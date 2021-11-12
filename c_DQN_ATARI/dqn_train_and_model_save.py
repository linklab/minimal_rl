import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import wandb
import sys, os

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from a_common.a_commons import Transition
from a_common.b_models import AtariCNN
from a_common.c_buffers import ReplayBuffer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# local
ENV_NAME = "PongNoFrameskip-v4"

MODEL_DIR = os.path.join(PROJECT_HOME, "c_DQN_ATARI", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN():
    def __init__(
            self, use_wandb, wandb_entity,
            max_num_episodes, batch_size, learning_rate,
            gamma, target_sync_step_interval,
            replay_buffer_size, min_buffer_size_for_training,
            epsilon_start, epsilon_end,
            epsilon_scheduled_last_episode,
            print_episode_interval,
            test_episode_interval, test_num_episodes,
            episode_reward_avg_solved, episode_reward_std_solved
    ):
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.wandb = wandb.init(
                entity=wandb_entity,
                project="DQN_PONG"
            )
        self.max_num_episodes = max_num_episodes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_sync_step_interval = target_sync_step_interval
        self.replay_buffer_size = replay_buffer_size
        self.min_buffer_size_for_training = min_buffer_size_for_training
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_scheduled_last_episode = epsilon_scheduled_last_episode
        self.print_episode_interval = print_episode_interval
        self.test_episode_interval = test_episode_interval
        self.test_num_episodes = test_num_episodes
        self.episode_reward_avg_solved = episode_reward_avg_solved
        self.episode_reward_std_solved = episode_reward_std_solved

        # env
        self.env = gym.make(ENV_NAME)
        self.env = gym.wrappers.AtariPreprocessing(
            self.env, grayscale_obs=True, scale_obs=True
        )
        self.env = gym.wrappers.FrameStack(
            self.env, num_stack=4, lz4_compress=True
        )

        # test_env
        self.test_env = gym.make(ENV_NAME)
        self.test_env = gym.wrappers.AtariPreprocessing(
            self.test_env, grayscale_obs=True, scale_obs=True
        )
        self.test_env = gym.wrappers.FrameStack(self.test_env, num_stack=4, lz4_compress=True)

        # self.env = make_sleepy_toy_env(self.env_name)

        obs_shape = self.env.observation_space.shape
        #n_actions = self.env.action_space.n
        n_actions = 3

        # network
        self.q = AtariCNN(obs_shape, n_actions).to(DEVICE)
        self.target_q = AtariCNN(obs_shape, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, device=DEVICE)

        # init rewards
        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

    def epsilon_scheduled(self, current_episode):
        fraction = min(current_episode / self.epsilon_scheduled_last_episode, 1.0)
        epsilon = min(
            self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start),
            self.epsilon_start
        )
        return epsilon

    def train_loop(self):
        loss = 0.0

        total_train_start_time = time.time()

        test_episode_reward_avg = -21
        test_episode_reward_std = 0.0

        is_terminated = False

        for n_episode in range(self.max_num_episodes):
            epsilon = self.epsilon_scheduled(n_episode)

            episode_start_time = time.time()

            episode_reward = 0

            # Environment 초기화와 변수 초기화
            observation = self.env.reset()

            while True:
                self.time_steps += 1

                action, gym_action = self.q.get_action(observation, epsilon)

                # do step in the environment
                next_observation, reward, done, _ = self.env.step(gym_action)

                transition = Transition(
                    observation, action, next_observation, reward, done
                )
                self.replay_buffer.append(transition)

                if self.time_steps > self.min_buffer_size_for_training:
                    loss = self.train_step()

                episode_reward += reward
                observation = next_observation

                if done:
                    self.episode_reward_lst.append(episode_reward)

                    per_episode_time = time.time() - episode_start_time
                    per_episode_time = time.strftime('%H:%M:%S', time.gmtime(per_episode_time))

                    mean_episode_reward = np.mean(self.episode_reward_lst[-100:])

                    total_training_time = time.time() - total_train_start_time
                    total_training_time = time.strftime(
                        '%H:%M:%S', time.gmtime(total_training_time)
                    )

                    if self.training_time_steps > 0 and n_episode % self.test_episode_interval == 0:
                        test_episode_reward_avg, test_episode_reward_std = self.q_testing(
                            self.test_num_episodes
                        )

                        print("[Test Episode Reward] Average: {0:.3f}, Standard Dev.: {1:.3f}".format(
                            test_episode_reward_avg, test_episode_reward_std
                        ))

                        termination_conditions = [
                            test_episode_reward_avg > self.episode_reward_avg_solved,
                            test_episode_reward_std < self.episode_reward_std_solved
                        ]

                        if all(termination_conditions):
                            print("Solved in {0} steps ({1} training steps)!".format(
                                self.time_steps, self.training_time_steps
                            ))
                            self.model_save(
                                test_episode_reward_avg, test_episode_reward_std
                            )
                            is_terminated = True

                    if (n_episode + 1) % self.print_episode_interval == 0:
                        print(
                            "[Episode {:3}, Steps {:6}]".format(
                                n_episode + 1, self.time_steps
                            ),
                            "Episode Reward: {:>5},".format(episode_reward),
                            "Mean Episode Reward: {:.3f},".format(mean_episode_reward),
                            "size of replay buffer: {:>6}".format(
                                self.replay_buffer.size()
                            ),
                            "Loss: {:.3f},".format(loss),
                            "Epsilon: {:.2f},".format(epsilon),
                            "Num Training Steps: {:4},".format(self.training_time_steps),
                            "Per-Episode Time: {}".format(per_episode_time),
                            "Total Elapsed Time {}".format(total_training_time)
                        )

                    if self.use_wandb:
                        self.wandb.log({
                            "[TEST] Average Episode Reward": test_episode_reward_avg,
                            "[TEST] Std. Episode Reward": test_episode_reward_std,
                            "Episode Reward": episode_reward,
                            "Loss": loss if loss != 0.0 else 0.0,
                            "Epsilon": epsilon,
                            "Mean Episode Reward": mean_episode_reward,
                            "Episode": n_episode,
                            "Size of replay buffer": self.replay_buffer.size(),
                            "Number of Training Steps": self.training_time_steps,
                        })

                    break

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))

    def train_step(self):
        self.training_time_steps += 1

        batch = self.replay_buffer.sample(self.batch_size)

        # observations.shape: torch.Size([32, 4, 84, 84]),
        # actions.shape: torch.Size([32]),
        # next_observations.shape: torch.Size([32, 4, 84, 84]),
        # rewards.shape: torch.Size([32]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = batch

        # state_action_values.shape: torch.Size([32])
        state_action_values = self.q(observations).gather(1, actions)

        with torch.no_grad():
            # next_state_values.shape: torch.Size([32])
            next_state_values = self.target_q(next_observations).max(dim=1).values
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

            # target_state_action_values.shape: torch.Size([32])
            target_state_action_values = rewards + self.gamma * next_state_values

        loss = F.mse_loss(state_action_values, target_state_action_values)

        # print("observations.shape: {0}, actions.shape: {1}, next_observations.shape: {2}, rewards.shape: {3}, dones.shape: {4}".format(
        #     observations.shape, actions.shape, next_observations.shape, rewards.shape, dones.shape
        # ))
        # print("state_action_values.shape: {0}".format(state_action_values.shape))
        # print("next_state_values.shape: {0}".format(next_state_values.shape))
        # print("target_state_action_values.shape: {0}".format(
        #     target_state_action_values.shape
        # ))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # sync
        if self.time_steps % self.target_sync_step_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return loss.item()

    def model_save(self, test_episode_reward_avg, test_episode_reward_std):
        torch.save(
            self.q.state_dict(),
            os.path.join(MODEL_DIR, "dqn_{0}_{1:4.1f}_{2:3.1f}.pth".format(
                ENV_NAME, test_episode_reward_avg, test_episode_reward_std
            ))
        )

    def q_testing(self, num_episodes):
        episode_reward_lst = []

        for i in range(num_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation = self.test_env.reset()

            while True:
                _, gym_action = self.q.get_action(observation, epsilon=0.0)

                # action을 통해서 next_state, reward, done, info를 받아온다
                next_observation, reward, done, _ = self.test_env.step(gym_action)

                episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

        return np.average(episode_reward_lst), np.std(episode_reward_lst)


def main():
    ENV_NAME = "PongNoFrameskip-v4"

    # env
    env = gym.make(ENV_NAME)
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4, lz4_compress=True)

    # test_env
    test_env = gym.make(ENV_NAME)
    test_env = gym.wrappers.AtariPreprocessing(test_env, grayscale_obs=True, scale_obs=True
    )
    test_env = gym.wrappers.FrameStack(test_env, num_stack=4, lz4_compress=True)

    dqn = DQN(
        env_name=ENV_NAME,
        use_wandb=True,                         # WANDB 연결 및 로깅 유무
        wandb_entity="link-koreatech",          # WANDB 개인 계정
        env=env,
        test_env=test_env,
        max_num_episodes=None,                  # 훈련을 위한 최대 에피소드 횟수
        batch_size=None,                        # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        learning_rate=None,                     # 학습율
        gamma=None,                             # 감가율
        target_sync_step_interval=None,         # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
        replay_buffer_size=None,                # 리플레이 버퍼 사이즈
        min_buffer_size_for_training=None,      # 훈련을 위한 최소 리플레이 버퍼 사이즈
        epsilon_start=None,                     # Epsilon 초기 값
        epsilon_end=None,                       # Epsilon 최종 값
        epsilon_scheduled_last_episode=None,    # Epsilon 최종 값으로 스케줄되어지는 마지막 에피소드
        print_episode_interval=1,               # Episode 통계 출력에 관한 에피소드 간격
        test_episode_interval=None,             # 테스트를 위한 training_step 간격
        test_num_episodes=None,                 # 테스트시에 수행하는 에피소드 횟수
        episode_reward_avg_solved=0,            # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        episode_reward_std_solved=3             # 훈련 종료를 위한 테스트 에피소드 리워드의 Standard Deviation
    )
    dqn.train_loop()


if __name__ == '__main__':
    main()
