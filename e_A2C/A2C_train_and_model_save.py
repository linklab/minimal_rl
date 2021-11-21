import sys
import os
import time

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from gym.vector import AsyncVectorEnv

import wandb

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from a_common.a_commons import VectorizedTransitions, make_gym_env
from a_common.b_models import ActorCritic
from a_common.c_buffers import ReplayBufferForVectorizedEnvs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_DIR = os.path.join(PROJECT_HOME, "e_A2C", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A2C:
    def __init__(
            self, env_name, n_vec_envs, env, test_env, use_wandb, wandb_entity,
            max_num_time_steps, batch_size, learning_rate, gamma,
            print_time_step_interval,
            test_training_time_step_interval, test_num_episodes,
            episode_reward_avg_solved, episode_reward_std_solved
    ):
        self.env_name = env_name
        self.n_vec_envs = n_vec_envs
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.wandb = wandb.init(
                entity=wandb_entity,
                project="A2C_{0}".format(self.env_name)
            )
        self.max_num_time_steps = max_num_time_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.print_time_step_interval = print_time_step_interval
        self.test_training_time_step_interval = test_training_time_step_interval
        self.test_num_episodes = test_num_episodes
        self.episode_reward_avg_solved = episode_reward_avg_solved
        self.episode_reward_std_solved = episode_reward_std_solved

        # env
        self.env = env
        self.test_env = test_env

        self.buffer_for_vectorized_envs = ReplayBufferForVectorizedEnvs(
            capacity=batch_size
        )

        self.actor_critic_model = ActorCritic(
            n_features=4, n_actions=2, device=DEVICE
        )
        self.optimizer = optim.Adam(
            self.actor_critic_model.parameters(), lr=learning_rate
        )

        # init rewards
        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

        self.next_test_training_time_step = self.test_training_time_step_interval

    def train_loop(self):
        episode_rewards = np.zeros((self.n_vec_envs,))
        n_episode = 0
        mean_episode_reward = 0.0

        actor_objective = 0.0
        critic_loss = 0.0
        is_terminated = False

        total_train_start_time = time.time()

        observations = self.env.reset()

        while self.time_steps < self.max_num_time_steps:
            actions = self.actor_critic_model.get_action(observations)
            next_observations, rewards, dones, infos = self.env.step(actions)
            self.time_steps += self.n_vec_envs

            vectorized_transitions = VectorizedTransitions(
                observations, actions, next_observations, rewards, dones
            )
            self.buffer_for_vectorized_envs.append(vectorized_transitions)

            observations = next_observations
            episode_rewards += rewards

            # TRAIN
            if len(self.buffer_for_vectorized_envs) >= self.batch_size:
                actor_objective, critic_loss = self.train_step()

            if any(dones):
                self.episode_reward_lst.extend([episode_reward for episode_reward in episode_rewards[dones]])
                episode_rewards[dones] = 0.0
                mean_episode_reward = np.mean(self.episode_reward_lst[-100:])
                n_episode += sum(dones)

            if self.training_time_steps >= self.next_test_training_time_step:
                test_episode_reward_avg, test_episode_reward_std = self.a2c_testing(
                    self.test_num_episodes
                )

                print("[Test Episode Reward (Training Time Steps: {0:3})] "
                      "Average: {1:7.3f}, Standard Dev.: {2:.3f}".format(
                    self.training_time_steps,
                    test_episode_reward_avg,
                    test_episode_reward_std
                ))

                termination_conditions = [
                    test_episode_reward_avg > self.episode_reward_avg_solved,
                    test_episode_reward_std < self.episode_reward_std_solved
                ]

                if self.use_wandb:
                    self.wandb.log({
                        "[TEST] Average Episode Reward": test_episode_reward_avg,
                        "[TEST] Std. Episode Reward": test_episode_reward_std,
                        "Episode": n_episode,
                        "Actor Objective": actor_objective if actor_objective != 0.0 else 0.0,
                        "Critic Loss": critic_loss if critic_loss != 0.0 else 0.0,
                        "Mean Episode Reward": mean_episode_reward,
                        "Number of Training Steps": self.training_time_steps,
                    })

                if all(termination_conditions):
                    print("Solved in {0} steps ({1} training steps)!".format(
                        self.time_steps, self.training_time_steps
                    ))
                    self.model_save(
                        test_episode_reward_avg, test_episode_reward_std
                    )
                    is_terminated = True

                self.next_test_training_time_step += self.test_training_time_step_interval

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime(
                '%H:%M:%S', time.gmtime(total_training_time)
            )

            if self.time_steps % self.print_time_step_interval == 0:
                print(
                    "[Episode {:3}, Steps {:6}, Training Steps {:6}]".format(
                        n_episode + 1, self.time_steps, self.training_time_steps
                    ),
                    "Mean Episode Reward: {:.3f},".format(mean_episode_reward),
                    "Actor Objective: {:.3f},".format(actor_objective),
                    "Critic Loss: {:.3f},".format(critic_loss),
                    "Total Elapsed Time {}".format(total_training_time)
                )

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))

    def train_step(self):
        self.training_time_steps += 1

        batch = self.buffer_for_vectorized_envs.sample_all()

        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = batch

        self.optimizer.zero_grad()

        ###################################
        #  Critic (Value) 손실 산출 - BEGIN #
        ###################################
        # next_values.shape: (32, 1)
        next_values = self.actor_critic_model.v(next_observations)
        td_target_value_lst = []

        for reward, next_value, done in zip(rewards, next_values, dones):
            td_target = reward + self.gamma * next_value * (0.0 if done else 1.0)
            td_target_value_lst.append(td_target)

        # td_target_values.shape: (32, 1)
        td_target_values = torch.tensor(
            td_target_value_lst, dtype=torch.float32
        ).unsqueeze(dim=-1)

        # values.shape: (32, 1)
        values = self.actor_critic_model.v(observations)
        # loss_critic.shape: (,) <--  값 1개
        critic_loss = F.mse_loss(td_target_values.detach(), values)
        ###################################
        #  Critic (Value)  Loss 산출 - END #
        ###################################

        ################################
        #  Actor Objective 산출 - BEGIN #
        ################################
        q_values = td_target_values
        advantages = (q_values - values).detach()

        action_probs = self.actor_critic_model.pi(observations)
        action_prob_selected = action_probs.gather(dim=1, index=actions)

        # action_prob_selected.shape: (32, 1)
        # advantage.shape: (32, 1)
        # log_pi_advantages.shape: (32, 1)
        log_pi_advantages = torch.multiply(
            torch.log(action_prob_selected), advantages
        )

        # actor_objective.shape: (,) <--  값 1개
        actor_objective = torch.sum(log_pi_advantages)
        actor_loss = torch.multiply(actor_objective, -1.0)
        ##############################
        #  Actor Objective 산출 - END #
        ##############################

        loss = critic_loss * 0.5 + actor_loss

        loss.backward()
        self.optimizer.step()

        self.buffer_for_vectorized_envs.clear()

        return actor_objective.item(), critic_loss.item()

    def model_save(self, test_episode_reward_avg, test_episode_reward_std):
        torch.save(
            self.actor_critic_model.state_dict(),
            os.path.join(MODEL_DIR, "a2c_{0}_{1:4.1f}_{2:3.1f}.pth".format(
                self.env_name, test_episode_reward_avg, test_episode_reward_std
            ))
        )

    def a2c_testing(self, num_episodes):
        episode_reward_lst = []

        for i in range(num_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation = self.test_env.reset()

            while True:
                action = self.actor_critic_model.get_action(observation, mode="test")

                next_observation, reward, done, _ = self.test_env.step(action)

                episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

        return np.average(episode_reward_lst), np.std(episode_reward_lst)


def main():
    ENV_NAME = "CartPole-v1"

    # env
    n_vec_envs = 4
    env = AsyncVectorEnv(env_fns=[make_gym_env(ENV_NAME) for _ in range(n_vec_envs)])
    test_env = gym.make(ENV_NAME)

    a2c = A2C(
        env_name=ENV_NAME,
        n_vec_envs=n_vec_envs,
        env=env,
        test_env=test_env,
        use_wandb=True,                         # WANDB 연결 및 로깅 유무
        wandb_entity="link-koreatech",          # WANDB 개인 계정
        max_num_time_steps=1_000_000,           # 훈련을 위한 최대 타임 스텝 수
        batch_size=64,                          # 훈련시 버퍼에서 한번에 가져오는 전체 배치 사이즈
        learning_rate=0.0005,                   # 학습율
        gamma=0.99,                             # 감가율
        print_time_step_interval=500,           # 통계 출력에 관한 time_step 간격
        test_training_time_step_interval=100,   # 테스트를 위한 training_time_step 간격
        test_num_episodes=3,                    # 테스트시에 수행하는 에피소드 횟수
        episode_reward_avg_solved=450,          # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        episode_reward_std_solved=10            # 훈련 종료를 위한 테스트 에피소드 리워드의 Standard Deviation
    )
    a2c.train_loop()


if __name__ == '__main__':
    main()
