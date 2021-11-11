import sys
import os
import time

from torch.distributions import Categorical

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import wandb

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

MODEL_DIR = os.path.join(PROJECT_HOME, "d_REINFORCE", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x


class REINFORCE:
    def __init__(
            self, env_name, env, test_env, use_wandb, wandb_entity,
            max_num_episodes, learning_rate, gamma,
            print_episode_interval,
            test_episode_interval, test_num_episodes,
            episode_reward_avg_solved, episode_reward_std_solved
    ):
        self.env_name = env_name
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.wandb = wandb.init(
                entity=wandb_entity,
                project="REINFOCE_{0}".format(self.env_name)
            )
        self.max_num_episodes = max_num_episodes
        self.gamma = gamma
        self.print_episode_interval = print_episode_interval
        self.test_episode_interval = test_episode_interval
        self.test_num_episodes = test_num_episodes
        self.episode_reward_avg_solved = episode_reward_avg_solved
        self.episode_reward_std_solved = episode_reward_std_solved

        #env
        self.env = env
        self.test_env = test_env

        self.data = []

        self.pi = Policy()
        self.optimizer = optim.Adam(self.pi.parameters(), lr=learning_rate)

        # init rewards
        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

    def train_net(self):
        R = 0
        loss = 0.0

        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

        return loss.item()

    def get_action(self, observation):
        action_prob = self.pi(torch.from_numpy(observation).float())
        m = Categorical(action_prob)
        action = m.sample()
        return action_prob, action.item()

    def train_loop(self):
        total_train_start_time = time.time()

        is_terminated = False

        for n_episode in range(self.max_num_episodes):
            episode_start_time = time.time()
            episode_reward = 0

            # Environment 초기화와 변수 초기화
            observation = self.env.reset()

            while True:
                self.time_steps += 1
                action_prob, action = self.get_action(observation)

                next_observation, reward, done, _ = self.env.step(action)

                self.data.append((reward, action_prob[action]))
                observation = next_observation
                episode_reward += reward

                if done:
                    break

            # TRAIN
            loss = self.train_net()
            self.training_time_steps = n_episode

            self.episode_reward_lst.append(episode_reward)

            per_episode_time = time.time() - episode_start_time
            per_episode_time = time.strftime('%H:%M:%S', time.gmtime(per_episode_time))

            mean_episode_reward = np.mean(self.episode_reward_lst[-100:])

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime(
                '%H:%M:%S', time.gmtime(total_training_time)
            )

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {:3}, Steps {:6}]".format(
                        n_episode, self.time_steps
                    ),
                    "Episode Reward: {:>5},".format(episode_reward),
                    "Mean Episode Reward: {:.3f},".format(mean_episode_reward),
                    "Loss: {:.3f},".format(loss),
                    "Per-Episode Time: {}".format(per_episode_time),
                    "Total Elapsed Time {}".format(total_training_time)
                )

            if self.training_time_steps > 0 and n_episode % self.test_episode_interval == 0:
                test_episode_reward_avg, test_episode_reward_std = self.reinforce_testing(
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

            if is_terminated:
                break

    def model_save(self, test_episode_reward_avg, test_episode_reward_std):
        print("Solved in {0} steps ({1} training steps)!".format(
            self.time_steps, self.training_time_steps
        ))
        torch.save(
            self.pi.state_dict(),
            os.path.join(MODEL_DIR, "reinforce_{0}_{1:4.1f}_{2:3.1f}.pth".format(
                self.env_name, test_episode_reward_avg, test_episode_reward_std
            ))
        )

    def reinforce_testing(self, num_episodes):
        episode_reward_lst = []

        for i in range(num_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment 초기화와 변수 초기화
            observation = self.test_env.reset()

            while True:
                _, action = self.get_action(observation)

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
    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    reinforce = REINFORCE(
        env_name=ENV_NAME,
        env=env,
        test_env=test_env,
        use_wandb=False,                            # WANDB 연결 및 로깅 유무
        wandb_entity="link-koreatech",          # WANDB 개인 계정
        max_num_episodes=None,                  # 훈련을 위한 최대 에피소드 횟수
        learning_rate=None,                     # 학습율
        gamma=None,                             # 감가율
        print_episode_interval=None,            # Episode 통계 출력에 관한 에피소드 간격
        test_episode_interval=None,             # 테스트를 위한 episode 간격
        test_num_episodes=None,                 # 테스트시에 수행하는 에피소드 횟수
        episode_reward_avg_solved=450,          # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        episode_reward_std_solved=10            # 훈련 종료를 위한 테스트 에피소드 리워드의 Standard Deviation
    )
    reinforce.train_loop()


if __name__ == '__main__':
    main()