# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
# -*- coding: utf-8 -*-
import collections
import time
from collections import namedtuple
import random
import sys
import os
from typing import Tuple

from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

ENV_NAME = "CartPole-v1"

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

MODEL_DIR = os.path.join(PROJECT_HOME, "b_DQN", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'reward', 'next_observation', 'done']
)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # fully connected
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, obs, epsilon):
        out = self.forward(obs)

        coin = random.random() # 0.0과 1.0사이의 임의의 값을 반환
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()  # argmax: 더 큰 값에 대응되는 인덱스 반환


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def reset(self):
        self.buffer.clear()

    def sample(self, batch_size: int) -> Tuple:
        # Get index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        # Sample
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=bool)

        # Convert to tensor
        states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return states, actions, rewards, next_states, dones


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
                project="DQN_CARTPOLE"
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

        # test_env
        self.test_env = gym.make(ENV_NAME)

        # network
        self.q = Qnet().to(DEVICE)
        self.target_q = Qnet().to(DEVICE)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # init rewards
        self.total_rewards = []
        self.best_mean_reward = -1000000000

        self.total_step_idx = 0
        self.training_steps = 0

    def get_action(self, observation, epsilon):
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            # Convert to Tensor
            observation = torch.tensor(observation, dtype=torch.float32, device=DEVICE)

            # Add batch-dim
            if len(observation.shape) == 3:
                observation = observation.unsqueeze(dim=0)

            q_values = self.q(observation)
            action = torch.argmax(q_values, dim=-1)
            action = int(action.item())

        return action

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

        test_episode_reward_avg = 0.0
        test_episode_reward_std = 0.0

        is_terminated = False

        for n_episode in range(self.max_num_episodes):
            epsilon = self.epsilon_scheduled(n_episode)

            episode_reward = 0

            # Environment 초기화와 변수 초기화
            observation = self.env.reset()

            while True:
                self.total_step_idx += 1

                action = self.get_action(observation, epsilon)

                # do step in the environment
                next_observation, reward, done, _ = self.env.step(action)

                transition = Transition(
                    observation, action, reward, next_observation, done
                )
                self.replay_buffer.append(transition)

                if self.total_step_idx > self.min_buffer_size_for_training:
                    loss = self.train_step()

                episode_reward += reward
                observation = next_observation

                if done:
                    self.total_rewards.append(episode_reward)

                    mean_episode_reward = np.mean(self.total_rewards[-100:])

                    total_training_time = time.time() - total_train_start_time
                    total_training_time = time.strftime(
                        '%H:%M:%S', time.gmtime(total_training_time)
                    )

                    if n_episode % self.print_episode_interval == 0:
                        print(
                            "[Episode {:3}, Steps {:6}]".format(
                                n_episode, self.total_step_idx
                            ),
                            "Episode Reward: {:>5},".format(episode_reward),
                            "Mean Episode Reward: {:.3f},".format(mean_episode_reward),
                            "size of replay buffer: {:>6}".format(
                                self.replay_buffer.size()
                            ),
                            "Loss: {:.3f},".format(loss),
                            "Epsilon: {:.2f},".format(epsilon),
                            "Num Training Steps: {:4},".format(self.training_steps),
                            "Total Elapsed Time {}".format(total_training_time)
                        )

                    if self.training_steps > 0 and n_episode % self.test_episode_interval == 0:
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
                                self.total_step_idx, self.training_steps
                            ))
                            self.model_save(
                                test_episode_reward_avg, test_episode_reward_std
                            )
                            is_terminated = True

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
                            "Number of Training Steps": self.training_steps
                        })

                    break

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))

    def train_step(self):
        self.training_steps += 1

        batch = self.replay_buffer.sample(self.batch_size)

        # states.shape: torch.Size([32, 4, 84, 84]), actions.shape: torch.Size([32]),
        # rewards.shape: torch.Size([32]), next_states.shape: torch.Size([32, 4, 84, 84]),
        # dones.shape: torch.Size([32])
        states, actions, rewards, next_states, dones = batch

        # state_action_values.shape: torch.Size([32])
        state_action_values = self.q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # next_state_values.shape: torch.Size([32])
            next_state_values = self.target_q(next_states).max(dim=1).values
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

            # target_state_action_values.shape: torch.Size([32])
            target_state_action_values = rewards + self.gamma * next_state_values

        loss = F.mse_loss(state_action_values, target_state_action_values)

        # print("states.shape: {0}, actions.shape: {1}, rewards.shape: {2}, "
        #       "next_states.shape: {3}, dones.shape: {4}".format(
        #     states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape
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
        if self.total_step_idx % self.target_sync_step_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return loss.item()

    def model_save(self, test_episode_reward_avg, test_episode_reward_std):
        print("Solved in {0} steps ({1} training steps)!".format(
            self.total_step_idx, self.training_steps
        ))
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
                action = self.get_action(observation, epsilon=0.0)

                # action을 통해서 next_state, reward, done, info를 받아온다
                next_observation, reward, done, _ = self.test_env.step(action)

                episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

        return np.average(episode_reward_lst), np.std(episode_reward_lst)


def main():
    dqn = DQN(
        use_wandb=True,                            # WANDB 연결 및 로깅 유무
        wandb_entity="link-koreatech",          # WANDB 개인 계정
        max_num_episodes=1000,                  # 훈련을 위한 최대 에피소드 횟수
        batch_size=32,                          # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        learning_rate=0.0001,                   # 학습율
        gamma=0.99,                             # 감가율
        target_sync_step_interval=500,          # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
        replay_buffer_size=5_000,               # 리플레이 버퍼 사이즈
        min_buffer_size_for_training=100,       # 훈련을 위한 최소 리플레이 버퍼 사이즈
        epsilon_start=0.5,                      # Epsilon 초기 값
        epsilon_end=0.01,                       # Epsilon 최종 값
        epsilon_scheduled_last_episode=300,     # Epsilon 최종 값으로 스케줄되어지는 마지막 에피소드
        print_episode_interval=10,              # Episode 통계 출력에 관한 에피소드 간격
        test_episode_interval=50,               # 테스트를 위한 episode 간격
        test_num_episodes=3,                    # 테스트시에 수행하는 에피소드 횟수
        episode_reward_avg_solved=450,          # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        episode_reward_std_solved=10            # 훈련 종료를 위한 테스트 에피소드 리워드의 Standard Deviation
    )
    dqn.train_loop()


if __name__ == '__main__':
    main()
