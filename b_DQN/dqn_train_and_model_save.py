# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
# -*- coding: utf-8 -*-
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gym
import numpy as np
import matplotlib.pyplot as plt
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

from b_DQN.dqn import Qnet, ReplayMemory


MODEL_DIR = os.path.join(PROJECT_HOME, "b_DQN", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

WANDB = False

if WANDB:
    wandb = wandb.init(
        entity="link-koreatech",
        project="DQN"
    )


def q_learning(
        env, test_env, num_episodes=1000, learning_rate=0.0001, gamma=0.99,
        epsilon_start=0.2, epsilon_end=0.01, batch_size=32,
        train_step_interval=4, target_update_step_interval=100,
        print_episode_interval=10, test_num_episodes=3,
        episode_reward_solved=475, episode_reward_std_solved=25
):
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayMemory()

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    training_step_lst = []
    episode_reward_lst = []

    training_steps = 0
    last_episode_reward = 0

    total_step_idx = 0

    test_training_step_lst = []
    test_avg_episode_reward_lst = []

    test_avg_episode_reward = None
    test_std_episode_reward = None
    is_terminated = False

    for i in range(num_episodes):
        epsilon = max(epsilon_end, epsilon_start - 0.5 * (i / num_episodes))

        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        #env.render()

        while True:
            total_step_idx += 1

            # 가장 Q값이 높은 action을 결정함
            action = q.get_action(torch.from_numpy(observation).float(), epsilon)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(action)
            #env.render()
            done_mask = 0.0 if done else 1.0

            memory.put(
                transition=(observation, action, reward / 100.0, next_observation, done_mask)
            )

            if memory.size() > 2000 and total_step_idx % train_step_interval == 0:
                observation_t, action_t, reward_t, next_observation_t, done_mask_t = memory.sample(batch_size)

                q_out = q(observation_t)                                   # q_out.shape: (32, 2)
                q_a = q_out.gather(dim=1, index=action_t)                  # q_a.shape: (32, 2)

                q_prime_out = q_target(next_observation_t)                 # q_prime_out.shape: (32, 2)

                # q_prime_out.max(dim=1).values.shape: (32,)
                max_q_prime = q_prime_out.max(dim=1).values.unsqueeze(dim=-1)  # max_q_prime.shape: (32, 1)

                target = reward_t + gamma * max_q_prime * done_mask_t
                loss = F.mse_loss(target, q_a)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_steps += 1    # Q 업데이트 횟수 증가
                training_step_lst.append(training_steps)
                episode_reward_lst.append(last_episode_reward)

                if WANDB:
                    wandb.log({
                        "EPISODE": i,
                        "EPISODE_REWARD": episode_reward,
                        "LOSS": loss.item(),
                        "SIZE_OF_REPLAY_BUFFER": memory.size(),
                        "EPSILON": epsilon,
                        "TRAINING_STEPS": training_steps,
                        "TEST_AVG_EPISODE_REWARD": test_avg_episode_reward,
                        "TEST_STD_EPISODE_REWARD": test_std_episode_reward
                    })

            if total_step_idx % target_update_step_interval == 0:
                q_target.load_state_dict(q.state_dict())

            if training_steps != 0 and training_steps % 10 == 0:
                test_avg_episode_reward, test_std_episode_reward = q_testing(
                    test_env, q, test_num_episodes
                )
                test_training_step_lst.append(training_steps)
                test_avg_episode_reward_lst.append(test_avg_episode_reward)

                termination_conditions = [
                    test_avg_episode_reward > episode_reward_solved,
                    test_std_episode_reward < episode_reward_std_solved
                ]

                if all(termination_conditions):
                    torch.save(
                        q.state_dict(),
                        os.path.join(MODEL_DIR, "dqn_{0}_{1:4.1f}_{2:3.1f}.pth".format(
                            ENV_NAME, test_avg_episode_reward, test_std_episode_reward
                        ))
                    )
                    is_terminated = True

            # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            episode_reward += reward
            observation = next_observation

            if done or is_terminated:
                last_episode_reward = episode_reward
                break

        if i % print_episode_interval == 0 and i != 0:
            print("EPISODE: {0:3d}, EPISODE_REWARD: {1:5.1f}, "
                  "SIZE_OF_REPLAY_BUFFER: {2:5d}, EPSILON: {3:.3f}, "
                  "TRAINING_STEPS: {4:5d}".format(
                i, episode_reward, memory.size(), epsilon, training_steps
            ))

        if is_terminated:
            break

    return training_step_lst, episode_reward_lst, test_training_step_lst, \
           test_avg_episode_reward_lst


def q_testing(test_env, q, num_episodes):
    episode_reward_lst = []

    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = test_env.reset()

        while True:
            action = q.get_action(torch.from_numpy(observation).float(), epsilon=0.0)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = test_env.step(action)

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            if done:
                break

        episode_reward_lst.append(episode_reward)

    return np.average(episode_reward_lst), np.std(episode_reward_lst)


def main_q_learning():
    NUM_EPISODES = 300
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 0.7
    EPSILON_END = 0.01
    BATCH_SIZE = 32
    TRAIN_STEP_INTERVAL = 4
    TARGET_UPDATE_STEP_INTERVAL = 100
    PRINT_EPISODE_INTERVAL = 10
    TEST_NUM_EPISODES = 7
    EPISODE_REWARD_SOLVED = 475
    EPISODE_REWARD_STD_SOLVED = 25

    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    training_step_lst, episode_reward_lst, \
    test_training_step_lst, test_avg_episode_reward_lst = q_learning(
        env, test_env,
        NUM_EPISODES, LEARNING_RATE, GAMMA,
        EPSILON_START, EPSILON_END, BATCH_SIZE, TRAIN_STEP_INTERVAL,
        TARGET_UPDATE_STEP_INTERVAL, PRINT_EPISODE_INTERVAL, TEST_NUM_EPISODES,
        EPISODE_REWARD_SOLVED, EPISODE_REWARD_STD_SOLVED
    )

    plt.plot(training_step_lst, episode_reward_lst, color="Blue")
    plt.xlabel("training steps")
    plt.ylabel("episode reward")
    plt.show()

    plt.plot(test_training_step_lst, test_avg_episode_reward_lst, color="Red")
    plt.xlabel("training steps")
    plt.ylabel("test episode reward")
    plt.show()


if __name__ == "__main__":
    main_q_learning()
