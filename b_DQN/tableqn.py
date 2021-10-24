# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
# -*- coding: utf-8 -*-
import gym
import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# Environment
# Slippery environment (stochastic policy, move left probability = 1/3) comes by default!
# If we want deterministic policy, we need to create new environment
# Make environment No Slippery (deterministic policy, move left = 100% left)

MAX_EPISODE_STEPS = 250

gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4', 'is_slippery': False
    },
    max_episode_steps=MAX_EPISODE_STEPS,
)

# You can only register once
# To delete any new environment
# del gym.envs.registry.env_specs['FrozenLakeNotSlippery-v0']

# Make the environment based on deterministic policy
env = gym.make('FrozenLakeNotSlippery-v0')
# env = gym.make('FrozenLake-v0')


# Q값이 모두 같을때 랜덤한 action을 구해주기 위한 함수
def greedy_action(action_values):
    max_value = np.max(action_values)
    return np.random.choice([action_ for action_, value_ in enumerate(action_values) if value_ == max_value])


def epsilon_greedy_action(action_values, epsilon):
    if np.random.rand() < epsilon:
        return random.choice(range(len(action_values)))
    else:
        max_value = np.max(action_values)
        return np.random.choice([action_ for action_, value_ in enumerate(action_values) if value_ == max_value])


def q_table_learning(num_episodes=200, alpha=0.1, gamma=0.95, epsilon=0.1):
    # Q-Table 초기화
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    episode_reward_list = []

    training_steps = 0
    last_episode_reward = 0

    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        print("EPISODE: {0} - Initial State: {1}".format(i, observation), end=" ")
        sList = [observation]

        episode_step = 0

        # The Q-Table 알고리즘
        while True:
            episode_step += 1
            # 가장 Q값이 높은 action을 결정함

            # action = greedy_action(q_table[observation, :])

            action = epsilon_greedy_action(q_table[observation, :], epsilon)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(action)

            # Q-Learning
            q_table[observation, action] = q_table[observation, action] + alpha * (reward + gamma * np.max(q_table[next_observation, :]) - q_table[observation, action])
            #episode_reward += (gamma ** (episode_step - 1)) * reward
            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            training_steps += 1  # Q-table 업데이트 횟수
            episode_reward_list.append(last_episode_reward)

            sList.append(next_observation)
            observation = next_observation

            if done or episode_step >= MAX_EPISODE_STEPS:
                print(sList, done, "GOAL" if done and observation == 15 else "")
                break

        last_episode_reward = episode_reward

    return q_table, training_steps, episode_reward_list


def q_table_testing(num_episodes, q_table):
    episode_reward_list = []

    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        pass # 숙제

        episode_reward_list.append(episode_reward)

    return np.average(episode_reward_list)


def main_env_info():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print(env.observation_space)
    print(env.observation_space.n)
    # We should expect to see 15 possible grids from 0 to 15 when
    # we uniformly randomly sample from our observation space
    for i in range(10):
        print(env.observation_space.sample(), end=" ")
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print(env.action_space)
    print(env.action_space.n)
    # We should expect to see 4 actions when
    # we uniformly randomly sample:
    #     1. LEFT: 0
    #     2. DOWN: 1
    #     3. RIGHT: 2
    #     4. UP: 3
    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    observation = env.reset()
    print(observation, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    env.render()

    action = 2  # RIGHT
    observation, reward, done, info = env.step(action)
    env.render()

    # Observation = 1: move to grid number 1 (unchanged)
    # Prob = 1: deterministic policy, if we choose to go right, we'll go right
    print("Observation: {0}, Action: {1}, Reward: {2}, Done: {3}, Info: {4}".format(
        observation, action, reward, done, info
    ))

    action = 1  # DOWN
    observation, reward, done, info = env.step(action)
    env.render()

    # Observation = 5: move to grid number 5 (unchanged)
    print("Observation: {0}, Action: {1}, Reward: {2}, Done: {3}, Info: {4}".format(
        observation, action, reward, done, info
    ))

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    env.reset()
    env.render()

    actions = [2, 2, 1, 1, 1, 2]
    for action in actions:
        observation, reward, done, info = env.step(action)
        env.render()
        print("Observation: {0}, Action: {1}, Reward: {2}, Done: {3}, Info: {4}".format(
            observation, action, reward, done, info
        ))


def main_q_table_learning():
    NUM_EPISODES = 200
    ALPHA = 0.1
    GAMMA = 0.95
    EPSILON = 0.1

    q_table, training_steps, episode_reward_list = q_table_learning(NUM_EPISODES, ALPHA, GAMMA, EPSILON)
    print("\nFinal Q-Table Values")
    print("    LEFT   DOWN  RIGHT     UP")
    for idx, observation in enumerate(q_table):
        print("{0:2d}".format(idx), end=":")
        for action_state in observation:
            print("{0:5.3f} ".format(action_state), end=" ")
        print()

    plt.plot(range(training_steps), episode_reward_list, color="Blue")
    plt.xlabel("training steps")
    plt.ylabel("episode reward")
    plt.show()


if __name__ == "__main__":
    main_env_info()
    main_q_table_learning()
