# -*- coding: utf-8 -*-
import os
import sys
import gym

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

ENV_NAME = "PongNoFrameskip-v4"


def main_env_info():
    env = gym.make(ENV_NAME)
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4, lz4_compress=True)

    print("*" * 80)
    print(env.observation_space)
    # for i in range(1):
    #     print(env.observation_space.sample())
    # print()

    ################
    # action space #
    ################
    print("*" * 80)
    print(env.action_space)
    print(env.action_space.n)
    print(env.get_action_meanings())
    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
    # Starting State:
    # All observations are assigned a uniform random value in [-0.05..0.05]
    observation = env.reset()
    print(observation.shape)

    # Reward:
    # Reward is 1 for every step taken, including the termination step
    action = 0  # LEFT
    next_observation, reward, done, info = env.step(action)

    # Observation = 1: move to grid number 1 (unchanged)
    # Prob = 1: deterministic policy, if we choose to go right, we'll go right
    print("Observation: {0}, Action: {1}, Next Observation: {3}, Reward: {2}, Done: {4}, Info: {5}".format(
        observation.shape, action, next_observation.shape, reward, done, info
    ))

    observation = next_observation

    action = 1
    next_observation, reward, done, info = env.step(action)

    print("Observation: {0}, Action: {1}, Next Observation: {4}, Reward: {3}, Done: {4}, Info: {5}".format(
        observation.shape, action, next_observation.shape, reward, done, info
    ))

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    observation = env.reset()
    env.render()

    # actions = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]

    actions = ([2] * 3 + [3] * 3) * 500

    for action in actions:
        next_observation, reward, done, info = env.step(action)
        env.render()
        print("Observation: {0}, Action: {1}, Next Observation: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
            observation.shape, action, next_observation.shape, reward, done, info
        ))
        observation = next_observation

    env.close()


if __name__ == "__main__":
    main_env_info()
