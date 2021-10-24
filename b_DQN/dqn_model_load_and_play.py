# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
# -*- coding: utf-8 -*-
import sys
import gym
import numpy as np
import torch
import os

ENV_NAME = "CartPole-v1"

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from b_DQN.dqn import Qnet

MODEL_DIR = os.path.join(PROJECT_HOME, "b_DQN", "models")


def play(q, num_episodes):
    env = gym.make(ENV_NAME)

    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        env.render()

        episode_steps = 0

        while True:
            episode_steps += 1
            action = q.get_action(torch.from_numpy(observation).float(), epsilon=0.0)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(action)
            env.render()

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            if done:
                break

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


def main_q_play():
    q = Qnet()
    model_params = torch.load(
        os.path.join(MODEL_DIR, "dqn_CartPole-v1_500.0_0.0.pth")
    )
    q.load_state_dict(model_params)
    play(q, num_episodes=3)


if __name__ == "__main__":
    main_q_play()
