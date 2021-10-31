# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
# -*- coding: utf-8 -*-
import random
import sys
import time

import gym
import torch
import os

from b_DQN.dqn_train_and_model_save import Qnet

ENV_NAME = "CartPole-v1"

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

MODEL_DIR = os.path.join(PROJECT_HOME, "b_DQN", "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_action(q, observation, epsilon):
    if random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        # Convert to Tensor
        observation = torch.tensor(observation, dtype=torch.float32, device=DEVICE)

        # Add batch-dim
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(dim=0)

        q_values = q(observation)
        action = torch.argmax(q_values, dim=-1)
        action = int(action.item())

    return action


def play(env, q, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        env.render()

        episode_steps = 0

        while True:
            episode_steps += 1
            action = get_action(q, observation, epsilon=0.0)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(action)
            env.render()

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            time.sleep(0.05)
            if done:
                break

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


def main_q_play(num_episodes):
    env = gym.make(ENV_NAME)

    q = Qnet().to(DEVICE)
    model_params = torch.load(
        os.path.join(MODEL_DIR, "dqn_CartPole-v1_500.0_0.0.pth")
    )
    q.load_state_dict(model_params)
    play(env, q, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 5
    main_q_play(num_episodes=NUM_EPISODES)
