import sys
import time

import gym
import torch
import os

from torch.distributions import Categorical

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from d_REINFORCE.REINFORCE_train_and_model_save import Policy

ENV_NAME = "CartPole-v1"

MODEL_DIR = os.path.join(PROJECT_HOME, "d_REINFORCE", "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_action(pi, observation):
    action_prob = pi(torch.from_numpy(observation).float())
    m = Categorical(action_prob)
    action = m.sample()
    return action_prob, action.item()


def play(env, pi, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        env.render()

        episode_steps = 0

        while True:
            episode_steps += 1
            _, action = get_action(pi, observation)

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


def main_reinforce_play(num_episodes):
    env = gym.make(ENV_NAME)

    pi = Policy()
    model_params = torch.load(
        os.path.join(MODEL_DIR, "reinforce_CartPole-v1_500.0_0.0.pth")
    )
    pi.load_state_dict(model_params)
    play(env, pi, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 5
    main_reinforce_play(num_episodes=NUM_EPISODES)
