import sys
import time
import gym
import torch
import os


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from c_DQN_ATARI.dqn_train_and_model_save import AtariCNN

ENV_NAME = "PongNoFrameskip-v4"

MODEL_DIR = os.path.join(PROJECT_HOME, "c_DQN_ATARI", "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(env, q, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()
        env.render()

        episode_steps = 0

        while True:
            episode_steps += 1
            _, gym_action = q.get_action(observation, epsilon=0.0)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, _ = env.step(gym_action)
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
    env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = gym.wrappers.FrameStack(env, num_stack=4, lz4_compress=True)

    obs_shape = env.observation_space.shape
    n_actions = 3
    q = AtariCNN(obs_shape, n_actions).to(DEVICE)
    model_params = torch.load(
        os.path.join(MODEL_DIR, "dqn_PongNoFrameskip-v4_6.0_1.4.pth")
    )
    q.load_state_dict(model_params)
    play(env, q, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 5
    main_q_play(num_episodes=NUM_EPISODES)
