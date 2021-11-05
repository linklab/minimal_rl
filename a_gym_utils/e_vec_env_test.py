import random
import time
from collections import namedtuple

import numpy as np
import gym
from gym.vector import SyncVectorEnv, AsyncVectorEnv

from b_DQN.dqn_train_and_model_save import ReplayBuffer, Transition


VectorizedTransitions = namedtuple(
    typename='VectorizedTransitions',
    field_names=[
        'observations', 'actions', 'rewards', 'next_observations', 'dones'
    ]
)


class SleepyToyEnv(gym.Env):
    """
    Environment with observation 0..4 and actions 0..2
    """

    def __init__(self):
        super(SleepyToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=2)
        self.current_state = -1
        self.terminal_state = 4
        self.max_sleep_time = 2

    def reset(self):
        self.current_state = 0
        #print("RESET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return self.current_state

    def step(self, action):
        assert self.action_space.contains(action), \
            "Action {0} is not valid".format(action)

        # 0 <= sleep_time < self.max_sleep_time
        sleep_time = random.randrange(0, self.max_sleep_time)
        time.sleep(sleep_time)

        self.current_state += sleep_time
        is_done = self.current_state >= self.terminal_state
        if is_done:
            reward = 10.0
            return self.terminal_state, reward, True, {}
        else:
            reward = 0.0
            return self.current_state, reward, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def make_env():
    def _make():
        # env = gym.make("CartPole-v0")
        env = SleepyToyEnv()
        return env
    return _make


class ExtendedReplayBuffer(ReplayBuffer):
    def append_rollout_vectorized_transitions(self, vectorized_transitions):
        for observation, action, reward, next_observation, done in zip(
                *vectorized_transitions
        ):
            transition = Transition(
                observation, action, reward, next_observation, done
            )
            self.buffer.append(transition)


def main():
    n_envs = 8
    env = AsyncVectorEnv(env_fns=[make_env() for _ in range(n_envs)])
    extended_replay_buffer = ExtendedReplayBuffer(capacity=10_000)

    total_train_start_time = time.time()

    total_time_steps = 20
    episode_rewards = np.zeros((n_envs,))
    episode_reward_lst = []

    observations = env.reset()

    for time_step in range(total_time_steps):
        actions = env.action_space.sample()
        next_observations, rewards, dones, infos = env.step(actions)

        transitions = VectorizedTransitions(
            observations, actions, rewards, next_observations, dones
        )
        extended_replay_buffer.append_vectorized_transitions(transitions)
        print("[{0:>3}] Observations: {1}, Actions: {2}, "
              "Rewards: {3}, Next Observations: {4}, Dones: {5}".format(
            time_step, observations, actions, rewards, next_observations, dones
        ))

        episode_rewards += rewards

        if any(dones):
            # print(episode_rewards[dones], len(episode_rewards[dones]), "****")
            episode_reward_lst.extend([episode_reward for episode_reward in episode_rewards[dones]])
            episode_rewards[dones] = 0.0
        # print(episode_reward_lst, "##########################")

        observations = next_observations

    total_training_time = time.time() - total_train_start_time
    total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
    print("Total Training End : {}".format(total_training_time))


if __name__ == "__main__":
    main()
