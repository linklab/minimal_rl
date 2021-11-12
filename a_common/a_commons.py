import random
import time
from collections import namedtuple
import numpy as np
import gym
import torch
from gym.vector import AsyncVectorEnv

Transition = namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done']
)


VectorizedTransitions = namedtuple(
    typename='VectorizedTransitions',
    field_names=[
        'observations', 'actions', 'next_observations', 'rewards', 'dones'
    ]
)


ParallelVectorizedTransitions = namedtuple(
    typename='ParallelVectorizedTransitions',
    field_names=[
        'actor_id', 'time_step', 'observations', 'actions', 'next_observations',
        'rewards', 'dones', 'infos'
    ]
)

NStepParallelVectorizedTransition = namedtuple(
    typename='NStepParallelVectorizedTransition',
    field_names=[
        'actor_id', 'time_step', 'observation', 'action', 'next_observation',
        'reward', 'done', 'info'
    ]
)


class SleepyToyEnv(gym.Env):
    """
    Environment with observation 0..3 and actions 0..2
    """

    def __init__(self):
        super(SleepyToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=4)  # 0, 1, 2, 3
        self.action_space = gym.spaces.Discrete(n=3) # 0, 1, 2
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

        time.sleep(action)

        self.current_state += action

        is_done = self.current_state >= self.terminal_state
        if is_done:
            reward = 10.0
            return None, reward, True, {}
        else:
            reward = 0.0
            return self.current_state, reward, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.discrete_observation_space_n = self.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.discrete_observation_space_n,)
        )  # [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]

    def observation(self, observation):  # Observation --> One-hot vector
        if observation is None:
            return None
        new_obs = np.zeros(self.discrete_observation_space_n) # [0, 0, 0, 0]
        new_obs[observation] = 1  # [0, 1, 0, 0]
        return new_obs


class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # modify rew
        if reward == 0.0:
            reward = -1.0
        return reward


class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if action < 0:
            action = 0
        return action


def make_sleepy_toy_env():
    # env = SleepyToyEnv()

    env = CustomActionWrapper(CustomRewardWrapper(CustomObservationWrapper(
        SleepyToyEnv()
    )))
    return env


def make_gym_env(env_name):
    def _make():
        env = gym.make(env_name)
        return env

    return _make


def make_gym_vec_env(env_name, n_vec_envs=4):
    env = AsyncVectorEnv(env_fns=[make_gym_env(env_name) for _ in range(n_vec_envs)])
    test_env = gym.make(env_name)
    return env, test_env
