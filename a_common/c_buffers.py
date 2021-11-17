import collections
from typing import Tuple

import numpy as np
import torch

from a_common.a_commons import Transition


class ReplayBuffer:
    def __init__(self, capacity, device=None):
        self.buffer = collections.deque(maxlen=capacity)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size: int) -> Tuple:
        # Get index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        # Sample
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (64, 4), (64, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions

        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards

        dones = np.array(dones, dtype=bool)
        # actions.shape, rewards.shape, dones.shape: (64, 1) (64, 1) (64,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        return observations, actions, next_observations, rewards, dones


class ReplayBufferForVectorizedEnvs(ReplayBuffer):
    def append(self, vectorized_transitions):
        """
        vectorized_transitions
          - observations: 15개의 observation
          - actions: 15개의 action
          - next_observations: 15개의 next_observations
          - rewards
          - dones
        :param vectorized_transitions:
        :return:
        """
        for observation, action, next_observation, reward, done in zip(
                *vectorized_transitions
        ):
            transition = Transition(
                observation, action, next_observation, reward, done
            )
            self.buffer.append(transition)

    def sample_all(self):
        return self.sample(batch_size=len(self.buffer))


class ReplayBufferForParallelVectorizedEnvs(ReplayBuffer):
    def append(self, parallel_vectorized_transitions):
        for observation, action, next_observation, reward, done in zip(
                parallel_vectorized_transitions.observations,
                parallel_vectorized_transitions.actions,
                parallel_vectorized_transitions.next_observations,
                parallel_vectorized_transitions.rewards,
                parallel_vectorized_transitions.dones
        ):
            transition = Transition(
                observation, action, next_observation, reward, done
            )
            self.buffer.append(transition)
