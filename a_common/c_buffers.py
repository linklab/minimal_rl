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

    def reset(self):
        self.buffer.clear()

    def sample(self, batch_size: int) -> Tuple:
        # Get index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        # Sample
        states, actions, next_states, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)

        # Convert to tensor
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        return states, actions, next_states, rewards, dones


class ReplayBufferForVectorizedEnvs(ReplayBuffer):
    def append(self, vectorized_transitions):
        for observation, action, next_observation, reward, done in zip(
                *vectorized_transitions
        ):
            if done:
                next_observation = None

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
            if done:
                next_observation = None

            transition = Transition(
                observation, action, next_observation, reward, done
            )
            self.buffer.append(transition)
