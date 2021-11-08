import random
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class QNet(nn.Module):
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(QNet, self).__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_features, 128)  # fully connected
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action(self, obs, epsilon=0.1):
        out = self.forward(obs)

        coin = random.random() # 0.0과 1.0사이의 임의의 값을 반환
        if coin < epsilon:
            return random.randrange(0, self.n_actions)
        else:
            return out.argmax().item()  # argmax: 더 큰 값에 대응되는 인덱스 반환


class AtariCNN(nn.Module):
    def __init__(
            self, obs_shape: Tuple[int], n_actions: int, hidden_size: int = 256,
            device=torch.device("cpu")
    ):
        super(AtariCNN, self).__init__()

        input_channel = obs_shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        self.device = device

    def _get_conv_out(self, shape):
        cont_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(cont_out.size()))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        conv_out = self.conv(x)
        conv_out = torch.flatten(conv_out, 1)
        out = self.fc(conv_out)
        return out

    def get_action(self, observation, epsilon):
        def get_gym_action(action):
            if action == 0:
                gym_action = 0
            elif action == 1:
                gym_action = 2
            elif action == 2:
                gym_action = 3
            else:
                raise ValueError()

            return gym_action

        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            # Convert to Tensor
            observation = np.array(observation, copy=False)
            observation = torch.tensor(observation, device=self.device)

            # Add batch-dim
            if len(observation.shape) == 3:
                observation = observation.unsqueeze(dim=0)

            q_values = self.forward(observation)
            action = torch.argmax(q_values, dim=1)
            action = int(action.item())

        gym_action = get_gym_action(action)

        return action, gym_action


class Policy(nn.Module):
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1 if x.dim() == 2 else 0)
        return x

    def get_action(self, x, mode="train"):
        action_prob = self.forward(x)
        m = Categorical(probs=action_prob)

        if mode == "train":
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=1 if action_prob.dim() == 2 else 0)
        return action.numpy()

    def get_action_with_action_prob(self, x, mode="train"):
        action_prob = self.forward(x)
        m = Categorical(probs=action_prob)

        if mode == "train":
            action = m.sample()
            action_prob_selected = action_prob[action]
        else:
            action = torch.argmax(m.probs, dim=1 if action_prob.dim() == 2 else 0)
            action_prob_selected = None
        return action.numpy(), action_prob_selected


class ActorCritic(nn.Module):
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_pi = nn.Linear(128, n_actions)
        self.fc_v = nn.Linear(128, 1)
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def pi(self, x, softmax_dim=1):
        x = self.forward(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.forward(x)
        v = self.fc_v(x)
        return v

    def get_action(self, x, mode="train"):
        action_prob = self.pi(x)
        m = Categorical(probs=action_prob)

        if mode == "train":
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=1 if action_prob.dim() == 2 else 0)
        return action.numpy()
