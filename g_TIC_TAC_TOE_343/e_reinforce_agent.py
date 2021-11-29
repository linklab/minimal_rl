import random
from torch import optim

from g_TIC_TAC_TOE_343.d_common import Policy


class TTTAgentReinforce:
    def __init__(self, name, env, gamma, learning_rate):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.buffer = []

        self.policy = Policy(n_features=12, n_actions=12)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

    def get_action(self, state, epsilon=0.0):
        available_action_ids = state.get_available_actions()
        action_id = random.choice(available_action_ids)

        # TODO

        return action_id

    def learning(self, state, action, next_state, reward, done, epsilon):
        loss = 0.0

        # TODO

        return loss
