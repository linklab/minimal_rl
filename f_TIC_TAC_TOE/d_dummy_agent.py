import random

class Dummy_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def get_action(self, state):
        available_action_ids = state.get_available_actions()
        # E.g., available_action_ids = [2, 3, 5, 6, 7]
        action_id = random.choice(available_action_ids)
        return action_id
