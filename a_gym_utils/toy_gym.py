# https://alexandervandekleut.github.io/gym-wrappers/
import gym
import numpy as np


class ToyEnv(gym.Env):
    """
    Environment with observation 0..4 and actions 0..2
    """

    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=2)
        self.current_state = -1
        self.terminal_state = 4

    def reset(self):
        self.current_state = 0
        print("RESET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return self.current_state

    def step(self, action):
        assert self.action_space.contains(action), \
            "Action {0} is not valid".format(action)

        self.current_state += action
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


class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.n = self.observation_space.n
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.n,))

    def observation(self, obs):
        new_obs = np.zeros(self.n)
        new_obs[obs] = 1
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
        assert isinstance(action, int)
        if action < 0:
            action = 0
        elif action > 1:
            action = 1
        else:
            action = action

        return action


def main():
    env = CustomActionWrapper(CustomRewardWrapper(CustomObservationWrapper(
        ToyEnv()
    )))
    T = 12
    observation = env.reset()
    for t in range(T):
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)

        print("[{0:>3}] Observation: {1}, Action: {2}, "
              "Reward: {3}, Next Observation: {4}, Done: {5}".format(
            t, observation, action, reward, next_observation, done
        ))

        if done:
            observation = env.reset()
        else:
            observation = next_observation


if __name__ == "__main__":
    main()
