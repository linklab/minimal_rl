import random
import time

import gym
from gym.vector import SyncVectorEnv, AsyncVectorEnv


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
        self.max_sleep_time = 3

    def reset(self):
        self.current_state = 0
        print("RESET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # 0 <= rand < self.max_sleep_time
        time.sleep(random.randrange(0, self.max_sleep_time))
        return self.current_state

    def step(self, action):
        assert self.action_space.contains(action), \
            "Action {0} is not valid".format(action)

        # 0 <= rand < self.max_sleep_time
        time.sleep(random.randrange(0, self.max_sleep_time))

        self.current_state += 1
        is_done = self.current_state == self.terminal_state
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


def main():
    n_env = 5
    env = AsyncVectorEnv(env_fns=[make_env() for _ in range(n_env)])

    total_train_start_time = time.time()

    T = 12
    observations = env.reset()
    for t in range(T):
        actions = env.action_space.sample()
        next_observations, rewards, dones, infos = env.step(actions)

        print("[{0}] Observation: {1}, Action: {2}, "
              "Reward: {3}, Next Observation: {4}".format(
            t, observations, actions, rewards, next_observations, dones
        ))

        observations = next_observations

    total_training_time = time.time() - total_train_start_time
    total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
    print("Total Training End : {}".format(total_training_time))


if __name__ == "__main__":
    main()
