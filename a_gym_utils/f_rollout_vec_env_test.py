import os
import time
import multiprocessing as mp
from collections import namedtuple

import numpy as np
from gym.vector import AsyncVectorEnv

from a_gym_utils.e_vec_env_test import SleepyToyEnv
from b_DQN.dqn_train_and_model_save import ReplayBuffer, Transition

RolloutVectorizedTransitions = namedtuple(
    typename='RolloutVectorizedTransitions',
    field_names=[
        'time_step', 'observations', 'actions', 'rewards', 'next_observations',
        'dones', 'infos'
    ]
)


def make_env():
    def _make():
        # env = gym.make("CartPole-v0")
        env = SleepyToyEnv()
        return env
    return _make


class RolloutExtendedReplayBuffer(ReplayBuffer):
    def append_rollout_vectorized_transitions(self, rollout_vectorized_transitions):
        for observation, action, reward, next_observation, done in zip(
                rollout_vectorized_transitions.observations,
                rollout_vectorized_transitions.actions,
                rollout_vectorized_transitions.rewards,
                rollout_vectorized_transitions.next_observations,
                rollout_vectorized_transitions.dones
        ):
            transition = Transition(
                observation, action, reward, next_observation, done
            )
            self.buffer.append(transition)


def learning(n_envs, buffer_capacity, pipe_conn):
    rollout_extended_replay_buffer = RolloutExtendedReplayBuffer(
        capacity=buffer_capacity
    )

    total_train_start_time = time.time()

    episode_rewards = np.zeros((n_envs,))
    episode_reward_lst = []

    while True:
        message = pipe_conn.recv()

        if message == "finish":
            break
        else:
            rollout_vectorized_transitions = message

        rollout_extended_replay_buffer.append_rollout_vectorized_transitions(
            rollout_vectorized_transitions
        )

        time_step, observations, actions, rewards, next_observations, dones, infos = rollout_vectorized_transitions

        print("[{0:>3}] Observations: {1}, Actions: {2}, Rewards: {3}, "
              "Next Observations: {4}, Dones: {5}, Replay Buffer: {6}".format(
            time_step, observations, actions, rewards, next_observations,
            dones, rollout_extended_replay_buffer.size()
        ))

        episode_rewards += rewards

        if any(dones):
            episode_reward_lst.extend(
                [episode_reward for episode_reward in episode_rewards[dones]]
            )
            episode_rewards[dones] = 0.0

    total_training_time = time.time() - total_train_start_time
    total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
    print("Total Training End : {}".format(total_training_time))


def rollout(n_envs, total_time_steps, pipe_conn):
    env = AsyncVectorEnv(env_fns=[make_env() for _ in range(n_envs)])

    observations = env.reset()

    for time_step in range(total_time_steps):
        actions = env.action_space.sample()
        next_observations, rewards, dones, infos = env.step(actions)

        rollout_vectorized_transitions = RolloutVectorizedTransitions(
            time_step, observations, actions, rewards, next_observations, dones, infos
        )

        pipe_conn.send(rollout_vectorized_transitions)

        observations = next_observations

    pipe_conn.send("finish")


def main():
    n_envs = 32
    actor_pipe_conn, agent_pipe_conn = mp.Pipe()

    actor = mp.Process(target=rollout, args=(n_envs, 20, actor_pipe_conn))
    agent = mp.Process(target=learning, args=(n_envs, 1000, agent_pipe_conn))
    actor.start()
    agent.start()

    while actor.is_alive() or agent.is_alive():
        time.sleep(1)
        actor.join()
        agent.join()


if __name__ == "__main__":
    main()
