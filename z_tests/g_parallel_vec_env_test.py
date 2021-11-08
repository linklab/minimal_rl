import time
import torch.multiprocessing as mp
import numpy as np
from gym.vector import AsyncVectorEnv

from a_common.a_commons import make_env, ParallelVectorizedTransitions
from a_common.b_models import Policy
from a_common.c_buffers import ReplayBufferForParallelVectorizedEnvs


def rollout(n_envs, policy, queue, time_steps):
    env = AsyncVectorEnv(env_fns=[make_env for _ in range(n_envs)])

    observations = env.reset()

    for time_step in range(time_steps):
        actions = policy.get_action(observations)
        next_observations, rewards, dones, infos = env.step(actions)

        parallel_vectorized_transitions = ParallelVectorizedTransitions(
            time_step, observations, actions, next_observations, rewards, dones, infos
        )

        queue.put(parallel_vectorized_transitions)

        observations = next_observations

    queue.put(None)


def learning(n_envs, policy, queue, n_actors, buffer_capacity):
    replay_buffer_for_parallel_vectorized_envs = ReplayBufferForParallelVectorizedEnvs(
        capacity=buffer_capacity
    )

    episode_rewards = np.zeros((n_envs,))
    episode_reward_lst = []
    num_train_steps = 0

    n_actor_terminations = 0

    total_train_start_time = time.time()

    while True:
        parallel_vectorized_transitions = queue.get()

        if parallel_vectorized_transitions is None:
            n_actor_terminations += 1
            if n_actor_terminations == n_actors:
                break
            else:
                continue

        replay_buffer_for_parallel_vectorized_envs.append(
            parallel_vectorized_transitions
        )

        time_step, observations, actions, next_observations, rewards, dones, infos = parallel_vectorized_transitions

        episode_rewards += rewards

        if len(replay_buffer_for_parallel_vectorized_envs) > 1:
            # TRAIN POLICY
            num_train_steps += 1

        print("[{0:>3}] Observations: {1}, Actions: {2}, Next Observations: {3}, "
              "Rewards: {4}, Dones: {5} || "
              "Replay Buffer: {6}, Training Steps: {7}".format(
            time_step + 1,
            str(np.array(observations).argmax(axis=1)),
            actions,
            str(np.array(next_observations).argmax(axis=1)),
            rewards,
            str(dones),
            replay_buffer_for_parallel_vectorized_envs.size(), num_train_steps
        ))

        if any(dones):
            episode_reward_lst.extend(
                [episode_reward for episode_reward in episode_rewards[dones]]
            )
            episode_rewards[dones] = 0.0

    total_training_time = time.time() - total_train_start_time
    formatted_total_training_time = time.strftime(
        '%H:%M:%S', time.gmtime(total_training_time)
    )
    print("Total Training End : {}".format(formatted_total_training_time))
    print("Rate of Buffer Increase: {0:.3f}/1sec.".format(
        replay_buffer_for_parallel_vectorized_envs.size() / total_training_time
    ))
    print("Rate of Training Steps: {0:.3f}/1sec.".format(
        num_train_steps / total_training_time
    ))


def main():
    n_envs = 4
    n_actors = 2
    time_steps = 10
    buffer_capacity = 1000

    queue = mp.SimpleQueue()
    policy = Policy(n_features=4, n_actions=3)

    actors = []

    for _ in range(n_actors):
        actor = mp.Process(
            target=rollout,
            args=(n_envs, policy, queue, time_steps)
        )
        actors.append(actor)
        actor.start()

    agent = mp.Process(
        target=learning,
        args=(n_envs, policy, queue, n_actors, buffer_capacity)
    )
    agent.start()

    while agent.is_alive():
        agent.join(timeout=1)

    for idx in range(n_actors):
        while actors[idx].is_alive():
            actors[idx].join(timeout=1)


if __name__ == "__main__":
    main()
