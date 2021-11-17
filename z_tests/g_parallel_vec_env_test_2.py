import time
import torch.multiprocessing as mp
import numpy as np
from gym.vector import AsyncVectorEnv
from torch.multiprocessing import Process, cpu_count

from a_common.a_commons import make_sleepy_toy_env, ParallelVectorizedTransitions
from a_common.b_models import Policy
from a_common.c_buffers import ReplayBufferForParallelVectorizedEnvs


def rollout(actor_id, n_vec_envs, policy, queue, time_steps):
    env = AsyncVectorEnv(env_fns=[make_sleepy_toy_env for _ in range(n_vec_envs)])

    observations = env.reset()

    for time_step in range(time_steps):
        actions = policy.get_action(observations)
        next_observations, rewards, dones, infos = env.step(actions)

        parallel_vectorized_transitions = ParallelVectorizedTransitions(
            actor_id=actor_id,
            model_version=0,
            time_step=time_step,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            dones=dones,
            infos=infos
        )

        queue.put(parallel_vectorized_transitions)

        observations = next_observations

    queue.put(None)


def learning(n_vec_envs, policy, queue, n_actors, buffer_capacity):
    replay_buffer_for_parallel_vectorized_envs = ReplayBufferForParallelVectorizedEnvs(
        capacity=buffer_capacity
    )

    episode_rewards = np.zeros((n_vec_envs,))
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

        actor_id, model_version, time_step, observations, actions, next_observations, rewards, dones, infos \
            = parallel_vectorized_transitions

        episode_rewards += rewards

        if len(replay_buffer_for_parallel_vectorized_envs) > 1:
            # TRAIN POLICY
            num_train_steps += 1

            print("[Actor ID: {0:2}, Model Version: {1:2}, Time Step: {2:>3}] "
                  "Observations: {3}, Actions: {4}, Next Observations: {5}, "
                  "Rewards: {6}, Dones: {7} || Replay Buffer: {8}, Training Steps: {9}".format(
                actor_id,
                model_version,
                time_step + 1,
                str(np.array(observations).argmax(axis=1)),
                actions,
                str(np.array(next_observations).argmax(axis=1)),
                rewards,
                str(dones),
                replay_buffer_for_parallel_vectorized_envs.size(),
                num_train_steps
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
    n_vec_envs = 4
    time_steps = 10
    buffer_capacity = 1000

    queue = mp.Queue()
    policy = Policy(n_features=4, n_actions=3)

    n_cpu_cores = cpu_count()
    n_actors = n_cpu_cores - 1

    print("******************************************")
    print("CPU Cores: {0}".format(n_cpu_cores))
    print("Actors: {0}".format(n_actors))
    print("Envs per actor: {0}".format(n_vec_envs))
    print("Total numbers of envs: {0}".format(n_actors * n_vec_envs))
    print("******************************************")

    actors = []
    for actor_id in range(n_actors):
        actor = Process(
            target=rollout,
            args=(actor_id, n_vec_envs, policy, queue, time_steps)
        )
        actors.append(actor)
        actor.start()

    agent = Process(
        target=learning,
        args=(n_vec_envs, policy, queue, n_actors, buffer_capacity)
    )
    agent.start()

    while agent.is_alive():
        agent.join(timeout=1)

    for idx in range(n_actors):
        while actors[idx].is_alive():
            actors[idx].join(timeout=1)


if __name__ == "__main__":
    main()
