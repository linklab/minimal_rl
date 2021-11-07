import time
import multiprocessing as mp
from collections import deque

from gym.vector import AsyncVectorEnv

from a_common.a_commons import make_env, NStepParallelVectorizedTransition
from a_common.b_models import Policy
from a_common.c_buffers import ReplayBuffer


def rollout(n_envs, policy, queue, total_time_steps, n_step, gamma):
    env = AsyncVectorEnv(env_fns=[make_env for _ in range(n_envs)])
    histories = []
    for _ in range(n_envs):
        histories.append(deque(maxlen=n_step))

    observations = env.reset()

    for time_step in range(total_time_steps):
        actions = policy.get_action(observations)
        next_observations, rewards, dones, infos = env.step(actions)

        for env_idx, (observation, action, next_observation, reward, done, info) in enumerate(
                zip(observations, actions, next_observations, rewards, dones, infos)
        ):
            histories[env_idx].append(NStepParallelVectorizedTransition(
                time_step, observation, action, next_observation, reward, done, info
            ))

            print(env_idx, '->', observation, action, next_observation, reward, done, info, len(histories[env_idx]))

            if len(histories[env_idx]) == n_step or done:
                n_step_transitions = tuple(histories[env_idx])
                if n_step_transitions[-1].done:
                    next_observation = None
                    done = True
                else:
                    next_observation = n_step_transitions[-1].next_observation
                    done = False

                n_step_reward = 0.0
                for n_step_transition in reversed(n_step_transitions):
                    n_step_reward = n_step_transition.reward + gamma * n_step_reward

                n_step_transition = NStepParallelVectorizedTransition(
                    n_step_transitions[0].time_step,
                    n_step_transitions[0].observation,
                    n_step_transitions[0].action,
                    next_observation,
                    n_step_reward,
                    done,
                    n_step_transitions[-1].info
                )

                n_step_transition.info["real_num_steps"] = len(n_step_transitions)
                n_step_transition.info["env_idx"] = env_idx

                queue.put(n_step_transition)

                histories[env_idx].clear()

        observations = next_observations

    queue.put(None)


def learning(policy, queue, n_actors, buffer_capacity):
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    episode_reward = 0.0
    episode_reward_lst = []
    num_train_steps = 0

    n_actor_terminations = 0

    total_train_start_time = time.time()

    while True:
        n_step_transition = queue.get()

        if n_step_transition is None:
            n_actor_terminations += 1
            if n_actor_terminations == n_actors:
                break
            else:
                continue

        replay_buffer.append(n_step_transition)

        time_step, observation, action, next_observation, reward, done, info = n_step_transition

        episode_reward += reward

        if len(replay_buffer) > 1:
            # TRAIN POLICY
            num_train_steps += 1

        print("[{0:>3}] Observation: {1}, Action: {2}, Next Observation: {3:14}, "
              "Reward: {4}, Done: {5:5} || "
              "Replay Buffer: {6}, Training Steps: {7}".format(
            time_step + 1,
            str(observation), action, str(next_observation), reward, str(done),
            replay_buffer.size(), num_train_steps
        ))

        if done:
            episode_reward_lst.append(episode_reward)
            episode_reward = 0.0

    total_training_time = time.time() - total_train_start_time
    formatted_total_training_time = time.strftime(
        '%H:%M:%S', time.gmtime(total_training_time)
    )
    print("Total Training End : {}".format(formatted_total_training_time))
    print("Rate of Buffer Increase: {0:.3f}/1sec.".format(
        replay_buffer.size() / total_training_time
    ))
    print("Rate of Training Steps: {0:.3f}/1sec.".format(
        num_train_steps / total_training_time
    ))


def main():
    n_envs = 4
    n_actors = 2
    total_time_steps = 10
    buffer_capacity = 1000
    n_step = 2
    gamma = 0.99

    queue = mp.SimpleQueue()
    policy = Policy(n_features=4, n_actions=3)

    actors = []

    for _ in range(n_actors):
        actor = mp.Process(
            target=rollout,
            args=(n_envs, policy, queue, total_time_steps, n_step, gamma)
        )
        actors.append(actor)
        actor.start()

    agent = mp.Process(
        target=learning,
        args=(policy, queue, n_actors, buffer_capacity)
    )
    agent.start()

    while agent.is_alive():
        agent.join(timeout=1)

    for idx in range(n_actors):
        while actors[idx].is_alive():
            actors[idx].join(timeout=1)


if __name__ == "__main__":
    main()
