import time
import multiprocessing as mp
from collections import deque
from gym.vector import SyncVectorEnv
from torch.multiprocessing import Process, cpu_count

from a_common.a_commons import make_sleepy_toy_env, NStepParallelVectorizedTransition
from a_common.b_models import Policy
from a_common.c_buffers import ReplayBuffer


class Actor(Process):
    def __init__(self, actor_id, n_vec_envs, policy, queue, time_steps, n_step, gamma):
        super(Actor, self).__init__()
        self.actor_id = actor_id
        self.n_vec_envs = n_vec_envs
        self.policy = policy
        self.queue = queue
        self.time_steps = time_steps
        self.n_step = n_step
        self.gamma = gamma
        self.is_vectorized_env_created = mp.Value('i', False)

    def run(self):
        env = SyncVectorEnv(env_fns=[make_sleepy_toy_env for _ in range(self.n_vec_envs)])
        self.is_vectorized_env_created.value = True

        histories = []
        for _ in range(self.n_vec_envs):
            histories.append(deque(maxlen=self.n_step))

        observations = env.reset()

        for time_step in range(self.time_steps):
            actions = self.policy.get_action(observations)
            next_observations, rewards, dones, infos = env.step(actions)

            for env_idx, (observation, action, next_observation, reward, done, info) in enumerate(
                    zip(observations, actions, next_observations, rewards, dones, infos)
            ):
                histories[env_idx].append(NStepParallelVectorizedTransition(
                    self.actor_id, time_step,
                    observation, action, next_observation, reward, done, info
                ))

                # print(env_idx, '->', observation, action, next_observation, reward, done, info, len(histories[env_idx]))

                if len(histories[env_idx]) == self.n_step or done:
                    n_step_transitions = tuple(histories[env_idx])
                    if n_step_transitions[-1].done:
                        next_observation = None
                        done = True
                    else:
                        next_observation = n_step_transitions[-1].next_observation
                        done = False

                    n_step_reward = 0.0
                    for n_step_transition in reversed(n_step_transitions):
                        n_step_reward = n_step_transition.reward + self.gamma * n_step_reward

                    n_step_transition = NStepParallelVectorizedTransition(
                        self.actor_id,
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

                    self.queue.put(n_step_transition)

                    histories[env_idx].clear()

            observations = next_observations

        self.queue.put(None)


class Learner(Process):
    def __init__(self, n_vec_envs, policy, queue, n_actors, buffer_capacity):
        super(Learner, self).__init__()
        self.n_vec_envs = n_vec_envs
        self.policy = policy
        self.queue = queue
        self.n_actors = n_actors

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.episode_reward = 0.0
        self.episode_reward_lst = []
        self.num_train_steps = 0

        self.n_actor_terminations = 0

    def run(self):
        total_train_start_time = time.time()

        while True:
            n_step_transition = self.queue.get()

            if n_step_transition is None:
                self.n_actor_terminations += 1
                if self.n_actor_terminations >= self.n_actors:
                    break
                else:
                    continue

            self.replay_buffer.append(n_step_transition)

            actor_id, time_step, observation, action, next_observation, reward, done, info = n_step_transition

            self.episode_reward += reward

            if len(self.replay_buffer) > 1:
                # TRAIN POLICY
                self.num_train_steps += 1

            print("[Actor ID: {0:2}, Time Step: {1:>3}] "
                  "Observation: {1}, Action: {2}, Next Observation: {3:14}, "
                  "Reward: {4}, Done: {5:5} || "
                  "Replay Buffer: {6}, Training Steps: {7}".format(
                actor_id,
                time_step + 1,
                str(observation), action, str(next_observation), reward, str(done),
                self.replay_buffer.size(), self.num_train_steps
            ))

            if done:
                self.episode_reward_lst.append(self.episode_reward)
                self.episode_reward = 0.0

        total_training_time = time.time() - total_train_start_time
        formatted_total_training_time = time.strftime(
            '%H:%M:%S', time.gmtime(total_training_time)
        )
        print("Total Training End : {}".format(formatted_total_training_time))
        print("Rate of Buffer Increase: {0:.3f}/1sec.".format(
            self.replay_buffer.size() / total_training_time
        ))
        print("Rate of Training Steps: {0:.3f}/1sec.".format(
            self.num_train_steps / total_training_time
        ))


def main():
    n_vec_envs = 4
    time_steps = 10
    buffer_capacity = 1000
    n_step = 2
    gamma = 0.99

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

    actors = [
        Actor(
            actor_id, n_vec_envs, policy, queue, time_steps, n_step, gamma
        ) for actor_id in range(n_actors)
    ]

    learner = Learner(n_vec_envs, policy, queue, n_actors, buffer_capacity)
    for actor in actors:
        actor.start()

    for actor in actors:
        while not actor.is_vectorized_env_created.value:
            time.sleep(0.1)

    learner.start()

    for actor in actors:
        while actor.is_alive():
            actor.join(timeout=1)

    while learner.is_alive():
        learner.join(timeout=1)


if __name__ == "__main__":
    main()
