# https://alexandervandekleut.github.io/gym-wrappers/
import time

from a_common.a_commons import make_sleepy_toy_env, Transition
from a_common.b_models import Policy
from a_common.c_buffers import ReplayBuffer


def rl_main():
    env = make_sleepy_toy_env()
    policy = Policy(n_features=4, n_actions=3)
    replay_buffer = ReplayBuffer(capacity=1000)

    time_steps = 10
    episode_reward = 0.0
    num_train_steps = 0

    total_train_start_time = time.time()

    observation = env.reset()

    for time_step in range(time_steps):
        action = policy.get_action(observation)
        next_observation, reward, done, info = env.step(action)

        replay_buffer.append(Transition(
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=done
        ))

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
            episode_reward = 0.0
            observation = env.reset()
        else:
            observation = next_observation

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


if __name__ == "__main__":
    rl_main()
