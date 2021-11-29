# 선수 에이전트: RL 에이전트, 후수 에이전트: RL 에이전트
from g_TIC_TAC_TOE_343.f_game_stats import draw_performance, print_game_statistics, print_step_status, \
    epsilon_scheduled, GameStatus
from g_TIC_TAC_TOE_343.a_env_tic_tac_toe_343 import TicTacToe343
from g_TIC_TAC_TOE_343.e_a2c_agent import TTTAgentA2c
from g_TIC_TAC_TOE_343.e_dqn_agent import TTTAgentDqn
from g_TIC_TAC_TOE_343.e_reinforce_agent import TTTAgentReinforce

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 50_000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 100_000

STEP_VERBOSE = False
BOARD_RENDER = False

def learning_for_self_play():
    game_status = GameStatus()

    env = TicTacToe343()

    self_agent_1 = TTTAgentDqn(
        name="AGENT_1", env=env, gamma=0.99, learning_rate=0.001,
        replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=500,
        min_buffer_size_for_training=100
    )
    # self_agent_1 = TTTAgentReinforce(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.001
    # )
    # self_agent_1 = TTTAgentA2c(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.001, batch_size=32
    # )

    self_agent_2 = TTTAgentDqn(
        name="AGENT_2", env=env, gamma=0.99, learning_rate=0.001,
        replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=500,
        min_buffer_size_for_training=100
    )
    self_agent_2.q = self_agent_1.q

    # self_agent_2 = TTTAgentReinforce(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.001
    # )
    # self_agent_2.policy = self_agent_1.policy

    # self_agent_2 = TTTAgentA2c(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.001, batch_size=32
    # )
    # self_agent_2.actor_critic_model = self_agent_1.actor_critic_model



    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_1_episode_td_error = 0.0
        agent_2_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # self_agent_1 스텝 수행
            action = self_agent_1.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(
                self_agent_1, state, action, next_state,
                reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps,
                    game_status, self_agent_1, self_agent_2
                )

                # reward: self_agent_1가 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                agent_1_episode_td_error += self_agent_1.learning(
                    state, action, None, reward, done, epsilon
                )

                # 미루워 두었던 self_agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += self_agent_2.learning(
                        STATE_2, ACTION_2, None, -1.0 * reward, done, epsilon
                    )
            else:
                # 미루워 두었던 self_agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += self_agent_2.learning(
                        STATE_2, ACTION_2, next_state, reward, done, epsilon
                    )

                # self_agent_1이 방문한 현재 상태 및 수행한
                # 행동 정보를 저장해 두었다가 추후 활용
                STATE_1 = state
                ACTION_1 = action

                # self_agent_2 스텝 수행
                state = next_state
                action = self_agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)
                print_step_status(
                    self_agent_2, state, action, next_state,
                    reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps,
                        game_status, self_agent_1, self_agent_2
                    )

                    # reward: self_agent_2가 착수하여 done=True
                    # self_agent_2가 이기면 -1.0, 비기면 0.0
                    agent_2_episode_td_error += self_agent_2.learning(
                        state, action, None, -1.0 * reward, done, epsilon
                    )

                    # 미루워 두었던 self_agent_1의 배치에 transition 정보 추가
                    agent_1_episode_td_error += self_agent_1.learning(
                        STATE_1, ACTION_1, None, reward, done, epsilon
                    )
                else:
                    # self_agent_2이 방문한 현재 상태 및 수행한
                    # 행동 정보를 저장해 두었다가 추후 활용
                    STATE_2 = state
                    ACTION_2 = action

                    # 미루워 두었던 self_agent_1의 배치에 transition 정보 추가
                    agent_1_episode_td_error += self_agent_1.learning(
                        STATE_1, ACTION_1, next_state, reward, done, epsilon
                    )

            state = next_state

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)
        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

    game_status.agent_1_count_state_updates = self_agent_1.count_state_updates
    game_status.agent_2_count_state_updates = self_agent_2.count_state_updates
    draw_performance(game_status, MAX_EPISODES)

    # 훈련 종료 직후 완전 탐욕적으로 정책 설정
    # self_agent_1과 self_agent_2는 동일한 에이전트이므로 self_agent_1에 대해서만 정책 설정
    # 아래 내용 불필요 --> agent_1.get_action(state, epsilon=0.0)으로 해결 가능
    # self_agent_1.make_greedy_policy()

    # self_agent_1과 self_agent_2는 동일한 에이전트이므로 self_agent_1만 반환
    return self_agent_1


def self_play(self_agent):
    MAX_EPISODES = 10000
    VERBOSE = False

    env = TicTacToe343()

    agent_1 = self_agent
    agent_2 = self_agent

    agent_2.q_table = agent_1.q_table
    agent_2.policy = agent_1.policy

    current_agent = agent_1

    game_status = GameStatus()
    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        if VERBOSE:
            print("[시작 상태]")
            env.render()

        done = False
        while not done:
            total_steps += 1
            action = current_agent.get_action(state)

            next_state, _, done, info = env.step(action)

            if VERBOSE:
                print("[{0}]".format("Q-Learning 에이전트 1" if current_agent == agent_1 else "Q-Learning 에이전트 2"))
                env.render()

            if done:
                if VERBOSE:
                    if info['winner'] == 1:
                        print("Q-Learning 에이전트 1이 이겼습니다.")
                    elif info['winner'] == -1:
                        print("Q-Learning 에이전트 2가 이겼습니다!")
                    else:
                        print("비겼습니다!")

                done = done
                print_game_statistics(info, episode, 0.0, total_steps, game_status)
            else:
                state = next_state

            if current_agent == agent_1:
                current_agent = agent_2
            else:
                current_agent = agent_1


if __name__ == '__main__':
    trained_self_agent = learning_for_self_play()
    self_play(trained_self_agent)
