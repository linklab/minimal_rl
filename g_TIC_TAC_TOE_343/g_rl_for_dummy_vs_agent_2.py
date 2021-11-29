# 선수 에이전트: Dummy 에이전트, 후수 에이전트: RL 에이전트
from g_TIC_TAC_TOE_343.c_human_agent import Human_Agent
from g_TIC_TAC_TOE_343.b_dummy_agent import Dummy_Agent
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


def learning_for_dummy_vs_agent_2():
    game_status = GameStatus()

    env = TicTacToe343()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = TTTAgentDqn(
        name="AGENT_1", env=env, gamma=0.99, learning_rate=0.001,
        replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=500,
        min_buffer_size_for_training=100
    )
    # agent_2 = TTTAgentReinforce(
    #     name="AGENT_2", env=env, gamma=0.99, learning_rate=0.001
    # )
    # agent_2 = TTTAgentA2c(
    #     name="AGENT_2", env=env, gamma=0.99, learning_rate=0.001, batch_size=32
    # )

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        current_agent = agent_1

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_2_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # agent_1 스텝 수행
            action_1 = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action_1)
            print_step_status(
                current_agent, state, action_1, next_state,
                reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps,
                    game_status, agent_1, agent_2
                )

                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                # reward: agent_1이 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += agent_2.learning(
                        STATE_2, ACTION_2, None, -1.0 * reward, done, epsilon
                    )
            else:
                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += agent_2.learning(
                        STATE_2, ACTION_2, next_state, reward, done, epsilon
                    )

                # agent_2 스텝 수행
                state = next_state
                action = agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)
                print_step_status(
                    agent_2, state, action, next_state,
                    reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps,
                        game_status, agent_1, agent_2
                    )

                    # reward: agent_2가 착수하여 done=True
                    # agent_2가 이기면 -1.0, 비기면 0.0
                    agent_2_episode_td_error += agent_2.learning(
                        state, action, None, -1.0 * reward, done, epsilon
                    )
                else:
                    # agent_2에 방문한 현재 상태 및 수행한 행동 정보를
                    # 저장해 두었다가 추후 활용
                    STATE_2 = state
                    ACTION_2 = action

            state = next_state

        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

    game_status.agent_2_count_state_updates = agent_2.count_state_updates
    draw_performance(game_status, MAX_EPISODES)

    # 훈련 종료 직후 완전 탐욕적으로 정책 설정
    # 아래 내용 불필요 --> agent_1.get_action(state, epsilon=0.0)으로 해결 가능
    # agent_2.make_greedy_policy()

    return agent_2


def play_with_agent_2(agent_2):
    env = TicTacToe343()
    env.print_board_idx()
    state = env.reset()

    agent_1 = Human_Agent(name="AGENT_1", env=env)
    current_agent = agent_1

    print()

    print("[당신(사람) 차례]")
    env.render()

    done = False
    while not done:
        action = current_agent.get_action(state)
        next_state, _, done, info = env.step(action)
        if current_agent == agent_2:
            print("     State:", state)
            print("   Q-value:", current_agent.get_q_values_for_one_state(state))
            print("    Policy:", current_agent.get_policy_for_one_state(state))
            print("    Action:", action)
            print("Next State:", next_state, end="\n\n")

        print("[{0}]".format(
            "Q-Learning 에이전트 차례" if current_agent == agent_1 \
            else "당신(사람) 차례"
        ))
        env.render()

        if done:
            if info['winner'] == 1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            elif info['winner'] == -1:
                print("Q-Learning 에이전트가 이겼습니다.")
            else:
                print("비겼습니다. 잘했습니다!")
        else:
            state = next_state

        if current_agent == agent_1:
            current_agent = agent_2
        else:
            current_agent = agent_1


if __name__ == '__main__':
    trained_agent_2 = learning_for_dummy_vs_agent_2()
    play_with_agent_2(trained_agent_2)