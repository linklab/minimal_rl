from f_TIC_TAC_TOE.a_env_tic_tac_toe import TicTacToe
from f_TIC_TAC_TOE.b_q_learning_agent import Q_Learning_Agent
from f_TIC_TAC_TOE.c_human_agent import Human_Agent
from f_TIC_TAC_TOE.d_dummy_agent import Dummy_Agent
from f_TIC_TAC_TOE.e_game_stats import draw_performance, print_game_statistics, print_step_status, GameStatus, \
    epsilon_scheduled

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 100

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 200

STEP_VERBOSE = False
BOARD_RENDER = False


# 선수 에이전트: Q-Learning 에이전트, 후수 에이전트: Dummy 에이전트
def q_learning_for_agent_1_vs_dummy():
    game_status = GameStatus()
    env = TicTacToe()

    agent_1 = Q_Learning_Agent(name="AGENT_1", env=env)
    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False

        agent_1_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # agent_1 스텝 수행
            action = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(
                agent_1, state, action, next_state,
                reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # reward: agent_1이 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                agent_1_episode_td_error += agent_1.q_learning(
                    state, action, None, reward, done, epsilon
                )

                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps,
                    game_status, agent_1, agent_2
                )
            else:
                # agent_2 스텝 수행
                action_2 = agent_2.get_action(next_state)
                next_state, reward, done, info = env.step(action_2)
                print_step_status(
                    agent_2, state, action_2, next_state,
                    reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # reward: agent_2가 착수하여 done=True
                    # agent_2가 이기면 -1.0, 비기면 0.0
                    agent_1_episode_td_error += agent_1.q_learning(
                        state, action, None, reward, done, epsilon
                    )

                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps,
                        game_status, agent_1, agent_2
                    )
                else:
                    agent_1_episode_td_error += agent_1.q_learning(
                        state, action, next_state, reward, done, epsilon
                    )

            state = next_state

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)

    draw_performance(game_status, MAX_EPISODES)

    # 훈련 종료 직후 완전 탐욕적으로 정책 설정
    agent_1.make_greedy_policy()

    return agent_1


def play_with_agent_1(agent_1):
    env = TicTacToe()
    env.print_board_idx()
    state = env.reset()

    agent_2 = Human_Agent(name="AGENT_2", env=env)
    current_agent = agent_1

    print()

    print("[Q-Learning 에이전트 차례]")
    env.render()

    done = False
    while not done:
        if isinstance(current_agent, Human_Agent):
            action = current_agent.get_action(state)
        else:
            action = current_agent.get_action(state, mode="PLAY")

        next_state, _, done, info = env.step(action)
        if current_agent == agent_1:
            print("     State:", state)
            print("   Q-value:", current_agent.get_q_values_for_one_state(state))
            print("    Policy:", current_agent.get_policy_for_one_state(state))
            print("    Action:", action)
            print("Next State:", next_state, end="\n\n")

        print("[{0}]".format(
            "당신(사람) 차례" if current_agent == agent_1 \
            else "Q-Learning 에이전트 차례"
        ))
        env.render()

        if done:
            if info['winner'] == 1:
                print("Q-Learning 에이전트가 이겼습니다.")
            elif info['winner'] == -1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            else:
                print("비겼습니다. 잘했습니다!")
        else:
            state = next_state

        if current_agent == agent_1:
            current_agent = agent_2
        else:
            current_agent = agent_1


if __name__ == '__main__':
    trained_agent_1 = q_learning_for_agent_1_vs_dummy()
    play_with_agent_1(trained_agent_1)