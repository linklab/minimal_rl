# conda install matplotlib or pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np


linestyles = ['-', '--', ':']
legends = ["AGENT-1 WIN", "AGENT-2 WIN", "DRAW"]
GAME_STATUS_PRINT_PERIOD_EPISODES = 5000


class GameStatus:
    def __init__(self):
        self.num_player_1_win = 0
        self.num_player_2_win = 0
        self.num_draw = 0

        self.player_1_win_info_list = []
        self.player_2_win_info_list = []
        self.draw_info_list = []
        self.agent_1_episode_td_error = []
        self.agent_2_episode_td_error = []
        self.epsilon_list = []

        self.player_1_win_rate_over_100_games = []
        self.player_2_win_rate_over_100_games = []
        self.draw_rate_over_100_games = []

    def set_agent_1_episode_td_error(self, agent_1_episode_td_error):
        self.agent_1_episode_td_error.append(agent_1_episode_td_error)

    def set_agent_2_episode_td_error(self, agent_2_episode_td_error):
        self.agent_2_episode_td_error.append(agent_2_episode_td_error)


def draw_performance(game_status, max_episodes):
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 6)

    figure, _ = plt.subplots(nrows=4, ncols=1)

    plt.subplot(211)
    if game_status.agent_1_episode_td_error:
        plt.plot(
            range(1, max_episodes + 1, 100),
            game_status.agent_1_episode_td_error[::100],
            label="Agent 1"
        )
        #plt.plot(range(1, max_episodes + 1, 10000), game_status.agent_1_episode_td_error[::10000], label="Agent 1")
    if game_status.agent_2_episode_td_error:
        plt.plot(
            range(1, max_episodes + 1, 100),
            game_status.agent_2_episode_td_error[::100],
            label="Agent 2"
        )
        #plt.plot(range(1, max_episodes + 1, 10000), game_status.agent_2_episode_td_error[::10000], label="Agent 2")

    plt.xlabel('Episode')
    plt.ylabel('Q Learning TD Error')
    plt.legend()

    plt.subplot(212)
    for i in range(3):
        if i == 0:
            values = game_status.player_1_win_rate_over_100_games[::100]
        elif i == 1:
            values = game_status.player_2_win_rate_over_100_games[::100]
        else:
            values = game_status.draw_rate_over_100_games[::100]
        plt.plot(
            range(1, max_episodes + 1, 100),
            values,
            linestyle=linestyles[i],
            label=legends[i]
        )

    plt.xlabel('Episode')
    plt.ylabel('Winning Rate')
    plt.legend()


    plt.tight_layout()

    plt.show()
    plt.close()


def epsilon_scheduled(
    current_episode, last_scheduled_episodes,
    initial_epsilon, final_epsilon):
    fraction = min(current_episode / last_scheduled_episodes, 1.0)
    epsilon = min(
        initial_epsilon + fraction * (final_epsilon - initial_epsilon),
        initial_epsilon
    )
    return epsilon


def print_game_statistics(
    info, episode, epsilon, total_steps,
    game_status, agent_1=None, agent_2=None):
    if info['winner'] == 1:
        game_status.num_player_1_win += 1
    elif info['winner'] == -1:
        game_status.num_player_2_win += 1
    else:
        game_status.num_draw += 1

    game_status.player_1_win_info_list.append(1 if info['winner'] == 1 else 0)
    game_status.player_2_win_info_list.append(1 if info['winner'] == -1 else 0)
    game_status.draw_info_list.append(1 if info['winner'] == 0 else 0)

    game_status.player_1_win_rate_over_100_games.append(
        np.average(game_status.player_1_win_info_list[-100:]) * 100
    )
    game_status.player_2_win_rate_over_100_games.append(
        np.average(game_status.player_2_win_info_list[-100:]) * 100
    )
    game_status.draw_rate_over_100_games.append(
        np.average(game_status.draw_info_list[-100:]) * 100
    )

    if episode % GAME_STATUS_PRINT_PERIOD_EPISODES == 0:
        print("### GAMES DONE: {0} | episolon: {1:.2f} | total_steps: {2} | "
              "agent_1_win : agent_2_win : draw = {3} : {4} : {5} | "
              "winning_rate_over_recent_100_games --> {6:4.1f}% : {7:4.1f}% : {8:4.1f}%".format(
            episode, epsilon, total_steps,
            game_status.num_player_1_win, game_status.num_player_2_win, game_status.num_draw,
            game_status.player_1_win_rate_over_100_games[-1],
            game_status.player_2_win_rate_over_100_games[-1],
            game_status.draw_rate_over_100_games[-1]
        ))


def print_step_status(
    agent, state, action, next_state,
    reward, done, info, env, step_verbose, board_render
):
    if step_verbose:
        print("[{0}]|{1}|action: {2}|next_state: {3}|"
              "reward: {4:4.1f}|done: {5:5s}|info: {6}".format(
            agent.name, state, action, next_state, reward, str(done), info
        ))
    if board_render:
        env.BOARD_RENDER()