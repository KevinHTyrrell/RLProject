"""
Script to train an RL agent to play Pong on Atari
"""
import argparse

from ale_py import ALEInterface
from ale_py.roms import Pong
from Bots.pong import PongBot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Model configuration yaml file')
    parser.add_argument('--n_games', type=int, required=True, help='Number of games used to train model')

    args = parser.parse_args()
    config_yml              = args.config_file
    n_games_train           = args.n_games

    ale = ALEInterface()
    ale.loadROM(Pong)

    action_map = {
        0: 0,
        1: 3,
        2: 4
    }

    action_space_size = action_map.__len__()
    state_space_size = ale.getScreenRGB().shape

    config_file = f'Configs/{config_yml}.yml'
    atari_bot = PongBot(state_space_size, config_file, action_space_size)
    GAME_OVER = False
    for game_n in range(n_games_train):
        while not GAME_OVER:
            curr_space  = ale.getScreenRGB()
            bot_action  = atari_bot.get_action(curr_space)
            reward      = ale.act(action_map[bot_action])
            next_space  = ale.getScreenRGB()
            atari_bot.store_memory(curr_space, bot_action, next_space, reward)
            print(ale.getEpisodeFrameNumber(), '\t\t', bot_action, '\t\t', reward)
            GAME_OVER   = ale.game_over()
        atari_bot.end_game()
        GAME_OVER = False
