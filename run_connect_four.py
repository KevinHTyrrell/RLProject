from Bots.connect_four import BotConnectFour
from Environments.connect_four_game import ConnectFourGame


if __name__ == "__main__":
    config_file = 'Configs/base_bot.yml'
    connect_four = ConnectFourGame()
    connect_four_shape = connect_four.get_grid_dims()
    player_one = BotConnectFour(state_shape=connect_four_shape, config_file=config_file, action_space=connect_four_shape[0])
    player_two = BotConnectFour(state_shape=connect_four_shape, config_file=config_file, action_space=connect_four_shape[0])
    connect_four.add_bot(player=1, bot=player_one)
    connect_four.add_bot(player=2, bot=player_two)
    connect_four.run_game()