from Bots.maze import BotMaze
from Environments.maze_game import MazeGame


if __name__ == "__main__":
    config_file = 'Configs/maze.yml'
    game = MazeGame(10, 10, 20)
    state_space = game.get_grid_size()
    action_space = 4
    player = BotMaze(state_space, config_file, action_space)
    game.add_player(player)
    game.run()