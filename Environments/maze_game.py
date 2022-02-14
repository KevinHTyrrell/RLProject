import time
from Bots.maze import BotMaze
from Environments.maze_grid import MazeGrid


class MazeGame:
    def __init__(self, height: int = 10, width: int = 10, n_pits: int = 10):
        self._grid = MazeGrid(height, width, n_pits)
        self._height    = height
        self._width     = width
        self._n_pits    = n_pits
        self._player = None

    def _reset(self):
        self._grid = MazeGrid(self._height, self._width, self._n_pits)

    def add_player(self, bot: BotMaze):
        self._player = bot

    def run(self, n_games: int = 100, time_delay: int = 2):
        for n_game in range(n_games):
            while self._grid.run_game():
                self._grid.print()
                curr_grid = self._grid.get_grid()
                action = self._player.get_action(curr_grid)
                reward = self._grid.move_player(action)
                next_grid = self._grid.get_grid()
                self._player.store_memory(curr_grid, action, next_grid, reward)
                time.sleep(time_delay)
            self._player.end_game()
            self._reset()

    def get_grid_size(self):
        return self._grid.get_shape()