import numpy as np
from random import randint


class MazeGrid:
    def __init__(self, height: int = 10, width: int = 10, n_pits: int = 10):
        self._height        = height
        self._width         = width
        self._n_pits        = n_pits
        self._player_loc    = [0, 0]
        self._trophy_loc    = [9, 9]
        self._max_dist      = height + width
        self._player_char   = 1
        self._pit_char      = 5
        self._trophy_char   = 9
        self._grid_char     = 0
        self._pit_reward    = -10
        self._trophy_reward = 10
        self._ill_reward    = 0
        self._pit_list      = list()
        self._grid          = None
        self._run_game      = True
        self._initial_config()

    def _initial_config(self):
        self._create_pits()
        self._render_grid()

    def _reset_grid(self):
        grid_flat = np.repeat(self._grid_char, self._width * self._height)
        grid_reshaped = np.reshape(grid_flat, newshape=(self._height, self._width))
        return grid_reshaped

    def _create_pits(self):
        while len(self._pit_list) < self._n_pits:
            pit_x = randint(0, self._width - 1)
            pit_y = randint(0, self._height - 1)
            if pit_x == 0 and pit_y == 0:
                continue
            if pit_x == 9 and pit_y == 9:
                continue
            self._pit_list.append((pit_y, pit_x))

    def _render_grid(self):
        self._grid = self._reset_grid()
        self._grid[self._player_loc[0], self._player_loc[1]] = self._player_char
        self._grid[self._trophy_loc[0], self._trophy_loc[1]] = self._trophy_char
        for i in range(self._n_pits):
            x, y = self._pit_list[i]
            self._grid[x, y] = self._pit_char
        return True

    def _check_collision(self):
        for i in range(self._n_pits):
            x, y = self._pit_list[i]
            if x == self._player_loc[0] and y == self._player_loc[1]:
                return True
        return False

    def _check_win(self):
        if self._player_loc[0] == self._trophy_loc[0] and self._player_loc[1] == self._trophy_loc[1]:
            return True
        return False

    def _get_dist(self, x_loc, y_loc):
        x_diff = abs(x_loc - self._trophy_loc[0])
        y_diff = abs(y_loc - self._trophy_loc[1])
        return x_diff + y_diff

    def _calc_reward(self, x_loc: int, y_loc: int):
        if x_loc < 0 or x_loc > 9 or y_loc < 0 or y_loc > 9:        # illegal move
            return self._ill_reward
        if self._grid[y_loc, x_loc] == self._pit_char:
            self._run_game = False
            return self._pit_reward
        elif self._grid[y_loc, x_loc] == self._trophy_char:
            self._run_game = False
            return self._trophy_reward
        else:
            curr_dist = self._get_dist(self._player_loc[1], self._player_loc[0])
            new_dist = self._get_dist(x_loc, y_loc)
            return curr_dist - new_dist

    def get_grid(self):
        return self._grid

    def move_player(self, player_move: int):
        # 0 down 1 right 2 up 3 left
        y_loc = self._player_loc[0]
        x_loc = self._player_loc[1]
        if player_move == 0:
            y_loc -= 1
        elif player_move == 1:
            x_loc += 1
        elif player_move == 2:
            y_loc += 1
        elif player_move == 3:
            x_loc -= 1
        reward = self._calc_reward(x_loc, y_loc)
        if reward != self._ill_reward:
            self._grid[self._player_loc[0], self._player_loc[1]] = self._grid_char
            self._player_loc = [y_loc, x_loc]
        return reward

    def print(self):
        self._render_grid()
        print(self._grid[::-1])
        print('')

    def run_game(self):
        return self._run_game

    def get_shape(self):
        return *self._grid.shape, 1
