import collections
import numpy as np
from Bots.base_bot import BaseBot
from Environments.connect_four_grid import ConnectFourGrid


class ConnectFourGame:
    def __init__(self):
        self._player_one    = 1
        self._player_two    = 2
        self._player_dict   = dict()
        self._connection    = 4
        self._grid          = ConnectFourGrid()
        self._run_game      = True
        self._initialize_players()

    def reset(self):
        self._grid = ConnectFourGrid()
        self._run_game = True

    def _initialize_players(self):
        self._player_dict[1] = None
        self._player_dict[2] = None

    def add_bot(self, player, bot: BaseBot):
        self._player_dict[player] = bot

    def _check_arr(self, selected_arr):
        piece_count_dict = collections.Counter(selected_arr)
        for piece in [self._player_one, self._player_two]:
            piece_count = piece_count_dict.get(piece)
            if piece_count is None:
                continue
            if piece_count >= self._connection:
                split_row = np.split(selected_arr, np.where(np.diff(selected_arr) != 0)[0] + 1)
                count_result = [len(x) >= self._connection for x in split_row]
                if any(count_result):
                    return True
        return False

    def _check_cols(self):
        n_cols = self._grid.get_width()
        for col in range(n_cols):
            selected_col = self._grid.get_grid()[:, col]
            if self._check_arr(selected_col):
                return True
        return False

    def _check_rows(self):
        n_rows = self._grid.get_height()
        for row in range(n_rows):
            selected_row = self._grid.get_grid()[row, :]
            if self._check_arr(selected_row):
                return True
        return False

    def _check_diag(self):
        for row in range(self._grid.get_height() - self._connection):
            for col in range(self._grid.get_width() - self._connection):
                selected_grid = self._grid.get_grid()[row:(row+self._connection), col:(col+self._connection)]
                grid_result = [self._check_arr(np.diag(grd)) for grd in [selected_grid, selected_grid[:, ::-1]]]
                if any(grid_result):
                    return True
        return False

    def _check_win(self):
        row_win = self._check_rows()
        col_win = self._check_cols()
        diag_win = self._check_diag()
        if any([row_win, col_win, diag_win]):
            return True
        return False

    def _end_game(self):
        self._run_game = False

    def _get_max(self, player):
        row_max = self._get_rows_max(player)
        col_max = self._get_cols_max(player)
        diag_max = self._get_diag_max(player)
        return np.max([row_max, col_max, diag_max])

    def _get_arr_max(self, player, selected_arr):
        global_max = 0
        max_length = 0
        split_idx = np.where(np.diff(selected_arr) != 0)[0] + 1
        split_row = np.split(selected_arr, split_idx)
        for chunk_idx in range(len(split_row)):
            chunk = split_row[chunk_idx]
            if player in chunk and len(chunk) > global_max:
                if chunk_idx == 0:
                    if split_row.__len__() > 1 and split_row[chunk_idx+1][0] == 0:
                        max_length = len(chunk)
                elif chunk_idx == len(split_row) - 1:
                    if split_row[chunk_idx-1][-1] == 0:
                        max_length = len(chunk)
                else:
                    if split_row[chunk_idx + 1][0] == 0 or split_row[chunk_idx-1][-1] == 0:
                        max_length = len(chunk)
                if max_length > global_max:
                    global_max = max_length
        return global_max

    def _get_cols_max(self, player: int):
        global_max = -1
        n_cols = self._grid.get_width()
        for col in range(n_cols):
            selected_col = self._grid.get_grid()[:, col]
            row_max = self._get_arr_max(player, selected_col)
            if row_max > global_max:
                global_max = row_max
        return global_max

    def _get_diag_max(self, player: int):
        global_max = -1
        for row in range(self._grid.get_height() - self._connection):
            for col in range(self._grid.get_width() - self._connection):
                selected_grid = self._grid.get_grid()[row:(row+self._connection), col:(col+self._connection)]
                diag_max = np.max([self._get_arr_max(player, np.diag(grd)) for grd in [selected_grid, selected_grid[:, ::-1]]])
                if diag_max > global_max:
                    global_max = diag_max
        return global_max

    def _get_rows_max(self, player: int):
        global_max = -1
        n_rows = self._grid.get_height()
        for row in range(n_rows):
            selected_row = self._grid.get_grid()[row, :]
            row_max = self._get_arr_max(player, selected_row)
            if row_max > global_max:
                global_max = row_max
        return global_max

    def _get_player_input(self, player: int):
        print(f'Player {player} enter column?:', end='', flush=True)
        insert_col = input()
        valid_move = self._validate_input(insert_col)
        if valid_move:
            insert_col_int = int(insert_col)
            return insert_col_int
        else:
            print('INVALID MOVE')
            input()
            return None

    def _validate_input(self, user_input):
        if user_input.isnumeric():
            user_input = int(user_input)
        else:
            return False
        if user_input >= self._grid.get_width():
            return False
        if user_input < 0:
            return False
        return self._grid.check_valid_move(col=user_input)

    def run_game(self):
        n_iterations = 0
        while self._run_game:
            n_player = (n_iterations % 2) + 1
            player = self._player_dict.get(n_player)
            self._grid.print()
            if player is None:
                col_to_insert = self._get_player_input(n_player)
                if col_to_insert is None:
                    continue
                self._grid.insert_piece(piece=n_player, col=col_to_insert)
            else:
                # get bot action, insert piece #
                initial_state = self._grid.get_grid().copy()
                col_to_insert = player.get_action(initial_state)
                self._grid.insert_piece(piece=n_player, col=col_to_insert)
                result_state = self._grid.get_grid().copy()
                self_reward, opp_reward = self.calc_reward(n_player)
                reward_dict = {'self_reward': self_reward, 'opp_reward': opp_reward}
                player.store_memory(initial_state, col_to_insert, result_state, reward_args=reward_dict)
            n_iterations += 1
            is_win = self._check_win()
            if is_win:
                print(f'\n\n\n\n\n\n\nPLAYER {n_player} WINS')
                self._grid.print()
                self._end_game()

    def calc_reward(self, player: int):
        opp_player = (player % 2) + 1
        return self._get_max(player), self._get_max(opp_player)

    def get_grid_dims(self):
        return self._grid.get_grid_shape()