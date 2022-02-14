import numpy as np
from random import choices
from Bots.base_bot import BaseBot
from Misc.numpy_fn import concat


class PongBot(BaseBot):
    def _set_bot_train_vals(self):
        self._train_args['batch_size']          = 128
        self._train_args['clear_memory']        = False
        self._train_args['train_epochs']        = 10
        self._train_args['train_sample_size']   = 1028
        self._train_args['incremental_save']    = True
        self._train_args['_save_dir']           = 'Models/Pong/'

    def _set_bot_vals(self):
        self._bot_args['alpha']                 = 0.005
        self._bot_args['epsilon']               = 1.0
        self._bot_args['epsilon_decay']         = 0.000001
        self._bot_args['epsilon_limit']         = 0.2
        self._bot_args['gamma']                 = 0.1

    def _set_reward_args(self):
        self._reward_args['coeff_reward']       = 10

    def _calc_reward(self, reward_args):
        coeff_rew = self._reward_args['coeff_reward']
        reward_raw = self._reward_args['reward']
        return coeff_rew * reward_raw

    def _get_train_data(self):
        """
        Modified to calculate q-value in a batch, instead of per instance in order to increase speed
        :return: None
        """
        train_y_list = list()
        sample_memory = choices(self._memory, k=self._train_args['train_sample_size'])

        initial_state_arr = None
        action_arr = None
        next_state_arr = None
        reward_arr = None
        for initial_state, action, next_state, reward in sample_memory:
            initial_state_arr = concat(initial_state, initial_state_arr)
            action_arr = concat(action, action_arr)
            next_state_arr = concat(next_state, next_state_arr)
            reward_arr = concat(reward, reward_arr)
        initial_state_arr = initial_state_arr.squeeze()
        next_state_arr = next_state_arr.squeeze()
        q_vals = self._get_q_values(initial_state_arr, action_arr, next_state_arr, reward_arr)
        current_rewards_arr = self._model.predict(initial_state_arr)

        for i in range(len(initial_state_arr)):
            current_rewards = current_rewards_arr[i]
            q_val = q_vals[i]
            current_rewards[action_arr[i]] = q_val
            current_rewards = np.expand_dims(current_rewards, axis=0)
            train_y_list.append(current_rewards)
        train_x = initial_state_arr
        train_y = np.concatenate(train_y_list, axis=0)
        return train_x, train_y

    def _get_q_values(self, initial_state_arr, action_arr, next_state_arr, reward_arr):
        """
        Modified to calculate q-value in a batch, instead of per instance in order to increase speed
        :return: None
        """
        initial_state_predictions_arr = self._model.predict(initial_state_arr, verbose=False).squeeze()
        next_state_predictions_arr = self._model.predict(next_state_arr, verbose=False).squeeze()

        reward_list = list()
        for i in range(len(initial_state_predictions_arr)):
            initial_state_predictions = initial_state_predictions_arr[i]
            next_state_predictions = next_state_predictions_arr[i]
            action = action_arr[i]
            reward = reward_arr[i]
            initial_q_max = initial_state_predictions[action]
            next_q_max = next_state_predictions[np.argmax(next_state_predictions)]
            q_current = (1 - self._bot_args['alpha']) * initial_q_max
            q_future = self._bot_args['alpha'] * (reward * self._bot_args['gamma'] * next_q_max)
            reward_list.append(q_current + q_future)
        return reward_list
