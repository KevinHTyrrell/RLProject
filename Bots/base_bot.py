import numpy as np
import os.path
import yaml
from random import randint, choices
from tensorflow import keras

from abc import ABC
from Framework.model_builder import ModelBuilder


class BaseBot(ABC):
    def __init__(self, state_shape, config_file, action_space):
        self._action_space_size     = action_space
        self._state_shape           = state_shape
        self._move_n                = 0
        self._game_number           = 0
        self._model                 = None
        self._memory                = list()
        self._bot_args              = dict()
        self._reward_args           = dict()
        self._train_args            = dict()
        self._model_config_file     = config_file
        self._model_builder         = ModelBuilder()
        self._build_model()
        self._set_bot_train_vals()
        self._set_bot_vals()

    def _set_bot_train_vals(self):
        self._train_args['batch_size']          = 32
        self._train_args['clear_memory']        = False
        self._train_args['train_epochs']        = 10
        self._train_args['train_sample_size']   = 256
        self._train_args['incremental_save']    = False
        self._train_args['_save_dir']           = None

    def _set_bot_vals(self):
        self._bot_args['alpha']                 = 0.005
        self._bot_args['epsilon']               = 1.0
        self._bot_args['epsilon_decay']         = 0.001
        self._bot_args['epsilon_limit']         = 0.2
        self._bot_args['gamma']                 = 0.1

    def _set_reward_args(self):
        NotImplementedError("NO REWARD ARGUMENTS SUPPLIED")

    def get_action(self, current_state):
        if self._bot_args['epsilon'] > self._bot_args['epsilon_limit']:
            action = randint(0, self._action_space_size - 1)
            self._bot_args['epsilon'] *= (1 - self._bot_args['epsilon_decay'])
        else:
            current_state = np.expand_dims(current_state, axis=-1)
            action = self._compute_action(current_state)
        if len(self._memory) != 0 and len(self._memory) % self._train_args['train_sample_size']  == 0:
            self._train()
        self._move_n += 1
        return action

    def store_memory(self, initial_state, action, next_state, reward=None, reward_args=None):
        assert reward is not None or reward_args is not None, "NEED A REWARD OR ARGS TO CALCULATE A REWARD"
        if reward_args is not None:
            reward = self._calc_reward(reward_args)
        initial_state = np.expand_dims(initial_state, axis=(0, -1))
        next_state = np.expand_dims(next_state, axis=(0, -1))
        self._memory.append((initial_state, action, next_state, reward))

    def _compute_action(self, curr_state):
        curr_state = np.expand_dims(curr_state, axis=0)
        model_predictions = self._model.predict(curr_state).squeeze()
        action = np.argmax(model_predictions)
        return action

    def _build_model(self):
        with open(self._model_config_file, 'r') as f:
            yaml_contents = yaml.safe_load(f)
        bot_config = yaml_contents['Bot']

        builder_output = self._model_builder.build_tensor_model(input_dims=self._state_shape, model_config=bot_config)
        input_layer, current_layer, build_layer_list = builder_output
        output_layer = keras.layers.Dense(self._action_space_size, activation='linear')(current_layer)
        model = keras.models.Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='mean_squared_error')
        self._model = model

    def _train(self):
        train_x, train_y = self._get_train_data()
        print('TRAINING...')
        self._model.fit(train_x, train_y, batch_size=self._train_args['batch_size'],
                        epochs=self._train_args['train_epochs'])
        print('COMPLETE')
        if self._train_args['clear_memory']:
            self._memory.clear()
        if self._train_args['incremental_save']:
            self._save_game()

    def _get_train_data(self):
        train_x_list = list()
        train_y_list = list()
        sample_memory = choices(self._memory, k=self._train_args['train_sample_size'])

        for initial_state, action, next_state, reward in sample_memory:
            current_rewards = self._model.predict(initial_state).squeeze()
            q_val = self._get_q_value(initial_state, action, reward, next_state)
            current_rewards[action] = q_val
            current_rewards = np.expand_dims(current_rewards, axis=0)
            train_x_list.append(initial_state)
            train_y_list.append(current_rewards)
        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        return train_x, train_y

    def _get_q_value(self, initial_state, action, reward, next_state):
        initial_state_predictions = self._model.predict(initial_state, verbose=False).squeeze()
        next_state_predictions = self._model.predict(next_state, verbose=False).squeeze()
        initial_q_max = initial_state_predictions[action]
        next_q_max = next_state_predictions[np.argmax(next_state_predictions)]
        q_current = (1 - self._bot_args['alpha']) * initial_q_max
        q_future = self._bot_args['alpha'] * (reward + self._bot_args['gamma'] * next_q_max)
        return q_current + q_future

    def _calc_reward(self, reward_args):
        NotImplementedError('REWARD FUNCTION NOT IMPLEMENTED')

    def end_game(self):
        self._game_number += 1
        self._train()
        self._save_game()

    def _save_game(self):
        outpath = os.path.join(self._train_args['_save_dir'], f'game_{self._game_number}_iteration_{self._move_n}')
        self._model.save(outpath)
