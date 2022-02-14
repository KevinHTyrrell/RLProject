from Bots.base_bot import BaseBot


class BotMaze(BaseBot):
    def _set_bot_train_vals(self):
        self._train_args['batch_size'] = 32
        self._train_args['clear_memory'] = False
        self._train_args['train_epochs'] = 10
        self._train_args['train_sample_size'] = 32
        self._train_args['incremental_save'] = True
        self._train_args['_save_dir'] = 'Models/Maze/'

    def _set_bot_vals(self):
        self._bot_args['alpha'] = 0.005
        self._bot_args['epsilon'] = 1.0
        self._bot_args['epsilon_decay'] = 0.00001
        self._bot_args['epsilon_limit'] = 0.2
        self._bot_args['gamma'] = 0.1