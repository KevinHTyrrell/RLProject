from Bots.base_bot import BaseBot


class BotConnectFour(BaseBot):
    def _set_reward_args(self):
        self._reward_args['coeff_off']          = 1
        self._reward_args['def_off']            = 1
    def _calc_reward(self, reward_args):
        coeff_off   = self._reward_args.get('coeff_def', 1)
        coeff_def   = self._reward_args.get('coeff_def', 1)
        self_reward = reward_args.get('self_reward')
        opp_reward  = reward_args.get('opp_reward')
        return coeff_off * self_reward - coeff_def * opp_reward