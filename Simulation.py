from SimulationBasic import SimulationBasic
import os
import pandas as pd
import numpy as np
class Simulation(SimulationBasic):

    def strategy(self):
        # write your strategy here
        pass
    def generate_position_signal(self,):
        [window_period, std_num] = [self.window_period, self.std_num]
        open_focused, high_focused, low_focused, close_focused = self.price_focused_list[0].values, self.price_focused_list[1].values, \
                                 self.price_focused_list[2].values, self.price_focused_list[3].values
        volume_focused = self.volume_focused.values
        datetime_focused = self.datetime_focused.iloc[:, 0].values
        instrument_focused = self.instrument_focused.values
        position_signal_focused = np.zeros(self.instrument_focused.shape)
        price_used = np.copy(open_focused)

        if self.coinType1==self.coinType2:
            two_contract_diff = price_used[:, 0] - price_used[:, 1]
        else:
            two_contract_diff = np.log(price_used[:, 0]) - np.log(price_used[:, 1])
            # two_contract_diff = price_used[:, 0] - price_used[:, 1]

        period_mean = (pd.Series(two_contract_diff).rolling(window_period).mean()).values
        period_std = (pd.Series(two_contract_diff).rolling(window_period).std()).values
        ceil_price = period_mean + period_std * std_num
        floor_price = period_mean - period_std * std_num

        cleartype_coefficient = 0.5
        clear_ceil = period_mean + period_std * std_num * cleartype_coefficient
        clear_floor = period_mean - period_std * std_num * cleartype_coefficient

        instrument_contract1, instrument_contract2 = instrument_focused[:, 0], instrument_focused[:,1]
        for i in range(window_period - 1, period_mean.shape[0]):

            delivery_time1 = self.is_delivery_time(instrument_contract1[i], datetime_focused[i])
            delivery_time2 = self.is_delivery_time(instrument_contract2[i], datetime_focused[i])
            if delivery_time1 == True or delivery_time2 == True:
                position_signal_focused[i][0] = 0
                position_signal_focused[i][1] = 0
            else:
                if two_contract_diff[i] >= ceil_price[i]:
                    position_signal_focused[i][0] = -1
                    position_signal_focused[i][1] = 1

                elif two_contract_diff[i] <= floor_price[i]:
                    position_signal_focused[i][0] = 1
                    position_signal_focused[i][1] = -1
                else:
                    position_signal_focused[i][0] = position_signal_focused[i-1][0]
                    position_signal_focused[i][1] = position_signal_focused[i-1][1]
                if two_contract_diff[i] >= clear_ceil[i] and two_contract_diff[i] < ceil_price[i]:
                    if position_signal_focused[i-1][0] == 1:
                        position_signal_focused[i][0] = 0
                        position_signal_focused[i][1] = 0
                if two_contract_diff[i] <= clear_floor[i] and two_contract_diff[i] > floor_price[i]:
                    if position_signal_focused[i-1][0] == -1:
                        position_signal_focused[i][0] = 0
                        position_signal_focused[i][1] = 0

        position_signal_focused[:,0] = position_signal_focused[:,0]*0.5
        position_signal_focused[:,1] = position_signal_focused[:,1]*0.5
        position_signal_focused[0,:] = 0
        position_signal_focused[1:,:] = position_signal_focused[:-1,:]
        position_signal_focused[-1, :] = 0
        return position_signal_focused

if __name__ == '__main__':
    cta = Simulation()
    cta.start_time = '201808100000'
    cta.end_time =   '201808140000'
    cta.coin_list = ['btc', 'bch','eth', 'etc','eos']#  'btc', 'bch','eth', 'etc', 'eos'
    cta.two_contract = ['quarter','week']
    cta.strategy_name = 'medium'
    cta.loss_threshold = 0.01
    cta.cool_time = 100
    cta.window_period_list = [4500,5000,5500] #
    cta.std_num_list = [3,3.25,3.5] #2.5, 3, 3.25, 3.5, 3.75, 4

    # initialize variable values
    cta.project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    cta.buy_commission_rate = 0.0003
    cta.sell_commission_rate = cta.buy_commission_rate

    # start the program
    cta.start()

    cta.two_contract = ['quarter','quarter']
    # cta.start2()