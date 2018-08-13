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

        # open, high, low, close = self.price_df_list[0].values, self.price_df_list[1].values, \
        #                          self.price_df_list[2].values, self.price_df_list[3].values
        # volume = self.volume.values
        # datetime = self.datetime.iloc[:,0].values
        # instrument = self.instrument.values

        position_signal_focused = np.zeros(self.instrument_focused.shape)

        two_contract_diff = close_focused[:, 0] - close_focused[:, 1]
        period_mean = (pd.Series(two_contract_diff).rolling(window_period).mean()).values
        period_std = (pd.Series(two_contract_diff).rolling(window_period).std()).values
        ceil_price = period_mean + period_std * std_num
        floor_price = period_mean - period_std * std_num

        cleartype_coefficient = 0.5
        clear_ceil = period_mean + period_std * std_num * cleartype_coefficient
        clear_floor = period_mean - period_std * std_num * cleartype_coefficient

        instrument_contract1, instrument_contract2 = instrument_focused[:, 0], instrument_focused[:,1]
        for i in range(window_period - 1, period_mean.shape[0]):
            # volume_not_nan = not np.isnan(volume_focused[i][0]) and not np.isnan(volume_focused[i][1])
            # if volume_not_nan==False:
            #     pass
            # else:
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
                if two_contract_diff[i] > floor_price[i] and two_contract_diff[i] <= clear_floor[i]:
                    if position_signal_focused[i-1][0] == -1:
                        position_signal_focused[i][0] = 0
                        position_signal_focused[i][1] = 0

        position_signal_focused[:,0] = position_signal_focused[:,0]*0.5
        position_signal_focused[:,1] = position_signal_focused[:,1]*0.5
        position_signal_focused[0,:] = 0
        position_signal_focused[1:,:] = position_signal_focused[:-1,:]
        return position_signal_focused

if __name__ == '__main__':
    cta = Simulation()
    cta.end_time =   '201808130000'
    cta.coin_list = ['btc', 'bch','eth', 'etc', 'eos']#  'btc', 'bch','eth', 'etc', 'eos'
    cta.weight_list = [50,50]
    # cta.strategy_name = 'medium-'+'-'+str(cta.weight_list[1]) #[ceilfloor,medium,half-stdnum,threeQuarters-stdnum]
    cta.strategy_name = 'medium-'+str(cta.weight_list[0])+'-'+str(cta.weight_list[1])
    cta.window_period_list = [5000] #
    cta.std_num_list = [3] #2.5, 3, 3.25, 3.5, 3.75, 4

    # initialize variable values
    cta.project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    cta.buy_commission_rate = 0.0003
    cta.sell_commission_rate = cta.buy_commission_rate

    # start the program
    cta.start()