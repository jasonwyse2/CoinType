from SimulationBasic import SimulationBasic
import os
import pandas as pd
class Simulation(SimulationBasic):
    def __init__(self):
        pass
    def generate_position_signal(self,):
        [window_period, std_num, cleartype] = [self.window_period, self.std_num, self.cleartype]
        open, high, low, close = self.price_focused_list[0], self.price_focused_list[1], self.price_focused_list[2], \
                                 self.price_focused_list[3]
        volume = self.volume_focused

        two_contract_diff = close.iloc[:, 0] - close.iloc[:, 1]
        period_mean = two_contract_diff.rolling(window_period).mean()
        period_std = two_contract_diff.rolling(window_period).std()

        ceil_price = period_mean + period_std * std_num
        floor_price = period_mean - period_std * std_num

        cleartype_coefficient = self.cleartype_coefficient_dict[cleartype]
        clear_ceil = period_mean + period_std * std_num * cleartype_coefficient
        clear_floor = period_mean - period_std * std_num * cleartype_coefficient

        position_signal1 = pd.Series([0] * two_contract_diff.shape[0])
        position_signal2 = pd.Series([0] * two_contract_diff.shape[0])
        instrument_contract1, instrument_contract2 = \
            self.instrument_focused.iloc[:, 0], self.instrument_focused.iloc[:,1]
        for i in range(window_period - 1, period_mean.shape[0]):
            delivery_time1 = self.is_delivery_time(instrument_contract1.iloc[i], self.datetime_focused.iloc[i])
            delivery_time2 = self.is_delivery_time(instrument_contract2.iloc[i], self.datetime_focused.iloc[i])
            if delivery_time1 == True or delivery_time2 == True:
                position_signal1[i] = 0
                position_signal2[i] = 0
            else:
                if two_contract_diff[i] >= ceil_price[i]:
                    position_signal1[i] = -1
                    position_signal2[i] = 1

                elif two_contract_diff[i] <= floor_price[i]:
                    position_signal1[i] = 1
                    position_signal2[i] = -1
                else:
                    position_signal1[i] = position_signal1[i - 1]
                    position_signal2[i] = position_signal2[i - 1]

                if two_contract_diff[i] >= clear_ceil[i] and two_contract_diff[i] < ceil_price[i]:
                    if position_signal1[i - 1] == 1:
                        position_signal1[i] = 0
                        position_signal2[i] = 0
                if two_contract_diff[i] > floor_price[i] and two_contract_diff[i] <= clear_floor[i]:
                    if position_signal1[i - 1] == -1:
                        position_signal1[i] = 0
                        position_signal2[i] = 0
        position_signal_list = [position_signal1*0.5, position_signal2*0.5]
        return position_signal_list

if __name__ == '__main__':

    end_time =   '201808060000'#201806160030, 201807090000,201807170000
    coin_list = ['btc', 'bch']#  'btc', 'bch','eth', 'etc', 'eos',
    cleartype = 'medium' #[ceilfloor,medium,half-stdnum,threeQuarters-stdnum]
    window_period_list = [4000,5000] #
    std_num_list = [2.5,3] #2.5, 3, 3.25, 3.5, 3.75, 4
    cta = Simulation()
    # initialize variable values
    cta.project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    cta.buy_commission_rate = 0.0003
    cta.sell_commission_rate = cta.buy_commission_rate
    cta.end_time = end_time
    cta.coin_list = coin_list
    cta.cleartype = cleartype
    cta.window_period_list = window_period_list
    cta.std_num_list = std_num_list
    # start the program
    cta.start()