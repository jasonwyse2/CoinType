import pandas as pd
import pymysql
from datetime import datetime
import numpy as np
import time
from dateutil.relativedelta import relativedelta

import tool
class Indicator:
    minutes_in_uninTime = 24 * 60
    return_for_unitTime_flag = 0
    def __init__(self,cta):
        self.cta = cta
        self.position_signal_list = cta.position_signal_list
        # self.single_return_add = cta.single_return_list[0] + cta.single_return_list[1]
        self.datetime_focused = cta.datetime_focused
        self.start_time = cta.start_time
        self.end_time = cta.end_time
    def get_buysell_signal(self,):
        buysell_signal_list = []
        for i in range(len(self.position_signal_list)):
            position_signal = self.position_signal_list[i]
            buysell_signal = pd.Series([0.0] * position_signal.shape[0])
            position_diff = np.diff(position_signal)
            buysell_signal[2:] = position_diff[:-1]
            buysell_signal_list.append(buysell_signal)
        return buysell_signal_list
    def get_single_return(self):
        price_focused_df = self.cta.price_focused_list[3]
        period = self.cta.window_period
        single_return_list = []
        buysell_signal_list = self.get_buysell_signal()
        for i in range(len(self.position_signal_list)):
            price_focused = price_focused_df.iloc[:, i]
            per_return_contract = np.diff(price_focused) / price_focused[:-1]
            position_signal = self.position_signal_list[i]

            buysell_signal = buysell_signal_list[i]
            contract_commission = pd.Series([0] * price_focused.shape[0])
            contract_commission[buysell_signal > 0] = (buysell_signal * self.cta.buy_commission_rate)[
                                                          buysell_signal > 0] * (-1)
            contract_commission[buysell_signal < 0] = (buysell_signal * self.cta.sell_commission_rate)[buysell_signal < 0]

            contract_return = pd.Series([0] * price_focused.shape[0])
            contract_return[period + 1:] = np.array(position_signal[period - 1:-2]) * np.array(
                per_return_contract[period:])
            contract_single_return = contract_return + contract_commission
            single_return_list.append(contract_single_return)

        return single_return_list

    def get_compound_return(self):
        price_focused_list = self.cta.price_focused_list
        commission_rate_buy, commission_rate_sell = self.buy_commission_rate, self.sell_commission_rate
        open_focused, high_focused, low_focused, close_focused = price_focused_list[0], price_focused_list[1], \
                                                                 price_focused_list[2], \
                                                                price_focused_list[3]
        compound_return_list = []
        buysell_signal_list = self.get_buysell_signal()
        for i in range(2):
            total_return = pd.Series([0.0] * close_focused.shape[0])
            BuySell_signal = buysell_signal_list[i]
            price = close_focused.iloc[:, i]
            buysell_signal = BuySell_signal[BuySell_signal != 0]
            holding_position = np.cumsum(buysell_signal)
            buysell_price = price[BuySell_signal != 0]
            buysell_price_return = pd.Series([0] * buysell_price.shape[0])
            buysell_price_return[1:] = np.diff(buysell_price) / buysell_price[:-1]
            # iterate the price series, which only contains buy and sell points.
            buysell_rate = commission_rate_buy + commission_rate_sell
            index = buysell_signal.index
            for j in range(1, index.shape[0]):
                if (buysell_signal[index[j]] == -1 and holding_position[index[j - 1]] == 1):
                    total_return[index[j]] = buysell_price[index[j]] / buysell_price[index[j - 1]] - 1 - buysell_rate
                elif (buysell_signal[index[j]] == -2 and holding_position[index[j - 1]] == 1):
                    total_return[index[j]] = buysell_price[index[j]] / buysell_price[index[j - 1]] - 1 - buysell_rate
                elif (buysell_signal[index[j]] == 1 and holding_position[index[j - 1]] == -1):
                    total_return[index[j]] = 1 - buysell_price[index[j]] / buysell_price[index[j - 1]] - buysell_rate
                elif (buysell_signal[index[j]] == 2 and holding_position[index[j - 1]] == -1):
                    total_return[index[j]] = 1 - buysell_price[index[j]] / buysell_price[index[j - 1]] - buysell_rate
            compound_return_list.append(total_return)
        return compound_return_list
    def get_return_for_unitTime(self,):
        if self.return_for_unitTime_flag == 0:
            last_time_point = self.start_time
            datetime_focused = self.datetime_focused
            end_time = self.end_time
            cur_time_point = tool.currentTime_forward_delta(self.start_time, self.minutes_in_uninTime)
            return_for_unitTime_list = []
            unitEndTime_list = []
            single_return_list = self.get_single_return()
            single_return_add = single_return_list[0] + single_return_list[1]
            # datetime_focused.reshape(datetime_focused.size())
            while last_time_point < end_time:
                idx1 = (datetime_focused >= int(last_time_point))
                idx2 = (datetime_focused < int(cur_time_point))
                unitTime_totalreturn = np.sum(single_return_add[(idx1 & idx2)])
                unitEndTime_list.append(cur_time_point)
                return_for_unitTime_list.append(unitTime_totalreturn)
                last_time_point = cur_time_point
                cur_time_point = tool.currentTime_forward_delta(cur_time_point, self.minutes_in_uninTime)
                # print('%s'%currTime)
            if unitEndTime_list[-1] > end_time:
                unitEndTime_list.pop()
                unitEndTime_list.append(end_time)
            self.return_for_unitTime_list = pd.Series(return_for_unitTime_list)
            self.unitEndTime_list = pd.Series(unitEndTime_list)
            self.return_for_unitTime_flag = 1
        return self.return_for_unitTime_list,self.unitEndTime_list

    def get_average_annual_return(self,total_return, datetime_focused, start_time, end_time):
        return_for_unitTime_list,unitTime_list = self.get_return_for_unitTime(total_return, datetime_focused, start_time, end_time)

    def get_max_drawdown(self,):

        return_for_unitTime_list,unitTime_list = self.get_return_for_unitTime()
        return_for_unitTime_series = pd.Series(return_for_unitTime_list)
        return_for_unitTime_cum = np.cumsum(return_for_unitTime_series)+1

        cum_max = np.maximum.accumulate(return_for_unitTime_cum)
        dd_array = return_for_unitTime_cum/cum_max-1
        max_dd = np.min(dd_array)

        dd_end_idx = np.argwhere(dd_array == max_dd)[0][0]
        dd_start_idx_array = np.argwhere(return_for_unitTime_cum==cum_max[dd_end_idx])
        dd_start_idx = 0
        for i in range(dd_start_idx_array.shape[0]-1,-1,-1):
            if(dd_start_idx_array[i][0]<dd_end_idx):
                dd_start_idx = dd_start_idx_array[i][0]
                break

        self.dd_startTime = unitTime_list[dd_start_idx]
        self.dd_endTime = unitTime_list[dd_end_idx]
        self.max_dd = max_dd
        return self.max_dd,self.dd_startTime,self.dd_endTime

    def get_margin_bp(self):
        total_turnover = self.get_total_turnover()
        mg_bp = self.get_total_return()/total_turnover
        return mg_bp
    def get_return_divide_dd(self,):
        max_dd, a ,b  = self.get_max_drawdown()
        self.return2dd = self.get_total_return() / abs(max_dd)
        return self.return2dd
    def get_total_return(self):
        single_return_list = self.get_single_return()
        return np.sum(single_return_list[0]+single_return_list[1])
    def get_total_turnover(self,):
        buysell_signal_list = self.get_buysell_signal()
        buysell_signal = buysell_signal_list[0]
        self.turn_over  = np.sum(buysell_signal[buysell_signal>0])*2
        return self.turn_over
    def get_mean_turnover(self):
        total_turnover = self.get_total_turnover()
        return_for_unitTime_list, unitEndTime_list = self.get_return_for_unitTime()
        mean_turnover = float(total_turnover)/len(unitEndTime_list)
        return mean_turnover
    def get_sharp(self, ):
        return_for_day_list,unitTime_list = self.get_return_for_unitTime()
        return_mean_day = np.mean(return_for_day_list)
        return_std_day = np.std(return_for_day_list)
        if return_std_day==0:
            sharp = 0
        else:
            sharp = return_mean_day/return_std_day*np.sqrt(365)
        self.sharp = sharp
        return sharp

    def get_std_divide_price(self,):
        price_focused, window_period = self.cta.price_focused_list[3],self.cta.window_period
        two_contract_diff = price_focused.iloc[:, 0] - price_focused.iloc[:, 1]
        price1 = price_focused.iloc[:, 0]
        period_std = two_contract_diff.rolling(window_period).std()
        std_divide_price = np.mean(period_std / price1)
        self.std_divide_price = std_divide_price
        return std_divide_price

