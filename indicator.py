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
    __sigleton_variable_flag= 0
    __sigleton_buysell_signal_flag=0
    __sigleton_single_return_flag = 0
    def __init__(self,cta,position_signal_array):
        self.cta = cta
        self.position_signal_array = position_signal_array
        self.datetime = cta.datetime
        self.start_time = cta.start_time
        self.end_time = cta.end_time
        self.single_return_array = np.zeros(self.position_signal_array.shape)
        self.compound_return_array = np.zeros(self.position_signal_array.shape)
    def normalize_position_signal_array(self):
        position_signal_array = self.position_signal_array.copy()
        divisor = len(self.cta.coin_list)*len(self.cta.window_period_list)*len(self.cta.std_num_list)
        row_sum_array = np.sum(np.absolute(position_signal_array),axis = 1)
        (rows,cols) = position_signal_array.shape
        divisor_array = np.array([divisor]*rows*cols).reshape(position_signal_array.shape)
        position_signal_array_norm = position_signal_array/divisor_array
        return position_signal_array_norm
    def initialize_variables(self):
        if self.__sigleton_variable_flag==0:
            self.buysell_signal_array = np.zeros(self.position_signal_array.shape)
            self.single_return_array = np.zeros(self.position_signal_array.shape)

    def get_buysell_signal(self,):
        position_signal_array = self.position_signal_array
        position_diff_array = np.diff(position_signal_array, axis=0)
        buysell_signal_array = np.zeros(self.position_signal_array.shape)
        buysell_signal_array[1:, :] = position_diff_array
        self.buysell_signal_array = buysell_signal_array
        return buysell_signal_array

    def get_single_return(self):

        if self.__sigleton_single_return_flag == 0:
            position_signal_array = self.position_signal_array
            window_period = self.cta.window_period
            datetime = self.datetime
            datetime_period = self.cta.datetime_focused.iloc[window_period,0]
            period = datetime[datetime.values==datetime_period].index.tolist()[0]
            open, high, low, close = self.cta.price_df_list[0].values, self.cta.price_df_list[1].values, \
                                          self.cta.price_df_list[2].values, self.cta.price_df_list[3].values
            price_diff = np.diff(close,axis=0)
            per_return_array = np.zeros(close.shape)
            per_return_array[1:,:] = price_diff / close[:-1,:]
            commission_array = np.zeros(close.shape)
            buysell_signal_array = self.get_buysell_signal()
            commission_array[period:, :] = np.abs(buysell_signal_array[period:, :]) * self.cta.buy_commission_rate
            self.single_return_array[period:-1,:] = position_signal_array[period:-1,:]*per_return_array[period+1:]-commission_array[period:-1, :]

            self.__sigleton_single_return_flag = 1
        return self.single_return_array

    def get_compound_return(self):
        price_focused_list = self.cta.price_df_list
        commission_rate_buy, commission_rate_sell = self.cta.buy_commission_rate, self.cta.sell_commission_rate
        open_focused, high_focused, low_focused, close_focused = \
            price_focused_list[0], price_focused_list[1], price_focused_list[2], price_focused_list[3]
        close = self.cta.price_df_list[3].values
        compound_return_array = self.compound_return_array
        buysell_signal_array = self.get_buysell_signal()
        buysell_signal_df = pd.DataFrame(buysell_signal_array,index =self.cta.instrument.index)
        buysell_signal = buysell_signal_df.iloc[np.sum(np.abs(buysell_signal_df.values),axis=1)>0,:]
        holding_position = buysell_signal.cumsum().values
        buysell_price = close[np.sum(np.abs(buysell_signal_df.values),axis=1)>0,:]
        price_diff = np.diff(close, axis=0)
        per_return_array = np.zeros(close.shape)
        per_return_array[1:,:] = price_diff / close[:-1, :]
        buysell_rate = commission_rate_buy + commission_rate_sell
        total_return = np.zeros(close.shape)

        index = buysell_signal.index
        buysell_signal_arr = buysell_signal.values
        for i in range(buysell_signal.shape[1]):
            unit = abs(buysell_signal.values[:,i][0])
            for j in range(1, index.shape[0]):
                if (buysell_signal_arr[j,i] == -unit and holding_position[j - 1,i] == unit):
                    total_return[index[j],i] = buysell_price[j,i] / buysell_price[j - 1,i] - 1 - buysell_rate
                elif (buysell_signal_arr[j,i] == -2*unit and holding_position[j - 1,i] == unit):
                    total_return[index[j],i] = buysell_price[j,i] / buysell_price[j - 1,i] - 1 - buysell_rate
                elif (buysell_signal_arr[j,i] == unit and holding_position[j - 1,i] == -unit):
                    total_return[index[j],i] = 1 - buysell_price[j,i] / buysell_price[j - 1,i] - buysell_rate
                elif (buysell_signal_arr[j,i] == 2*unit and holding_position[j - 1,i] == -unit):
                    total_return[index[j],i] = 1 - buysell_price[j,i] / buysell_price[j - 1,i] - buysell_rate

        return total_return


        # for i in range(2):
        #     total_return = pd.Series([0.0] * close_focused.shape[0])
        #     BuySell_signal = buysell_signal_list[i]
        #     price = close_focused.iloc[:, i]
        #     buysell_signal = BuySell_signal[BuySell_signal != 0]
        #     holding_position = np.cumsum(buysell_signal)
        #     buysell_price = price[BuySell_signal != 0]
        #     buysell_price_return = pd.Series([0] * buysell_price.shape[0])
        #     buysell_price_return[1:] = np.diff(buysell_price) / buysell_price[:-1]
        #     # iterate the price series, which only contains buy and sell points.
        #     buysell_rate = commission_rate_buy + commission_rate_sell
        #     index = buysell_signal.index
        #     for j in range(1, index.shape[0]):
        #         if (buysell_signal[index[j]] == -1 and holding_position[index[j - 1]] == 1):
        #             total_return[index[j]] = buysell_price[index[j]] / buysell_price[index[j - 1]] - 1 - buysell_rate
        #         elif (buysell_signal[index[j]] == -2 and holding_position[index[j - 1]] == 1):
        #             total_return[index[j]] = buysell_price[index[j]] / buysell_price[index[j - 1]] - 1 - buysell_rate
        #         elif (buysell_signal[index[j]] == 1 and holding_position[index[j - 1]] == -1):
        #             total_return[index[j]] = 1 - buysell_price[index[j]] / buysell_price[index[j - 1]] - buysell_rate
        #         elif (buysell_signal[index[j]] == 2 and holding_position[index[j - 1]] == -1):
        #             total_return[index[j]] = 1 - buysell_price[index[j]] / buysell_price[index[j - 1]] - buysell_rate
        #     compound_return_list.append(total_return)
        # return compound_return_list


    def get_return_for_unitTime(self,):
        if self.return_for_unitTime_flag == 0:
            last_time_point = self.start_time
            datetime = self.datetime.iloc[:,0].values
            end_time = self.end_time
            cur_time_point = tool.currentTime_forward_delta(self.start_time, self.minutes_in_uninTime)
            return_for_unitTime_list = []
            unitEndTime_list = []
            single_return_array = self.get_single_return()
            single_return_sum = np.sum(single_return_array,axis=1)
            while last_time_point < end_time:
                idx1 = (datetime >= int(last_time_point))
                idx2 = (datetime < int(cur_time_point))
                unitTime_totalreturn = np.sum(single_return_sum[(idx1 & idx2)])
                unitEndTime_list.append(cur_time_point)
                return_for_unitTime_list.append(unitTime_totalreturn)
                last_time_point = cur_time_point
                cur_time_point = tool.currentTime_forward_delta(cur_time_point, self.minutes_in_uninTime)
            if unitEndTime_list[-1] > end_time:
                unitEndTime_list.pop()
                unitEndTime_list.append(end_time)

            self.return_for_unitTime_list = (return_for_unitTime_list)
            self.unitEndTime_list = (unitEndTime_list)
            self.return_for_unitTime_flag = 1
        return self.return_for_unitTime_list,self.unitEndTime_list

    def get_average_annual_return(self,total_return, datetime_focused, start_time, end_time):
        return_for_unitTime_list,unitTime_list = self.get_return_for_unitTime(total_return, datetime_focused, start_time, end_time)

    def get_max_drawdown(self,):

        return_for_unitTime_list,unitTime_list = self.get_return_for_unitTime()
        return_for_unitTime_cum = np.cumsum(return_for_unitTime_list)+1

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

        dd_startTime = unitTime_list[dd_start_idx]
        dd_endTime = unitTime_list[dd_end_idx]
        dd_startTime = dd_startTime[:-4]
        dd_endTime = dd_endTime[:-4]
        max_dd = round(max_dd*100,2)
        return max_dd,dd_startTime,dd_endTime

    def get_margin_bp(self):
        total_turnover = self.get_total_turnover()
        mg_bp = self.get_total_return()/total_turnover
        mg_bp = round(mg_bp*100,2)
        return mg_bp
    def get_return_divide_dd(self,):
        max_dd, a ,b  = self.get_max_drawdown()
        self.return2dd = self.get_total_return() / abs(max_dd)
        self.return2dd = round(self.return2dd,2)
        return self.return2dd
    def get_total_return(self):
        single_return_array = self.get_single_return()
        total_return = np.sum(single_return_array)
        total_return = round(total_return * 100, 4)
        return total_return
    def get_total_turnover(self,):
        buysell_signal_array = self.get_buysell_signal()
        turn_over = np.sum(np.abs(buysell_signal_array))
        return turn_over
    def get_mean_turnover(self):
        total_turnover = self.get_total_turnover()
        return_for_unitTime_list, unitEndTime_list = self.get_return_for_unitTime()
        mean_turnover = float(total_turnover)/len(unitEndTime_list)
        mean_turnover = round(mean_turnover * 100, 2)
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
        sharp = round(sharp, 2)
        return sharp

    def get_std_divide_price(self,):
        price_focused, window_period = self.cta.price_focused_list[3],self.cta.window_period
        two_contract_diff = price_focused.iloc[:, 0] - price_focused.iloc[:, 1]
        price1 = price_focused.iloc[:, 0]
        period_std = two_contract_diff.rolling(window_period).std()
        std_divide_price = np.mean(period_std / price1)
        self.std_divide_price = std_divide_price
        return std_divide_price

