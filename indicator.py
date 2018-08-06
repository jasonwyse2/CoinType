import pandas as pd
import pymysql
from datetime import datetime
import numpy as np
import time

from dateutil.relativedelta import relativedelta

def get_commission_signal(position_signal):
    buysell_signal = pd.Series([0] * position_signal.shape[0])
    position_diff = np.diff(position_signal)
    buysell_signal[2:] = position_diff[:-1]
    return buysell_signal

def currentTime_forward_delta(currentTime, min_deltaTime):

    time_format = '%Y%m%d%H%M'
    curr = datetime.strptime(currentTime, time_format)
    forward = (curr + relativedelta(minutes=+min_deltaTime))
    currTime = forward.strftime(time_format)
    return currTime

def get_return_for_unitTime(total_return, datetime_focused, start_time, end_time, minutes_in_uninTime =24 * 60):
    last_time_point = start_time
    # datetime_focused = datetime_focused.iloc[:, 0]
    cur_time_point = currentTime_forward_delta(start_time, minutes_in_uninTime)
    return_for_unitTime_list = []
    unitEndTime_list = []
    # datetime_focused.reshape(datetime_focused.size())
    while last_time_point < end_time:
        idx1 = (datetime_focused >= int(last_time_point))
        idx2 = (datetime_focused < int(cur_time_point))
        unitTime_totalreturn = np.sum(total_return[(idx1 & idx2)])
        unitEndTime_list.append(cur_time_point)
        return_for_unitTime_list.append(unitTime_totalreturn)
        last_time_point = cur_time_point
        cur_time_point = currentTime_forward_delta(cur_time_point, minutes_in_uninTime)
        # print('%s'%currTime)
    if unitEndTime_list[-1] > end_time:
        unitEndTime_list.pop()
        unitEndTime_list.append(end_time)
    return return_for_unitTime_list,unitEndTime_list

def get_average_annual_return(total_return, datetime_focused, start_time, end_time):
    return_for_unitTime_list,unitTime_list = get_return_for_unitTime(total_return, datetime_focused, start_time, end_time)

def get_max_drawdown(total_return, datetime_focused, start_time, end_time):
    return_for_unitTime_list,unitTime_list = get_return_for_unitTime(total_return, datetime_focused, start_time, end_time)

    return_for_unitTime_series = pd.Series(return_for_unitTime_list)
    return_for_unitTime_cum = np.cumsum(return_for_unitTime_series)+1

    cum_max = np.maximum.accumulate(return_for_unitTime_cum)
    dd_array = return_for_unitTime_cum/cum_max-1
    max_dd = np.min(dd_array)

    dd_end_idx = np.argwhere(dd_array == max_dd)[0][0]
    dd_start_idx = np.argwhere(return_for_unitTime_cum==cum_max[dd_end_idx])[0][0]

    dd_startTime = unitTime_list[dd_start_idx]
    dd_endTime = unitTime_list[dd_end_idx]
    return max_dd,dd_start_idx,dd_end_idx

def get_total_turnover(position_signal_list):
    position_signal = position_signal_list[0]
    buysell_signal = get_commission_signal(position_signal)
    turn_over  = np.sum(buysell_signal[buysell_signal>0])
    return turn_over
def sharp_ratio(total_return,datetime_focused,start_time,end_time):
    return_for_day_list,unitTime_list = get_return_for_unitTime(total_return, datetime_focused, start_time, end_time)
    return_mean_day = np.mean(return_for_day_list)
    return_std_day = np.std(return_for_day_list)
    if return_std_day==0:
        sharp = 0
    else:
        sharp = return_mean_day/return_std_day*np.sqrt(365)
    return sharp

def get_std_divide_price(price_focused, period):
    # price_focused = self.price_focused_list[3]
    two_contract_diff = price_focused.iloc[:, 0] - price_focused.iloc[:, 1]
    price1 = price_focused.iloc[:, 0]
    period_std = two_contract_diff.rolling(period).std()
    std_divide_price = np.mean(period_std / price1)
    return std_divide_price

