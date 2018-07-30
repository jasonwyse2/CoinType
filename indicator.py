import pandas as pd
import pymysql
from datetime import datetime
import numpy as np
import time
from CTA_main import get_commission_signal
from dateutil.relativedelta import relativedelta

def currentTime_forward_delta(currentTime, min_deltaTime):

    time_format = '%Y%m%d%H%M'
    curr = datetime.strptime(currentTime, time_format)
    forward = (curr + relativedelta(minutes=+min_deltaTime))
    currTime = forward.strftime(time_format)
    return currTime

def get_return_for_unitTime(total_return, datetime_focused, start_time, end_time, minutes_in_uninTime =24 * 60):
    last_time_point = start_time
    datetime_focused = datetime_focused.iloc[:, 0]
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
    drawdown_list = []
    return_for_unitTime_series = pd.Series(return_for_unitTime_list)
    return_for_unitTime_cum = np.cumsum(return_for_unitTime_series)+1

    for i in range(1,len(return_for_unitTime_cum)):
        drawdown_list.append(return_for_unitTime_cum[i]/np.max(return_for_unitTime_cum[:i])-1)
    max_drawdown = np.min(drawdown_list)

    cum_max = np.maximum.accumulate(return_for_unitTime_cum)
    dd_array = return_for_unitTime_cum/cum_max-1
    max_dd = np.min(dd_array)

    dd_end_idx = np.argwhere(dd_array == max_dd)[0][0]
    dd_start_idx = np.argwhere(return_for_unitTime_cum==cum_max[dd_end_idx])[0][0]


    return max_drawdown

def get_total_turnover(position_contract1,position_contract2,period):
    position_signal1 = get_commission_signal(position_contract1, period)
    turn_over  = np.sum(position_signal1[position_signal1>0])
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

