import os
from datetime import datetime as dt

import time
import pandas as pd
import numpy as np
from tool import *
import pymysql
import indicator
import tool
import re
class DATA:
    db_host_list = ['192.168.0.113', '206.189.89.22', '192.168.0.113', '127.0.0.1']  # 192.168.0.113, 206.189.89.22
    db_port_list = [3306, 5555, 3306, 3306]  # 3306 , 5555
    db_user_list = ['root', 'linjuninvestment', 'root', 'root']  # root, linjuninvestment
    db_pass_list = ['1qazxsw2', '123456', '1qazxsw2', '1qazxsw2']  # 1qazxsw2, 123456
    db_name_list = ['okex', 'tradingdata', 'yg', 'okex']  # okcoin, tradingdata

    contract_timeType = ['week', 'nextweek', 'quarter']
    __fourPrice_type_list = ['open', 'high', 'low', 'close']
    coinType_list = ['bch', 'btc', 'btg', 'eos', 'etc', 'eth', 'ltc', 'xrp']  # do not change this variable
    one_week_seconds = 7 * 24 * 3600
    db_table = '1min'
    def __init__(self):
        pass
    def login_MySQL(self,num):
        db_host = self.db_host_list[num]
        db_port = self.db_port_list[num]
        db_user = self.db_user_list[num]
        db_pass = self.db_pass_list[num]
        db_name = self.db_name_list[num]
        conn = pymysql.connect(host=db_host, port=db_port, user=db_user, passwd=db_pass, db=db_name)
        return conn

    def get_unique_datetime(self,start_time, end_time):
        df = self.get_datetime_allUnique_df(start_time, end_time)
        unique_datetime = pd.unique(df.datetime)
        df = pd.DataFrame({'datetime': unique_datetime})
        return df

    def getTimeDiff(self,timeStra, timeStrb):
        if timeStra <= timeStrb:
            return 0
        ta = time.strptime(timeStra, "%Y%m%d%H%M")
        tb = time.strptime(timeStrb, "%Y%m%d%H%M")
        y, m, d, H, M, S = ta[0:6]
        dataTimea = datetime(y, m, d, H, M, S)
        y, m, d, H, M, S = tb[0:6]
        dataTimeb = datetime(y, m, d, H, M, S)
        secondsDiff = (dataTimea - dataTimeb).total_seconds()

        return secondsDiff

    def get_datetime_allUnique_df(self,start_time, end_time):
        db_table = self.db_table
        conn = self.login_MySQL(3)
        field_list = ['open','high','close','low' ,'datetime', 'instrument','volume']
        fields = ','.join(field_list)
        sql = 'SELECT '+ fields +' FROM okex.' + db_table + \
              ' WHERE DATETIME >=%d and datetime<%d and volume>0 order by datetime asc'%(int(start_time), int(end_time))
        df = pd.read_sql(sql, conn)
        return df

    def get_columns_cointype_df(self,coinType, start_time, end_time):
        conn = self.login_MySQL(3)
        field_list = ['open', 'high', 'close', 'low', 'datetime', 'instrument', 'volume']
        fields = ','.join(field_list)
        sql_datetime = 'SELECT '+fields+' FROM okex.' + self.db_table +\
                       ' WHERE DATETIME >=%d and datetime<%d and volume>0'%(int(start_time), int(end_time))
        sql_filter = 'instrument LIKE "%s' % (coinType) + '%"'
        # concatenate sql string
        sql = sql_datetime + ' and ' + sql_filter + ' order by instrument'
        df = pd.read_sql(sql, conn)
        return df

    def add_contract_price(self,df_data, contract_timeType, coinType, datetime_open, datetime_high, datetime_close,
                           datetime_low):
        datetime_open = pd.merge(datetime_open, df_data[['open', 'datetime']], on='datetime', how='left')
        datetime_open.rename(columns={'open': coinType + '_' + contract_timeType}, inplace=True)

        datetime_high = pd.merge(datetime_high, df_data[['high', 'datetime']], on='datetime', how='left')
        datetime_high.rename(columns={'high': coinType + '_' + contract_timeType}, inplace=True)

        datetime_close = pd.merge(datetime_close, df_data[['close', 'datetime']], on='datetime', how='left')
        datetime_close.rename(columns={'close': coinType + '_' + contract_timeType}, inplace=True)

        datetime_low = pd.merge(datetime_low, df_data[['low', 'datetime']], on='datetime', how='left')
        datetime_low.rename(columns={'low': coinType + '_' + contract_timeType}, inplace=True)
        return [datetime_open, datetime_high, datetime_close, datetime_low]

    def add_contract_instrument(self,df_data, contract_timeType, coinType, instrument):
        instrument = pd.merge(instrument, df_data[['instrument', 'datetime']], on='datetime', how='left')
        instrument.rename(columns={'instrument': coinType + '_' + contract_timeType}, inplace=True)
        return instrument

    def add_contract_volume(self,df_data, contract_timeType, coinType, volume):
        volume = pd.merge(volume, df_data[['volume', 'datetime']], on='datetime', how='left')
        volume.rename(columns={'volume': coinType + '_' + contract_timeType}, inplace=True)
        return volume

    def quick_datetime_symbol(self,start_time, end_time):

        datetime_open = self.get_unique_datetime(start_time, end_time)
        datetime_high, datetime_close, datetime_low = datetime_open, datetime_open, datetime_open
        instrument = datetime_open
        volume = datetime_open
        one_week_seconds = self.one_week_seconds
        for coin_Type in self.coinType_list:
            df = self.get_columns_cointype_df(coin_Type, start_time, end_time)
            df['instrument_precise'] = df.instrument.map(lambda x: x[len(coin_Type) + 1:] + '1600')
            # transfer 'datetime' and instrument into date format
            df['datetime_str'] = df.datetime.map(lambda x: str(x))
            df['seconds_diff'] = df.apply(lambda x: (dt.strptime(x['instrument_precise'], "%Y%m%d%H%M") - dt.strptime(x['datetime_str'],"%Y%m%d%H%M")).total_seconds(), axis=1)

            df_week = df[(df['seconds_diff'] >= 0) & (df['seconds_diff'] < one_week_seconds)]
            df_nextweek = df[(df['seconds_diff'] > 0) & (df['seconds_diff'] >= one_week_seconds) & (
                        df['seconds_diff'] <= 2 * one_week_seconds)]
            df_quarter = df[(df['seconds_diff'] >= 0) & (df['seconds_diff'] > 2 * one_week_seconds)]

            df_list = [df_week, df_nextweek, df_quarter]
            contract_timetype_list = ['week', 'nextweek', 'quarter']
            for i in range(len(df_list)):
                [datetime_open, datetime_high, datetime_close, datetime_low] \
                    = self.add_contract_price(df_list[i], contract_timetype_list[i], coin_Type, datetime_open, datetime_high,
                                              datetime_close, datetime_low)
                instrument = self.add_contract_instrument(df_list[i], contract_timetype_list[i], coin_Type, instrument)
                volume = self.add_contract_volume(df_list[i], contract_timetype_list[i], coin_Type, volume)

        price_df_list = [datetime_open, datetime_high, datetime_close, datetime_low]
        return [price_df_list, instrument, volume]

    def check_update_data(self,end_time,data_dir):
        price_df_list, instrument, datetime, volume = self.load_data_from_file(data_dir)
        datetime = datetime.iloc[:,0] #convert DataFrame into Series
        recorded_end_time = datetime[-1]
        if recorded_end_time<end_time:
            start_time = tool.currentTime_forward_delta(recorded_end_time, deltaTime=1)
            [price_df_list, instrument, volume] = self.quick_datetime_symbol(start_time, end_time)
            pass
    def get_data(self, start_time, end_time, data_dir):
        time_start = time.clock()
        flag = mkdir(data_dir)
        p_list = ['open', 'high', 'close', 'low']  # the order can't be changed
        if flag == 0 or not os.listdir(data_dir):
            [price_df_list, instrument, volume] = self.quick_datetime_symbol(start_time, end_time)
            datetime = instrument.iloc[:, 0]

            write_df_to_file(datetime, data_dir, filename='datetime')
            write_df_to_file(instrument, data_dir, filename='instrument')
            write_df_to_file(volume, data_dir, filename='volume')
            for i in range(len(p_list)):
                write_df_to_file(price_df_list[i], data_dir, filename=p_list[i])
        time_end = time.clock()
        elapsed = time_end - time_start

    def load_data_from_file(self, dest_dir):
        price_df_list = []
        for price_type in self.__fourPrice_type_list:
            fileName_price = price_type + '.pkl'
            fullfileName_price = os.path.join(dest_dir, fileName_price)
            price = pd.DataFrame(getpkl(fullfileName_price)[:, 1:])
            price_df_list.append(price)

        fullfileName_instrument = os.path.join(dest_dir, 'instrument.pkl')
        instrument = pd.DataFrame(getpkl(fullfileName_instrument)[:, 1:])
        fullfileName_datetime = os.path.join(dest_dir, 'datetime.pkl')
        datetime = pd.DataFrame(getpkl(fullfileName_datetime))
        fullfileName_volume = os.path.join(dest_dir, 'volume.pkl')
        volume = pd.DataFrame(getpkl(fullfileName_volume)[:, 1:])
        return price_df_list, instrument, datetime, volume

class CTA:
    __contract_map = {'week': 0, 'nextweek': 1, 'quarter': 2}
    __cleartype_coefficient_dict = {'ceilfloor': 1, 'half-stdnum': 0.5, 'medium': 0, 'threeQuarters-stdnum': 0.75}
    __coinType_list = ['bch', 'btc', 'btg', 'eos', 'etc', 'eth', 'ltc', 'xrp']  # do not change this variable
    __fourPrice_type_list = ['open', 'high', 'low', 'close']
    __project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    __root_data = os.path.join(__project_dir, 'data')
    __root_result = os.path.join(__project_dir, 'backtest')
    __root_average = os.path.join(__project_dir, 'average')
    __root_buysellInfo = os.path.join(__project_dir, 'buysell')
    __singleton = 0
    start_time = '201806160000'
    db_table = '1min'  # ['1min','5min']
    three_contract = ['week', 'nextweek', 'quarter']
    bp_list=[]
    comp_bp_list=[]
    return_dd_list = []
    std_price_list = []
    turnover_list = []
    average_type_list = []
    average_result_list = [bp_list,comp_bp_list,return_dd_list,std_price_list,turnover_list]
    buy_commission_rate = 0.0000
    sell_commission_rate = 0.0000
    __data = DATA()
    data_dir = ''
    def __init__(self,):
        pass

    @property
    def project_dir(self):
        return self.__project_dir
    @project_dir.setter
    def project_dir(self,value):
        self.__project_dir = value

    def set_directory(self,start_time,end_time,cleartype):
        # Singleton mode, this function can only be called once
        if self.__singleton == 0:
            db_table = self.__data.db_table
            father_dir = self.__project_dir
            root_data = os.path.join(father_dir, 'data')
            root_result = os.path.join(father_dir, 'backtest')
            root_average = os.path.join(father_dir, 'average')
            root_buysellInfo = os.path.join(father_dir, 'buysell')

            fileName_item_list = [db_table, start_time, end_time]  # [db_table, start_time, end_time]
            db_start_end = '-'.join(fileName_item_list)
            self.data_dir = os.path.join(root_data, db_start_end)

            now_str = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
            self.result_dir = os.path.join(root_result, db_start_end, cleartype, now_str)
            self.average_dir = os.path.join(root_average, db_start_end, cleartype, now_str)
            self.buysell_dir = os.path.join(root_buysellInfo, db_start_end, cleartype, now_str)

            mkdir(self.data_dir)
            mkdir(self.result_dir)
            mkdir(self.average_dir)
            mkdir(self.buysell_dir)

            self.__singleton = 1
    def __is_delivery_time(self,instrument,date_time):
        time_instrument = dt.strptime(instrument[4:] + '1600',"%Y%m%d%H%M")
        time_datetime = dt.strptime(str(date_time),"%Y%m%d%H%M")
        time_diff = (time_instrument - time_datetime).total_seconds()
        delivery_time = False
        if abs(time_diff)<2:
            delivery_time = True
        return delivery_time

    def open_position(self,args_list):
        [period, std_num, cleartype] = args_list

        price_focused_list = self.price_focused_list
        instrument_focused = self.instrument_focused
        volume_focused = self.volume_focused
        open, high, low, close = price_focused_list[0], price_focused_list[1], price_focused_list[2], \
                                 price_focused_list[3]
        two_contract_diff = close.iloc[:, 0] - close.iloc[:, 1]

        period_mean = two_contract_diff.rolling(period).mean()
        period_std = two_contract_diff.rolling(period).std()

        ceil_price = period_mean + period_std * std_num
        floor_price = period_mean - period_std * std_num
        instrument_contract1, instrument_contract2 = instrument_focused.iloc[:, 0], instrument_focused.iloc[:, 1]
        position_signal1, position_signal2 = 0, 0
        open_position_satisfied = False
        if two_contract_diff[-1] >= ceil_price[-1]:
            open_position_satisfied = True
            position_signal1 = -1
            position_signal2 = 1

        elif two_contract_diff[-1] <= floor_price[-1]:
            open_position_satisfied = True
            position_signal1 = 1
            position_signal2 = -1
        return open_position_satisfied,position_signal1, position_signal2

    def clear_position(self,args_list):
        pass
    # def generate_position_signal(self, price_focused_list, volume_focused, instrument_focused, args_list):
    #     [period, std_num, cleartype] = args_list
    #
    #     open, high, low, close = price_focused_list[0],price_focused_list[1],price_focused_list[2],price_focused_list[3]
    #     two_contract_diff = close.iloc[:, 0] - close.iloc[:, 1]
    #
    #     period_mean = two_contract_diff.rolling(period).mean()
    #     period_std = two_contract_diff.rolling(period).std()
    #
    #     ceil_price = period_mean + period_std * std_num
    #     floor_price = period_mean - period_std * std_num
    #     cleartype_coefficient = self.__cleartype_coefficient_dict[cleartype]
    #     clear_ceil = period_mean + period_std * std_num * cleartype_coefficient
    #     clear_floor = period_mean - period_std * std_num * cleartype_coefficient
    #     position_signal1 = pd.Series([0] * two_contract_diff.shape[0])
    #     position_signal2 = pd.Series([0] * two_contract_diff.shape[0])
    #
    #     instrument_contract1, instrument_contract2 = instrument_focused.iloc[:, 0], instrument_focused.iloc[:, 1]
    #
    #     for i in range(period, period_mean.shape[0]):
    #         idx = i
    #         if instrument_contract1[i] != instrument_contract1[i - 1]:
    #             position_signal1[i - 1] = 0
    #             position_signal2[i - 1] = 0
    #         else:
    #             if two_contract_diff[i] >= ceil_price[idx]:
    #                 position_signal1[i] = -1
    #                 position_signal2[i] = 1
    #
    #             elif two_contract_diff[i] <= floor_price[idx]:
    #                 position_signal1[i] = 1
    #                 position_signal2[i] = -1
    #             else:
    #                 position_signal1[i] = position_signal1[i - 1]
    #                 position_signal2[i] = position_signal2[i - 1]
    #
    #             if two_contract_diff[i] >= clear_ceil[idx] and two_contract_diff[i] < ceil_price[idx]:
    #                 if position_signal1[i - 1] == 1:
    #                     position_signal1[i] = 0
    #                     position_signal2[i] = 0
    #             if two_contract_diff[i] > floor_price[idx] and two_contract_diff[i] <= clear_floor[idx]:
    #                 if position_signal1[i - 1] == -1:
    #                     position_signal1[i] = 0
    #                     position_signal2[i] = 0
    #     return [position_signal1, position_signal2]
    def generate_position_signal(self, args_list):

        [period, std_num, cleartype] = args_list
        open, high, low, close = self.price_focused_list[0], self.price_focused_list[1], self.price_focused_list[2], self.price_focused_list[3]
        two_contract_diff = close.iloc[:, 0] - close.iloc[:, 1]

        period_mean = two_contract_diff.rolling(period).mean()
        period_std = two_contract_diff.rolling(period).std()

        ceil_price = period_mean + period_std * std_num
        floor_price = period_mean - period_std * std_num

        cleartype_coefficient = self.__cleartype_coefficient_dict[cleartype]
        clear_ceil = period_mean + period_std * std_num * cleartype_coefficient
        clear_floor = period_mean - period_std * std_num * cleartype_coefficient

        position_signal1 = pd.Series([0] * two_contract_diff.shape[0])
        position_signal2 = pd.Series([0] * two_contract_diff.shape[0])
        instrument_contract1, instrument_contract2 = self.instrument_focused.iloc[:, 0], self.instrument_focused.iloc[:, 1]
        for i in range(period-1, period_mean.shape[0]):
            delivery_time1 = self.__is_delivery_time(instrument_contract1.iloc[i], self.datetime_focused.iloc[i])
            delivery_time2 = self.__is_delivery_time(instrument_contract2.iloc[i], self.datetime_focused.iloc[i])
            if delivery_time1==True or delivery_time2==True:
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
        return [position_signal1, position_signal2]

    def get_compound_return(self, buysell_signal_list):
        price_focused_list = self.price_focused_list
        commission_rate_buy,commission_rate_sell = self.buy_commission_rate, self.sell_commission_rate
        open_focused, high_focused, low_focused, close_focused = price_focused_list[0], price_focused_list[1], price_focused_list[2], \
                                 price_focused_list[3]
        compound_return_list = []
        for i in range(2):
            total_return = pd.Series([0.0] * close_focused.shape[0])
            BuySell_signal = buysell_signal_list[i]
            price = close_focused.iloc[:,i]
            buysell_signal = BuySell_signal[BuySell_signal!=0]
            holding_position = np.cumsum(buysell_signal)
            buysell_price = price[BuySell_signal!=0]
            buysell_price_return = pd.Series([0] * buysell_price.shape[0])
            buysell_price_return[1:] = np.diff(buysell_price)/buysell_price[:-1]
            # iterate the price series, which only contains buy and sell points.
            buysell_rate = commission_rate_buy + commission_rate_sell
            index = buysell_signal.index
            for j in range(1,index.shape[0]):
                if (buysell_signal[index[j]] == -1 and holding_position[index[j-1]] == 1):
                    total_return[index[j]] = buysell_price[index[j]] / buysell_price[index[j-1]] - 1 - buysell_rate
                elif (buysell_signal[index[j]] == -2 and holding_position[index[j-1]] == 1):
                    total_return[index[j]] = buysell_price[index[j]] / buysell_price[index[j-1]] - 1 - buysell_rate
                elif (buysell_signal[index[j]] == 1 and holding_position[index[j-1]] == -1):
                    total_return[index[j]] = 1 - buysell_price[index[j]] / buysell_price[index[j-1]] - buysell_rate
                elif (buysell_signal[index[j]] == 2 and holding_position[index[j-1]] == -1):
                    total_return[index[j]] = 1 - buysell_price[index[j]] / buysell_price[index[j-1]] - buysell_rate
            compound_return_list.append(total_return)
        return compound_return_list

    def get_buysell_signal(self,position_signal):
        buysell_signal = indicator.get_commission_signal(position_signal)
        return buysell_signal

    def get_buysell_info(self,price_focused_df,position_signal_list, buysell_signal_list, single_return_list, compound_return_list):
        # price_focused_df = self.price_focused_list[3]
        buysell_info = pd.concat(
            [self.datetime_focused,
             price_focused_df.iloc[:, 0], position_signal_list[0], buysell_signal_list[0], single_return_list[0],
             compound_return_list[0],
             price_focused_df.iloc[:, 1], position_signal_list[1], buysell_signal_list[1], single_return_list[1],
             compound_return_list[1]], axis=1)
        buysell_info.columns = ['datetime',
                                'price1', 'position1', 'buysell1', 'single_return1', 'compound_return1',
                                'price2', 'position2', 'buysell2', 'single_return2', 'compound_return2']
        return buysell_info

    def get_single_return_with_position(self, position_signal_list, period):
        # price_focused = price_focused_list[3]
        price_focused_df = self.price_focused_list[3]
        single_return_list = []
        buysell_signal_list = []
        for i in range(len(position_signal_list)):
            price_focused = price_focused_df.iloc[:,i]
            position_signal = position_signal_list[i]
            per_return_contract = np.diff(price_focused) / price_focused[:-1]
            buysell_signal = indicator.get_commission_signal(position_signal)

            contract_commission = pd.Series([0] * price_focused.shape[0])
            contract_commission[buysell_signal > 0] = (buysell_signal * self.buy_commission_rate)[buysell_signal > 0] * (-1)
            contract_commission[buysell_signal < 0] = (buysell_signal * self.sell_commission_rate)[buysell_signal < 0]

            contract_return = pd.Series([0] * price_focused.shape[0])
            contract_return[period + 1:] = np.array(position_signal[period - 1:-2]) * np.array(
                per_return_contract[period:])

            contract_single_return = contract_return + contract_commission
            single_return_list.append(contract_single_return)
            buysell_signal_list.append(buysell_signal)
        return single_return_list, buysell_signal_list

    def get_focused_info(self,price_df_list, instrument, datetime, volume, coinType, two_contract):
        column_offset = self.__coinType_list.index(coinType) * 3
        contract1_idx = self.__contract_map[two_contract[0]]
        contract2_idx = self.__contract_map[two_contract[1]]

        # fourPrice_type_list = ['open', 'high', 'low', 'close']
        price_focused_list = []
        for i in range(len(self.__fourPrice_type_list)):
            price = price_df_list[i]
            price_focused_nan = price[[column_offset + contract1_idx, column_offset + contract2_idx]]
            price_focused = price_focused_nan.dropna(axis=0, how='any')
            price_focused_list.append(price_focused)

        instrument_focused_nan = instrument[[column_offset + contract1_idx, column_offset + contract2_idx]]
        volume_focused_nan = volume[[column_offset + contract1_idx, column_offset + contract2_idx]]
        # remove rows that contains 'nan'
        price_focused = price_focused_list[0]
        instrument_focused = instrument_focused_nan.iloc[price_focused.index.tolist(), :]
        datetime_focused = datetime.iloc[price_focused.index.tolist(), :]
        volume_focused = volume_focused_nan.iloc[price_focused.index.tolist(), :]
        # reset index from 0, because some index has been removed
        for price_focused in price_focused_list:
            price_focused.index = range(len(price_focused.index))
        instrument_focused.index = range(len(instrument_focused.index))
        datetime_focused.index = range(len(datetime_focused.index))
        volume_focused.index = range(len(volume_focused.index))
        # datetime_focused is a 'Dataframe' variable, but it only contains one column, so convert it to Series
        datetime_focused = datetime_focused.iloc[:,0]
        return [price_focused_list, volume_focused, instrument_focused, datetime_focused]

    def run(self, period, std_num):
        # get position signal {1:'long',-1:'short','0':clear}
        args_list = [period, std_num, self.cleartype]
        position_signal_list = self.generate_position_signal(args_list)
        single_return_list, buysell_signal_list = self.get_single_return_with_position(position_signal_list, period)
        compound_return_list = self.get_compound_return(buysell_signal_list)
        res_list = [position_signal_list, buysell_signal_list,single_return_list, compound_return_list]
        return res_list
    def start(self):
        for coinType in self.coin_list:
            for i in range(len(self.three_contract)):
                if i == 2:
                    time_start = time.clock()
                    two_contract = [self.three_contract[i % 3], self.three_contract[(i + 1) % 3]]
                    self.two_contract = two_contract
                    self.roll_test(coinType, two_contract)
                    time_end = time.clock()
                    elapsed = time_end - time_start
                    print('coinType:%s, two_contract:%s,%s complete! elapsed time is:%s' % (
                    coinType, two_contract[0], two_contract[1], str(elapsed)))

        average_type_list = ['bp', 'comp-bp', 'return-dd', 'std-price', 'turnover']
        self.average_result(average_type_list)
    def write_buysell_info_to_file(self, buysell_dir, seq, buysell_info_df):
        file_name = '_'.join(seq) + '.csv'
        result_full_filename = os.path.join(buysell_dir, file_name)
        buysell_info_df.to_csv(result_full_filename)

    def write_result_to_file(self, result_dir, seq, results, index, columns):
        # index , columns = args_list
        file_name = '_'.join(seq) + '.csv'
        result_full_filename = os.path.join(result_dir, file_name)
        result_df = pd.DataFrame(results)
        result_df.index = index
        result_df.columns = columns
        result_df.to_csv(result_full_filename)


    def roll_test(self,coinType, two_contract):

        self.set_directory(self.start_time, self.end_time, self.cleartype)

        data_obj = self.__data
        data_obj.get_data(self.start_time, self.end_time, self.data_dir)
        price_df_list, instrument, datetime, volume = data_obj.load_data_from_file(self.data_dir)
        [self.price_focused_list, self.volume_focused, self.instrument_focused, self.datetime_focused] = \
            self.get_focused_info(price_df_list, instrument, datetime, volume, coinType, two_contract)

        # initialize the variables we need to record
        results_single_return_bp = np.zeros((len(self.window_period_list), len(self.std_num_list)))
        results_compound_return_bp = results_single_return_bp
        results_return_dd = results_single_return_bp
        results_std_price = results_single_return_bp
        results_turnover = results_single_return_bp

        for i in range(len(self.window_period_list)):
            for j in range(len(self.std_num_list)):
                window_period = self.window_period_list[i]
                std_num = self.std_num_list[j]
                # basic routine run(),
                [position_signal_list, buysell_signal_list, single_return_list, compound_return_list] = \
                    self.run(window_period, std_num)

                price_focused_df = self.price_focused_list[3]
                std_divide_price = indicator.get_std_divide_price(price_focused_df, window_period)
                buysell_info = self.get_buysell_info(price_focused_df,position_signal_list, buysell_signal_list, single_return_list, compound_return_list)
                total_turnover = indicator.get_total_turnover(position_signal_list)
                single_return_add = single_return_list[0] + single_return_list[1]

                seq = [coinType, two_contract[0], two_contract[1], self.cleartype, str(window_period),str(std_num)]
                self.write_buysell_info_to_file(self.buysell_dir, seq, buysell_info)

                max_withdraw, dd_start_idx, dd_end_idx = indicator.get_max_drawdown(single_return_add,
                                                                                    self.datetime_focused, self.start_time,
                                                                                    self.end_time)

                results_return_dd[i][j] = np.sum(single_return_add) / abs(max_withdraw)
                results_std_price[i][j] = std_divide_price
                results_turnover[i][j] = total_turnover

                if total_turnover == 0:
                    ave_single_return_bp = 0
                    ave_compound_return_bp = 0
                else:
                    compound_return_sum = np.sum(compound_return_list[0] + compound_return_list[1])
                    single_return_sum = np.sum(single_return_list[0] + single_return_list[1])
                    ave_single_return_bp = single_return_sum / 2 / total_turnover
                    ave_compound_return_bp = compound_return_sum / 2 / total_turnover
                results_single_return_bp[i][j] = ave_single_return_bp
                results_compound_return_bp[i][j] = ave_compound_return_bp

        results_list = [results_single_return_bp,results_compound_return_bp,results_return_dd,results_std_price,results_turnover]
        for i in range(len(results_list)):
            self.average_result_list[i].append(results_list[i])

        seq = [coinType, two_contract[0], two_contract[1], self.cleartype]
        results_list = [results_single_return_bp,results_compound_return_bp,results_return_dd,results_std_price,results_turnover]
        results_tag_list = ['bp','comp-bp','return-dd','std-price','turnover']
        for i in range(len(results_list)):
            seq.append(results_tag_list[i])
            self.write_result_to_file(self.result_dir, seq, results_list[i], self.window_period_list, self.std_num_list)
            del seq[-1]

    # def average_allCoin(self, two_contract, cleartype, average_type):
    #     result_dir, average_dir = self.result_dir,self.average_dir
    #     compile_str = '[a-z]+_%s_%s_%s_%s.csv' % (two_contract[0], two_contract[1], cleartype, average_type)
    #     pattern = re.compile(compile_str)
    #     fileNames = os.listdir(result_dir)
    #     full_fileName = os.path.join(result_dir, fileNames[0])
    #     tmp_df = pd.read_csv(full_fileName)
    #     sum_array = np.zeros(tmp_df.shape)
    #     count = 0
    #     for file in fileNames:
    #         m = pattern.match(file)
    #         if m is not None:
    #             count += 1
    #             filename = m.group()
    #             full_fileName = os.path.join(result_dir, filename)
    #             df = pd.read_csv(full_fileName)
    #             ndarray = df.values
    #             sum_array += ndarray
    #     average_array = sum_array / count
    #     ave_result = average_array[:, 1:]
    #
    #     seq = ['average', two_contract[0], two_contract[1], cleartype, average_type]
    #     self.write_result_to_file(self.average_dir, seq, ave_result, index = tmp_df.iloc[:, 0], columns = tmp_df.columns[1:])

    def average_result(self,average_type_list):
        for i in range(len(average_type_list)):
            sum_array = np.zeros(self.average_result_list[i][0].shape)
            for j in range(len(self.average_result_list[i])):
                sum_array+= self.average_result_list[i][j]
            average_array = sum_array / len(self.average_result_list[i])
            seq = ['average', self.two_contract[0], self.two_contract[1], self.cleartype, average_type_list[i]]
            self.write_result_to_file(self.result_dir, seq, average_array, index=self.window_period_list,
                                      columns=self.std_num_list)


if __name__ == '__main__':
    pass

