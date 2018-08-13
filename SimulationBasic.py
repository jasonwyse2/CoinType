import os
from datetime import datetime as dt

import time
import pandas as pd
import numpy as np
from tool import *
import pymysql
from indicator import Indicator
import tool
class Data:
    db_host_list = ['192.168.0.113', '206.189.89.22', '192.168.0.113', '127.0.0.1']  # 192.168.0.113, 206.189.89.22
    db_port_list = [3306, 5555, 3306, 3306]  # 3306 , 5555
    db_user_list = ['root', 'linjuninvestment', 'root', 'root']  # root, linjuninvestment
    db_pass_list = ['1qazxsw2', '123456', '1qazxsw2', '1qazxsw2']  # 1qazxsw2, 123456
    db_name_list = ['okex', 'tradingdata', 'yg', 'okex']  # okcoin, tradingdata

    contract_timeType = ['week', 'nextweek', 'quarter']
    __fourPrice_type_list = ['open', 'high', 'low', 'close']
    # 'btc', 'bch', 'eth', 'etc', 'eos'
    coinType_list = ['bch', 'btc', 'eos', 'etc', 'eth']#['bch', 'btc', 'eos', 'etc', 'eth']  # do not change this variable
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
        field_list = ['datetime']
        fields = ','.join(field_list)
        sql = 'SELECT '+ fields +' FROM okex.' + db_table + \
              ' WHERE DATETIME >=%d and datetime<%d and volume>0 order by datetime asc'%(int(start_time), int(end_time))
        df = pd.read_sql(sql, conn)
        return df

    def get_columns_cointype_df(self,coinType, start_time, end_time):
        conn = self.login_MySQL(3)
        field_list = ['open', 'high', 'low','close', 'datetime', 'instrument', 'volume']
        fields = ','.join(field_list)
        sql_datetime = 'SELECT '+fields+' FROM okex.' + self.db_table +\
                       ' WHERE DATETIME >=%d and datetime<%d and volume>0'%(int(start_time), int(end_time))
        sql_filter = 'instrument LIKE "%s' % (coinType) + '%"'
        # concatenate sql string
        sql = sql_datetime + ' and ' + sql_filter + ' order by instrument'
        df = pd.read_sql(sql, conn)
        return df

    def add_contract_price(self,df_data, contract_timeType, coinType, datetime_open, datetime_high, datetime_low,
                           datetime_close):
        datetime_open = pd.merge(datetime_open, df_data[['open', 'datetime']], on='datetime', how='left')
        datetime_open.rename(columns={'open': coinType + '_' + contract_timeType}, inplace=True)

        datetime_high = pd.merge(datetime_high, df_data[['high', 'datetime']], on='datetime', how='left')
        datetime_high.rename(columns={'high': coinType + '_' + contract_timeType}, inplace=True)


        datetime_low = pd.merge(datetime_low, df_data[['low', 'datetime']], on='datetime', how='left')
        datetime_low.rename(columns={'low': coinType + '_' + contract_timeType}, inplace=True)

        datetime_close = pd.merge(datetime_close, df_data[['close', 'datetime']], on='datetime', how='left')
        datetime_close.rename(columns={'close': coinType + '_' + contract_timeType}, inplace=True)

        return [datetime_open, datetime_high, datetime_low, datetime_close]

    def add_contract_instrument(self,df_data, contract_timeType, coinType, instrument):
        instrument = pd.merge(instrument, df_data[['instrument', 'datetime']], on='datetime', how='left')
        instrument.rename(columns={'instrument': coinType + '_' + contract_timeType}, inplace=True)
        return instrument

    def add_contract_volume(self,df_data, contract_timeType, coinType, volume):
        volume = pd.merge(volume, df_data[['volume', 'datetime']], on='datetime', how='left')
        volume.rename(columns={'volume': coinType + '_' + contract_timeType}, inplace=True)
        return volume

    def quick_datetime_symbol(self,start_time, end_time):
        datetime = self.get_unique_datetime(start_time, end_time)
        datetime_open = datetime.copy()
        datetime_high, datetime_low, datetime_close = datetime.copy(), datetime.copy(), datetime.copy()
        instrument = datetime.copy()
        volume = datetime.copy()
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
                [datetime_open, datetime_high, datetime_low,datetime_close] \
                    = self.add_contract_price(df_list[i], contract_timetype_list[i], coin_Type, datetime_open, datetime_high,
                                              datetime_low, datetime_close)
                instrument = self.add_contract_instrument(df_list[i], contract_timetype_list[i], coin_Type, instrument)
                volume = self.add_contract_volume(df_list[i], contract_timetype_list[i], coin_Type, volume)

        price_df_list = [datetime_open, datetime_high, datetime_low, datetime_close]
        return [price_df_list, instrument, volume,datetime]

    def append_df_to_file(self):
        pass
    def check_update_data(self,end_time,data_dir):

        datetime = self.load_datetime_from_file(data_dir)
        recorded_end_time = str(datetime.iloc[-1,0])
        if recorded_end_time<end_time:
            start_time = tool.currentTime_forward_delta(recorded_end_time, min_deltaTime=1)
            if(start_time<end_time):
                [price_df_list_added, instrument_added, volume_added,datetime_added] = self.quick_datetime_symbol(start_time, end_time)
                append_df_to_file(datetime_added,data_dir,filename='datetime')
                append_df_to_file(instrument_added, data_dir, filename='instrument')
                append_df_to_file(volume_added, data_dir, filename='volume')
                p_list = self.__fourPrice_type_list
                for i in range(len(p_list)):
                    append_df_to_file(price_df_list_added[i], data_dir, filename=p_list[i])

    def get_data(self, start_time, end_time, data_dir):
        time_start = time.clock()
        flag = mkdir(data_dir)
        p_list = self.__fourPrice_type_list
        if flag == 0 or not os.listdir(data_dir):
            [price_df_list, instrument, volume,datetime] = self.quick_datetime_symbol(start_time, end_time)

            write_df_to_file(datetime, data_dir, filename='datetime')
            write_df_to_file(instrument, data_dir, filename='instrument')
            write_df_to_file(volume, data_dir, filename='volume')
            for i in range(len(p_list)):
                write_df_to_file(price_df_list[i], data_dir, filename=p_list[i])
        else:
            pass
            self.check_update_data(end_time, data_dir)
        time_end = time.clock()
        elapsed = time_end - time_start

    def load_data_from_file(self, start_time, end_time,dest_dir):
        fullfileName_datetime = os.path.join(dest_dir, 'datetime.csv')
        datetime = pd.read_csv(fullfileName_datetime)
        datetime_array = datetime.iloc[:,0].values
        index_range1 = datetime_array >= int(start_time)
        index_range2 = datetime_array < int(end_time)
        index_range = index_range1&index_range2
        price_df_list = []
        for price_type in self.__fourPrice_type_list:
            fileName_price = price_type + '.csv'
            fullfileName_price = os.path.join(dest_dir, fileName_price)
            price = pd.read_csv(fullfileName_price).iloc[:,1:]
            price = price[index_range]
            price_df_list.append(price)


        datetime = datetime[index_range]
        fullfileName_instrument = os.path.join(dest_dir, 'instrument.csv')
        instrument = pd.read_csv(fullfileName_instrument).iloc[:,1:]
        instrument = instrument[index_range]
        fullfileName_volume = os.path.join(dest_dir, 'volume.csv')
        volume = pd.read_csv(fullfileName_volume).iloc[:,1:]
        volume = volume[index_range]
        return price_df_list, instrument, datetime, volume

    def load_datetime_from_file(self,dest_dir):
        fullfileName_datetime = os.path.join(dest_dir, 'datetime.csv')
        datetime = pd.read_csv(fullfileName_datetime)
        return datetime

class SimulationBasic:
    __contract_map = {'week': 0, 'nextweek': 1, 'quarter': 2}
    strategy_coefficient_dict = {'ceilfloor': 1, 'half-stdnum': 0.5, 'medium': 0, 'threeQuarters-stdnum': 0.75}
    __coinType_list = ['bch', 'btc', 'eos', 'etc', 'eth']#['bch', 'btc', 'eos', 'etc', 'eth']  # do not change this variable
    __fourPrice_type_list = ['open', 'high', 'low', 'close']
    __project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    __root_data = os.path.join(__project_dir, 'data')
    __root_result = os.path.join(__project_dir, 'backtest')
    # __root_average = os.path.join(__project_dir, 'average')
    __root_buysellInfo = os.path.join(__project_dir, 'buysell')
    __directory_singleton_flag = 0
    __average_result_tag= 0
    __position_signal_array_flag=0
    __buysell_signal_array_flag = 0
    indi_list_list = []
    position_signal_array_list = []
    start_time = '201806160000'
    db_table = '1min'  # ['1min','5min']
    three_contract = ['week', 'nextweek', 'quarter']
    buy_commission_rate = 0.0000
    sell_commission_rate = 0.0000
    timestamp = 0
    __data = Data()
    data_dir = ''

    def __init__(self,):
        if not hasattr(self, 'end_time'):
            self.end_time = dt.now().strftime('%Y%m%d')+'0000'
    def initialize_variables(self):
        if self.__sigleton_variable_flag==0:
            self.buysell_signal_array = np.zeros(self.position_signal_array.shape)
            self.single_return_array = np.zeros(self.position_signal_array.shape)
    @property
    def project_dir(self):
        return self.__project_dir
    @project_dir.setter
    def project_dir(self,value):
        self.__project_dir = value

    def get_coinType_contractType_index(self,coinType,contractType):
        column_offset = self.__coinType_list.index(coinType) * 3
        contract_idx = self.__contract_map[contractType]
        idx = column_offset + contract_idx
        return idx
    def set_directory(self, start_time, end_time, strategy_name):
        # Singleton mode, this function can only be called once
        if self.__directory_singleton_flag == 0:
            db_table = self.__data.db_table
            father_dir = self.__project_dir
            root_data = os.path.join(father_dir, 'data')
            root_result = os.path.join(father_dir, 'backtest')
            root_buysellInfo = os.path.join(father_dir, 'buysell')

            fileName_item_list = [db_table, start_time, end_time]  # [db_table, start_time, end_time]
            db_start_end = '-'.join(fileName_item_list)

            self.data_dir = os.path.join(root_data, db_table)
            if self.timestamp ==1:
                now_str = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
            else:
                now_str=''
            self.result_dir = os.path.join(root_result, db_start_end, strategy_name, now_str)
            self.buysell_dir = os.path.join(root_buysellInfo, db_start_end, strategy_name, now_str)

            mkdir(self.data_dir)
            mkdir(self.result_dir)
            mkdir(self.buysell_dir)

            self.__directory_singleton_flag = 1
    def is_delivery_time(self, instrument, date_time):
        time_instrument = dt.strptime(instrument[4:] + '1600',"%Y%m%d%H%M")
        time_datetime = dt.strptime(str(date_time),"%Y%m%d%H%M")
        time_diff = (time_instrument - time_datetime).total_seconds()
        delivery_time = False
        if abs(time_diff)<2:
            delivery_time = True
        return delivery_time

    def generate_position_signal(self,):

        [window_period, std_num, strategy_name] = [self.window_period, self.std_num, self.strategy_name]
        open, high, low, close = self.price_focused_list[0].values, self.price_focused_list[1].values, \
                                 self.price_focused_list[2].values, self.price_focused_list[3].values
        volume_focused = self.volume_focused.values
        datetime_focused = self.datetime_focused.values
        instrument_focused = self.instrument_focused.values

        two_contract_diff = close[:, 0] - close[:, 1]
        period_mean = (pd.Series(two_contract_diff).rolling(window_period).mean()).values
        period_std = (pd.Series(two_contract_diff).rolling(window_period).std()).values
        ceil_price = period_mean + period_std * std_num
        floor_price = period_mean - period_std * std_num

        cleartype_coefficient = 0.5
        clear_ceil = period_mean + period_std * std_num * cleartype_coefficient
        clear_floor = period_mean - period_std * std_num * cleartype_coefficient

        position_signal1 = np.array([0] * two_contract_diff.shape[0])
        position_signal2 = np.array([0] * two_contract_diff.shape[0])
        instrument_contract1, instrument_contract2 = \
            instrument_focused[:, 0], instrument_focused[:, 1]
        for i in range(window_period - 1, period_mean.shape[0]):
            delivery_time1 = self.is_delivery_time(instrument_contract1[i], datetime_focused[i])
            delivery_time2 = self.is_delivery_time(instrument_contract2[i], datetime_focused[i])
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
        position_signal_list = [pd.Series(position_signal1 * 0.5), pd.Series(position_signal2 * 0.5)]
        return position_signal_list

    def get_buysell_info(self,):
        price_df_list = self.price_df_list_nan
        close = price_df_list[3]
        idx1 = self.idx1
        idx2 = self.idx2
        buysell_info = pd.concat(
            [self.datetime,
             close.iloc[:, idx1], pd.Series(self.position_signal_array[:,idx1]), pd.Series(self.buysell_signal_array[:,idx1]), pd.Series(self.single_return_array[:,idx1]),
             # self.compound_return_list[0],
             close.iloc[:, idx2], pd.Series(self.position_signal_array[:,idx2]), pd.Series(self.buysell_signal_array[:,idx2]), pd.Series(self.single_return_array[:,idx2]),
             # self.compound_return_list[1]
             ],
            axis=1)
        buysell_info.columns = ['datetime',
                                'price1', 'position1', 'buysell1', 'single_return1',
                                # 'compound_return1',
                                'price2', 'position2', 'buysell2', 'single_return2',
                                # 'compound_return2'
                                ]
        return buysell_info
    def fill_NaN_with_previous(self,price_df_list_nan):
        price_df_list = []
        for i in range(len(price_df_list_nan)):
            price_df = price_df_list_nan[i].copy()
            price_array = (price_df.values).copy()

            idx = np.argwhere(np.isnan(price_array))

            for j in range(idx.shape[0]):
                price_array[idx[j][0],idx[j][1]] = price_array[idx[j][0]-1,idx[j][1]]

            price_df_list.append(pd.DataFrame(price_array))
        return price_df_list
    def get_focused_info(self,):
        data_obj = self.__data
        data_obj.get_data(self.start_time, self.end_time, self.data_dir)
        if not hasattr(self, 'price_df_list'):
            self.price_df_list_nan, self.instrument, self.datetime, self.volume = \
                data_obj.load_data_from_file(self.start_time, self.end_time,self.data_dir)
            self.price_df_list = self.fill_NaN_with_previous(self.price_df_list_nan)
        column_offset = self.__coinType_list.index(self.coinType) * 3
        contract1_idx = self.__contract_map[self.two_contract[0]]
        contract2_idx = self.__contract_map[self.two_contract[1]]

        price_focused_list = []
        for i in range(len(self.__fourPrice_type_list)):
            # price = self.price_df_list[i]
            price = self.price_df_list_nan[i]
            price.columns = range(len(price.columns))
            price_focused_nan = price[[column_offset + contract1_idx, column_offset + contract2_idx]]
            price_focused = price_focused_nan.dropna(axis=0, how='any')
            # price_focused = price_focused_nan
            price_focused_list.append(price_focused)
        self.instrument.columns = range(len(self.instrument.columns))
        self.volume.columns = range(len(self.volume.columns))

        instrument_focused_nan = self.instrument[[column_offset + contract1_idx, column_offset + contract2_idx]]
        volume_focused_nan = self.volume[[column_offset + contract1_idx, column_offset + contract2_idx]]
        # remove rows that contains 'nan'
        price_focused = price_focused_list[0]
        instrument_focused = instrument_focused_nan.iloc[price_focused.index.tolist(), :]
        datetime_focused = self.datetime.iloc[price_focused.index.tolist(), :]
        volume_focused = volume_focused_nan.iloc[price_focused.index.tolist(), :]
        # reset index from 0, because some index has been removed
        # for price_focused in price_focused_list:
        #     price_focused.index = range(len(price_focused.index))
        # instrument_focused.index = range(len(instrument_focused.index))
        # datetime_focused.index = range(len(datetime_focused.index))
        # volume_focused.index = range(len(volume_focused.index))
        # # datetime_focused is a 'Dataframe' variable, but it only contains one column, so convert it to Series
        # datetime_focused = datetime_focused.iloc[:,0]
        return [price_focused_list, volume_focused, instrument_focused, datetime_focused]
    def get_positon_signal_array(self):
        if self.__position_signal_array_flag==0:
            self.position_signal_array = np.zeros(self.instrument.shape)
            self.__position_signal_array_flag == 1
        return self.position_signal_array
    def get_buysell_signal_array(self):
        if self.__buysell_signal_array_flag==0:
            self.buysell_signal_array = np.zeros(self.instrument.shape)
            self.__buysell_signal_array_flag == 1
        return self.buysell_signal_array
    def map_signal_focused_to_array(self, position_signal_focused):
        price = self.price_df_list_nan[3].values
        # position_signal_array = self.get_positon_signal_array()
        coinType = self.coinType
        two_contract = self.two_contract
        idx1 = self.get_coinType_contractType_index(coinType, two_contract[0])
        idx2 = self.get_coinType_contractType_index(coinType, two_contract[1])
        self.idx1,self.idx2 = idx1,idx2
        price_diff = price[:,idx1]-price[:,idx2]

        position_signal_array = self.get_positon_signal_array()
        index = self.volume_focused.index.values
        position_signal_array[index,idx1] = position_signal_focused[:,0]
        position_signal_array[index,idx2] = position_signal_focused[:,1]

        index_row = np.argwhere(np.isnan(price_diff))[:, 0]
        for i in range(len(index_row)):
            position_signal_array[index_row[i],idx1]=self.position_signal_array[index_row[i]-1,idx1]
            position_signal_array[index_row[i], idx2] = self.position_signal_array[index_row[i] - 1, idx2]
        return position_signal_array

    def write_buysell_info_to_file(self, buysell_dir, seq, buysell_info_df):
        file_name = '_'.join(seq) + '.csv'
        result_full_filename = os.path.join(buysell_dir, file_name)
        buysell_info_df.to_csv(result_full_filename)

    def write_result_to_file(self, result_dir, seq, results, index, columns):
        file_name = '_'.join(seq) + '.csv'
        result_full_filename = os.path.join(result_dir, file_name)
        result_df = pd.DataFrame(results)
        result_df.index = index
        result_df.columns = columns
        result_df.to_csv(result_full_filename)

    def write_indicator_to_file(self):
        file_name = 'perf'+'.csv'
        full_filename = os.path.join(self.result_dir,file_name)
        f = open(full_filename, "w")
        line = 'cointype,window,std,from,to,ret,tvr,sharp,ret2dd,dd,dd_start,dd_end,mg_bp'
        cols = ['cointype','window','std','from','to','ret','tvr','sharp','ret2dd','dd','dd_start','dd_end','mg_bp']

        f.writelines(line)
        f.write('\n')
        for i in range(len(self.indi_list_list)):
            li = (self.indi_list_list[i])
            li[3] = str(li[3])[:-4]
            li[4] = str(li[4])[:-4]
            line = ",".join(str(it) for it in li)
            f.writelines(line)
            f.write('\n')
        perf_res = pd.DataFrame(self.indi_list_list)
        perf_res.columns = cols
        print perf_res
    def get_average_result_list(self,results_tag_list):
        if self.__average_result_tag ==0:
            average_result_list = []
            for i in range(len(results_tag_list)):
                average_result_list.append([])
            self.average_result_list = average_result_list
            self.__average_result_tag = 1
        return self.average_result_list

    def average_result(self,):
        for i in range(len(self.results_tag_list)):
            sum_array = np.zeros(self.average_result_list[i][0].shape)
            for j in range(len(self.average_result_list[i])):
                sum_array+= self.average_result_list[i][j]
            average_array = sum_array / len(self.average_result_list[i])
            seq = ['average', self.two_contract[0], self.two_contract[1], self.strategy_name, self.results_tag_list[i]]
            self.write_result_to_file(self.result_dir, seq, average_array, index=self.window_period_list,
                                      columns=self.std_num_list)
    def get_average_position_signal_array(self):
        position_signal_array_list = self.position_signal_array_list
        average_position_signal_array = np.zeros(position_signal_array_list[0].shape)
        for i in range(len(position_signal_array_list)):
            average_position_signal_array += position_signal_array_list[i]

        divisor = len(self.coin_list) * len(self.window_period_list) * len(self.std_num_list)
        (rows, cols) = position_signal_array_list[i].shape
        divisor_array = np.array([divisor] * rows * cols).reshape((rows, cols))
        average_position_signal_array_norm = average_position_signal_array / divisor_array
        return average_position_signal_array_norm
    def start(self):
        self.indi_list_list = []
        for coinType in self.coin_list:
            for i in range(len(self.three_contract)):
                if i == 2:
                    time_start = time.clock()
                    two_contract = [self.three_contract[i % 3], self.three_contract[(i + 1) % 3]]
                    self.two_contract = two_contract
                    self.coinType = coinType
                    self.roll_test()
                    time_end = time.clock()
                    elapsed = time_end - time_start
                    print('coinType:%s, two_contract:%s,%s complete! elapsed time is:%s' % (
                    coinType, two_contract[0], two_contract[1], str(elapsed)))



        cols = ['cointype', 'window', 'std', 'from', 'to', 'ret', 'tvr', 'sharp', 'ret2dd', 'dd', 'dd_start', 'dd_end',
                'mg_bp']
        average_position_signal_array_norm = self.get_average_position_signal_array()
        indi_obj = Indicator(self, average_position_signal_array_norm)
        ret = indi_obj.get_total_return()
        mean_tvr = indi_obj.get_mean_turnover()
        dd, dd_start, dd_end = indi_obj.get_max_drawdown()
        sharp = indi_obj.get_sharp()
        ret2dd = indi_obj.get_return_divide_dd()
        mg_bp = indi_obj.get_margin_bp()

        indi_list = ['average', self.window_period, self.std_num, self.start_time[:-4], self.end_time[:-4], ret,
                     mean_tvr, sharp, ret2dd, dd, dd_start, dd_end, mg_bp]
        self.indi_list_list.append(indi_list)

        perf_df = pd.DataFrame(self.indi_list_list)
        perf_df.columns = cols
        print(perf_df)
        # self.average_result()
        # self.write_indicator_to_file()
    def get_buysell_signal(self, position_signal_focused):

        position_diff_array = np.diff(position_signal_focused, axis=0)
        buysell_signal_focused = np.zeros(self.instrument_focused.shape)
        # buysell_signal_focused[2:,:] = position_diff_array[:-1,:]
        buysell_signal_focused[1:, :] = position_diff_array
        return buysell_signal_focused
    def map_buysell_signal_focused_to_position_signal_array(self,buysell_signal_focused,buysell_signal_array):
        price = self.price_df_list_nan[3].values

        coinType = self.coinType
        two_contract = self.two_contract
        idx1 = self.get_coinType_contractType_index(coinType, two_contract[0])
        idx2 = self.get_coinType_contractType_index(coinType, two_contract[1])
        self.idx1, self.idx2 = idx1, idx2
        price_diff = price[:, idx1] - price[:, idx2]

        values_pos = self.position_signal_focused[self.position_signal_focused != 0].reshape((-1, 2))
        # position_signal_array = self.get_positon_signal_array()
        index = self.volume_focused.index.values
        buysell_signal_array[index, idx1] += buysell_signal_focused[:, 0]
        buysell_signal_array[index, idx2] += buysell_signal_focused[:, 1]

        index = np.argwhere(buysell_signal_array!=0)
        values = buysell_signal_array[buysell_signal_array!=0].reshape((-1,2))
        return buysell_signal_array

    def roll_test(self,):

        self.set_directory(self.start_time, self.end_time, self.strategy_name)
        [self.price_focused_list, self.volume_focused,
         self.instrument_focused, self.datetime_focused] = self.get_focused_info()
        results_single_return_bp = np.zeros((len(self.window_period_list), len(self.std_num_list)))
        results_return_dd = np.zeros((len(self.window_period_list), len(self.std_num_list)))
        results_turnover = np.zeros((len(self.window_period_list), len(self.std_num_list)))

        for i in range(len(self.window_period_list)):
            for j in range(len(self.std_num_list)):
                self.window_period = self.window_period_list[i]
                self.std_num = self.std_num_list[j]
                position_signal_focused = self.generate_position_signal()

                position_signal_array = self.map_signal_focused_to_array(position_signal_focused)
                self.position_signal_array_list.append(position_signal_array)

                indi_obj = Indicator(self,position_signal_array)
                self.buysell_signal_array = indi_obj.get_buysell_signal()
                # self.compound_return_array = indi_obj.get_compound_return()
                self.single_return_array = indi_obj.get_single_return()

                self.buysell_info = self.get_buysell_info()
                seq = [self.coinType, self.two_contract[0], self.two_contract[1], self.strategy_name,
                       str(self.window_period), str(self.std_num)]
                self.write_buysell_info_to_file(self.buysell_dir, seq, self.buysell_info)
                seq.append('day-return')
                ret_unitTime_list, endtime_list = indi_obj.get_return_for_unitTime()
                ret_unit_df = pd.DataFrame({'endtime': endtime_list, 'return': ret_unitTime_list})
                self.write_buysell_info_to_file(self.buysell_dir, seq, ret_unit_df)
                del seq[-1]

                ret = indi_obj.get_total_return()
                tvr = indi_obj.get_total_turnover()
                mean_tvr = indi_obj.get_mean_turnover()
                dd, dd_start, dd_end = indi_obj.get_max_drawdown()
                ret_unitTime_list, endtime_list = indi_obj.get_return_for_unitTime()
                sharp = indi_obj.get_sharp()
                ret2dd = indi_obj.get_return_divide_dd()
                mg_bp = indi_obj.get_margin_bp()

                results_return_dd[i][j] = ret2dd
                results_turnover[i][j] = mean_tvr
                results_single_return_bp[i][j] = mg_bp
                indi_list = [self.coinType, self.window_period, self.std_num, self.start_time[:-4], self.end_time[:-4], ret,
                             mean_tvr, sharp, ret2dd, dd, dd_start, dd_end, mg_bp]
                self.indi_list_list.append(indi_list)

        seq = [self.coinType, self.two_contract[0], self.two_contract[1], self.strategy_name]
        results_list = [results_single_return_bp, results_return_dd, results_turnover]
        self.results_tag_list = ['mg-bp', 'ret2dd', 'turnover']
        self.get_average_result_list(self.results_tag_list)
        for i in range(len(results_list)):
            # record the data results for each coin type, it's used for calculating the average results
            self.average_result_list[i].append(results_list[i])
            seq.append(self.results_tag_list[i])
            self.write_result_to_file(self.result_dir, seq, results_list[i], self.window_period_list, self.std_num_list)
            del seq[-1]

if __name__ == '__main__':
    pass

