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

class SimulationBasic:
    __contract_map = {'week': 0, 'nextweek': 1, 'quarter': 2}
    cleartype_coefficient_dict = {'ceilfloor': 1, 'half-stdnum': 0.5, 'medium': 0, 'threeQuarters-stdnum': 0.75}
    __coinType_list = ['bch', 'btc', 'btg', 'eos', 'etc', 'eth', 'ltc', 'xrp']  # do not change this variable
    __fourPrice_type_list = ['open', 'high', 'low', 'close']
    __project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    __root_data = os.path.join(__project_dir, 'data')
    __root_result = os.path.join(__project_dir, 'backtest')
    __root_average = os.path.join(__project_dir, 'average')
    __root_buysellInfo = os.path.join(__project_dir, 'buysell')
    __singleton = 0
    __average_result_tag= 0
    indi_list_list = []
    start_time = '201806160000'
    db_table = '1min'  # ['1min','5min']
    three_contract = ['week', 'nextweek', 'quarter']
    buy_commission_rate = 0.0000
    sell_commission_rate = 0.0000
    __data = Data()
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
    def is_delivery_time(self, instrument, date_time):
        time_instrument = dt.strptime(instrument[4:] + '1600',"%Y%m%d%H%M")
        time_datetime = dt.strptime(str(date_time),"%Y%m%d%H%M")
        time_diff = (time_instrument - time_datetime).total_seconds()
        delivery_time = False
        if abs(time_diff)<2:
            delivery_time = True
        return delivery_time

    def generate_position_signal(self,):

        [window_period, std_num, cleartype] = [self.window_period,self.std_num,self.cleartype]
        open, high, low, close = self.price_focused_list[0], self.price_focused_list[1], self.price_focused_list[2], self.price_focused_list[3]
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
        instrument_contract1, instrument_contract2 = self.instrument_focused.iloc[:, 0], self.instrument_focused.iloc[:, 1]
        # for i in range(window_period-1, period_mean.shape[0]):
        #     delivery_time1 = self.is_delivery_time(instrument_contract1.iloc[i], self.datetime_focused.iloc[i])
        #     delivery_time2 = self.is_delivery_time(instrument_contract2.iloc[i], self.datetime_focused.iloc[i])
        #     if delivery_time1==True or delivery_time2==True:
        #         position_signal1[i] = 0
        #         position_signal2[i] = 0
        #     else:
        #         if two_contract_diff[i] >= ceil_price[i]:
        #             position_signal1[i] = -1
        #             position_signal2[i] = 1
        #
        #         elif two_contract_diff[i] <= floor_price[i]:
        #             position_signal1[i] = 1
        #             position_signal2[i] = -1
        #         else:
        #             position_signal1[i] = position_signal1[i - 1]
        #             position_signal2[i] = position_signal2[i - 1]
        #
        #         if two_contract_diff[i] >= clear_ceil[i] and two_contract_diff[i] < ceil_price[i]:
        #             if position_signal1[i - 1] == 1:
        #                 position_signal1[i] = 0
        #                 position_signal2[i] = 0
        #         if two_contract_diff[i] > floor_price[i] and two_contract_diff[i] <= clear_floor[i]:
        #             if position_signal1[i - 1] == -1:
        #                 position_signal1[i] = 0
        #                 position_signal2[i] = 0
        position_signal_list = [position_signal1*0.5, position_signal2*0.5]
        return position_signal_list


    def get_buysell_info(self,):
        price_focused_df = self.price_focused_list[3]
        buysell_info = pd.concat(
            [self.datetime_focused,
             price_focused_df.iloc[:, 0], self.position_signal_list[0], self.buysell_signal_list[0], self.single_return_list[0],
             # self.compound_return_list[0],
             price_focused_df.iloc[:, 1], self.position_signal_list[1], self.buysell_signal_list[1], self.single_return_list[1],
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

    def get_focused_info(self,):
        data_obj = self.__data
        data_obj.get_data(self.start_time, self.end_time, self.data_dir)
        price_df_list, instrument, datetime, volume = data_obj.load_data_from_file(self.data_dir)

        column_offset = self.__coinType_list.index(self.coinType) * 3
        contract1_idx = self.__contract_map[self.two_contract[0]]
        contract2_idx = self.__contract_map[self.two_contract[1]]

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

    def start(self):
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
        self.average_result()
        self.write_indicator_to_file()

    def write_indicator_to_file(self):
        file_name = 'indicator'+'.csv'
        full_filename = os.path.join(self.result_dir,file_name)
        f = open(full_filename, "w")
        line = 'cointype,window,std,from,to,ret,tvr,sharp,ret2dd,dd,dd_start,dd_end,mg_bp'
        f.writelines(line)
        f.write('\n')
        for i in range(len(self.indi_list_list)):
            li = list(self.indi_list_list[i])
            li[3+2] = round(li[3+2]*100,4)
            li[4+2] = round(li[4+2]*100,2)
            li[5+2] = round(li[5+2],2)
            li[6+2] = round(li[6+2],2)
            li[7+2] = round(li[7+2]*100,2)
            li[10+2] = round(li[10+2]*10000,2)
            line = ",".join(str(it) for it in li)
            f.writelines(line)
            f.write('\n')
    def roll_test(self,):

        self.set_directory(self.start_time, self.end_time, self.cleartype)
        [self.price_focused_list, self.volume_focused, self.instrument_focused, self.datetime_focused] = self.get_focused_info()

        # initialize the variables we need to record
        results_single_return_bp = np.zeros((len(self.window_period_list), len(self.std_num_list)))
        results_compound_return_bp = np.zeros((len(self.window_period_list), len(self.std_num_list)))
        results_return_dd = np.zeros((len(self.window_period_list), len(self.std_num_list)))
        results_std_price = np.zeros((len(self.window_period_list), len(self.std_num_list)))
        results_turnover = np.zeros((len(self.window_period_list), len(self.std_num_list)))

        for i in range(len(self.window_period_list)):
            for j in range(len(self.std_num_list)):
                self.window_period = self.window_period_list[i]
                self.std_num = self.std_num_list[j]
                self.position_signal_list = self.generate_position_signal()
                # self.compound_return_list = self.get_compound_return(self.buysell_signal_list)
                indi_obj = Indicator(self)
                self.single_return_list  = indi_obj.get_single_return()
                self.buysell_signal_list = indi_obj.get_buysell_signal()
                std2price = indi_obj.get_std_divide_price()

                ret = indi_obj.get_total_return()
                tvr = indi_obj.get_total_turnover()
                mean_tvr = indi_obj.get_mean_turnover()
                dd, dd_start, dd_end = indi_obj.get_max_drawdown()
                ret_unitTime_list,endtime_list = indi_obj.get_return_for_unitTime()
                sharp = indi_obj.get_sharp()
                ret_unit_df = pd.DataFrame({'endtime':endtime_list,'return':ret_unitTime_list})
                ret2dd = indi_obj.get_return_divide_dd()
                mg_bp = indi_obj.get_margin_bp()
                # long,short = 0.5,-0.5
                indi_list = [self.coinType, self.window_period, self.std_num, self.start_time, self.end_time, ret, mean_tvr,sharp,ret2dd,dd,dd_start,dd_end,mg_bp]
                self.indi_list_list.append(indi_list)
                # print(indi_list)
                self.buysell_info = self.get_buysell_info()
                seq = [self.coinType, self.two_contract[0], self.two_contract[1], self.cleartype, str(self.window_period),str(self.std_num)]
                self.write_buysell_info_to_file(self.buysell_dir, seq, self.buysell_info)
                seq.append('return')
                self.write_buysell_info_to_file(self.buysell_dir, seq, ret_unit_df)
                del seq[-1]

                results_return_dd[i][j] = ret2dd
                results_std_price[i][j] = std2price
                results_turnover[i][j] = tvr
                results_single_return_bp[i][j] = mg_bp

        seq = [self.coinType, self.two_contract[0], self.two_contract[1], self.cleartype]
        results_list = [results_single_return_bp,results_return_dd,results_turnover]
        self.results_tag_list = ['mg-bp','ret2dd','turnover']
        self.get_average_result_list(self.results_tag_list)
        for i in range(len(results_list)):
            # record the data results for each coin type, it's used for calculating the average results
            self.average_result_list[i].append(results_list[i])
            seq.append(self.results_tag_list[i])
            self.write_result_to_file(self.result_dir, seq, results_list[i], self.window_period_list, self.std_num_list)
            del seq[-1]

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
            seq = ['average', self.two_contract[0], self.two_contract[1], self.cleartype, self.results_tag_list[i]]
            self.write_result_to_file(self.result_dir, seq, average_array, index=self.window_period_list,
                                      columns=self.std_num_list)


if __name__ == '__main__':
    pass

