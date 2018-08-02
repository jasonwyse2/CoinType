from CTA import mkdir
import os
from datetime import datetime
import time
import pandas as pd
import numpy as np
from tool import *
import pymysql
import indicator
import re
class DATA:
    db_host_list = ['192.168.0.113', '206.189.89.22', '192.168.0.113', '127.0.0.1']  # 192.168.0.113, 206.189.89.22
    db_port_list = [3306, 5555, 3306, 3306]  # 3306 , 5555
    db_user_list = ['root', 'linjuninvestment', 'root', 'root']  # root, linjuninvestment
    db_pass_list = ['1qazxsw2', '123456', '1qazxsw2', '1qazxsw2']  # 1qazxsw2, 123456
    db_name_list = ['okex', 'tradingdata', 'yg', 'okex']  # okcoin, tradingdata

    contract_timeType = ['week', 'nextweek', 'quarter']
    fourPrice_type_list = ['open', 'high', 'low', 'close']
    coinType_list = ['bch', 'btc', 'btg', 'eos', 'etc', 'eth', 'ltc', 'xrp']
    one_week_seconds = 7 * 24 * 3600
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

    def get_unique_datetime(self,start_time, end_time, db_table):
        df = self.get_datetime_allUnique_df(start_time, end_time, db_table)
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

    def get_datetime_allUnique_df(self,start_time, end_time, db_table):
        conn = self.login_MySQL(3)
        field_list = ['open','high','close','low' ,'datetime', 'instrument','volume']
        fields = ','.join(field_list)
        sql = 'SELECT '+ fields +' FROM okex.' + db_table + \
              ' WHERE DATETIME >=%d and datetime<=%d and volume>0 order by datetime asc'%(int(start_time), int(end_time))
        df = pd.read_sql(sql, conn)
        return df

    def get_columns_cointype_df(self,coinType, start_time, end_time, db_table):
        conn = self.login_MySQL(3)
        field_list = ['open', 'high', 'close', 'low', 'datetime', 'instrument', 'volume']
        fields = ','.join(field_list)
        sql_datetime = 'SELECT '+fields+' FROM okex.' + db_table +\
                       ' WHERE DATETIME >=%d and datetime<=%d and volume>0'%(int(start_time), int(end_time))
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

    def quick_datetime_symbol(self,start_time, end_time, db_table):

        datetime_open = self.get_unique_datetime(start_time, end_time, db_table)
        datetime_high, datetime_close, datetime_low = datetime_open, datetime_open, datetime_open
        instrument = self.get_unique_datetime(start_time, end_time, db_table)
        volume = self.get_unique_datetime(start_time, end_time, db_table)
        one_week_seconds = self.one_week_seconds
        for coin_Type in self.coinType_list:
            df = self.get_columns_cointype_df(coin_Type, start_time, end_time, db_table)
            df['instrument_precise'] = df.instrument.map(lambda x: x[len(coin_Type) + 1:] + '1600')
            # transfer 'datetime' and instrument into date format
            df['datetime_str'] = df.datetime.map(lambda x: str(x))
            df['seconds_diff'] = df.apply(lambda x: (datetime.strptime(x['instrument_precise'], "%Y%m%d%H%M") - datetime.strptime(x['datetime_str'],"%Y%m%d%H%M")).total_seconds(), axis=1)

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

    def get_data(self, start_time, end_time, data_dir, db_table):
        time_start = time.clock()
        flag = mkdir(data_dir)
        p_list = ['open', 'high', 'close', 'low']  # the order can't be changed
        if flag == 0 or not os.listdir(data_dir):
            df_list = self.quick_datetime_symbol(start_time, end_time, db_table)
            price_df_list = df_list[0]
            instrument = df_list[1]
            volume = df_list[2]
            datetime = instrument.iloc[:, 0]

            write_df_to_file(datetime, data_dir, filename='datetime')
            write_df_to_file(instrument, data_dir, filename='instrument')
            write_df_to_file(volume, data_dir, filename='volume')
            for i in range(len(p_list)):
                write_df_to_file(price_df_list[i], data_dir, filename=p_list[i])
        time_end = time.clock()
        elapsed = time_end - time_start

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
    __data = DATA()
    def __init__(self,):
        pass

    @property
    def project_dir(self):
        return self.__project_dir
    @project_dir.setter
    def project_dir(self,value):
        self.__project_dir = value

    def generate_position_signal(self, price_focused_list, volume_focused, instrument_focused, args_list):
        [period, std_num, cleartype] = args_list

        open, high, low, close = price_focused_list[0],price_focused_list[1],price_focused_list[2],price_focused_list[3]
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

        instrument_contract1, instrument_contract2 = instrument_focused.iloc[:, 0], instrument_focused.iloc[:, 1]

        for i in range(period, period_mean.shape[0]):
            idx = i
            if instrument_contract1[i] != instrument_contract1[i - 1]:
                position_signal1[i - 1] = 0
                position_signal2[i - 1] = 0
            else:
                if two_contract_diff[i] >= ceil_price[idx]:
                    position_signal1[i] = -1
                    position_signal2[i] = 1

                elif two_contract_diff[i] <= floor_price[idx]:
                    position_signal1[i] = 1
                    position_signal2[i] = -1
                else:
                    position_signal1[i] = position_signal1[i - 1]
                    position_signal2[i] = position_signal2[i - 1]

                if two_contract_diff[i] >= clear_ceil[idx] and two_contract_diff[i] < ceil_price[idx]:
                    if position_signal1[i - 1] == 1:
                        position_signal1[i] = 0
                        position_signal2[i] = 0
                if two_contract_diff[i] > floor_price[idx] and two_contract_diff[i] <= clear_floor[idx]:
                    if position_signal1[i - 1] == -1:
                        position_signal1[i] = 0
                        position_signal2[i] = 0
        return [position_signal1, position_signal2]

    def get_std_divide_price(self,price_focused, period):
        two_contract_diff = price_focused.iloc[:, 0] - price_focused.iloc[:, 1]
        price1 = price_focused.iloc[:, 0]
        period_mean = two_contract_diff.rolling(period).mean()
        period_std = two_contract_diff.rolling(period).std()
        std_divide_price = np.mean(period_std / price1)
        return std_divide_price

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

    def get_return(self, price_focused_list, BuySell_signal_list, args_list):
        [commission_rate_buy,commission_rate_sell] = args_list
        open_focused, high_focused, low_focused, close_focused = price_focused_list[0], price_focused_list[1], price_focused_list[2], \
                                 price_focused_list[3]

        return_list = []
        for i in range(2):
            total_return = pd.Series([0.0] * close_focused.shape[0])
            BuySell_signal = BuySell_signal_list[i]
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
            return_list.append(total_return)
            # for j in range(1,buysell_price.shape[0]):
            #     if(buysell_signal[j]==-1 and holding_position[j-1]==1):
            #         total_return += buysell_price[j]/buysell_price[j-1]-1-(commission_rate_buy+commission_rate_sell)
            #     elif(buysell_signal[j]==-2 and holding_position[j-1]==1):
            #         total_return += buysell_price[j] / buysell_price[j-1] - 1-(commission_rate_buy+commission_rate_sell)
            #     elif(buysell_signal[j]==1 and holding_position[j-1]==-1):
            #         total_return += 1-buysell_price[j-1]/buysell_price[j] - (commission_rate_buy+commission_rate_sell)
            #     elif(buysell_signal[j]==2 and holding_position[j-1]==-1):
            #         total_return += 1-buysell_price[j-1]/buysell_price[j] - (commission_rate_buy+commission_rate_sell)

        return return_list

    def get_buysell_signal(self,position_signal):
        BuySell_signal = indicator.get_commission_signal(position_signal)
        return BuySell_signal
    def get_return_plus_commission(self, price_focused, position_signal, period, commission_rate_buy=0,
                                   commission_rate_sell=0):
        # price_focused = price_focused_list[3]
        per_return_contract = np.diff(price_focused) / price_focused[:-1]
        BuySell_signal = indicator.get_commission_signal(position_signal)
        contract_commission = pd.Series([0] * price_focused.shape[0])
        contract_commission[BuySell_signal > 0] = (BuySell_signal * commission_rate_buy)[BuySell_signal > 0] * (-1)
        contract_commission[BuySell_signal < 0] = (BuySell_signal * commission_rate_sell)[BuySell_signal < 0]

        contract_return = pd.Series([0] * price_focused.shape[0])
        contract_return[period + 1:] = np.array(position_signal[period - 1:-2]) * np.array(
            per_return_contract[period:])

        contract_return_plus_commission = contract_return + contract_commission
        return contract_return_plus_commission,BuySell_signal

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
        return [price_focused_list, instrument_focused, datetime_focused,volume_focused]

    def run(self, start_time, end_time, db_table, price_type, period, coinType, std_num, two_contract, data_dir, cleartype):
        # read data from file, if data doesn't exist, get_data() will generate data automaticly
        data_obj = self.__data
        data_obj.get_data(start_time, end_time, data_dir, db_table)
        price_df_list, instrument, datetime, volume = self.load_data_from_file(data_dir)
        res = self.get_focused_info(price_df_list, instrument, datetime, volume, coinType, two_contract)
        price_focused_list, instrument_focused, datetime_focused, volume_focused = res[0],res[1],res[2],res[3]
        # get position signal {1:'long',-1:'short','0':clear}
        args_list = [period, std_num, cleartype]
        position_list = self.generate_position_signal(price_focused_list, volume_focused, instrument_focused, args_list)
        [position_signal1, position_signal2]= position_list
        # get return for the two contract with their corresponding position signals
        price_focused = price_focused_list[3] #'close'
        return_contract1,BuySell_signal1 = self.get_return_plus_commission(price_focused.iloc[:, 0], position_signal1, period)
        return_contract2,BuySell_signal2 = self.get_return_plus_commission(price_focused.iloc[:, 1], position_signal2, period)
        BuySell_signal_list = [BuySell_signal1,BuySell_signal2]
        args_list = [0,0]
        return_list = self.get_return(price_focused_list, BuySell_signal_list, args_list)
        std_divide_price = self.get_std_divide_price(price_focused, period)

        buysell_info = pd.concat(
            [datetime_focused,
             price_focused.iloc[:, 0], position_signal1, BuySell_signal1, return_contract1, return_list[0],
             price_focused.iloc[:, 1], position_signal2, BuySell_signal2, return_contract2, return_list[1]], axis=1)
        buysell_info.columns = ['datetime',
                                'price1', 'position1', 'buysell1', 'return1','return1_1',
                                'price2', 'position2','buysell2', 'return2','return2_1']
        res_list = [position_signal1, position_signal2, return_contract1, return_contract2, datetime_focused,
                    std_divide_price, price_focused, buysell_info,return_list]
        return res_list

    # def mkdir(path):
    #     path = path.strip()
    #     isExists = os.path.exists(path)
    #     flag = 1
    #     if not isExists:
    #         # print path + 'create success'
    #         os.makedirs(path)
    #         flag = 0
    #     return flag

    def write_tradeInfo_to_file(self,buysell_dir, seq, buysell_info_df):
        file_name = '_'.join(seq) + '.csv'
        result_full_filename = os.path.join(buysell_dir, file_name)
        buysell_info_df.to_csv(result_full_filename)

    def write_result_to_file(self,result_dir, seq, results, window_period_list, std_num_list, result_type):
        seq.append(result_type)
        file_name = '_'.join(seq) + '.csv'
        # mkdir(result_dir)
        result_full_filename = os.path.join(result_dir, file_name)
        result_df = pd.DataFrame(results)
        result_df.index = window_period_list
        result_df.columns = std_num_list
        result_df.to_csv(result_full_filename)
        del seq[-1]

    def roll_test(self,start_time, end_time, db_table, price_type, window_period_list, coinType, std_num_list, two_contract,
                  data_dir, result_dir, root_buysell, cleartype):

        results_bp = np.zeros((len(window_period_list), len(std_num_list)))
        results_return_dd = np.zeros((len(window_period_list), len(std_num_list)))
        results_std_price = np.zeros((len(window_period_list), len(std_num_list)))
        results_turnover = np.zeros((len(window_period_list), len(std_num_list)))
        for i in range(len(window_period_list)):
            for j in range(len(std_num_list)):
                window_period = window_period_list[i]
                std_num = std_num_list[j]
                # basic routine
                res_list = self.run(start_time, end_time, db_table, price_type, window_period, coinType, std_num,
                               two_contract, data_dir, cleartype)

                position_signal1, position_signal2 = res_list[0], res_list[1]
                total_turnover = indicator.get_total_turnover(position_signal1, position_signal2)
                total_return_sum = np.sum(res_list[2] + res_list[3])

                contract_1and2_return = res_list[2] + res_list[3]
                datetime_focused = res_list[4]
                std_divide_price = res_list[5]
                price_focused = res_list[6]
                buysell_info_df = res_list[7]
                return_list = res_list[8]
                seq = [coinType, price_type, two_contract[0], two_contract[1], cleartype, str(window_period_list[i]),
                       str(std_num_list[j])]
                self.write_tradeInfo_to_file(root_buysell, seq, buysell_info_df)

                max_withdraw, dd_start_idx, dd_end_idx = indicator.get_max_drawdown(contract_1and2_return,
                                                                                    datetime_focused, start_time,
                                                                                    end_time)
                results_return_dd[i][j] = np.sum(contract_1and2_return) / abs(max_withdraw)
                results_std_price[i][j] = std_divide_price
                results_turnover[i][j] = total_turnover

                if total_turnover == 0:
                    ave_return_turnover = 0
                else:
                    ave_return_turnover = total_return_sum / 2 / total_turnover
                    # print('total return bp:%d'%np.sum(return_list[0].to_list()+return_list[1])/2/total_turnover)
                results_bp[i][j] = ave_return_turnover

        seq = [coinType, price_type, two_contract[0], two_contract[1], cleartype]

        self.write_result_to_file(result_dir, seq, results_bp, window_period_list, std_num_list, result_type='bp')
        self.write_result_to_file(result_dir, seq, results_return_dd, window_period_list, std_num_list,
                             result_type='return-dd')
        self.write_result_to_file(result_dir, seq, results_std_price, window_period_list, std_num_list,
                             result_type='std-price')
        self.write_result_to_file(result_dir, seq, results_turnover, window_period_list, std_num_list,
                             result_type='turnover')

    def average_allCoin(self,result_dir, average_dir, price_type, two_contract, cleartype, average_type):

        compile_str = '[a-z]+_%s_%s_%s_%s_%s.csv' % (
        price_type, two_contract[0], two_contract[1], cleartype, average_type)
        pattern = re.compile(compile_str)
        fileNames = os.listdir(result_dir)
        full_fileName = os.path.join(result_dir, fileNames[0])
        tmp_df = pd.read_csv(full_fileName)
        sum_array = np.zeros(tmp_df.shape)
        count = 0
        for file in fileNames:
            m = pattern.match(file)
            if m is not None:
                count += 1
                filename = m.group()
                full_fileName = os.path.join(result_dir, filename)
                df = pd.read_csv(full_fileName)
                ndarray = df.values
                sum_array += ndarray
        average_array = sum_array / count
        df = pd.DataFrame(average_array[:, 1:])
        df.index = tmp_df.iloc[:, 0]
        df.columns = tmp_df.columns[1:]
        seq = ['average', price_type, two_contract[0], two_contract[1], cleartype, average_type]
        file_name = '_'.join(seq) + '.csv'
        average_full_fileName = os.path.join(average_dir, file_name)
        df.to_csv(average_full_fileName)

if __name__ == '__main__':
    pass
    # start_time = '201806160000'  # 20180616000
    # end_time = '201807260000'  # 201806160030, 201807090000,201807170000
    # db_table = '1min'  # ['1min','5min']
    # price_type = 'close'  # []
    # minutes_in_uninTime = 24 * 60
    #
    # coinType_list = ['bch', 'btc', 'btg', 'eos', 'etc', 'eth', 'ltc', 'xrp']  # do not change this variable
    # # two_contract = ['week', 'quarter']#['week','nextweek', 'quarter']
    # three_contract = ['week', 'nextweek', 'quarter']
    # coin_list = ['btc']  # 'btc', 'bch','eth', 'etc', 'eos',    , 'ltc', 'xrp', 'btg'
    #
    # # cleartype_coefficient_dict = {'ceilfloor': 1, 'half_stdnum': 0.5, 'medium': 0}
    # cleartype = 'medium'  # [ceilfloor,medium,half-stdnum,threeQuarters-stdnum]
    # window_period_list = [5000]  #
    # std_num_list = [4]  # 2.5, 3, 3.25, 3.5, 3.75, 4
    #
    # father_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # root_data = os.path.join(father_dir, 'data')
    # root_result = os.path.join(father_dir, 'backtest')
    # root_average = os.path.join(father_dir, 'average')
    # root_buysellInfo = os.path.join(father_dir, 'buysell')
    #
    # fileName_item_list = [db_table, start_time, end_time]
    # db_start_end = '-'.join(fileName_item_list)
    # data_dir = os.path.join(root_data, db_start_end)
    #
    # now_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # result_dir = os.path.join(root_result, db_start_end, cleartype, now_str)
    # average_dir = os.path.join(root_average, db_start_end, cleartype, now_str)
    # buysell_dir = os.path.join(root_buysellInfo, db_start_end, cleartype, now_str)
    #
    # mkdir(result_dir)
    # mkdir(average_dir)
    # mkdir(buysell_dir)
    # commission_rate_buy = 0.0000
    # commission_rate_sell = commission_rate_buy

    # for coinType in coin_list:
    #     for i in range(len(three_contract)):
    #         if i == 2:
    #             time_start = time.clock()
    #             two_contract = [three_contract[i % 3], three_contract[(i + 1) % 3]]
    #             roll_test(start_time, end_time, db_table, price_type, window_period_list, coinType, std_num_list,
    #                       two_contract, data_dir, result_dir, buysell_dir, cleartype)
    #             time_end = time.clock()
    #             elapsed = time_end - time_start
    #             print('coinType:%s, two_contract:%s,%s complete! elapsed time is:%s' % (
    #             coinType, two_contract[0], two_contract[1], str(elapsed)))
    #
    # average_type_list = ['bp', 'return-dd', 'std-price', 'turnover']
    # for average_type in average_type_list:
    #     for i in range(len(three_contract)):
    #         if i == 2:
    #             time_start = time.clock()
    #             two_contract = [three_contract[i % 3], three_contract[(i + 1) % 3]]
    #             average_allCoin(result_dir, average_dir, price_type, two_contract, cleartype, average_type)
