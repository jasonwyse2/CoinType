# coding=utf-8
# from pyalgotrade import strategy
# from pyalgotrade.bar import Frequency
# from pyalgotrade.barfeed.csvfeed import GenericBarFeed
import file_io
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import quick_transfer_data as qtd
import indicator
import os
import datetime
import time
import re

contract_map = {'week':0,'nextweek':1,'quarter':2}
cleartype_coefficient_dict = {'ceilfloor':1,'half-stdnum':0.5,'medium':0,'threeQuarters-stdnum':0.75}
def generate_signal(price_focused,instrument_focused,period,std_num,cleartype):

    two_contract_diff = price_focused.iloc[:, 0] - price_focused.iloc[:, 1]

    period_mean = two_contract_diff.rolling(period).mean()
    period_std = two_contract_diff.rolling(period).std()

    ceil_price = period_mean + period_std * std_num
    floor_price = period_mean - period_std * std_num
    cleartype_coefficient = cleartype_coefficient_dict[cleartype]
    clear_ceil = period_mean + period_std * std_num * cleartype_coefficient
    clear_floor = period_mean - period_std * std_num * cleartype_coefficient
    position_contract1 = pd.Series([0] * two_contract_diff.shape[0])
    position_contract2 = pd.Series([0] * two_contract_diff.shape[0])

    instrument_contract1, instrument_contract2 = instrument_focused.iloc[:,0],instrument_focused.iloc[:,1]

    for i in range(period, period_mean.shape[0]):
        idx = i
        if instrument_contract1[i] != instrument_contract1[i - 1]:
            position_contract1[i - 1] = 0
            position_contract2[i - 1] = 0
        else:
            if two_contract_diff[i] >= ceil_price[idx]:
                position_contract1[i] = -1
                position_contract2[i] = 1

            elif two_contract_diff[i] <= floor_price[idx]:
                position_contract1[i] = 1
                position_contract2[i] = -1
            else:
                position_contract1[i] = position_contract1[i - 1]
                position_contract2[i] = position_contract2[i - 1]

            if two_contract_diff[i]>=clear_ceil[idx] and two_contract_diff[i]<ceil_price[idx]:
                if position_contract1[i - 1] == 1:
                    position_contract1[i] = 0
                    position_contract2[i] = 0
            if two_contract_diff[i] > floor_price[idx] and two_contract_diff[i] <= clear_floor[idx]:
                if position_contract1[i - 1] == -1:
                    position_contract1[i] = 0
                    position_contract2[i] = 0
    return position_contract1,position_contract2
def get_std_divide_price(price_focused,period):
    two_contract_diff = price_focused.iloc[:, 0] - price_focused.iloc[:, 1]
    price1 = price_focused.iloc[:, 0]
    period_mean = two_contract_diff.rolling(period).mean()
    period_std = two_contract_diff.rolling(period).std()
    std_divide_price = np.mean(period_std / price1)
    return std_divide_price

def load_data_from_file(price_type,dest_dir,datetime_exist=0):
    if datetime_exist==1:
        fileName_price = price_type + '.csv'
        fullfileName_price = os.path.join(dest_dir,fileName_price)
        fileName_instrument = 'instrument.csv'
        fullfileName_instrument = os.path.join(dest_dir,fileName_instrument)
        price = pd.read_csv(fullfileName_price).iloc[:,1:]
        instrument = pd.read_csv(fullfileName_instrument).iloc[:,1:]
        datetime = price.iloc[:,0]
    elif datetime_exist==0:
        fileName_price = price_type + '.pkl'
        fullfileName_price = os.path.join(dest_dir, fileName_price)
        fileName_instrument = 'instrument.pkl'
        fullfileName_instrument = os.path.join(dest_dir, fileName_instrument)
        fullfileName_datetime = os.path.join(dest_dir, 'datetime.pkl')
        price = pd.DataFrame(file_io.getpkl(fullfileName_price))
        instrument = pd.DataFrame(file_io.getpkl(fullfileName_instrument))
        datetime = pd.DataFrame(file_io.getpkl(fullfileName_datetime))
    else:
        print('invalid datetime_exist:%s'%str(datetime_exist))
        exit()
    return price,instrument,datetime

# def show_figure(plt_obj_list,title_list,figure_num=1):
#     plt.figure(figure_num)
#     # plt.title(title)
#     for i in range(len(plt_obj_list)):
#         plt.plot(plt_obj_list[i], label=title_list[i])
#         plt.scatter(np.arange(0, plt_obj_list[i].shape[0]), plt_obj_list[i], marker='D')
#         plt.plot()
#     plt.grid()
#     plt.legend()
#
#
# def show_figure1(price_focused,period,std_num,figure_num=2):
#     plt.figure(figure_num)
#     two_contract_diff = price_focused.iloc[:, 0] - price_focused.iloc[:, 1]
#     period_mean = two_contract_diff.rolling(period).mean()
#     period_std = two_contract_diff.rolling(period).std()
#     ceil_price = period_mean + period_std * std_num
#     floor_price = period_mean - period_std * std_num
#     plt_obj_list = [two_contract_diff,ceil_price,floor_price,period_mean]
#     for i in range(len(plt_obj_list)):
#         plt.plot(plt_obj_list[i])
#         plt.scatter(np.arange(0, plt_obj_list[i].shape[0]), plt_obj_list[i], marker='D')
#         plt.plot()
#     plt.grid()
#     plt.legend()

def get_commission_signal(position_contract, period):
    BuySell_signal = pd.Series([0] * position_contract.shape[0])
    position_diff = np.diff(position_contract)
    BuySell_signal[2:] = position_diff[:-1]
    return BuySell_signal

def get_return_plus_commission(price_focused, position_contract, period):
    per_return_contract = np.diff(price_focused) / price_focused[:-1]
    BuySell_signal = get_commission_signal(position_contract, period)
    commission_contract = pd.Series([0] * price_focused.shape[0])
    commission_contract[BuySell_signal>0] = (BuySell_signal* commission_rate_buy)[BuySell_signal>0]*(-1)
    commission_contract[BuySell_signal < 0] = (BuySell_signal * commission_rate_sell)[BuySell_signal < 0]

    return_contract = pd.Series([0] * price_focused.shape[0])
    return_contract[period + 1:] = np.array(position_contract[period - 1:-2]) * np.array(per_return_contract[period:])

    return_contract_plus_commission = return_contract + commission_contract
    return return_contract_plus_commission

def get_focused_PriceInstrumentDatetime(price,instrument,datetime,coinType):
    column_offset = coinType_list.index(coinType) * 3
    contract1_idx = contract_map[two_contract[0]]
    contract2_idx = contract_map[two_contract[1]]

    price_focused_nan = price[[column_offset + contract1_idx, column_offset + contract2_idx]]
    instrument_focused_nan = instrument[[column_offset + contract1_idx, column_offset + contract2_idx]]
    # remove rows that contains 'nan'
    price_focused = price_focused_nan.dropna(axis=0, how='any')
    instrument_focused = instrument_focused_nan.dropna(axis=0, how='any')
    datetime_focused = datetime.iloc[price_focused.index.tolist(), :]
    # reset index from 0, because some index has been removed
    price_focused.index = range(len(price_focused.index))
    instrument_focused.index = range(len(instrument_focused.index))
    datetime_focused.index = range(len(datetime_focused.index))
    return price_focused,instrument_focused,datetime_focused

def run(start_time,end_time,db_table,price_type,period,coinType,std_num,two_contract,dest_dir, cleartype,datetime_exist=0):
    # read data from file, if data doesn't exist, get_data() will generate data automaticly
    qtd.get_data(start_time,end_time,dest_dir,db_table)
    price, instrument, datetime = load_data_from_file(price_type, dest_dir,datetime_exist)
    price_focused, instrument_focused, datetime_focused = get_focused_PriceInstrumentDatetime(price,instrument,datetime,coinType)
    # get position signal {1:'long',-1:'short','0':clear}
    position_contract1, position_contract2 = generate_signal(price_focused, instrument_focused, period, std_num,cleartype)

    return_contract1 = get_return_plus_commission(price_focused.iloc[:,0], position_contract1, period)
    return_contract2 = get_return_plus_commission(price_focused.iloc[:,1], position_contract2, period)

    std_divide_price = get_std_divide_price(price_focused,period)

    buysell_info = pd.concat([datetime_focused, price_focused.iloc[:,0], position_contract1, return_contract1, price_focused.iloc[:,1], position_contract2, return_contract2], axis=1)
    buysell_info.columns = ['datetime','price1','position1','return1','price2','position2','return2']
    # print(tmp)
    # show_figure1(price_focused, period, std_num)
    # show_figure([return_contract1,return_contract2], ['return_contract1','return_contract2'])
    # plt.show()
    # print ('total buy and sell number:%d'%sum(buy_signal1+sell_signal1+buy_signal2+sell_signal2))
    res_list = [position_contract1, position_contract2, return_contract1, return_contract2, datetime_focused, std_divide_price, price_focused, buysell_info]
    return res_list

def mkdir(path):
    path = path.strip()
    isExists = os.path.exists(path)
    flag =1
    if not isExists:
        # print path + 'create success'
        os.makedirs(path)
        flag = 0
    return flag
def cleardir(path):
    import shutil
    isExists = os.path.exists(path)
    if isExists:
        shutil.rmtree(path)
    os.makedirs(path)

def write_tradeInfo_to_file(buysell_dir, seq, buysell_info_df):
    file_name = '_'.join(seq) + '.csv'
    result_full_filename = os.path.join(buysell_dir, file_name)
    buysell_info_df.to_csv(result_full_filename)

def write_result_to_file(result_dir,seq, results,result_type):
    seq.append(result_type)
    file_name = '_'.join(seq) + '.csv'
    # mkdir(result_dir)
    result_full_filename = os.path.join(result_dir, file_name)
    result_df = pd.DataFrame(results)
    result_df.index = window_period_list
    result_df.columns = std_num_list
    result_df.to_csv(result_full_filename)
    del seq[-1]

def roll_test(start_time,end_time,db_table,price_type,window_period_list,coinType,std_num_list,two_contract,data_dir,result_dir,root_buysellInfo,cleartype):

    results_bp = np.zeros((len(window_period_list), len(std_num_list)))
    results_return_dd = np.zeros((len(window_period_list), len(std_num_list)))
    results_std_price = np.zeros((len(window_period_list), len(std_num_list)))
    results_turnover = np.zeros((len(window_period_list), len(std_num_list)))
    for i in range(len(window_period_list)):
        for j in range(len(std_num_list)):
            window_period = window_period_list[i]
            std_num = std_num_list[j]
            # basic routine
            res_list = run(start_time, end_time, db_table, price_type, window_period, coinType, std_num, two_contract, data_dir, cleartype)

            position_contract1, position_contract2 = res_list[0], res_list[1]
            total_turnover = indicator.get_total_turnover(position_contract1, position_contract2, window_period)
            total_return_sum = np.sum(res_list[2] + res_list[3])

            contract_1and2_return = res_list[2] + res_list[3]
            datetime_focused = res_list[4]
            std_divide_price = res_list[5]
            price_focused = res_list[6]
            buysell_info_df = res_list[7]
            seq = [coinType, price_type, two_contract[0], two_contract[1], cleartype, str(window_period_list[i]), str(std_num_list[j])]
            write_tradeInfo_to_file(buysell_dir, seq, buysell_info_df)
            max_withdraw = indicator.get_max_drawdown(contract_1and2_return, datetime_focused, start_time, end_time)
            results_return_dd[i][j] = np.sum(contract_1and2_return)/abs(max_withdraw)
            results_std_price[i][j] = std_divide_price
            results_turnover[i][j] = total_turnover

            if total_turnover == 0:
                ave_return_turnover = 0
            else:
                ave_return_turnover = total_return_sum / 2 / total_turnover
            results_bp[i][j] = ave_return_turnover

    seq = [coinType, price_type, two_contract[0], two_contract[1],cleartype]

    write_result_to_file(result_dir, seq, results_bp,result_type='bp')
    write_result_to_file(result_dir, seq, results_return_dd, result_type='return-dd')
    write_result_to_file(result_dir, seq, results_std_price, result_type='std-price')
    write_result_to_file(result_dir, seq, results_turnover, result_type='turnover')

def average_allCoin(result_dir, average_dir, price_type, two_contract, cleartype, average_type):

    compile_str = '[a-z]+_%s_%s_%s_%s_%s.csv' % (price_type, two_contract[0], two_contract[1], cleartype, average_type)

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
            full_fileName = os.path.join(result_dir,filename)
            df = pd.read_csv(full_fileName)
            ndarray = df.values
            sum_array += ndarray
    average_array = sum_array/count
    df = pd.DataFrame(average_array[:,1:])
    df.index = tmp_df.iloc[:,0]
    df.columns = tmp_df.columns[1:]
    seq = ['average' ,price_type, two_contract[0], two_contract[1],cleartype, average_type]
    file_name = '_'.join(seq) + '.csv'
    average_full_fileName = os.path.join(average_dir, file_name)
    df.to_csv(average_full_fileName)


if __name__ == '__main__':
    start_time = '201807220000'#20180616000
    end_time =   '201807291200'#201806160030, 201807090000,201807170000
    db_table = '1min' #['1min','5min']
    price_type = 'close'#[]

    coinType_list = ['bch', 'btc', 'btg', 'eos', 'etc', 'eth', 'ltc', 'xrp'] #do not change this variable
    # two_contract = ['week', 'quarter']#['week','nextweek', 'quarter']
    three_contract = ['week', 'nextweek','quarter']
    coin_list = ['btc', 'bch','eth', 'etc', 'eos']#  'btc', 'bch','eth', 'etc', 'eos',    , 'ltc', 'xrp', 'btg'

    # cleartype_coefficient_dict = {'ceilfloor': 1, 'half_stdnum': 0.5, 'medium': 0}
    cleartype = 'medium' #[ceilfloor,medium,half-stdnum,threeQuarters-stdnum]
    window_period_list = [2000,3000,4000,5000,7000] #
    std_num_list = [2.5, 3, 3.25, 3.5, 3.75, 4] #

    father_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    root_data = os.path.join(father_dir,'data')
    root_result = os.path.join(father_dir,'backtest')
    root_average = os.path.join(father_dir,'average')
    root_buysellInfo = os.path.join(father_dir,'buysell')

    fileName_item_list = [db_table, start_time, end_time]
    db_start_end = '-'.join(fileName_item_list)
    data_dir = os.path.join(root_data, db_start_end)

    now_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(root_result,db_start_end,cleartype,now_str)
    average_dir = os.path.join(root_average,db_start_end,cleartype,now_str)
    buysell_dir = os.path.join(root_buysellInfo,db_start_end,cleartype,now_str)

    mkdir(result_dir)
    mkdir(average_dir)
    mkdir(buysell_dir)
    commission_rate_buy = 0.0000
    commission_rate_sell = commission_rate_buy

    for coinType in coin_list:
        for i in range(len(three_contract)):
            if i==2:
                time_start = time.clock()
                two_contract = [three_contract[i%3],three_contract[(i+1)%3]]
                roll_test(start_time, end_time, db_table, price_type, window_period_list, coinType, std_num_list, two_contract, data_dir, result_dir, buysell_dir,cleartype)
                time_end = time.clock()
                elapsed = time_end - time_start
                print('coinType:%s, two_contract:%s,%s complete! elapsed time is:%s'%(coinType,two_contract[0],two_contract[1],str(elapsed)))

    average_type_list = ['bp', 'return-dd', 'std-price', 'turnover']
    for average_type in average_type_list:
        for i in range(len(three_contract)):
            if i == 2:
                time_start = time.clock()
                two_contract = [three_contract[i % 3], three_contract[(i + 1) % 3]]
                average_allCoin(result_dir, average_dir, price_type, two_contract, cleartype, average_type)
