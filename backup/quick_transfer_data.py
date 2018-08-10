import pandas as pd
import pymysql
from datetime import datetime
import time
import numpy as np
import os
import strategy.tool


db_host_list = ['192.168.0.113','206.189.89.22','192.168.0.113','127.0.0.1'] #192.168.0.113, 206.189.89.22
db_port_list = [3306,5555,3306,3306] #3306 , 5555
db_user_list = ['root','linjuninvestment','root','root'] # root, linjuninvestment
db_pass_list = ['1qazxsw2','123456','1qazxsw2','1qazxsw2'] #1qazxsw2, 123456
db_name_list = ['okex','tradingdata','yg','okex'] #okcoin, tradingdata

contract_timeType = ['week', 'nextweek', 'quarter']
fourPrice_type_list = ['open', 'high', 'low', 'close']
coinType_list = ['bch', 'btc','btg','eos','etc','eth','ltc','xrp']  # 'bch', 'btc','btg','eos','etc','eth','ltc','xrp'
one_week_seconds = 7*24*3600
def get_unique_datetime(start_time,end_time,db_table):
    df = get_datetime_allUnique_df(start_time, end_time,db_table)
    unique_datetime = pd.unique(df.datetime)
    df = pd.DataFrame({'datetime': unique_datetime})
    return df

def login_MySQL(num):
    db_host = db_host_list[num]
    db_port = db_port_list[num]
    db_user = db_user_list[num]
    db_pass = db_pass_list[num]
    db_name = db_name_list[num]
    conn = pymysql.connect(host=db_host, port=db_port, user=db_user, passwd=db_pass, db=db_name)
    return  conn
def getTimeDiff(timeStra, timeStrb):
    if timeStra <= timeStrb:
        return 0
    ta = time.strptime(timeStra, "%Y%m%d%H%M")
    tb = time.strptime(timeStrb, "%Y%m%d%H%M")
    y, m, d, H, M, S = ta[0:6]
    dataTimea = datetime(y, m, d, H, M, S)
    y, m, d, H, M, S = tb[0:6]
    dataTimeb = datetime(y, m, d, H, M, S)
    secondsDiff = (dataTimea - dataTimeb).total_seconds()
    #
    # minutesDiff = round(secondsDiff / 60, 1)
    return secondsDiff
def get_datetime_allUnique_df(start_time, end_time,db_table):
    conn = login_MySQL(3)
    sql = 'SELECT open,high,close,low ,datetime, instrument,volume FROM okex.'+db_table+' WHERE DATETIME >=%d and datetime<=%d and volume>0 order by datetime asc' % (int(
        start_time), int(end_time))
    # read codes from database and save as dataframe
    df = pd.read_sql(sql, conn)
    return df
def get_columns_cointype_df(coinType,start_time,end_time,db_table):
    conn = login_MySQL(3)
    sql_datetime = 'SELECT open,high,close,low ,datetime, instrument,volume FROM okex.'+db_table+' WHERE DATETIME >=%d and datetime<=%d and volume>0' % (
        int(start_time), int(end_time))

    sql_filter = 'instrument LIKE "%s' % (coinType) + '%"'
    # concatenate sql string
    sql = sql_datetime + ' and ' + sql_filter + ' order by instrument'
    df = pd.read_sql(sql, conn)
    return df

def add_contract_price(df_data, contract_timeType, coinType, datetime_open, datetime_high, datetime_close, datetime_low):
    datetime_open = pd.merge(datetime_open, df_data[['open', 'datetime']], on='datetime', how='left')
    datetime_open.rename(columns={'open': coinType +'_' + contract_timeType}, inplace = True)

    datetime_high = pd.merge(datetime_high, df_data[['high', 'datetime']], on='datetime', how='left')
    datetime_high.rename(columns={'high': coinType + '_' + contract_timeType}, inplace = True)

    datetime_close = pd.merge(datetime_close, df_data[['close', 'datetime']], on='datetime', how='left')
    datetime_close.rename(columns={'close': coinType + '_' + contract_timeType}, inplace = True)

    datetime_low = pd.merge(datetime_low, df_data[['low', 'datetime']], on='datetime', how='left')
    datetime_low.rename(columns={'low': coinType + '_' + contract_timeType}, inplace = True)
    return [datetime_open,datetime_high,datetime_close,datetime_low]

def add_contract_instrument(df_data, contract_timeType, coinType, instrument):

    instrument = pd.merge(instrument, df_data[['instrument', 'datetime']], on='datetime', how='left')
    instrument.rename(columns={'instrument': coinType + '_' + contract_timeType}, inplace=True)
    return instrument

def add_contract_volume(df_data,contract_timeType,coinType,volume):
    volume = pd.merge(volume,df_data[['volume','datetime']],on='datetime',how='left')
    volume.rename(columns={'volume':coinType + '_' + contract_timeType},inplace = True)
    return volume

def quick_datetime_symbol(start_time, end_time,db_table):

    # read codes from database and save as dataframe
    datetime_open = get_unique_datetime(start_time,end_time,db_table)
    datetime_high = get_unique_datetime(start_time,end_time,db_table)
    datetime_close = get_unique_datetime(start_time,end_time,db_table)
    datetime_low = get_unique_datetime(start_time,end_time,db_table)
    instrument = get_unique_datetime(start_time, end_time, db_table)
    volume = get_unique_datetime(start_time,end_time,db_table)
    for coin_Type in coinType_list:
        df = get_columns_cointype_df(coin_Type, start_time, end_time,db_table)
        df['instrument_precise'] = df.instrument.map(lambda x: x[len(coin_Type) + 1:] + '1600')
        # transfer 'datetime' and instrument into date format
        df['datetime_str'] = df.datetime.map(lambda x: str(x))
        df['seconds_diff'] = df.apply(lambda x: (datetime.strptime(x['instrument_precise'], "%Y%m%d%H%M") - datetime.strptime(x['datetime_str'], "%Y%m%d%H%M")).total_seconds(), axis=1)

        df_week = df[(df['seconds_diff']>=0) & (df['seconds_diff']<one_week_seconds)]
        contract_timeType = 'week'
        [datetime_open, datetime_high, datetime_close, datetime_low]\
            =add_contract_price(df_week, contract_timeType, coin_Type, datetime_open, datetime_high, datetime_close, datetime_low)
        instrument = add_contract_instrument(df_week, contract_timeType, coin_Type, instrument)
        volume = add_contract_volume(df_week,contract_timeType,coin_Type,volume)

        df_nextweek = df[(df['seconds_diff'] > 0) & (df['seconds_diff'] >= one_week_seconds) & (df['seconds_diff'] <= 2 * one_week_seconds)]
        contract_timeType = 'nextweek'
        [datetime_open, datetime_high, datetime_close, datetime_low] \
            = add_contract_price(df_nextweek, contract_timeType, coin_Type, datetime_open, datetime_high, datetime_close, datetime_low)
        instrument = add_contract_instrument(df_nextweek, contract_timeType, coin_Type, instrument)
        volume = add_contract_volume(df_nextweek, contract_timeType, coin_Type, volume)

        df_quarter = df[(df['seconds_diff'] >= 0) & (df['seconds_diff'] > 2 * one_week_seconds)]
        contract_timeType = 'quarter'
        [datetime_open, datetime_high, datetime_close, datetime_low] \
            = add_contract_price(df_quarter, contract_timeType, coin_Type, datetime_open, datetime_high, datetime_close,
                                 datetime_low)
        instrument = add_contract_instrument(df_quarter, contract_timeType, coin_Type, instrument)
        volume = add_contract_volume(df_quarter, contract_timeType, coin_Type, volume)

    price_df_list = [datetime_open, datetime_high, datetime_close, datetime_low]
    return [price_df_list,instrument,volume]

def mkdir(path):
    path = path.strip()
    # path = path.rstrip("\\")
    isExists = os.path.exists(path)
    flag =1
    if not isExists:
        os.makedirs(path)
        flag = 0
    return flag


def write_df_to_file(df, project_path, type):

    df_values = df.values
    fileName = ''.join([type, '.pkl'])
    full_fileName = os.path.join(project_path, fileName)
    if not os.path.exists(full_fileName):
        strategy.tool.dumppkl(df_values, full_fileName)
    fileName = ''.join([type, '.csv'])
    full_fileName = os.path.join(project_path, fileName)
    if not os.path.exists(full_fileName):
        df.to_csv(full_fileName,index=False)

def get_data(start_time,end_time,dest_dir,db_table):
    time_start = time.clock()
    dest_includeTime_dir = dest_dir
    flag = mkdir(dest_includeTime_dir)
    p_list = ['open', 'high', 'close', 'low'] #the order can't be changed
    if flag==0 or not os.listdir(dest_includeTime_dir):
        df_list = quick_datetime_symbol(start_time, end_time,db_table)
        price_df_list = df_list[0]
        instrument = df_list[1]
        volume = df_list[2]
        datetime = instrument.iloc[:,0]

        write_df_to_file(datetime, dest_includeTime_dir, type='datetime')
        write_df_to_file(instrument, dest_includeTime_dir, type='instrument')
        write_df_to_file(volume, dest_includeTime_dir, type='volume')
        for i in range(len(p_list)):
            write_df_to_file(price_df_list[i], dest_includeTime_dir, type=p_list[i])
    time_end = time.clock()
    elapsed = time_end - time_start
    # print('load data elapsed time:%s'%str(elapsed))

if __name__ == '__main__':

    # start_time = '201806160000' #201806160000 #sldkddjkjkjkd
    # end_time   = '201807170000'#[201806230014,201806230018)201807090000
    # dest_dir = 'F:\\Projects\\CTA\\output_files\\'
    # db_table = '5min'
    # dest_includeTime_dir = dest_dir + start_time + '-' + end_time +'-'+db_table+ os.sep
    # [price_df_list,instrument_df_list] = quick_datetime_symbol(start_time, end_time,db_table)
    #
    # write_df_to_files(price_df_list, dest_includeTime_dir, type='price')
    # write_df_to_files(instrument_df_list, dest_includeTime_dir, type='instrument')
    # nd = fio.getpkl(dest_includeTime_dir + 'high_price.pkl')
    pass



