from CTA import *
if __name__ == '__main__':
    start_time = '201806160000'#20180616000
    end_time =   '201807260000'#201806160030, 201807090000,201807170000
    db_table = '1min' #['1min','5min']
    price_type = 'close'#[]
    minutes_in_uninTime = 24 * 60

    coinType_list = ['bch', 'btc', 'btg', 'eos', 'etc', 'eth', 'ltc', 'xrp'] #do not change this variable
    # two_contract = ['week', 'quarter']#['week','nextweek', 'quarter']
    three_contract = ['week', 'nextweek','quarter']
    coin_list = ['btc']#  'btc', 'bch','eth', 'etc', 'eos',    , 'ltc', 'xrp', 'btg'

    # cleartype_coefficient_dict = {'ceilfloor': 1, 'half_stdnum': 0.5, 'medium': 0}
    cleartype = 'medium' #[ceilfloor,medium,half-stdnum,threeQuarters-stdnum]
    window_period_list = [5000] #
    std_num_list = [4] #2.5, 3, 3.25, 3.5, 3.75, 4

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
