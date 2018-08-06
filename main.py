from tool import *
import os
import datetime,time
from CTA import CTA
if __name__ == '__main__':
    end_time =   '201807260000'#201806160030, 201807090000,201807170000
    coin_list = ['btc','bch']#  'btc', 'bch','eth', 'etc', 'eos',
    cleartype = 'medium' #[ceilfloor,medium,half-stdnum,threeQuarters-stdnum]
    window_period_list = [5000] #
    std_num_list = [3,3.5] #2.5, 3, 3.25, 3.5, 3.75, 4
    cta = CTA()
    # initialize variable values
    cta.project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    cta.buy_commission_rate = 0.0000
    cta.sell_commission_rate = cta.buy_commission_rate
    cta.end_time = '201807260000'
    cta.coin_list = coin_list
    cta.cleartype = cleartype
    cta.window_period_list = window_period_list
    cta.std_num_list = std_num_list
    # start the program
    cta.start()




