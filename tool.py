import pickle as pk
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
def dumppkl(data, path):
    datafile = open(path, 'w')
    pk.dump(data, datafile)
    datafile.close()
    return


def getpkl(path):
    datafile = open(path, 'r')
    ret = pk.load(datafile)
    datafile.close()
    return ret


def write_df_to_file(df, dest_dir, filename):

    fileName = ''.join([filename, '.csv'])
    full_fileName = os.path.join(dest_dir, fileName)
    if not os.path.exists(full_fileName):
        df.to_csv(full_fileName, index=False)

def append_df_to_file(df,dest_dir,filename):
    fileName_tmp = ''.join([filename+'tmp', '.csv'])
    full_fileName_tmp = os.path.join(dest_dir, fileName_tmp)
    df.to_csv(full_fileName_tmp, index=False, header = False)
    fileName = ''.join([filename, '.csv'])
    full_fileName = os.path.join(dest_dir, fileName)
    with open(full_fileName, 'ab') as f:
        f.write(open(full_fileName_tmp, 'rb').read())

    if os.path.exists(full_fileName_tmp):
        os.remove(full_fileName_tmp)

def mkdir(path):
    path = path.strip()
    # path = path.rstrip("\\")
    isExists = os.path.exists(path)
    flag = 1
    if not isExists:
        os.makedirs(path)
        flag = 0
    return flag

def currentTime_forward_delta(currentTime, min_deltaTime):
    time_format = '%Y%m%d%H%M'
    curr = datetime.strptime(currentTime, time_format)
    forward = (curr + relativedelta(minutes=+min_deltaTime))
    currTime = forward.strftime(time_format)
    return currTime