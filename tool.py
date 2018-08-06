import pickle as pk
import os
import datetime
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


def write_df_to_file(df, project_path, filename):
    df_values = df.values
    fileName = ''.join([filename, '.pkl'])
    full_fileName = os.path.join(project_path, fileName)
    if not os.path.exists(full_fileName):
        dumppkl(df_values, full_fileName)
    fileName = ''.join([filename, '.csv'])
    full_fileName = os.path.join(project_path, fileName)
    if not os.path.exists(full_fileName):
        df.to_csv(full_fileName, index=False)

def mkdir(path):
    path = path.strip()
    # path = path.rstrip("\\")
    isExists = os.path.exists(path)
    flag = 1
    if not isExists:
        os.makedirs(path)
        flag = 0
    return flag

def currentTime_forward_deltaMin(currentTime, deltaTime):
    time_format = "%Y%m%d%H%M"
    curr = datetime.strptime(currentTime, time_format)
    forward = (curr + relativedelta(months=+deltaTime))
    cutTime = forward.strftime(time_format)
    return cutTime

def currentTime_forward_delta(currentTime, min_deltaTime):
    time_format = '%Y%m%d%H%M'
    curr = datetime.strptime(currentTime, time_format)
    forward = (curr + relativedelta(minutes=+min_deltaTime))
    currTime = forward.strftime(time_format)
    return currTime