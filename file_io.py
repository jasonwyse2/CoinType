import pickle as pk
def dumppkl(data,path):
    datafile = open(path,'w')
    pk.dump(data, datafile)
    datafile.close()
    return

def getpkl(path):
    datafile = open(path, 'r')
    ret = pk.load(datafile)
    datafile.close()
    return ret