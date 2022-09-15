import pandas as pd
import numpy as np



"""
treatment - 50
death - y - 49
caseID - 0

"""
def convert_file(x):
    x = x.values
    x = x.astype(float)
    return x


def load_and_format_covariates_mbqip(file_path):

    data = np.loadtxt(file_path, delimiter=',')

    contfeats = [1]
    binfeats = [i for i in range(2,49) if i not in contfeats]

    perm = contfeats + binfeats
    x = data[:, perm]
    return x


def load_all_other_crap(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    contfeats = [1]
    binfeats = [i for i in range(2,49) if i not in contfeats]   
    perm = contfeats + binfeats 
    t, y = data[:, 50], data[:,perm][:, None]
    
    print(y)
    return t.reshape(-1, 1), y


def main():
    pass
    #load_all_other_crap("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/Death/mbqip_1.csv")
    #load_and_format_covariates_mbqip("/Users/jiaweizhang/med/dragonnet/dat/ihdp/csv/ihdp_npci_1.csv")

if __name__ == '__main__':
    main()
