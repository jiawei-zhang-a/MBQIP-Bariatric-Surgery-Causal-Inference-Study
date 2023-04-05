import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
import numpy as np
from econml.orf import DROrthoForest
from sklearn.ensemble import RandomForestRegressor

def main():
    PATH = "/scratch/jz4721/dragonnet/"
    est = DROrthoForest(model_Y = RandomForestRegressor())
    ans = mbqip_utils.run_mbqip(PATH + "dat/mbqip/csv/BMI",est)
    print(ans)

if __name__ == '__main__':
    main()