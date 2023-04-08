import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
import numpy as np
from econml.orf import DROrthoForest
from sklearn.ensemble import RandomForestRegressor

def main():
    PATH = "/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"
    est = DROrthoForest(model_Y = RandomForestRegressor())
    print(mbqip_utils.run_mbqip(est,PATH))

if __name__ == '__main__':
    main()