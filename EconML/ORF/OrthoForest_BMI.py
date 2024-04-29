from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
from econml.orf import DROrthoForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def main():
    est = DROrthoForest(
        model_Y=RandomForestRegressor(),
        model_Y_final=RandomForestRegressor(),
    )
    print("OrthoForest with random forest regressor for Y and Y_final ")


    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

    est = DROrthoForest(
        model_Y=RandomForestRegressor(),
    )
    print("OrthoForest with random forest regressor only for Y")
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

    print("OrthoForest with lightgbm")
    est = DROrthoForest(
        model_Y=lgb.LGBMRegressor(),
        model_Y_final=lgb.LGBMRegressor(),
    )
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()





