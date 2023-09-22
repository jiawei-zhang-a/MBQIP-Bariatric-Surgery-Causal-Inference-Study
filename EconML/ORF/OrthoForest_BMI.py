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
        propensity_model=RandomForestClassifier(),
        model_Y_final=RandomForestRegressor(),
        propensity_model_final=RandomForestClassifier(),
    )
    print("OrthoForest with random forest")
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

    print("OrthoForest with lightgbm")
    est = DROrthoForest(
        model_Y=lgb.LGBMRegressor(),
        propensity_model=lgb.LGBMClassifier(),
        model_Y_final=lgb.LGBMRegressor(),
        propensity_model_final=lgb.LGBMClassifier(),
    )
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()





