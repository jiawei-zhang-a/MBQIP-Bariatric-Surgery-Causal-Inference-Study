from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("ForestDRLearner")
    est1 = ForestDRLearner(model_regression=lgb.LGBMRegressor(),
                            model_propensity=lgb.LGBMClassifier())  
    print(mbqip_utils.run_mbqip(est1, "/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()

