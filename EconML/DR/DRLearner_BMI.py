from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.ensemble import RandomForestClassifier

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("ForestDRLearner")
    est = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)
    mbqip_utils.run_mbqip(est, "/scratch/jz4721/dragonnet/dat/mbqip/csv/BMI")
    
    print("ForestDRLearner with XGB")
    est = ForestDRLearner(model_regression=xgb.XGBRegressor(),
                            model_propensity=xgb.XGBClassifier(),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)
    mbqip_utils.run_mbqip(est, "/scratch/jz4721/dragonnet/dat/mbqip/csv/BMI")

    print("LinearDRLearner")
    est = LinearDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)
    mbqip_utils.run_mbqip(est, "/scratch/jz4721/dragonnet/dat/mbqip/csv/BMI")
    
if __name__ == '__main__':
    main()

