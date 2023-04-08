from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.ensemble import RandomForestClassifier

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("ForestDRLearner")
    est1 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)
    print(mbqip_utils.run_mbqip(est1, "/scratch/jz4721/Observational-Study/dat/mbqip/csv/BMI"))

    print("LinearDRLearner")
    est2 = LinearDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10))
    print(mbqip_utils.run_mbqip(est2, "/scratch/jz4721/dragonnet/dat/mbqip/csv/BMI"))
    
if __name__ == '__main__':
    main()

