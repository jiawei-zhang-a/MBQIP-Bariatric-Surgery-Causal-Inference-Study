
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk
from sklearn.ensemble import RandomForestRegressor
sys.path.append('../../utils')
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.ensemble import RandomForestClassifier

def main():
    PATH = "/scratch/jz4721/Observational-Study"
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")

    print("\nDeath")
    print("ForestDRLearner")
    est1 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)  
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))

    print("LinearDRLearner")
    est1 = LinearDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))


    print("\nintervention")  
    print("ForestDRLearner")
    est2 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)      
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("LinearDRLearner")
    est2 = LinearDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10))
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission")
    print("ForestDRLearner")
    est3 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)  
    print(mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission"))

    print("LinearDRLearner")
    est3 = LinearDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10))
    print(mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    print("ForestDRLearner")
    est4 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10),
                            cv=3,
                            n_estimators=4000,
                            min_samples_leaf=10,
                            verbose=0,
                            min_weight_fraction_leaf=.005)  
    print(mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation"))

    print("LinearDRLearner")
    est4 = LinearDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier(min_samples_leaf=10))
    print(mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation"))
    
if __name__ == '__main__':
    main()


