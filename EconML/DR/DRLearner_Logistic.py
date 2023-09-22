
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk
from sklearn.ensemble import RandomForestRegressor
sys.path.append('../../utils')
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def main():
    PATH = "/scratch/jz4721/Observational-Study"
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")

    print("\nDeath")
    print("ForestDRLearner")
    est1 = ForestDRLearner(model_regression=lgb.LGBMRegressor(),
                            model_propensity=lgb.LGBMClassifier())  
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))

    print("\nintervention")  
    print("ForestDRLearner")
    est2 = ForestDRLearner(model_regression=lgb.LGBMRegressor(),
                            model_propensity=lgb.LGBMClassifier())      
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission")
    print("ForestDRLearner")
    est3 = ForestDRLearner(model_regression=lgb.LGBMRegressor(),
                            model_propensity=lgb.LGBMClassifier())   
    print(mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    print("ForestDRLearner")
    est4 = ForestDRLearner(model_regression=lgb.LGBMRegressor(),
                            model_propensity=lgb.LGBMClassifier())  
    print(mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation"))

if __name__ == '__main__':
    main()


