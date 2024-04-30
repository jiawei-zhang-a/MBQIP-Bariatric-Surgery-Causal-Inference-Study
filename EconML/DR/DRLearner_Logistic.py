import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import mbqip_risk_rate as mbqip_risk
from sklearn.ensemble import RandomForestRegressor
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import numpy as np
np.random.seed(0)

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")
    print("ForestDRLearner with random forest")
    print("\nDeath")
    est1 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier())
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))
    
    print("\nintervention")
    est2 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier())
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission")
    est3 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier())
    print(mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    est4 = ForestDRLearner(model_regression=RandomForestRegressor(),
                            model_propensity=RandomForestClassifier())
    print(mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation"))


    print("ForestDRLearner with lgbm")
    print("\nDeath")
    print("ForestDRLearner + ")
    est1 = ForestDRLearner(model_regression=lgb.LGBMRegressor(verbosity = -1),
                            model_propensity=lgb.LGBMClassifier(verbosity = -1))  
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))

    print("\nintervention")  
    print("ForestDRLearner")
    est2 = ForestDRLearner(model_regression=lgb.LGBMRegressor(verbosity = -1),
                            model_propensity=lgb.LGBMClassifier(verbosity = -1))      
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission")
    print("ForestDRLearner")
    est3 = ForestDRLearner(model_regression=lgb.LGBMRegressor(verbosity = -1),
                            model_propensity=lgb.LGBMClassifier(verbosity = -1))   
    print(mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    print("ForestDRLearner")
    est4 = ForestDRLearner(model_regression=lgb.LGBMRegressor(verbosity = -1),
                            model_propensity=lgb.LGBMClassifier(verbosity = -1))  
    print(mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation"))

    print("ForestDRLearner with xgboost")
    print("\nDeath")
    est1 = ForestDRLearner(model_regression=xgb.XGBRegressor(),
                            model_propensity=xgb.XGBClassifier())
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))

    print("\nintervention")
    est2 = ForestDRLearner(model_regression=xgb.XGBRegressor(),
                            model_propensity=xgb.XGBClassifier())
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission")
    est3 = ForestDRLearner(model_regression=xgb.XGBRegressor(),
                            model_propensity=xgb.XGBClassifier())
    print(mbqip_risk.run_mbqip_risk(est3, PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    est4 = ForestDRLearner(model_regression=xgb.XGBRegressor(),
                            model_propensity=xgb.XGBClassifier())
    print(mbqip_risk.run_mbqip_risk(est4, PATH + "/data/mbqip/csv/reoperation"))



if __name__ == '__main__':
    main()


