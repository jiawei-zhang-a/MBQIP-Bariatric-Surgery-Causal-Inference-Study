from econml.metalearners import TLearner, SLearner, DomainAdaptationLearner, XLearner
from sklearn.ensemble import RandomForestRegressor
import sys
import lightgbm as lgb
import xgboost as xgb
import numpy as np
np.random.seed(0)

sys.path.append('../../utils')
import mbqip_risk_rate_metalearners as mbqip_risk  # Replace with your actual module and functions

PATH = "../../"


def main():
    PATH = "/scratch/jz4721/Observational-Study"
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")
    
    run_metalearner_for_each_outcome_and_model(TLearner, RandomForestRegressor(), "Random Forest")
    run_metalearner_for_each_outcome_and_model(SLearner, RandomForestRegressor(), "Random Forest")
    run_metalearner_for_each_outcome_and_model(DomainAdaptationLearner, RandomForestRegressor(), "Random Forest")
    run_metalearner_for_each_outcome_and_model(XLearner, RandomForestRegressor(), "Random Forest")

    run_metalearner_for_each_outcome_and_model(TLearner, lgb.LGBMRegressor(verbosity = -1), "LightGBM")
    run_metalearner_for_each_outcome_and_model(SLearner, lgb.LGBMRegressor(verbosity = -1), "LightGBM")
    run_metalearner_for_each_outcome_and_model(DomainAdaptationLearner, lgb.LGBMRegressor(verbosity = -1), "LightGBM")
    run_metalearner_for_each_outcome_and_model(XLearner, lgb.LGBMRegressor(verbosity = -1), "LightGBM")

    run_metalearner_for_each_outcome_and_model(TLearner, xgb.XGBRegressor(), "XGBoost")
    run_metalearner_for_each_outcome_and_model(SLearner, xgb.XGBRegressor(), "XGBoost")
    run_metalearner_for_each_outcome_and_model(DomainAdaptationLearner, xgb.XGBRegressor(), "XGBoost")
    run_metalearner_for_each_outcome_and_model(XLearner, xgb.XGBRegressor(), "XGBoost")

def run_metalearner_for_each_outcome_and_model(MetaLearner, model, model_name):
    outcomes = ["Death", "intervention", "readmission", "reoperation"]
    for outcome in outcomes:
        print(f"\n{MetaLearner.__name__} with {model_name} for {outcome}")
        est = MetaLearner(models=model) if MetaLearner != SLearner else MetaLearner(overall_model=model)
        print(mbqip_risk.run_mbqip_risk(est, f"{PATH}/data/mbqip/csv/{outcome}"))

if __name__ == '__main__':
    main()
