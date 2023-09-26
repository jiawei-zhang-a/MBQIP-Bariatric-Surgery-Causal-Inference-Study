from econml.metalearners import TLearner, SLearner, DomainAdaptationLearner, XLearner
from sklearn.ensemble import RandomForestRegressor
import sys
import lightgbm as lgb
import xgboost as xgb

sys.path.append('../../utils')
import mbqip_read_run_metalearners as mbqip_utils  # Replace with your actual module and functions

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")
    
    run_metalearner_with_model(TLearner, RandomForestRegressor(), "TLearner with Random Forest")
    run_metalearner_with_model(SLearner, RandomForestRegressor(), "SLearner with Random Forest")
    run_metalearner_with_model(DomainAdaptationLearner, RandomForestRegressor(), "Domain Adaptation Learner with Random Forest")
    run_metalearner_with_model(XLearner, RandomForestRegressor(), "XLearner with Random Forest")

    run_metalearner_with_model(TLearner, lgb.LGBMRegressor(), "TLearner with LightGBM")
    run_metalearner_with_model(SLearner, lgb.LGBMRegressor(), "SLearner with LightGBM")
    run_metalearner_with_model(DomainAdaptationLearner, lgb.LGBMRegressor(), "Domain Adaptation Learner with LightGBM")
    run_metalearner_with_model(XLearner, lgb.LGBMRegressor(), "XLearner with LightGBM")

    run_metalearner_with_model(TLearner, xgb.XGBRegressor(), "TLearner with XGBoost")
    run_metalearner_with_model(SLearner, xgb.XGBRegressor(), "SLearner with XGBoost")
    run_metalearner_with_model(DomainAdaptationLearner, xgb.XGBRegressor(), "Domain Adaptation Learner with XGBoost")
    run_metalearner_with_model(XLearner, xgb.XGBRegressor(), "XLearner with XGBoost")

def run_metalearner_with_model(MetaLearner, model, model_name):
    print(f"{model_name}")
    est = MetaLearner(models=model) if MetaLearner != SLearner else MetaLearner(overall_model=model)
    print(mbqip_utils.run_mbqip(est, "/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()
