from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from econml.metalearners import XLearner,TLearner,SLearner,DomainAdaptationLearner
import xgboost as xgb
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("SLearner")
    est = SLearner(overall_model=RandomForestRegressor())
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/dat/mbqip/csv/BMI"))

    print("TLearner")
    est = TLearner(models=RandomForestRegressor())
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/dat/mbqip/csv/BMI"))

    print("XLearner")
    est = XLearner(models=RandomForestRegressor(), cate_models=RandomForestRegressor())
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/dat/mbqip/csv/BMI"))

    print("DomainAdaptationLearner")
    est = DomainAdaptationLearner(models=RandomForestRegressor(), final_model=RandomForestRegressor())
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/dat/mbqip/csv/BMI"))

    
if __name__ == '__main__':
    main()