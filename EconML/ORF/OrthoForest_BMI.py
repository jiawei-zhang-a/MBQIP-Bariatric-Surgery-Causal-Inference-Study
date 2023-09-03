from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
from econml.orf import DROrthoForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


def main():
    est = DROrthoForest(
        model_Y=RandomForestRegressor(),
        model_T=RandomForestClassifier(),
        propensity_model=RandomForestClassifier(),
        model_Y_final=RandomForestRegressor(),
        propensity_model_final=RandomForestClassifier(),
    )
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()





