from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
from econml.orf import DROrthoForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


def main():
    PATH = "/scratch/jz4721/Observational-Study"
    est = DROrthoForest(propensity_model = RandomForestClassifier(),
                            model_Y = RandomForestRegressor(),
                        propensity_model_final = RandomForestClassifier(),
                         model_Y_final = RandomForestRegressor()
                        )
    print(mbqip_utils.run_mbqip(est,PATH))

if __name__ == '__main__':
    main()





