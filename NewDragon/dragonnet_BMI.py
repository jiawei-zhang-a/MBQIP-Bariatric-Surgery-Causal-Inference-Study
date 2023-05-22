from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.ensemble import RandomForestClassifier
from exdragonnet import EXdragonnet
from dragonnet.dragonnet import DragonNet


def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("ForestDRLearner")
    est = EXdragonnet(DragonNet)
    print(mbqip_utils.run_mbqip(est, "/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()

