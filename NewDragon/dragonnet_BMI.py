from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../utils')
import mbqip_read_run_dragon as mbqip_utils
from exdragonnet import EXdragonnet
from dragonnet.dragonnet import DragonNet


def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("DragonNet")
    print(mbqip_utils.run_mbqip( "/scratch/jz4721/SCI/data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()

