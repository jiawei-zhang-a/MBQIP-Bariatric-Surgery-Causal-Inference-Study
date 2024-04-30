import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
import mbqip_read_run_dragon as mbqip_utils
import numpy as np
np.random.seed(0)

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("DragonNet")
    print(mbqip_utils.run_mbqip( PATH+"data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()

