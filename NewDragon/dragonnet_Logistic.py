import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
import mbqip_risk_rate_dragon as mbqip_risk
import numpy as np
np.random.seed(0)


def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")

    """print("\nDeath")
    print("DragonNet")
    print(mbqip_risk.run_mbqip_risk(PATH + "/data/mbqip/csv/Death"))

    print("\nintervention")  
    print("DragonNet")
    print(mbqip_risk.run_mbqip_risk(PATH + "/data/mbqip/csv/intervention"))"""

    print("\nreadmission")
    print("DragonNet")
    print(mbqip_risk.run_mbqip_risk(PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    print("DragonNet")
    print(mbqip_risk.run_mbqip_risk(PATH + "/data/mbqip/csv/reoperation"))

if __name__ == '__main__':
    main()


