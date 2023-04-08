
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk
from econml.orf import DROrthoForest



def main():
    PATH = "/scratch/jz4721/Observational-Study"

    PATH = "/Users/jiaweizhang/med/Observational-Study/"
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")

    print("\nDeath")
    est1 = DROrthoForest(model_Y = RandomForestRegressor())
   
    mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death")

    print("\nintervention")    
    est2 = DROrthoForest(model_Y = RandomForestRegressor())
    mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention")

    print("\nreadmission")
    est3 = DROrthoForest(model_Y = RandomForestRegressor())
    mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission")

    print("\nreoperation")
    est4 = DROrthoForest(model_Y = RandomForestRegressor())
    mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation")
    
if __name__ == '__main__':
    main()


