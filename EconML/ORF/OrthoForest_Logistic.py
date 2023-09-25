
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk
from econml.orf import DROrthoForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def main():
    PATH = "/scratch/jz4721/Observational-Study"

    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")

    print("OrthoForest with rnadom forest")
    print("\nDeath")
    est1 = DROrthoForest(
        model_Y=RandomForestRegressor(),
        model_Y_final=RandomForestRegressor(),
    )
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))

    print("\nintervention")    
    est2 = DROrthoForest(
        model_Y=RandomForestRegressor(),
        model_Y_final=RandomForestRegressor(),
    )
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission")
    est3 = DROrthoForest(
        model_Y=RandomForestRegressor(),
        model_Y_final=RandomForestRegressor(),
    )
    print(mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    est4 = DROrthoForest(
        model_Y=RandomForestRegressor(),
        model_Y_final=RandomForestRegressor(),
    )
    print(mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation"))
    
    print("\nOrthoForest with lightgbm")
    print("\nDeath")
    est1 = DROrthoForest(
        model_Y=lgb.LGBMRegressor(),
        model_Y_final=lgb.LGBMRegressor(),
    )
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))

    print("\nintervention")
    est2 = DROrthoForest(
        model_Y=lgb.LGBMRegressor(),
        model_Y_final=lgb.LGBMRegressor(),
    )

    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission")
    est3 = DROrthoForest(
        model_Y=lgb.LGBMRegressor(),
        model_Y_final=lgb.LGBMRegressor(),
    )

    print(mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    est4 = DROrthoForest(
        model_Y=lgb.LGBMRegressor(),
        model_Y_final=lgb.LGBMRegressor(),
    )

    print(mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation"))


if __name__ == '__main__':
    main()


