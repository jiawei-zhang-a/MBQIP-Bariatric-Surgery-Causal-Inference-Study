
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk


def main():
    PATH = "/scratch/jz4721/Observational-Study"
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")

    print("\nDeath")
    est1 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        random_state=123)    
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))

    print("\nintervention")    
    est2 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        random_state=123)    
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission")
    est3 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        random_state=123)    
    print(mbqip_risk.run_mbqip_risk(est3,PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation")
    est4 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        random_state=123)    
    print(mbqip_risk.run_mbqip_risk(est4,PATH + "/data/mbqip/csv/reoperation"))
    
if __name__ == '__main__':
    main()


