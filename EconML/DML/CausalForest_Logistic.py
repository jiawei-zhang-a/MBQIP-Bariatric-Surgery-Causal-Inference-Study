
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk
import os
import mbqip_bootstrap as mbqip_bootstrap


def main():

    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
    else:
        print("Please add the job number like this\nEx.python CausalForest_logistic.py 1")
        exit()

    if os.path.exists("Result") == False:
        os.mkdir("Result")

    est = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        random_state=123)   
    est = CausalForestDML()

    PATH = "/scratch/jz4721/Observational-Study"
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")
    print("\nBMI")
    mbqip_bootstrap.bootstrap(est, PATH + "/data/mbqip/csv/Death", "Death", task_id)

    print("\nintervention")    
    mbqip_bootstrap.bootstrap(est, PATH + "/data/mbqip/csv/intervention", "intervention", task_id)

    print("\nreadmission") 
    mbqip_bootstrap.bootstrap(est, PATH + "/data/mbqip/csv/readmission", "readmission", task_id)

    print("\nreoperation") 
    mbqip_bootstrap.bootstrap(est, PATH + "/data/mbqip/csv/reoperation", "reoperation", task_id)
    
if __name__ == '__main__':
    main()


