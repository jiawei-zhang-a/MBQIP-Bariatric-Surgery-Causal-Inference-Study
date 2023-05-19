from sklearn.base import clone
import os
from mbqip_risk_rate import run_mbqip_risk
import numpy as np
import copy

def bootstrap(est, PATH, type, task_id):
    estimator_copy = copy.deepcopy(est)
    if os.path.exists("Result") == False:
        os.mkdir("Result")
    if os.path.exists("Result/" + type) == False:
        os.mkdir("Result/" + type)
    result = run_mbqip_risk(estimator_copy, PATH)
    result = np.array(result)
    np.save('Result/%s/risk_%d.npy' % (type, task_id), result)




