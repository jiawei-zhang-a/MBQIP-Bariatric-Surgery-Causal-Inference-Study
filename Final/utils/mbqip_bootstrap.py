from sklearn.base import clone
import os
from mbqip_risk_rate import run_mbqip_risk
import numpy as np
import copy
from sklearn.utils import resample


def bootstrap(est, PATH, type, task_id):
    #estimator_copy = copy.deepcopy(est)
    if os.path.exists("Result") == False:
        os.mkdir("Result")
    if os.path.exists("Result/" + type) == False:
        os.mkdir("Result/" + type)
    result = run_mbqip_risk(est, PATH)
    result = np.array(result)
    np.save('Result/%s/risk_%d.npy' % (type, task_id), result)

def bootstrap_analysis(PATH, type, n_bootstrap=1000, alpha=0.05):
    bootstrap_results = []
    bootstrap_conf_intervals = []

    # list all .npy files in the Result folder
    file_list = [f for f in os.listdir(PATH + "/" + type) if f.endswith('.npy')]

    # perform bootstrap analysis for each file
    for file in file_list:
        # load the saved numpy array
        data = np.load(PATH + "/" + type + "/" + file)

        # transpose the data so that each column is treated as an independent sample
        data = data.T

        # perform bootstrap analysis for each column
        for i in range(data.shape[0]):
            bootstrap_samples = []
            for _ in range(n_bootstrap):
                boot_sample = resample(data[i])  # by default, resample samples with replacement
                bootstrap_samples.append(boot_sample)

            # save the bootstrap samples for each column in the file
            file_bootstrap_results = np.array(bootstrap_samples)
            bootstrap_results.append(file_bootstrap_results)

            # calculate confidence interval for each column
            lower = np.percentile(file_bootstrap_results, 100 * alpha / 2.)
            upper = np.percentile(file_bootstrap_results, 100 * (1 - alpha / 2.))
            bootstrap_conf_intervals.append((lower, upper))

    return bootstrap_results, bootstrap_conf_intervals
