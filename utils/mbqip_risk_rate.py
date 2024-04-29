import numpy as np
import glob 
import mbqip_read_run
import os
import scipy.stats as stats

def get_counterfactual_outcome (est,X, Y, T):

    # Get the counterfactual outcome for each sample
    cate_pred = est.effect(X)
    y_0 = Y - T * cate_pred
    y_1 = y_0 + cate_pred

    return y_0, y_1

def get_risk_causal_relative_ratio(y_0, y_1):

    # Probability of the event occurring in the untreated group
    prob_event_y0 = np.mean(y_0)

    # Probability of the event occurring in the treated group
    prob_event_y1 = np.mean(y_1)

    # Relative risk ratio
    relative_risk_ratio = prob_event_y1 / prob_event_y0

    return relative_risk_ratio

def get_risk_confidence_interval(y_0, y_1, alpha=0.05):
    # Ensure numpy arrays
    y_0, y_1 = np.array(y_0), np.array(y_1)
    
    # Number of events and total observations in each group
    x0, n0 = np.sum(y_0), len(y_0)
    x1, n1 = np.sum(y_1), len(y_1)
    
    # Check for division by zero
    if x0 == 0 or x1 == 0 or n0 - x0 == 0 or n1 - x1 == 0:
        raise ValueError("Division by zero encountered in calculation")
    
    # Probability of the event occurring in each group
    p0, p1 = x0 / n0, x1 / n1
    
    # Relative risk ratio
    relative_risk_ratio = get_risk_causal_relative_ratio(y_0, y_1)
    
    # Standard error calculation
    SE = np.sqrt((1 / x1) + (1 / x0) - (1 / (n1)) - (1 / (n0)))
    
    # Confidence interval calculation
    z_value = stats.norm.ppf(1 - alpha / 2)
    log_CI = [np.log(relative_risk_ratio) + -1 * z_value * SE, np.log(relative_risk_ratio) + z_value * SE]
    log_CI = np.array(log_CI)
    CI = np.exp(log_CI)
    
    return CI


def run_mbqip_risk(est, data_base_dir):

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    ans = []
    for idx, simulation_file in enumerate(simulation_files):

        x = mbqip_read_run.load_and_format_covariates_mbqip(simulation_file)
        t, y = mbqip_read_run.load_all_other_crap(simulation_file)

        est.fit(y,t,X = x,W=None)

        y_0, y_1 = get_counterfactual_outcome(est, x, y, t)
        #risk = get_risk_causal_relative_ratio(y_0, y_1)

        np.savez('result.npz', y_0=y_0, y_1=y_1)


        ans.append("Finished simulation")
    return ans

