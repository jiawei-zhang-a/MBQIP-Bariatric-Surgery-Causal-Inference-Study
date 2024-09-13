import numpy as np
import glob
from dragonnet.dragonnet import DragonNet
from exdragonnet import EXdragonnet
import torch
import scipy.stats as stats

def load_and_format_covariates_mbqip(file_path):

    data = np.loadtxt(file_path, delimiter=',')

    contfeats = [1]
    binfeats = [i for i in range(2,49) if i not in contfeats]

    perm = contfeats + binfeats
    x = data[:, perm]
    return x

def load_all_other_crap(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    perm = 49
    t, y = data[:, 50], data[:,perm][:, None]
    
    return  t.reshape(-1,), y.reshape(-1,)

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
        return "Division by zero encountered in calculation"
    
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

def run_mbqip_risk(data_base_dir):
    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    ans = []
    for idx, simulation_file in enumerate(simulation_files):

        x = load_and_format_covariates_mbqip(simulation_file)
        t, y = load_all_other_crap(simulation_file)
        # Check if GPU is available
        model = DragonNet(x.shape[1])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        est = EXdragonnet(model)
        est.fit(y,t,X = x)
        cate_pred = est.ate(x)
        y_0 = y - t * cate_pred
        y_1 = y_0 + cate_pred


        #np.savez('result.npz', y_0=y_0, y_1=y_1)

        risk = get_risk_causal_relative_ratio(y_0, y_1)
        CI = get_risk_confidence_interval(y_0, y_1)

        ans.append([risk, CI])

    return ans
