import numpy as np
import glob 
import mbqip_read_run
from dragonnet.dragonnet import DragonNet
from exdragonnet import EXdragonnet
import torch


def get_counterfactual_outcome (est,X, Y, T):

    # Get the counterfactual outcome for each sample
    cate_pred = est.effect(X)
    y_0 = Y - T * cate_pred
    y_1 = y_0 + cate_pred

    return y_0, y_1

def get_risk_causal_ratio(y_0, y_1, T, event_threshold = 0.5):

    # Event indicator for the untreated group
    event_y0 = (y_0 >= event_threshold).astype(int)
    event_y1 = (y_1 >= event_threshold).astype(int)

    # Probability of the event occurring in the untreated group
    prob_event_y0 = np.mean(event_y0[T.flatten() == 0])

    # Probability of the event occurring in the treated group
    prob_event_y1 = np.mean(event_y1[T.flatten() == 1])

    # Causal risk ratio
    causal_risk_ratio = prob_event_y1 / prob_event_y0

    return causal_risk_ratio

def get_risk_relative_ratio(y_0, y_1):

    # Probability of the event occurring in the untreated group
    prob_event_y0 = np.mean(y_0)

    # Probability of the event occurring in the treated group
    prob_event_y1 = np.mean(y_1)

    # Relative risk ratio
    relative_risk_ratio = prob_event_y1 / prob_event_y0

    return relative_risk_ratio

def run_mbqip_risk(data_base_dir):

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    ans = []
    for idx, simulation_file in enumerate(simulation_files):

        x = mbqip_read_run.load_and_format_covariates_mbqip(simulation_file)
        t, y = mbqip_read_run.load_all_other_crap(simulation_file)

        model = DragonNet(x.shape[1])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        est = EXdragonnet(model)
        est.fit(y,t,X = x)
        ans.append((est.ate(x),est.ate_interval(x)))

        y_0, y_1 = get_counterfactual_outcome(est, x, y, t)
        risk = get_risk_relative_ratio(y_0, y_1)

        ans.append(risk)

    return ans

