import numpy as np
import scipy.stats as stats

class EXdragonnet:
    def __init__(self, model):
        self.model = model

    def ate(self, X):

        y0_pred, y1_pred, t_pred, _ = self.model.predict(X)

        y1_pred_np = y1_pred.detach().cpu().numpy()
        y0_pred_np = y0_pred.detach().cpu().numpy()

        ate = np.mean(y1_pred_np - y0_pred_np)

        return ate

    def fit(self, y, t, X):
        X_train = X
        y_train = y
        t_train = t
        self.model.fit(X_train, y_train, t_train)

    # assuming y0_pred and y1_pred are numpy arrays
    def ate_interval(self, X, alpha = 0.95):

        y0_pred, y1_pred, t_pred, _ = self.model.predict(X)

        y1_pred_np = y1_pred.detach().cpu().numpy()
        y0_pred_np = y0_pred.detach().cpu().numpy()

        # Calculate ITE
        ite = y1_pred_np - y0_pred_np

        # Calculate ATE
        ate = np.mean(ite)

        # Calculate standard error (SE)
        se = np.std(ite, ddof=1) / np.sqrt(len(ite))

        # Calculate the 95% confidence interval of ATE
        lower, upper = stats.norm.interval(alpha, loc=ate, scale=se)

        return lower, upper
