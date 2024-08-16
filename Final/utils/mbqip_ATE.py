import numpy as np
import scipy.stats as stats

class ATEEstimator:
    def __init__(self, model):
        self.model = model

    def ate(self, X):
        y0_pred, y1_pred, t_pred, _ = self.model.predict(X)
        ate = np.mean(y1_pred - y0_pred)
        return ate

    def bootstrap_CI(self, X, n_bootstrap_samples=1000, alpha=0.05):
        bootstrap_ate_values = []
        for _ in range(n_bootstrap_samples):
            X_sample = np.random.choice(X, size=len(X), replace=True)
            ate_sample = self.ate(X_sample)
            bootstrap_ate_values.append(ate_sample)
        lower = np.percentile(bootstrap_ate_values, 100 * alpha / 2)
        upper = np.percentile(bootstrap_ate_values, 100 * (1 - alpha / 2))
        return lower, upper

    def ate_interval(self, X, n_bootstrap_samples=1000, alpha=0.05):
        ate = self.ate(X)
        lower, upper = self.bootstrap_CI(X, n_bootstrap_samples, alpha)
        return ate, lower, upper
