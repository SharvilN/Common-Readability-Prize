import numpy as np
from sklearn import metrics

def rmse(targets, preds):
    return np.sqrt(metrics.mean_squared_error(targets, preds))