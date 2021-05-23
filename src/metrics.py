from . import loss
from sklearn import metrics

def rmse_scorer():
    return metrics.make_scorer(loss.rmse, greater_is_better=False)