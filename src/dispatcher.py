import lightgbm as lgb

from sklearn import ensemble
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import svm

MODELS = {
    'RFR': ensemble.RandomForestRegressor(),
    'ETR': ensemble.ExtraTreesRegressor(),
    'RIDGE': linear_model.Ridge(alpha=0.5, solver='auto'),
    'LASSO': linear_model.Lasso(normalize=True),
    'OLS': linear_model.LinearRegression(normalize=True, fit_intercept=False),
    'SVM': svm.SVR(),
    'RIDGE_PIPE': pipeline.Pipeline([('poly', preprocessing.StandardScaler()),
                 ('fit', linear_model.Ridge(alpha=1))]),
    'LASSO_PIPE': pipeline.Pipeline([('poly', preprocessing.PolynomialFeatures()),
                 ('fit', linear_model.Lasso())]),
    'OLS_PIPE': pipeline.Pipeline([('poly', preprocessing.PolynomialFeatures()),
                 ('fit', linear_model.LinearRegression())]),
}

PARAMS = {
    'RIDGE_PIPE': {'fit__alpha':[550, 580, 600, 620, 650]},
    'LASSO_PIPE': {'fit__alpha':[0.005, 0.02, 0.03, 0.05, 0.06]}
}