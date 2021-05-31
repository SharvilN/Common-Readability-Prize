import os
import joblib
import argparse
import numpy as np
import pandas as pd

import neptune.new as neptune

from modeling import dispatcher
from evaluation import loss

def train(train_path, fold, store_model_at, model):
    df = pd.read_csv(train_path)

    train = df[df.kfold != fold]
    val = df[df.kfold == fold]

    ytrain = train.target.values
    yval = val.target.values

    cols_to_drop = ['id', 'url_legal', 'license', 'excerpt', 'standard_error', 'target', 'kfold']
    xtrain = train.drop(cols_to_drop, axis=1)
    xval = val.drop(cols_to_drop, axis=1)

    model = dispatcher.MODELS[model]

    model.fit(xtrain, ytrain)
    preds = model.predict(xval)

    rmse = loss.rmse(preds, yval)
    print(f'Loss for fold={fold} : rmse={rmse}')

    joblib.dump(model, f'{store_model_at}/{model}_{fold}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type='int', help='to specify which fold of cv to run')
    parser.add_argument('--train_path', type='str', help='to provide training data path')
    parser.add_argument('--store_model_at', type='str', help='to provide path for storing trained model')
    parser.add_argument('--model', type='str', help='to dispatch a pariticular model (RF, LGB, etc.)')

    args = parser.parse_args()

    train(train_path=args.train_path, fold=args.fold, store_model_at=args.store_model_at, model=args.model)