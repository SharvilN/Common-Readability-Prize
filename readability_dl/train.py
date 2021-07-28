import os
from readability_dl.model import CLRPModel
from readability_dl.metrics import AverageMeter
import transformers

from readability_dl import config
import torch
import pandas as pd

from torch.utils.data import DataLoader
import torch.nn as nn
from readability_dl.dataset import CLRPDataset
from transformers import AdamW, get_cosine_schedule_with_warmup
from readability_baseline.dataset.cross_validation import CrossValidation, ProblemType

from engine import Trainer

def run(df, fold, model_dir):

    xtrain = df[df["fold"] == fold]
    xvalid = df[df ["fold"] != fold]
    dtrain = CLRPDataset(xtrain, config.CHECKPOINT, config.MAX_LEN)
    dvalid = CLRPDataset(xvalid, config.CHECKPOINT, config.MAX_LEN)

    train_loader = DataLoader(
        dtrain,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        dvalid,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=True
    )

    num_training_steps = config.EPOCHS*len(dtrain)
    model = CLRPModel(checkpoint=config.CHECKPOINT)
    optimizer = AdamW(model.parameters(), lr=config.LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    print(f"Fold: {fold}")
    print(f"Num training steps: {num_training_steps}")

    result_dict = {}
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=config.EPOCHS,
        log_interval=config.LOG_INTERVAL,
        eval_interval=config.EVAL_INTERVAL,
        model_dir=model_dir
    )
    result_dict = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        result_dict=result_dict,
        fold=fold
    )

    


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, help="to specify # of fold")
    parser.add_argument("--train_path", help="to specify train data path")
    parser.add_argument("--cv_path", help="to specify cross validated data path")
    parser.add_argument("--model_dir", help="to specify path for storing trained model")
    args = parser.parse_args()

    if not os.path.exists(args.cv_path):
        raw_train = pd.read_csv(args.train_path)
        cv = CrossValidation(
            raw_train, 
            config.TARGET_COLS, 
            problem_type=ProblemType.REGRESSION, 
            shuffle=True,
            n_folds=5
        )
        df_folds = cv.split()
        df_folds.to_csv(args.cv_path)
    else:
        df_folds = pd.read_csv(args.cv_path)

    run(df_folds, args.fold, args.model_dir)


