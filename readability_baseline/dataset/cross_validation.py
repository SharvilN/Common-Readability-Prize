import pandas as pd
import numpy as np

from enum import Enum, auto
from sklearn import model_selection

'''
- binary classification
- multiclass classification
- multilabel classification
- single column regression
- multi column regression
- holdout
'''

class ProblemType(Enum):
        BINARY = auto(),
        MULTICLASS = auto(),
        REGRESSION = auto(),
        MULTI_COL_REGRESSION = auto(),
        HOLDOUT = auto(),
        MULTILABEL = auto()


class CrossValidation:

    def __init__(
            self,
            df,
            target_cols,
            shuffle,
            problem_type,
            holdout_pct=None,
            n_folds=5,
            multilabel_delimiter=',',
            random_state=31
        ):
        self.df = df
        self.target_cols = target_cols
        self.n_targets = len(self.target_cols)
        self.problem_type = problem_type
        self.shuffle = shuffle
        self.n_folds = n_folds
        self.holdout_pct = holdout_pct
        self.multilabel_delimiter = multilabel_delimiter
        self.random_state = random_state

        if self.shuffle is True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df['kfold'] = -1

    def split(self):
        if self.problem_type in (ProblemType.BINARY, ProblemType.MULTICLASS):
            if self.n_targets != 1:
                raise Exception(f'Invalid number of targets {self.n_targets} for selected problem type: {self.problem_type}') 
            
            target = self.target_cols[0]
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds)
        
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                print(len(train_idx), len(val_idx))
                self.df.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in (ProblemType.REGRESSION, ProblemType.MULTI_COL_REGRESSION):
            if self.n_targets != 1 and self.problem_type == ProblemType.REGRESSION:
                raise Exception(f'Invalid combination of number of targets {self.n_targets} and problem type {self.problem_type}')
            if self.n_targets < 2 and self.problem_type == ProblemType.MULTI_COL_REGRESSION:
                raise Exception(f'Invalid combination of number of targets {self.n_targets} and problem type {self.problem_type}')
            
            target = self.target_cols[0]

            # calculate number of bins by Sturge's rule
            # I take the floor of the value, you can also
            # just round it
            num_bins = int(np.floor(1 + np.log2(len(self.df))))
            
            # bin targets
            self.df.loc[:, "bins"] = pd.cut(
                self.df["target"], bins=num_bins, labels=False
            )
            
            # initiate the kfold class from model_selection module
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds)
            
            # fill the new kfold column
            # note that, instead of targets, we use bins!
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.df, y=self.df.bins.values)):
                print(len(train_idx), len(valid_idx))
                self.df.loc[valid_idx, 'kfold'] = fold
            
            # drop the bins column
            self.df = self.df.drop("bins", axis=1)

        
        elif self.problem_type == ProblemType.HOLDOUT:
            holdout_pctg = self.holdout_pct
            n_holdout_samples = int(len(self.df) * holdout_pctg / 100)
            self.df.loc[:n_holdout_samples, 'kfold'] = 0
            self.df.loc[n_holdout_samples: , 'kfold'] = 1
            print(n_holdout_samples)

        elif self.problem_type == ProblemType.MULTILABEL:
            if self.n_targets != 1:
                raise Exception(f'Invalid combination of number of targets {self.n_targets} and problem type {self.problem_type}')

            targets = self.df[self.target_cols[0]].apply(lambda x: len(x.split(self.multilabel_delimiter)))
            print(targets.value_counts())
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds)

            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.df, y=targets)):
                print(len(train_idx), len(valid_idx))
                self.df.loc[valid_idx, 'kfold'] = fold

        else:
            raise Exception(f'Invalid problem type found : {self.problem_type}')
        return self.df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='to specify path to raw training data')
    parser.add_argument('--output_path', help='to specify path to generated k-fold data')
    parser.add_argument('--target_cols', help='to specify target columns of data')
    parser.add_argument('--problem_type', help='regression, classification, etc.')
    parser.add_argument('--n_folds', type=int, default=5, help='specifies the number of splits to perform on data')

    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    target_cols = args.target_cols.split(',')
    n_folds = args.n_folds
    problem_type=ProblemType[args.problem_type.upper()]

    print(f'Target columns : {target_cols}')
    print(f'Reading data from {data_path}')
    df = pd.read_csv(data_path)
    cv = CrossValidation(
                    df,
                    target_cols,
                    n_folds=n_folds,
                    shuffle=True,
                    problem_type=problem_type
                )
    
    print(f'Generating folds...')
    df_split = cv.split()
    print(df_split.head())
    print(df_split.tail())
    print(df_split.groupby(by=['kfold'])['target'].median())

    print(f'Saving train folds to {output_path}')
    df_split.to_csv(output_path, index=False)
