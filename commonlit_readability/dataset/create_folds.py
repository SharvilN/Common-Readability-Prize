import argparse
import pandas as pd

from typing import List, Optional

from commonlit_readability.dataset import cross_validation

def run(data_path: str, output_path: str, target_cols: List[str], problem_type: cross_validation.ProblemType,  n_folds: Optional[int] = 5) -> None:

    print(f'Target columns : {target_cols}')
    print(f'Reading data from {data_path}')
    df = pd.read_csv(data_path)
    cv = cross_validation.CrossValidation(df,
                        target_cols,
                        n_folds=n_folds,
                        shuffle=True,
                        problem_type=problem_type)
    
    print(f'Generating folds...')
    df_split = cv.split()
    print(df_split.head())
    print(df_split.tail())
    print(df_split.groupby(by=['kfold'])['target'].median())

    print(f'Saving train folds to {output_path}')
    df_split.to_csv(output_path, index=False)

if __name__ == '__main__':
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
    problem_type=cross_validation.ProblemType[args.problem_type.upper()]

    run(data_path, output_path, target_cols, problem_type, n_folds)

