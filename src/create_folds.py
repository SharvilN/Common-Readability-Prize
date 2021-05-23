import argparse
import pandas as pd
from cross_validation import CrossValidation

def run(args):
    data_path = args.data_path
    output_path = args.output_path
    target_cols = args.target_cols.split(',')
    n_folds = args.n_folds

    print(target_cols)
    print(f'Reading data from {data_path}')
    df = pd.read_csv(data_path)
    cv = CrossValidation(df,
                        target_cols,
                        n_folds=n_folds,
                        shuffle=True,
                        problem_type=CrossValidation.ProblemType.SINGLE_COL_REGRESSION)
    
    print(f'Generating folds...')
    df_split = cv.split()
    print(df_split.head())
    print(df_split.tail())
    print(df_split.groupby(by=['kfold'])['target'].median())

    print(f'Saving train folds to {output_path}')
    df_split.to_csv(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='to specify path to raw training data')
    parser.add_argument('--output_path', help='to specify path to generated k-fold data')
    parser.add_argument('--target_cols', help='to specify target columns of data')
    parser.add_argument('--n_folds', type=int, default=5, help='specifies the number of splits to perform on data')

    run(parser.parse_args())

