export PYTHONPATH=$PYTHONPATH:/home/sharvil/studies/ml/Common-Readability-Prize/readability_baseline

# declare cv variables

DATA_PATH=data/processed/non-nn-processed-v1.csv
CV_DATA_PATH=data/processed/non-nn-processed-v1-folds.csv
TARGET_COLS=target
N_FOLDS=5

python readability_baseline/dataset/cross_validation.py --data_path=$DATA_PATH \
                       --output_path=$CV_DATA_PATH \
                       --target_cols=$TARGET_COLS \
                       --n_folds=$N_FOLDS \
                       --problem_type=REGRESSION
