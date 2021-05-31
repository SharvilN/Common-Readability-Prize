# declare cv variables

export PYTHONPATH=$PYTHONPATH:/home/sharvil/studies/ml/Common-Readability-Prize/commonlit_readability

# DATA_PATH=data/processed/non-nn-processed-v1.csv
# CV_DATA_PATH=data/processed/non-nn-processed-v1-folds.csv
# TARGET_COLS=target
# N_FOLDS=5

# python commonlit_readability/dataset/create_folds.py --data_path=$DATA_PATH \
#                        --output_path=$CV_DATA_PATH \
#                        --target_cols=$TARGET_COLS \
#                        --n_folds=$N_FOLDS


# declare training variables

TRAIN_PATH=data/processed/non-nn-processed-v1-folds.csv
MODEL_DIR=models/
MODEL=RIDGE

python modeling/train.py --fold=1 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=MODEL
python modeling/train.py --fold=2 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=MODEL
python modeling/train.py --fold=3 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=MODEL
python modeling/train.py --fold=4 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=MODEL
python modeling/train.py --fold=5 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=MODEL

# declare evaluation variables

# TEST_PATH=../data/raw/test.csv
# MODEL_DIR=../models/

# python modeling/predict.py --test_path=$TEST_PATH --model_dir=$MODEL_DIR

