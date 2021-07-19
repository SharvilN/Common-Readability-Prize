export PYTHONPATH=$PYTHONPATH:/home/sharvil/studies/ml/Common-Readability-Prize/commonlit_readability

# declare training variables

TRAIN_PATH=data/processed/non-nn-processed-v1-folds.csv
MODEL_DIR=models/
MODEL=RIDGE

python commonlit_readability/modeling/train.py --fold=0 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=$MODEL
python commonlit_readability/modeling/train.py --fold=1 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=$MODEL
python commonlit_readability/modeling/train.py --fold=2 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=$MODEL
python commonlit_readability/modeling/train.py --fold=3 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=$MODEL
python commonlit_readability/modeling/train.py --fold=4 --train_path=$TRAIN_PATH --store_model_at=$MODEL_DIR --model=$MODEL




