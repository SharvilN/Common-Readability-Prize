export PYTHONPATH=$PYTHONPATH:/home/sharvil/studies/ml/Common-Readability-Prize/readability_dl

TRAIN_PATH=data/raw/train.csv
TEST_PATH=data/raw/test.csv
CV_PATH=data/interim/clrp-dl-folds.csv

python readability_dl/train.py --fold=0 --train_path=$TRAIN_PATH --cv_path=$CV_PATH
python readability_dl/train.py --fold=1 --train_path=$TRAIN_PATH --cv_path=$CV_PATH
python readability_dl/train.py --fold=2 --train_path=$TRAIN_PATH --cv_path=$CV_PATH
python readability_dl/train.py --fold=3 --train_path=$TRAIN_PATH --cv_path=$CV_PATH
python readability_dl/train.py --fold=4 --train_path=$TRAIN_PATH --cv_path=$CV_PATH
