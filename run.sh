
export MODEL=$1

python create_folds.py --data_path=../data/raw/train.csv \
                       --output_path=../data/interim/train_folds.csv \
                       --target_cols=target,standard_error \
                       --n_folds=5

python train.py --fold=1 --train_path=../data/processed/train_final.csv --store_model_at=../models/
python train.py --fold=2 --train_path=../data/processed/train_final.csv --store_model_at=../models/
python train.py --fold=3 --train_path=../data/processed/train_final.csv --store_model_at=../models/
python train.py --fold=4 --train_path=../data/processed/train_final.csv --store_model_at=../models/
python train.py --fold=5 --train_path=../data/processed/train_final.csv --store_model_at=../models/