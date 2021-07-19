export PYTHONPATH=$PYTHONPATH:/home/sharvil/studies/ml/Common-Readability-Prize/readability_baseline

# declare evaluation variables

TEST_PATH=data/processed/test-processed.csv
MODEL_DIR=models/
MODEL_PREFIX=Ridge

python readability_baseline/modeling/predict.py --test_path=$TEST_PATH --model_dir=$MODEL_DIR --model_prefix=$MODEL_PREFIX