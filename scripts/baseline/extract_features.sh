export PYTHONPATH=$PYTHONPATH:/home/sharvil/studies/ml/Common-Readability-Prize/readability_baseline

INPUT_PATH=data/raw/test.csv
OUTPUT_PATH=data/processed/test-processed.csv

python readability_baseline/features/extract_features.py --input_path=$INPUT_PATH --output_path=$OUTPUT_PATH