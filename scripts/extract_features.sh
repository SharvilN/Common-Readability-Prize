export PYTHONPATH=$PYTHONPATH:/home/sharvil/studies/ml/Common-Readability-Prize/commonlit_readability

INPUT_PATH=data/raw/test.csv
OUTPUT_PATH=data/processed/test-processed.csv

python commonlit_readability/features/extract_features.py --input_path=$INPUT_PATH --output_path=$OUTPUT_PATH