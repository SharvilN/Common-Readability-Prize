TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
CV_PATH = "data/interim/clrp-dl-folds.csv"

TARGET_COLS = ["target"]
MAX_LEN = 128
CHECKPOINT = "distil-bert-uncased"
EPOCHS = 5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8
LR = 5e-5
DEVICE = "cpu"
EVAL_INTERVAL = 10
LOG_INTERVAL = 10