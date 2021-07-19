import pandas as pd
import os
import joblib
import argparse
import numpy as np

def predict(test_path: str, model_dir: str, model_prefix: str):
    test_df = pd.read_csv(test_path)
    cols_to_drop = ['id', 'url_legal', 'license', 'excerpt', 'preprocessed_excerpt', 'n_longest_sent']
    test = test_df.drop(cols_to_drop, axis=1)

    models = list()
    for dirname, _, filenames in os.walk(model_dir):
        for filename in filenames:
            if model_prefix in filename:
                models.append(joblib.load(os.path.join(dirname, filename)))

    preds = [model.predict(test) for model in models]
    preds = np.mean(preds, axis=0)

    print(preds)
    submission = pd.DataFrame({"id": test_df["id"], "target": preds})
    submission.to_csv("reports/submissions/submission_1.csv", index=False)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='to provide test data path')
    parser.add_argument('--model_dir', help='to provide path for stored trained model')
    parser.add_argument('--model_prefix', help='to provide prefix of model names trained on multiple folds')

    args = parser.parse_args()

    predict(args.test_path, args.model_dir, args.model_prefix)