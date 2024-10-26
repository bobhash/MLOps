import argparse

import numpy as np
import pandas as pd
from catboost import Pool, CatBoostClassifier
from termcolor import colored
from sklearn.metrics import roc_auc_score


def configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", type=str, required=True)
    parser.add_argument("--test_csv_path", type=str, required=True)
    return parser


def infer(saved_model_path: str, test_csv_path: str):
    test_df = pd.read_csv(test_csv_path)
    test_df = test_df.fillna(-999)

    x = test_df.drop("Survived", axis=1)
    y = test_df["Survived"]
    cate_features_index = np.where(x.dtypes != float)[0]

    test_pool = Pool(data=x, label=y, cat_features=cate_features_index)

    model = CatBoostClassifier()
    model.load_model(saved_model_path, format="cbm")

    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    auc_score = roc_auc_score(y, y_pred_proba)
    print(
        colored("Test AUC Score:", color="green", attrs=["bold"]),
        colored(f"{round(auc_score, 2)}", color="white", attrs=["bold"]),
    )


if __name__ == "__main__":
    args = configure_parser().parse_args()
    infer(args.saved_model_path, args.test_csv_path)
