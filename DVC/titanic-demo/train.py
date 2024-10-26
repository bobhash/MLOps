import argparse

import numpy as np
import pandas as pd
from termcolor import colored
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepath", type=str, required=True)
    parser.add_argument("--train_csv_path", type=str, required=True)
    return parser


def train(train_csv_path: str, savepath: str):
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.fillna(-999)

    x = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]
    cate_features_index = np.where(x.dtypes != float)[0]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    train_pool = Pool(x_train, y_train, cat_features=cate_features_index)
    val_pool = Pool(x_val, y_val, cat_features=cate_features_index)

    model = CatBoostClassifier(
        depth=5,
        iterations=1000,
        eval_metric="AUC",
        use_best_model=True,
        random_seed=42,
        early_stopping_rounds=100,
        metric_period=10,
    )
    model.fit(train_pool, eval_set=val_pool)

    auc_score = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
    print(
        colored("Val AUC Score:", color="green", attrs=["bold"]),
        colored(f"{round(auc_score, 2)}", color="white", attrs=["bold"]),
    )
    model.save_model(savepath, format="cbm")


if __name__ == "__main__":
    args = configure_parser().parse_args()
    train(args.train_csv_path, args.savepath)
