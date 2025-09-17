from catboost import CatBoostRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():
    X, y = load_iris(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostRegressor()
    model.fit(X_train, y_train)
    model.save_model("trained_model.cbm", format="cbm")


if __name__ == "__main__":
    main()
