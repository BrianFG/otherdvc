import pandas as pd
import sys
import os
import pickle
from  sklearn.ensemble import RandomForestRegressor


def main():
    in_path = sys.argv[1]


    x_train_path = os.path.join(in_path, "X_train.csv")
    y_train_path = os.path.join(in_path, "y_train.csv")

    X = pd.read_csv(x_train_path)
    y = pd.read_csv(y_train_path)


    regressor = RandomForestRegressor(random_state=0, n_estimators=1000).fit(X, y)

    with open("model2.pkl", "wb") as f:
        pickle.dump(regressor, f)


if __name__ == "__main__":
    main()
