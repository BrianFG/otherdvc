import pandas as pd
import sys
import os
import pickle
import argparse
import yaml
from sklearn.tree import DecisionTreeRegressor

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params = yaml.safe_load(open(args.config))
    print(params)
    in_path = params["base"]["data_dir"]

    x_train_path = os.path.join(in_path, "prepared", "X_train.pkl")
    y_train_path = os.path.join(in_path, "prepared", "y_train.pkl")

    X = pd.read_pickle(x_train_path)
    y = pd.read_pickle(y_train_path)


    regressor = DecisionTreeRegressor(random_state=0).fit(X, y)
    model_name = params["base"]["model_name"]
    model_dir = params["base"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    with open(model_path, "wb") as f:
        pickle.dump(regressor, f)


if __name__ == "__main__":
    main()
