import os
import random
import sys
import pandas as pd

import yaml
from sklearn.model_selection import train_test_split


def main():
    params = yaml.safe_load(open("./params.yaml"))["prepare"]
    data_path = sys.argv[1]

    df = pd.read_csv(data_path)
    df = df.dropna()
    X, y = df.iloc[:, :-2], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["split"], random_state=params["seed"])

    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)
    output_x_train = os.path.join("data", "prepared", "X_train.csv")
    output_x_test = os.path.join("data", "prepared", "X_test.csv")
    output_y_train = os.path.join("data", "prepared", "y_train.csv")
    output_y_test = os.path.join("data", "prepared", "y_test.csv")

    X_train.to_csv(output_x_train, index=False)
    X_test.to_csv(output_x_test, index=False)
    y_train.to_csv(output_y_train, index=False)
    y_test.to_csv(output_y_test, index=False)



if __name__ == "__main__":
    main()
