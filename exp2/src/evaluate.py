import json
import math
import os
import pickle
import sys

import pandas as pd

from dvclive import Live




def main():
    model_file = sys.argv[1]

    x_test_path = os.path.join("data", "prepared", "X_test.csv")
    y_test_path = os.path.join("data", "prepared", "y_test.csv")

    X = pd.read_csv(x_test_path)
    y = pd.read_csv(y_test_path)

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    with Live() as live:
        accuracy = model.score(X, y)
        live.log_metric("accuracy", accuracy)
        live.make_summary()