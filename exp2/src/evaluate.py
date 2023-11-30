import json
import math
import os
import pickle
import sys
import json

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

    with Live("evaluate", save_dvc_exp=True) as live:
        accuracy = model.score(X, y)
        print("Accuracy:", accuracy)
        #json.dump({"accuracy": accuracy}, open("metrics.json", "w"))
        live.log_metric("accuracy", accuracy)
        live.log_metric("accuracy2", accuracy * 2)
        live.make_summary()

if __name__ == "__main__":
    main()