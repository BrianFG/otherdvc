import json
import math
import os
import pickle
import sys
import json
import argparse
import yaml
import pandas as pd

from dvclive import Live




def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params = yaml.safe_load(open(args.config))

    model_file = os.path.join(params["base"]["model_dir"], params["base"]["model_name"])
    data_dir = params["base"]["data_dir"]


    x_test_path = os.path.join(data_dir, "prepared", "X_test.pkl")
    y_test_path = os.path.join(data_dir, "prepared", "y_test.pkl")

    X = pd.read_pickle(x_test_path)
    y = pd.read_pickle(y_test_path)

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    with Live("eval_plots", save_dvc_exp=True) as live:
        accuracy = model.score(X, y)
        print("Accuracy:", accuracy)
        json.dump({"accuracy": accuracy}, open("metrics.json", "w"))



if __name__ == "__main__":
    main()