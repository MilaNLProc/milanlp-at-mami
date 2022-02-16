from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset
import os
import pandas as pd


TEST_FILE = os.path.join("data", "test", "Test.csv")


def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def compute_metrics_multi_task(predictions, targets, names):
    d = dict()
    for i, n in enumerate(names):
        d[n] = compute_metrics(targets[:, i], predictions[:, i])
    return d


def read_test_dataset():
    df = pd.read_csv(TEST_FILE, sep="\t")
    return Dataset.from_pandas(df)