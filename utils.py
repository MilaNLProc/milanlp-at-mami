from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset
import os
import pandas as pd
from PIL import Image


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


NAMES = ["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"]


class VizHelper:
    
    def __init__(self):
        data_dir = "./data"
        self.data_dir = data_dir
        
        self.train_df = pd.read_csv(f"{data_dir}/training/training.tsv", sep="\t")
        self.test_df = pd.read_csv(f"{data_dir}/test_labels.txt", sep="\t", header=None, names=NAMES)
        
        original_test = pd.read_csv(f"{data_dir}/test/Test.csv", sep="\t")
        self.test_df["Text Transcription"] = original_test["Text Transcription"]
        
        cat_df = pd.concat([self.train_df, self.test_df])
        
        self.web_df = pd.read_csv(f"{data_dir}/web_entities.tsv", sep="\t")
        self.nsfw = pd.read_csv(f"{data_dir}/nsfw.tsv", sep="\t")
        self.captions = pd.read_csv(f"{data_dir}/image_captions.tsv", sep="\t").rename(columns={"image": "file_name"})
        self.ff = pd.read_csv(f"{data_dir}/fairface.tsv", sep="\t")
        
        cat_df = cat_df.merge(self.web_df, on="file_name").merge(self.captions, on="file_name") 
        cat_df["nsfw"] = self.nsfw["is_safe_bool"]
        
        self.data = cat_df.set_index("file_name")
        
        
    def show_sample(self, file_name=None):
        if not file_name:
            sample = self.data.sample(1)
        else:
            sample = self.data.loc[file_name]

        img = Image.open(f"{self.data_dir}/images/{file_name}").convert("RGB")
        display(img)
        
        faces = self.ff.loc[self.ff.file_name == file_name]
        faces = faces[["race", "age", "gender"]]
        display(faces)
        
        return sample