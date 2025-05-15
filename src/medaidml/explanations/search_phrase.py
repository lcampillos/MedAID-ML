import pandas as pd
import numpy as np
import argparse
import re
import os

from medaidml import DATA_TEST_JSON, DATA_TRAIN_JSON
from medaidml.utils import json_to_dataframe

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for a phrase in the dataset and show frequency in human and LLM generated text")
    parser.add_argument("--dataset", type=str, default="all", choices=["train", "test", "all"], help="Dataset to use")
    parser.add_argument("--phrase", type=str, required=True, help="Phrase to search for in the dataset")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PHRASE = args.phrase
    DATASET = args.dataset
    if DATASET == "train":
        dataset_path = DATA_TRAIN_JSON
    elif DATASET == "test":
        dataset_path = DATA_TEST_JSON
    else:
        dataset_path = [DATA_TRAIN_JSON, DATA_TEST_JSON]
    if isinstance(dataset_path, list):
        dataset_path = [json_to_dataframe(path) for path in dataset_path]
        dataset = pd.concat(dataset_path, ignore_index=True)
    else:
        dataset = json_to_dataframe(dataset_path)
    dataset = dataset.dropna(subset=["text"])
    dataset = dataset[dataset["text"].str.contains(PHRASE, case=False, na=False)]
    
    print(f"Phrase '{PHRASE}' found in {len(dataset)} samples.")
    print(f"HUMAN: Frequency: {dataset[dataset['target'] == 0]['text'].str.count(PHRASE, flags=re.IGNORECASE).sum()}")
    print(f"LLM: Frequency: {dataset[dataset['target'] == 1]['text'].str.count(PHRASE, flags=re.IGNORECASE).sum()}")
    
    print(f"{dataset[dataset['target'] == 1]['text'].str.count(PHRASE, flags=re.IGNORECASE).sum()} / {dataset['text'].str.count(PHRASE, flags=re.IGNORECASE).sum()}")
    print(f"{dataset[dataset['target'] == 1]['text'].str.count(PHRASE, flags=re.IGNORECASE).sum() / dataset['text'].str.count(PHRASE, flags=re.IGNORECASE).sum() * 100:.2f}%")