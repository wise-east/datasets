import os
from subprocess import run 
import shlex 

DATASETS = {
    "chemprot": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/chemprot/",
        "dataset_size": 4169
    },
    "rct-20k": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-20k/",
        "dataset_size": 180040
    },
    "rct-sample": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-sample/",
        "dataset_size": 500
    },
    "citation_intent": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/citation_intent/",
        "dataset_size": 1688
    },
    "sciie": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/sciie/",
        "dataset_size": 3219
    },
    "ag": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/ag/",
        "dataset_size": 115000
    },
    "hyperpartisan_news": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/hyperpartisan_news/",
        "dataset_size": 500
    },
    "imdb": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/imdb/",
        "dataset_size": 20000
    },
    "amazon": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/amazon/",
        "dataset_size": 115251
    }
}


splits = ["train", "dev", "test"]

for dataset_name, dataset_configs in DATASETS.items(): 
    for split in splits: 
        fp = os.path.join(dataset_configs["data_dir"], f"{split}.jsonl")
        dataset_path = f"tapt_tasks/{dataset_name}"
        os.makedirs(dataset_path, exist_ok=True)
        cmd = f"curl -Lo {os.path.join(dataset_path, f'{split}.jsonl')} {fp}"
        print(cmd)
        run(shlex.split(cmd))