# Datasets
# - CIFAR10 Corrupted [1]
#   - In tensorflow datasets: cifar10_corrupted
# - CIFAR 10 [2]
#   - In tensorflow datasets: cifar10
# - CIFAR 100 [2]
#   - In tensorflow datasets: cifar100

# Refs
# [1] Hendrycks, Dan, and Thomas Dietterich. "Benchmarking neural network robustness to common corruptions and perturbations." 
#     arXiv preprint arXiv:1903.12261 (2019).
# [2] Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. 
#     "Human-level concept learning through probabilistic program induction." Science 350.6266 (2015): 1332-1338.


import tensorflow_datasets as tfds
import numpy as np
from collections import namedtuple

from sklearn.model_selection import train_test_split
from pathlib import Path

# TODO: refactor to avoid copying
# TODO: add evaluation dataset construction

DataState = namedtuple("data", ["xtr", "xval", "xte", "ytr", "yval", "yte"])
DataPair = namedtuple("DataPair", ["x", "y"])

CORRUPT_DATA_DIR = Path(__file__).parent
CORRUPT_DATA_DIR.mkdir(exist_ok=True)

CORRUPT_DATASETS = ['cifar10', 'cifar100', 'cifar10_corrupted']

def load_tf_img_dataset(dataset, rng_seed=None, use_validation=True, take=-1):
    data = {"train": (None, None), "validation": (None,None), "test": (None,None)}
    dss = tfds.load(dataset, data_dir=CORRUPT_DATA_DIR)

    for split in set(dss.keys()):
        ds = dss[split]
        data[split] = tuple(map(np.stack,
            zip(*((datum["image"].numpy().squeeze(), datum["label"].numpy().squeeze())
                for datum in ds.take(take)
            ))))

    if use_validation and data['train'] is not None and data["validation"] is None:
        xtr, xval, ytr, yval = train_test_split(
            *data['train'],
            train_size=0.9,
            random_state=rng_seed,
        )
        data['train'] = xtr, ytr
        data['validation'] = xval, yval

    dataset = DataState( *data['train'], *data['validation'], *data['test'])

    return dataset

def load_datasets(rng_seed, use_val, take):
    datasets = {dataset: load_tf_img_dataset(dataset, rng_seed=rng_seed, use_validation=use_val, take=take) for dataset in CORRUPT_DATASETS}
    return datasets

load_datasets(0, False, -1)