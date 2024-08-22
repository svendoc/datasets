
# Datasets
# - CIFAR10 Corrupted [1]
#   - In tensorflow datasets: cifar10_corrupted
# - CIFAR100 Corrupted [1]
#   - In tensorflow datasets: cifar100_corrupted
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
DataState = namedtuple("data", ["xtr", "xval", "xte", "ytr", "yval", "yte"])
DataPair = namedtuple("DataPair", ["x", "y"])

CORRUPT_DATA_DIR = Path(__file__).parent
CORRUPT_DATA_DIR.mkdir(exist_ok=True)

CORRUPT_DATASETS = ['cifar10', 'cifar100', 'cifar10_corrupted', 'cifar100_corrupted']

def load_tf_img_dataset(dataset, rng_seed=None, use_validation=True, take=-1):
    data = {}
    dss = tfds.load(dataset, data_dir=CORRUPT_DATA_DIR)
    splits = set(dss.keys()) & {"train", "validation", "test"}

    for split in splits:
        ds = dss[split]
        data[split] = np.stack(
            tuple(
                np.concatenate(
                    (
                        datum["image"].numpy().squeeze().reshape(-1),
                        datum["label"].numpy().squeeze().reshape(-1),
                    )
                )
                for datum in ds.take(take)
            )
        )

    if "validation" in splits:
        tr = np.vstack(
            [data["train"], data["validation"]]
        )  # Use own train/validate split
    else:
        tr = data["train"]

    if use_validation:
        tr, val = train_test_split(
            tr,
            train_size=0.9,
            random_state=rng_seed,
        )
        dataset = DataState(
            *map(
                partial(jnp.array, dtype=int),
                [tr[:, :-1], val[:, :-1], data["test"][:, :-1]],
            ),
            *map(
                partial(jnp.array, dtype=int),
                [tr[:, -1], val[:, -1], data["test"][:, -1]],
            ),
        )
    else:
        dataset = DataState(
            np.array(tr[:, :-1], dtype=int),
            None,
            np.array(data["test"][:, :-1]),
            np.array(tr[:, -1], dtype=int),
            None,
            np.array(data["test"][:, -1]),
        )

    return dataset

def load_datasets(rng_seed, use_val, take):
    datasets = {dataset: load_tf_img_dataset(dataset, rng_seed=rng_seed, use_validation=use_val, take=take) for dataset in CORRUPT_DATASETS}
    return datasets

load_datasets(0, False, -1)