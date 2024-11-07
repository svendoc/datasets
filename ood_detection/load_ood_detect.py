# Datasets
# - Street View House Numbers  [1]
#   - In tensorflow datasets: svhn_cropped
# - MNIST [???]
#   - In tensorflow-datasets: mnist
# - CIFAR-10 [3]
#   - In tensorflow datasets: cifar10
# - OMNIGLOT [4]
#   - In tensorflow datasets: omniglot
# - Fashion-MNIST [4]
#   - In tensorflow datasets: fashion-mnist

# Refs
# [1] Netzer, Yuval, et al. "Reading digits in natural images with unsupervised feature learning."
#     NIPS workshop on deep learning and unsupervised feature learning. Vol. 2011. No. 2. 2011.
# [2] Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009): 7.
# [3] Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum.
#     "Human-level concept learning through probabilistic program induction." Science 350.6266 (2015): 1332-1338.
# [4] Xiao, Han, Kashif Rasul, and Roland Vollgraf.
#     "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms." arXiv preprint arXiv:1708.07747 (2017).


import tensorflow_datasets as tfds
from collections import namedtuple
import jax.numpy as jnp

from sklearn.model_selection import train_test_split
from pathlib import Path

Dataset = namedtuple("Dataset", ["xtr", "ytr", "xval", "yval", "xte", "yte"])
DataPair = namedtuple("DataPair", ["x", "y"])

OOD_DATA_DIR = Path(__file__).parent
OOD_DATA_DIR.mkdir(exist_ok=True)

OOD_DATASETS = ["mnist", "svhn_cropped", "cifar10", "omniglot", "fashion_mnist"]


def load_dataset(dataset, rng_seed, use_val, take):
    data = {}
    dss = tfds.load(dataset, data_dir=OOD_DATA_DIR)
    splits = set(dss.keys()) & {"train", "validation", "test"}

    for split in splits:
        ds = dss[split]

        itms = ds.take(take)
        data[split] = DataPair(
            jnp.stack([itm["image"].numpy() / 255.0 for itm in itms]),
            jnp.stack([itm["label"].numpy() for itm in itms]).astype(int),
        )

    if use_val and "validation" not in splits:
        xtr, xval, ytr, yval = train_test_split(
            *data["train"],
            train_size=0.9,
            random_state=rng_seed,
        )
        data["train"] = DataPair(xtr, ytr)
        data["validation"] = DataPair(xval, yval)
    else:
        data["validation"] = DataPair(None, None)

    return Dataset(
        *data["train"],
        *data["validation"],
        *data["test"],
    )

    return dataset


def load_datasets(rng_seed, use_val, take):
    datasets = {
        dataset: load_dataset(dataset, rng_seed=rng_seed, use_val=use_val, take=take)
        for dataset in OOD_DATASETS
    }
    return datasets
