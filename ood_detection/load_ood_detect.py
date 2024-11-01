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
import numpy as np
from collections import namedtuple

from sklearn.model_selection import train_test_split
from pathlib import Path

DataState = namedtuple("data", ["xtr", "xval", "xte", "ytr", "yval", "yte"])
DataPair = namedtuple("DataPair", ["x", "y"])

OOD_DATA_DIR = Path(__file__).parent
OOD_DATA_DIR.mkdir(exist_ok=True)

OOD_DATASETS = ["mnist", "svhn_cropped", "cifar10", "omniglot", "fashion_mnist"]


def load_tf_img_dataset(dataset, rng_seed=None, use_validation=True, take=-1):
    data = {}
    dss = tfds.load(dataset, data_dir=OOD_DATA_DIR)
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
    datasets = {
        dataset: load_tf_img_dataset(
            dataset, rng_seed=rng_seed, use_validation=use_val, take=take
        )
        for dataset in OOD_DATASETS
    }
    return datasets


load_datasets(0, False, -1)
