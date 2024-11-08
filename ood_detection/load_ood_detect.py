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
import tensorflow as tf
from pathlib import Path

OOD_DATA_DIR = Path(__file__).parent
OOD_DATA_DIR.mkdir(exist_ok=True)

OOD_DATASETS = ["mnist", "svhn_cropped", "cifar10", "omniglot", "fashion_mnist"]


batch_size = 512

def train_batches(batch_size, dataset):
    assert dataset in OOD_DATASETS

    # as_supervised=True gives us the (image, label) as a tuple instead of a dict
    ds = tfds.load(name=dataset, split='train', as_supervised=True, data_dir=OOD_DATA_DIR)
    n = tf.data.experimental.cardinality(ds).numpy().item()

    # You can build up an arbitrary tf.data input pipeline
    ds = ds.shuffle(buffer_size=10_000).batch(batch_size, drop_remainder=True).prefetch(1)

    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(ds), n


def test_data(dataset):
    assert dataset in OOD_DATASETS

    # as_supervised=True gives us the (image, label) as a tuple instead of a dict
    ds = tfds.load(name=dataset, split='test', as_supervised=True, data_dir=OOD_DATA_DIR)
    n = tf.data.experimental.cardinality(ds).numpy().item()

    x,y = list(tfds.as_numpy(ds.batch(n)))[0]
    return x, y, n

