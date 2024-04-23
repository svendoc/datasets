from numpy import loadtxt, concatenate, argsort
from pathlib import Path
from jax import numpy as jnp
from collections import defaultdict
from sklearn.model_selection import train_test_split

UCI_DIR = Path(__file__).parent

def load_datasets():
    uci_bench = {}

    for dataset in UCI_DIR.iterdir():
        filename = dataset / 'data' / 'data.txt'
        if filename.exists():
            data = loadtxt(filename)
            uci_bench[dataset.stem] = (jnp.array(data[:, :-1]), jnp.array(data[:,-1]))
    return uci_bench


def load_gap_splits():
    splits = defaultdict(list)
    for dataset, (x,y) in load_datasets().items():
        for dim in range(x.shape[1]):
            ordering = argsort(x[:,dim])
            tr, te = concatenate((ordering[:x.shape[0]//3], ordering[2*x.shape[0]//3:])), ordering[x.shape[0] // 3 + 1: 2 * x.shape[0]//3]
            tr, val = train_test_split(tr, test_size=.1)
            splits[dataset].append({'tr': tr, 'val':val, 'te':te})
    return splits


def load_standard_splits():
    splits = defaultdict(list)
    for dataset in UCI_DIR.iterdir():
        if (dataset / 'data' / f"{'_'.join(('index', 'train', str(0)))}.txt").exists():
            n = int(loadtxt(dataset / 'data' / 'n_splits.txt').item())
            for i in range(n):
                tr =loadtxt(dataset / 'data' / f"{'_'.join(('index', 'train', str(i)))}.txt")
                tr, val = train_test_split(tr, test_size=.1)
                te =loadtxt(dataset / 'data' / f"{'_'.join(('index', 'test', str(i)))}.txt")
                splits[dataset.stem].append({'tr': tr, 'val': val, 'te': te})
    return splits

print(len(load_standard_splits()['boston-housing']))