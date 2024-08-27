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


def load_gap_splits(use_validation):
    splits = defaultdict(list)
    for dataset, (x,y) in load_datasets().items():
        for dim in range(x.shape[1]):
            ordering = argsort(x[:,dim])
            tr, te = concatenate((ordering[:x.shape[0]//3], ordering[2*x.shape[0]//3:])), ordering[x.shape[0] // 3 + 1: 2 * x.shape[0]//3]
            te = te.astype(int)
            if use_validation:
                tr, val = train_test_split(tr, test_size=.1)
                val = val.astype(int)
            else:
                val = None
            tr = tr.astype(int)
            splits[dataset].append({'tr': tr, 'val':val, 'te':te})
    return splits


def load_standard_splits(use_validation):
    splits = defaultdict(list)
    for dataset in UCI_DIR.iterdir():
        if dataset.is_dir() and (dataset / 'data').exists():
            n = int(loadtxt(dataset / 'data' / 'n_splits.txt').item())
            assert n > 0
            for i in range(n):
                if (dataset / 'data' / f'index_train_{i}.txt').exists():

                    tr =loadtxt(dataset / 'data' / f'index_train_{i}.txt').astype(int)
                    assert tr.dtype == int, 'UCI Train index must have int type'

                    if use_validation:
                        tr, val = train_test_split(tr, test_size=.1)
                        assert tr.dtype == int, 'UCI Train indices must have int type'
                        assert val.dtype == int, 'UCI Validations indices must have int type'
                    else:
                        val = None
                    te = loadtxt(dataset / 'data' / f"{'_'.join(('index', 'test', str(i)))}.txt").astype(int)
                    assert te.dtype == int, 'UCI Test indices must have int type'

                    splits[dataset.stem].append({'tr': tr, 'val': val, 'te': te})
    return splits


def pprint_summary_latex(split, use_validation):
    assert split in ['std', 'gap'], 'Unkwown dataset split'
    datasets = load_datasets()
    match split:
        case 'std':
            indices = load_standard_splits(use_validation)
        case 'gap':
            indices = load_gap_splits(use_validation)

    if use_validation:
        header = r'''\begin{tabular}{lccccc}
        \toprule 
        Dataset & Train size & Val size & Test size & Features & Splits \\
        \midrule'''
    else:
        header = r'''\begin{tabular}{lcccc}
        \toprule 
        Dataset & Train size & Test size & Features & Splits \\
        \midrule'''
    
    rows = []

    for dataset, (x,_) in sorted(datasets.items(), key=lambda item: item[0]):
        for l, ind in enumerate(indices[dataset]):
            xtr  = x[ind["tr"]]
            if l == 0:
                n, m = xtr.shape
            assert xtr.shape == (n, m), f"Train splits not equally sized for {dataset}."

        if use_validation:
            lens = [len(ind['tr']),  len(ind['val']), len(ind['te'])]
        else: 
            lens = [len(ind['tr']), len(ind['te'])]

        row = [dataset.split('-')[0].capitalize()] + lens + [m, len(indices[dataset])]

        rows.append(r'        ' + ' & '.join(map(str, row)) + r' \\')
    data = '\n'.join(rows)

    footer = r'''        \bottomrule
\end{tabular}'''
    return '\n'.join([header, data, footer])
    