import time

import numpy as np

from vsearch.index import IVF, LSH, BruteForce, recall_at_k


def make_vectors(n=20000, d=64, centers=50, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(centers, d))
    return C[rng.integers(0, centers, n)] + 0.6 * rng.normal(size=(n, d))


def bench(n=20000, d=64, n_queries=200, k=10):
    X = make_vectors(n, d)
    rng = np.random.default_rng(1)
    Q = X[rng.choice(n, n_queries, replace=False)] + 0.3 * rng.normal(size=(n_queries, d))
    exact = BruteForce(X)
    truth = [exact.search(q, k) for q in Q]
    rows = {}
    configs = {"brute_force": exact, "lsh(8 tables,12 bits)": LSH(X), "ivf(nprobe=1)": IVF(X, nprobe=1),
               "ivf(nprobe=4)": IVF(X, nprobe=4), "ivf(nprobe=8)": IVF(X, nprobe=8)}
    for name, idx in configs.items():
        t = time.perf_counter()
        res = [idx.search(q, k) for q in Q]
        ms = (time.perf_counter() - t) / n_queries * 1000
        rows[name] = (round(float(np.mean([recall_at_k(a, b) for a, b in zip(truth, res)])), 3), round(ms, 3))
    return rows


if __name__ == "__main__":
    for name, (rec, ms) in bench().items():
        print(f"{name:24s} recall@10={rec:<6} ms/query={ms}")
