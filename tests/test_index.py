import numpy as np

from benchmark import make_vectors
from vsearch.index import IVF, LSH, BruteForce, normalize, recall_at_k


def data():
    return make_vectors(3000, 32, 20)


def test_bruteforce_matches_naive_sort():
    X = data()
    q = X[5] + 0.1
    Xn = normalize(X.astype(np.float32))
    expected = np.argsort(-(Xn @ normalize(q.reshape(1, -1).astype(np.float32))[0]))[:10]
    assert list(BruteForce(X).search(q, 10)) == list(expected)


def test_self_query_returns_itself_first():
    X = data()
    for cls in (BruteForce, LSH, IVF):
        idx = cls(X) if cls is not IVF else IVF(X, n_clusters=16, nprobe=4)
        assert idx.search(X[7], 1)[0] == 7


def test_ivf_full_probe_is_exact():
    X = data()
    ivf = IVF(X, n_clusters=16, nprobe=16)
    bf = BruteForce(X)
    for i in range(0, 50, 10):
        assert recall_at_k(bf.search(X[i] + 0.2, 10), ivf.search(X[i] + 0.2, 10)) == 1.0


def test_more_probes_never_reduce_recall_on_average():
    X = data()
    bf = BruteForce(X)
    Q = X[:40] + 0.3 * np.random.default_rng(2).normal(size=(40, 32))
    def avg(nprobe):
        ivf = IVF(X, n_clusters=16, nprobe=nprobe)
        return np.mean([recall_at_k(bf.search(q), ivf.search(q)) for q in Q])
    assert avg(8) >= avg(1)


def test_recall_function():
    assert recall_at_k([1, 2, 3, 4], [1, 2, 9, 8]) == 0.5
