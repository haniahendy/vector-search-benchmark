"""Three nearest-neighbour indexes on L2-normalised vectors (cosine similarity), written with numpy.

- BruteForce: exact, the reference answer.
- LSH: random-hyperplane hashing in several tables.
- IVF: k-means clusters, search only the `nprobe` closest clusters.
"""
import numpy as np


def normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return X / n


class BruteForce:
    def __init__(self, X):
        self.X = normalize(np.asarray(X, dtype=np.float32))

    def search(self, q, k=10):
        q = normalize(np.asarray(q, dtype=np.float32).reshape(1, -1))[0]
        s = self.X @ q
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return idx[np.argsort(-s[idx])]


class LSH:
    def __init__(self, X, n_tables=8, n_bits=12, seed=0):
        self.X = normalize(np.asarray(X, dtype=np.float32))
        rng = np.random.default_rng(seed)
        self.planes = [rng.normal(size=(self.X.shape[1], n_bits)).astype(np.float32) for _ in range(n_tables)]
        self.tables = []
        for P in self.planes:
            keys = self._keys(self.X, P)
            table = {}
            for i, key in enumerate(keys):
                table.setdefault(int(key), []).append(i)
            self.tables.append(table)

    @staticmethod
    def _keys(X, P):
        bits = (X @ P) > 0
        return bits.astype(np.int64) @ (1 << np.arange(P.shape[1], dtype=np.int64))

    def search(self, q, k=10):
        q = normalize(np.asarray(q, dtype=np.float32).reshape(1, -1))
        cand = set()
        for P, table in zip(self.planes, self.tables):
            cand.update(table.get(int(self._keys(q, P)[0]), []))
        if not cand:
            return np.array([], dtype=int)
        cand = np.fromiter(cand, dtype=int)
        s = self.X[cand] @ q[0]
        return cand[np.argsort(-s)[:k]]


class IVF:
    def __init__(self, X, n_clusters=64, nprobe=4, seed=0, iters=8):
        self.X = normalize(np.asarray(X, dtype=np.float32))
        self.nprobe = nprobe
        rng = np.random.default_rng(seed)
        C = self.X[rng.choice(len(self.X), n_clusters, replace=False)].copy()
        for _ in range(iters):  # a few k-means steps (spherical)
            assign = np.argmax(self.X @ C.T, axis=1)
            for c in range(n_clusters):
                m = self.X[assign == c]
                if len(m):
                    C[c] = m.mean(0)
            C = normalize(C)
        self.C = C
        assign = np.argmax(self.X @ C.T, axis=1)
        self.lists = [np.flatnonzero(assign == c) for c in range(n_clusters)]

    def search(self, q, k=10):
        q = normalize(np.asarray(q, dtype=np.float32).reshape(1, -1))[0]
        probe = np.argsort(-(self.C @ q))[: self.nprobe]
        cand = np.concatenate([self.lists[c] for c in probe])
        if len(cand) == 0:
            return np.array([], dtype=int)
        s = self.X[cand] @ q
        return cand[np.argsort(-s)[:k]]


def recall_at_k(truth, found) -> float:
    return len(set(map(int, truth)) & set(map(int, found))) / max(len(truth), 1)
