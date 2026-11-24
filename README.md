# vector-search-benchmark

Three nearest-neighbour search methods written from scratch with numpy, compared on recall@10 and query time against exact brute force: random-hyperplane LSH, and an inverted-file index (IVF, k-means clusters, probe the closest `nprobe` clusters). This is the idea behind vector databases such as FAISS, at toy scale.

## Run
```bash
pip install -r requirements.txt
python -m pytest -q      # 5 tests (exactness checks, recall monotonicity, self-query)
python benchmark.py
```

## Measured result (23 Nov 2026, 20,000 vectors, 64 dims, 200 queries, 2-CPU sandbox)
| method | recall@10 | ms/query |
|---|---|---|
| brute force (exact) | 1.00 | 0.181 |
| LSH (8 tables, 12 bits) | 0.465 | 0.070 |
| IVF nprobe=1 | 0.91 | 0.034 |
| IVF nprobe=4 | 1.00 | 0.073 |
| IVF nprobe=8 | 1.00 | 0.135 |

## Limits
Synthetic clustered Gaussian vectors, which suits IVF. Real embeddings behave differently. Small scale (brute force is already fast), timings are single-run and vary by machine, my LSH settings were not tuned, k-means runs only 8 iterations, no product quantization or graph (HNSW) index. Use FAISS or a vector database in production.

## Provenance
A learning and portfolio project.
