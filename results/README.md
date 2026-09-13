# results/ — reference outputs of every run in the paper

These are the files the paper's tables were generated from (`paper/make_results.py` reads this folder). Re-running a
script writes a fresh copy next to the script; `python3 compare.py <name>_results.json` diffs it against the copy here.

| file | produced by | paper | contents |
|---|---|---|---|
| `predictive_results.json`, `predictive.log` | `predictive.py` | Table 4, Figure 4 | per-(hidden, M) Seed bits, held-out Seed bits, Footprint/object, break-even M* |
| `designA_long.log` | `amortize.py` | §5.2 negative result | teacher-forced loss vs free-running exactness of the latent-Seed design |
| `realfile_results.json`, `realfile.log` | `realfile.py` | Table 5 | per file and mode: Seed bytes, bits/byte, gzip/bz2/xz, write/read seconds, sha-256 check |
| `n8_results.json`, `n8.log` | `n8_bigengine.py` | Table 6 | 148 KB engine on pg12/pg84: float-stream and fixed/C Seeds vs gzip/bz2/xz; training curve in the log |
| `baselines_results.json`, `baselines.log` | `baselines.py` | Table 7 | PPMd o8/o16, brotli, zstd+trained dictionary, and xz/PPMd given the 12 MB corpus, on the same bytes |
| `multiengine_results.json`, `multiengine.log` | `multiengine.py` | Table 8 | engine chosen per object, Seed, ratio, total with engine, read seconds; the log shows every engine tried |
| `libwork_results.json`, `libwork.log` | `libwork.py` | §5.4 (curve) | library-document ratio at each 15-minute training checkpoint |
| `editaware_results.json`, `editaware.log` | `editaware.py` | Table 9 | four versions: engine, base, edits, Seed bytes, ratio; totals vs xz |
| `retrieval_results.json`, `retrieval.log` | `retrieval.py` | Table 10 | per object: LIB/delta/retrieval candidates, runs, chosen engine, Seed, ratio |
| `determinism_results.json` | `determinism.py` | Table 11 | 18 (engine, variant, object) rows: same Seed? tables identical? regenerates reference? |
| `fastcoder_results.json` | `fastcoder.py` | Table 12 | C vs Python Seed identity; read seconds and KB/s by thread count |
| `n2b_results.json`, `n2b.log` | `n2b_deepdata.py` | Table 13 | telemetry and DEM: Seed, program bytes, ratio, gzip/xz, regeneration seconds |
| `demo2_summary.json`, `demo2_REPORT.md` | `demo2/run_demo2.py` | §5.6 scaling check | 1000-file totals (copies of `demo2/summary.json`, `demo2/REPORT.md`) |

Table 14 (ten files) is read from `demo/report.json`. All timings are from one Intel Xeon Platinum 8259CL core at
2.5 GHz with two hardware threads, 8 GB RAM, no GPU (the paper's "test machine"). Read times in `editaware_results.json`
were measured while a training job shared that core (see the caption of Table 9).
