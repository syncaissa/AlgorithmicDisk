# Step-by-step: running the tests on Ubuntu (22.04 / 24.04)

Everything below was run on a 2-core, 8 GB Ubuntu 22.04 machine with no GPU. Times are for that machine.

## Step 1 — system packages
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip curl git build-essential   # gcc is needed only for fastcoder.py
sudo apt-get install -y p7zip-full brotli zstd                                    # only for baselines.py (PPMd / brotli / zstd)
python3 --version        # 3.10 or newer
```

## Step 2 — get the code
```bash
git clone <this repository>
cd <repository>/forGithub        # the folder that contains verify.py
```

## Step 3 — Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # numpy only
```
GPU (optional, CUDA 12): `pip install cupy-cuda12x`, then add `--device cuda` to any command below.

## Step 4 — download the public-domain corpus (~3.5 MB)
```bash
./get_data.sh
```
Creates `alice.txt` and `books/pg12.txt pg1342.txt pg84.txt pg2701.txt pg98.txt`.
If your network blocks gutenberg.org, download the six files by hand from
`https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt` and place them as above.

## Step 5 — verify the paper's results WITHOUT training (about 6 minutes)
```bash
python3 verify.py
```
Expected last line: `24/24 objects regenerated bit-exactly`.
Every line shows an object, the engine that regenerated it, its Seed size and ratio, and PASS.
Key lines to look for:
```
lorenz.csv        1,153,112 B  ANET-PROC  Seed     118 B  ratio 0.00010  PASS
pg12.txt            176,754 B  ANET-RET   Seed      10 B  ratio 0.00006  PASS
rev1_60words.txt    176,941 B  ANET-DELTA Seed     390 B  ratio 0.00220  PASS
pg12.txt:stream     196,534 B             Seed  65,093 B  ratio 0.3312   PASS
image.png            88,791 B  VERBATIM   Seed  88,791 B  ratio 1.00000  PASS
```
A FAIL on any line means the shipped engine and your machine's floating point disagree — please report
your CPU/BLAS (`python3 -c "import numpy; numpy.show_config()"`); see "Determinism" in README.md.

## Step 6 — reproduce the experiments one by one
Run them in this order; each writes its own `*.adisk` / `*_results.json` in the current folder.
Reference outputs to compare against are in `results/`.

| # | paper | command | time | expected |
|---|---|---|---|---|
| 6.1 | §5.1 | `python3 algorithmic_storage.py --device cpu demo --sweep` | 2 min | `Round trip exact: True`; sweep: 256 dims × 4 bits found in ~300 iterations |
| 6.2 | §5.2 predictive | `python3 predictive.py --device cpu --epochs 60` | 40 min | `train exact 100.0%` on every row; footprint/obj < 232 bits at M = 2000, 4000 |
| 6.3 | §5.2 latent-Seed (negative) | `python3 amortize.py --device cpu --epochs 300 --dims 32 --M-sweep 500` | 30 min | bit-acc about 93–95 %, exact about 0 % (this is the reported negative result) |
| 6.4 | §5.3 | `python3 realfile.py --device cpu` | 8 min (+5 min training if `engine_books.json` is deleted) | `pg12.txt stream: Seed 65,093 B`, all `sha256 PASS`; restored files in `restored/` |
| 6.5 | §5.4 | `python3 multiengine.py --device cpu` | 10 min with shipped engines; +45 min if `engine_code.json` / `engine_lib.json` are deleted | `lorenz.csv ... ratio 0.00010x`, `image.png ... VERBATIM` |
| 6.6 | §5.4 deltas | `python3 editaware.py` | 5 min | `rev1_60words.txt ... Seed 390 B ratio 0.00220x` |
| 6.7 | §5.4 retrieval | `python3 retrieval.py` | 8 min | `pg12.txt ... Seed 10 B ratio 0.00006x` |
| 6.8 | §5.4 memorisation | `python3 libwork.py --device cpu --minutes 150 --every 15` | 150 min | `libwork_results.json` with a checkpoint every 15 min; ratio falls from 0.140 toward 0.10 |
| 6.9 | §5.5 | `python3 determinism.py --device cpu` | 6 min | `12/18 ... bit-identical`: float32 fails chunked/f64, fixed-point passes all |
| 6.11 | §5.5 compiled | `python3 fastcoder.py` | 8 min | `C vs Python fixed-point Seeds identical: True`; read ≈ 60 KB/s per core (vs 6 KB/s Python); threads 1/2/4 timings |
| 6.12 | §5.6 | `python3 n2b_deepdata.py` | 4 min | `telemetry.csv ... ratio 1.17e-06`, `planet_dem.u16 ... exact True`; delete the two large files afterwards |
| 6.13 | §5.3 big engine | `python3 n8_bigengine.py` | 2 min with shipped `engine_big.json` (90 min if deleted) | `pg12.txt ... beats xz: True beats bz2: True`, same for pg84 |
| 6.10 | §5.5, GPU | `pip install cupy-cuda12x && python3 determinism.py --device cuda` | 6 min | on a CUDA machine: the GPU rows should read PASS for fixed-point; float32 rows are expected to FAIL |

Or run 6.1, 6.4–6.7 and 6.9 in one go (about 40 minutes; skips the long training runs):
```bash
./run_tests.sh
```

## Step 7 — check a run against the reference
```bash
python3 compare.py retrieval_results.json      # any *_results.json
```
Seed sizes match exactly when the shipped engines are used. If you retrained an engine, sizes differ by a few
percent (random initialisation) but `exact` must be `True` for every object — bit-exact regeneration is the invariant.

## Troubleshooting
* **`ModuleNotFoundError: numpy`** — the venv is not active: `source .venv/bin/activate`.
* **`retrieval index mismatch: run get_data.sh`** — `alice.txt` or `books/pg12.txt` missing or modified.
* **Disk space** — the whole folder plus corpus is under 10 MB; training writes nothing large.
* **It is slow** — expected: the range coder is pure Python (about 6 KB/s). Reads of a 200 KB file take about 30 s.
  That time is the Horizon side of the Footprint–Compute–Horizon triangle; a compiled coder is listed as future work.
* **Different numbers after retraining** — see Step 7; only bit-exactness is guaranteed, not identical Seed sizes.
