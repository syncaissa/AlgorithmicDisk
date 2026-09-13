# Algorithmic Storage — reference implementation

Code for the paper *"Algorithmic Storage: The AlgorithmicDisk on the Footprint–Compute–Horizon Triangle"*
(M. K. Patil, Syncaissa Systems Inc., 2026).

Paradigm **Algorithmic Storage** → device **AlgorithmicDisk** → engines **AlgorithmicNET** (neural, predictive),
**ANET-PROC** (procedural), **ANET-DELTA** (edit-aware), **ANET-RET** (retrieval index). A stored object is a
Regeneration Manifest (engine id + Seed + length + sha-256); reading it regenerates the object bit-for-bit.
Weights are trained once per domain and never change when objects are written — only Seeds are added.

Pure Python + NumPy. Same code runs on GPU with CuPy (`--device cuda`). No other dependency.

**Step-by-step Ubuntu instructions with the expected output of every test: [INSTALL_UBUNTU.md](INSTALL_UBUNTU.md).**
**Which script produced which table, and what will match exactly: [PAPER_TO_CODE.md](PAPER_TO_CODE.md).**

Layout — every folder has its own README:
`./` the experiment scripts, shipped engines (`engine_*.json`) and disks (`*.adisk`) · `results/` the reference outputs the
paper's tables were generated from · `paper/` LaTeX source, `make_results.py` (JSON → tables) and the current PDF ·
`native/` the C fixed-point engine and coder · `demo/` ten real files through one disk (§5.7) · `demo2/` the
1000-file scaling check (§5.6).

## 1. Verify the paper's results without training (5 minutes)

```bash
pip install -r requirements.txt
./get_data.sh                       # public-domain corpus (Project Gutenberg), ~3.5 MB
python3 verify.py                   # opens every shipped .adisk, regenerates every object, checks sha-256
```
Expected: `24/24 objects regenerated bit-exactly`, with the Seed sizes and ratios of Tables 2, 5, 8, 9, 10 and 13 of the paper
(e.g. `lorenz.csv` 1,153,112 B from a 118 B Seed = 0.00010×; `pg12.txt` on the retrieval disk from 10 B; `rev1_60words.txt`
from 390 B; `image.png` verbatim 1.000×). Reads are slow (pure-Python range coder, ~6 KB/s); that is the Horizon side of
the triangle, not a limitation of the method.

Shipped engines (int8, trained once): `engine_books.json` (English, 55,720 B), `engine_code.json` (Python source),
`engine_lib.json` / `engine_lib_phase2.json` (library engine, hidden 384), `engine_big.json` (English, hidden 256, 148 KB; below xz and bz2 on unseen books, above PPMd — Table 7). Shipped disks: `demo.adisk`, `realfile.adisk`, `multi.adisk`,
`editaware.adisk`, `retrieval.adisk`, `deepdata.adisk` (895 B for 239 MB of objects). Reference outputs of every run are in `results/`.

## 2. Reproduce each experiment (order of the paper)

| paper section | command | time (2 CPU cores) | produces |
|---|---|---|---|
| §5.1 first object + Seed sweep | `python3 algorithmic_storage.py demo --sweep` | 2 min | `demo.adisk` |
| §5.2 amortisation, latent-Seed design (negative) | `python3 -c "..."` see `results/designA_long.log` header, or `python3 amortize.py` | 1 h | `amortize_results.json` |
| §5.2 amortisation, predictive design | `python3 predictive.py --epochs 60` | 40 min | `predictive_results.json` |
| §5.3 real files | `python3 realfile.py` (trains the English engine, 5 min, unless `engine_books.json` exists) | 8 min | `realfile.adisk`, `restored/` |
| §5.4 domain engines | `python3 multiengine.py` (trains CODE 5 min + LIB 40 min unless the engine files exist) | 10 min with shipped engines | `multi.adisk` |
| §5.4 memorisation curve | `python3 libwork.py --minutes 150 --every 15` | 150 min | `libwork_results.json` |
| §5.4 edit-aware Seeds | `python3 editaware.py` | 5 min | `editaware.adisk` |
| §5.4 retrieval engine | `python3 retrieval.py` | 8 min | `retrieval.adisk` |
| §5.5 determinism | `python3 determinism.py --device cpu` | 6 min | `determinism_results.json` |
| §5.5 compiled engine | `python3 fastcoder.py` | 8 min | `fastcoder_results.json` (needs gcc) |
| §5.6 100 MB deep data | `python3 n2b_deepdata.py` | 4 min | `deepdata.adisk`, two regenerable files (239 MB, delete after) |
| §5.3 engine that beats xz | `python3 n8_bigengine.py` | 90 min training (skipped if `engine_big.json` exists) + 2 min eval | `n8_results.json` |
| §5.3 stronger baselines (PPMd, brotli, zstd+dict, same-corpus prior) | `python3 baselines.py` (needs `7z`, `brotli`, `zstd`: `sudo apt-get install p7zip-full brotli zstd`) | 2 min | `baselines_results.json` |

Delete an `engine_*.json` file to force retraining from scratch; results will differ slightly (random init, float order)
but every object must still regenerate bit-exactly — the sha-256 check is the invariant.

## 3. Storing your own file

```bash
python3 algorithmic_storage.py write --text "The fox ran over the lazy dog" --name p1 --q-bits 4   # gradient Seed search, untrained engine
python3 algorithmic_storage.py read  --name p1
python3 realfile.py --files path/to/your.txt                                                       # closed-form Seed, shared English engine
```

## 3b. Compiled engine (paper §5.5)
`native/anet_fixed.c` is the fixed-point engine and range coder in C with OpenMP over blocks; `fastcoder.py` builds it
(`gcc -O3 -march=native -fopenmp`) and cross-checks it against the Python engine — Seeds are byte-identical in both
directions. ~10× faster than the Python path; `python3 fastcoder.py` runs the check and the Horizon-vs-threads timing
(E5). Needs `gcc` (`sudo apt-get install -y build-essential`).

## 4. Determinism (paper §5.5)
The float32 engine is **not** portable: change the summation order or precision of a dot product (a different BLAS,
core count, or a GPU) and its Seeds no longer regenerate (6/6 cross-variant cases fail). The **fixed-point engine**
(`fixedpoint.py`, integer-only) is bit-identical under every variant (9/9) at the same Footprint.

```bash
python3 determinism.py --device cpu     # 18 (engine, variant, object) cases against determinism_reference.json
python3 determinism.py --device cuda    # on a CUDA machine with cupy: compares the GPU against the CPU reference
```
`determinism_reference.json` holds the Seeds and table hashes produced on the original CPU, so a run on any other
machine compares against the *original*, not against itself. **We have not yet run this on a GPU** — if you have one,
this is the most valuable test you can contribute: please report the last line of the output.

## Demo: ten files through one disk (paper §5.7)
`demo/` holds `InputFiles/` (10 files, 1-5 MB: 4 books, 3 PDFs, 3 videos), `AlgorithmicDisk/` (engines, Seeds, Manifests —
everything needed to regenerate), `OutputFiles/` (regenerated from the disk folder alone) and `REPORT.md` with the
byte-for-byte comparison. `python3 demo/run_demo.py read` regenerates the outputs again; `demo/README.md` has the details.
`OutputFiles/` is regenerable and `InputFiles/` is rebuilt by `demo/make_inputs.py`, so both can be omitted from a clone.

## Scaling check with 1000 files (paper §5.6)
`demo2/` stores one thousand ~105 MB telemetry files as a catalogue of 1000 Manifests plus one generator program, then
regenerates and verifies every file from the catalogue alone (`python3 demo2/run_demo2.py all 1000`, ~2 h; interrupt-safe).
Outputs: `demo2/AlgorithmicDisk/catalogue.json`, `results.json`, `summary.json`, `REPORT.md`.

## Attribution
Building blocks used, with credit (also cited in the paper): arithmetic coding (Witten, Neal & Cleary 1987) in
Subbotin's carry-less range-coder form (1999); gear-hash content-defined chunking after FastCDC (Xia et al. 2016);
Ratcliff–Obershelp matching via Python `difflib`; teacher forcing (Williams & Zipser 1989), straight-through estimator
(Bengio et al. 2013), Adam (Kingma & Ba 2015); integer-only inference (Jacob et al. 2018); Lorenz (1963), fractal value
noise (Perlin 1985), SplitMix64 mixing (Steele, Lea & Flood 2014); NumPy (Harris et al. 2020); texts from Project
Gutenberg (public domain). Prior neural compressors: cmix (Knoll), NNCP (Bellard), Delétang et al. 2023.

## Files
`algorithmic_storage.py` demo engine + gradient Seed Discovery · `amortize.py` Design A · `predictive.py` Design B + range coder ·
`realfile.py` whole files, stream/blocks modes, verbatim escape · `multiengine.py` multi-engine disk, procedural engine ·
`editaware.py` delta engine · `retrieval.py` retrieval index engine · `libwork.py` memorisation curve · `n2b_deepdata.py` 100 MB deep-data objects · `n8_bigengine.py` larger English engine (clean split) · `baselines.py` PPMd/brotli/zstd baselines on the same books · `fixedpoint.py` integer-only engine · `determinism.py` cross-device harness · `fastcoder.py` + `native/anet_fixed.c` compiled engine+coder (C, OpenMP) · `verify.py` one-command check ·
`get_data.sh` corpus · `run_tests.sh` quick suite · `compare.py` diff against reference · `image.png` the shallow test object · `results/` reference outputs and logs · `paper/` LaTeX source and table generator · `PAPER_TO_CODE.md` table-by-table map.
