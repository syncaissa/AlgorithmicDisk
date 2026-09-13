# demo2 — Scaling check with 1000 files

**Claim being checked** (paper §5.6): one thousand telemetry files of ~105 MB each (~105 GB in all) can be stored as
one small generator program plus a catalogue of one thousand Manifests (a parameter record, a length and a sha-256
each), and every file regenerated from that catalogue alone, byte-for-byte.

**What is on the disk after the run** — and nothing else:

```
AlgorithmicDisk/
  engines/ANET-PROC-telemetry-q20-v1.py   the generator program (about 1 KB): a Lorenz system integrated in
                                          Q20 fixed-point INTEGER arithmetic, so it is bit-identical on any machine
  catalogue.json                          1000 Manifests: {id, gid, seed (7 parameters), n_bytes, sha256}
```

## Reproduce it (Ubuntu, any machine with Python 3.8+ and NumPy)

```bash
cd demo2
python3 run_demo2.py all 1000        # ~2 hours on one core; see below to run it in the background
```
That single command runs the three phases and writes `REPORT.md`, `summary.json` and `results.json`:

| phase | what happens | how long |
|---|---|---|
| A `write` | generates each of the 1000 files once, records its length and sha-256 in its Manifest, **discards the file** (this is the moment at which, in a real system, the file existed and was written) | ~1 h (≈3.7 s per file) |
| B `read`  | regenerates every file **from the catalogue alone** and checks length and sha-256 against the Manifest | ~1 h |
| C `summary` | a metadata query over the catalogue (no regeneration) + the report | seconds |

The phases can also be run separately (`write`, `read`, `summary`) and the run is **interrupt-safe**: `catalogue.json`
and `results.json` are saved every 10 files, so re-running `all 1000` resumes where it stopped. To run in the background:
```bash
nohup python3 -u run_demo2.py all 1000 > demo2.log 2>&1 &
tail -f demo2.log
```

## What you should see

`REPORT.md` begins with the three headline numbers:
```
Total size of the 1000 files: 104,471,513,533 bytes (104.47 GB)
Total size of what is stored (Seeds 123,907 B + generator program 1,060 B + Manifest overhead) = whole disk folder: 317,228 bytes
Efficiency ratio: stored / original = 3.04e-06  (the original is 329,326 times larger than what is stored)
```
followed by `1000/1000 bit-exact, 0 failed`, the write and read times, and the catalogue query result. The exact byte
counts are in `summary.json`; the per-file regeneration times and digests in `results.json`. Your digests must equal
ours (`catalogue.json` is shipped): the generator is integer-only, so a different CPU, OS or Python version produces the
same bytes. If a digest differs, please report it with `python3 -c "import platform,sys;print(platform.platform(),sys.version)"`.

## Notes for a careful reader

* **No 105 MB file is ever kept.** ~105 GB of telemetry exists only while each file is being hashed; free disk needed: a
  few megabytes. To *see* a file, regenerate one: 
  `python3 -c "import json,importlib.util as u;s=u.spec_from_file_location('g','AlgorithmicDisk/engines/ANET-PROC-telemetry-q20-v1.py');g=u.module_from_spec(s);s.loader.exec_module(g);m=json.load(open('AlgorithmicDisk/catalogue.json'))['manifests'][0];open(m['id'],'wb').write(g.telemetry_csv(m['seed']))"`
* **What this proves and what it does not.** It checks the scaling claim end to end: a thousand distinct objects,
  independent Manifests, one shared program, ratio ~10⁻⁶, every object bit-exact, and a catalogue that is itself the
  archive's searchable index. It does *not* make the result less "by construction": the files are outputs of a program
  we wrote, so their tiny Footprint is provenance capture (paper §5.6). The paradigm's value for a real mission is
  proportional to the share of its data that is program output — all of simulation, rendering and synthetic products,
  none of raw sensor data.
* **Changing the experiment.** `records(n)` in `run_demo2.py` draws the 1000 parameter records deterministically
  (`numpy.random.default_rng(20260913)`); change the seed, the ranges, or `steps` (file length) and the run reproduces
  with your parameters. `python3 run_demo2.py all 20` is a five-minute version.
