#!/usr/bin/env python3
"""
Stronger baselines for the two held-out books of the N8 experiment (Table "The larger English
engine on two books it never saw").  A compression reviewer would ask for these, so they are in
the paper:

  * PPMd (7-Zip, -m0=PPMd, order 8 / 16, no training data)      -- the classical text compressor
  * brotli -q 11 (built-in 122 KB static dictionary)            -- a shipped "shared prior"
  * zstd --ultra -22 with a dictionary trained on the engine's  -- a learned shared prior of the
    own 12 MB training corpus, capped at the engine's byte size    same size as the engine
  * conditional baselines: compress(train + test) - compress(train) for xz -9e and PPMd o16,
    i.e. what a standard compressor achieves when handed the SAME 12 MB corpus the engine saw.

Exactly the same byte strings as n8_bigengine.py: gutenberg_body() of each file, and the
training corpus = concatenation of engine_big.json['train_books'] truncated to train_bytes.
Needs: 7z (p7zip-full), brotli, zstd on PATH.  Output: baselines_results.json.
"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realfile import gutenberg_body

TEST = ["books/pg12.txt", "books/pg84.txt"]

def sz(cmd_out): return len(cmd_out)
def run(cmd, inp=None): return subprocess.run(cmd, input=inp, capture_output=True, check=True).stdout
def seven_zip(path, order, mem, outp):
    if os.path.exists(outp): os.remove(outp)
    subprocess.run(["7z", "a", "-bso0", "-bsp0", f"-m0=PPMd:mem={mem}:o={order}", outp, path], check=True)
    return os.path.getsize(outp)

def main():
    ej = json.load(open("engine_big.json"))
    corpus = b"".join(gutenberg_body(f) for f in ej["train_books"])[: ej.get("train_bytes", 12_000_000)]
    res = dict(engine_bytes=148024, train_bytes=len(corpus), train_texts=len(ej["train_books"]), tests=[])
    n8 = json.load(open("n8_results.json"))
    with tempfile.TemporaryDirectory() as td:
        tr = os.path.join(td, "train.txt"); open(tr, "wb").write(corpus)
        ch = os.path.join(td, "chunks"); os.makedirs(ch)
        for i in range(0, len(corpus), 4096): open(os.path.join(ch, f"c{i//4096:05d}"), "wb").write(corpus[i:i+4096])
        dic = os.path.join(td, "dict")
        subprocess.run(["zstd", "-q", "--train", *[os.path.join(ch, f) for f in sorted(os.listdir(ch))],
                        f"--maxdict={res['engine_bytes']}", "-o", dic], check=True, capture_output=True)
        res["zstd_dict_bytes"] = os.path.getsize(dic)
        xz_train = sz(run(["xz", "-9e", "-c", tr]))
        ppmd_train = seven_zip(tr, 16, "1024m", os.path.join(td, "tr.7z"))
        for f in TEST:
            data = gutenberg_body(f); n = len(data); p = os.path.join(td, os.path.basename(f)); open(p, "wb").write(data)
            tt = os.path.join(td, "tt.txt"); open(tt, "wb").write(corpus + data)
            row = dict(file=os.path.basename(f), bytes=n,
                       xz9e=sz(run(["xz", "-9e", "-c", p])), bz2=sz(run(["bzip2", "-9", "-c", p])),
                       zstd22=sz(run(["zstd", "-q", "--ultra", "-22", "-c", p])),
                       zstd22_dict=sz(run(["zstd", "-q", "--ultra", "-22", "-D", dic, "-c", p])),
                       brotli11=sz(run(["brotli", "-q", "11", "--lgwin=24", "-c", p])),
                       ppmd_o8=seven_zip(p, 8, "256m", os.path.join(td, "o8.7z")),
                       ppmd_o16=seven_zip(p, 16, "256m", os.path.join(td, "o16.7z")),
                       xz9e_prior=sz(run(["xz", "-9e", "-c", tt])) - xz_train,
                       ppmd_o16_prior=seven_zip(tt, 16, "1024m", os.path.join(td, "tt.7z")) - ppmd_train)
            t = next(x for x in n8["tests"] if x["file"] == row["file"])
            row["engine_float"] = t["float_stream"]["seed"]; row["engine_fixed"] = t["fixed_blocks_C"]["seed"]
            res["tests"].append(row)
            print(f"{row['file']}: n={n:,} engine float {row['engine_float']:,} fixed {row['engine_fixed']:,} | xz9e {row['xz9e']:,} bz2 {row['bz2']:,} "
                  f"zstd22 {row['zstd22']:,} zstd22+dict {row['zstd22_dict']:,} brotli {row['brotli11']:,} PPMd-o8 {row['ppmd_o8']:,} PPMd-o16 {row['ppmd_o16']:,} | "
                  f"with 12 MB prior: xz {row['xz9e_prior']:,} PPMd {row['ppmd_o16_prior']:,}", flush=True)
    json.dump(res, open("baselines_results.json", "w"), indent=1); print("saved baselines_results.json")

if __name__ == "__main__":
    main()
