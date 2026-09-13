#!/usr/bin/env python3
"""
One-command verification for critics: open every shipped AlgorithmicDisk, regenerate every
object from its Seed with the shipped engines, and check the sha-256 recorded in the Manifest.
No training, no network access (except that retrieval.adisk rebuilds its chunk index from the
two library files, which get_data.sh downloads).

    python3 verify.py            # all disks
    python3 verify.py demo.adisk # one disk
"""
import base64, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from algorithmic_storage import get_backend, AlgorithmicDisk, bits_to_text
from realfile import Disk as RealDisk, engine_from_json, gutenberg_body
from retrieval import RetrievalDisk

def load_multi(path, xp):
    d = json.load(open(path)); disk = RetrievalDisk(path); disk.d = d
    for gid, g in d["generators"].items():
        if g["kind"] == "predictive": disk.nets[gid] = engine_from_json(xp, g)
        elif g["kind"] == "retrieval":
            docs = [(n, gutenberg_body(n if os.path.exists(n) else os.path.join("books", n))) for n in g["docs"]]
            disk.add_retrieval(docs); assert disk.ret_gid == gid, "retrieval index mismatch: run get_data.sh"
    return disk

def main():
    xp, dev = get_backend("cpu"); disks = sys.argv[1:] or ["demo.adisk", "realfile.adisk", "multi.adisk", "editaware.adisk", "retrieval.adisk", "deepdata.adisk"]
    total = ok_n = 0
    for path in disks:
        if not os.path.exists(path): print(f"-- {path}: missing"); continue
        print(f"== {path}")
        if path.endswith("demo.adisk"):
            disk = AlgorithmicDisk(path)
            for m in disk.d["manifests"]:
                bits, ok, rm = disk.read(xp, m["id"]); total += 1; ok_n += ok
                print(f"   {m['id']:24s} {rm['n_bits']:>10} bits  Seed {len(base64.b64decode(rm['seed_b64'])):>7} B  -> {bits_to_text(bits)!r}  {'PASS' if ok else 'FAIL'}")
            continue
        if path.endswith("deepdata.adisk"):
            from n2b_deepdata import ENGINES
            for m in json.load(open(path))["manifests"]:
                t0 = time.time(); data = ENGINES[m["gid"]](m["seed"]); dt = time.time() - t0
                ok = hashlib.sha256(data).hexdigest() == m["sha256"]; total += 1; ok_n += ok; del data
                print(f"   {m['id']:24s} {m['n_bytes']:>11,} B  {m['gid'][:20]:20s} Seed {len(json.dumps(m['seed'], sort_keys=True)):>6} B  ratio {len(json.dumps(m['seed'], sort_keys=True))/m['n_bytes']:.2e}  regen {dt:5.1f}s  {'PASS' if ok else 'FAIL'}")
            continue
        if path.endswith("realfile.adisk"):
            disk = RealDisk(path)
            for m in disk.d["manifests"]:
                t0 = time.time(); data, ok, dt, rm = disk.read(xp, m["id"]); total += 1; ok_n += ok
                print(f"   {m['id']:34s} {rm['n_bytes']:>9,} B  Seed {len(base64.b64decode(rm['seed_b64'])):>9,} B  ratio {len(base64.b64decode(rm['seed_b64']))/rm['n_bytes']:.4f}  read {dt:5.1f}s  {'PASS' if ok else 'FAIL'}")
            continue
        disk = load_multi(path, xp)
        for m in disk.d["manifests"]:
            data, ok, dt, rm = disk.read(m["id"]); total += 1; ok_n += ok
            print(f"   {m['id']:24s} {rm['n_bytes']:>9,} B  {rm['gid'][:14]:14s} Seed {rm['seed_bytes']:>9,} B  ratio {rm['seed_bytes']/rm['n_bytes']:.5f}  read {dt:5.1f}s  {'PASS' if ok else 'FAIL'}")
    print(f"\n{ok_n}/{total} objects regenerated bit-exactly")
    sys.exit(0 if ok_n == total else 1)

if __name__ == "__main__":
    main()
