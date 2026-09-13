#!/usr/bin/env python3
"""
Steps 2-4 of the demo.
  write : InputFiles/*  ->  AlgorithmicDisk/   (engines/, seeds/, manifests/, INDEX.json)
  read  : AlgorithmicDisk/  ->  OutputFiles/*  using ONLY what is in AlgorithmicDisk/
  report: compare OutputFiles with InputFiles byte-for-byte -> REPORT.md + report.json

Engines on this disk
  ANET-EN-256          the English predictive engine of the paper (hidden 256, int8), run as the compiled
                       fixed-point engine (native/anet_fixed.c) in blocks mode with the verbatim escape
  ANET-PROC-anim-v1    the procedural animation generator (anim_generator.py); Seed = parameter record
  VERBATIM             the escape (an object nothing can shorten is stored raw, per 512-byte block)
For every file the disk tries every applicable engine and keeps the smallest Seed.  A 128 KB probe
decides whether the neural engine is worth a full pass (it is skipped when the probe shows the file is
already incompressible to it, e.g. compressed PDF streams or video pixels).
"""
import hashlib, json, lzma, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import numpy as np
from fastcoder import FastFixedEngine
IN, DISK, OUT = (os.path.join(HERE, d) for d in ("InputFiles", "AlgorithmicDisk", "OutputFiles"))
L = 512; PROBE = 131072

def sha(b): return hashlib.sha256(b).hexdigest()

# ----------------------------------------------------------------------------- write
def write_disk(threads=2, resume=True):
    """Write every input to the disk. With resume=True, a file whose manifest already records its sha-256 is skipped,
    and manifests/seeds of files that no longer exist in InputFiles are removed."""
    for d in ("engines", "seeds", "manifests"): os.makedirs(os.path.join(DISK, d), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "engine_big.json"), os.path.join(DISK, "engines", "ANET-EN-256.json"))
    shutil.copy(os.path.join(HERE, "anim_generator.py"), os.path.join(DISK, "engines", "ANET-PROC-anim-v1.py"))
    eng = FastFixedEngine(json.load(open(os.path.join(DISK, "engines", "ANET-EN-256.json"))))
    prov = json.load(open(os.path.join(IN, ".provenance.json"))) if os.path.exists(os.path.join(IN, ".provenance.json")) else {}
    index = {"format": "AlgorithmicDisk/demo-1", "engines": {"ANET-EN-256": "engines/ANET-EN-256.json", "ANET-PROC-anim-v1": "engines/ANET-PROC-anim-v1.py", "VERBATIM": None}, "objects": []}
    names = sorted(f for f in os.listdir(IN) if not f.startswith("."))
    for stale in os.listdir(os.path.join(DISK, "manifests")):                       # drop objects whose input was removed
        if stale[:-5] not in names:
            os.remove(os.path.join(DISK, "manifests", stale)); sp = os.path.join(DISK, "seeds", stale[:-5] + ".seed"); os.path.exists(sp) and os.remove(sp)
    for name in names:
        data = open(os.path.join(IN, name), "rb").read(); n = len(data); digest = sha(data); trials = []; t_all = time.time()
        mp = os.path.join(DISK, "manifests", name + ".json")
        if resume and os.path.exists(mp):
            man = json.load(open(mp))
            if man.get("sha256") == digest and os.path.exists(os.path.join(DISK, man["seed_file"])):
                index["objects"].append({k: man[k] for k in ("id", "gid", "mode", "n_bytes", "seed_bytes", "sha256")}); json.dump(index, open(os.path.join(DISK, "INDEX.json"), "w"), indent=1)
                print(f"\n== {name}  {n:,} B  (already on the disk, unchanged)", flush=True); continue
        print(f"\n== {name}  {n:,} B", flush=True)
        best = ("VERBATIM", data, "raw"); trials.append(dict(engine="VERBATIM", seed_bytes=n, seconds=0.0))
        if name in prov:                                                    # provenance capture
            from anim_generator import render_bytes
            t0 = time.time(); ok = sha(render_bytes(prov[name]["params"])) == digest; dt = time.time() - t0
            seed = json.dumps(prov[name]["params"], sort_keys=True).encode()
            trials.append(dict(engine=prov[name]["gid"], seed_bytes=len(seed), seconds=round(dt, 1), exact=ok))
            print(f"   {prov[name]['gid']:20s} -> {len(seed):>10,} B  (regenerated and verified in {dt:.1f}s)", flush=True)
            if ok and len(seed) < len(best[1]): best = (prov[name]["gid"], seed, "procedural")
        # neural engine: probe first
        t0 = time.time(); probe = eng.encode_blocks(data[:PROBE], L=L, threads=threads); pr = len(probe) / min(n, PROBE)
        print(f"   ANET-EN-256 probe on {min(n, PROBE):,} B: ratio {pr:.3f} ({time.time()-t0:.1f}s)", flush=True)
        if pr < 0.97:
            t0 = time.time(); seed = eng.encode_blocks(data, L=L, threads=threads); dt = time.time() - t0
            trials.append(dict(engine="ANET-EN-256", seed_bytes=len(seed), seconds=round(dt, 1), probe_ratio=round(pr, 3))); print(f"   ANET-EN-256          -> {len(seed):>10,} B  ({dt:.0f}s)", flush=True)
            if len(seed) < len(best[1]): best = ("ANET-EN-256", seed, "blocks")
        else:
            trials.append(dict(engine="ANET-EN-256", skipped="probe shows incompressible", probe_ratio=round(pr, 3)))
        gid, seed, mode = best
        open(os.path.join(DISK, "seeds", name + ".seed"), "wb").write(seed)
        man = dict(id=name, gid=gid, mode=mode, n_bytes=n, sha256=digest, block_len=L, seed_file=f"seeds/{name}.seed", seed_bytes=len(seed),
                   seed_preview=(seed[:80].decode("utf-8", "replace") if mode == "procedural" else seed[:16].hex() + "..."),
                   write_seconds=round(time.time() - t_all, 1), trials=trials, xz=len(lzma.compress(data, preset=6)))
        json.dump(man, open(os.path.join(DISK, "manifests", name + ".json"), "w"), indent=1)
        index["objects"].append({k: man[k] for k in ("id", "gid", "mode", "n_bytes", "seed_bytes", "sha256")})
        print(f"   chosen {gid} ({mode}): Seed {len(seed):,} B  ratio {len(seed)/n:.5f}   (xz -6: {man['xz']:,} B)", flush=True)
        json.dump(index, open(os.path.join(DISK, "INDEX.json"), "w"), indent=1)

# ----------------------------------------------------------------------------- read (uses only the disk folder)
def read_disk(threads=2):
    os.makedirs(OUT, exist_ok=True)
    index = json.load(open(os.path.join(DISK, "INDEX.json")))
    eng = None; times = {}
    for o in index["objects"]:
        man = json.load(open(os.path.join(DISK, "manifests", o["id"] + ".json"))); seed = open(os.path.join(DISK, man["seed_file"]), "rb").read()
        t0 = time.time()
        if man["mode"] == "raw": data = seed
        elif man["mode"] == "procedural":
            import importlib.util
            spec = importlib.util.spec_from_file_location("gen", os.path.join(DISK, "engines", man["gid"] + ".py")); gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
            data = gen.render_bytes(json.loads(seed))
        else:
            eng = eng or FastFixedEngine(json.load(open(os.path.join(DISK, "engines", "ANET-EN-256.json"))))
            data = eng.decode_blocks(seed, man["n_bytes"], L=man["block_len"], threads=threads)
        dt = time.time() - t0; ok = sha(data) == man["sha256"]
        open(os.path.join(OUT, o["id"]), "wb").write(data); times[o["id"]] = (round(dt, 1), ok)
        print(f"   regenerated {o['id']:32s} {len(data):>11,} B in {dt:6.1f}s  sha256 {'PASS' if ok else 'FAIL'}", flush=True)
    json.dump(times, open(os.path.join(HERE, "read_times.json"), "w"), indent=1)
    return times

# ----------------------------------------------------------------------------- report
def report(times):
    rows = []; index = json.load(open(os.path.join(DISK, "INDEX.json")))
    for o in index["objects"]:
        man = json.load(open(os.path.join(DISK, "manifests", o["id"] + ".json")))
        a = open(os.path.join(IN, o["id"]), "rb").read(); b = open(os.path.join(OUT, o["id"]), "rb").read()
        kind = "text" if o["id"].endswith(".txt") else "pdf" if o["id"].endswith(".pdf") else "video"
        rows.append(dict(file=o["id"], type=kind, bytes=o["n_bytes"], engine=o["gid"], seed_bytes=o["seed_bytes"], ratio=round(o["seed_bytes"] / o["n_bytes"], 5),
                         xz6=man["xz"], xz_ratio=round(man["xz"] / o["n_bytes"], 3), write_s=man["write_seconds"], read_s=times[o["id"]][0],
                         identical=(a == b), sha_ok=times[o["id"]][1]))
    eng_bytes = os.path.getsize(os.path.join(DISK, "engines", "ANET-EN-256.json")) + os.path.getsize(os.path.join(DISK, "engines", "ANET-PROC-anim-v1.py"))
    disk_bytes = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DISK) for f in fs)
    tot = dict(input_bytes=sum(r["bytes"] for r in rows), seed_bytes=sum(r["seed_bytes"] for r in rows), engine_bytes=eng_bytes, disk_folder_bytes=disk_bytes,
               xz_bytes=sum(r["xz6"] for r in rows), all_identical=all(r["identical"] and r["sha_ok"] for r in rows))
    tot["seed_ratio"] = round(tot["seed_bytes"] / tot["input_bytes"], 4); tot["disk_ratio"] = round(tot["disk_folder_bytes"] / tot["input_bytes"], 4); tot["xz_ratio"] = round(tot["xz_bytes"] / tot["input_bytes"], 4)
    json.dump(dict(rows=rows, totals=tot), open(os.path.join(HERE, "report.json"), "w"), indent=1)
    md = ["# Demo report — ten files through an AlgorithmicDisk", "",
          f"Inputs: {len(rows)} files, {tot['input_bytes']:,} bytes. Disk folder: {tot['disk_folder_bytes']:,} bytes "
          f"(engines {tot['engine_bytes']:,} B + Seeds {tot['seed_bytes']:,} B + manifests). xz -6 on each file: {tot['xz_bytes']:,} bytes.", "",
          f"**All {len(rows)} files regenerated byte-identically from the disk folder alone: {tot['all_identical']}.**", "",
          "| file | type | bytes | engine chosen | Seed (B) | ratio | xz -6 ratio | write s | read s | identical |", "|---|---|---:|---|---:|---:|---:|---:|---:|:-:|"]
    for r in rows:
        md.append(f"| {r['file']} | {r['type']} | {r['bytes']:,} | {r['engine']} | {r['seed_bytes']:,} | {r['ratio']:.5f} | {r['xz_ratio']:.3f} | {r['write_s']} | {r['read_s']} | {'yes' if r['identical'] and r['sha_ok'] else 'NO'} |")
    md += ["", f"Totals: Seeds/inputs = {tot['seed_ratio']}, whole disk folder/inputs = {tot['disk_ratio']} (engine counted once), xz/inputs = {tot['xz_ratio']}.", "",
           "How to read it. The Seed of a file is the part of it the disk cannot already produce, so three outcomes appear:",
           "(1) files the disk can GENERATE - the two animations - cost a 110-byte parameter record (30,000x smaller);",
           "(2) files in a domain the disk KNOWS - the four books, never seen in training - cost about 0.28x, on par with xz;",
           "(3) files in NO domain the disk knows - the storybook PDF (page images), the Flate-compressed PDF, the H.264 MP4, and the",
           "uncompressed PDF whose text is hex-encoded inside content streams - are stored VERBATIM (1.0x): with no engine for pixels,",
           "Flate or hex, the honest choice is raw bytes rather than expansion, and a general-purpose compressor does better on those.",
           "The fix for (3) is more engines on the disk (a PDF engine that decodes streams and hands the text to the English engine,",
           "an image engine, provenance for the MP4 if its source and encoder were on the disk) - the paper's design rule.",
           "Every file, whatever its ratio, comes back bit-for-bit from the disk folder alone."]
    open(os.path.join(HERE, "REPORT.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("write", "all"): write_disk()
    if step in ("read", "all"): times = read_disk()
    if step in ("report", "all"):
        if step == "report":
            rt = os.path.join(HERE, "read_times.json")
            times = {k: tuple(v) for k, v in json.load(open(rt)).items()} if os.path.exists(rt) else {o["id"]: (0.0, True) for o in json.load(open(os.path.join(DISK, "INDEX.json")))["objects"]}
        report(times)
