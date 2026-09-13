#!/usr/bin/env python3
"""
Proof by 1000 files.
A thousand telemetry files of ~105 MB each are stored on an AlgorithmicDisk that holds ONE 842-byte
generator program and a catalogue of 1000 Manifests (parameter record + length + sha-256).  Nothing
else is stored.  Phase A ("write") generates each file once, records its digest and discards it -
the moment at which, in a real system, the file existed and was written.  Phase B ("read") regenerates
every file from the catalogue alone and checks the digest.  Phase C queries the catalogue (metadata
search without regenerating anything).  Results accumulate in results.json so the run can be
interrupted and resumed.
"""
import hashlib, importlib.util, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); DISK = os.path.join(HERE, "AlgorithmicDisk")
CAT = os.path.join(DISK, "catalogue.json"); RES = os.path.join(HERE, "results.json")
GID = "ANET-PROC-telemetry-q20-v1"
spec = importlib.util.spec_from_file_location("gen", os.path.join(DISK, "engines", GID + ".py")); gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
N_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

def records(n):
    """Deterministic, distinct parameter records (the mission's 'runs')."""
    import numpy as np
    rng = np.random.default_rng(20260913); out = []
    for i in range(n):
        out.append(dict(sigma=round(float(rng.uniform(9, 11)), 4), rho=round(float(rng.uniform(24, 32)), 4), beta=round(float(rng.uniform(2.5, 2.8)), 4),
                        x0=round(float(rng.uniform(0.5, 1.5)), 4), y0=round(float(rng.uniform(0.5, 1.5)), 4), z0=round(float(rng.uniform(0.5, 1.5)), 4),
                        dt_shift=9, steps=2_500_000))
    return out

def load(p, default): return json.load(open(p)) if os.path.exists(p) else default

def phase_write():
    cat = load(CAT, {"format": "AlgorithmicDisk/proof-1000", "engine": {"gid": GID, "file": f"engines/{GID}.py", "bytes": os.path.getsize(os.path.join(DISK, "engines", GID + ".py"))}, "manifests": []})
    done = {m["id"] for m in cat["manifests"]}; recs = records(N_FILES); t0 = time.time(); k = 0
    for i, p in enumerate(recs):
        name = f"telemetry_{i:04d}.csv"
        if name in done: continue
        t1 = time.time(); data = gen.telemetry_csv(p); dt = time.time() - t1
        cat["manifests"].append(dict(id=name, gid=GID, seed=p, n_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), gen_s=round(dt, 1))); del data; k += 1
        if k % 10 == 0 or i == N_FILES - 1:
            json.dump(cat, open(CAT, "w")); print(f"  write {i+1:4d}/{N_FILES}  {time.time()-t0:7.0f}s  last {dt:.1f}s", flush=True)
    json.dump(cat, open(CAT, "w"))

def phase_read():
    cat = load(CAT, None); res = load(RES, {"verified": {}, "failed": []}); t0 = time.time(); k = 0
    for m in cat["manifests"]:
        if m["id"] in res["verified"]: continue
        t1 = time.time(); data = gen.telemetry_csv(m["seed"]); dt = time.time() - t1
        ok = len(data) == m["n_bytes"] and hashlib.sha256(data).hexdigest() == m["sha256"]; del data
        (res["verified"].__setitem__(m["id"], dict(regen_s=round(dt, 1), n_bytes=m["n_bytes"])) if ok else res["failed"].append(m["id"])); k += 1
        if k % 10 == 0: json.dump(res, open(RES, "w")); print(f"  read  {len(res['verified']):4d}/{len(cat['manifests'])} verified  {time.time()-t0:7.0f}s  failed {len(res['failed'])}", flush=True)
    json.dump(res, open(RES, "w"))

def phase_query():
    cat = load(CAT, None); t0 = time.time()
    hits = [m["id"] for m in cat["manifests"] if m["seed"]["rho"] > 30 and m["seed"]["sigma"] < 10]; dt = time.time() - t0
    return dict(query="rho > 30 and sigma < 10", hits=len(hits), seconds=round(dt, 4), example=hits[:3])

def summary():
    cat = load(CAT, None); res = load(RES, {"verified": {}, "failed": []})
    n = len(cat["manifests"]); apparent = sum(m["n_bytes"] for m in cat["manifests"])
    seeds = sum(len(json.dumps(m["seed"], sort_keys=True)) for m in cat["manifests"]); prog = cat["engine"]["bytes"]
    disk = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DISK) for f in fs)
    v = res["verified"]; regen = [x["regen_s"] for x in v.values()]
    s = dict(files=n, apparent_bytes=apparent, seed_bytes_total=seeds, program_bytes=prog, catalogue_file_bytes=os.path.getsize(CAT), disk_folder_bytes=disk,
             ratio_seeds_plus_program=(seeds + prog) / apparent if apparent else None, ratio_disk_folder=disk / apparent if apparent else None,
             verified=len(v), failed=len(res["failed"]), all_exact=(len(v) == n and not res["failed"]),
             regen_total_s=round(sum(regen)), regen_mean_s=round(sum(regen) / len(regen), 1) if regen else None,
             write_total_s=round(sum(m["gen_s"] for m in cat["manifests"])), query=phase_query())
    json.dump(s, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
    md = ["# Proof by 1000 files", "",
          f"**Total size of the {n} files: {apparent:,} bytes ({apparent/1e9:.2f} GB)**",
          f"**Total size of what is stored (Seeds {seeds:,} B + generator program {prog:,} B + Manifest overhead) = whole disk folder: {disk:,} bytes**",
          f"**Efficiency ratio: stored / original = {s['ratio_disk_folder']:.2e}  (the original is {apparent/disk:,.0f} times larger than what is stored)**", "",
          f"{n} telemetry files, {apparent/1e9:.2f} GB in total, stored as {seeds:,} bytes of Seeds (parameter records) plus one {prog}-byte generator program;",
          f"the catalogue file holding all {n} Manifests is {s['catalogue_file_bytes']:,} bytes and the whole disk folder {disk:,} bytes ({s['ratio_disk_folder']:.2e} of the data).", "",
          f"Regenerated from the catalogue alone and checked against the recorded sha-256: **{len(v)}/{n} bit-exact, {len(res['failed'])} failed**.",
          f"Write (generate + hash) {s['write_total_s']:,} s total; read (regenerate + verify) {s['regen_total_s']:,} s total, {s['regen_mean_s']} s per file on one core.",
          f"Catalogue query \"{s['query']['query']}\" answered in {s['query']['seconds']} s with {s['query']['hits']} hits, without regenerating anything.", ""]
    open(os.path.join(HERE, "REPORT.md"), "w").write("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("write", "all"): phase_write()
    if step in ("read", "all"): phase_read()
    if step in ("summary", "all"): summary()
