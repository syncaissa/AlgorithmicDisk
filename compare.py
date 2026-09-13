#!/usr/bin/env python3
"""Compare a fresh *_results.json with the shipped reference in results/.
    python3 compare.py retrieval_results.json
"""
import json, os, sys

def rows(d):
    if "objects" in d: return [(o["file"], o["seed_bytes"], o.get("exact")) for o in d["objects"]]
    if "files" in d:   return [(f["file"] + ":" + m, x["seed_bytes"], x["exact"]) for f in d["files"] for m, x in f["modes"].items()]
    if "runs" in d:    return [(f"hidden{r['hidden']}/M{r['M']}", r["train"]["seed_bits_mean"], r["train"]["exact"] == 1.0) for r in d["runs"]]
    if "points" in d:  return [(f"{p['minutes']} min", p["seed_bytes"], p["exact"]) for p in d["points"]]
    return []

name = sys.argv[1]
mine, ref = json.load(open(name)), json.load(open(os.path.join("results", os.path.basename(name))))
ok = True
for (k, s, e), (_, rs, _) in zip(rows(mine), rows(ref)):
    flag = "OK " if e else "FAIL"; ok &= bool(e)
    print(f"{flag} {k:32s} reference {rs:>12,}   yours {s:>12,}   {'same' if s == rs else f'{100*(s-rs)/rs:+.1f}%'}")
print("all objects bit-exact" if ok else "SOME OBJECTS NOT EXACT — please report")
sys.exit(0 if ok else 1)
