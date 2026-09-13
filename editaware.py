#!/usr/bin/env python3
"""
N9 — EDIT-AWARE SEEDS.

A revision of an object already on the disk should not pay for the whole object again.
The DELTA engine (ANET-DELTA) makes an object's Seed a reference to a *base* object plus an
edit script; Manifestation regenerates the base (through whatever engine the base uses) and
applies the edits.  Weights are untouched: the shared engines stay exactly as they are, only
Seeds are added.  Chains are allowed (a revision of a revision).

At write time the multi-engine disk now also tries "delta against every existing text object"
and keeps the smallest Seed, so lineage is discovered, not declared.
"""
import base64, difflib, hashlib, json, os, re, sys, time, gzip, lzma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from algorithmic_storage import get_backend
from realfile import gutenberg_body
from multiengine import MultiDisk, PROC_GID

TOK = re.compile(rb"\s+|[^\s]+")
def tokens(b): return TOK.findall(b)
def doc_tokens(b): return [x for l in b.splitlines(keepends=True) for x in tokens(l)]   # same tokenisation as make_delta

def make_delta(base: bytes, new: bytes):
    """Token-level edit script: list of [i1, i2, replacement_bytes] against base tokens.
    Two-level diff (lines first, then tokens inside changed line ranges) keeps it fast on long documents."""
    la, lb = base.splitlines(keepends=True), new.splitlines(keepends=True)
    ta = [tokens(l) for l in la]; tb = [tokens(l) for l in lb]
    starts_a = [0]
    for l in ta: starts_a.append(starts_a[-1] + len(l))
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, la, lb, autojunk=False).get_opcodes():
        if tag == "equal": continue
        a_tok = [x for l in ta[i1:i2] for x in l]; b_tok = [x for l in tb[j1:j2] for x in l]; off = starts_a[i1]
        for tg, p1, p2, q1, q2 in difflib.SequenceMatcher(None, a_tok, b_tok, autojunk=False).get_opcodes():
            if tg != "equal": ops.append([off + p1, off + p2, b"".join(b_tok[q1:q2])])
    blob = b"".join(i1.to_bytes(4, "big") + i2.to_bytes(4, "big") + len(r).to_bytes(4, "big") + r for i1, i2, r in ops)
    z = lzma.compress(blob, preset=9)                       # the edit script itself is shallow data: entropy-code it
    blob = b"\x01" + z if len(z) + 1 < len(blob) + 1 else b"\x00" + blob
    return blob, len(ops)

def apply_delta(base: bytes, blob: bytes):
    blob = lzma.decompress(blob[1:]) if blob[:1] == b"\x01" else blob[1:]
    a = doc_tokens(base); out = []; pos = 0; p = 0
    while p < len(blob):
        i1, i2, ln = (int.from_bytes(blob[p:p+4], "big"), int.from_bytes(blob[p+4:p+8], "big"), int.from_bytes(blob[p+8:p+12], "big"))
        r = blob[p+12:p+12+ln]; p += 12 + ln
        out.append(b"".join(a[pos:i1])); out.append(r); pos = i2
    out.append(b"".join(a[pos:]))
    return b"".join(out)

class EditAwareDisk(MultiDisk):
    def write(self, name, data, L=256, provenance=None, candidates=None, log=print):
        rm = super().write(name, data, L, provenance, candidates, log)
        best = None
        for m in self.d["manifests"]:
            if m["id"] == name or m["mode"] in ("procedural",): continue
            base, ok, _, _ = self.read(m["id"])
            if not ok: continue
            t0 = time.time(); blob, nops = make_delta(base, data); t = time.time() - t0
            seed = m["id"].encode() + b"\0" + blob
            rm["trials"].append(dict(gid="ANET-DELTA", base=m["id"], edits=nops, seed_bytes=len(seed), seconds=round(t, 1)))
            log(f"      try delta vs {m['id']:22s} -> {len(seed):>9,} B  ({nops} edits, {t:.1f}s)")
            if best is None or len(seed) < len(best[1]): best = (m["id"], seed)
        if best and len(best[1]) < rm["seed_bytes"]:
            rm.update(gid="ANET-DELTA", mode="delta", base=best[0], seed_b64=base64.b64encode(best[1]).decode(), seed_bytes=len(best[1]))
        return rm
    def read(self, name):
        rm = next(m for m in self.d["manifests"] if m["id"] == name)
        if rm["mode"] != "delta": return super().read(name)
        t0 = time.time(); seed = base64.b64decode(rm["seed_b64"]); base_id, blob = seed.split(b"\0", 1)
        base, ok, _, _ = self.read(base_id.decode()); data = apply_delta(base, blob)
        return data, ok and hashlib.sha256(data).hexdigest() == rm["sha256"], time.time() - t0, rm
    def engine_bytes(self, gid):
        return 0 if gid in ("ANET-DELTA", "VERBATIM") else super().engine_bytes(gid)

def main():
    xp, dev = get_backend("cpu"); L = 256
    disk = EditAwareDisk("editaware.adisk")
    for gid, f in (("ANET-EN", "engine_books.json"), ("ANET-LIB", "engine_lib.json")): disk.add_neural(xp, gid, json.load(open(f)))
    disk.add_procedural()
    base = gutenberg_body("books/pg12.txt"); words = base.split(b" "); rng = np.random.default_rng(7)
    rev1 = list(words)
    for i in rng.choice(len(words), 60, replace=False): rev1[i] = b"[REVISED]"
    rev1 = b" ".join(rev1)
    rev2 = rev1.replace(b"Alice", b"ALICE", 40) + b"\n\nAPPENDIX A. Added in revision 2: procedures updated for the new mission phase. " * 20
    rev3 = list(rev2.split(b" "))
    for i in rng.choice(len(rev3), int(0.05 * len(rev3)), replace=False): rev3[i] = b"<edit>"      # 5% of words
    rev3 = b" ".join(rev3)
    objects = [("pg12.txt", base), ("rev1_60words.txt", rev1), ("rev2_appendix.txt", rev2), ("rev3_5pct.txt", rev3)]
    res = dict(objects=[])
    for name, data in objects:
        print(f"\n== {name}: {len(data):,} bytes   gzip {len(gzip.compress(data, 9)):,}  xz {len(lzma.compress(data, preset=9)):,}")
        rm = disk.write(name, data, L, candidates=["ANET-LIB"]); disk.save()
        back, ok, t_read, _ = disk.read(name)
        row = dict(file=name, bytes=len(data), gzip=len(gzip.compress(data, 9)), xz=len(lzma.compress(data, preset=9)), chosen=rm["gid"],
                   base=rm.get("base"), seed_bytes=rm["seed_bytes"], ratio=round(rm["seed_bytes"] / len(data), 5), read_s=round(t_read, 1), exact=ok,
                   trials=rm["trials"])
        res["objects"].append(row); json.dump(res, open("editaware_results.json", "w"), indent=1)
        print(f"   chosen {rm['gid']}{' (base '+rm['base']+')' if rm.get('base') else ''}: Seed {rm['seed_bytes']:,} B  ratio {row['ratio']:.5f}x  read {t_read:.1f}s  sha256 {'PASS' if ok else 'FAIL'}")
    # total disk accounting for the four versions
    tot_seed = sum(o["seed_bytes"] for o in res["objects"]); tot_bytes = sum(o["bytes"] for o in res["objects"])
    tot_xz = sum(o["xz"] for o in res["objects"])
    res["totals"] = dict(bytes=tot_bytes, seeds=tot_seed, xz_each=tot_xz, ratio=round(tot_seed / tot_bytes, 5))
    print(f"\nfour versions: {tot_bytes:,} B of documents -> {tot_seed:,} B of Seeds ({tot_seed/tot_bytes:.4f}x); xz on each separately {tot_xz:,} B")
    json.dump(res, open("editaware_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
