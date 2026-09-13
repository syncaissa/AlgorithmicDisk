#!/usr/bin/env python3
"""
N10 — RETRIEVAL LIBRARY ENGINE (ANET-RET).

Instead of *memorising* a library with a neural network (slow: Section 6.7), the engine
*stores* the library once as a content-addressed index of content-defined chunks and only
predicts what is not already there.  A Seed is a layout: runs of consecutive library chunks
(document, start, length) interleaved with new bytes; the new bytes are the neural residual,
coded with the shared English engine (or xz / raw, whichever is smaller).  Manifestation
retrieves the runs, decodes the residual, and assembles.  Weights and index are paid once and
shared; each object adds only its Seed.  Dedup fused with a predictor: the "cache corner".
"""
import base64, hashlib, json, lzma, os, sys, time, gzip
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from algorithmic_storage import get_backend
from realfile import gutenberg_body, encode_blocks, decode_blocks
from editaware import EditAwareDisk

# ----------------------------------------------------------------------------- content-defined chunking (gear hash)
_GEAR = [int(x) for x in np.random.default_rng(20260912).integers(0, 2**64, size=256, dtype=np.uint64)]
def cdc(data: bytes, avg=64, mn=24, mx=256):
    """Deterministic content-defined chunk boundaries (average `avg` bytes), gear rolling hash."""
    mask = (1 << (avg.bit_length() - 1)) - 1
    out, start, h = [], 0, 0
    g = _GEAR; n = len(data); M = 0xFFFFFFFFFFFFFFFF
    for i in range(n):
        h = ((h << 1) + g[data[i]]) & M
        if i - start + 1 >= mn and ((h & mask) == 0 or i - start + 1 >= mx):
            out.append(data[start:i + 1]); start = i + 1
    if start < n: out.append(data[start:])
    return out

def varint(x):
    out = bytearray()
    while True:
        b = x & 0x7F; x >>= 7
        if x: out.append(b | 0x80)
        else: out.append(b); return bytes(out)
def read_varint(buf, p):
    x = 0; s = 0
    while True:
        b = buf[p]; p += 1; x |= (b & 0x7F) << s; s += 7
        if not b & 0x80: return x, p

# ----------------------------------------------------------------------------- the engine
class RetrievalEngine:
    def __init__(self, docs):
        """docs: list of (name, bytes). Builds the chunk store and per-document chunk sequences."""
        self.store = {}; self.chunks = []; self.docs = []
        for name, data in docs:
            seq = []
            for c in cdc(data):
                h = hashlib.blake2b(c, digest_size=8).digest()
                if h not in self.store: self.store[h] = len(self.chunks); self.chunks.append(c)
                seq.append(self.store[h])
            self.docs.append((name, seq))
        self.pos = {}                                               # chunk id -> list of (doc, index)
        for d, (_, seq) in enumerate(self.docs):
            for i, cid in enumerate(seq): self.pos.setdefault(cid, []).append((d, i))
    def engine_bytes(self):
        blob = b"".join(self.chunks) + b"".join(varint(len(s)) + b"".join(varint(c) for c in s) for _, s in self.docs)
        return len(lzma.compress(blob, preset=9))               # the index is shipped once, compressed
    def gid(self):
        return "ANET-RET-" + hashlib.sha256(b"".join(self.chunks)).hexdigest()[:12]

    def discover(self, data: bytes, residual_engine=None, L=256):
        """Seed = layout of library runs and new segments + coded residual."""
        segs = []; residual = bytearray(); cur = None
        for c in cdc(data):
            h = hashlib.blake2b(c, digest_size=8).digest(); cid = self.store.get(h)
            if cid is not None:
                if cur and cur[0] == "run" and (cur[1], cur[2] + cur[3]) in [(d, i) for d, i in self.pos[cid]]:
                    cur[3] += 1
                else:
                    d, i = self.pos[cid][0]
                    if cur: segs.append(cur)
                    cur = ["run", d, i, 1]
            else:
                if cur and cur[0] == "new": cur[1] += len(c)
                else:
                    if cur: segs.append(cur)
                    cur = ["new", len(c)]
                residual += c
        if cur: segs.append(cur)
        layout = varint(len(segs)) + b"".join(
            (b"\x00" + varint(s[1]) + varint(s[2]) + varint(s[3])) if s[0] == "run" else (b"\x01" + varint(s[1])) for s in segs)
        # residual: best of neural (blocks mode), xz, raw
        cands = [(b"\x00", bytes(residual))]
        if residual:
            cands.append((b"\x01", lzma.compress(bytes(residual), preset=9)))
            if residual_engine is not None:
                cands.append((b"\x02", encode_blocks(residual_engine, bytes(residual), L)[0]))
        tag, res = min(cands, key=lambda t: len(t[1]))
        lz = lzma.compress(layout, preset=9); layout_blob = (b"\x01" + lz) if len(lz) < len(layout) else (b"\x00" + layout)
        seed = varint(len(layout_blob)) + layout_blob + tag + varint(len(residual)) + res
        stats = dict(segments=len(segs), runs=sum(1 for s in segs if s[0] == "run"), residual_bytes=len(residual),
                     residual_coded=len(res), residual_mode={b"\x00": "raw", b"\x01": "xz", b"\x02": "neural"}[tag], layout_bytes=len(layout_blob))
        return seed, stats

    def manifest(self, seed: bytes, residual_engine=None, L=256):
        p = 0; ln, p = read_varint(seed, p); lb = seed[p:p + ln]; p += ln
        layout = lzma.decompress(lb[1:]) if lb[:1] == b"\x01" else lb[1:]
        tag = seed[p:p + 1]; p += 1; rlen, p = read_varint(seed, p); res = seed[p:]
        residual = res if tag == b"\x00" else lzma.decompress(res) if tag == b"\x01" else decode_blocks(residual_engine, res, rlen, L)
        q = 0; nseg, q = read_varint(layout, q); out = []; rp = 0
        for _ in range(nseg):
            kind = layout[q]; q += 1
            if kind == 0:
                d, q = read_varint(layout, q); i, q = read_varint(layout, q); k, q = read_varint(layout, q)
                out.append(b"".join(self.chunks[c] for c in self.docs[d][1][i:i + k]))
            else:
                n, q = read_varint(layout, q); out.append(residual[rp:rp + n]); rp += n
        return b"".join(out)

# ----------------------------------------------------------------------------- disk with retrieval engine
class RetrievalDisk(EditAwareDisk):
    def add_retrieval(self, docs):
        self.ret = RetrievalEngine(docs); gid = self.ret.gid()
        self.d["generators"][gid] = {"kind": "retrieval", "docs": [n for n, _ in docs], "chunks": len(self.ret.chunks), "bytes": self.ret.engine_bytes()}
        self.ret_gid = gid
    def write(self, name, data, L=256, provenance=None, candidates=None, log=print):
        rm = super().write(name, data, L, provenance, candidates, log)
        t0 = time.time(); seed, st = self.ret.discover(data, self.nets.get("ANET-EN"), L); t = time.time() - t0
        rm["trials"].append(dict(gid=self.ret_gid, seed_bytes=len(seed), seconds=round(t, 1), **st))
        log(f"      try {self.ret_gid:12s} -> {len(seed):>9,} B  ({st['runs']} runs, residual {st['residual_bytes']:,} B -> {st['residual_coded']:,} {st['residual_mode']}, {t:.1f}s)")
        if len(seed) < rm["seed_bytes"]:
            rm.update(gid=self.ret_gid, mode="retrieval", base=None, seed_b64=base64.b64encode(seed).decode(), seed_bytes=len(seed))
        return rm
    def read(self, name):
        rm = next(m for m in self.d["manifests"] if m["id"] == name)
        if rm["mode"] != "retrieval": return super().read(name)
        t0 = time.time(); data = self.ret.manifest(base64.b64decode(rm["seed_b64"]), self.nets.get("ANET-EN"), rm["block_len"])
        return data, hashlib.sha256(data).hexdigest() == rm["sha256"], time.time() - t0, rm
    def engine_bytes(self, gid):
        return self.ret.engine_bytes() if gid == getattr(self, "ret_gid", None) else super().engine_bytes(gid)

def main():
    xp, dev = get_backend("cpu"); L = 256
    disk = RetrievalDisk("retrieval.adisk")
    for gid, f in (("ANET-EN", "engine_books.json"), ("ANET-LIB", "engine_lib.json")): disk.add_neural(xp, gid, json.load(open(f)))
    disk.add_procedural()
    looking = gutenberg_body("books/pg12.txt"); alice = gutenberg_body("alice.txt")
    disk.add_retrieval([("pg12.txt", looking), ("alice.txt", alice)])
    print(f"retrieval engine: {len(disk.ret.chunks):,} chunks from {len(looking)+len(alice):,} B of library -> {disk.ret.engine_bytes():,} B shipped (xz)")
    # test objects
    words = looking.split(b" "); rng = np.random.default_rng(7); r1 = list(words)
    for i in rng.choice(len(words), 60, replace=False): r1[i] = b"[REVISED]"
    rev1 = b" ".join(r1)
    r3 = list(rev1.split(b" "))
    for i in rng.choice(len(r3), int(0.05 * len(r3)), replace=False): r3[i] = b"<edit>"
    rev3 = b" ".join(r3)
    paras = [p for p in (looking + b"\n\n" + alice).split(b"\r\n\r\n") if len(p) > 200] or [p for p in (looking + b"\n\n" + alice).split(b"\n\n") if len(p) > 200]
    pick = rng.choice(len(paras), 60, replace=False)
    new_paras = [b"MISSION NOTE %d: this paragraph is new material written for the composite manual, describing procedures that do not appear in any library document. " % i * 3 for i in range(6)]
    composite = b"\n\n".join([paras[i] for i in pick[:30]] + new_paras[:3] + [paras[i] for i in pick[30:]] + new_paras[3:])
    frank = gutenberg_body("books/pg84.txt")[:200_000]
    objects = [("pg12.txt", looking), ("rev1_60words.txt", rev1), ("rev3_5pct.txt", rev3), ("composite_manual.txt", composite), ("pg84_unseen.txt", frank)]
    res = dict(engine_bytes=disk.ret.engine_bytes(), library_bytes=len(looking) + len(alice), chunks=len(disk.ret.chunks), objects=[])
    for name, data in objects:
        print(f"\n== {name}: {len(data):,} bytes   gzip {len(gzip.compress(data, 9)):,}  xz {len(lzma.compress(data, preset=9)):,}")
        rm = disk.write(name, data, L, candidates=["ANET-LIB"] if name != "pg84_unseen.txt" else ["ANET-EN"]); disk.save()
        back, ok, t_read, _ = disk.read(name)
        row = dict(file=name, bytes=len(data), gzip=len(gzip.compress(data, 9)), xz=len(lzma.compress(data, preset=9)), chosen=rm["gid"], base=rm.get("base"),
                   seed_bytes=rm["seed_bytes"], ratio=round(rm["seed_bytes"] / len(data), 5), read_s=round(t_read, 1), exact=ok, trials=rm["trials"])
        res["objects"].append(row); json.dump(res, open("retrieval_results.json", "w"), indent=1)
        print(f"   chosen {rm['gid']}{' (base '+rm['base']+')' if rm.get('base') else ''}: Seed {rm['seed_bytes']:,} B  ratio {row['ratio']:.5f}x  read {t_read:.1f}s  sha256 {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    main()
