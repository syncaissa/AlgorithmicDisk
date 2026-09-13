#!/usr/bin/env python3
"""
Experiment N2 — a REAL FILE stored as ONE object on an AlgorithmicDisk.

Engine: the predictive AlgorithmicNET of predictive.py (byte-level, output feedback), trained
once on a corpus of public-domain books and frozen as int8 weights = the shared generator.
Seed Discovery (write) = arithmetic-code the file under the engine (closed form).
Manifestation (read)   = arithmetic-decode, the engine driven by its own output.

The file is processed in blocks of L bytes; every block starts from the zero state, so blocks
are independent given the Seed and can be manifested in parallel:
    --mode stream : one range-coder stream over all blocks (smallest Seed; sequential read)
    --mode blocks : one stream per block (block-parallel read; 4-byte flush per block) with a
                    per-block VERBATIM ESCAPE: a block the engine cannot shorten is stored raw,
                    so the Footprint never exceeds the file size by more than 2 bytes per block
                    (the ceiling of the conservation law, implemented)

Reports, per file: size, Seed bytes (Footprint), ratio, gzip/bz2/xz baselines, write Work
(seconds), read Horizon (seconds, sequential and block-parallel), sha-256 verification.
Device flag --device cpu|cuda|auto (numpy / cupy).
"""
import argparse, base64, bz2, gzip, hashlib, json, lzma, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from algorithmic_storage import get_backend, to_np
from amortize import Adam
from predictive import PredictiveANET, RangeEncoder, RangeDecoder, START

# ----------------------------------------------------------------------------- corpus
def gutenberg_body(path):
    raw = open(path, "rb").read()
    a = raw.find(b"*** START OF"); a = raw.find(b"\n", a) + 1 if a >= 0 else 0
    b = raw.find(b"*** END OF"); b = b if b >= 0 else len(raw)
    return raw[a:b]

def windows(data: bytes, L: int):
    n = len(data) // L
    return np.frombuffer(data[:n * L], dtype=np.uint8).reshape(n, L).astype(np.int64)

# ----------------------------------------------------------------------------- training
def train_engine(xp, corpus: bytes, hidden, emb, L, epochs, batch=128, lr=3e-3, log=print, max_bytes=None):
    if max_bytes: corpus = corpus[:max_bytes]
    X = xp.asarray(windows(corpus, L))
    net = PredictiveANET(xp, hidden=hidden, emb=emb)
    opt = Adam(xp, {k: v.shape for k, v in net.params().items()}, lr)
    rng = np.random.default_rng(0); t0 = time.time(); hist = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(X.shape[0]); tot = 0.0
        for b0 in range(0, X.shape[0], batch):
            idx = xp.asarray(perm[b0:b0 + batch])
            loss, g, _ = net.loss_and_grads(X[idx]); tot += loss * len(idx)
            p = opt.step(net.params(), g)
            for k, v in p.items(): setattr(net, k, v)
        bpc = tot / X.shape[0] / math.log(2)
        hist.append(dict(epoch=ep, bits_per_byte=round(bpc, 3), sec=round(time.time() - t0)))
        log(f"    ep {ep:3d}  train {bpc:.3f} bits/byte  {time.time()-t0:5.0f}s")
    net.cast_int8()
    return net, hist

# ----------------------------------------------------------------------------- engine (de)serialisation
def engine_to_json(net):
    out = {"arch": net.arch, "weights": {}}
    for k, v in net.params().items():
        v = to_np(net.xp, v); s = float(np.max(np.abs(v))) / 127.0 or 1.0
        q = np.clip(np.rint(v / s), -127, 127).astype(np.int8)
        out["weights"][k] = {"shape": list(v.shape), "scale": s, "int8_b64": base64.b64encode(q.tobytes()).decode()}
    return out

def engine_from_json(xp, d):
    net = PredictiveANET(xp, hidden=d["arch"]["hidden"], emb=d["arch"]["emb"])
    for k, w in d["weights"].items():
        q = np.frombuffer(base64.b64decode(w["int8_b64"]), dtype=np.int8).astype(np.float32).reshape(w["shape"])
        setattr(net, k, xp.asarray(q * w["scale"], dtype=xp.float32))
    return net

def engine_bytes(d):
    return sum(len(base64.b64decode(w["int8_b64"])) + 4 for w in d["weights"].values())

# ----------------------------------------------------------------------------- Seed Discovery / Manifestation on blocks
def cum_table(net, logit):
    f = net.freqs(logit); return f, np.concatenate([[0], np.cumsum(f)])

def encode_stream(net, data: bytes, L):
    """One range-coder stream over all blocks; engine state reset at every block start."""
    xp = net.xp; enc = RangeEncoder(); ideal = 0.0
    for b0 in range(0, len(data), L):
        H = xp.zeros((1, net.W_h.shape[0]), dtype=xp.float32); prev = xp.asarray([START])
        for ch in data[b0:b0 + L]:
            H, logit = net.step(H, prev); f, cum = cum_table(net, logit[0])
            enc.encode(int(cum[ch]), int(f[ch]), int(cum[-1])); ideal += -math.log2(f[ch] / cum[-1]); prev = xp.asarray([ch])
    return enc.finish(), ideal

def decode_stream(net, seed: bytes, n, L):
    xp = net.xp; dec = RangeDecoder(seed); out = bytearray()
    for b0 in range(0, n, L):
        H = xp.zeros((1, net.W_h.shape[0]), dtype=xp.float32); prev = xp.asarray([START])
        for _ in range(min(L, n - b0)):
            H, logit = net.step(H, prev); f, cum = cum_table(net, logit[0])
            tgt = dec.get_target(int(cum[-1])); ch = int(np.searchsorted(cum, tgt, side="right") - 1)
            dec.decode(int(cum[ch]), int(f[ch])); out.append(ch); prev = xp.asarray([ch])
    return bytes(out)

def encode_blocks(net, data: bytes, L):
    """One stream per block, engine batched across ALL blocks (block-parallel write)."""
    xp = net.xp
    n = len(data); nb = math.ceil(n / L)
    padded = data + b"\0" * (nb * L - n)
    X = np.frombuffer(padded, dtype=np.uint8).reshape(nb, L)
    encs = [RangeEncoder() for _ in range(nb)]; lens = [min(L, n - j * L) for j in range(nb)]
    H = xp.zeros((nb, net.W_h.shape[0]), dtype=xp.float32); prev = xp.full((nb,), START, dtype=xp.int64); ideal = 0.0
    for t in range(L):
        H, logits = net.step(H, prev); lg = to_np(xp, logits)
        for j in range(nb):
            if t < lens[j]:
                f, cum = cum_table(net, lg[j]); ch = int(X[j, t])
                encs[j].encode(int(cum[ch]), int(f[ch]), int(cum[-1])); ideal += -math.log2(f[ch] / cum[-1])
        prev = xp.asarray(X[:, t].astype(np.int64))
    parts = [e.finish() for e in encs]
    # container: 2-byte header per block = [raw flag (1 bit) | length (15 bits)].  Verbatim escape:
    # a block whose coded stream is not shorter than the block itself is stored raw (Theorem ceiling).
    out = []
    for j, p in enumerate(parts):
        raw = padded[j * L:j * L + lens[j]]
        if len(p) >= lens[j]:
            out.append((0x8000 | lens[j]).to_bytes(2, "big") + raw)
        else:
            out.append(len(p).to_bytes(2, "big") + p)
    return b"".join(out), ideal

def decode_blocks(net, seed: bytes, n, L):
    xp = net.xp; nb = math.ceil(n / L); lens = [min(L, n - j * L) for j in range(nb)]
    parts = []; pos = 0; out = np.zeros((nb, L), dtype=np.int64); raw_blocks = set()
    for j in range(nb):
        hdr = int.from_bytes(seed[pos:pos + 2], "big"); pos += 2; ln = hdr & 0x7FFF
        if hdr & 0x8000:
            out[j, :ln] = np.frombuffer(seed[pos:pos + ln], dtype=np.uint8); raw_blocks.add(j); parts.append(None)
        else:
            parts.append(seed[pos:pos + ln])
        pos += ln
    decs = [RangeDecoder(p) if p is not None else None for p in parts]
    H = xp.zeros((nb, net.W_h.shape[0]), dtype=xp.float32); prev = xp.full((nb,), START, dtype=xp.int64)
    for t in range(L):
        H, logits = net.step(H, prev); lg = to_np(xp, logits)
        for j in range(nb):
            if t < lens[j] and j not in raw_blocks:
                f, cum = cum_table(net, lg[j]); tgt = decs[j].get_target(int(cum[-1]))
                ch = int(np.searchsorted(cum, tgt, side="right") - 1); decs[j].decode(int(cum[ch]), int(f[ch])); out[j, t] = ch
        prev = xp.asarray(out[:, t])
    return bytes(out.reshape(-1)[:n].astype(np.uint8))

# ----------------------------------------------------------------------------- AlgorithmicDisk v0.2 (predictive objects)
class Disk:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {"format": "AlgorithmicDisk/0.2", "generators": {}, "manifests": []}
    def save(self): json.dump(self.d, open(self.path, "w"))
    def add_engine(self, net):
        ej = engine_to_json(net); gid = "ANET-P-" + hashlib.sha256(json.dumps(ej, sort_keys=True).encode()).hexdigest()[:16]
        self.d["generators"][gid] = ej; return gid
    def write(self, gid, name, seed, n, digest, L, mode, stats):
        rm = dict(id=name, gid=gid, n_bytes=n, sha256=digest, block_len=L, mode=mode, seed_b64=base64.b64encode(seed).decode(), discovery=stats)
        self.d["manifests"] = [m for m in self.d["manifests"] if m["id"] != name] + [rm]; return rm
    def read(self, xp, name):
        rm = next(m for m in self.d["manifests"] if m["id"] == name)
        net = engine_from_json(xp, self.d["generators"][rm["gid"]]); seed = base64.b64decode(rm["seed_b64"])
        t0 = time.time()
        data = decode_stream(net, seed, rm["n_bytes"], rm["block_len"]) if rm["mode"] == "stream" else decode_blocks(net, seed, rm["n_bytes"], rm["block_len"])
        ok = hashlib.sha256(data).hexdigest() == rm["sha256"]
        return data, ok, time.time() - t0, rm

# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train-books", nargs="+", default=["alice.txt", "books/pg1342.txt", "books/pg84.txt", "books/pg2701.txt", "books/pg98.txt"])
    ap.add_argument("--train-bytes", type=int, default=2_000_000)
    ap.add_argument("--files", nargs="+", default=["books/pg12.txt", "../ideaBrainStormFile.txt", "algorithmic_storage.py"])
    ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--block-len", type=int, default=256); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--disk", default="realfile.adisk"); ap.add_argument("--engine", default="engine_books.json")
    ap.add_argument("--out", default="realfile_results.json"); ap.add_argument("--restore-dir", default="restored")
    args = ap.parse_args()
    xp, dev = get_backend(args.device)
    L = args.block_len
    if os.path.exists(args.engine):
        ej = json.load(open(args.engine)); net = engine_from_json(xp, ej); hist = ej.get("hist", [])
        print(f"[device {dev}] loaded engine {args.engine}")
    else:
        corpus = b"".join(gutenberg_body(p) for p in args.train_books)
        print(f"[device {dev}] training engine on {min(len(corpus), args.train_bytes):,} bytes from {len(args.train_books)} books, "
              f"hidden {args.hidden}, block {L}, {args.epochs} epochs")
        net, hist = train_engine(xp, corpus, args.hidden, args.emb, L, args.epochs, max_bytes=args.train_bytes)
        ej = engine_to_json(net); ej["hist"] = hist; ej["train_books"] = args.train_books; ej["train_bytes"] = min(len(corpus), args.train_bytes)
        json.dump(ej, open(args.engine, "w"))
    wbytes = engine_bytes(ej)
    print(f"  engine (shared generator): {net.n_params():,} params -> {wbytes:,} bytes int8")
    if os.path.exists(args.disk): os.remove(args.disk)
    disk = Disk(args.disk); gid = disk.add_engine(net); disk.save()
    os.makedirs(args.restore_dir, exist_ok=True)
    res = dict(device=dev, engine=dict(params=net.n_params(), bytes=wbytes, hidden=args.hidden, emb=args.emb, block_len=L,
                                       train_bytes=ej.get("train_bytes"), train_books=ej.get("train_books"), hist=hist), files=[])
    missing = [p for p in args.files if not os.path.exists(p)]
    if missing: print(f"  skipping missing input(s): {missing}  (the paper's ideaBrainStormFile.txt row was a snapshot of the research log; its object is still on realfile.adisk and is regenerated by verify.py)")
    for path in [p for p in args.files if os.path.exists(p)]:
        data = open(path, "rb").read(); n = len(data); name = os.path.basename(path)
        digest = hashlib.sha256(data).hexdigest()
        base = dict(gzip9=len(gzip.compress(data, 9)), bz2=len(bz2.compress(data, 9)), xz=len(lzma.compress(data, preset=9)))
        print(f"\n== {name}: {n:,} bytes   gzip {base['gzip9']:,}  bz2 {base['bz2']:,}  xz {base['xz']:,}")
        row = dict(file=name, bytes=n, sha256=digest, baselines=base, modes={})
        for mode in ("stream", "blocks"):
            t0 = time.time()
            seed, ideal = (encode_stream if mode == "stream" else encode_blocks)(net, data, L)
            t_write = time.time() - t0
            rm = disk.write(gid, f"{name}:{mode}", seed, n, digest, L, mode, dict(seconds=round(t_write, 1))); disk.save()
            back, ok, t_read, _ = Disk(args.disk).read(xp, f"{name}:{mode}")
            if mode == "stream":
                open(os.path.join(args.restore_dir, name), "wb").write(back)
            row["modes"][mode] = dict(seed_bytes=len(seed), ideal_bytes=round(ideal / 8), ratio=round(len(seed) / n, 4),
                                      bits_per_byte=round(len(seed) * 8 / n, 3), write_s=round(t_write, 1), read_s=round(t_read, 1),
                                      exact=ok, seed_plus_engine=len(seed) + wbytes)
            print(f"   {mode:6s}: Seed {len(seed):,} B ({len(seed)*8/n:.2f} bits/byte, ratio {len(seed)/n:.3f}; ideal {ideal/8:,.0f} B)  "
                  f"write {t_write:.1f}s  read {t_read:.1f}s  sha256 {'PASS' if ok else 'FAIL'}  |  Seed+engine {len(seed)+wbytes:,} B")
        res["files"].append(row); json.dump(res, open(args.out, "w"), indent=1)
    print(f"\ndisk {args.disk}: {os.path.getsize(args.disk):,} bytes on disk (JSON incl. base64 overhead)")
    print("saved", args.out)

if __name__ == "__main__":
    main()
