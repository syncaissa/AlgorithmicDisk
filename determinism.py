#!/usr/bin/env python3
"""
N3 — CROSS-DEVICE DETERMINISM TEST.

Question: does a Seed written on one device regenerate the same object on another?
Two engines are tested on the same objects:
    float32 predictive engine   (predictive.py / realfile.py, the one used in Sections 6.4-6.9)
    fixed-point engine          (fixedpoint.py, integer-only)
under "device variants" that change how a dot product is evaluated, which is exactly what a
different BLAS, a different core count, or a GPU does:
    ref       plain numpy matmul
    chunked   the reduction split in two halves and added (a different summation order)
    f64       accumulation in float64 (a different precision, like a fused/TF32 kernel)
    cuda      CuPy on a real GPU (if installed)                       -> python3 determinism.py --device cuda
For every (engine, variant) the object is (a) decoded from the reference Seed and (b) re-encoded,
and the number of differing probability tables is counted.  PASS = identical Seed AND sha-256.

Reference Seeds produced on the CPU are stored in determinism_reference.json so that a run on
another machine (or GPU) compares against the *original* Seeds, not against itself.
"""
import argparse, base64, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from algorithmic_storage import get_backend, to_np
from realfile import engine_from_json, gutenberg_body
from predictive import RangeEncoder, RangeDecoder, START
from fixedpoint import FixedPointANET, QW, QA, TANH_N, TANH_RANGE

# ----------------------------------------------------------------------------- matmul variants
def mm_ref(a, b): return a @ b
def mm_chunked(a, b):
    k = a.shape[-1] // 2
    return a[..., :k] @ b[:k] + a[..., k:] @ b[k:]
def mm_f64(a, b):
    xp = np if isinstance(a, np.ndarray) else __import__("cupy")
    out = a.astype(xp.float64) @ b.astype(xp.float64)
    return out.astype(a.dtype) if a.dtype.kind == "f" else xp.rint(out).astype(a.dtype)
VARIANTS = {"ref": mm_ref, "chunked": mm_chunked, "f64": mm_f64}

# ----------------------------------------------------------------------------- engines with pluggable matmul
class FloatEngine:
    def __init__(self, xp, ej, mm): self.net = engine_from_json(xp, ej); self.mm = mm; self.xp = xp; self.hidden = self.net.W_h.shape[0]
    def zero(self): return self.xp.zeros((1, self.hidden), dtype=self.xp.float32)
    def step(self, H, prev):
        n = self.net; xp = self.xp
        H = xp.tanh(self.mm(H, n.W_h.T) + self.mm(n.E[prev], n.W_x.T) + n.b_h)
        return H, self.mm(H, n.W_o.T) + n.b_o
    def freqs(self, logit): return self.net.freqs(logit)

class FixedEngine:
    def __init__(self, xp, ej, mm): self.net = FixedPointANET(xp, ej); self.mm = mm; self.xp = xp; self.hidden = self.net.hidden
    def zero(self): return self.xp.zeros((1, self.hidden), dtype=self.xp.int64)
    def step(self, H, prev):
        n = self.net; xp = self.xp
        u = (self.mm(H, n.W_h.T) >> QW) + (self.mm(n.E[prev], n.W_x.T) >> (2 * QW - QA)) + (n.b_h << (QA - QW))
        idx = ((u + (int(TANH_RANGE) << QA)) * TANH_N) >> (QA + 4)
        H = n.tanh_lut[xp.clip(idx, 0, TANH_N - 1)]
        return H, (self.mm(H, n.W_o.T) >> QW) + (n.b_o << (QA - QW))
    def freqs(self, logit): return self.net.freqs(logit)

def encode(eng, data, L=256, tables=None):
    xp = eng.xp; enc = RangeEncoder()
    for b0 in range(0, len(data), L):
        H = eng.zero(); prev = xp.asarray([START])
        for ch in data[b0:b0 + L]:
            H, logit = eng.step(H, prev); f = eng.freqs(logit[0]); cum = np.concatenate([[0], np.cumsum(f)])
            if tables is not None: tables.append(f)
            enc.encode(int(cum[ch]), int(f[ch]), int(cum[-1])); prev = xp.asarray([ch])
    return enc.finish()

def decode(eng, seed, n, L=256):
    xp = eng.xp; dec = RangeDecoder(seed); out = bytearray()
    for b0 in range(0, n, L):
        H = eng.zero(); prev = xp.asarray([START])
        for _ in range(min(L, n - b0)):
            H, logit = eng.step(H, prev); f = eng.freqs(logit[0]); cum = np.concatenate([[0], np.cumsum(f)])
            tgt = dec.get_target(int(cum[-1])); ch = int(np.searchsorted(cum, tgt, side="right") - 1)
            dec.decode(int(cum[ch]), int(f[ch])); out.append(ch); prev = xp.asarray([ch])
    return bytes(out)

# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--engine", default="engine_books.json")
    ap.add_argument("--ref", default="determinism_reference.json"); ap.add_argument("--out", default="determinism_results.json")
    ap.add_argument("--bytes", type=int, default=30000)
    args = ap.parse_args()
    xp, dev = get_backend(args.device); ej = json.load(open(args.engine))
    objects = [("pg12.txt[:30000]", gutenberg_body("books/pg12.txt")[:args.bytes]),
               ("algorithmic_storage.py[:10000]", open("algorithmic_storage.py", "rb").read()[:10000]),
               ("image.png[:8000]", open("image.png", "rb").read()[:8000])]
    ref = json.load(open(args.ref)) if os.path.exists(args.ref) else {"produced_on": None, "seeds": {}}
    variants = dict(VARIANTS)
    results = dict(device=dev, reference_from=ref["produced_on"], rows=[])
    for eng_name, cls in (("float32", FloatEngine), ("fixed-point", FixedEngine)):
        for var, mm in variants.items():
            eng = cls(xp, ej, mm)
            for oname, data in objects:
                key = f"{eng_name}|{oname}"
                digest = hashlib.sha256(data).hexdigest()
                t0 = time.time(); tables = []; seed = encode(eng, data, tables=tables); t_enc = time.time() - t0
                if key not in ref["seeds"]:                                 # first machine: establish the reference
                    ref["seeds"][key] = dict(seed_b64=base64.b64encode(seed).decode(), sha256=digest,
                                             tables_sha=hashlib.sha256(np.concatenate(tables).tobytes()).hexdigest(), n=len(data))
                    if ref["produced_on"] is None: ref["produced_on"] = f"{dev}/ref/{os.uname().machine}"
                r = ref["seeds"][key]; rseed = base64.b64decode(r["seed_b64"])
                same_seed = seed == rseed
                tables_same = hashlib.sha256(np.concatenate(tables).tobytes()).hexdigest() == r["tables_sha"]
                t0 = time.time()
                try: back = decode(eng, rseed, r["n"]); exact = hashlib.sha256(back).hexdigest() == r["sha256"]
                except Exception: exact = False                              # desynchronised coder = garbage
                t_dec = time.time() - t0
                row = dict(engine=eng_name, variant=var if dev == "cpu" else f"{dev}:{var}", object=oname, seed_bytes=len(seed),
                           same_seed=same_seed, tables_identical=tables_same, decodes_reference_seed_exactly=exact,
                           encode_s=round(t_enc, 1), decode_s=round(t_dec, 1))
                results["rows"].append(row)
                print(f"{eng_name:11s} {row['variant']:12s} {oname:32s} Seed {len(seed):>6,} B  same-Seed {str(same_seed):5s}  "
                      f"tables-identical {str(tables_same):5s}  decode-ref-exact {str(exact):5s}  {'PASS' if same_seed and exact else 'FAIL'}")
            json.dump(results, open(args.out, "w"), indent=1)
    json.dump(ref, open(args.ref, "w"), indent=1)
    n_pass = sum(1 for r in results["rows"] if r["same_seed"] and r["decodes_reference_seed_exactly"])
    print(f"\n{n_pass}/{len(results['rows'])} (engine, variant, object) combinations bit-identical to the reference")

if __name__ == "__main__":
    main()
