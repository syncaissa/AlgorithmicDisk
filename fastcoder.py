#!/usr/bin/env python3
"""
N6 — compiled coder: the fixed-point AlgorithmicNET and its range coder in C (native/anet_fixed.c),
called through ctypes.  Same container format as realfile.encode_blocks (2-byte header per block,
verbatim escape), same integer engine as fixedpoint.py, same coder as predictive.py -> Seeds are
interchangeable between Python and C, and blocks run in parallel with OpenMP (E5: Horizon vs cores).

    python3 fastcoder.py            # builds the library if needed, cross-checks against Python, runs E5
"""
import ctypes, json, math, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fixedpoint import FixedPointANET, TANH_LUT, NEG2_LUT, START
from predictive import RangeEncoder

HERE = os.path.dirname(os.path.abspath(__file__)); LIB = os.path.join(HERE, "native", "libanet_fixed.so")

def build():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(os.path.join(HERE, "native", "anet_fixed.c")):
        subprocess.check_call(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
                               os.path.join(HERE, "native", "anet_fixed.c"), "-o", LIB])
    return ctypes.CDLL(LIB)

class FastFixedEngine:
    def __init__(self, ej):
        self.lib = build(); fx = FixedPointANET(np, ej)
        i32 = lambda a: np.ascontiguousarray(np.asarray(a, dtype=np.int64).astype(np.int32))
        self.E, self.W_x, self.W_h, self.b_h, self.W_o, self.b_o = (i32(a) for a in (fx.E, fx.W_x, fx.W_h, fx.b_h, fx.W_o, fx.b_o))
        self.tanh = np.ascontiguousarray(TANH_LUT); self.neg2 = np.ascontiguousarray(NEG2_LUT)
        self.hidden, self.emb = fx.W_h.shape[0], fx.E.shape[1]
        P = ctypes.POINTER
        self.lib.encode_blocks.restype = ctypes.c_int; self.lib.decode_blocks.restype = ctypes.c_int
    def _args(self):
        p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        return [ctypes.c_int(self.hidden), ctypes.c_int(self.emb), p(self.E), p(self.W_x), p(self.W_h), p(self.b_h), p(self.W_o), p(self.b_o), p(self.tanh), p(self.neg2)]

    def encode_blocks(self, data: bytes, L=256, threads=0):
        n = len(data); nb = math.ceil(n / L); cap = 2 * L + 16
        buf = np.zeros(nb * cap, dtype=np.uint8); lens = np.zeros(nb, dtype=np.int64)
        d = np.frombuffer(data, dtype=np.uint8)
        self.lib.encode_blocks(*self._args(), d.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(n), ctypes.c_int(L),
                               buf.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(cap), lens.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(threads))
        out = []
        for j in range(nb):
            ln = int(lens[j]); blen = min(L, n - j * L); raw = data[j * L:j * L + blen]
            if ln >= blen: out.append((0x8000 | blen).to_bytes(2, "big") + raw)
            else: out.append(ln.to_bytes(2, "big") + buf[j * cap:j * cap + ln].tobytes())
        return b"".join(out)

    def decode_blocks(self, seed: bytes, n, L=256, threads=0):
        nb = math.ceil(n / L); pos = 0; offs = np.zeros(nb, dtype=np.int64); lens = np.zeros(nb, dtype=np.int64)
        data = np.zeros(n, dtype=np.uint8); raw_blocks = []
        for j in range(nb):
            hdr = int.from_bytes(seed[pos:pos + 2], "big"); pos += 2; ln = hdr & 0x7FFF
            if hdr & 0x8000: data[j * L:j * L + ln] = np.frombuffer(seed[pos:pos + ln], dtype=np.uint8); raw_blocks.append(j); offs[j] = pos; lens[j] = 0
            else: offs[j] = pos; lens[j] = ln
            pos += ln
        s = np.frombuffer(seed, dtype=np.uint8)
        # raw blocks: give the C decoder a zero-length stream and let it decode garbage there, then overwrite
        self.lib.decode_blocks(*self._args(), s.ctypes.data_as(ctypes.c_void_p), offs.ctypes.data_as(ctypes.c_void_p), lens.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_int(nb), data.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(n), ctypes.c_int(L), ctypes.c_int(threads))
        pos = 0
        for j in range(nb):
            hdr = int.from_bytes(seed[pos:pos + 2], "big"); pos += 2; ln = hdr & 0x7FFF
            if hdr & 0x8000: data[j * L:j * L + ln] = np.frombuffer(seed[pos:pos + ln], dtype=np.uint8)
            pos += ln
        return data.tobytes()

# ----------------------------------------------------------------------------- python reference (same container) for the cross-check
def py_encode_blocks(fx, data, L=256):
    from fixedpoint import encode as fx_encode
    out = []
    for j in range(0, len(data), L):
        blk = data[j:j + L]; s, _ = fx_encode(fx, blk, L)
        out.append(((0x8000 | len(blk)).to_bytes(2, "big") + blk) if len(s) >= len(blk) else (len(s).to_bytes(2, "big") + s))
    return b"".join(out)

if __name__ == "__main__":
    import hashlib, lzma
    from realfile import gutenberg_body
    ej = json.load(open(os.path.join(HERE, "engine_books.json"))); eng = FastFixedEngine(ej); fx = FixedPointANET(np, ej)
    # 1. cross-check C vs Python on 20 KB
    data = gutenberg_body(os.path.join(HERE, "books/pg12.txt"))[:20000]
    t0 = time.time(); sc = eng.encode_blocks(data); tc = time.time() - t0
    t0 = time.time(); sp = py_encode_blocks(fx, data); tp = time.time() - t0
    print(f"C vs Python fixed-point Seeds identical: {sc == sp}  ({len(sc):,} B)   C encode {tc:.2f}s  Python encode {tp:.1f}s  -> {tp/tc:.0f}x faster")
    print(f"C decodes Python Seed exactly: {eng.decode_blocks(sp, len(data)) == data}")
    # 2. E5: Horizon vs threads on real objects
    res = dict(objects=[])
    objs = [("pg12.txt", gutenberg_body(os.path.join(HERE, "books/pg12.txt")))]
    big = b"".join(gutenberg_body(os.path.join(HERE, f)) for f in ("alice.txt", "books/pg1342.txt", "books/pg84.txt", "books/pg2701.txt", "books/pg98.txt"))
    objs.append(("five_books.txt", big))
    for name, d in objs:
        row = dict(file=name, bytes=len(d), xz=len(lzma.compress(d, preset=9)), runs=[])
        seed = None
        for th in (1, 2, 4):
            t0 = time.time(); s = eng.encode_blocks(d, threads=th); te = time.time() - t0
            t0 = time.time(); back = eng.decode_blocks(s, len(d), threads=th); td = time.time() - t0
            ok = back == d and (seed is None or s == seed); seed = s
            row["runs"].append(dict(threads=th, encode_s=round(te, 2), decode_s=round(td, 2), read_MBps=round(len(d) / td / 1e6, 2), exact=ok))
            print(f"{name:16s} {len(d):>10,} B  Seed {len(s):>9,} B (ratio {len(s)/len(d):.3f}, xz {row['xz']/len(d):.3f})  threads {th}: write {te:6.2f}s  read {td:6.2f}s = {len(d)/td/1e6:5.2f} MB/s  exact {ok}")
        row["seed_bytes"] = len(seed); row["ratio"] = round(len(seed) / len(d), 4); res["objects"].append(row)
    res["cores"] = os.cpu_count(); res["python_coder_KBps"] = 6.5
    json.dump(res, open(os.path.join(HERE, "fastcoder_results.json"), "w"), indent=1)
    print("saved fastcoder_results.json")
