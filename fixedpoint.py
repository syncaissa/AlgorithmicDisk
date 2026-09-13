#!/usr/bin/env python3
"""
N3 — FIXED-POINT AlgorithmicNET: bit-identical Manifestation on any device by construction.

A float engine's next-byte probabilities depend on the order in which a BLAS (or a GPU) sums a
dot product; one differing frequency table desynchronises the range coder and the object is
lost.  The fixed-point engine uses ONLY integer arithmetic (int64 accumulators, Q12 weights,
Q15 activations, table-driven tanh and exp), so every device that implements two's-complement
integers produces the same tables, the same Seed and the same object.

Post-training quantisation of a trained PredictiveANET (no retraining):
    weights   w_int = round(w * 2^12)           (int32, clipped to int16 range)
    state     h_int in Q15  [-32768, 32767]
    u_int     = (W_h_int @ h_int + W_x_int @ e_int + b_int) >> 12       (Q15)
    h_int     = TANH_LUT[u_int]                                          (Q15, 4096-entry table over [-8, 8))
    logit_int = (W_o_int @ h_int) >> 12 + b_o_int                        (Q15)
    freq      = integer softmax via EXP2_LUT (256-entry) and shifts -> table summing to 2^16
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from predictive import RangeEncoder, RangeDecoder, START

QW, QA = 12, 15                                    # weight and activation fractional bits
TANH_N, TANH_RANGE = 4096, 8.0
TANH_LUT = np.round(np.tanh((np.arange(TANH_N) / TANH_N * 2 - 1) * TANH_RANGE) * (1 << QA)).astype(np.int64)
NEG2_LUT = np.round(2.0 ** (-np.arange(256) / 256.0) * (1 << 16)).astype(np.int64)   # 2^(-k/256), Q16

class FixedPointANET:
    def __init__(self, xp, ej):
        """Build from an engine JSON (int8 weights + scales) — the same file the float engine uses."""
        import base64
        self.xp = xp; self.arch = ej["arch"]
        def W(k):
            w = ej["weights"][k]
            q = np.frombuffer(base64.b64decode(w["int8_b64"]), dtype=np.int8).astype(np.float64).reshape(w["shape"]) * w["scale"]
            return xp.asarray(np.clip(np.round(q * (1 << QW)), -32768, 32767).astype(np.int64))
        self.E, self.W_x, self.W_h, self.b_h, self.W_o, self.b_o = (W(k) for k in ("E", "W_x", "W_h", "b_h", "W_o", "b_o"))
        self.tanh_lut = xp.asarray(TANH_LUT); self.neg2_lut = xp.asarray(NEG2_LUT)
        self.hidden = self.W_h.shape[0]

    def gid(self):
        import hashlib
        return "ANET-FX-" + hashlib.sha256(b"".join(np.asarray(self.xp.asnumpy(p) if hasattr(self.xp, "asnumpy") else p).tobytes()
                                                    for p in (self.E, self.W_x, self.W_h, self.b_h, self.W_o, self.b_o))).hexdigest()[:12]

    def step(self, H, x_idx):
        """H (m, hidden) int64 Q15; x_idx (m,) -> H', logits (m, 256) int64 Q15. Integer only."""
        xp = self.xp
        e = self.E[x_idx]                                                   # (m, emb)  Q12
        u = ((H @ self.W_h.T) >> QW) + ((e @ self.W_x.T) >> (2 * QW - QA)) + (self.b_h << (QA - QW))   # Q15
        idx = ((u + (int(TANH_RANGE) << QA)) * TANH_N) >> (QA + 4)         # [-8,8) in Q15 -> [0, 4096)
        H = self.tanh_lut[xp.clip(idx, 0, TANH_N - 1)]                     # Q15
        logit = ((H @ self.W_o.T) >> QW) + (self.b_o << (QA - QW))          # Q15
        return H, logit

    def freqs(self, logit, total=1 << 16):
        """Integer softmax -> frequency table summing to `total`, every symbol >= 1. Deterministic."""
        xp = self.xp
        z = logit - int(xp.max(logit))                                      # <= 0, Q15 (real value z / 2^15)
        q = ((-z) * 47274) >> 22                                            # -z / ln2 in Q8   (47274 = 2^22 * 256 / (ln2 * 2^15))
        ip = q >> 8; fp = q & 255                                           # 2^(-q/256) = 2^(-fp/256) >> ip
        p = xp.where(ip >= 62, 0, self.neg2_lut[fp] >> xp.minimum(ip, 62))  # Q16
        s = int(xp.sum(p)) or 1
        f = ((p * (total - 256)) // s) + 1
        f_np = np.asarray(xp.asnumpy(f) if hasattr(xp, "asnumpy") else f, dtype=np.int64)
        f_np[int(np.argmax(f_np))] += total - int(f_np.sum())
        return f_np

def encode(net, data: bytes, L=256):
    xp = net.xp; enc = RangeEncoder(); ideal = 0.0
    for b0 in range(0, len(data), L):
        H = xp.zeros((1, net.hidden), dtype=xp.int64); prev = xp.asarray([START])
        for ch in data[b0:b0 + L]:
            H, logit = net.step(H, prev); f = net.freqs(logit[0]); cum = np.concatenate([[0], np.cumsum(f)])
            enc.encode(int(cum[ch]), int(f[ch]), int(cum[-1])); ideal += -math.log2(f[ch] / cum[-1]); prev = xp.asarray([ch])
    return enc.finish(), ideal

def decode(net, seed: bytes, n, L=256):
    xp = net.xp; dec = RangeDecoder(seed); out = bytearray()
    for b0 in range(0, n, L):
        H = xp.zeros((1, net.hidden), dtype=xp.int64); prev = xp.asarray([START])
        for _ in range(min(L, n - b0)):
            H, logit = net.step(H, prev); f = net.freqs(logit[0]); cum = np.concatenate([[0], np.cumsum(f)])
            tgt = dec.get_target(int(cum[-1])); ch = int(np.searchsorted(cum, tgt, side="right") - 1)
            dec.decode(int(cum[ch]), int(f[ch])); out.append(ch); prev = xp.asarray([ch])
    return bytes(out)

if __name__ == "__main__":
    from realfile import gutenberg_body, engine_from_json
    ej = json.load(open("engine_books.json")); fx = FixedPointANET(np, ej); fl = engine_from_json(np, ej)
    data = gutenberg_body("books/pg12.txt")[:20000]
    seed, ideal = encode(fx, data); back = decode(fx, seed, len(data))
    from realfile import encode_stream
    fseed, fideal = encode_stream(fl, data, 256)
    print(f"fixed-point: Seed {len(seed):,} B ({len(seed)*8/len(data):.3f} bits/byte)  exact {back == data}   |  float32: {len(fseed):,} B ({len(fseed)*8/len(data):.3f} bits/byte)")
