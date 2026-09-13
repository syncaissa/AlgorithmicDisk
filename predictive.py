#!/usr/bin/env python3
"""
Experiment E3-B — AMORTISATION with a PREDICTIVE AlgorithmicNET (hybrid core).

Design B: the engine is a seed-conditioned *predictor*.  At every step it emits a probability
distribution over the next byte given the bytes it has already manifested (output feedback).
The Seed of an object is the *arithmetic code* of the object under that predictor.  Seed
Discovery therefore has a CLOSED FORM (entropy coding): no search, one forward pass, and
Seed length = surprisal = -log2 P(object | engine) bits.  Manifestation is the mirror image:
the engine consumes the Seed bit by bit and regenerates the object exactly (lossless by
construction).

Engine (same recurrent form as Design A, but with a 256-way soft-max read-out):
    h_t = tanh(W_h h_{t-1} + W_x e(x_{t-1}) + b_h)          x_{-1} = start symbol
    p_t = softmax(W_o h_t + b_o)                              distribution over the next byte
Weights are trained on the corpus by maximum likelihood (teacher forcing), then frozen and
rounded to float16.  A 32-bit range coder turns p_t into Seed bits.

Footprint per object (bits) = |weights| / M + |Seed|.
Device flag --device cpu|cuda|auto (numpy / cupy).
"""
import argparse, gzip, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithmic_storage import get_backend, to_np
from amortize import load_corpus, Adam

START = 256          # start symbol index (embedding table has 257 rows)

# ----------------------------------------------------------------------------- engine
class PredictiveANET:
    def __init__(self, xp, hidden=128, emb=16, init_seed=20260912, spectral=1.0):
        import numpy as np
        self.xp = xp
        self.arch = dict(hidden=hidden, emb=emb, init_seed=init_seed, spectral=spectral, vocab=256)
        rng = np.random.default_rng(init_seed)
        f = lambda a: xp.asarray(a, dtype=xp.float32)
        self.E   = f(rng.standard_normal((257, emb)) * 0.3)
        self.W_x = f(rng.standard_normal((hidden, emb)) / math.sqrt(emb))
        self.W_h = f(rng.standard_normal((hidden, hidden)) / math.sqrt(hidden) * spectral)
        self.b_h = f(np.zeros(hidden))
        self.W_o = f(rng.standard_normal((256, hidden)) / math.sqrt(hidden))
        self.b_o = f(np.zeros(256))

    def params(self):
        return {"E": self.E, "W_x": self.W_x, "W_h": self.W_h, "b_h": self.b_h, "W_o": self.W_o, "b_o": self.b_o}

    def n_params(self):
        return sum(int(p.size) for p in self.params().values())

    def cast_fp16(self):
        for k, v in self.params().items():
            setattr(self, k, v.astype(self.xp.float16).astype(self.xp.float32))

    def cast_int8(self):
        """Per-tensor symmetric int8 weights (what a production AlgorithmicDisk would hold): 8 bits/param + 1 scale."""
        xp = self.xp
        for k, v in self.params().items():
            s = float(xp.max(xp.abs(v))) / 127.0 or 1.0
            setattr(self, k, (xp.clip(xp.rint(v / s), -127, 127) * s).astype(xp.float32))

    def step(self, H, x_idx):
        """One engine step for a batch: H (m, hidden), x_idx (m,) previous symbols -> H', logits (m, 256)."""
        xp = self.xp
        H = xp.tanh(H @ self.W_h.T + self.E[x_idx] @ self.W_x.T + self.b_h)
        return H, H @ self.W_o.T + self.b_o

    def forward(self, Xidx, keep=False):
        """Teacher-forced. Xidx (m, L) symbol indices -> logits (m, L, 256) for predicting Xidx[:, t]."""
        xp = self.xp
        m, L = Xidx.shape
        H = xp.zeros((m, self.W_h.shape[0]), dtype=xp.float32)
        prev = xp.full((m,), START, dtype=xp.int64)
        Hs, Ls, Ps = [], [], []
        for t in range(L):
            H, logit = self.step(H, prev)
            Ls.append(logit)
            if keep: Hs.append(H); Ps.append(prev)
            prev = Xidx[:, t]
        L_ = xp.stack(Ls, axis=1)
        return (L_, Hs, Ps) if keep else L_

    def loss_and_grads(self, Xidx):
        """Mean cross-entropy (nats) with manual BPTT."""
        xp = self.xp
        m, L = Xidx.shape
        logits, Hs, Ps = self.forward(Xidx, keep=True)
        z = logits - xp.max(logits, axis=2, keepdims=True)
        P = xp.exp(z); P = P / xp.sum(P, axis=2, keepdims=True)
        rows = xp.arange(m)[:, None]; cols = xp.arange(L)[None, :]
        nll = -xp.log(P[rows, cols, Xidx] + 1e-12)
        loss = float(xp.mean(nll))
        dlog = P.copy(); dlog[rows, cols, Xidx] -= 1.0; dlog /= (m * L)
        g = {k: xp.zeros_like(v) for k, v in self.params().items()}
        g_next = xp.zeros_like(Hs[0])
        for t in range(L - 1, -1, -1):
            Ht = Hs[t]; Hprev = Hs[t - 1] if t > 0 else None
            g["W_o"] += dlog[:, t, :].T @ Ht; g["b_o"] += xp.sum(dlog[:, t, :], axis=0)
            gH = dlog[:, t, :] @ self.W_o + g_next @ self.W_h
            gU = gH * (1 - Ht ** 2)
            if Hprev is not None: g["W_h"] += gU.T @ Hprev
            g["b_h"] += xp.sum(gU, axis=0)
            emb = self.E[Ps[t]]                                   # (m, emb)
            g["W_x"] += gU.T @ emb
            gE = gU @ self.W_x                                    # (m, emb)
            xp.add.at(g["E"], Ps[t], gE) if hasattr(xp, "add") and hasattr(xp.add, "at") else None
            if not (hasattr(xp, "add") and hasattr(xp.add, "at")):
                import cupyx; cupyx.scatter_add(g["E"], Ps[t], gE)
            g_next = gU
        return loss, g, nll

    # -- probabilities as integer frequencies (deterministic, what the coder sees)
    def freqs(self, logit, total=1 << 16):
        """logit (256,) -> integer frequency table summing to `total`, every symbol >= 1."""
        import numpy as np
        l = to_np(self.xp, logit).astype(np.float64)
        p = np.exp(l - l.max()); p /= p.sum()
        f = np.maximum(1, np.floor(p * (total - 256)).astype(np.int64)) + 1
        f[np.argmax(p)] += total - f.sum()
        return f

# ----------------------------------------------------------------------------- range coder (32-bit, carry-less, Subbotin style)
class RangeEncoder:
    TOP, BOT = 1 << 24, 1 << 16
    def __init__(self):
        self.low, self.range, self.out = 0, 0xFFFFFFFF, bytearray()
    def encode(self, cum, freq, total):
        r = self.range // total
        self.low += r * cum; self.range = r * freq
        while True:
            if (self.low ^ (self.low + self.range)) < self.TOP:
                pass
            elif self.range < self.BOT:
                self.range = (-self.low) & (self.BOT - 1)
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & 0xFFFFFFFF; self.range = (self.range << 8) & 0xFFFFFFFF
    def finish(self):
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF); self.low = (self.low << 8) & 0xFFFFFFFF
        return bytes(self.out)

class RangeDecoder:
    TOP, BOT = 1 << 24, 1 << 16
    def __init__(self, data):
        self.data, self.pos = data + b"\0" * 8, 0
        self.low, self.range, self.code = 0, 0xFFFFFFFF, 0
        for _ in range(4): self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
    def _byte(self):
        b = self.data[self.pos]; self.pos += 1; return b
    def get_target(self, total):
        self.r = self.range // total
        return min(total - 1, (self.code - self.low) // self.r)
    def decode(self, cum, freq):
        self.low += self.r * cum; self.range = self.r * freq
        while True:
            if (self.low ^ (self.low + self.range)) < self.TOP:
                pass
            elif self.range < self.BOT:
                self.range = (-self.low) & (self.BOT - 1)
            else:
                break
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
            self.low = (self.low << 8) & 0xFFFFFFFF; self.range = (self.range << 8) & 0xFFFFFFFF

# ----------------------------------------------------------------------------- Seed Discovery (closed form) and Manifestation
def discover(net, text):
    """Closed-form Seed Discovery: arithmetic-code the object under the engine. Returns (seed_bytes, ideal_bits)."""
    import numpy as np
    xp = net.xp
    enc = RangeEncoder(); H = xp.zeros((1, net.W_h.shape[0]), dtype=xp.float32); prev = xp.asarray([START]); ideal = 0.0
    for ch in text.encode("ascii"):
        H, logit = net.step(H, prev)
        f = net.freqs(logit[0]); cum = np.concatenate([[0], np.cumsum(f)])
        enc.encode(int(cum[ch]), int(f[ch]), int(cum[-1]))
        ideal += -math.log2(f[ch] / cum[-1])
        prev = xp.asarray([ch])
    return enc.finish(), ideal

def manifest(net, seed, n_chars):
    """Manifestation: regenerate n_chars from the Seed by driving the engine with its own output."""
    import numpy as np
    xp = net.xp
    dec = RangeDecoder(seed); H = xp.zeros((1, net.W_h.shape[0]), dtype=xp.float32); prev = xp.asarray([START]); out = bytearray()
    for _ in range(n_chars):
        H, logit = net.step(H, prev)
        f = net.freqs(logit[0]); cum = np.concatenate([[0], np.cumsum(f)])
        tgt = dec.get_target(int(cum[-1])); ch = int(np.searchsorted(cum, tgt, side="right") - 1)
        dec.decode(int(cum[ch]), int(f[ch])); out.append(ch); prev = xp.asarray([ch])
    return out.decode("ascii")

# ----------------------------------------------------------------------------- training
def train_lm(xp, chunks, hidden, epochs, lr=3e-3, batch=250, log=print, seed=0, emb=16):
    import numpy as np
    X = xp.asarray(np.array([[c for c in ch.encode("ascii")] for ch in chunks]), dtype=xp.int64)
    net = PredictiveANET(xp, hidden=hidden, emb=emb)
    opt = Adam(xp, {k: v.shape for k, v in net.params().items()}, lr)
    rng = np.random.default_rng(seed); t0 = time.time(); hist = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(chunks)); tot = 0.0
        for b0 in range(0, len(chunks), batch):
            idx = xp.asarray(perm[b0:b0 + batch])
            loss, g, _ = net.loss_and_grads(X[idx]); tot += loss * len(idx)
            p = opt.step(net.params(), g)
            for k, v in p.items(): setattr(net, k, v)
        if ep % 10 == 0 or ep == epochs:
            bpc = tot / len(chunks) / math.log(2)
            hist.append(dict(epoch=ep, bits_per_char=round(bpc, 3), sec=round(time.time() - t0, 1)))
            log(f"    ep {ep:4d} train {bpc:.3f} bits/char   {time.time()-t0:5.0f}s")
    net.cast_int8()
    return net, hist

def evaluate(net, chunks, verify=True, log=print):
    """Seed every chunk, manifest it back, verify exactness. Returns per-chunk stats."""
    t0 = time.time(); seed_bits = []; ideal = []; ok = 0
    for i, ch in enumerate(chunks):
        seed, ib = discover(net, ch); seed_bits.append(len(seed) * 8); ideal.append(ib)
        if verify:
            ok += int(manifest(net, seed, len(ch)) == ch)
    return dict(n=len(chunks), exact=ok / len(chunks) if verify else None,
                seed_bits_mean=round(sum(seed_bits) / len(chunks), 1), ideal_bits_mean=round(sum(ideal) / len(chunks), 1),
                seconds=round(time.time() - t0, 1))

# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--corpus", default="alice.txt")
    ap.add_argument("--chunk-chars", type=int, default=29)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--M-sweep", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000])
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--out", default="predictive_results.json")
    args = ap.parse_args()
    xp, dev = get_backend(args.device)
    chunks = load_corpus(args.corpus, args.chunk_chars)
    hold = chunks[-args.holdout:]
    N = args.chunk_chars * 8
    joined = "".join(chunks).encode()
    res = dict(device=dev, chunk_chars=args.chunk_chars, N_bits=N, corpus_chunks=len(chunks), runs=[],
               baselines=dict(raw_bits=N, gzip9_bits_per_chunk=round(len(gzip.compress(joined, 9)) * 8 / len(chunks), 1),
                              untrained_engine_bits=1024))
    print(f"[device {dev}] {len(chunks)} chunks; baselines {res['baselines']}")
    for hid in args.hidden:
        for M in args.M_sweep:
            if M > len(chunks) - args.holdout: break
            print(f"== hidden {hid}  M {M}")
            net, hist = train_lm(xp, chunks[:M], hid, args.epochs)
            tr = evaluate(net, chunks[:M], verify=True)
            ho = evaluate(net, hold, verify=True)
            wbits = net.n_params() * 8 + 6 * 32                       # int8 weights + one fp32 scale per tensor
            breakeven = wbits / max(1e-9, N - tr["seed_bits_mean"])
            row = dict(hidden=hid, M=M, weight_params=net.n_params(), weight_bits=wbits, weight_format="int8",
                       train=tr, holdout=ho,
                       footprint_bits_per_obj=round(wbits / M + tr["seed_bits_mean"], 1),
                       footprint_bits_per_obj_ideal=round(wbits / M + tr["ideal_bits_mean"], 1),
                       breakeven_M=round(breakeven),
                       total_footprint_bytes=round((wbits + M * tr["seed_bits_mean"]) / 8), total_data_bytes=M * args.chunk_chars,
                       hist=hist)
            res["runs"].append(row)
            print(f"   -> train exact {tr['exact']*100:.1f}%  seed {tr['seed_bits_mean']} bits (ideal {tr['ideal_bits_mean']})  "
                  f"holdout exact {ho['exact']*100:.1f}% seed {ho['seed_bits_mean']}  |  footprint/obj {row['footprint_bits_per_obj']} bits "
                  f"(raw {N}, gzip {res['baselines']['gzip9_bits_per_chunk']})  total {row['total_footprint_bytes']} B vs data {row['total_data_bytes']} B  break-even M* {row['breakeven_M']}")
            json.dump(res, open(args.out, "w"), indent=1)
    print("saved", args.out)

if __name__ == "__main__":
    main()
