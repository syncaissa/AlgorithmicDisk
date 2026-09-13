#!/usr/bin/env python3
"""
Experiment E3 — AMORTISATION: thousands of objects on ONE trained AlgorithmicNET.

The engine is the same AlgorithmicNET as in algorithmic_storage.py, but now the weights
(W_h, W_e, W_o, b_h, b_o) are TRAINED on a corpus of M fixed-length text chunks while every
chunk simultaneously gets its own Seed (an "auto-decoder": shared deterministic decoder,
one latent code per object).  Block codes p_j, clock codes c_t and sign-masks m_{j,t} stay
generated from the init seed (zero Footprint).

Footprint accounting (per object, bits):
    F(M) = |weights| / M  +  seed_dim * q_bits
Baselines: raw = 8 bits/char; gzip -9 of the whole corpus / M; untrained engine (Session 3).

Device flag --device cpu|cuda|auto (numpy / cupy).
"""
import argparse, gzip, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithmic_storage import get_backend, to_np, text_to_bits, bits_to_text, quantize_seed, dequantize_seed

# ----------------------------------------------------------------------------- corpus
def load_corpus(path, chunk_chars=29, max_chunks=None):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    a = raw.find("CHAPTER I.", raw.find("CHAPTER I.") + 1)          # skip the table of contents
    b = raw.find("*** END OF THE PROJECT")
    body = " ".join(raw[a:b].split())                                   # collapse whitespace
    body = body.encode("ascii", errors="ignore").decode()              # keep it 7-bit
    chunks = [body[i:i + chunk_chars] for i in range(0, len(body) - chunk_chars, chunk_chars)]
    return chunks[:max_chunks] if max_chunks else chunks

# ----------------------------------------------------------------------------- trainable engine
class TrainableANET:
    def __init__(self, xp, seed_dim=32, hidden=128, bits_per_step=8, block_steps=8, max_blocks=8,
                 init_seed=20260912, spectral=1.2):
        import numpy as np
        self.xp = xp
        self.arch = dict(seed_dim=seed_dim, hidden=hidden, bits_per_step=bits_per_step, block_steps=block_steps,
                         max_blocks=max_blocks, init_seed=init_seed, spectral=spectral)
        rng = np.random.default_rng(init_seed)
        f = lambda a: xp.asarray(a, dtype=xp.float32)
        # trainable
        self.W_h = f(rng.standard_normal((hidden, hidden)) / math.sqrt(hidden) * spectral)
        self.W_e = f(rng.standard_normal((hidden, seed_dim)) / math.sqrt(seed_dim))
        self.W_o = f(rng.standard_normal((bits_per_step, hidden)) / math.sqrt(hidden))
        self.b_h = f(rng.standard_normal(hidden) * 0.1)
        self.b_o = f(rng.standard_normal(bits_per_step) * 0.1)
        self.W_x = f(rng.standard_normal((hidden, bits_per_step)) / math.sqrt(bits_per_step) * 0.5)   # output feedback
        # generated (never stored)
        self.M = f(rng.choice([-1.0, 1.0], size=(max_blocks, block_steps, seed_dim)))
        self.P = f(rng.standard_normal((max_blocks, hidden)) * 0.5)
        self.C = f(rng.standard_normal((block_steps, hidden)) * 0.5)

    def params(self):
        return {"W_h": self.W_h, "W_e": self.W_e, "W_o": self.W_o, "b_h": self.b_h, "b_o": self.b_o, "W_x": self.W_x}

    def n_weight_params(self):
        return sum(int(p.size) for p in self.params().values())

    def _prep(self, S, B):
        xp = self.xp
        m, L = S.shape[0], self.arch["block_steps"]
        X = (S[:, None, None, :] * self.M[None, :B, :, :]).reshape(m * B, L, -1)   # masked seeds (mB, L, d)
        inj = X @ self.W_e.T                                                        # (mB, L, hidden)
        ctx = xp.tile(self.P[:B] + self.b_h[None, :], (m, 1))                       # (mB, hidden)
        return X, inj, ctx

    def forward(self, S, n_blocks, Xin=None, keep=False):
        """Output-feedback engine.  The bits emitted at step t-1 are fed back as input at step t
        (x_{-1} = 0 at the start of every block, so blocks stay independent / parallel).
        Xin (mB, L, bps) in {-1,+1}: teacher-forced previous-step bits (training); None = free-running.
        Returns logits (m, B, L, bps)."""
        xp = self.xp
        m, L = S.shape[0], self.arch["block_steps"]
        B = n_blocks
        X, inj, ctx = self._prep(S, B)
        H = xp.zeros((m * B, self.W_h.shape[0]), dtype=xp.float32)
        xprev = xp.zeros((m * B, self.arch["bits_per_step"]), dtype=xp.float32)
        Hs, Os, Xs = [], [], []
        for t in range(L):
            if Xin is not None and t > 0:
                xprev = Xin[:, t - 1, :]
            H = xp.tanh(H @ self.W_h.T + xprev @ self.W_x.T + ctx + inj[:, t, :] + self.C[t][None, :])
            O = H @ self.W_o.T + self.b_o
            Os.append(O)
            if keep: Hs.append(H); Xs.append(xprev)
            if Xin is None:
                xprev = xp.where(O > 0, 1.0, -1.0).astype(xp.float32)      # free-running feedback
        O = xp.stack(Os, axis=1).reshape(m, B, L, -1)
        return (O, Hs, X, Xs) if keep else O

    def loss_and_grads(self, S, Y, margin, train_weights=True):
        """Teacher-forced hinge loss over real bits (Y in {-1,0,+1}).
        Returns loss, grads dict (incl. 'S'), min-margin."""
        xp = self.xp
        m, B, L, bps = Y.shape
        Yf = Y.reshape(m * B, L, bps)
        Xin = xp.where(Yf == 0, 0.0, Yf).astype(xp.float32)               # previous true bits as +-1 (0 = padding)
        O, Hs, X, Xs = self.forward(S, B, Xin=Xin, keep=True)
        care = (Y != 0).astype(xp.float32); n_care = float(xp.sum(care))
        Z = (margin - Y * O) * care
        loss = float(xp.sum(xp.maximum(Z, 0))) / n_care
        dO = (-Y * (Z > 0).astype(xp.float32) / n_care).reshape(m * B, L, bps)
        g = {k: xp.zeros_like(v) for k, v in self.params().items()} if train_weights else {}
        gS = xp.zeros_like(S)
        g_next = xp.zeros_like(Hs[0])
        for t in range(L - 1, -1, -1):
            Ht = Hs[t]; Hprev = Hs[t - 1] if t > 0 else None
            if train_weights:
                g["W_o"] += dO[:, t, :].T @ Ht; g["b_o"] += xp.sum(dO[:, t, :], axis=0)
            gH = dO[:, t, :] @ self.W_o + g_next @ self.W_h
            gU = gH * (1 - Ht ** 2)                                       # (mB, hidden)
            if train_weights:
                if Hprev is not None: g["W_h"] += gU.T @ Hprev
                g["b_h"] += xp.sum(gU, axis=0)
                g["W_e"] += gU.T @ X[:, t, :]
                g["W_x"] += gU.T @ Xs[t]
            gX = (gU @ self.W_e).reshape(m, B, -1) * self.M[None, :B, t, :]  # (m, B, d)
            gS += xp.sum(gX, axis=1)
            g_next = gU
        g["S"] = gS
        mm = float(xp.min(xp.where(care > 0, Y * O, xp.inf)))
        return loss, g, mm

    def bits(self, S, n_blocks):
        """Free-running Manifestation (what the AlgorithmicDisk does at read time)."""
        return (to_np(self.xp, self.forward(S, n_blocks)) > 0).astype(int)

    def cast_fp16(self):
        """Round the stored weights to float16 (what the AlgorithmicDisk would hold)."""
        xp = self.xp
        for k, v in self.params().items():
            setattr(self, k, v.astype(xp.float16).astype(xp.float32))

# ----------------------------------------------------------------------------- Adam
class Adam:
    def __init__(self, xp, shapes, lr):
        self.xp, self.lr, self.t = xp, lr, 0
        self.m = {k: xp.zeros(s, dtype=xp.float32) for k, s in shapes.items()}
        self.v = {k: xp.zeros(s, dtype=xp.float32) for k, s in shapes.items()}
    def step(self, params, grads, keys=None):
        xp = self.xp; self.t += 1
        for k in (keys or params.keys()):
            g = grads[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g; self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            params[k] = params[k] - self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (xp.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)
        return params

# ----------------------------------------------------------------------------- experiment core
def make_targets(xp, chunks, L, bps):
    N = len(chunks[0]) * 8
    B = math.ceil(N / (L * bps))
    Y = []
    for c in chunks:
        bits = text_to_bits(c)
        Y.append([2 * b - 1 for b in bits] + [0] * (B * L * bps - N))
    return xp.asarray(Y, dtype=xp.float32).reshape(len(chunks), B, L, bps), N, B

def exact_rate(net, S, Y, N, q_bits):
    """Fraction of objects reproduced bit-exactly from QUANTISED seeds."""
    xp = net.xp
    Sq = xp.stack([dequantize_seed(xp, *quantize_seed(xp, S[i], q_bits)) for i in range(S.shape[0])])
    got = net.bits(Sq, Y.shape[1]).reshape(S.shape[0], -1)[:, :N]
    tgt = (to_np(xp, Y).reshape(S.shape[0], -1)[:, :N] > 0).astype(int)
    per = (got == tgt).all(axis=1)
    return float(per.mean()), int((got == tgt).sum()), int(tgt.size), Sq

def train(xp, chunks, seed_dim, hidden, q_bits, epochs, margin=0.5, lr_w=2e-3, lr_s=0.03, radius=1.5,
          batch=500, ste_from=0.5, log=print, seed=0, block_steps=8):
    import numpy as np
    L, bps = block_steps, 8
    Y, N, B = make_targets(xp, chunks, L, bps)
    Mn = len(chunks)
    net = TrainableANET(xp, seed_dim=seed_dim, hidden=hidden, max_blocks=B, block_steps=L)
    rng = np.random.default_rng(seed)
    S = xp.asarray(rng.standard_normal((Mn, seed_dim)), dtype=xp.float32)
    R = radius * math.sqrt(seed_dim)
    S = S * (R / xp.linalg.norm(S, axis=1, keepdims=True))
    optW = Adam(xp, {k: v.shape for k, v in net.params().items()}, lr_w)
    optS = Adam(xp, {"S": S.shape}, lr_s)
    t0 = time.time(); best = 0.0; hist = []
    for ep in range(1, epochs + 1):
        ste = ep > ste_from * epochs
        perm = rng.permutation(Mn); tot = 0.0
        for b0 in range(0, Mn, batch):
            idx = perm[b0:b0 + batch]; idx_x = xp.asarray(idx)
            Sb = S[idx_x]
            if ste:
                Sb_eval = xp.stack([dequantize_seed(xp, *quantize_seed(xp, Sb[i], q_bits)) for i in range(Sb.shape[0])])
            else:
                Sb_eval = Sb
            loss, g, mm = net.loss_and_grads(Sb_eval, Y[idx_x], margin)
            tot += loss * len(idx)
            p = net.params(); p = optW.step(p, g)
            for k, v in p.items(): setattr(net, k, v)
            # seeds: per-row Adam state (indexed)
            gS = g["S"]
            optS.t += 1
            optS.m["S"][idx_x] = 0.9 * optS.m["S"][idx_x] + 0.1 * gS
            optS.v["S"][idx_x] = 0.999 * optS.v["S"][idx_x] + 0.001 * gS * gS
            upd = lr_s * (optS.m["S"][idx_x] / (1 - 0.9 ** min(optS.t, ep))) / (xp.sqrt(optS.v["S"][idx_x] / (1 - 0.999 ** min(optS.t, ep))) + 1e-8)
            Sb = Sb - upd
            nrm = xp.linalg.norm(Sb, axis=1, keepdims=True)
            Sb = xp.where(nrm > R, Sb * (R / nrm), Sb)
            S[idx_x] = Sb
        if ep % 10 == 0 or ep == epochs:
            rate, nb_ok, nb_tot, _ = exact_rate(net, S, Y, N, q_bits)
            best = max(best, rate)
            hist.append(dict(epoch=ep, loss=tot / Mn, exact=rate, bit_acc=nb_ok / nb_tot, sec=round(time.time() - t0, 1)))
            log(f"    ep {ep:4d} loss {tot/Mn:.4f} exact {rate*100:5.1f}%  bit-acc {100*nb_ok/nb_tot:6.2f}%  "
                f"{'STE' if ste else '   '}  {time.time()-t0:6.0f}s")
            if rate == 1.0 and ste: break
    # final: fp16 weights, quantised seeds, then a short seed-only repair pass for any failures
    net.cast_fp16()
    rate, nb_ok, nb_tot, Sq = exact_rate(net, S, Y, N, q_bits)
    return net, S, Y, N, dict(exact=rate, bit_acc=nb_ok / nb_tot, epochs=ep, seconds=round(time.time() - t0, 1), hist=hist)

def write_new_objects(net, chunks, q_bits, iters=1500, margin=0.5, lr=0.05, radius=1.5, log=print, seed=1):
    """Write path on a FROZEN trained engine: discover Seeds for unseen chunks (batched gradient inversion)."""
    import numpy as np
    xp = net.xp
    L, bps = net.arch["block_steps"], 8
    Y, N, B = make_targets(xp, chunks, L, bps)
    Mn = len(chunks); d = net.arch["seed_dim"]
    rng = np.random.default_rng(seed)
    S = xp.asarray(rng.standard_normal((Mn, d)), dtype=xp.float32)
    R = radius * math.sqrt(d); S = S * (R / xp.linalg.norm(S, axis=1, keepdims=True))
    opt = Adam(xp, {"S": S.shape}, lr); t0 = time.time(); first = {}
    for it in range(1, iters + 1):
        Se = xp.stack([dequantize_seed(xp, *quantize_seed(xp, S[i], q_bits)) for i in range(Mn)]) if it > iters // 2 else S
        loss, g, mm = net.loss_and_grads(Se, Y, margin, train_weights=False)
        S = opt.step({"S": S}, g)["S"]
        nrm = xp.linalg.norm(S, axis=1, keepdims=True); S = xp.where(nrm > R, S * (R / nrm), S)
        if it % 100 == 0:
            rate, nb_ok, nb_tot, _ = exact_rate(net, S, Y, N, q_bits)
            log(f"    write iter {it:4d} loss {loss:.4f} exact {rate*100:5.1f}% bit-acc {100*nb_ok/nb_tot:6.2f}%")
            if rate == 1.0 and it > iters // 2: break
    rate, nb_ok, nb_tot, _ = exact_rate(net, S, Y, N, q_bits)
    return dict(exact=rate, bit_acc=nb_ok / nb_tot, iters=it, seconds=round(time.time() - t0, 1))

# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--corpus", default="alice.txt")
    ap.add_argument("--chunk-chars", type=int, default=29)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--q-bits", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 24, 32, 48])
    ap.add_argument("--M-sweep", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000])
    ap.add_argument("--M-for-dim-sweep", type=int, default=2000)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--out", default="amortize_results.json")
    args = ap.parse_args()
    xp, dev = get_backend(args.device)
    chunks_all = load_corpus(args.corpus, args.chunk_chars)
    print(f"[device {dev}] corpus {args.corpus}: {len(chunks_all)} chunks of {args.chunk_chars} chars")
    print(f"  e.g. {chunks_all[0]!r} | {chunks_all[1]!r}")
    N = args.chunk_chars * 8
    results = dict(device=dev, chunk_chars=args.chunk_chars, hidden=args.hidden, q_bits=args.q_bits, N_bits=N,
                   corpus_chunks=len(chunks_all), dim_sweep=[], M_sweep=[], holdout={})
    # baselines
    joined = "".join(chunks_all).encode()
    gz = len(gzip.compress(joined, 9)) * 8 / len(chunks_all)
    results["baselines"] = dict(raw_bits=N, gzip9_bits_per_chunk=round(gz, 1), untrained_engine_bits=1024)
    print(f"  baselines: raw {N} bits/chunk, gzip-9 {gz:.1f} bits/chunk, untrained engine 1024 bits/chunk")

    def fp(net, M, d):  # per-object footprint (bits) with fp16 weights
        return net.n_weight_params() * 16 / M + d * args.q_bits

    print("\n== A. seed-dim sweep at M =", args.M_for_dim_sweep)
    train_chunks = chunks_all[:args.M_for_dim_sweep]
    best_d = None
    for d in args.dims:
        print(f"  seed_dim {d}")
        net, S, Y, N_, st = train(xp, train_chunks, d, args.hidden, args.q_bits, args.epochs)
        row = dict(seed_dim=d, M=len(train_chunks), exact=st["exact"], bit_acc=st["bit_acc"], epochs=st["epochs"],
                   seconds=st["seconds"], weight_params=net.n_weight_params(),
                   footprint_bits_per_obj=round(fp(net, len(train_chunks), d), 1), seed_bits=d * args.q_bits)
        results["dim_sweep"].append(row); print("   ->", row)
        json.dump(results, open(args.out, "w"), indent=1)
        if st["exact"] == 1.0 and best_d is None: best_d = d
    if best_d is None: best_d = args.dims[-1]
    results["chosen_seed_dim"] = best_d

    print(f"\n== B. corpus-size sweep at seed_dim = {best_d}")
    for M in args.M_sweep:
        if M > len(chunks_all) - args.holdout: break
        print(f"  M {M}")
        net, S, Y, N_, st = train(xp, chunks_all[:M], best_d, args.hidden, args.q_bits, args.epochs)
        row = dict(M=M, seed_dim=best_d, exact=st["exact"], bit_acc=st["bit_acc"], epochs=st["epochs"], seconds=st["seconds"],
                   weight_params=net.n_weight_params(), footprint_bits_per_obj=round(fp(net, M, best_d), 1),
                   total_footprint_bytes=round((net.n_weight_params() * 16 + M * best_d * args.q_bits) / 8),
                   total_data_bytes=M * args.chunk_chars)
        results["M_sweep"].append(row); print("   ->", row)
        json.dump(results, open(args.out, "w"), indent=1)
        last_net = net
    print(f"\n== C. write path on unseen chunks (frozen engine from the largest M run), holdout = {args.holdout}")
    hold = chunks_all[-args.holdout:]
    results["holdout"] = write_new_objects(last_net, hold, args.q_bits)
    print("   ->", results["holdout"])
    json.dump(results, open(args.out, "w"), indent=1)
    print("\nsaved", args.out)

if __name__ == "__main__":
    main()
