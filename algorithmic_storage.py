#!/usr/bin/env python3
"""
Algorithmic Storage — reference demo  (paradigm)
  AlgorithmicDisk  — the device: a JSON file holding generator identities + Regeneration Manifests
  AlgorithmicNET   — the engine: a seeded, deterministic recurrent generator with feedback

Pipeline demonstrated here
  text -> bits -> Seed Discovery (gradient inversion, write time) -> Manifest stored on the disk
       -> Manifestation (regenerate bits from Seed, read time) -> verify hash -> text

Device flag:  --device cpu | cuda | auto
  The same code runs on CPU (numpy) or GPU (cupy, drop-in numpy API).  No other dependency.

Everything about the engine is derived from the Generator Identity (GID):
  weights are *generated* from a 32-bit init seed, so the weight Footprint is ~0 bytes.
"""
import argparse, base64, hashlib, json, math, os, sys, time

# ----------------------------------------------------------------------------- device backend
def get_backend(device: str):
    if device == "auto":
        try:
            import cupy  # noqa
            device = "cuda"
        except Exception:
            device = "cpu"
    if device == "cuda":
        import cupy as xp
        return xp, "cuda"
    import numpy as xp
    return xp, "cpu"

def to_np(xp, a):
    return xp.asnumpy(a) if hasattr(xp, "asnumpy") else a

# ----------------------------------------------------------------------------- bits <-> text
def text_to_bits(s: str):
    data = s.encode("utf-8")
    return [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]

def bits_to_text(bits):
    out = bytearray()
    for i in range(0, len(bits) - len(bits) % 8, 8):
        out.append(int("".join(str(b) for b in bits[i:i + 8]), 2))
    return out.decode("utf-8", errors="replace")

def bits_to_str(bits):
    return "".join(str(b) for b in bits)

# ----------------------------------------------------------------------------- AlgorithmicNET
class AlgorithmicNET:
    """
    Seeded deterministic generator with feedback (a small recurrent net).

      h_t^{(j)} = tanh(W_h h_{t-1} + W_e (s * m_{j,t}) + p_j + c_t + b_h)   h_{-1} = 0
                  feedback: state -> next state; the Seed s is injected at every step, modulated
                  by a context sign-mask m_{j,t} in {-1,+1}^seed_dim so that every (block, step)
                  context sees a different random projection of the Seed;
                  p_j = block code (parallel blocks), c_t = clock code (step within block)
      o_t       = W_o h_t + b_o                              (read-out: `bits_per_step` logits)
      bit       = 1[o_t > 0]

    All weights are *generated* from the 32-bit `init_seed`, so the Generator Identity (GID)
    is a handful of integers.  Nothing about the weights needs to be stored.
    Blocks are independent given (s, j), so blocks manifest in parallel (Compute <-> Horizon).
    """
    def __init__(self, xp, seed_dim=64, hidden=256, bits_per_step=8, block_steps=8,
                 max_blocks=64, init_seed=20260911, spectral=1.4):
        self.xp = xp
        self.arch = dict(seed_dim=seed_dim, hidden=hidden, bits_per_step=bits_per_step,
                         block_steps=block_steps, max_blocks=max_blocks, init_seed=init_seed,
                         spectral=spectral)
        import numpy as np
        rng = np.random.default_rng(init_seed)
        W_h = rng.standard_normal((hidden, hidden)) / math.sqrt(hidden) * spectral
        W_e = rng.standard_normal((hidden, seed_dim)) / math.sqrt(seed_dim)
        M   = rng.choice([-1.0, 1.0], size=(max_blocks, block_steps, seed_dim))   # context sign-masks on the Seed
        P   = rng.standard_normal((max_blocks, hidden)) * 0.5
        C   = rng.standard_normal((block_steps, hidden)) * 0.5        # per-step clock code
        b_h = rng.standard_normal(hidden) * 0.1
        W_o = rng.standard_normal((bits_per_step, hidden)) / math.sqrt(hidden)
        b_o = rng.standard_normal(bits_per_step) * 0.1
        f = lambda a: xp.asarray(a, dtype=xp.float32)
        self.W_h, self.W_e, self.M, self.P, self.C, self.b_h, self.W_o, self.b_o = map(f, (W_h, W_e, M, P, C, b_h, W_o, b_o))

    # -- identity -------------------------------------------------------------------------
    def gid(self):
        spec = json.dumps(self.arch, sort_keys=True).encode()
        return "ANET-" + hashlib.sha256(spec).hexdigest()[:16]

    # -- forward (Manifestation) -----------------------------------------------------------
    def forward(self, s, n_blocks, keep=False):
        """s: (seed_dim,) -> logits (n_blocks, block_steps, bits_per_step)."""
        xp = self.xp
        L, B = self.arch["block_steps"], n_blocks
        ctx = self.P[:B] + self.b_h[None, :]                                    # (B, hidden) block context
        inj = (self.M[:B] * s[None, None, :]) @ self.W_e.T                       # (B, L, hidden) masked Seed injection
        h = xp.zeros((B, self.W_h.shape[0]), dtype=xp.float32)
        hs, outs = [], []
        for t in range(L):
            h = xp.tanh(h @ self.W_h.T + ctx + inj[:, t, :] + self.C[t][None, :])
            outs.append(h @ self.W_o.T + self.b_o)
            if keep: hs.append(h)
        o = xp.stack(outs, axis=1)
        return (o, hs) if keep else o

    def manifest_bits(self, s, n_bits):
        L, bps = self.arch["block_steps"], self.arch["bits_per_step"]
        n_blocks = math.ceil(n_bits / (L * bps))
        o = self.forward(s, n_blocks)
        bits = (to_np(self.xp, o) > 0).astype(int).reshape(-1)[:n_bits]
        return bits.tolist(), to_np(self.xp, o).reshape(-1)[:n_bits]

    def manifest_bits_sequential(self, s, n_bits):
        """Same result, one block at a time (Horizon instead of Compute)."""
        xp = self.xp
        L, bps = self.arch["block_steps"], self.arch["bits_per_step"]
        n_blocks = math.ceil(n_bits / (L * bps))
        outs = []
        for j in range(n_blocks):
            ctx = self.P[j] + self.b_h
            h = xp.zeros(self.W_h.shape[0], dtype=xp.float32)
            for t in range(L):
                h = xp.tanh(self.W_h @ h + ctx + self.W_e @ (s * self.M[j, t]) + self.C[t])
                outs.append(self.W_o @ h + self.b_o)
        o = to_np(xp, xp.stack(outs)).reshape(-1)[:n_bits]
        return (o > 0).astype(int).tolist()

    # -- backward w.r.t. the Seed only (weights are frozen) ----------------------------------
    def seed_grad(self, s, y, margin):
        """y: (n_blocks, block_steps, bits_per_step) in {-1,+1} (0 = don't care).
        Hinge loss  mean( max(0, margin - y*o) )  over real bits: satisfied bits stop pulling,
        which keeps the Seed norm (and therefore tanh saturation) under control.
        Returns (loss, grad_s, min_margin)."""
        xp = self.xp
        o, hs = self.forward(s, y.shape[0], keep=True)
        care = (y != 0).astype(xp.float32)
        n_care = float(xp.sum(care))
        z = (margin - y * o) * care
        loss = float(xp.sum(xp.maximum(z, 0))) / n_care
        dL_do = (-y * (z > 0).astype(xp.float32)) / n_care                   # (B, L, bps)
        L, B = self.arch["block_steps"], y.shape[0]
        g_s = xp.zeros_like(s)
        g_next = xp.zeros_like(hs[0])                                        # grad wrt u_{t+1}
        for t in range(L - 1, -1, -1):
            g_h = dL_do[:, t, :] @ self.W_o + g_next @ self.W_h              # (B, hidden)
            g_u = g_h * (1 - hs[t] ** 2)
            g_s = g_s + xp.sum((g_u @ self.W_e) * self.M[:B, t, :], axis=0)   # injection = W_e (s * m_{j,t})
            g_next = g_u
        mm = float(xp.min(xp.where(care > 0, y * o, xp.inf)))
        return loss, g_s, mm

# ----------------------------------------------------------------------------- Seed Discovery
def quantize_seed(xp, s, bits=8):
    """Symmetric uniform quantisation of the Seed to `bits` bits per dimension.
    bits=1 is a sign (binary) Seed: q in {-1,+1}."""
    amax = float(xp.max(xp.abs(s)))
    if bits == 1:
        q = xp.where(s >= 0, 1.0, -1.0)
        scale = float(xp.mean(xp.abs(s))) if amax > 0 else 1.0
        return q.astype(xp.int32), scale
    qmax = 2 ** (bits - 1) - 1
    scale = amax / qmax if amax > 0 else 1.0
    q = xp.clip(xp.rint(s / scale), -qmax, qmax)
    return q.astype(xp.int32), scale

def dequantize_seed(xp, q, scale):
    return (q.astype(xp.float32) * scale)

def pack_seed(q_np, bits):
    """Pack integer Seed levels into a byte string using exactly `bits` bits per dimension."""
    import numpy as np
    if bits == 1:
        u = ((q_np + 1) // 2).astype(np.uint8)
        return np.packbits(u).tobytes()
    qmax = 2 ** (bits - 1) - 1
    u = (q_np + qmax).astype(np.uint16)                      # 0 .. 2*qmax  (fits in `bits` bits)
    bitplane = ((u[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bitplane).tobytes()

def unpack_seed(buf, bits, dim):
    import numpy as np
    plane = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))
    if bits == 1:
        return plane[:dim].astype(np.int32) * 2 - 1
    plane = plane[:dim * bits].reshape(dim, bits)
    u = (plane * (1 << np.arange(bits - 1, -1, -1))).sum(axis=1)
    return (u - (2 ** (bits - 1) - 1)).astype(np.int32)

def discover_seed(net, bits, margin=0.5, q_bits=8, max_iters=3000, lr=0.05,
                  restarts=4, radius=1.5, verbose=True, rng_seed=0):
    """Gradient inversion: find s such that Man(s)[:N] == bits, robust to quantisation.
    Adam on the Seed only; after every step the Seed is projected back onto the ball
    ||s|| <= radius*sqrt(seed_dim) (keeps the engine out of tanh saturation).
    Acceptance requires the *quantised* Seed to reproduce every bit with a safety margin.
    Returns (q, scale, stats) or (None, None, stats)."""
    xp = net.xp
    L, bps = net.arch["block_steps"], net.arch["bits_per_step"]
    N = len(bits)
    n_blocks = math.ceil(N / (L * bps))
    y_flat = [2 * b - 1 for b in bits] + [0] * (n_blocks * L * bps - N)       # 0 = don't care
    y = xp.asarray(y_flat, dtype=xp.float32).reshape(n_blocks, L, bps)
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    R = radius * math.sqrt(net.arch["seed_dim"])
    t0 = time.time(); total_iters = 0; best = 0
    for r in range(restarts):
        s = xp.asarray(rng.standard_normal(net.arch["seed_dim"]), dtype=xp.float32)
        s = s * (R / float(xp.linalg.norm(s)))
        m = xp.zeros_like(s); v = xp.zeros_like(s)
        for it in range(1, max_iters + 1):
            total_iters += 1
            if q_bits <= 4:                                   # straight-through: gradient at the quantised Seed
                q_, sc_ = quantize_seed(xp, s, q_bits)
                loss, g, mm = net.seed_grad(dequantize_seed(xp, q_, sc_), y, margin)
            else:
                loss, g, mm = net.seed_grad(s, y, margin)
            # Adam
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            s = s - lr * (m / (1 - 0.9 ** it)) / (xp.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
            n = float(xp.linalg.norm(s))
            if n > R: s = s * (R / n)
            if it % 25 == 0 or mm > 0:
                # quantisation-aware check: does the *quantised* Seed reproduce every bit with margin?
                q, scale = quantize_seed(xp, s, q_bits)
                sq = dequantize_seed(xp, q, scale)
                got, logits = net.manifest_bits(sq, N)
                match = sum(int(a == b) for a, b in zip(got, bits)); best = max(best, match)
                if verbose and it % 100 == 0:
                    print(f"  restart {r} iter {it:5d} loss {loss:.4f} min-margin {mm:+.3f} "
                          f"quantised-match {match}/{N}")
                if match == N and float(np.min(np.abs(logits))) > margin * 0.25:
                    stats = dict(restart=r, iters=total_iters, seconds=round(time.time() - t0, 2),
                                 min_abs_logit=float(np.min(np.abs(logits))), best_match=N)
                    return q, scale, stats
    return None, None, dict(restart=restarts, iters=total_iters, seconds=round(time.time() - t0, 2), best_match=best)

# ----------------------------------------------------------------------------- AlgorithmicDisk
class AlgorithmicDisk:
    """A JSON file: generator identities (architecture + init seed) and Regeneration Manifests."""
    FORMAT = "AlgorithmicDisk/0.1"

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.d = json.load(open(path))
        else:
            self.d = {"format": self.FORMAT, "generators": {}, "manifests": []}

    def save(self):
        json.dump(self.d, open(self.path, "w"), indent=1)

    def register(self, net):
        self.d["generators"][net.gid()] = net.arch
        return net.gid()

    def write(self, net, name, q, scale, n_bits, digest, q_bits, stats):
        import numpy as np
        qn = to_np(net.xp, q).astype(np.int32)
        rm = {"id": name, "gid": net.gid(), "n_bits": n_bits, "sha256": digest,
              "seed_q_bits": q_bits, "seed_dim": int(qn.size), "seed_scale": scale,
              "seed_b64": base64.b64encode(pack_seed(qn, q_bits)).decode(),
              "discovery": stats}
        self.d["manifests"] = [m for m in self.d["manifests"] if m["id"] != name] + [rm]
        return rm

    def read(self, xp, name, sequential=False):
        import numpy as np
        rm = next(m for m in self.d["manifests"] if m["id"] == name)
        net = AlgorithmicNET(xp, **self.d["generators"][rm["gid"]])
        qn = unpack_seed(base64.b64decode(rm["seed_b64"]), rm["seed_q_bits"], rm["seed_dim"]).astype(np.float32)
        s = xp.asarray(qn * rm["seed_scale"], dtype=xp.float32)
        if sequential:
            bits = net.manifest_bits_sequential(s, rm["n_bits"])
        else:
            bits, _ = net.manifest_bits(s, rm["n_bits"])
        ok = hashlib.sha256(bits_to_str(bits).encode()).hexdigest() == rm["sha256"]
        return bits, ok, rm

    def footprint_report(self):
        gens = sum(len(json.dumps(a)) for a in self.d["generators"].values())
        seeds = sum(len(base64.b64decode(m["seed_b64"])) for m in self.d["manifests"])
        apparent = sum(m["n_bits"] for m in self.d["manifests"]) / 8
        return dict(generator_spec_bytes=gens, seed_bytes_total=seeds,
                    objects=len(self.d["manifests"]), apparent_capacity_bytes=apparent)

# ----------------------------------------------------------------------------- commands
def cmd_write(args):
    xp, dev = get_backend(args.device)
    text = args.text
    bits = text_to_bits(text)
    print(f"[device {dev}]  text: {text!r}")
    print(f"binary ({len(bits)} bits): {bits_to_str(bits)}")
    net = AlgorithmicNET(xp, seed_dim=args.seed_dim, hidden=args.hidden,
                         block_steps=args.block_steps, init_seed=args.init_seed)
    print(f"AlgorithmicNET GID {net.gid()}  arch {net.arch}")
    print("Seed Discovery (gradient inversion, weights frozen) ...")
    q, scale, stats = discover_seed(net, bits, margin=args.margin, q_bits=args.q_bits,
                                    max_iters=args.iters, lr=args.lr, restarts=args.restarts,
                                    radius=args.radius, verbose=not args.quiet)
    if q is None:
        print("FAILED to discover a Seed with these settings:", stats); return 1
    digest = hashlib.sha256(bits_to_str(bits).encode()).hexdigest()
    disk = AlgorithmicDisk(args.disk)
    disk.register(net)
    rm = disk.write(net, args.name, q, scale, len(bits), digest, args.q_bits, stats)
    disk.save()
    seed_bytes = len(base64.b64decode(rm["seed_b64"]))
    print(f"Seed found in {stats['iters']} iterations / {stats['seconds']:.1f}s  "
          f"(restart {stats['restart']}, min |logit| {stats['min_abs_logit']:.3f})")
    print(f"Manifest written to {args.disk}: id={args.name} gid={rm['gid']} "
          f"Seed={seed_bytes} bytes ({seed_bytes*8} bits) for {len(bits)} data bits")
    return 0

def cmd_read(args):
    xp, dev = get_backend(args.device)
    disk = AlgorithmicDisk(args.disk)
    t0 = time.time()
    bits, ok, rm = disk.read(xp, args.name, sequential=args.sequential)
    dt = time.time() - t0
    text = bits_to_text(bits)
    print(f"[device {dev}] Manifestation of {args.name} ({'sequential' if args.sequential else 'block-parallel'}) "
          f"in {dt*1000:.1f} ms")
    print(f"binary: {bits_to_str(bits)}")
    print(f"text  : {text!r}")
    print(f"sha256 verification: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1

def cmd_demo(args):
    """Full round trip on the canonical phrase, then a Seed-size sweep."""
    phrase = args.text
    disk = args.disk
    if os.path.exists(disk): os.remove(disk)
    print("=" * 78); print("STEP 1  write: text -> bits -> Seed Discovery -> AlgorithmicDisk"); print("=" * 78)
    a = argparse.Namespace(**vars(args)); a.name = "phrase-1"
    if cmd_write(a): return 1
    print(); print("=" * 78); print("STEP 2  read: AlgorithmicDisk -> Manifestation -> verify -> text"); print("=" * 78)
    a.sequential = False; cmd_read(a)
    a.sequential = True;  cmd_read(a)
    xp, _ = get_backend(args.device)
    bits, ok, rm = AlgorithmicDisk(disk).read(xp, "phrase-1")
    print(f"\nRound trip exact: {bits_to_text(bits) == phrase}   ({len(phrase)} chars, {len(bits)} bits)")
    print("Disk footprint:", AlgorithmicDisk(disk).footprint_report())
    if args.sweep:
        print(); print("=" * 78); print("STEP 3  Seed-size sweep (Footprint vs write-time Work)"); print("=" * 78)
        print(f"{'seed_dim':>8} {'q_bits':>6} {'Seed bits':>9} {'data bits':>9} {'ratio':>6} {'found':>5} {'best':>5} {'iters':>6} {'sec':>6}")
        bitsv = text_to_bits(phrase)
        for sd in args.sweep_dims:
            for qb in args.sweep_qbits:
                net = AlgorithmicNET(xp, seed_dim=sd, hidden=args.hidden, block_steps=args.block_steps,
                                     init_seed=args.init_seed)
                q, scale, st = discover_seed(net, bitsv, margin=args.margin, q_bits=qb, max_iters=args.iters,
                                             lr=args.lr, restarts=args.restarts, radius=args.radius, verbose=False)
                sb = sd * qb
                print(f"{sd:>8} {qb:>6} {sb:>9} {len(bitsv):>9} {sb/len(bitsv):>6.2f} "
                      f"{'yes' if q is not None else 'no':>5} {st['best_match']:>5} {st['iters']:>6} {st['seconds']:>6.1f}")
    return 0

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--disk", default="demo.adisk")
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(sp):
        sp.add_argument("--seed-dim", type=int, default=256)
        sp.add_argument("--hidden", type=int, default=256)
        sp.add_argument("--block-steps", type=int, default=8, help="steps (chars) per parallel block")
        sp.add_argument("--init-seed", type=int, default=20260911, help="32-bit seed that generates the weights")
        sp.add_argument("--margin", type=float, default=0.5)
        sp.add_argument("--radius", type=float, default=1.5, help="Seed norm bound, in units of sqrt(seed_dim)")
        sp.add_argument("--q-bits", type=int, default=8, help="bits per Seed dimension after quantisation")
        sp.add_argument("--iters", type=int, default=3000)
        sp.add_argument("--lr", type=float, default=0.05)
        sp.add_argument("--restarts", type=int, default=3)
        sp.add_argument("--quiet", action="store_true")
    w = sub.add_parser("write"); common(w)
    w.add_argument("--text", required=True); w.add_argument("--name", default="obj-1")
    r = sub.add_parser("read"); r.add_argument("--name", default="obj-1"); r.add_argument("--sequential", action="store_true")
    d = sub.add_parser("demo"); common(d)
    d.add_argument("--text", default="The fox ran over the lazy dog")
    d.add_argument("--sweep", action="store_true")
    d.add_argument("--sweep-dims", type=int, nargs="+", default=[128, 256, 384, 512])
    d.add_argument("--sweep-qbits", type=int, nargs="+", default=[8, 4, 2, 1])
    args = p.parse_args()
    if args.cmd == "read": args.sequential = getattr(args, "sequential", False)
    return {"write": cmd_write, "read": cmd_read, "demo": cmd_demo}[args.cmd](args)

if __name__ == "__main__":
    sys.exit(main())
