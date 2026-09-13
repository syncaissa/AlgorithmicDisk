#!/usr/bin/env python3
"""
Experiment N7 — DOMAIN ENGINES ON ONE ALGORITHMICDISK.

One disk, several generators; the Regeneration Manifest's GID selects the engine per object.
At write time the disk tries every applicable engine and keeps the smallest Seed.

Engines on this disk
  ANET-EN    predictive engine trained on English books      (engine_books.json, Session 5)
  ANET-CODE  predictive engine trained on Python source      (the CPython standard library)
  ANET-LIB   "library" engine: the English engine fine-tuned on a specific document library
             until its surprisal is tiny -> per-document Seeds far below 0.1x. The engine is
             the library's cost, paid once; new revisions of library documents cost only their
             edits.
  ANET-PROC  procedural engine: a deterministic simulator (Lorenz system, RK4, CSV output).
             Seed = the simulator's parameters (provenance capture, strategy D0). This is DEEP
             DATA: megabytes regenerated from ~100 bytes.
  VERBATIM   the escape (Theorem ceiling): a block/object nothing can shorten is stored raw.

Footprint accounting is reported two ways: per-object (Seed only; engines shared and paid
once) and total (Seed + engine), so the reader can see exactly when 0.01x is real.
"""
import argparse, base64, glob, hashlib, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from algorithmic_storage import get_backend
from realfile import (train_engine, engine_to_json, engine_from_json, engine_bytes, encode_stream, decode_stream,
                      encode_blocks, decode_blocks, gutenberg_body)
from predictive import PredictiveANET
from amortize import Adam

# ----------------------------------------------------------------------------- procedural engine (deep data)
def lorenz_csv(params):
    """Deterministic simulator: Lorenz system, RK4, float64, CSV text. Seed = params (dict)."""
    s, r, b = params["sigma"], params["rho"], params["beta"]
    x, y, z = params["x0"], params["y0"], params["z0"]
    dt, n = params["dt"], params["steps"]
    def f(x, y, z): return s * (y - x), x * (r - z) - y, x * y - b * z
    lines = ["t,x,y,z"]
    for i in range(n):
        k1 = f(x, y, z)
        k2 = f(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], z + 0.5 * dt * k1[2])
        k3 = f(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2])
        k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        lines.append(f"{(i+1)*dt:.6f},{x:.12f},{y:.12f},{z:.12f}")
    return ("\n".join(lines) + "\n").encode()

PROC_GID = "ANET-PROC-lorenz-rk4-v1"

# ----------------------------------------------------------------------------- library engine (fine-tune to memorise)
def finetune_library(xp, base_json, library: bytes, L, minutes, target_bpb, log=print, lr=2e-3, batch=64, hidden=None):
    """Continue training an engine on a specific library until surprisal <= target or time is up.
    If `hidden` differs from the base engine, a fresh engine of that size is trained instead."""
    if hidden and hidden != base_json["arch"]["hidden"]:
        net = PredictiveANET(xp, hidden=hidden, emb=base_json["arch"]["emb"]); log(f"    fresh library engine hidden {hidden}")
    else:
        net = engine_from_json(xp, base_json)
    n = len(library) // L
    X = xp.asarray(np.frombuffer(library[:n * L], dtype=np.uint8).reshape(n, L).astype(np.int64))
    opt = Adam(xp, {k: v.shape for k, v in net.params().items()}, lr)
    rng = np.random.default_rng(0); t0 = time.time(); hist = []; ep = 0
    while True:
        ep += 1; perm = rng.permutation(n); tot = 0.0
        for b0 in range(0, n, batch):
            idx = xp.asarray(perm[b0:b0 + batch]); loss, g, _ = net.loss_and_grads(X[idx]); tot += loss * len(idx)
            p = opt.step(net.params(), g)
            for k, v in p.items(): setattr(net, k, v)
        bpb = tot / n / math.log(2); hist.append(dict(epoch=ep, bits_per_byte=round(bpb, 4), sec=round(time.time() - t0)))
        if ep % 5 == 0 or bpb <= target_bpb: log(f"    lib ep {ep:3d}  {bpb:.4f} bits/byte  {time.time()-t0:5.0f}s")
        if bpb <= target_bpb or time.time() - t0 > minutes * 60: break
        if ep == 40: opt.lr = lr * 0.5
        if ep == 80: opt.lr = lr * 0.25
    net.cast_int8()
    return net, hist

# ----------------------------------------------------------------------------- multi-engine disk
class MultiDisk:
    def __init__(self, path):
        self.path = path
        self.d = {"format": "AlgorithmicDisk/0.3", "generators": {}, "manifests": []}
        self.nets = {}
    def add_neural(self, xp, gid, ej):
        self.d["generators"][gid] = {"kind": "predictive", **{k: ej[k] for k in ("arch", "weights")}}
        self.nets[gid] = engine_from_json(xp, ej)
    def add_procedural(self):
        self.d["generators"][PROC_GID] = {"kind": "procedural", "program": "lorenz_csv (RK4, float64, CSV)"}
    def save(self): json.dump(self.d, open(self.path, "w"))
    def engine_bytes(self, gid):
        g = self.d["generators"][gid]
        return engine_bytes(g) if g["kind"] == "predictive" else len(g["program"])

    def write(self, name, data: bytes, L=256, provenance=None, candidates=None, log=print):
        """Try every applicable engine; keep the smallest Seed. Returns the manifest and the trial table."""
        n = len(data); digest = hashlib.sha256(data).hexdigest(); trials = []
        best = None
        if provenance is not None:                                     # D0 provenance capture
            seed = json.dumps(provenance, sort_keys=True).encode(); t0 = time.time()
            ok = hashlib.sha256(lorenz_csv(provenance)).hexdigest() == digest; t = time.time() - t0
            trials.append(dict(gid=PROC_GID, seed_bytes=len(seed), seconds=round(t, 1), exact=ok))
            if ok: best = (PROC_GID, seed, "procedural")
        for gid in (candidates or [g for g in self.nets]):
            t0 = time.time(); seed, ideal = encode_blocks(self.nets[gid], data, L); t = time.time() - t0
            trials.append(dict(gid=gid, seed_bytes=len(seed), ideal_bytes=round(ideal / 8), seconds=round(t, 1)))
            log(f"      try {gid:12s} -> {len(seed):>9,} B  ({t:.1f}s)")
            if best is None or len(seed) < len(best[1]): best = (gid, seed, "blocks")
        raw = n + 2 * math.ceil(n / L)
        trials.append(dict(gid="VERBATIM", seed_bytes=raw, seconds=0.0))
        if len(best[1]) >= raw: best = ("VERBATIM", data, "raw")     # ties go to the escape: no engine needed
        gid, seed, mode = best
        rm = dict(id=name, gid=gid, mode=mode, n_bytes=n, sha256=digest, block_len=L,
                  seed_b64=base64.b64encode(seed).decode(), seed_bytes=len(seed), trials=trials)
        self.d["manifests"] = [m for m in self.d["manifests"] if m["id"] != name] + [rm]
        return rm

    def read(self, name):
        rm = next(m for m in self.d["manifests"] if m["id"] == name); seed = base64.b64decode(rm["seed_b64"]); t0 = time.time()
        if rm["mode"] == "procedural": data = lorenz_csv(json.loads(seed))
        elif rm["mode"] == "raw": data = seed
        else: data = decode_blocks(self.nets[rm["gid"]], seed, rm["n_bytes"], rm["block_len"])
        return data, hashlib.sha256(data).hexdigest() == rm["sha256"], time.time() - t0, rm

# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lib-minutes", type=float, default=40); ap.add_argument("--lib-target", type=float, default=0.06)
    ap.add_argument("--lib-hidden", type=int, default=384)
    ap.add_argument("--code-epochs", type=int, default=10); ap.add_argument("--code-bytes", type=int, default=2_000_000)
    ap.add_argument("--out", default="multiengine_results.json"); ap.add_argument("--disk", default="multi.adisk")
    args = ap.parse_args()
    xp, dev = get_backend(args.device); L = 256
    res = dict(device=dev, engines={}, objects=[])

    # --- engines -------------------------------------------------------------------------
    en = json.load(open("engine_books.json"))
    if not os.path.exists("engine_code.json"):
        import sysconfig
        src = b"".join(open(p, "rb").read() for p in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))))
        print(f"[device {dev}] training CODE engine on {min(len(src), args.code_bytes):,} bytes of CPython stdlib source")
        net, hist = train_engine(xp, src, 128, 16, L, args.code_epochs, max_bytes=args.code_bytes)
        ej = engine_to_json(net); ej["hist"] = hist; ej["train"] = "CPython 3.10 stdlib *.py"; json.dump(ej, open("engine_code.json", "w"))
    code = json.load(open("engine_code.json"))
    library = gutenberg_body("books/pg12.txt")                                    # the document library: Looking-Glass
    if not os.path.exists("engine_lib.json"):
        print(f"[device {dev}] fine-tuning LIBRARY engine on {len(library):,} bytes (target {args.lib_target} bits/byte, "
              f"budget {args.lib_minutes} min, hidden {args.lib_hidden})")
        net, hist = finetune_library(xp, en, library, L, args.lib_minutes, args.lib_target, hidden=args.lib_hidden)
        ej = engine_to_json(net); ej["hist"] = hist; ej["library"] = "books/pg12.txt body"; json.dump(ej, open("engine_lib.json", "w"))
    lib = json.load(open("engine_lib.json"))

    disk = MultiDisk(args.disk)
    for gid, ej in (("ANET-EN", en), ("ANET-CODE", code), ("ANET-LIB", lib)): disk.add_neural(xp, gid, ej)
    disk.add_procedural()
    for gid in disk.d["generators"]:
        res["engines"][gid] = dict(bytes=disk.engine_bytes(gid), kind=disk.d["generators"][gid]["kind"],
                                   hist=(en if gid == "ANET-EN" else code if gid == "ANET-CODE" else lib if gid == "ANET-LIB" else {}).get("hist", [])[-1:] )
        print(f"  engine {gid:12s} {res['engines'][gid]['bytes']:>9,} B  {res['engines'][gid]['kind']}")

    # --- objects ------------------------------------------------------------------------
    proc_params = dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0, x0=1.0, y0=1.0, z0=1.0, dt=0.001, steps=20_000)
    t0 = time.time(); sim = lorenz_csv(proc_params); print(f"\n  simulated Lorenz CSV: {len(sim):,} bytes in {time.time()-t0:.1f}s")
    open("lorenz.csv", "wb").write(sim)
    # a revised library document: Looking-Glass with 60 edited words (a "manual update")
    words = library.split(b" "); rng = np.random.default_rng(7)
    for i in rng.choice(len(words), 60, replace=False): words[i] = b"[REVISED]"
    revised = b" ".join(words); open("pg12_revised.txt", "wb").write(revised)
    objects = [("lorenz.csv", sim, proc_params), ("pg12.txt", library, None), ("pg12_revised.txt", revised, None),
               ("pg84_frankenstein.txt", gutenberg_body("books/pg84.txt")[:200_000], None),
               ("algorithmic_storage.py", open("algorithmic_storage.py", "rb").read(), None),
               ("image.png", open("image.png", "rb").read(), None)]
    import gzip, lzma
    for name, data, prov in objects:
        print(f"\n== {name}: {len(data):,} bytes   gzip {len(gzip.compress(data, 9)):,}  xz {len(lzma.compress(data, preset=9)):,}")
        rm = disk.write(name, data, L, provenance=prov); disk.save()
        back, ok, t_read, _ = disk.read(name)
        eb = disk.engine_bytes(rm["gid"]) if rm["gid"] != "VERBATIM" else 0
        row = dict(file=name, bytes=len(data), gzip=len(gzip.compress(data, 9)), xz=len(lzma.compress(data, preset=9)),
                   chosen=rm["gid"], mode=rm["mode"], seed_bytes=rm["seed_bytes"], ratio=round(rm["seed_bytes"] / len(data), 5),
                   engine_bytes=eb, total_ratio=round((rm["seed_bytes"] + eb) / len(data), 4), read_s=round(t_read, 1), exact=ok, trials=rm["trials"])
        res["objects"].append(row); json.dump(res, open(args.out, "w"), indent=1)
        print(f"   chosen {rm['gid']} ({rm['mode']}): Seed {rm['seed_bytes']:,} B  ratio {row['ratio']:.5f}x  "
              f"(+engine {eb:,} B -> total {row['total_ratio']:.4f}x)  read {t_read:.1f}s  sha256 {'PASS' if ok else 'FAIL'}")
    print(f"\ndisk {args.disk}: {os.path.getsize(args.disk):,} bytes; saved {args.out}")

if __name__ == "__main__":
    main()
