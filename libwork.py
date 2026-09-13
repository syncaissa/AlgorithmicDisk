#!/usr/bin/env python3
"""
N7b — buy per-document Footprint with training Work.
Continue training the LIBRARY engine (engine_lib.json) on its library at a constant learning
rate; every `--every` minutes freeze an int8 copy, write the library document to a disk with it
(blocks mode) and record (minutes of Work, Seed bytes, ratio).  Output: libwork_results.json.
"""
import argparse, copy, json, lzma, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from algorithmic_storage import get_backend
from realfile import engine_from_json, engine_to_json, engine_bytes, encode_blocks, decode_blocks, gutenberg_body
from amortize import Adam

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto"); ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--every", type=float, default=15); ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--engine", default="engine_lib.json"); ap.add_argument("--out", default="libwork_results.json")
    args = ap.parse_args()
    xp, dev = get_backend(args.device); L = 256
    ej = json.load(open(args.engine)); net = engine_from_json(xp, ej)
    prior_min = (ej.get("hist") or [{}])[-1].get("sec", 0) / 60
    library = gutenberg_body("books/pg12.txt"); n = len(library) // L
    X = xp.asarray(np.frombuffer(library[:n * L], dtype=np.uint8).reshape(n, L).astype(np.int64))
    xz_ratio = len(lzma.compress(library, preset=9)) / len(library)
    opt = Adam(xp, {k: v.shape for k, v in net.params().items()}, args.lr)
    rng = np.random.default_rng(1); t0 = time.time(); ep = 0; next_eval = 0.0
    res = dict(device=dev, hidden=ej["arch"]["hidden"], engine_bytes=engine_bytes(ej), xz_ratio=round(xz_ratio, 4),
               library_bytes=len(library), prior_minutes=round(prior_min, 1), points=[])
    def evaluate(bpb):
        frozen = engine_from_json(xp, engine_to_json(net))          # int8 snapshot (engine_to_json quantises)
        seed, ideal = encode_blocks(frozen, library, L); ok = decode_blocks(frozen, seed, len(library), L) == library
        mins = prior_min + (time.time() - t0) / 60
        res["points"].append(dict(minutes=round(mins, 1), epoch=ep, train_bits_per_byte=round(bpb, 4), seed_bytes=len(seed),
                                  ratio=round(len(seed) / len(library), 5), exact=ok))
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"  [{mins:6.1f} min | ep {ep:4d}] train {bpb:.4f} b/B -> Seed {len(seed):,} B  ratio {len(seed)/len(library):.5f}x  exact {ok}", flush=True)
        json.dump({**engine_to_json(frozen), "hist": [dict(epoch=ep, bits_per_byte=round(bpb, 4), sec=round(mins * 60))]}, open("engine_lib_phase2.json", "w"))
    bpb = float("nan")
    while True:
        if (time.time() - t0) / 60 >= next_eval:
            if ep > 0 or next_eval == 0.0: evaluate(bpb if ep else float("nan"))
            next_eval += args.every
        if (time.time() - t0) / 60 >= args.minutes: break
        ep += 1; perm = rng.permutation(n); tot = 0.0
        for b0 in range(0, n, 64):
            idx = xp.asarray(perm[b0:b0 + 64]); loss, g, _ = net.loss_and_grads(X[idx]); tot += loss * len(idx)
            p = opt.step(net.params(), g)
            for k, v in p.items(): setattr(net, k, v)
        bpb = tot / n / math.log(2)
        if ep % 25 == 0: print(f"    ep {ep:4d} {bpb:.4f} bits/byte {(time.time()-t0)/60:5.1f} min", flush=True)
    evaluate(bpb)
    print("saved", args.out)

if __name__ == "__main__":
    main()
