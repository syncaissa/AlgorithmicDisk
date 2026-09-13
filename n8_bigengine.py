#!/usr/bin/env python3
"""
N8 — an English engine that beats xz on its domain.
Clean split: TRAIN on 15 Gutenberg books (~13 MB); TEST on two books never seen in training
(pg12 Through the Looking-Glass, pg84 Frankenstein). Larger engine (hidden 256, emb 32),
longer context blocks (L=512), longer training.  Evaluation with the float engine (stream mode)
and with the compiled fixed-point engine (blocks mode), against gzip -9 / bz2 / xz -9.
"""
import argparse, bz2, glob, gzip, json, lzma, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from algorithmic_storage import get_backend
from realfile import gutenberg_body, train_engine, engine_to_json, engine_from_json, engine_bytes, encode_stream, decode_stream

TEST = ["books/pg12.txt", "books/pg84.txt"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--block", type=int, default=512); ap.add_argument("--epochs", type=int, default=14); ap.add_argument("--train-bytes", type=int, default=12_000_000)
    ap.add_argument("--engine", default="engine_big.json"); ap.add_argument("--out", default="n8_results.json")
    args = ap.parse_args(); xp, dev = get_backend(args.device); L = args.block
    train_files = sorted(f for f in glob.glob("books/pg*.txt") if f not in TEST) + ["alice.txt"]
    if os.path.exists(args.engine):
        ej = json.load(open(args.engine)); net = engine_from_json(xp, ej); print("loaded", args.engine)
    else:
        corpus = b"".join(gutenberg_body(f) for f in train_files)
        print(f"[device {dev}] training hidden {args.hidden} emb {args.emb} block {L} on {min(len(corpus), args.train_bytes):,} bytes from {len(train_files)} books, {args.epochs} epochs", flush=True)
        def log(s): print(s, flush=True)
        # learning-rate schedule: 3e-3 for the first 60% of epochs, then 1e-3
        net, hist = train_engine(xp, corpus, args.hidden, args.emb, L, int(args.epochs * 0.6), lr=3e-3, log=log, max_bytes=args.train_bytes)
        from amortize import Adam
        # continue at lower lr (train_engine builds its own optimiser; re-enter with the trained net)
        from realfile import windows
        X = xp.asarray(windows(corpus[:args.train_bytes], L)); opt = Adam(xp, {k: v.shape for k, v in net.params().items()}, 1e-3)
        # un-quantise not needed: train_engine cast to int8 at the end -> rebuild a float net from the json to continue
        net = engine_from_json(xp, engine_to_json(net)); opt = Adam(xp, {k: v.shape for k, v in net.params().items()}, 1e-3)
        rng = np.random.default_rng(1); t0 = time.time()
        for ep in range(int(args.epochs * 0.6) + 1, args.epochs + 1):
            perm = rng.permutation(X.shape[0]); tot = 0.0
            for b0 in range(0, X.shape[0], 128):
                idx = xp.asarray(perm[b0:b0 + 128]); loss, g, _ = net.loss_and_grads(X[idx]); tot += loss * len(idx)
                p = opt.step(net.params(), g)
                for k, v in p.items(): setattr(net, k, v)
            bpb = tot / X.shape[0] / math.log(2); hist.append(dict(epoch=ep, bits_per_byte=round(bpb, 3), sec=round(time.time() - t0)))
            log(f"    ep {ep:3d}  train {bpb:.3f} bits/byte  (lr 1e-3)  {time.time()-t0:5.0f}s")
        net.cast_int8(); ej = engine_to_json(net); ej["hist"] = hist; ej["train_books"] = train_files; ej["train_bytes"] = min(len(corpus), args.train_bytes)
        json.dump(ej, open(args.engine, "w"))
    wb = engine_bytes(ej); print(f"engine: {net.n_params():,} params -> {wb:,} B int8", flush=True)
    res = dict(engine=dict(hidden=args.hidden, emb=args.emb, block=L, bytes=wb, params=net.n_params(), train_books=len(train_files), train_bytes=ej.get("train_bytes")), tests=[])
    from fastcoder import FastFixedEngine
    fast = FastFixedEngine(ej)
    for f in TEST:
        data = gutenberg_body(f); n = len(data)
        base = dict(gzip9=len(gzip.compress(data, 9)), bz2=len(bz2.compress(data, 9)), xz=len(lzma.compress(data, preset=9)))
        t0 = time.time(); seed_f, ideal = encode_stream(net, data, L); t_f = time.time() - t0
        ok_f = decode_stream(net, seed_f, n, L) == data
        t0 = time.time(); seed_c = fast.encode_blocks(data, L=L, threads=2); t_c = time.time() - t0
        t0 = time.time(); ok_c = fast.decode_blocks(seed_c, n, L=L, threads=2) == data; t_cd = time.time() - t0
        row = dict(file=os.path.basename(f), bytes=n, baselines=base,
                   float_stream=dict(seed=len(seed_f), ratio=round(len(seed_f) / n, 4), bpb=round(len(seed_f) * 8 / n, 3), exact=ok_f, write_s=round(t_f, 1)),
                   fixed_blocks_C=dict(seed=len(seed_c), ratio=round(len(seed_c) / n, 4), bpb=round(len(seed_c) * 8 / n, 3), exact=ok_c, write_s=round(t_c, 1), read_s=round(t_cd, 1)),
                   beats_xz=len(seed_f) < base["xz"], beats_bz2=len(seed_f) < base["bz2"])
        res["tests"].append(row); json.dump(res, open(args.out, "w"), indent=1)
        print(f"{row['file']}: {n:,} B | gzip {base['gzip9']:,} bz2 {base['bz2']:,} xz {base['xz']:,} | float stream Seed {len(seed_f):,} ({len(seed_f)*8/n:.3f} b/B, exact {ok_f}) | "
              f"C fixed blocks Seed {len(seed_c):,} ({len(seed_c)*8/n:.3f} b/B, exact {ok_c}, read {t_cd:.1f}s) | beats xz: {len(seed_f) < base['xz']}  beats bz2: {len(seed_f) < base['bz2']}", flush=True)
    print("saved", args.out)

if __name__ == "__main__":
    main()
