#!/usr/bin/env python3
"""
N2b — 100 MB+ DEEP-DATA OBJECTS ON A SPACECRAFT BUDGET.

Two mission-like objects are stored on an AlgorithmicDisk as procedural Manifests and regenerated:
  telemetry.csv   ~100 MB of simulated attitude/dynamics telemetry: a Lorenz system integrated in
                  FIXED-POINT INTEGER arithmetic (Q20, int64 range) -> bit-identical on any machine
                  (this closes the float64 caveat of the earlier Lorenz engine)
  planet_dem.u16  a 8192 x 8192 uint16 digital elevation model (128 MB) from integer fractal value
                  noise (hash-based, vectorised numpy integer ops) -> bit-identical on any machine
The Seed of each object is its parameter record (~150 bytes); the generator program is the engine.

Budget: Footprint = Seed + generator source; Horizon measured on this machine (one core) and
derived for a RAD750-class flight computer (~400 MIPS) from the counted integer operations.
"""
import hashlib, inspect, json, lzma, gzip, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

# ----------------------------------------------------------------------------- engine 1: fixed-point Lorenz telemetry (exact integers)
Q = 20                                   # fractional bits; |x|,|y|,|z| < 64 -> 2^26; products 2^52 < 2^63
def telemetry_csv(p):
    """Euler integration of the Lorenz system in Q20 fixed point with Python ints (exact, portable)."""
    one = 1 << Q
    s, r, b = int(p["sigma"] * one), int(p["rho"] * one), int(p["beta"] * one)
    x, y, z = int(p["x0"] * one), int(p["y0"] * one), int(p["z0"] * one)
    dt_shift = p["dt_shift"]                                 # dt = 2^-dt_shift
    n = p["steps"]; out = ["t_ms,x,y,z,mode"]; ops = 0
    for i in range(1, n + 1):
        dx = (s * (y - x)) >> Q
        dy = ((x * (r - z)) >> Q) - y
        dz = ((x * y) >> Q) - ((b * z) >> Q)
        x += dx >> dt_shift; y += dy >> dt_shift; z += dz >> dt_shift
        mode = "SAFE" if z > (40 << Q) else "NOMINAL"
        out.append(f"{i},{x},{y},{z},{mode}")           # integers exactly as the state holds them
    return ("\n".join(out) + "\n").encode()
TELEMETRY_OPS_PER_STEP = 16                                # multiplies, shifts, adds, compare (counted from the loop)

# ----------------------------------------------------------------------------- engine 2: integer fractal value noise DEM
def _hash2(ix, iy, seed):
    h = (ix.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + iy.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F) + np.uint64(seed)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    h ^= h >> np.uint64(29); h = (h * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF); h ^= h >> np.uint64(32)
    return (h & np.uint64(0xFFFF)).astype(np.int64)                                   # 16-bit value in [0, 65535]

def planet_dem(p):
    """NxN uint16 heightmap: sum of octaves of integer value noise with bilinear interpolation (all int64).
    Computed in row strips (memory), which does not change a single bit of the result."""
    N, seed, octaves = p["size"], p["seed"], p["octaves"]; out = []; strip = 256
    for r0 in range(0, N, strip):
        ys, xs = np.mgrid[r0:r0 + strip, 0:N].astype(np.int64); acc = np.zeros((strip, N), dtype=np.int64); amp_sum = 0
        for o in range(octaves):
            cell = p["base_cell"] >> o; amp = p["amp"] >> o                              # cell size halves, amplitude halves
            cx, cy = xs // cell, ys // cell; fx, fy = (xs % cell) * 256 // cell, (ys % cell) * 256 // cell   # fractions in Q8
            v00 = _hash2(cx, cy, seed + o); v10 = _hash2(cx + 1, cy, seed + o); v01 = _hash2(cx, cy + 1, seed + o); v11 = _hash2(cx + 1, cy + 1, seed + o)
            top = (v00 * (256 - fx) + v10 * fx) >> 8; bot = (v01 * (256 - fx) + v11 * fx) >> 8
            acc += ((top * (256 - fy) + bot * fy) >> 8) * amp; amp_sum += amp
        out.append((acc // amp_sum).astype(np.uint16).tobytes())
    return b"".join(out)
DEM_OPS_PER_PIXEL_PER_OCTAVE = 40

ENGINES = {"ANET-PROC-telemetry-q20-v1": telemetry_csv, "ANET-PROC-dem-inoise-v1": planet_dem}
def program_bytes(fn): return len(inspect.getsource(fn).encode()) + (len(inspect.getsource(_hash2).encode()) if fn is planet_dem else 0)

def main():
    RAD750_MIPS = 400; here_ops_per_s = None
    objects = [("telemetry.csv", "ANET-PROC-telemetry-q20-v1",
                dict(sigma=10.0, rho=28.0, beta=8 / 3, x0=1.0, y0=1.0, z0=1.0, dt_shift=9, steps=2_500_000)),
               ("planet_dem.u16", "ANET-PROC-dem-inoise-v1", dict(size=8192, seed=20260912, octaves=6, base_cell=1024, amp=1 << 10))]
    disk = {"format": "AlgorithmicDisk/0.4", "generators": {g: {"kind": "procedural", "program_bytes": program_bytes(f)} for g, f in ENGINES.items()}, "manifests": []}
    res = dict(objects=[])
    for name, gid, params in objects:
        fn = ENGINES[gid]; seed = json.dumps(params, sort_keys=True).encode()
        print(f"\n== {name}: generating from a {len(seed)}-byte Seed ...", flush=True)
        t0 = time.time(); data = fn(params); t_gen = time.time() - t0; n = len(data); digest = hashlib.sha256(data).hexdigest()
        t0 = time.time(); again = fn(params); t_gen2 = time.time() - t0; exact = again == data; del again
        open(name, "wb").write(data)
        t0 = time.time(); gz = len(gzip.compress(data, 6)); t_gz = time.time() - t0
        t0 = time.time(); xz = len(lzma.compress(data, preset=6)); t_xz = time.time() - t0
        ops = (params["steps"] * TELEMETRY_OPS_PER_STEP) if "steps" in params else (params["size"] ** 2 * params["octaves"] * DEM_OPS_PER_PIXEL_PER_OCTAVE)
        disk["manifests"].append(dict(id=name, gid=gid, n_bytes=n, sha256=digest, seed=params))
        row = dict(file=name, bytes=n, seed_bytes=len(seed), program_bytes=program_bytes(fn), footprint_bytes=len(seed) + program_bytes(fn),
                   ratio=len(seed) / n, ratio_with_program=(len(seed) + program_bytes(fn)) / n, gzip6=gz, xz6=xz, gzip_ratio=gz / n, xz_ratio=xz / n,
                   gen_s=round(t_gen, 1), regen_s=round(t_gen2, 1), exact=exact, MBps=round(n / t_gen / 1e6, 2), int_ops=ops,
                   ops_per_byte=round(ops / n, 1), horizon_rad750_s=round(ops / (RAD750_MIPS * 1e6), 1), xz_compress_s=round(t_xz, 1), gzip_compress_s=round(t_gz, 1))
        res["objects"].append(row); json.dump(res, open("n2b_results.json", "w"), indent=1); json.dump(disk, open("deepdata.adisk", "w"), indent=1)
        print(f"   {n:,} B  Seed {len(seed)} B (+ program {program_bytes(fn):,} B)  ratio {row['ratio']:.2e} (with program {row['ratio_with_program']:.2e})  "
              f"gzip {gz:,} ({gz/n:.3f}, {t_gz:.0f}s)  xz {xz:,} ({xz/n:.3f}, {t_xz:.0f}s)\n"
              f"   regenerate: {t_gen:.1f}s here ({n/t_gen/1e6:.1f} MB/s), exact {exact}; {ops:,} integer ops = {ops/n:.1f}/byte -> RAD750-class (400 MIPS): {ops/4e8:.0f}s", flush=True)
        del data
    print(f"\ndisk deepdata.adisk: {os.path.getsize('deepdata.adisk'):,} bytes for {sum(o['bytes'] for o in res['objects'])/1e6:.0f} MB of objects; saved n2b_results.json")

if __name__ == "__main__":
    main()
