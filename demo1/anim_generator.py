#!/usr/bin/env python3
"""
ANET-PROC-anim-v1 — a procedural animation engine (integer arithmetic only, bit-identical on any machine).
Each frame: a colour gradient that drifts with time plus a bright dot that follows a Lorenz trajectory
integrated in Q20 fixed point. The Seed of a video made by this engine is its parameter record.
Returns raw bottom-up BGR frames (rows padded to 4 bytes) for an uncompressed AVI.
"""
import numpy as np
GEN_ID = "ANET-PROC-anim-v1"
Q = 20

def render_frames(p):
    w, h, n = p["width"], p["height"], p["frames"]
    one = 1 << Q; s, r, b = int(p["sigma"] * one), int(p["rho"] * one), int(p["beta"] * one)
    x, y, z = int(1.0 * one) + p["seed"] * 1000, int(1.0 * one), int(1.0 * one)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.int64); rowlen = (w * 3 + 3) & ~3; frames = []
    for t in range(n):
        for _ in range(40):                                        # 40 Euler steps of dt = 2^-9 per frame
            dx = (s * (y - x)) >> Q; dy = ((x * (r - z)) >> Q) - y; dz = ((x * y) >> Q) - ((b * z) >> Q)
            x += dx >> 9; y += dy >> 9; z += dz >> 9
        R = ((xs * 255) // max(w - 1, 1) + t * 5) & 255; G = ((ys * 255) // max(h - 1, 1) + t * 3) & 255; B = ((xs ^ ys) + t * 7) & 255
        px = int(((x >> Q) + 25) * (w - 8) // 50); py = int(((z >> Q)) * (h - 8) // 50); px = min(max(px, 0), w - 8); py = min(max(py, 0), h - 8)
        img = np.stack([B, G, R], axis=-1).astype(np.uint8)         # BGR
        img[py:py + 8, px:px + 8, :] = 255
        row = np.zeros((h, rowlen), np.uint8); row[:, :w * 3] = img[::-1].reshape(h, w * 3)   # bottom-up
        frames.append(row.tobytes())
    return frames

def render_bytes(p):
    """Whole video file bytes (the AVI container is part of the generator, so the Seed regenerates the file)."""
    from make_inputs import write_avi
    import tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".avi"); os.close(fd)
    write_avi(tmp, render_frames(p), p["width"], p["height"]); data = open(tmp, "rb").read(); os.remove(tmp); return data
