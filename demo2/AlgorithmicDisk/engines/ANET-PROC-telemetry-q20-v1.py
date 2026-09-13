"""ANET-PROC-telemetry-q20-v1: Lorenz telemetry in Q20 fixed-point integers (bit-identical on any machine)."""
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
