# native/ — the fixed-point engine and range coder in C

`anet_fixed.c` is the integer-only AlgorithmicNET step (Q12 weights, Q15 state, table-driven tanh and 2^x, int64
accumulators, integer soft-max) and the carry-less 32-bit range coder, mirroring `fixedpoint.py` and
`predictive.py` exactly: Seeds produced by the C code and by the Python code are byte-identical, and each decodes the
other's (paper §5.5, Table 12). Blocks are independent, so OpenMP parallelises over blocks (`--threads`).

Build and test — `fastcoder.py` does both:
```bash
sudo apt-get install -y build-essential        # gcc with OpenMP
python3 ../fastcoder.py                        # compiles libanet_fixed.so here, cross-checks against Python, times threads 1/2/4
```
Manual build: `gcc -O3 -march=native -fopenmp -shared -fPIC anet_fixed.c -o libanet_fixed.so`. The library is loaded
through `ctypes` by `fastcoder.py` (class `FastFixedEngine`), which `n8_bigengine.py` and `demo/run_demo.py` also use.

Why it exists: the float engine's probability tables depend on summation order, so its Seeds are not portable across
BLAS libraries, core counts or GPUs (Table 11). The integer engine has no floating point between Seed and object, so any
machine with 64-bit two's-complement integers should produce the same bytes; two independent implementations (this
file and `fixedpoint.py`) agreeing is the evidence the paper offers. A run on a GPU or a non-x86 CPU is the most useful
test a reader can contribute — `python3 determinism.py` reports it.
