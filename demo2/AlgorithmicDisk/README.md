# AlgorithmicDisk/ — the whole 104 GB archive: 317 KB

```
engines/ANET-PROC-telemetry-q20-v1.py   the generator (about 1 KB): a Lorenz system integrated in Q20 fixed-point
                                        integer arithmetic, so it is bit-identical on any machine
catalogue.json                          1000 Manifests: {id, gid, seed (7 parameters), n_bytes, sha256}
```
No telemetry file is stored. `python3 ../run_demo2.py read` regenerates all 1000 (~1 h) and checks each sha-256 against
the catalogue; the one-liner in `../README.md` regenerates a single file. The catalogue is also the archive's index:
`python3 ../run_demo2.py summary` answers a parameter query over it in milliseconds without regenerating anything.
The files are outputs of this program, so the ratio is provenance capture by construction (paper §5.6), not
compression of arbitrary data.
