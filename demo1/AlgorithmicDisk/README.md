# AlgorithmicDisk/ — everything needed to regenerate the ten files, and nothing else

```
INDEX.json           the catalogue: one entry per object (name, engine id, Seed file, length, sha-256, mode)
engines/
  ANET-EN-256.json   the 148 KB English engine (int8 weights, hidden 256), used through the compiled fixed-point coder
  ANET-PROC-anim-v1.py  the animation generator; its Seed is a 110-byte parameter record
manifests/*.json     one Regeneration Manifest per object: engine id (GID), Seed reference, length, sha-256
seeds/*.seed         the Seed bytes: the arithmetic code of a book, 110 bytes for an animation, or the raw bytes for
                     an object no engine could shorten (the verbatim escape: the three PDFs and the NASA video)
```
`python3 ../run_demo.py read` rebuilds `../OutputFiles/` from this folder alone and checks every sha-256; `../REPORT.md`
is the byte-for-byte comparison. The folder is 12.6 MB for 25.5 MB of inputs (Table 14): 0.49x, against 0.42x for
xz -6 — the disk loses on this deliberately mixed set because four files have no engine and are stored raw.
