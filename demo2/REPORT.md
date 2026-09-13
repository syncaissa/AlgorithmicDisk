# Proof by 1000 files

**Total size of the 1000 files: 104,471,513,533 bytes (104.47 GB)**
**Total size of what is stored (Seeds 123,907 B + generator program 1,060 B + Manifest overhead) = whole disk folder: 317,228 bytes**
**Efficiency ratio: stored / original = 3.04e-06  (the original is 329,326 times larger than what is stored)**

1000 telemetry files, 104.47 GB in total, stored as 123,907 bytes of Seeds (parameter records) plus one 1060-byte generator program;
the catalogue file holding all 1000 Manifests is 315,075 bytes and the whole disk folder 317,228 bytes (3.04e-06 of the data).

Regenerated from the catalogue alone and checked against the recorded sha-256: **1000/1000 bit-exact, 0 failed**.
Write (generate + hash) 3,836 s total; read (regenerate + verify) 3,640 s total, 3.6 s per file on one core.
Catalogue query "rho > 30 and sigma < 10" answered in 0.0002 s with 134 hits, without regenerating anything.
