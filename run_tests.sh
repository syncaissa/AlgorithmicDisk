#!/bin/bash
# Runs the quick experiments in paper order using the shipped engines (~45 min on 2 cores). Logs to logs/.
# Long training runs (predictive.py, amortize.py, libwork.py, retraining engine_big.json) are not included;
# see README.md section 2 for those.
set -e
mkdir -p logs
if [ ! -f alice.txt ] || [ ! -f books/pg12.txt ]; then ./get_data.sh; fi
echo "== verify every shipped disk (Tables 2, 5, 8, 9, 10, 13)"; python3 verify.py                                  | tee logs/verify.log
echo "== 5.1 first object + Seed sweep (Tables 2-3)";        python3 algorithmic_storage.py --device cpu demo --sweep | tee logs/demo.log
echo "== 5.3 real files (Table 5)";                          python3 realfile.py --device cpu                   | tee logs/realfile.log
echo "== 5.3 larger engine, two unseen books (Table 6)";     python3 n8_bigengine.py                            | tee logs/n8.log
if command -v 7z >/dev/null && command -v brotli >/dev/null && command -v zstd >/dev/null; then
echo "== 5.3 stronger baselines (Table 7)";                  python3 baselines.py                               | tee logs/baselines.log
else echo "== 5.3 stronger baselines skipped (install p7zip-full brotli zstd to run baselines.py)"; fi
echo "== 5.4 domain engines (Table 8)";                      python3 multiengine.py --device cpu                | tee logs/multiengine.log
echo "== 5.4 revisions cost their edits (Table 9)";          python3 editaware.py                               | tee logs/editaware.log
echo "== 5.4 retrieval engine (Table 10)";                   python3 retrieval.py                               | tee logs/retrieval.log
echo "== 5.5 determinism (Table 11)";                        python3 determinism.py --device cpu                | tee logs/determinism.log
echo "== 5.5 compiled coder (Table 12)";                     python3 fastcoder.py                               | tee logs/fastcoder.log
echo "== 5.6 deep data at scale (Table 13; writes 239 MB, then deletes it)"; python3 n2b_deepdata.py           | tee logs/n2b.log
echo "== compare with the reference results/"
for f in realfile_results.json multiengine_results.json editaware_results.json retrieval_results.json; do python3 compare.py "$f"; done
echo "all done — logs/ holds every run; results/ holds the paper's reference outputs"
