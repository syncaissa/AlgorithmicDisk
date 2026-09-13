#!/bin/bash
# Build ../AlgorithmicStorage_paper.html: a single self-contained HTML file that reproduces the PDF exactly
# (embedded fonts, vector graphics, selectable text, outline). Uses pdf2htmlEX through Docker.
set -e
cd "$(dirname "$0")"
IMG=pdf2htmlex/pdf2htmlex:0.18.8.rc2-master-20200820-ubuntu-20.04-x86_64
docker run --rm -v "$PWD":/pdf -w /pdf $IMG --zoom 1.5 --embed-css 1 --embed-font 1 --embed-image 1 \
  --embed-javascript 1 --embed-outline 1 --process-outline 1 --dest-dir /pdf \
  AlgorithmicStorage_paper.pdf AlgorithmicStorage_paper.html
sudo chown "$(id -u):$(id -g)" AlgorithmicStorage_paper.html 2>/dev/null || true
echo "built AlgorithmicStorage_paper.html ($(du -h AlgorithmicStorage_paper.html | cut -f1))"
