#!/bin/bash
# Build the paper: pdflatex -> bibtex -> pdflatex x2. Output: main.pdf
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "built main.pdf ($(pdfinfo main.pdf 2>/dev/null | grep Pages || echo pages unknown))"
cp main.pdf AlgorithmicStorage_paper.pdf && echo "copied to AlgorithmicStorage_paper.pdf"
