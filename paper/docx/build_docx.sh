#!/bin/bash
# Build ../../AlgorithmicStorage_paper.docx from main.tex via pandoc (>= 3.1). Steps:
#   render_figures.py  -> figN.png (TikZ/pgfplots rendered with the paper's preamble) + main_docx.tex
#   prepare_tex.py     -> main_pandoc.tex (refs resolved from main.aux, captions numbered, title metadata)
#   make_reference.py  -> reference.docx (page geometry, header/footer, fonts, Key Insight styles)
#   pandoc + filter.lua -> DOCX ; postprocess.py -> table column widths
# Requires: python3 with pymupdf and python-docx; pdflatex; pandoc 3 (set PANDOC=/path/to/pandoc if not on PATH).
set -e
cd "$(dirname "$0")"
PANDOC=${PANDOC:-pandoc}
[ -f ../main.aux ] || (cd .. && ./build.sh >/dev/null)
python3 render_figures.py
python3 prepare_tex.py
PANDOC=$PANDOC python3 make_reference.py
OUT=../AlgorithmicStorage_paper.docx
$PANDOC main_pandoc.tex -f latex -t docx --citeproc --bibliography ../refs.bib --lua-filter filter.lua --csl apa.csl \
        -M reference-section-title=References --reference-doc reference.docx --number-sections --resource-path . -o "$OUT"
python3 postprocess.py "$OUT"
echo "built $OUT"
