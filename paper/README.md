# paper/ — LaTeX source of the paper and the table generator

```bash
python3 make_results.py     # reads ../results/*.json, ../demo/report.json, ../demo2/summary.json -> results.tex
./build.sh                  # pdflatex + bibtex -> main.pdf -> AlgorithmicStorage_paper.pdf  (needs texlive: pdflatex, bibtex, tikz, pgfplots, tcolorbox)
```
`main.tex` contains no result numbers of its own for the measured tables: it `\input`s `results.tex`, whose macros
(`\realfileTable`, `\nEightTable`, `\baselineTable`, ... — 39 in all) are written by `make_results.py` from the JSON
files. Re-run an experiment, copy its `*_results.json` into `../results/`, re-run the two commands, and the paper's
tables update. `AlgorithmicStorage_paper.pdf` is the current build. On Ubuntu:
`sudo apt-get install -y texlive-latex-extra texlive-pictures texlive-science texlive-bibtex-extra`.

## DOCX and HTML versions

```bash
docx/build_docx.sh          # -> AlgorithmicStorage_paper.docx  (needs pandoc >= 3, python3 pymupdf python-docx, pdflatex)
./build_html.sh             # -> AlgorithmicStorage_paper.html  (needs Docker; runs pdf2htmlEX on the PDF)
```
The HTML is a single self-contained file (fonts, figures and outline embedded) that reproduces the PDF exactly, with
selectable text. The DOCX is built from the LaTeX source: real Word headings, tables, equations (OMML) and
citations, with the TikZ figures rendered to 300 dpi images by the paper's own preamble; `docx/prepare_tex.py`
resolves every cross-reference to the number the PDF prints, `docx/make_reference.py` sets the page geometry,
running header, "n of N" footer and the Key Insight box styles, `docx/filter.lua` maps the LaTeX environments
onto those styles, and `docx/postprocess.py` sizes table columns and sets the two-column reference list.
Set `PANDOC=/path/to/pandoc` if pandoc 3 is not on your PATH.
