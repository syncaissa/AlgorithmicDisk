#!/usr/bin/env python3
"""Render every figure of main.tex (TikZ / pgfplots) to PNG at 300 dpi, using the paper's own preamble
and results.tex, so the images in the DOCX are the ones in the PDF.  Writes docx/figN.png and
docx/main_docx.tex, a copy of main.tex in which each figure body is replaced by \\includegraphics."""
import os, re, subprocess, sys, fitz
here = os.path.dirname(os.path.abspath(__file__)); paper = os.path.dirname(here)
src = open(os.path.join(paper, "main.tex")).read()
pre, body = src.split("\\begin{document}", 1)
pre = pre.replace("\\input{results.tex}", "\\input{../results.tex}")
figs = list(re.finditer(r"\\begin\{figure\}(\[[^\]]*\])?\s*\\centering\s*(.*?)(\\caption\{)", body, re.S))
out = body
for i, m in enumerate(figs, 1):
    inner = m.group(2)
    tex = (pre + "\\usepackage[active,tightpage]{preview}\\PreviewEnvironment{tikzpicture}\n\\begin{document}\n"
           + inner + "\n\\end{document}\n")
    fn = os.path.join(here, f"fig{i}.tex"); open(fn, "w").write(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"fig{i}.tex"], cwd=here,
                   stdout=subprocess.DEVNULL, check=True)
    d = fitz.open(os.path.join(here, f"fig{i}.pdf")); p = d[0]
    p.get_pixmap(dpi=300).save(os.path.join(here, f"fig{i}.png"))
    w_in = p.rect.width / 72
    print(f"fig{i}.png  {w_in:.2f} in wide")
    out = out.replace(inner, "\\includegraphics[width=%.2fin]{fig%d.png}\n" % (min(w_in, 6.3), i), 1)
    for ext in ("aux", "log", "tex", "pdf", "out"):
        if os.path.exists(os.path.join(here, f"fig{i}.{ext}")): os.remove(os.path.join(here, f"fig{i}.{ext}"))
open(os.path.join(here, "main_docx.tex"), "w").write(pre + "\\begin{document}" + out)
print(len(figs), "figures rendered; main_docx.tex written")
