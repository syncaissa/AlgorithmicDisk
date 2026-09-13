#!/usr/bin/env python3
"""Turn docx/main_docx.tex (figures already replaced by PNGs) into docx/main_pandoc.tex:
   * every \\ref / \\eqref / \\pageref is replaced by the number the PDF shows (read from main.aux),
     so cross-references in the DOCX are identical to the PDF's;
   * numbered equations get their "(n)" tag;
   * the hand-built title block becomes pandoc metadata (Title / Subtitle / Author / Date styles);
   * the "Abstract" and "Keywords" blocks are marked so the filter can style them."""
import os, re
here = os.path.dirname(os.path.abspath(__file__)); paper = os.path.dirname(here)
aux = open(os.path.join(paper, "main.aux")).read()
labels = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))
s = open(os.path.join(here, "main_docx.tex")).read()
# inline the generated results.tex so every substitution below also applies to the generated tables
s = s.replace("\\input{../results.tex}", open(os.path.join(paper, "results.tex")).read())
missing = set()
def ref(m):
    k = m.group(1)
    if k not in labels: missing.add(k); return "??"
    return labels[k]
s = re.sub(r"\\ref\{([^}]*)\}", ref, s)
s = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + ref(m) + ")", s)
s = s.replace("\\pageref{LastPage}", "")
# equation tags
def eqtag(m):
    body = m.group(1); lab = re.search(r"\\label\{([^}]*)\}", body)
    n = labels.get(lab.group(1), "") if lab else ""
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    return "\\begin{equation}" + body.rstrip() + (" \\qquad (%s)" % n if n else "") + "\\end{equation}"
s = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eqtag, s, flags=re.S)
s = re.sub(r"\\label\{[^}]*\}", "", s)
# \bm -> \boldsymbol (texmath), and the two-line aligned manifestation equation -> two display lines
s = s.replace("\\bm{", "\\boldsymbol{")
def split_aligned(m):
    body = m.group(1); tag = m.group(2) or ""
    lines = [re.sub(r"&", "", l).strip() for l in re.split(r"\\\\", body) if l.strip()]
    return "".join("\\[" + l + ("\\qquad " + tag if i == len(lines) - 1 else "") + "\\]\n" for i, l in enumerate(lines))
s = re.sub(r"\\begin\{equation\}\s*\\begin\{aligned\}(.*?)\\end\{aligned\}\s*(?:\\qquad (\(\d+\)))?\\end\{equation\}", split_aligned, s, flags=re.S)
# caption prefixes in source order = LaTeX numbering order
counters = {"table": 0, "figure": 0}
def number_captions(kind):
    global s
    pre, body = s.split("\\begin{document}", 1); s = body
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{%s\}.*?\\end\{%s\}" % (kind, kind), s, re.S):
        block = m.group(0); counters[kind] += 1
        block = block.replace("\\caption{", "\\caption{%s %d: " % (kind.capitalize(), counters[kind]), 1)
        out.append(s[pos:m.start()]); out.append(block); pos = m.end()
    out.append(s[pos:]); s = pre + "\\begin{document}" + "".join(out)
number_captions("table"); number_captions("figure")
# title block -> metadata
title_block = re.search(r"\\begin\{center\}\s*\{\\Large\\bfseries(.*?)\\end\{center\}", s, re.S)
tb = title_block.group(0)
title = re.search(r"\{\\Large\\bfseries (.*?)\\par\}", tb, re.S).group(1)
subtitle = re.search(r"\{\\normalsize A Seed-Driven(.*?)\\par\}", tb, re.S).group(0)[len("{\\normalsize "):-len("\\par}")]
clean = lambda t: re.sub(r"\\\\\[[^\]]*\]|\\\\", " ", t).replace("\n", " ").strip()
clean = lambda t: re.sub(r"\s+", " ", re.sub(r"\\\\\[[^\]]*\]|\\\\", " ", t)).strip()
meta = ("\\title{%s}\n\\subtitle{%s}\n\\author{Milind K. Patil \\\\ Syncaissa Systems Inc. \\\\ \\texttt{syncaissa@outlook.com}}\n\\date{September 2026}\n"
        % (clean(title), clean(subtitle)))
s = s.replace(tb, "")
s = s.replace("\\begin{document}", meta + "\\begin{document}\n\\maketitle\n", 1)
# abstract + keywords markers
s = s.replace("\\begin{center}\\textbf{Abstract}\\end{center}\n\\begin{quote}\\small", "\\begin{abstractbox}")
s = s.replace("\\end{quote}\n\\noindent\\small\\textbf{Keywords:}", "\\end{abstractbox}\n\\begin{keywordsbox}\\textbf{Keywords:}")
s = s.replace("millennial archive\n\\normalsize", "millennial archive\n\\end{keywordsbox}")
# bibliography: pandoc --citeproc places it at the end; drop the manual multicols block
s = re.sub(r"\{\\scriptsize\\setlength\{\\bibsep\}.*?\\end\{multicols\}\}", "", s, flags=re.S)
s = s.replace("\\begin{center}\\small Syncaissa Systems Inc. \\quad$\\bullet$\\quad \\texttt{syncaissa@outlook.com}\\end{center}", "")
# two caption formulas that some Word renderers set badly: write them as text
s = s.replace("``$\\mid$ prior''", "``| prior''").replace("$\\mid$ prior", "| prior")
s = s.replace("$|\\mathrm{compress}(\\mathrm{corpus}\\,\\|\\,\\mathrm{book})|-|\\mathrm{compress}(\\mathrm{corpus})|$", "|compress(corpus\\,$\\|$\\,book)| $-$ |compress(corpus)|")
s = s.replace("$+$", "+").replace("$-$", "\u2212").replace("$\\times$", "\u00d7").replace("$/$", "/")
open(os.path.join(here, "main_pandoc.tex"), "w").write(s)
print("unresolved labels:", sorted(missing) or "none", "| tables", counters["table"], "figures", counters["figure"])
