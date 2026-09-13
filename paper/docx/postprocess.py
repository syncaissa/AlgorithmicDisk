#!/usr/bin/env python3
"""Post-process the pandoc DOCX: size every table's columns by their content (pandoc gives equal widths,
which wraps numbers), use 8.5 pt in tables, centre tables, and make code runs inherit the table size."""
import sys
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
path = sys.argv[1]; doc = Document(path)
TEXT_W = 16.2   # cm between the paper's margins
for t in doc.tables:
    ncol = max(len(r.cells) for r in t.rows)
    pref, need = [0.0] * ncol, [0.0] * ncol      # preferred (no wrap) and minimum (longest word) widths, cm
    cw = 0.185 if ncol <= 8 else 0.165
    pad = 0.45 if ncol <= 8 else 0.32
    for r in t.rows:
        for j, c in enumerate(r.cells):
            for p in c.paragraphs:
                width = 0.0; longest = 0.0; word = 0.0
                for run in p.runs:
                    run.font.size = Pt(8.5 if ncol <= 8 else 7.5)
                    mono = run.style is not None and run.style.name == "Verbatim Char"
                    if mono: run.font.size = Pt(8 if ncol <= 8 else 7)
                    f = cw * (1.25 if mono else 1.0) * (1.08 if run.font.bold else 1.0)
                    for ch in run.text:
                        if ch == " ": longest = max(longest, word); word = 0.0
                        else: word += f
                        width += f
                longest = max(longest, word)
                p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                pref[j] = max(pref[j], width + pad); need[j] = max(need[j], longest + pad)
    if sum(pref) <= TEXT_W: widths = pref
    else:
        # text columns (those that can wrap) give up width first; numeric columns keep their full width
        widths = list(pref); slack = sum(pref) - TEXT_W
        flex = [pref[j] - need[j] for j in range(ncol)]
        if sum(flex) > 0:
            k = min(1.0, slack / sum(flex))
            widths = [pref[j] - k * flex[j] for j in range(ncol)]
        if sum(widths) > TEXT_W: widths = [w * TEXT_W / sum(widths) for w in widths]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = tblPr.find(qn("w:tblLayout"))
    if lay is None: lay = OxmlElement("w:tblLayout"); tblPr.append(lay)
    lay.set(qn("w:type"), "fixed")
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None: tblW = OxmlElement("w:tblW"); tblPr.append(tblW)
    tblW.set(qn("w:type"), "dxa"); tblW.set(qn("w:w"), str(int(sum(widths) / 2.54 * 1440)))
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths): gc.set(qn("w:w"), str(int(w / 2.54 * 1440)))
    # tighter cell margins for wide tables
    if ncol > 8:
        mar = tblPr.find(qn("w:tblCellMar"))
        if mar is None: mar = OxmlElement("w:tblCellMar"); tblPr.append(mar)
        for side in ("left", "right"):
            e = mar.find(qn("w:" + side))
            if e is None: e = OxmlElement("w:" + side); mar.append(e)
            e.set(qn("w:w"), "40"); e.set(qn("w:type"), "dxa")
    for r in t.rows:
        for j, c in enumerate(r.cells): c.width = Cm(widths[j])
# two-column reference list, as in the PDF: a continuous section break before the "References" heading
import copy
body = doc.element.body; main_sect = body.find(qn("w:sectPr"))
paras = body.findall(qn("w:p"))
for i, p in enumerate(paras):
    ps = p.find(qn("w:pPr"))
    if ps is not None and ps.find(qn("w:pStyle")) is not None and ps.find(qn("w:pStyle")).get(qn("w:val")) == "Heading1" and "".join(t.text or "" for t in p.iter(qn("w:t"))).strip().endswith("References"):
        sp = copy.deepcopy(main_sect); ty = OxmlElement("w:type"); ty.set(qn("w:val"), "continuous")
        sp.insert(len(sp.findall(qn("w:headerReference"))) + len(sp.findall(qn("w:footerReference"))), ty)
        ps.append(sp)   # the heading paragraph ends the single-column section
        ty2 = OxmlElement("w:type"); ty2.set(qn("w:val"), "continuous")
        main_sect.insert(len(main_sect.findall(qn("w:headerReference"))) + len(main_sect.findall(qn("w:footerReference"))), ty2)
        cols = OxmlElement("w:cols"); cols.set(qn("w:num"), "2"); cols.set(qn("w:space"), "400"); main_sect.append(cols)
        break
doc.save(path); print("post-processed", len(doc.tables), "tables")
