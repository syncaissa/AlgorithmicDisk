#!/usr/bin/env python3
"""Build reference.docx for pandoc: A4 page and margins of the paper, running header, "n of N" footer,
Cambria body text, heading styles, and the paragraph styles the Lua filter uses (Key Insight box,
Abstract, Keywords, Theorem). Starts from pandoc's own default reference document."""
import os, subprocess, sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
here = os.path.dirname(os.path.abspath(__file__))
pandoc = os.environ.get("PANDOC", "pandoc")
subprocess.run([pandoc, "-o", os.path.join(here, "ref_default.docx"), "--print-default-data-file", "reference.docx"], check=True)
doc = Document(os.path.join(here, "ref_default.docx"))
KEYBLUE, KEYBACK = "1A1ACC", "EDEFFF"
FONT, SIZE = "Cambria", 11

def set_font(style, name=FONT, size=SIZE, bold=None, italic=None, color=None):
    f = style.font; f.name = name; f.size = Pt(size)
    rpr = style.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), name)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
    if bold is not None: f.bold = bold
    if italic is not None: f.italic = italic
    if color: f.color.rgb = RGBColor.from_string(color)

def para_fmt(style, before=None, after=None, line=None, align=None, left=None, right=None, keep=None):
    pf = style.paragraph_format
    if before is not None: pf.space_before = Pt(before)
    if after is not None: pf.space_after = Pt(after)
    if line is not None: pf.line_spacing = line
    if align is not None: pf.alignment = align
    if left is not None: pf.left_indent = Cm(left)
    if right is not None: pf.right_indent = Cm(right)
    if keep is not None: pf.keep_with_next = keep

def shade(style, fill, border=None, border_sz=8, space="4"):
    ppr = style.element.get_or_add_pPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill); ppr.append(shd)
    if border:
        b = OxmlElement("w:pBdr")
        for side in ("top", "left", "bottom", "right"):
            e = OxmlElement("w:" + side); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(border_sz)); e.set(qn("w:space"), space); e.set(qn("w:color"), border); b.append(e)
        ppr.append(b)

def S(name):
    for st in doc.styles:
        if st.name == name: return st
    raise KeyError(name)
def has(name): return any(st.name == name for st in doc.styles)
def style(name, base="Normal"):
    from docx.enum.style import WD_STYLE_TYPE
    st = S(name) if has(name) else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    if name != base and has(base): st.base_style = S(base)
    return st

# --- page geometry (matches geometry: a4paper, top 2.3, bottom 2.4, left/right 2.4 cm)
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin, sec.bottom_margin, sec.left_margin, sec.right_margin = Cm(2.3), Cm(2.4), Cm(2.4), Cm(2.4)
sec.header_distance, sec.footer_distance = Cm(1.2), Cm(1.2)

# --- body text
normal = S("Normal"); set_font(normal); para_fmt(normal, after=4, line=1.10, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
for n in ("Body Text", "First Paragraph", "Compact"):
    if has(n): set_font(S(n)); para_fmt(S(n), after=4, line=1.10, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
for n, sz, before, after in (("Heading 1", 14, 16, 6), ("Heading 2", 12, 12, 4), ("Heading 3", 11, 8, 2)):
    st = S(n); set_font(st, size=sz, bold=True, color="000000"); para_fmt(st, before=before, after=after, align=WD_ALIGN_PARAGRAPH.LEFT, keep=True)
    # remove the theme colour pandoc's default gives headings
    rpr = st.element.get_or_add_rPr()
    for c in rpr.findall(qn("w:color")): rpr.remove(c)
    col = OxmlElement("w:color"); col.set(qn("w:val"), "000000"); rpr.append(col)
st = S("Title"); set_font(st, size=15, bold=True, color="000000"); para_fmt(st, before=0, after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
st = S("Subtitle"); set_font(st, size=11, bold=False, italic=False, color="000000"); para_fmt(st, before=2, after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
for n in ("Author", "Date"):
    st = S(n); set_font(st, size=11, color="000000"); para_fmt(st, before=0, after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
for n in ("Caption", "Table Caption", "Image Caption"):
    if has(n):
        st = S(n); set_font(st, size=9, italic=False, color="000000"); para_fmt(st, before=4, after=4, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
if has("Bibliography"):
    st = S("Bibliography"); set_font(st, size=8.5); para_fmt(st, after=1, line=1.0, align=WD_ALIGN_PARAGRAPH.LEFT)
if has("Block Text"):
    st = S("Block Text"); set_font(st, size=10); para_fmt(st, left=1.0, right=1.0)
# --- custom styles used by filter.lua
st = style("Abstract Title"); set_font(st, size=11, bold=True, color="000000"); para_fmt(st, before=10, after=2, align=WD_ALIGN_PARAGRAPH.CENTER, keep=True)
st = style("Abstract"); set_font(st, size=10); para_fmt(st, after=4, line=1.10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, left=1.0, right=1.0)
st = style("Keywords"); set_font(st, size=10); para_fmt(st, before=4, after=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
st = style("Key Insight Title"); set_font(st, size=10, bold=True, color="FFFFFF"); para_fmt(st, before=10, after=6, line=1.0, align=WD_ALIGN_PARAGRAPH.LEFT, keep=True); shade(st, KEYBLUE, KEYBLUE, space="2")
st = style("Key Insight"); set_font(st, size=10.5); para_fmt(st, before=0, after=0, line=1.10, align=WD_ALIGN_PARAGRAPH.JUSTIFY); shade(st, KEYBACK, KEYBLUE, space="2")
st = style("Theorem"); set_font(st); para_fmt(st, before=4, after=4, line=1.10, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
# --- character styles
for st in doc.styles:
    if st.name == "Verbatim Char":
        set_font(st, name="Consolas", size=9.5)
        rpr = st.element.get_or_add_rPr()
        for c in rpr.findall(qn("w:color")): rpr.remove(c)
# --- header and footer
hdr = sec.header; hp = hdr.paragraphs[0]; hp.text = ""
r = hp.add_run("Algorithmic Storage: The AlgorithmicDisk and the Footprint–Compute–Horizon Triangle"); r.font.size = Pt(9); r.font.name = FONT
hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
ppr = hp._p.get_or_add_pPr(); b = OxmlElement("w:pBdr"); e = OxmlElement("w:bottom"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "1"); e.set(qn("w:color"), "000000"); b.append(e); ppr.append(b)
ftr = sec.footer; fp = ftr.paragraphs[0]; fp.text = ""; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
def field(par, instr):
    r = par.add_run(); r.font.size = Pt(9); r.font.name = FONT
    for tag, txt in (("begin", None), (None, instr), ("separate", None), (None, "1"), ("end", None)):
        if tag: fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), tag); r._r.append(fc)
        elif txt == instr: it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = " " + instr + " "; r._r.append(it)
        else: t = OxmlElement("w:t"); t.text = txt; r._r.append(t)
field(fp, "PAGE"); rr = fp.add_run(" of "); rr.font.size = Pt(9); rr.font.name = FONT; field(fp, "NUMPAGES")
doc.save(os.path.join(here, "reference.docx")); os.remove(os.path.join(here, "ref_default.docx")); print("reference.docx written")
