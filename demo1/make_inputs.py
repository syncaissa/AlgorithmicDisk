#!/usr/bin/env python3
"""
Step 1 of the demo: create ten input files of 1-5 MB each in InputFiles/  (text, PDF, video).

  4 text  : public-domain books never used to train any engine (Project Gutenberg 996, 1184, 28054, 3300)
  3 PDF   : a Cinderella storybook (public-domain text, typeset and rendered to page images); a book laid out as
            text with UNCOMPRESSED streams; the same book PDF with compressed streams (built with PyMuPDF)
  3 video : two procedurally generated animations (integer-only generator, uncompressed AVI) and one real
            H.264 MP4 from NASA (2016 Mercury transit, Goddard Space Flight Center; public domain)

Only this step needs PyMuPDF (to make the PDFs and rasterise pages); the disk itself needs NumPy only.
Provenance of the two procedural videos (generator id + parameters) is written to InputFiles/.provenance.json —
provenance capture requires knowing the generator, and the demo is honest about which files have it.
"""
import json, os, struct, sys, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from anim_generator import render_frames, GEN_ID
IN = os.path.join(HERE, "InputFiles")

BOOKS = {"996": "don_quixote.txt", "1184": "count_of_monte_cristo.txt", "28054": "brothers_karamazov.txt", "3300": "wealth_of_nations.txt"}

def fetch_books():
    for gid, name in BOOKS.items():
        dst = os.path.join(IN, name)
        if os.path.exists(dst): continue
        src = os.path.join(ROOT, "books", f"pg{gid}.txt")
        if not os.path.exists(src):
            src = os.path.join(ROOT, "..", "books", f"pg{gid}.txt")           # development layout
        if not os.path.exists(src):
            os.makedirs(os.path.join(ROOT, "books"), exist_ok=True); src = os.path.join(ROOT, "books", f"pg{gid}.txt")
            urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", src)
        open(dst, "wb").write(open(src, "rb").read())

def write_avi(path, frames, w, h, fps=12):
    """Minimal uncompressed 24-bit AVI (RIFF/AVI, 'DIB ' frames, bottom-up BGR rows padded to 4 bytes)."""
    row = (w * 3 + 3) & ~3; fsize = row * h
    def chunk(tag, data): return tag + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")
    def lst(tag, data): return chunk(b"LIST", tag + data)
    avih = struct.pack("<IIIIIIIIIIIIII", int(1e6 / fps), fsize * fps, 0, 0x10, len(frames), 0, 1, fsize, w, h, 0, 0, 0, 0)
    strh = b"vids" + b"DIB " + struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, 1, fps, 0, len(frames), fsize, 0, 0, 0, 0, w, h)
    strf = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, fsize, 0, 0, 0, 0)
    hdrl = lst(b"hdrl", chunk(b"avih", avih) + lst(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movi_body = b""; idx = b""; off = 4
    for f in frames:
        c = chunk(b"00db", f); idx += b"00db" + struct.pack("<III", 0x10, off, len(f)); off += len(c); movi_body += c
    movi = lst(b"movi", movi_body); idx1 = chunk(b"idx1", idx)
    body = b"AVI " + hdrl + movi + idx1
    open(path, "wb").write(b"RIFF" + struct.pack("<I", len(body)) + body)

def make_videos():
    prov = {}
    specs = [("animation_A.avi", dict(width=160, height=120, frames=60, sigma=10.0, rho=28.0, beta=8 / 3, seed=1)),
             ("animation_B.avi", dict(width=200, height=150, frames=30, sigma=10.0, rho=35.0, beta=8 / 3, seed=7))]
    for name, p in specs:
        dst = os.path.join(IN, name)
        if not os.path.exists(dst):
            fr = render_frames(p); write_avi(dst, fr, p["width"], p["height"])
        prov[name] = {"gid": GEN_ID, "params": p}
    json.dump(prov, open(os.path.join(IN, ".provenance.json"), "w"), indent=1)

def make_pdfs_and_recording():
    import pymupdf
    paper = os.path.join(ROOT, "..", "..", "AlgorithmicStorage_paper.pdf")
    if not os.path.exists(paper): paper = os.path.join(ROOT, "..", "AlgorithmicStorage_paper.pdf")
    # (a) Cinderella storybook PDF: public-domain text (Andrew Lang, The Blue Fairy Book, 1889, Project Gutenberg #503),
    #     typeset in large type and then rendered page-by-page to images (a scanned-storybook style PDF)
    dst = os.path.join(IN, "cinderella_storybook.pdf")
    if not os.path.exists(dst):
        src = os.path.join(ROOT, "books", "pg503.txt")
        if not os.path.exists(src): src = os.path.join(ROOT, "..", "books", "pg503.txt")
        if not os.path.exists(src):
            os.makedirs(os.path.join(ROOT, "books"), exist_ok=True); src = os.path.join(ROOT, "books", "pg503.txt")
            urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/503/pg503.txt", src)
        raw = open(src, encoding="utf-8", errors="ignore").read()
        a = raw.index("CINDERELLA, OR THE LITTLE GLASS SLIPPER"); b = raw.index("ALADDIN AND THE WONDERFUL LAMP", a)
        body = raw[a + len("CINDERELLA, OR THE LITTLE GLASS SLIPPER"):b].strip()
        paras = [" ".join(p.split()) for p in body.split("\n\n")]                 # reflow: join wrapped lines within a paragraph
        story = "CINDERELLA, OR THE LITTLE GLASS SLIPPER\n(Andrew Lang, The Blue Fairy Book, 1889 - public domain)\n\n" + "\n\n".join(paras)
        story = story.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2014", " - ").replace("\u2013", "-")
        title, credit, body_paras = paras_title = "CINDERELLA", "or, The Little Glass Slipper\n\nAndrew Lang, The Blue Fairy Book (1889)\nProject Gutenberg #503 - public domain", story.split("\n\n")[1:]
        typeset = pymupdf.open()
        pg = typeset.new_page(width=595, height=842)                                  # title page
        pg.insert_textbox(pymupdf.Rect(60, 260, 535, 340), title, fontsize=40, fontname="tibo", align=pymupdf.TEXT_ALIGN_CENTER)
        pg.insert_textbox(pymupdf.Rect(60, 350, 535, 480), credit, fontsize=16, fontname="tiit", align=pymupdf.TEXT_ALIGN_CENTER, lineheight=1.4)
        i = 0; pageno = 1
        while i < len(body_paras):
            pg = typeset.new_page(width=595, height=842); pageno += 1; chunk = ""
            while i < len(body_paras) and len(chunk) + len(body_paras[i]) < 2700: chunk += "    " + body_paras[i] + "\n\n"; i += 1
            pg.insert_textbox(pymupdf.Rect(70, 70, 525, 770), chunk, fontsize=14, fontname="tiro", lineheight=1.45, align=pymupdf.TEXT_ALIGN_JUSTIFY)
            pg.insert_textbox(pymupdf.Rect(60, 790, 535, 815), f"- {pageno} -", fontsize=11, fontname="tiro", align=pymupdf.TEXT_ALIGN_CENTER)
        out = pymupdf.open()
        for p in typeset:
            pix = p.get_pixmap(dpi=500); pg = out.new_page(width=p.rect.width, height=p.rect.height); pg.insert_image(pg.rect, pixmap=pix)
        out.save(dst, deflate=True)
    # (b)/(c) text PDFs of Don Quixote, uncompressed and compressed streams
    text = open(os.path.join(IN, BOOKS["996"]), encoding="utf-8", errors="ignore").read()
    for name, deflate in (("quixote_text_uncompressed.pdf", False), ("quixote_text_compressed.pdf", True)):
        dst = os.path.join(IN, name)
        if os.path.exists(dst): continue
        doc = pymupdf.open(); lines = text.splitlines(); i = 0
        while i < len(lines) and doc.page_count < 440:
            pg = doc.new_page(width=595, height=842); chunk = "\n".join(lines[i:i + 60]); i += 60
            pg.insert_textbox(pymupdf.Rect(50, 50, 545, 800), chunk, fontsize=9, fontname="helv")
        if deflate: doc.save(dst, deflate=True, garbage=3)
        else: doc.save(dst, expand=255, deflate=False)              # expand=255 decompresses every stream
    # (d) a real video: NASA, "2016 Mercury Transit" (Goddard Space Flight Center), mobile-resolution MP4, public domain
    dst = os.path.join(IN, "nasa_mercury_transit_2016.mp4")
    if not os.path.exists(dst):
        urllib.request.urlretrieve("https://images-assets.nasa.gov/video/GSFC_20160601_Mercury_m12268_Transit/GSFC_20160601_Mercury_m12268_Transit~mobile.mp4", dst)

if __name__ == "__main__":
    fetch_books(); make_videos(); make_pdfs_and_recording()
    for f in sorted(os.listdir(IN)):
        if not f.startswith("."): print(f"{f:36s} {os.path.getsize(os.path.join(IN, f)):>11,} B")
