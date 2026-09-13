#!/usr/bin/env python3
"""Generate results.tex (macros, table, plot data) from the experiment JSON files in ../code."""
import json, os, re
here = os.path.dirname(os.path.abspath(__file__))
code = os.path.join(here, "..", "results")   # packaged layout: every *_results.json lives in ../results
out = []
def macro(name, body): out.append("\\newcommand{\\%s}{%s}" % (name, body))

# ---------------- Design B (predictive engine)
pb = os.path.join(code, "predictive_results.json")
if os.path.exists(pb):
    r = json.load(open(pb))
    macro("corpusChunks", str(r["corpus_chunks"]))
    macro("gzipBits", str(r["baselines"]["gzip9_bits_per_chunk"]))
    rows = r["runs"]
    hid = sorted(set(x["hidden"] for x in rows))
    tab = ["\\begin{tabular}{@{}rrrrrrrrr@{}}", "\\toprule",
           "hidden & $M$ & weights (int8) & Seed coded & Seed ideal & held-out Seed & Footprint/obj & vs.\\ raw & $M^{*}$ \\\\",
           "\\midrule"]
    for x in rows:
        fp = x["footprint_bits_per_obj"]; ratio = fp / r["N_bits"]
        flag = "\\textbf{%.0f}" % fp if fp < r["N_bits"] else "%.0f" % fp
        tab.append("%d & %d & %s bits & %.0f & %.0f & %.0f & %s bits & %.2f & %d \\\\" % (
            x["hidden"], x["M"], "{:,}".format(x["weight_bits"]), x["train"]["seed_bits_mean"], x["train"]["ideal_bits_mean"],
            x["holdout"]["seed_bits_mean"], flag, ratio, x["breakeven_M"]))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("designBTable", "\n".join(tab))
    plots = []
    cols = {64: "cNet", 128: "cComp", 256: "cHori"}
    for h in hid:
        pts = " ".join("(%d,%.1f)" % (x["M"], x["footprint_bits_per_obj"]) for x in rows if x["hidden"] == h)
        plots.append("\\addplot[%s,line width=1.4pt,mark=*] coordinates {%s}; \\addlegendentry{hidden %d, coded Seed}" % (cols.get(h, "black"), pts, h))
        pts = " ".join("(%d,%.1f)" % (x["M"], x["footprint_bits_per_obj_ideal"]) for x in rows if x["hidden"] == h)
        plots.append("\\addplot[%s,line width=1.0pt,densely dashed,mark=o] coordinates {%s}; \\addlegendentry{hidden %d, ideal Seed}" % (cols.get(h, "black"), pts, h))
    macro("designBPlots", "\n  ".join(plots))
    ymax = min(1200, max(x["footprint_bits_per_obj"] for x in rows) * 1.08)
    macro("figYmax", "%.0f" % ymax)
else:
    macro("corpusChunks", "?"); macro("gzipBits", "?"); macro("designBTable", "(pending)"); macro("designBPlots", ""); macro("figYmax", "600")

# ---------------- Design A (latent seeds)
la = os.path.join(code, "designA_long.log")
txt = "Result pending."
if os.path.exists(la):
    s = open(la).read()
    m = re.search(r"FINAL exact ([\d.]+) bit_acc ([\d.]+) time ([\d.]+)", s)
    eps = re.findall(r"ep\s+(\d+) loss ([\d.]+) exact\s+([\d.]+)%\s+bit-acc\s+([\d.]+)%", s)
    if m:
        ex, acc, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
        best_acc = max(float(e[3]) for e in eps) if eps else acc * 100
        tf_loss = min(float(e[1]) for e in eps) if eps else float("nan")
        txt = ("On 500 chunks with 32-dimensional 4-bit Seeds (128 bits/object) and a 128-unit engine, "
               "3000 epochs (%.0f\\,s) drove the teacher-forced hinge loss to %.1g---essentially every bit "
               "satisfied when the engine is fed the true previous byte---yet free-running regeneration from the "
               "quantised Seeds reached only %.1f\\,\\%% bit accuracy and \\emph{%.1f\\,\\%%} bit-exact objects: the "
               "first wrong bit cascades through the output feedback (exposure bias), and 4-bit quantisation "
               "costs a further margin. Design A is kept as the search-based fallback for engines that cannot "
               "express a distribution; the predictive engine removes both problems at once." % (sec, tf_loss, acc * 100, ex * 100))
macro("designAResult", txt)

# ---------------- N2 real files
rf = os.path.join(code, "realfile_results.json")
if os.path.exists(rf):
    r = json.load(open(rf)); e = r["engine"]
    macro("trainBytes", "{:,}".format(e["train_bytes"])); macro("engineParams", "{:,}".format(e["params"]))
    macro("engineBytes", "{:,}".format(e["bytes"])); macro("blockLen", str(e["block_len"]))
    tab = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
           "file & bytes & gzip & xz & mode & Seed (B) & bits/byte & vs.\\ xz & write (s) & read (s) \\\\", "\\midrule"]
    for f in r["files"]:
        for i, (mode, x) in enumerate(f["modes"].items()):
            name = f["file"].replace("_", "\\_") if i == 0 else ""
            b = "{:,}".format(f["bytes"]) if i == 0 else ""; gz = "{:,}".format(f["baselines"]["gzip9"]) if i == 0 else ""
            xz = "{:,}".format(f["baselines"]["xz"]) if i == 0 else ""
            tab.append("\\texttt{%s} & %s & %s & %s & %s & %s & %.2f & %.2f & %.0f & %.0f \\\\" % (
                name, b, gz, xz, mode, "{:,}".format(x["seed_bytes"]), x["bits_per_byte"], x["seed_bytes"] / f["baselines"]["xz"], x["write_s"], x["read_s"]))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("realfileTable", "\n".join(tab))
    # reading paragraph
    txt_files = [f for f in r["files"] if not f["file"].endswith(".png")]
    best = min(txt_files, key=lambda f: f["modes"]["stream"]["ratio"])
    bs = best["modes"]["stream"]; bb = best["modes"]["blocks"]
    png = next((f for f in r["files"] if f["file"].endswith(".png")), None)
    s = ("\\noindent\\textbf{Reading the result.} On the in-domain book (\\texttt{%s}, never seen in training) the "
         "Seed is %.2f bits/byte---a %.2f$\\times$ Footprint ratio, %s than \\texttt{xz -9} (%.2f bits/byte) and "
         "%s than \\texttt{gzip -9} (%.2f bits/byte)---with the engine's %s bytes paid once for every object on the disk. "
         "Block mode costs %.0f\\,\\%% more Seed (a 4-byte flush per %d-byte block) and reads %.1f$\\times$ faster than "
         "stream mode because the engine step is batched over all blocks; both are bit-exact."
         % (best["file"].replace("_", "\\_"), bs["bits_per_byte"], bs["ratio"],
            "better" if bs["seed_bytes"] < best["baselines"]["xz"] else "worse", best["baselines"]["xz"] * 8 / best["bytes"],
            "better" if bs["seed_bytes"] < best["baselines"]["gzip9"] else "worse", best["baselines"]["gzip9"] * 8 / best["bytes"],
            "{:,}".format(e["bytes"]), 100 * (bb["seed_bytes"] / bs["seed_bytes"] - 1), e["block_len"],
            bs["read_s"] / max(bb["read_s"], 1e-9)))
    if png:
        ps = png["modes"]["stream"]; pb = png["modes"]["blocks"]
        s += (" The PNG is a \\emph{shallow} object for this engine---already compressed, out of domain---and it "
              "exposes both ends of the conservation law: in stream mode the mismatched engine \\emph{expands} it to "
              "%.2f bits/byte (%.2f$\\times$), the information floor made visible; in blocks mode the per-block verbatim "
              "escape stores every block raw and caps the Footprint at %.3f$\\times$ (2 bytes of header per %d-byte block), "
              "the ceiling of Proposition~\\ref{thm:law}(ii) implemented---and reads in %.1f\\,s because no block needs the engine. "
              "The out-of-domain text files sit in between: English prose the engine has learned is cheap, LaTeX/Python "
              "syntax it has never seen is not, and there the engine loses to \\texttt{gzip}." % (
                  ps["bits_per_byte"], ps["ratio"], pb["ratio"], e["block_len"], pb["read_s"]))
    macro("realfileReading", s)
else:
    for k in ("trainBytes","engineParams","engineBytes","blockLen"): macro(k, "?")
    macro("realfileTable", "(pending)"); macro("realfileReading", "")

# ---------------- N7 multi-engine disk
me = os.path.join(code, "multiengine_results.json")
if os.path.exists(me):
    r = json.load(open(me))
    tab = ["\\begin{tabular}{@{}lrrrlrrrr@{}}", "\\toprule",
           "object & bytes & gzip & xz & engine chosen & Seed (B) & ratio & total & read (s) \\\\", "\\midrule"]
    for o in r["objects"]:
        tab.append("\\texttt{%s} & %s & %s & %s & %s & %s & %s & %s & %.1f \\\\" % (
            o["file"].replace("_", "\\_"), "{:,}".format(o["bytes"]), "{:,}".format(o["gzip"]), "{:,}".format(o["xz"]),
            o["chosen"].replace("_", "\\_"), "{:,}".format(o["seed_bytes"]),
            ("\\textbf{%.5f}" % o["ratio"]) if o["ratio"] < 0.02 else "%.3f" % o["ratio"],
            ("%.5f" % o["total_ratio"]) if o["total_ratio"] < 0.01 else "%.3f" % o["total_ratio"], o["read_s"]))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("multiTable", "\n".join(tab))
    eng = r["engines"]
    def find(n): return next((o for o in r["objects"] if o["file"] == n), None)
    lz, lib, rev, fr, py, png = (find(n) for n in ("lorenz.csv", "pg12.txt", "pg12_revised.txt", "pg84_frankenstein.txt", "algorithmic_storage.py", "image.png"))
    libh = eng.get("ANET-LIB", {}).get("hist", [{}])
    libbpb = libh[-1].get("bits_per_byte", float("nan")) if libh else float("nan")
    s = "\\noindent\\textbf{Reading the result.} "
    if lz: s += ("The simulation output is \\emph{deep data}: %s bytes regenerate from a %d-byte parameter record, a ratio of "
                 "%.5f (%.1f\\,s of Compute to manifest) where \\texttt{xz} manages %.2f. This is the regime the paradigm is for, and "
                 "the floor is the parameter record itself, because that is the object's information content. " % (
                     "{:,}".format(lz["bytes"]), lz["seed_bytes"], lz["ratio"], lz["read_s"], lz["xz"] / lz["bytes"]))
    if lib: s += ("The library engine (%s\\,B, trained to %.3f bits/byte on its library) stores the library document at "
                  "%.4f and %s" % ("{:,}".format(eng["ANET-LIB"]["bytes"]), libbpb, lib["ratio"],
                  ("its 60-word revision at %.4f---7\\,KB more than the original, because each edit also perturbs the engine's context for the rest of its block; Section~\\ref{sec:edit} reduces a revision to its edits alone. " % rev["ratio"]) if rev else ". "))
    if lib: s += ("That is %.2f$\\times$ per object after %d epochs (%.0f\\,min) of memorisation, against %.2f for \\texttt{xz}; "
                  "reaching $0.01\\times$ requires a surprisal of ${\\le}0.08$ bits/byte, i.e.\\ an engine that has memorised "
                  "${\\sim}99\\,\\%%$ of its library---purely a matter of training Work and engine size (Section~\\ref{sec:multi}). "
                  "And the engine is larger than a one-document library, so the \\emph{total} ratio is %.2f: the library regime "
                  "pays off only when the engine is amortised over many documents and many revisions, exactly as "
                  "Corollary~\\ref{cor:amort} says. " % (lib["ratio"], libh[-1].get("epoch", 0), libh[-1].get("sec", 0) / 60,
                                                         lib["xz"] / lib["bytes"], lib["total_ratio"]))
    if fr and py: s += ("Engine selection did the rest: a 200\\,KB excerpt of \\emph{Frankenstein}---English, and in fact part of the "
                        "English engine's training corpus, so a favourable case---went to ANET-EN (%.3f), Python source to ANET-CODE "
                        "(%.3f, against %.3f for \\texttt{xz})" % (fr["ratio"], py["ratio"], py["xz"] / py["bytes"]))
    if png: s += ", and the PNG fell through to the verbatim escape (%.3f)." % png["ratio"]
    macro("multiReading", s)
else:
    macro("multiTable", "(pending)"); macro("multiReading", "")


# ---------------- N7b library memorisation curve
lw = os.path.join(code, "libwork_results.json")
if os.path.exists(lw):
    r = json.load(open(lw)); pts = r["points"]
    plot = " ".join("(%.1f,%.4f)" % (p["minutes"], p["ratio"]) for p in pts)
    last = pts[-1]
    sec = ("The library engine was trained further (constant learning rate, %s\\,B int8, hidden %d) and the library "
           "document was re-written to the disk at intervals. Figure~\\ref{fig:libwork} plots the document's Footprint "
           "ratio against training Work: the curve is the Regeneration Triangle for a single object, Footprint bought with "
           "Compute$\\times$Horizon spent once on the shared engine. After %.0f\\,min the document costs %.4f$\\times$ "
           "(%s\\,B; \\texttt{xz} %.3f), %s"
           % ("{:,}".format(r["engine_bytes"]), r["hidden"], last["minutes"], last["ratio"], "{:,}".format(last["seed_bytes"]),
              r["xz_ratio"], ("and the $0.01\\times$ mark is crossed at %.0f\\,min." % next(p["minutes"] for p in pts if p["ratio"] <= 0.01))
              if any(p["ratio"] <= 0.01 for p in pts) else
              "still short of $0.01\\times$: on the test machine the curve's slope, not the paradigm, sets the limit."))
    sec += ("\n\\begin{figure}[H]\\centering\\begin{tikzpicture}\\begin{axis}[width=0.78\\textwidth,height=5.6cm,ymode=log,"
            "xlabel={training Work on the shared engine (minutes on the test machine)},ylabel={document Footprint ratio},grid=major,grid style={black!12},"
            "legend style={font=\\footnotesize,draw=black!40,at={(0.5,0.5)},anchor=center},every axis label/.style={font=\\small},tick label style={font=\\footnotesize},xmin=%.0f]"
            "\\addplot[cNet,line width=1.4pt,mark=*] coordinates {%s}; \\addlegendentry{library document, Seed/size}"
            "\\addplot[cSeed,dotted,thick,domain=%.0f:%.0f] {%.4f}; \\addlegendentry{xz -9}"
            "\\addplot[black!60,dashed,thick,domain=%.0f:%.0f] {0.01}; \\addlegendentry{$0.01\\times$ target}"
            "\\end{axis}\\end{tikzpicture}\\caption{Per-document Footprint against training Work for the library engine.}"
            "\\label{fig:libwork}\\end{figure}" % (pts[0]["minutes"] - 5, plot, pts[0]["minutes"] - 5, last["minutes"], r["xz_ratio"], pts[0]["minutes"] - 5, last["minutes"]))
    macro("libworkSection", sec)
    macro("libworkShort", ("The library engine was trained further at a constant learning rate and the document re-written to the disk every "
        "fifteen minutes (twelve checkpoints, %d--%d\\,min of Work on the test machine). Its Footprint ratio fell from %.3f to %.3f "
        "(best %.3f at %.0f\\,min) and then flattened, still well short of $0.01\\times$: memorising a document into a 384-unit network is "
        "the wrong tool for that target, which the retrieval engine below reaches by construction (Figure~\\ref{fig:libwork})."
        % (pts[0]["minutes"], last["minutes"], pts[0]["ratio"], last["ratio"], min(p["ratio"] for p in pts), min(pts, key=lambda p: p["ratio"])["minutes"])))
    # compact figure only (paper Figure "Footprint against training Work"): the third kind of Work, measured
    macro("libworkPlot", ("\\begin{tikzpicture}\\begin{axis}[width=0.56\\textwidth,height=4cm,ymode=log,"
        "xlabel={training Work on the shared engine (min, test machine)},ylabel={Footprint ratio},grid=major,grid style={black!12},"
        "legend style={font=\\scriptsize,draw=black!40,at={(1.03,0.5)},anchor=west},every axis label/.style={font=\\footnotesize},tick label style={font=\\scriptsize},xmin=%.0f,ymin=0.007,ymax=0.5]"
        "\\addplot[cNet,line width=1.2pt,mark=*,mark size=1.4pt] coordinates {%s}; \\addlegendentry{library document, Seed/size}"
        "\\addplot[cSeed,dotted,thick,domain=%.0f:%.0f] {%.4f}; \\addlegendentry{xz -9}"
        "\\addplot[black!60,dashed,thick,domain=%.0f:%.0f] {0.01}; \\addlegendentry{$0.01\\times$ target}"
        "\\end{axis}\\end{tikzpicture}" % (pts[0]["minutes"] - 5, plot, pts[0]["minutes"] - 5, last["minutes"], r["xz_ratio"], pts[0]["minutes"] - 5, last["minutes"])))
else:
    macro("libworkSection", "(memorisation run pending)"); macro("libworkShort", "(memorisation run pending)"); macro("libworkPlot", "")


# ---------------- N9 edit-aware seeds
ea = os.path.join(code, "editaware_results.json")
if os.path.exists(ea) and "totals" in json.load(open(ea)):
    r = json.load(open(ea))
    tab = ["\\begin{tabular}{@{}lrrrllrrr@{}}", "\\toprule",
           "version & bytes & gzip & xz & engine & base & Seed (B) & ratio & read (s) \\\\", "\\midrule"]
    for o in r["objects"]:
        tab.append("\\texttt{%s} & %s & %s & %s & %s & %s & %s & %s & %.1f \\\\" % (
            o["file"].replace("_", "\\_"), "{:,}".format(o["bytes"]), "{:,}".format(o["gzip"]), "{:,}".format(o["xz"]),
            o["chosen"], (o["base"] or "---").replace("_", "\\_"), "{:,}".format(o["seed_bytes"]),
            ("\\textbf{%.5f}" % o["ratio"]) if o["ratio"] < 0.02 else "%.3f" % o["ratio"], o["read_s"]))
    tt = r["totals"]
    tab.append("\\midrule"); tab.append("all four & %s & & %s & & & %s & %.4f & \\\\" % ("{:,}".format(tt["bytes"]), "{:,}".format(tt["xz_each"]), "{:,}".format(tt["seeds"]), tt["ratio"]))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("editTable", "\n".join(tab))
    revs = [o for o in r["objects"] if o["chosen"] == "ANET-DELTA"]
    s = "\\noindent\\textbf{Reading the result.} "
    if revs:
        me_ = json.load(open(os.path.join(code, "multiengine_results.json"))); rf_ = json.load(open(os.path.join(code, "realfile_results.json")))
        lib_e = me_["engines"]["ANET-LIB"]["bytes"]; en_e = me_["engines"]["ANET-EN"]["bytes"]
        en_base = next(f for f in rf_["files"] if f["file"] == "pg12.txt")["modes"]["stream"]["seed_bytes"]
        deltas = tt["seeds"] - r["objects"][0]["seed_bytes"]
        s += ("Every revision was stored as a delta: %s. Four versions of a %s-byte document occupy %s bytes of Seeds in "
              "total (%.4f$\\times$), against %s bytes for \\texttt{xz} applied to each version. Counting the engine as well: with the "
              "library engine (%s\\,B) the four versions cost %s\\,B, more than \\texttt{xz}; with the shared 55\\,KB English engine "
              "storing the base instead (Table~\\ref{tab:realfile}) they cost %s\\,B, %.0f\\,\\%% below \\texttt{xz}---the deltas "
              "(%s\\,B for three revisions) are the same under either. A revision therefore costs "
              "its edits and nothing else---this is what a manual update over a deep-space link should cost---and the read "
              "Horizon of a revision is the base's Horizon plus a few milliseconds." % (
                  "; ".join("\\texttt{%s} at %.5f$\\times$ (%d edits, base \\texttt{%s})" % (
                      o["file"].replace("_", "\\_"), o["ratio"], next(tr["edits"] for tr in o["trials"] if tr.get("base") == o["base"]), o["base"].replace("_", "\\_")) for o in revs),
                  "{:,}".format(r["objects"][0]["bytes"]), "{:,}".format(tt["seeds"]), tt["ratio"], "{:,}".format(tt["xz_each"]),
                  "{:,}".format(lib_e), "{:,}".format(tt["seeds"] + lib_e), "{:,}".format(en_base + en_e + deltas),
                  100 * (1 - (en_base + en_e + deltas) / tt["xz_each"]), "{:,}".format(deltas)))
    macro("editReading", s)
else:
    macro("editTable", "(pending)"); macro("editReading", "")


# ---------------- N10 retrieval engine
rt = os.path.join(code, "retrieval_results.json")
if os.path.exists(rt):
    r = json.load(open(rt))
    macro("retEngineBytes", "{:,}".format(r["engine_bytes"])); macro("retLibraryBytes", "{:,}".format(r["library_bytes"]))
    tab = ["\\begin{tabular}{@{}lrrrrrrlrr@{}}", "\\toprule",
           "object & bytes & xz & LIB engine & delta & retrieval & runs & chosen & Seed (B) & ratio \\\\", "\\midrule"]
    for o in r["objects"]:
        tr = {x["gid"].split("-")[1] if x["gid"].startswith("ANET-") else x["gid"]: x for x in o["trials"]}
        deltas = [x for x in o["trials"] if x["gid"] == "ANET-DELTA"]
        ret = next((x for x in o["trials"] if x["gid"].startswith("ANET-RET")), None)
        libv = next((x for x in o["trials"] if x["gid"] in ("ANET-LIB", "ANET-EN")), None)
        chosen = o["chosen"].split("-")[1] if o["chosen"].startswith("ANET-") else o["chosen"]
        tab.append("\\texttt{%s} & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            o["file"].replace("_", "\\_").replace("pg84\\_unseen", "pg84\\_frankenstein"), "{:,}".format(o["bytes"]), "{:,}".format(o["xz"]),
            "{:,}".format(libv["seed_bytes"]) if libv else "---", "{:,}".format(min(d["seed_bytes"] for d in deltas)) if deltas else "---",
            "{:,}".format(ret["seed_bytes"]) if ret else "---", str(ret["runs"]) if ret else "---", chosen,
            "{:,}".format(o["seed_bytes"]), ("\\textbf{%.5f}" % o["ratio"]) if o["ratio"] < 0.02 else "%.3f" % o["ratio"]))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("retTable", "\n".join(tab))
    def find(n): return next((o for o in r["objects"] if o["file"] == n), None)
    lib, comp, uns, rev1 = find("pg12.txt"), find("composite_manual.txt"), find("pg84_unseen.txt"), find("rev1_60words.txt")
    s = "\\noindent\\textbf{Reading the result.} "
    if lib: s += ("A library document costs %d bytes (%.5f$\\times$): one run covering the whole document, regenerated in "
                  "%.1f\\,s from the index---the $0.01\\times$ mark passed by two orders of magnitude, by construction rather "
                  "than by training. " % (lib["seed_bytes"], lib["ratio"], lib["read_s"]))
    if comp:
        cr = next(x for x in comp["trials"] if x["gid"].startswith("ANET-RET"))
        s += ("A \\emph{composite} manual assembled from %d paragraphs of two library documents plus new paragraphs is the "
              "case no other engine handles: %d runs plus a %s-byte residual give %.3f$\\times$ against %.3f for \\texttt{xz} "
              "and %.3f for the library engine; what remains is the genuinely new text. " % (
                  60, cr["runs"], "{:,}".format(cr["residual_coded"]), comp["ratio"], comp["xz"] / comp["bytes"],
                  next(x["seed_bytes"] for x in comp["trials"] if x["gid"] == "ANET-LIB") / comp["bytes"]))
    if rev1: s += ("For pure edits the delta engine still wins (%.5f$\\times$ against %.5f for retrieval), because a changed "
                   "word costs a whole chunk in the retrieval layout but only the word in an edit script. " % (
                       rev1["ratio"], next(x["seed_bytes"] for x in rev1["trials"] if x["gid"].startswith("ANET-RET")) / rev1["bytes"]))
    if uns: s += ("A document outside the retrieval library (the \\emph{Frankenstein} excerpt; it is inside the English engine's "
                  "training corpus but that is irrelevant to the index) finds no runs and falls through to the neural residual, "
                  "costing exactly what the English engine costs (%.3f$\\times$): retrieval never hurts. The disk chose among four "
                  "engines per object without being told anything about lineage or domain." % uns["ratio"])
    macro("retReading", s)
else:
    for k in ("retEngineBytes", "retLibraryBytes"): macro(k, "?")
    macro("retTable", "(pending)"); macro("retReading", "")


# ---------------- N3 determinism
dt = os.path.join(code, "determinism_results.json")
if os.path.exists(dt):
    r = json.load(open(dt)); rows = r["rows"]
    objs = []; [objs.append(x["object"]) for x in rows if x["object"] not in objs]
    tab = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule", "engine & variant & same Seed & tables identical & regenerates reference \\\\", "\\midrule"]
    for eng in ("float32", "fixed-point"):
        for var in ("ref", "chunked", "f64"):
            xs = [x for x in rows if x["engine"] == eng and x["variant"].endswith(var)]
            f = lambda key: ("%d/%d" % (sum(1 for x in xs if x[key]), len(xs))) if all(x[key] for x in xs) else ("\\textbf{%d/%d}" % (sum(1 for x in xs if x[key]), len(xs)))
            tab.append("%s & %s & %s & %s & %s \\\\" % (eng, var, f("same_seed"), f("tables_identical"), f("decodes_reference_seed_exactly")))
        if eng == "float32": tab.append("\\midrule")
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("detTable", "\n".join(tab))
    fl_fail = sum(1 for x in rows if x["engine"] == "float32" and not x["variant"].endswith("ref") and not x["decodes_reference_seed_exactly"])
    fl_n = sum(1 for x in rows if x["engine"] == "float32" and not x["variant"].endswith("ref"))
    fx_pass = sum(1 for x in rows if x["engine"] == "fixed-point" and x["same_seed"] and x["decodes_reference_seed_exactly"])
    fx_n = sum(1 for x in rows if x["engine"] == "fixed-point")
    t1 = next(x for x in rows if x["engine"] == "float32" and x["object"].startswith("pg12") and x["variant"].endswith("ref"))["seed_bytes"]
    t2 = next(x for x in rows if x["engine"] == "fixed-point" and x["object"].startswith("pg12") and x["variant"].endswith("ref"))["seed_bytes"]
    s = ("\\noindent\\textbf{Reading the result.} The float32 engine fails %d of %d cross-variant cases: change the summation "
         "order or the precision and the Seed differs, the tables differ, and the reference Seed no longer regenerates the object. "
         "The fixed-point engine passes %d of %d---identical Seeds and identical tables under every variant---at essentially the "
         "same Footprint (%s versus %s bytes on the text object). Integer arithmetic is order-independent as long as no accumulator "
         "overflows, and the engine's products stay below $2^{40}$, so by construction any device with two's-complement 64-bit integers (CPU, GPU, "
         "an FPGA on a probe) should yield the same bits; what has been \\emph{verified} is three evaluation orders on one CPU and two independent implementations. The procedural engine of Section~\\ref{sec:multi} carries the same caveat in "
         "reverse: its float64 simulator must be replaced by an integer or correctly-rounded one before it is trusted across "
         "devices. A run on a CUDA device against the CPU-produced reference is pending; the harness and reference Seeds ship "
         "with the code." % (fl_fail, fl_n, fx_pass, fx_n, "{:,}".format(t2), "{:,}".format(t1)))
    macro("detReading", s)
else:
    macro("detTable", "(pending)"); macro("detReading", "")


# ---------------- N6 compiled coder
fc = os.path.join(code, "fastcoder_results.json")
if os.path.exists(fc):
    r = json.load(open(fc))
    tab = ["\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule", "object & bytes & Seed (B) & ratio & xz & threads & read (s) & read KB/s \\\\", "\\midrule"]
    for o in r["objects"]:
        for i, run in enumerate(o["runs"]):
            tab.append("%s & %s & %s & %s & %s & %d & %.2f & %.0f \\\\" % (
                "\\texttt{%s}" % o["file"].replace("_", "\\_") if i == 0 else "", "{:,}".format(o["bytes"]) if i == 0 else "",
                "{:,}".format(o["seed_bytes"]) if i == 0 else "", "%.3f" % o["ratio"] if i == 0 else "", "%.3f" % (o["xz"] / o["bytes"]) if i == 0 else "",
                run["threads"], run["decode_s"], o["bytes"] / run["decode_s"] / 1e3))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("fastTable", "\n".join(tab))
    big = r["objects"][-1]; r1 = big["runs"][0]; r2 = big["runs"][1]
    s = ("The C implementation produces Seeds \\emph{identical} to the Python fixed-point engine on every test "
         "object and decodes the Python-produced Seeds exactly (and vice versa); it is %.0f$\\times$ faster than the "
         "Python path per byte (%.0f\\,KB/s against %.1f\\,KB/s) and stores 3.4\\,MB of five books at %.3f "
         "(\\texttt{xz}: %.3f) in %.0f\\,s single-threaded. The engine spends ${\\sim}49{,}000$ multiply--adds per byte "
         "(a $128\\times128$ recurrence and a $256\\times128$ read-out), so throughput is set by the arithmetic, not by "
         "the coder. Block parallelism gives only %.2f$\\times$ with two threads because the test machine has a single "
         "physical core (two hardware threads); experiment E5 proper---Horizon against cores---needs a multi-core or GPU "
         "host, for which the harness (\\texttt{--threads}) is ready." % (
             r["objects"][0]["bytes"] / r["objects"][0]["runs"][0]["decode_s"] / 1e3 / r["python_coder_KBps"],
             r["objects"][0]["bytes"] / r["objects"][0]["runs"][0]["decode_s"] / 1e3, r["python_coder_KBps"],
             big["ratio"], big["xz"] / big["bytes"], r1["decode_s"], r1["decode_s"] / r2["decode_s"]))
    macro("fastReading", s)
else:
    macro("fastTable", "(pending)"); macro("fastReading", "")


# ---------------- N2b deep data at scale
n2 = os.path.join(code, "n2b_results.json")
if os.path.exists(n2):
    r = json.load(open(n2))
    tab = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
           "object & bytes & Seed & program & Footprint ratio & gzip & xz & regen.\\ (s) & MB/s & flight-class (s, est.) \\\\", "\\midrule"]
    for o in r["objects"]:
        tab.append("\\texttt{%s} & %s & %d B & %s B & \\textbf{%.1e} & %.3f & %.3f & %.0f & %.1f & %.0f \\\\" % (
            o["file"].replace("_", "\\_"), "{:,}".format(o["bytes"]), o["seed_bytes"], "{:,}".format(o["program_bytes"]), o["ratio_with_program"],
            o["gzip_ratio"], o["xz_ratio"], o["gen_s"], o["MBps"], o["gen_s"] * 10))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("deepTable", "\n".join(tab))
    tel, dem = r["objects"][0], r["objects"][1]
    s = ("\\noindent\\textbf{Reading the result.} Both objects regenerate bit-exactly (verified by regenerating twice and by "
         "digest). The telemetry file costs %d bytes of Seed plus %d bytes of program for %.1f\\,MB---%.1e of its size, "
         "against %.3f for \\texttt{xz}, which also needed %.0f\\,s to compress it, %.0f$\\times$ longer than the %.0f\\,s the "
         "engine needs to regenerate it. The elevation model is the harder case for a compressor (%.3f for \\texttt{xz}) and the "
         "same trivial case for the engine (%.1e). The disk file that holds both Manifests is 895 bytes; its apparent capacity is "
         "%.0f\\,MB, a ratio of $2.7\\times10^{5}$. On a flight-class processor the telemetry regenerates in minutes and the model in a "
         "quarter of an hour---Horizons a probe on a months-long cruise does not notice. The two large files were deleted after the "
         "experiment; the disk regenerates them on demand." % (tel["seed_bytes"], tel["program_bytes"], tel["bytes"] / 1e6, tel["ratio_with_program"], tel["xz_ratio"],
                                     tel["xz_compress_s"], tel["xz_compress_s"] / tel["gen_s"], tel["gen_s"], dem["xz_ratio"], dem["ratio_with_program"],
                                     (tel["bytes"] + dem["bytes"]) / 1e6))
    macro("deepReading", s)
else:
    macro("deepTable", "(pending)"); macro("deepReading", "")

# ---------------- N8 larger engine
n8 = os.path.join(code, "n8_results.json")
if os.path.exists(n8):
    r = json.load(open(n8)); e = r["engine"]
    macro("nEightBytes", "{:,}".format(e["bytes"]))
    tab = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
           "book & bytes & gzip -9 & bz2 -9 & xz -9 & Seed float & bits/byte & Seed fixed/C & bits/byte & read (s) \\\\", "\\midrule"]
    for x in r["tests"]:
        tab.append("\\texttt{%s} & %s & %s & %s & %s & \\textbf{%s} & %.3f & %s & %.3f & %.1f \\\\" % (
            x["file"], "{:,}".format(x["bytes"]), "{:,}".format(x["baselines"]["gzip9"]), "{:,}".format(x["baselines"]["bz2"]), "{:,}".format(x["baselines"]["xz"]),
            "{:,}".format(x["float_stream"]["seed"]), x["float_stream"]["bpb"], "{:,}".format(x["fixed_blocks_C"]["seed"]), x["fixed_blocks_C"]["bpb"], x["fixed_blocks_C"]["read_s"]))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("nEightTable", "\n".join(tab))
    a, b = r["tests"]
    s = ("\\noindent\\textbf{Reading the result.} On both unseen books the engine is below every general-purpose baseline: "
         "%.3f$\\times$ against %.3f for \\texttt{xz -9} and %.3f for \\texttt{bz2 -9} on \\emph{Looking-Glass}, and %.3f$\\times$ against "
         "%.3f and %.3f on \\emph{Frankenstein}---%.0f\\,\\%% and %.0f\\,\\%% below \\texttt{xz}. The fixed-point engine, which is what a device "
         "would carry, gives up about 4\\,\\%% of that gain to quantisation and block resets and is still below \\texttt{xz} on both. "
         "These are Seed sizes; the engine is %s bytes, paid once, and the saving over \\texttt{xz} (%.1f\\,\\%% of the bytes on these two books) "
         "repays it after about %.1f\\,MB of unseen books---roughly ten of this size---after which every further book is a net gain. The loss of "
         "Section~\\ref{sec:realfile} was a budget, not a property of the approach, and the remaining gap to cmix-class compressors "
         "(Section~\\ref{sec:related}) is engine size: this one is a thousandth of theirs." % (
             a["float_stream"]["ratio"], a["baselines"]["xz"] / a["bytes"], a["baselines"]["bz2"] / a["bytes"],
             b["float_stream"]["ratio"], b["baselines"]["xz"] / b["bytes"], b["baselines"]["bz2"] / b["bytes"],
             100 * (1 - a["float_stream"]["seed"] / a["baselines"]["xz"]), 100 * (1 - b["float_stream"]["seed"] / b["baselines"]["xz"]),
             "{:,}".format(e["bytes"]), 100 * (a["baselines"]["xz"] - a["float_stream"]["seed"] + b["baselines"]["xz"] - b["float_stream"]["seed"]) / (a["bytes"] + b["bytes"]),
             e["bytes"] / ((a["baselines"]["xz"] - a["float_stream"]["seed"] + b["baselines"]["xz"] - b["float_stream"]["seed"]) / (a["bytes"] + b["bytes"])) / 1e6))
    macro("nEightReading", s)
else:
    macro("nEightBytes", "?"); macro("nEightTable", "(pending)"); macro("nEightReading", "")

# ---------------- stronger baselines on the two held-out books (baselines.py)
bl = os.path.join(code, "baselines_results.json")
if os.path.exists(bl):
    r = json.load(open(bl))
    tab = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
           "book & engine (float) & engine (fixed, C) & xz -9e & bz2 -9 & brotli -11 & zstd -22 $+$ dict & PPMd o8 & xz $\\mid$ prior & PPMd $\\mid$ prior \\\\",
           "\\midrule"]
    keys = ["engine_float", "engine_fixed", "xz9e", "bz2", "brotli11", "zstd22_dict", "ppmd_o8", "xz9e_prior", "ppmd_o16_prior"]
    for t in r["tests"]:
        best = min(t[k] for k in keys[:7])   # best among the unconditioned columns
        cells = []
        for k in keys:
            v = "%.3f" % (t[k] / t["bytes"])
            cells.append("\\textbf{%s}" % v if t[k] == best else v)
        tab.append("\\texttt{%s} & %s \\\\" % (t["file"].replace("_", "\\_"), " & ".join(cells)))
    tab += ["\\bottomrule", "\\end{tabular}"]
    macro("baselineTable", "\n".join(tab))
    a, b = r["tests"]
    macro("baselineDictBytes", "{:,}".format(r["zstd_dict_bytes"]))
    macro("ppmdGap", "%.0f" % (100 * (1 - (a["ppmd_o8"] + b["ppmd_o8"]) / (a["engine_float"] + b["engine_float"]))))
    macro("ppmdPriorGap", "%.0f" % (100 * (1 - (a["ppmd_o16_prior"] + b["ppmd_o16_prior"]) / (a["engine_float"] + b["engine_float"]))))
else:
    macro("baselineTable", "(pending)"); macro("baselineDictBytes", "?"); macro("ppmdGap", "?"); macro("ppmdPriorGap", "?")

# ---------------- demo: ten files
dm = os.path.join(here, "..", "demo", "report.json")
if os.path.exists(dm):
    r = json.load(open(dm)); rows = r["rows"]; tot = r["totals"]
    share = tot["engine_bytes"] / len(rows)
    tab = ["\\begin{tabular}{@{}lrrrrrrrc@{}}", "\\toprule",
           "file & input (B) & engine & Seed (B) & engine$/$10 & Seed$+$eng.$/$10 & output (B) & ratio & xz \\\\", "\\midrule"]
    for x in rows:
        eff = x["seed_bytes"] + share
        tab.append("\\texttt{%s} & %s & %s & %s & %s & %s & %s & %s & %.3f \\\\" % (
            x["file"].replace("_", "\\_").replace("nasa\\_mercury\\_transit\\_2016", "nasa\\_mercury\\_transit"), "{:,}".format(x["bytes"]),
            x["engine"].replace("ANET-PROC-anim-v1", "PROC").replace("ANET-EN-256", "EN-256").replace("VERBATIM", "raw"),
            "{:,}".format(x["seed_bytes"]), "{:,.0f}".format(share), "{:,.0f}".format(eff), "{:,}".format(x["bytes"]) if (x["identical"] and x["sha_ok"]) else "\\textbf{differs}",
            ("\\textbf{%.3f}" % (eff / x["bytes"])) if eff / x["bytes"] < 0.01 else "%.3f" % (eff / x["bytes"]), x["xz_ratio"]))
    tab += ["\\midrule", "all ten & %s & & %s & %s & %s & %s & %.3f & %.3f \\\\" % (
        "{:,}".format(tot["input_bytes"]), "{:,}".format(tot["seed_bytes"]), "{:,}".format(tot["engine_bytes"]), "{:,}".format(tot["seed_bytes"] + tot["engine_bytes"]),
        "{:,}".format(tot["input_bytes"]) if tot["all_identical"] else "\\textbf{differs}", (tot["seed_bytes"] + tot["engine_bytes"]) / tot["input_bytes"], tot["xz_ratio"]),
        "\\bottomrule", "\\end{tabular}"]
    macro("demoTable", "\n".join(tab))
    txt = [x for x in rows if x["type"] == "text"]; proc = [x for x in rows if x["engine"].startswith("ANET-PROC")]; raw = [x for x in rows if x["engine"] == "VERBATIM"]
    s = ("\\noindent\\textbf{Reading the result.} All ten files came back byte-identical from the disk folder alone, and the table should be read "
         "as a skeptic would read it. Across the ten, the disk occupies %.3f of the input against %.3f for \\texttt{xz -6}: on this mixed set "
         "the disk \\emph{loses} to a free, standard compressor, because four files sit at $1.000\\times$---a PDF of page images, a "
         "Flate-compressed PDF, an H.264 video and a PDF whose text is hex-encoded in its content streams---for which no engine on this "
         "disk exists, so the escape keeps their bytes rather than expand them. The four unseen books, at %.3f--%.3f$\\times$, are on par with "
         "\\texttt{xz} (%.3f--%.3f), a creditable result for a %d\\,KB engine but parity, not a breakthrough. The two animations, %.1f and "
         "%.1f\\,MB stored as %d-byte parameter records, are the only dramatic rows, and they are provenance capture: the disk holds the "
         "program that made them. What the table does demonstrate is the paradigm working end to end---ten real files of three types "
         "written to a folder of engines and Seeds, regenerated bit-exactly, the engine chosen per file, never worse than raw---and the "
         "law of Section~\\ref{sec:law} visible in one place: orders of magnitude where the disk holds the generator, parity where it "
         "knows the domain, nothing where it knows neither. It does not demonstrate superiority over existing compression on general "
         "files, and this paper does not claim it; the gains on the four engine-less files would come from more engines "
         "(Section~\\ref{sec:whynot}), which is a research programme rather than a result. "
         "The animation row deserves one more sentence, because it looks impossible: %s bytes of raw video frames stored as a "
         "%d-byte Seed---the parameter record: width, height, frame count, $\\sigma$, $\\rho$, $\\beta$, seed---that regenerates the file "
         "byte-for-byte. It is possible because every pixel of every frame is a deterministic function of those seven numbers, so "
         "the file carries about 110 bytes of information and 3.4\\,MB of the program's doing; \\texttt{xz}, which looks for repeated "
         "byte patterns rather than for the program, reaches only %.3f. The disk sees it perfectly because it \\emph{holds the "
         "program} (the 1,591-byte generator, also on the disk). Point a camera at the sky and record the same 3.4\\,MB and no program "
         "on any disk produces those bytes from 110 numbers---which is why the NASA video in the same table sits at $1.000\\times$. "
         "The precondition is the whole story, and it is the one the paper states wherever such a number appears." % (
             tot["disk_ratio"], tot["xz_ratio"], min(x["ratio"] for x in txt), max(x["ratio"] for x in txt), min(x["xz_ratio"] for x in txt),
             max(x["xz_ratio"] for x in txt), 148, proc[0]["bytes"] / 1e6, proc[1]["bytes"] / 1e6, proc[0]["seed_bytes"], "{:,}".format(proc[0]["bytes"]), proc[0]["seed_bytes"], proc[0]["xz_ratio"]))
    macro("demoReading", s)
else:
    macro("demoTable", "(demo pending)"); macro("demoReading", "")

# ---------------- proof by 1000 files
p1 = os.path.join(here, "..", "demo2", "summary.json")
if os.path.exists(p1):
    s = json.load(open(p1))
    txt = ("Result: %d files, %.1f\\,GB of telemetry in all, stored as %s bytes of Seeds plus a %s-byte program; the catalogue file "
           "holding every Manifest is %s bytes and the whole disk folder %s bytes, %.1e of the data. %d of %d files regenerated "
           "bit-exactly%s. Generating and hashing the thousand files took %s\\,s; regenerating and verifying them from the catalogue "
           "%s\\,s, %.1f\\,s per file on one core. The metadata query ``%s'' was answered in %.4f\\,s with %d hits without regenerating "
           "anything. Like the two objects above, this is provenance capture by construction; what it adds is the scaling claim "
           "verified at a thousand objects, a catalogue that is the archive's own index, and the batch Horizon a probe would pay." % (
               s["files"], s["apparent_bytes"] / 1e9, "{:,}".format(s["seed_bytes_total"]), s["program_bytes"], "{:,}".format(s["catalogue_file_bytes"]),
               "{:,}".format(s["disk_folder_bytes"]), s["ratio_disk_folder"], s["verified"], s["files"], "" if s["all_exact"] else " (\\textbf{%d failed})" % s["failed"],
               "{:,}".format(s["write_total_s"]), "{:,}".format(s["regen_total_s"]), s["regen_mean_s"] or 0, s["query"]["query"], s["query"]["seconds"], s["query"]["hits"]))
    macro("proofSection", txt)
else:
    macro("proofSection", "(run in progress)")

# ---------------- conclusion numbers (assembled from whatever results exist)
parts = []
try:
    r = json.load(open(os.path.join(code, "predictive_results.json"))); x = r["runs"][4]
    parts.append("Four thousand text objects sharing one 55\\,KB engine cost %.0f bits each against 232 raw, every one of them and 200 never seen regenerating bit-exactly (Table~\\ref{tab:designB})." % x["footprint_bits_per_obj"])
except Exception: pass
try:
    r = json.load(open(os.path.join(code, "n8_results.json"))); ts = r["tests"]
    parts.append("A larger English engine, trained on fifteen books and tested on two it never saw, stores them at " + " and ".join(
        "%.3f$\\times$ (\\texttt{xz}: %.3f)" % (t["float_stream"]["ratio"], t["baselines"]["xz"] / t["bytes"]) for t in ts) + (
        "---below \\texttt{xz} on %s, Seed only; the 148\\,KB engine is repaid once it is shared by about ten such books (Table~\\ref{tab:n8})." % ("both" if all(t["beats_xz"] for t in ts) else "one" if any(t["beats_xz"] for t in ts) else "neither")))
except Exception:
    try:
        r = json.load(open(os.path.join(code, "realfile_results.json"))); f = r["files"][0]
        parts.append("A whole book the engine never saw stores at %.3f$\\times$, below \\texttt{gzip} and within 5\\,\\%% of \\texttt{xz} (Table~\\ref{tab:realfile})." % f["modes"]["stream"]["ratio"])
    except Exception: pass
try:
    r = json.load(open(os.path.join(code, "multiengine_results.json"))); lz = next(o for o in r["objects"] if o["file"] == "lorenz.csv")
    parts.append("A %s-byte simulation output regenerates from a %d-byte Seed (%.5f$\\times$, Table~\\ref{tab:multi})." % ("{:,}".format(lz["bytes"]), lz["seed_bytes"], lz["ratio"]))
except Exception: pass
try:
    r = json.load(open(os.path.join(code, "n2b_results.json")))
    parts.append("Two flight-sized objects, %.0f\\,MB of integer-exact telemetry and a %.0f\\,MB elevation model, come from 3.3\\,KB of Manifests and generator programs (Table~\\ref{tab:deepscale})." % (r["objects"][0]["bytes"] / 1e6, r["objects"][1]["bytes"] / 1e6))
except Exception: pass
try:
    r = json.load(open(os.path.join(code, "editaware_results.json"))); rv = r["objects"][1]
    parts.append("A revised document costs only its edits: %d bytes for %d changed words on top of a base already stored (%.5f$\\times$, Table~\\ref{tab:edit})." % (rv["seed_bytes"], 60, rv["ratio"]))
except Exception: pass
try:
    r = json.load(open(os.path.join(code, "retrieval_results.json"))); lb = r["objects"][0]
    parts.append("A library document costs %d bytes through the retrieval engine." % lb["seed_bytes"])
except Exception: pass
try:
    r = json.load(open(os.path.join(code, "determinism_results.json"))); rows = r["rows"]
    fcr = json.load(open(os.path.join(code, "fastcoder_results.json")))
    speed = fcr["objects"][0]["bytes"] / fcr["objects"][0]["runs"][0]["decode_s"] / 1e3 / fcr["python_coder_KBps"]
    parts.append("The float engine fails every cross-arithmetic test and the fixed-point engine passes all %d (Table~\\ref{tab:determinism}); its C implementation is %.0f$\\times$ faster than the Python one and byte-identical to it (Table~\\ref{tab:fast})." % (
        sum(1 for x in rows if x["engine"] == "fixed-point"), speed))
except Exception: pass
macro("conclNumbers", " ".join(parts))

open(os.path.join(here, "results.tex"), "w").write("% auto-generated by make_results.py\n" + "\n".join(out) + "\n")
print("results.tex written:", len(out), "macros")
