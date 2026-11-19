"""Build the project presentation (PDF + .pptx, 16:9) from saved figures and result CSVs.

  python report/make_presentation.py --results ../results/v3 --out report/presentation
Every table is rendered from a result CSV of this project; speaker notes (.pptx) hold what to say.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NAVY, GREY = "#1F3A5F", "#555555"


def table_png(df, out, col_w=None, fontsize=11, header_color="#DCE3EC", highlight_rows=()):
    import textwrap
    nrow, ncol = df.shape
    col_w = col_w or [1.0] * ncol
    df = df.copy()
    df.columns = [textwrap.fill(str(c), max(8, int(w * 11))) for c, w in zip(df.columns, col_w)]
    hdr_lines = max(str(c).count(chr(10)) + 1 for c in df.columns)
    fig_w = sum(col_w) * 1.35
    fig, ax = plt.subplots(figsize=(fig_w, 0.42 * (nrow + 1) + 0.1))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center",
                 colWidths=[w / sum(col_w) for w in col_w])
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#9AA5B1")
        if r == 0:
            cell.set_height(cell.get_height() * (0.75 + 0.55 * hdr_lines))
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        elif r - 1 in highlight_rows:
            cell.set_facecolor("#FFF4D6")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def build_tables(R, T):
    os.makedirs(T, exist_ok=True)
    tabs = {}
    sc = pd.read_csv(os.path.join(R, "source_concentration_class.csv"))
    d = sc[["cls", "clips", "sources", "largest_clips", "effective_sources"]].copy()
    d.columns = ["class", "clips", "recordings", "clips from largest recording", "effective no. of recordings"]
    d["effective no. of recordings"] = d["effective no. of recordings"].round(1)
    tabs["conc"] = table_png(d, os.path.join(T, "t_conc.png"), [1, 0.8, 1, 1.6, 1.6], highlight_rows=[0])

    dup = pd.read_csv(os.path.join(R, "duplicates.csv"))
    rc = pd.read_csv(os.path.join(REPO, "docs", "results", "audit", "duplicates_recheck.csv"))
    ctl = pd.read_csv(os.path.join(R, "duplicates_control.csv"))
    conf = dup[dup.status == "confirmed"]
    d = pd.DataFrame([["candidate pairs screened (AST similarity)", len(dup)],
                      ["confirmed same audio (waveform match >= 0.9)", len(conf)],
                      ["   ... across different websites", int(conf.cross_website.sum())],
                      ["   ... across the OFFICIAL train/test split", int(conf.cross_official_split.sum())],
                      ["   ... with conflicting labels", int(conf.label_conflict.sum())],
                      ["random pairs (control): highest similarity", f"{ctl.xcorr.max():.2f}"],
                      ["strict re-check (bounded score): pairs that still hold", f"{int((rc.xcorr_bounded[rc.status == 'confirmed'] >= 0.9).sum())} of {len(conf)}"]],
                     columns=["duplicate audit", "count"])
    tabs["dup"] = table_png(d, os.path.join(T, "t_dup.png"), [4.2, 1.2], highlight_rows=[3])

    sp = pd.read_csv(os.path.join(R, "dl", "splits.csv"))
    d = sp.groupby("source_disjoint").agg(clips=("filename", "size"), recordings=("group", "nunique")).reindex(["train", "val", "test"]).reset_index()
    d.columns = ["partition", "clips", "recordings"]
    tabs["split"] = table_png(d, os.path.join(T, "t_split.png"), [1.2, 1, 1])

    wh = pd.read_csv(os.path.join(R, "website_holdout_recordings.csv"))
    wh = wh[wh.heldout.isin(["aigei", "freesound"])]
    names = {"ast": "frozen AST", "struct-nf": "structure", "spec": "spectral"}
    d = pd.DataFrame({"held-out website": wh.heldout.str.capitalize(), "features": wh.featureset.map(names),
                      "trained WITH other recordings of it": wh.control.round(3), "trained WITHOUT it": wh.holdout.round(3),
                      "loss (macro-F1) [95% CI]": [f"{g:.3f} [{lo:.3f}, {hi:.3f}]" for g, lo, hi in zip(wh.gap, wh.ci_lo, wh.ci_hi)]})
    tabs["web"] = table_png(d, os.path.join(T, "t_web.png"), [1.3, 1.1, 2.0, 1.4, 2.0],
                            highlight_rows=[i for i, f in enumerate(d.features) if f == "frozen AST"])

    m = pd.read_csv(os.path.join(HERE, "metrics_summary.csv"))
    d = pd.DataFrame({"model": m.system, "accuracy": m.accuracy.round(3), "precision": m.macro_precision.round(3),
                      "recall": m.macro_recall.round(3), "F1 (macro)": m.macro_f1.round(3), "seeds": m.seeds})
    tabs["models"] = table_png(d, os.path.join(T, "t_models.png"), [3.0, 1, 1, 1, 1, 0.7], fontsize=10.5,
                               highlight_rows=[len(d) - 1, len(d) - 3])

    def eff(path):
        e = pd.read_csv(path).set_index("quantity")
        return [f"{e.loc[k, 'estimate']:+.3f} [{e.loc[k, 'ci_lo']:+.3f}, {e.loc[k, 'ci_hi']:+.3f}]"
                for k in ["balanced_main_effect", "contrastive_main_effect", "interaction"]]
    cpu = eff(os.path.join(R, "report", "ast_2x2_effects.csv"))
    gpu = eff(os.path.join(REPO, "docs", "results", "colab_k0_pc200", "ast_2x2_effects.csv"))
    d = pd.DataFrame({"effect on macro-F1": ["source-balanced sampling", "cross-source contrastive loss", "both together (interaction)"],
                      "laptop: last layer trained": cpu, "GPU: all 12 layers trained": gpu})
    tabs["eff"] = table_png(d, os.path.join(T, "t_eff.png"), [2.4, 2.2, 2.2], highlight_rows=[0])

    ec = pd.read_csv(os.path.join(R, "error_clustering.csv"))
    ecm = pd.read_csv(os.path.join(R, "error_clustering_ml.csv"))
    pick = {"ast_ft_source_disjoint_standard_s0": "AST fine-tuned", "ast_ftsc_source_disjoint_source_s0": "AST combined",
            "cnn_source_disjoint_standard_s0": "CNN", "ml_ast_source_disjoint": "frozen AST + LR"}
    e = pd.concat([ec, ecm])
    e = e[e.run.isin(pick)]
    d = pd.DataFrame({"model": e.run.map(pick), "error clustering (ICC)": e.error_icc.round(2),
                      "effective test size": [f"{x:.0f} of {int(n)}" for x, n in zip(e.effective_n, e.test_clips)],
                      "errors in worst 10% of recordings": (100 * e.top10pct_recordings_share_of_errors).round(0).astype(int).astype(str) + "%",
                      "CI width: per clip vs per recording": [f"{a:.3f} vs {b:.3f}" for a, b in zip(e.ci_width_clip_bootstrap, e.ci_width_source_bootstrap)]})
    tabs["err"] = table_png(d, os.path.join(T, "t_err.png"), [1.5, 1.4, 1.4, 1.9, 2.2])

    d = pd.DataFrame([["full clip", "0.788", "0.784"], ["vocalisation only", "0.732", "0.744"],
                      ["same amount cut at random (control)", "0.671", "0.655"], ["vocalisation REMOVED", "0.304", "0.277"],
                      ["chance level", "<= 0.16", "<= 0.16"]], columns=["audio given to the model", "AST ordinary", "AST combined"])
    tabs["bg"] = table_png(d, os.path.join(T, "t_bg.png"), [3.4, 1.2, 1.2], highlight_rows=[3])
    return tabs


def slides(R, T, tabs):
    rep, fig, scr = os.path.join(R, "report"), os.path.join(R, "figures"), os.path.join(HERE, "screens")
    return [
        dict(kind="title"),
        dict(title="The question",
             bullets=["Nonspeech7k: 7,014 clips of 7 human sounds (breath, cough, crying, laugh, scream, sneeze, yawn) from 3 websites",
                      "Clips were cut from longer recordings, so one recording can supply hundreds of clips",
                      "Do deep models recognise the SOUND, or the RECORDING? And how do we evaluate them honestly?"],
             notes="Nonspeech7k has 7,014 clips of seven human sounds. They were cut from longer recordings. We asked whether "
                   "deep models recognise the sound or just the recording, and how to evaluate them honestly."),
        dict(title="Hidden problem: a few recordings dominate some classes", imgs=[os.path.join(fig, "A2_source_concentration.png"), tabs["conc"]],
             bullets=["Breath: 1,850 clips, but 827 come from ONE YouTube video (effective number of recordings: 4.6)"],
             notes="The 1,850 breath clips come from only about five effective recordings; 827 come from one YouTube video. "
                   "A model can learn the recording instead of the breath. Source: source_concentration_class.csv."),
        dict(title="Duplicate audio and leakage", imgs=[tabs["dup"]],
             bullets=["Published File_IDs are wrong for Aigei (numbering restarts), so we rebuilt recording identities",
                      "7.2% of the official test clips share a recording or duplicate with the training set"],
             notes="We found 137 re-uploaded duplicate audio pairs; 23 cross the official train/test split. Random pairs never "
                   "exceed 0.41 similarity. A stricter re-check confirmed 133 of the 137. Source: duplicates.csv, duplicates_recheck.csv."),
        dict(title="Leakage-safe evaluation: split by recording", imgs=[os.path.join(T, "flowchart_wide.png")],
             bullets=["Train 5,127 clips / 978 recordings  |  validation 727 / 225  |  test 1,160 / 300",
                      "No recording, and no copy of one, appears in two partitions",
                      "Settings chosen on validation; the test set is used once per model"],
             notes="We split by recording, so the test set contains only recordings the model has never heard."),
        dict(title="Website shift is real", imgs=[tabs["web"]],
             bullets=["Never hearing a website costs about 6 macro-F1 points (frozen AST: Aigei 0.057, Freesound 0.056)",
                      "Training size, class mix and number of recordings matched; the CIs exclude zero"],
             notes="If the model never hears a website, it loses about 6 macro-F1 points, with everything else matched. "
                   "Source: website_holdout_recordings.csv."),
        dict(title="Model comparison (test = unseen recordings)", imgs=[tabs["models"]],
             bullets=["Handcrafted features about 0.45 F1; CNN from scratch 0.77; fine-tuned AST up to 0.82",
                      "Frozen AST + LR (0.85) used 10 s inputs and about 4x more training clips: not a like-for-like comparison"],
             notes="Precision, recall and F1 are macro-averaged. Fine-tuned transformer is the best like-for-like model. Source: metrics_summary.csv."),
        dict(title="Where the models go wrong", imgs=[os.path.join(rep, "G3_confusion_matrices.png")],
             bullets=["Crying is the hardest class, mostly confused with laughter"],
             notes="Row-normalised confusion matrices over 3 seeds."),
        dict(title="Training: loss falls, validation F1 rises", imgs=[os.path.join(T, "learning_curves_2x2.png")],
             bullets=["4 training recipes x 3 seeds each; gradients and weight changes of the pretrained layers were logged"],
             notes="Every run trained properly; we checked that the pretrained transformer weights really changed."),
        dict(title="Our experiment: does source-aware training help?", imgs=[os.path.join(rep, "G6_ast_2x2_source_aware_training.png"), tabs["eff"]],
             bullets=["Sampling by recording helps a little (+3.5 points on the laptop, significant; +1.9 on GPU)",
                      "The cross-source contrastive loss does NOT help, in either experiment (reported as a null result)"],
             notes="Two fixes in a 2x2 design, rules fixed before running. Sampling by recording helped modestly; the contrastive loss "
                   "did nothing. Source: ast_2x2_effects.csv (laptop and Colab)."),
        dict(title="Why it did not help: models organise sounds by recording", imgs=[os.path.join(rep, "G9_source_clustering_of_embeddings.png")],
             bullets=["For an unseen recording, the closest match is from the SAME recording 72% of the time (chance 7.6%)",
                      "No training recipe changed this"],
             notes="Even for new recordings, the model's nearest neighbour comes from the same recording most of the time."),
        dict(title="Errors come from a few recordings", imgs=[tabs["err"]],
             bullets=["10% of recordings cause about 80% of all errors", "Per-clip confidence intervals are about 2x too narrow: report per recording"],
             notes="Error clustering is strong for every model. Source: error_clustering.csv."),
        dict(title="What does the model listen to? (surprising result)", imgs=[os.path.join(HERE, "build", "background_probe.png"), tabs["bg"]],
             bullets=["Vocalisation alone keeps about 93% of the performance: models mainly listen to the sound",
                      "With the vocalisation cut out, models still score about 2x chance from the leftover audio"],
             notes="We cut the vocalisation out of each clip. Performance falls from 0.79 to 0.30, but 0.30 is still twice chance. "
                   "Source: background_probe SUMMARY.md / Colab output."),
        dict(title="Live demo: an unseen scream", imgs=[os.path.join(scr, "demo_cell7_0.png"), os.path.join(scr, "demo_cell11_1.png")],
             bullets=["CNN said crying; both fine-tuned AST models said screaming (raw softmax scores, not calibrated)"],
             notes="Optional: open professor_demo/LIVE_DL_DEMO_executed.ipynb."),
        dict(title="Conclusions and limitations",
             bullets=["Website shift is real (about 6 macro-F1 points)",
                      "Sampling by recording helps a little; cross-source contrastive learning does not",
                      "Errors cluster by recording: report results per recording, not per clip",
                      "Models mainly listen to the vocalisation, but leftover audio still reveals the class",
                      "Limits: one dataset, small training budget, the contrastive method is not new (applied under strict controls)"],
             notes="Four findings and honest limits."),
    ]


def layout(sizes, x0, y0, aw, ah, gap):
    """Place images side by side or stacked, whichever is bigger. Returns [(x, y_top, w, h)] (y from top)."""
    tot_r = sum(w / h for w, h in sizes)
    hh = min(ah, (aw - gap * (len(sizes) - 1)) / tot_r)
    tot_c = sum(h / w for w, h in sizes)
    ww = min(aw, (ah - gap * (len(sizes) - 1)) / tot_c)
    row_area = sum(hh * hh * w / h for w, h in sizes)
    col_area = sum(ww * ww * h / w for w, h in sizes)
    out = []
    if row_area >= col_area:
        x = x0 + (aw - (hh * tot_r + gap * (len(sizes) - 1))) / 2
        for w, h in sizes:
            out.append((x, y0 + (ah - hh) / 2, hh * w / h, hh)); x += hh * w / h + gap
    else:
        y = y0 + (ah - (ww * tot_c + gap * (len(sizes) - 1))) / 2
        for w, h in sizes:
            out.append((x0 + (aw - ww) / 2, y, ww, ww * h / w)); y += ww * h / w + gap
    return out


TITLE = "Recording Identity, Website Shift and What Deep Models Listen To"
SUB = "Leakage-safe deep learning for human non-speech vocalisation recognition (Nonspeech7k)"
WHO = ["Harbani Kaur (1024030848)  |  Ina Sharma (1024030866)", "Submitted to: Ms. Arushi Madaan",
       "Machine Learning Lab  |  Department of Computer Science and Engineering", "Thapar Institute of Engineering and Technology  |  2026-27"]


def build_pdf(S, out):
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    W, H = 13.333 * 72, 7.5 * 72
    c = canvas.Canvas(out, pagesize=(W, H))
    for i, s in enumerate(S):
        if s.get("kind") == "title":
            c.setFillColor(HexColor(NAVY)); c.rect(0, H * 0.45, W, H * 0.55, fill=1, stroke=0)
            from reportlab.lib.utils import simpleSplit
            c.setFillColor(HexColor("#FFFFFF")); c.setFont("Times-Bold", 34)
            tl = simpleSplit(TITLE, "Times-Bold", 34, W - 120)
            for k, line in enumerate(tl):
                c.drawCentredString(W / 2, H * 0.8 - k * 42, line)
            c.setFont("Times-Roman", 20); c.drawCentredString(W / 2, H * 0.8 - len(tl) * 42 - 12, SUB)
            c.setFillColor(HexColor("#222222")); c.setFont("Times-Roman", 18)
            for k, line in enumerate(WHO):
                c.drawCentredString(W / 2, H * 0.34 - k * 30, line)
            c.showPage(); continue
        c.setFillColor(HexColor(NAVY)); c.rect(0, H - 70, W, 70, fill=1, stroke=0)
        c.setFillColor(HexColor("#FFFFFF")); c.setFont("Times-Bold", 26); c.drawString(36, H - 47, s["title"])
        c.setFillColor(HexColor(GREY)); c.setFont("Times-Roman", 11); c.drawRightString(W - 30, 18, str(i))
        from reportlab.lib.utils import simpleSplit
        bl = s.get("bullets", [])
        imgs = s.get("imgs", [])
        fs = 17 if imgs else 24
        lines = []
        for b in bl:
            wrapped = simpleSplit(b, "Times-Roman", fs, W - 130)
            lines += [("•  " if k == 0 else "    ") + t for k, t in enumerate(wrapped)]
        step = fs * 1.45
        text_h = step * len(lines) + 16 if imgs else 0
        if imgs:
            top, bottom = H - 82, 22 + text_h
            sizes = [ImageReader(p).getSize() for p in imgs]
            for p, (bx, by_top, bw, bh) in zip(imgs, layout(sizes, 36, 0, W - 72, top - bottom, 18)):
                c.drawImage(ImageReader(p), bx, top - by_top - bh, bw, bh)
            y = 22 + text_h - step
        else:
            y = H - 150
        c.setFillColor(HexColor("#111111")); c.setFont("Times-Roman", fs)
        for line in lines:
            c.drawString(50, y, line)
            y -= step if imgs else step * 1.25
        c.showPage()
    c.save()


def build_pptx(S, out):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Emu, Inches, Pt
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    navy = RGBColor(0x1F, 0x3A, 0x5F)

    def textbox(sl, x, y, w, h, lines, size, bold=False, color=RGBColor(0x11, 0x11, 0x11), center=False, bullet=False):
        tb = sl.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
        from pptx.enum.text import PP_ALIGN
        for k, line in enumerate(lines):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.text = ("• " + line) if bullet else line
            p.font.size, p.font.bold, p.font.name = Pt(size), bold, "Times New Roman"
            p.font.color.rgb = color
            p.space_after = Pt(6)
            if center:
                p.alignment = PP_ALIGN.CENTER
        return tb

    from PIL import Image as PImage
    for i, s in enumerate(S):
        sl = prs.slides.add_slide(blank)
        if s.get("kind") == "title":
            bar = sl.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(4.1)); bar.fill.solid(); bar.fill.fore_color.rgb = navy; bar.line.fill.background()
            textbox(sl, Inches(0.5), Inches(1.2), Inches(12.3), Inches(1.2), [TITLE], 36, True, RGBColor(255, 255, 255), True)
            textbox(sl, Inches(0.5), Inches(2.6), Inches(12.3), Inches(0.8), [SUB], 20, False, RGBColor(255, 255, 255), True)
            textbox(sl, Inches(0.5), Inches(4.5), Inches(12.3), Inches(2.6), WHO, 18, False, center=True)
            sl.notes_slide.notes_text_frame.text = "Introduce the project and both team members."
            continue
        bar = sl.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.97)); bar.fill.solid(); bar.fill.fore_color.rgb = navy; bar.line.fill.background()
        textbox(sl, Inches(0.45), Inches(0.15), Inches(12.4), Inches(0.75), [s["title"]], 28, True, RGBColor(255, 255, 255))
        bl, imgs = s.get("bullets", []), s.get("imgs", [])
        if imgs:
            text_h = 0.36 * sum(1 + len(b) // 110 for b in bl) + 0.35
            top, avail_h, avail_w = 1.15, 7.5 - 1.15 - text_h - 0.25, 12.4
            sizes = [PImage.open(p).size for p in imgs]
            for p, (bx, by, bw, bh) in zip(imgs, layout(sizes, 0.45, top, avail_w, avail_h, 0.25)):
                sl.shapes.add_picture(p, Inches(bx), Inches(by), width=Inches(bw), height=Inches(bh))
            textbox(sl, Inches(0.6), Inches(7.5 - text_h - 0.15), Inches(12.2), Inches(text_h), bl, 17, bullet=True)
        else:
            textbox(sl, Inches(0.7), Inches(1.5), Inches(12), Inches(5.5), bl, 24, bullet=True)
        sl.notes_slide.notes_text_frame.text = s.get("notes", "")
    prs.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = os.path.join(a.out, "assets")
    os.makedirs(T, exist_ok=True)
    tabs = build_tables(a.results, T)
    # wide flowchart (horizontal) for a 16:9 slide
    from matplotlib.patches import FancyBboxPatch
    steps = ["Audio\n7,014 clips", "Rebuild\nrecordings", "Find\nduplicates", "Split by\nrecording", "Features\n(spectrograms)",
             "Models\nLR / CNN / AST", "Select on\nvalidation", "Test once\n+ diagnostics"]
    fig, ax = plt.subplots(figsize=(13, 1.9)); ax.axis("off")
    for k, t in enumerate(steps):
        x = k / len(steps)
        ax.add_patch(FancyBboxPatch((x + 0.008, 0.15), 1 / len(steps) - 0.03, 0.7, boxstyle="round,pad=0.01", fc="#E8EEF7", ec="#333"))
        ax.text(x + 0.5 / len(steps) - 0.007, 0.5, t, ha="center", va="center", fontsize=11, family="serif")
        if k < len(steps) - 1:
            ax.annotate("", xy=(x + 1 / len(steps) + 0.006, 0.5), xytext=(x + 1 / len(steps) - 0.02, 0.5), arrowprops=dict(arrowstyle="-|>", lw=1.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(os.path.join(T, "flowchart_wide.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    import shutil
    shutil.copy(os.path.join(HERE, "build", "learning_curves_2x2.png"), os.path.join(T, "learning_curves_2x2.png"))
    S = slides(a.results, T, tabs)
    build_pdf(S, os.path.join(a.out, "Nonspeech7k_Presentation.pdf"))
    build_pptx(S, os.path.join(a.out, "Nonspeech7k_Presentation.pptx"))
    print("wrote", a.out, len(S), "slides")


if __name__ == "__main__":
    main()
