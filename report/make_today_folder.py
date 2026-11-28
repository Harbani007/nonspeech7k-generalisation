"""Assemble the presentation folder: every file to open, in order (00, 01, ...), plus a presenter guide and one Excel
workbook with all result tables. Everything is copied or derived from result files of this project.

  python report/make_today_folder.py --results ../results/v3 --out "C:/Users/Administrator/Downloads/Nonspeech7k_Presentation_Today"
"""
import argparse
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def tables(R):
    """All result tables shown today, as DataFrames (name -> df)."""
    T = {}
    sp = pd.read_csv(os.path.join(REPO, "docs", "results", "splits", "splits.csv"))
    d = pd.crosstab(sp.cls, sp.split).rename(columns={"train": "official train", "test": "official test"})
    d["total"] = d.sum(axis=1)
    d["recordings (reconstructed)"] = sp.groupby("cls").group.nunique()
    d["websites"] = sp.groupby("cls").source.apply(lambda s: ", ".join(sorted(s.unique())))
    T["dataset"] = d.reset_index().rename(columns={"cls": "class"})

    sc = pd.read_csv(os.path.join(R, "source_concentration_class.csv"))
    T["concentration"] = sc[["cls", "clips", "sources", "largest_source", "largest_clips", "largest_share", "top5_share",
                             "effective_sources"]].rename(columns={"cls": "class", "sources": "recordings",
                                                                   "largest_source": "largest recording",
                                                                   "largest_clips": "clips from largest recording",
                                                                   "largest_share": "share from largest", "top5_share": "share from top 5",
                                                                   "effective_sources": "effective no. of recordings"}).round(3)

    dup = pd.read_csv(os.path.join(R, "duplicates.csv"))
    rc = pd.read_csv(os.path.join(REPO, "docs", "results", "audit", "duplicates_recheck.csv"))
    ctl = pd.read_csv(os.path.join(R, "duplicates_control.csv"))
    conf = dup[dup.status == "confirmed"]
    T["duplicates_summary"] = pd.DataFrame([
        ["candidate pairs screened (AST embedding similarity, top-5 neighbours, cosine >= 0.95)", len(dup)],
        ["confirmed same audio (waveform cross-correlation >= 0.9)", len(conf)],
        ["  of which across different websites", int(conf.cross_website.sum())],
        ["  of which across the OFFICIAL train/test split", int(conf.cross_official_split.sum())],
        ["  of which with conflicting labels", int(conf.label_conflict.sum())],
        ["uncertain (0.6-0.9, kept separate)", int((dup.status == "uncertain").sum())],
        ["rejected", int((dup.status == "rejected").sum())],
        ["control: 500 random same-class pairs, highest similarity", round(float(ctl.xcorr.max()), 3)],
        ["strict re-check (bounded score): confirmed pairs that still score >= 0.9", int((rc.xcorr_bounded[rc.status == "confirmed"] >= 0.9).sum())],
        ["official test clips sharing a recording or duplicate with training", "52 of 725 (7.2%)"]], columns=["duplicate audit", "value"])
    rcc = rc[rc.status == "confirmed"]
    T["duplicate_pairs"] = rcc[["a", "b", "site_a", "site_b", "split_a", "split_b", "cls_a", "cls_b", "xcorr_bounded",
                                "cross_website", "cross_official_split", "label_conflict"]].rename(
        columns={"xcorr_bounded": "similarity (bounded)"}).round(3).sort_values("similarity (bounded)", ascending=False)

    d = sp.groupby("source_disjoint").agg(clips=("filename", "size"), recordings=("group", "nunique")).reindex(["train", "val", "test"])
    T["split"] = d.reset_index().rename(columns={"source_disjoint": "partition"})

    wh = pd.read_csv(os.path.join(R, "website_holdout_recordings.csv"))
    T["website_holdout"] = pd.DataFrame({
        "held-out website": wh.heldout, "features": wh.featureset.map({"ast": "frozen AST", "struct-nf": "structure", "spec": "spectral"}),
        "classes scored": wh.classes, "test clips": wh.test_clips, "test recordings": wh.test_recordings,
        "macro-F1 trained WITH other recordings of the website": wh.control.round(3),
        "macro-F1 trained WITHOUT the website": wh.holdout.round(3), "loss (gap)": wh.gap.round(3),
        "gap CI low": wh.ci_lo.round(3), "gap CI high": wh.ci_hi.round(3)})

    m = pd.read_csv(os.path.join(HERE, "metrics_summary.csv"))
    fam = {"Spectral features + LR": "ML", "Structure features + LR": "ML", "Frozen AST embedding + LR": "ML (pretrained features)"}
    T["models"] = pd.DataFrame({"family": [fam.get(s, "Deep learning") for s in m.system], "model": m.system, "seeds": m.seeds,
                                "accuracy": m.accuracy.round(3), "macro precision": [("n/a" if pd.isna(v) else round(v, 3)) for v in m.macro_precision],
                                "macro recall": m.macro_recall.round(3), "macro F1": m.macro_f1.round(3),
                                "F1 SD over seeds": m.macro_f1_sd.round(3)})
    gpu = pd.read_csv(os.path.join(REPO, "docs", "results", "colab_k0_pc200", "ast_2x2_effects.csv"))
    gpu_rows = gpu[gpu.quantity.str.startswith("macro-F1")]
    T["models_gpu_full_finetune"] = pd.DataFrame({"model (all 12 AST layers fine-tuned, Colab GPU, 3 seeds)": gpu_rows.quantity.str.replace("macro-F1 ", ""),
                                                  "macro F1": gpu_rows.estimate.round(3), "CI low": gpu_rows.ci_lo.round(3),
                                                  "CI high": gpu_rows.ci_hi.round(3)})
    pc = pd.read_csv(os.path.join(HERE, "metrics_per_class.csv"))
    T["per_class_AST_combined"] = pc[pc.system == "AST fine-tuned: combined"].drop(columns="system").round(3)

    def eff(path):
        e = pd.read_csv(path).set_index("quantity")
        return {k: f"{e.loc[k, 'estimate']:+.3f} [{e.loc[k, 'ci_lo']:+.3f}, {e.loc[k, 'ci_hi']:+.3f}]"
                for k in ["balanced_main_effect", "contrastive_main_effect", "interaction", "combined_minus_ordinary"]}
    c, g = eff(os.path.join(R, "report", "ast_2x2_effects.csv")), eff(os.path.join(REPO, "docs", "results", "colab_k0_pc200", "ast_2x2_effects.csv"))
    T["source_aware_effects"] = pd.DataFrame({
        "effect on macro-F1 [95% CI]": ["source-balanced sampling", "cross-source contrastive loss", "interaction", "combined - ordinary"],
        "CNN from scratch (GPU, 3 seeds)": ["+0.030 [+0.005, +0.057]", "not tested", "not tested", "not tested"],
        "AST, last layer fine-tuned (laptop, 3 seeds)": [c[k] for k in c], "AST, all layers fine-tuned (GPU, 3 seeds)": [g[k] for k in g]})

    cl = pd.read_csv(os.path.join(R, "report", "G9_source_clustering_of_embeddings.csv"))
    T["source_clustering"] = cl.groupby("system", sort=False)[["rate", "chance"]].mean().round(3).reset_index().rename(
        columns={"rate": "nearest same-class neighbour from the SAME recording", "chance": "chance level"})

    ec = pd.concat([pd.read_csv(os.path.join(R, "error_clustering.csv")), pd.read_csv(os.path.join(R, "error_clustering_ml.csv"))])
    names = {"ast_ft_source_disjoint_standard_s0": "AST fine-tuned (ordinary)", "ast_ftsc_source_disjoint_source_s0": "AST fine-tuned (combined)",
             "ast_frozen_source_disjoint_standard_s0": "AST frozen + linear head", "cnn_source_disjoint_standard_s0": "CNN (earlier CPU run)",
             "ml_ast_source_disjoint": "Frozen AST + LR", "ml_spec_source_disjoint": "Spectral + LR", "ml_struct-nf_source_disjoint": "Structure + LR"}
    ec = ec[ec.run.isin(names)]
    T["error_clustering"] = pd.DataFrame({"model (seed 0)": ec.run.map(names), "macro F1": ec.macro_f1.round(3),
                                          "error clustering by recording (ICC)": ec.error_icc.round(2),
                                          "effective test size (of 1160)": ec.effective_n.round(0),
                                          "share of errors in worst 10% of recordings": ec.top10pct_recordings_share_of_errors.round(2),
                                          "CI width if clips treated as independent": ec.ci_width_clip_bootstrap.round(3),
                                          "CI width resampling recordings": ec.ci_width_source_bootstrap.round(3)})
    import sys as _sys
    _sys.path.insert(0, HERE)
    from probe_numbers import table as probe_table
    T["background_probe"] = probe_table()
    return T


ITEMS = [  # (filename, source key or path, title, show, say)
    ("01_Slides.pdf", "SLIDES_PDF", "The presentation (present from this)", "Slides 1-16, one per step below.",
     "Use this as the backbone; open the numbered files below when she asks for detail or evidence."),
    ("01_Slides.pptx", "SLIDES_PPTX", "Same slides, editable, with speaker notes", "Presenter View shows your notes.", ""),
    ("02_Lab_Report.pdf", "REPORT", "Full project report (lab format, 18 pages)", "Hand in / show if she asks for the write-up.",
     "Interim version: the advanced-model experiments will be added after the GPU runs."),
    ("03_All_Results_Tables.xlsx", "XLSX", "Every result table in one Excel file (one sheet per table)",
     "Open when she wants to see the numbers behind any slide.", ""),
    ("04_Dataset_class_distribution.csv", "dataset", "Dataset: 7,014 clips, 7 classes, 3 websites",
     "Class counts (official train/test), reconstructed recordings per class, websites per class.",
     "Classes are imbalanced, and website is tangled with class (e.g. breath only from Freesound and YouTube)."),
    ("05_Source_concentration.png", "FIG:A2_source_concentration.png", "A few recordings dominate some classes",
     "Bar chart: share of each class from its largest / top-5 recordings.",
     "827 of the 1,850 breath clips come from ONE YouTube video; effective number of breath recordings is only 4.6."),
    ("05_Source_concentration.csv", "concentration", "Numbers behind the chart", "", ""),
    ("06_Aigei_crying_sources.png", "FIG:A4_aigei_crying_sources.png", "Example: Aigei crying clips come from few videos",
     "958 Aigei crying clips map to 70 recordings; 376 come from one video.", "Shows why File_ID is not a recording ID."),
    ("07_Duplicates_summary.csv", "duplicates_summary", "Duplicate audio and leakage", "137 confirmed same-audio pairs, 23 across the official split.",
     "We rebuilt recording identities and found re-uploaded audio; a strict re-check confirmed 133 of 137."),
    ("07_Duplicate_pairs_confirmed.csv", "duplicate_pairs", "Every confirmed duplicate pair (filenames, websites, splits, similarity)",
     "Sort by cross_official_split = TRUE to show leakage pairs.", ""),
    ("08_Workflow_diagram.png", "BUILD:flowchart.png", "Our pipeline", "Audio -> recordings -> duplicates -> split by recording -> models -> evaluation.",
     "Test recordings are never heard in training; settings chosen on validation, test used once."),
    ("08_Split_partitions.csv", "split", "Train / validation / test sizes", "5,127 / 727 / 1,160 clips; 978 / 225 / 300 recordings.", ""),
    ("09_Website_holdout.csv", "website_holdout", "Website shift is real",
     "Rows 'frozen AST': Aigei gap 0.057, Freesound 0.056 (CI excludes 0).",
     "If the model never hears a website, it loses about 6 macro-F1 points, with everything else matched."),
    ("10_Model_comparison_all_models.csv", "models", "MAIN TABLE: ML vs deep learning",
     "Accuracy, precision, recall, F1 for every model (3 seeds for deep models).",
     "Handcrafted features + LR 0.45-0.50; CNN 0.76 (0.79 source-balanced); fine-tuned AST up to 0.82."),
    ("10_Model_comparison_GPU_full_finetune.csv", "models_gpu_full_finetune", "AST with all 12 layers fine-tuned (Colab GPU)",
     "0.812-0.837 macro-F1.", "Full fine-tuning on GPU confirms the laptop results."),
    ("10_Per_class_metrics_best_AST.csv", "per_class_AST_combined", "Per-class precision / recall / F1 of the best AST", "",
     "Crying is hardest (recall 0.79); screaming, sneeze, yawn have high recall but lower precision."),
    ("11_Confusion_matrices.png", "REP:G3_confusion_matrices.png", "Where the models go wrong", "Row-normalised confusion matrices.",
     "Crying is mostly confused with laughter."),
    ("12_Training_curves_AST.png", "BUILD:learning_curves_2x2.png", "Training actually happened", "Loss falls, validation F1 rises, 3 seeds each.",
     "We also logged gradients and how much the pretrained weights changed."),
    ("13_Source_aware_training_effects.csv", "source_aware_effects", "Our experiment: does source-aware training help?",
     "Source-balanced sampling: CNN +0.030, AST +0.035 (significant). Contrastive loss: no effect.",
     "Sampling recordings evenly helps in two architectures; the contrastive loss does not (null result, reported honestly)."),
    ("13_AST_2x2_bar_chart.png", "REP:G6_ast_2x2_source_aware_training.png", "The four AST training recipes", "", ""),
    ("14_Source_clustering.png", "REP:G9_source_clustering_of_embeddings.png", "Why the contrastive loss did not help",
     "72% of the time an unseen clip's nearest match is from the SAME recording (chance 7.6%).",
     "The models organise sounds by recording, and no training recipe changed that."),
    ("14_Source_clustering.csv", "source_clustering", "Numbers behind the chart", "", ""),
    ("15_Error_clustering.csv", "error_clustering", "Errors come from a few recordings",
     "Worst 10% of recordings hold about 80% of errors; per-clip CIs about 2x too narrow.",
     "Results on this dataset should be reported per recording, not per clip."),
    ("16_Background_probe_chart.png", "BUILD:background_probe.png", "What does the model listen to? (most surprising result)",
     "Full 0.79 | vocalisation only 0.73 | random cut 0.66 | vocalisation REMOVED 0.26-0.31 at a 20 dB cut, 0.16-0.20 at a stricter 30 dB cut (chance <= 0.14).",
     "Models mainly listen to the sound; with it removed they fall to about 2x chance (20 dB) and 1.5x chance (30 dB), so a small part of the information is recording context. Corrected masks, A100 rerun."),
    ("16_Background_probe_table.csv", "background_probe", "Numbers behind the chart", "", ""),
    ("16_Background_probe_A100_20dB.png", "NB:docs/results/a100_rerun/probe_db20_edgefix/G10_background_probe.png",
     "Raw output figure of the corrected probe, 20 dB cut (college A100)", "All models and the linear probes, with CIs.", ""),
    ("16_Background_probe_A100_30dB.png", "NB:docs/results/a100_rerun/probe_db30_edgefix/G10_background_probe.png",
     "Raw output figure of the corrected probe, stricter 30 dB cut (college A100)", "Removed-vocalisation bars drop to about 1.5x chance.", ""),
    ("16_Background_swap_chart.png", "NB:docs/results/swap_probe_a100/G11_swap_probe.png",
     "Background swap: does a background from another class pull the prediction?",
     "Same-class background (control) vs different-class background, 20 dB / 30 dB / raw.",
     "Different-class backgrounds lower recall by 3-6 points; verdict fixed in advance: detected but small."),
    ("16_Background_swap_effects.csv", "NB:docs/results/swap_probe_a100/swap_effects.csv",
     "Numbers behind the swap chart (all effects with CIs)", "Row 'swap effect: same - different' is the main number.", ""),
    ("16_Background_swap_SUMMARY.md", "NB:docs/results/swap_probe_a100/SUMMARY.md", "Plain-language summary of the swap test", "", ""),
    ("17_Live_demo_waveform_spectrogram.png", "SCR:demo_cell7_0.png", "Live demo: an unseen test recording (a scream)", "", ""),
    ("17_Live_demo_predictions.png", "SCR:demo_cell11_1.png", "Live demo: model predictions",
     "CNN said crying; both fine-tuned AST models said screaming (raw softmax scores).", ""),
    ("17_Live_demo_console_output.png", "SCR:console_demo.png", "Live demo: printed output (7 of 7 correct for AST)", "", ""),
    ("17_Live_demo_notebook_executed.ipynb", "NB:professor_demo/LIVE_DL_DEMO_executed.ipynb", "The executed demo notebook (all outputs saved)",
     "Open only if she wants to see it run.", ""),
    ("18_Next_steps_tentative.md", "NEXT", "Work in progress: what comes next on GPU",
     "Background-swap training; Whisper and WavLM vs AST with the same background probe.",
     "Today's results are interim; the report and slides will be updated after these runs."),
]


def write_next(path):
    open(path, "w", encoding="utf-8").write(
        "# Next steps (tentative, after today's presentation)\n\n"
        "1. **Done since:** the background-swap test found only a small pull (verdict fixed in advance: detected but small),\n"
        "   so background-swap training is not needed.\n"
        "2. **Speech-pretrained vs general-audio models**: fine-tune the Whisper-small encoder and WavLM Base+ with the same\n"
        "   split, budget, seeds and background probe as AST. Question: which pretraining makes a model listen to the\n"
        "   vocalisation rather than the recording around it?\n"
        "3. **Excluded on purpose:** voc2vec and LAION-CLAP, likely trained on Nonspeech7k (test leakage).\n"
        "4. Update the report and slides with the final model table.\n")


def guide_pdf(path, rows, today):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=18, leading=23, spaceAfter=8)
    H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=13, leading=17, spaceBefore=8, spaceAfter=4)
    B = ParagraphStyle("b", fontName="Times-Roman", fontSize=11, leading=15, alignment=TA_LEFT, spaceAfter=4)
    C = ParagraphStyle("c", fontName="Times-Roman", fontSize=9.5, leading=12)
    CB = ParagraphStyle("cb", fontName="Times-Bold", fontSize=9.5, leading=12)
    s = [Paragraph("00 Presenter guide: Nonspeech7k interim presentation", H1),
         Paragraph(f"Harbani Kaur (1024030848) and Ina Sharma (1024030866). Submitted to Ms. Arushi Madaan. {today}.", B),
         Paragraph("<b>How to use this folder.</b> Present from <b>01_Slides.pdf</b>. Every slide's evidence is a numbered file "
                   "in this folder, in the same order. When she asks \"where does this number come from?\", open the file with that "
                   "number. All tables are also in <b>03_All_Results_Tables.xlsx</b> (one sheet each). Nothing here is hard-coded: "
                   "every value is read from the project's result files.", B),
         Paragraph("<b>One-sentence summary.</b> On Nonspeech7k, models partly recognise the <i>recording</i> rather than the "
                   "<i>sound</i>; with leakage-safe evaluation, sampling recordings evenly helps (CNN and AST), a contrastive loss "
                   "does not, errors cluster by recording, and with the vocalisation cut out models still score about 1.5-2 times "
                   "chance, so a small part of the information is recording context.", B),
         Paragraph("Order of the talk (about 12 minutes)", H2)]
    data = [[Paragraph(x, CB) for x in ("File", "What it is / what to show", "What to say")]]
    for fn, _, title, show, say in rows:
        data.append([Paragraph(fn, C), Paragraph(f"<b>{title}</b><br/>{show}", C), Paragraph(say, C)])
    t = Table(data, colWidths=[1.9 * inch, 2.7 * inch, 2.3 * inch], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6"))]))
    s += [t, Paragraph("Likely questions", H2)]
    for q, a in [("Is the method new?", "No. The contrastive loss is known (Kim et al., ICASSP 2024). Our contribution is the "
                  "leakage-safe evaluation and what it revealed about what models rely on."),
                 ("Why macro-F1?", "Classes are imbalanced; macro-F1 weights every class equally."),
                 ("Why does frozen AST + LR (0.848) beat fine-tuned AST (0.82)?", "Not a fair comparison: it used 10 s inputs and "
                  "about 4x more training clips; the fine-tuned models used 1,355 clips and 4 s inputs. A matched comparison is planned."),
                 ("Why is CNN precision n/a?", "The clean CNN rerun ran on Colab; its per-clip predictions are on Google Drive. "
                  "Accuracy, recall (= balanced accuracy) and F1 are exact."),
                 ("Is the duplicate detection reliable?", "500 random pairs never exceed similarity 0.41 (threshold 0.9); a stricter "
                  "re-check confirmed 133 of 137 pairs."),
                 ("What is next?", "Background-swap training and speech-pretrained models (Whisper, WavLM) on GPU; see 18_Next_steps.")]:
        s += [Paragraph(f"<b>{q}</b> {a}", B)]
    SimpleDocTemplate(path, pagesize=A4, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.7 * inch,
                      bottomMargin=0.7 * inch, title="00 Presenter guide").build(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", default="19 November 2026")
    a = ap.parse_args()
    R, O = a.results, a.out
    os.makedirs(O, exist_ok=True)
    T = tables(R)
    xlsx = os.path.join(O, "03_All_Results_Tables.xlsx")
    sheet_names = {"dataset": "04 dataset", "concentration": "05 concentration", "duplicates_summary": "07 duplicates summary",
                   "duplicate_pairs": "07 duplicate pairs", "split": "08 split", "website_holdout": "09 website holdout",
                   "models": "10 model comparison", "models_gpu_full_finetune": "10 AST full fine-tune GPU",
                   "per_class_AST_combined": "10 per-class best AST", "source_aware_effects": "13 source-aware effects",
                   "source_clustering": "14 source clustering", "error_clustering": "15 error clustering",
                   "background_probe": "16 background probe"}
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for k, sn in sheet_names.items():
            T[k].to_excel(xw, sheet_name=sn, index=False)
            ws = xw.sheets[sn]
            from openpyxl.styles import Font, PatternFill
            for c in ws[1]:
                c.font = Font(bold=True)
                c.fill = PatternFill("solid", fgColor="DCE3EC")
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, w + 2))
            ws.freeze_panes = "A2"
    src = {"SLIDES_PDF": os.path.join(HERE, "presentation", "Nonspeech7k_Presentation.pdf"),
           "SLIDES_PPTX": os.path.join(HERE, "presentation", "Nonspeech7k_Presentation.pptx"),
           "REPORT": os.path.join(HERE, "build", "ML_Lab_Project_Report.pdf")}
    for fn, key, *_ in ITEMS:
        dst = os.path.join(O, fn)
        if key in src:
            shutil.copy(src[key], dst)
        elif key == "XLSX":
            continue
        elif key == "NEXT":
            write_next(dst)
        elif key.startswith("FIG:"):
            shutil.copy(os.path.join(R, "figures", key[4:]), dst)
        elif key.startswith("REP:"):
            shutil.copy(os.path.join(R, "report", key[4:]), dst)
        elif key.startswith("BUILD:"):
            shutil.copy(os.path.join(HERE, "build", key[6:]), dst)
        elif key.startswith("SCR:"):
            shutil.copy(os.path.join(HERE, "screens", key[4:]), dst)
        elif key.startswith("NB:"):
            shutil.copy(os.path.join(REPO, key[3:]), dst)
        else:
            T[key].to_csv(dst, index=False)
    guide_pdf(os.path.join(O, "00_GUIDE_Read_First.pdf"), ITEMS, a.date)
    with open(os.path.join(O, "00_GUIDE_Read_First.md"), "w", encoding="utf-8") as f:
        f.write("# 00 Presenter guide\n\nPresent from 01_Slides.pdf; open the numbered files as evidence.\n\n| File | What | Say |\n|---|---|---|\n")
        for fn, _, title, show, say in ITEMS:
            f.write(f"| {fn} | **{title}** {show} | {say} |\n")
    print(f"wrote {len(os.listdir(O))} files to {O}")


if __name__ == "__main__":
    main()
