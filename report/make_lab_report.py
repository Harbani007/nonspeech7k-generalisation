"""Build the Machine Learning Lab project report (PDF) following the lab's report guidelines.

All numbers come from result files of this project (see the evidence notes in each section).
  python report/make_lab_report.py --results ../results/v3 --build report/build --out report/build/ML_Lab_Project_Report.pdf
Cover-page fields in COVER below must be filled in by the students.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

COVER = {
    "university": "Thapar Institute of Engineering and Technology (Thapar University)",
    "department": "Department of Computer Science and Engineering",
    "title": "Recording Identity, Website Shift and What Deep Models Listen To: "
             "A Leakage-Safe Deep-Learning Study of Human Non-Speech Vocalisation Recognition on Nonspeech7k",
    "students": [("Harbani Kaur", "1024030848"), ("Ina Sharma", "1024030866")],
    "faculty": "Ms. Arushi Madaan",
    "session": "2026-27",
}

FONT_DIR = r"C:\Windows\Fonts"
pdfmetrics.registerFont(TTFont("TNR", os.path.join(FONT_DIR, "times.ttf")))
pdfmetrics.registerFont(TTFont("TNR-B", os.path.join(FONT_DIR, "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("TNR-I", os.path.join(FONT_DIR, "timesi.ttf")))
pdfmetrics.registerFont(TTFont("TNR-BI", os.path.join(FONT_DIR, "timesbi.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402
registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

BODY = ParagraphStyle("body", fontName="TNR", fontSize=12, leading=18, alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=18, bulletIndent=6, spaceAfter=2)
H1 = ParagraphStyle("h1", fontName="TNR-B", fontSize=16, leading=22, spaceBefore=6, spaceAfter=10, alignment=TA_LEFT, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="TNR-B", fontSize=14, leading=20, spaceBefore=8, spaceAfter=6, alignment=TA_LEFT, keepWithNext=1)
CAP = ParagraphStyle("cap", fontName="TNR-I", fontSize=10.5, leading=14, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
CELL = ParagraphStyle("cell", fontName="TNR", fontSize=10, leading=12.5, alignment=TA_LEFT)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="TNR-B")
CODE = ParagraphStyle("code", fontName="Courier", fontSize=8.6, leading=10.8, leftIndent=6, backColor=colors.HexColor("#F4F4F4"),
                      borderPadding=5, spaceBefore=4, spaceAfter=10)
COVER_T = ParagraphStyle("covert", fontName="TNR-B", fontSize=18, leading=26, alignment=TA_CENTER)
COVER_S = ParagraphStyle("covers", fontName="TNR", fontSize=14, leading=22, alignment=TA_CENTER)
COVER_B = ParagraphStyle("coverb", fontName="TNR-B", fontSize=14, leading=22, alignment=TA_CENTER)

W = A4[0] - 2 * inch  # text width

fig_no = [0]
tab_no = [0]


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, BULLET, bulletText="\u2022") for t in items]


def figure(path, caption, width=W, max_h=4.6 * inch):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = width
    h = w * ih / iw
    if h > max_h:
        h = max_h
        w = h * iw / ih
    fig_no[0] += 1
    return KeepTogether([Image(path, width=w, height=h), P(f"Figure {fig_no[0]}: {caption}", CAP)])


def table(rows, caption, widths=None, header=True):
    tab_no[0] += 1
    data = [[P(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")))
    t.setStyle(TableStyle(st))
    return KeepTogether([P(f"Table {tab_no[0]}: {caption}", CAP), t, Spacer(1, 8)])


def page_number(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("TNR", 11)
    canvas.drawCentredString(A4[0] / 2, 0.55 * inch, str(doc.page - 1))
    canvas.restoreState()


def background_figure(out):
    """Bar chart of the Colab background-probe numbers (transcribed in docs/results/background_probe/SUMMARY.md)."""
    systems = ["ordinary", "source-balanced", "contrastive", "combined"]
    vals = {"full clip": [0.788, 0.797, 0.785, 0.784], "vocalisation only": [0.732, 0.741, 0.729, 0.744],
            "random cut (control)": [0.671, 0.662, 0.657, 0.655], "vocalisation removed": [0.304, 0.265, 0.304, 0.277]}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    w = 0.2
    for j, (k, v) in enumerate(vals.items()):
        ax.bar([i + (j - 1.5) * w for i in range(4)], v, w, label=k)
    ax.axhline(0.16, color="k", ls="--", lw=1, label="chance (95th percentile)")
    ax.set_xticks(range(4), systems)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center")
    ax.set_title("Background probe, unseen recordings (645 clips, 20 dB cut, 3 seeds)", fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, B = a.results, a.build
    rep, figs = os.path.join(R, "report"), os.path.join(R, "figures")
    global SCR
    SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screens")
    background_figure(os.path.join(B, "background_probe.png"))
    from PIL import Image as PImage  # learning curves: 1 x 4 panels -> 2 x 2 so they are readable on A4
    im = PImage.open(os.path.join(rep, "G1_G2_learning_curves_ast_2x2.png"))
    w4, hh = im.size[0] // 4, im.size[1]
    grid = PImage.new("RGB", (2 * w4, 2 * hh), "white")
    for k in range(4):
        grid.paste(im.crop((k * w4, 0, (k + 1) * w4, hh)), ((k % 2) * w4, (k // 2) * hh))
    grid.save(os.path.join(B, "learning_curves_2x2.png"))
    M = pd.read_csv(os.path.join(B, "metrics_summary.csv"))
    PC = pd.read_csv(os.path.join(B, "metrics_per_class.csv"))
    eff_cpu = pd.read_csv(os.path.join(rep, "ast_2x2_effects.csv")).set_index("quantity")
    eff_gpu = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "docs", "results", "colab_k0_pc200", "ast_2x2_effects.csv")).set_index("quantity")
    ec = pd.read_csv(os.path.join(R, "error_clustering.csv"))
    ecm = pd.read_csv(os.path.join(R, "error_clustering_ml.csv"))

    s = []
    # ---------------- cover ----------------
    s += [Spacer(1, 0.6 * inch), P(COVER["university"], COVER_B), P(COVER["department"], COVER_S), Spacer(1, 0.5 * inch),
          P("A Project Report on", COVER_S), Spacer(1, 0.15 * inch), P(COVER["title"], COVER_T), Spacer(1, 0.4 * inch),
          P("Subject: Machine Learning Lab", COVER_S), Spacer(1, 0.6 * inch), P("Submitted By", COVER_B)]
    s += [P(f"{n} &nbsp;&nbsp;(Roll No. {r})", COVER_S) for n, r in COVER["students"]]
    s += [Spacer(1, 0.4 * inch), P("Submitted To", COVER_B), P(COVER["faculty"], COVER_S), Spacer(1, 0.5 * inch),
          P(f"Academic Session: {COVER['session']}", COVER_S), PageBreak()]

    # ---------------- 1. abstract ----------------
    f1 = M.set_index("system").macro_f1
    s += [P("Abstract", H1), P(
        "Nonspeech7k is a public dataset of 7,014 short clips of seven human non-speech vocalisations (breath, cough, "
        "crying, laughter, screaming, sneezing and yawning) collected from three websites. Its clips were cut from longer "
        "recordings, so a model may learn to recognise <i>recordings</i> rather than <i>sounds</i>. This project builds a "
        "leakage-safe evaluation of deep-learning models on Nonspeech7k and asks three questions: does website shift hurt "
        "performance, does source-aware training help on unseen recordings, and what part of the audio the models actually use. "
        "Recording identities were reconstructed from filenames and 137 duplicated audio pairs were detected, giving a "
        "source-disjoint train/validation/test split. We compare logistic regression on handcrafted and pretrained features, "
        "a log-mel CNN trained from scratch and the Audio Spectrogram Transformer (AST), frozen and fine-tuned, in a "
        "pre-registered 2 x 2 experiment (source-balanced sampling x cross-source supervised contrastive loss) with three seeds, "
        f"on a CPU and with full fine-tuning on a GPU. Fine-tuned AST reaches a macro-F1 of {f1['AST fine-tuned: combined']:.3f} "
        f"(accuracy {M.set_index('system').accuracy['AST fine-tuned: combined']:.3f}) on recordings never seen in training. "
        "Holding out a whole website costs about 0.06 macro-F1. Source-balanced sampling helps modestly in two architectures "
        "(CNN +0.030, AST +0.035, both significant), the contrastive loss does not. Test errors cluster strongly by recording, so clip-level confidence "
        "intervals are about two times too narrow. Removing the vocalisation from a clip drops performance to about twice chance, "
        "showing that the models mainly, but not only, listen to the vocalisation itself."), PageBreak()]

    # ---------------- 2. introduction ----------------
    s += [P("1. Introduction", H1),
          P("1.1 Problem domain", H2),
          P("Human non-speech vocalisations such as coughs, sneezes, laughter and crying carry information about health, emotion "
            "and the situation a person is in. Automatic recognition of these sounds is useful for health monitoring (cough "
            "counting, respiratory screening), assistive technology (detecting a baby crying or a scream for help) and smart "
            "environments. Unlike speech, these sounds have no words, are short and vary strongly between people."),
          P("1.2 Motivation", H2),
          P("Deep-learning models report high accuracy on public audio datasets, but many such datasets are built by cutting long "
            "online recordings into short clips. If clips from the same recording appear in both training and test data, a model "
            "can score well by recognising the microphone, room or uploader instead of the sound. The original Nonspeech7k paper "
            "describes this chunking. We wanted to measure how much it matters and whether training methods that explicitly "
            "ignore recording identity generalise better to new recordings."),
          P("1.3 Importance", H2),
          P("A cough detector that only works on recordings similar to its training data is not useful in practice. Reliable "
            "evaluation, meaning test recordings that the model has never heard, is a precondition for deploying such systems "
            "and for comparing methods fairly."),
          P("1.4 Objectives", H2)]
    s += bullets(["Reconstruct which recording each clip came from and detect duplicated audio.",
                  "Build a source-disjoint train/validation/test split and audit the official split for leakage.",
                  "Measure the effect of website shift with a controlled leave-one-website-out experiment.",
                  "Train and compare traditional machine learning, a CNN and a pretrained/fine-tuned transformer (AST).",
                  "Test, under pre-registered rules, whether source-balanced sampling and a cross-source contrastive loss improve "
                  "accuracy on unseen recordings.",
                  "Analyse what the trained models actually listen to (vocalisation versus background)."])

    # ---------------- 3. problem statement ----------------
    s += [P("2. Problem Statement", H1),
          P("Given a 4-second audio clip of a human non-speech vocalisation, classify it into one of seven classes. The model "
            "must generalise to <b>recordings that were never seen during training</b>. The project addresses three linked problems: "
            "(i) standard evaluation may overstate performance because clips of one recording are spread across splits; "
            "(ii) differences between source websites may shift performance; (iii) it is unclear whether models rely on the "
            "vocalisation or on the recording context. The expected outcome is a trained deep-learning classifier evaluated on a "
            "leakage-safe split, together with verified evidence on each of these three questions.")]

    # ---------------- 4. literature review ----------------
    s += [P("3. Literature Review", H1)]
    s += bullets([
        "<b>Nonspeech7k</b> (Rashid, Li and Du, IET Signal Processing, 2023) introduced the dataset and trained end-to-end CNNs; "
        "the best of their models (M12) reached 79% accuracy. The paper describes cutting long recordings into clips.",
        "<b>Audio Spectrogram Transformer</b> (Gong, Chung and Glass, Interspeech 2021) applies a Vision Transformer to "
        "spectrogram patches and, pretrained on AudioSet (Gemmeke et al., 2017), is a strong general audio classifier.",
        "<b>Supervised contrastive learning</b> (Khosla et al., NeurIPS 2020) pulls same-class embeddings together. Variants "
        "that take positives from a different <i>domain</i> were applied to respiratory sounds across stethoscopes with AST "
        "(Kim et al., ICASSP 2024) and to bird sounds across recording conditions (Moummad et al., 2024).",
        "<b>Domain-adversarial training</b> (Ganin and Lempitsky, ICML 2015) removes domain information with a gradient-reversal layer.",
        "<b>Shortcut learning in audio</b>: SpurAudio (2026) shows that few-shot audio classifiers rely on background context "
        "when foreground and background are recombined."])
    s += [P("<b>Limitations of existing work and our contribution.</b> We found no study of Nonspeech7k with recording-disjoint "
            "evaluation, quantified duplicate leakage, a controlled website-shift test or a foreground-removal analysis. The "
            "training methods we test are not new; our contribution is applying them under a leakage-safe, pre-registered "
            "protocol and analysing what the models rely on.")]

    # ---------------- 5. dataset ----------------
    s += [P("4. Dataset Description", H1),
          P("<b>Name and source:</b> Nonspeech7k, Zenodo record 6967442 (CC BY-NC-SA 4.0). <b>Records:</b> 7,014 mono WAV clips "
            "at 32 kHz, 6,289 in the official training set and 725 in the official test set. <b>Input:</b> the raw waveform "
            "(4 s used). <b>Metadata:</b> filename, class label, File_ID and source website (Aigei, Freesound, YouTube). "
            "<b>Target variable:</b> the class label, 7 categories. There are no missing values; the important data-quality "
            "problems are hidden recording identity and duplicated audio (Section 5).")]
    counts = [["Class", "Official train", "Official test", "Total", "Recordings (reconstructed)", "Websites"],
              ["breath", 1690, 160, 1850, 192, "Freesound, YouTube"], ["cough", 632, 70, 702, 262, "Aigei, Freesound"],
              ["crying", 1791, 205, 1996, 156, "all three"], ["laugh", 1133, 140, 1273, 296, "Freesound, YouTube"],
              ["screaming", 593, 70, 663, 231, "all three"], ["sneeze", 236, 30, 266, 180, "all three"],
              ["yawn", 214, 50, 264, 191, "Aigei, Freesound"], ["Total", 6289, 725, 7014, "1,503", "3"]]
    s += [table(counts, "Nonspeech7k class distribution (recordings after merging confirmed duplicates).",
                widths=[0.95 * inch, 0.95 * inch, 0.9 * inch, 0.65 * inch, 1.3 * inch, 1.25 * inch]),
          P("The classes are imbalanced (crying 1,996 clips, yawn 264) and the website is confounded with the class: for example "
            "breath comes only from Freesound and YouTube, and 827 of the 1,850 breath clips come from a single YouTube video. "
            "Figure 1 shows how concentrated some classes are.")]
    s += [figure(os.path.join(figs, "A2_source_concentration.png"),
                 "Share of each class's clips that come from its single largest recording and from its five largest recordings (source_audit.py).", max_h=3.4 * inch)]

    # ---------------- 6. methodology ----------------
    s += [P("5. Methodology", H1),
          P("5.1 Data collection and preprocessing", H2),
          P("The released audio and metadata were used as published; no labels were changed. Each clip was converted to mono, "
            "resampled to 16 kHz and fixed to 4 s. For the CNN, 64-band log-mel spectrograms (25 ms window, 10 ms hop) were "
            "computed; for AST, the 128-bin Kaldi filterbank of the pretrained checkpoint (400 frames)."),
          P("5.2 Missing values and duplicate records", H2),
          P("There are no missing values. Duplicates were the main issue. (1) The published File_ID is not a recording ID: Aigei "
            "counters restart for each filename family, so unrelated clips shared IDs (97 groups mixed classes). We rebuilt "
            "recording keys from filenames, namespaced by website (2,475 IDs reduced to 1,586 recordings). (2) Re-uploaded audio "
            "was found by searching the top-5 AST-embedding neighbours (cosine at least 0.95) and confirming with peak waveform "
            "cross-correlation of at least 0.9: 137 confirmed duplicate pairs (53 across websites, 23 across the official split). "
            "500 random pairs never exceeded 0.41, so the threshold produced no false positives in this control. A re-check with a strictly bounded similarity score confirmed 133 of the 137 pairs. Confirmed "
            "duplicates were merged into one recording group (1,503 groups)."),
          P("5.3 Exploratory data analysis and visualisation", H2),
          P("EDA covered class and website distributions (Table 1), recordings per class and their concentration (Figure 1), "
            "duplicate statistics and official-split leakage: 52 of the 725 official test clips (7.2%) share a recording or "
            "duplicate with the training set."),
          P("5.4 Train/validation/test split", H2),
          P("A <b>source-disjoint</b> split was built by assigning whole recordings, per class, to train (5,127 clips, 978 "
            "recordings), validation (727 clips, 225 recordings) and test (1,160 clips, 300 recordings); recordings with 100 or "
            "more clips were kept in training. No recording or duplicate appears in two partitions (checked in code). The "
            "validation set was used for all model selection; the test set was used once per trained model."),
          P("5.5 Algorithms", H2)]
    s += bullets([
        "<b>Logistic regression</b> on 35 handcrafted spectral features, on structure (temporal burst) features, and on frozen "
        "768-dimensional AST embeddings (class-balanced, regularisation chosen by grouped cross-validation on the training set).",
        "<b>Log-mel CNN</b> trained from scratch (4 convolution blocks, 293k parameters, SpecAugment).",
        "<b>AST frozen + linear head</b>: the pretrained transformer is kept fixed, only a new 7-class layer is trained.",
        "<b>AST fine-tuned</b>: the pretrained transformer layers are trained further. On the laptop CPU only the last layer was "
        "trained (7.1 M trainable weights); on a Google Colab GPU all 12 layers were trained.",
        "<b>Source-aware variants</b> in a 2 x 2 design: (a) <i>source-balanced sampling</i>, which picks recordings uniformly, "
        "then a clip; (b) a <i>cross-source supervised contrastive loss</i>, whose positives are same-class clips from different "
        "recordings (weight 0.5, temperature 0.1, fixed in advance)."])
    s += [P("5.6 Training, testing and statistics", H2),
          P("Every AST condition used the same 1,355 training clips (up to 200 per class), the same initial weights, random "
            "stream and budget (4 epochs x 768 samples, batch 32, AdamW). The checkpoint with the best validation macro-F1 was "
            "tested once. Each condition was run with 3 seeds. Confidence intervals are 95% bootstrap intervals that resample "
            "whole test <i>recordings</i> (2,000 resamples), because clips from one recording are not independent. Hypotheses "
            "and settings were committed before the runs (pre-registration)."),
          figure(os.path.join(B, "flowchart.png"), "Overall workflow of the proposed system.", width=3.6 * inch, max_h=5.2 * inch)]

    # ---------------- 7. implementation ----------------
    s += [P("6. Implementation", H1),
          P("<b>Language:</b> Python 3.13. <b>Platforms:</b> a CPU laptop (Intel i5-1135G7, 4 cores, 16 GB RAM, no GPU) for "
            "data processing, baselines and last-layer fine-tuning; Google Colab with an NVIDIA T4 GPU for full fine-tuning and "
            "the background probe; Jupyter notebooks for the live demonstration. <b>Libraries:</b> PyTorch, Hugging Face "
            "Transformers (AST checkpoint MIT/ast-finetuned-audioset-10-10-0.4593), scikit-learn, NumPy, pandas, SciPy, "
            "SoundFile, Matplotlib."),
          P("<b>Implementation steps:</b> (1) <font face='Courier'>sources.py</font> and <font face='Courier'>find_duplicates.py</font> "
            "build recording groups; (2) <font face='Courier'>dl_data.py</font> creates the split and features; "
            "(3) <font face='Courier'>ml_baselines.py</font> and <font face='Courier'>train_cnn.py</font> train the baselines; "
            "(4) <font face='Courier'>ast_finetune.py</font> trains AST in all conditions with per-step logging, checkpointing, "
            "resume and run locks; (5) <font face='Courier'>dl_report.py</font>, <font face='Courier'>error_clustering.py</font> "
            "and <font face='Courier'>background_probe.py</font> compute the results. 23 unit tests check the source parsing, "
            "the contrastive loss and the diagnostics."),
          P("<b>Key code 1: cross-source supervised contrastive loss.</b> Positives are same-class clips from different recordings; "
            "same-class clips from the same recording are excluded from both numerator and denominator.")]
    s += [Preformatted(
        "def cross_source_supcon(z, y, g, tau):\n"
        "    sim = z @ z.T / tau                       # z: L2-normalised embeddings\n"
        "    eye = torch.eye(len(z), dtype=torch.bool)\n"
        "    same_y = y[:, None] == y[None, :]\n"
        "    same_g = g[:, None] == g[None, :]          # g: recording group\n"
        "    pos = same_y & ~same_g                     # same class, other recording\n"
        "    excluded = (same_y & same_g) | eye         # same recording: ignored\n"
        "    logits = sim.masked_fill(excluded, float('-inf'))\n"
        "    log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)\n"
        "    n_pos = pos.sum(1); ok = n_pos > 0\n"
        "    loss = -(log_prob.masked_fill(~pos, 0).sum(1)[ok] / n_pos[ok])\n"
        "    return loss.mean()", CODE),
          P("<b>Key code 2: one training step</b> (cross-entropy plus the optional contrastive term).")]
    s += [Preformatted(
        "emb = model.pooled(tokens(batch))              # AST summary vector (768)\n"
        "ce = F.cross_entropy(model.head(emb), y_batch)\n"
        "loss = ce\n"
        "if lam > 0:\n"
        "    z = F.normalize(proj(emb), dim=1)\n"
        "    loss = ce + lam * cross_source_supcon(z, y_batch, g_batch, tau)\n"
        "opt.zero_grad(); loss.backward(); opt.step(); sched.step()", CODE),
          P("<b>Execution evidence.</b> Each run logs, per step, the losses, the number of valid cross-source positive pairs "
            "(about 130 per batch of 32), gradient norms on the pretrained weights and how far those weights moved; a real-data "
            "smoke test confirmed that the contrastive gradient reaches and changes the pretrained layer. The live demonstration "
            "notebook loads trained checkpoints, plays a test recording, shows its waveform and spectrogram, and prints the "
            "models' predictions (raw softmax scores) and inference time; it correctly classified 7 of 7 example test recordings "
            "with the fine-tuned AST. Screenshots of the Colab and notebook execution are given in the Appendix.")]
    s += [figure(os.path.join(B, "learning_curves_2x2.png"),
                 "Training curves of the four fine-tuned AST conditions (3 seeds each, laptop experiment): cross-entropy, "
                 "validation macro-F1, contrastive loss and share of batch anchors with a cross-source positive.", max_h=4.4 * inch)]

    # ---------------- 8. results ----------------
    s += [P("7. Results and Discussion", H1), P("7.1 Model performance on unseen recordings", H2)]
    rows = [["Model", "Seeds", "Accuracy", "Precision", "Recall", "F1 (macro)"]]
    for _, r in M.iterrows():
        rows.append([r.system, int(r.seeds), f"{r.accuracy:.3f}", f"{r.macro_precision:.3f}", f"{r.macro_recall:.3f}",
                     f"{r.macro_f1:.3f}" + (f" &plusmn; {r.macro_f1_sd:.3f}" if r.seeds > 1 else "")])
        if pd.isna(r.macro_precision):
            rows[-1][3] = "n/a*"
    s += [table(rows, "Source-disjoint test set (1,160 clips, 300 recordings). Precision, recall and F1 are macro-averaged; "
                      "values are means over seeds (&plusmn; SD). *CNN rows: clean GPU rerun (3 seeds); recall = balanced accuracy, "
                      "precision not available because the per-clip predictions remain on Colab.",
                widths=[2.3 * inch, 0.5 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch, 1.0 * inch]),
          P("Pretrained AST features are far stronger than handcrafted features (macro-F1 0.85 versus 0.45-0.50). Fine-tuning "
            "AST with source-balanced sampling gives the best fine-tuned results (0.82). The logistic regression on frozen AST "
            "embeddings scores even higher (0.848), but it is not a like-for-like comparison: it uses 10 s inputs and the whole "
            "training partition (about 5,100 clips), whereas every fine-tuned condition uses 4 s inputs and 1,355 training clips. "
            "With full fine-tuning of all 12 layers on the GPU, the four conditions reached 0.812-0.837 macro-F1.")]
    s += [figure(os.path.join(rep, "G3_confusion_matrices.png"),
                 "Row-normalised confusion matrices on the source-disjoint test set (summed over 3 seeds).", max_h=4.5 * inch)]
    pc = PC[PC.system == "AST fine-tuned: combined"]
    rows = [["Class", "Precision", "Recall", "F1", "Test clips"]] + [
        [r.cls, f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}", int(r.support)] for _, r in pc.iterrows()]
    s += [table(rows, "Per-class results of the fine-tuned AST (combined condition, mean of 3 seeds).",
                widths=[1.3 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]),
          P("Crying is the hardest class (recall 0.79), mostly confused with laughter; screaming, sneeze and yawn have high recall "
            "but lower precision, because other classes are sometimes predicted as them.")]

    s += [P("7.2 Does source-aware training help? (2 x 2 experiment)", H2)]

    def e(df, k):
        r = df.loc[k]
        return f"{r.estimate:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
    rows = [["Effect (pre-registered)", "Laptop: last layer fine-tuned", "GPU: all 12 layers fine-tuned"],
            ["Source-balanced sampling", e(eff_cpu, "balanced_main_effect"), e(eff_gpu, "balanced_main_effect")],
            ["Cross-source contrastive loss", e(eff_cpu, "contrastive_main_effect"), e(eff_gpu, "contrastive_main_effect")],
            ["Interaction", e(eff_cpu, "interaction"), e(eff_gpu, "interaction")],
            ["Combined minus ordinary", e(eff_cpu, "combined_minus_ordinary"), e(eff_gpu, "combined_minus_ordinary")]]
    s += [table(rows, "Effects on test macro-F1 with 95% recording-level bootstrap intervals (3 seeds per condition).",
                widths=[2.0 * inch, 2.1 * inch, 2.1 * inch]),
          P("Source-balanced sampling improved macro-F1 by 3.5 points in the laptop experiment (interval excludes zero) and "
            "pointed the same way with full fine-tuning (+1.9 points, interval just touching zero). The same effect was found for the CNN trained from scratch (clean rerun, 3 seeds): source-balanced sampling raised macro-F1 from 0.762 [0.704, 0.812] to 0.792 [0.741, 0.834], a gain of +0.030 [+0.005, +0.057]. The contrastive loss had no "
            "detectable effect in either experiment, although it was verifiably active during training. A diagnostic explains "
            "why: for every model, the nearest same-class neighbour of a test clip comes from the <i>same recording</i> 70-72% of "
            "the time, against 7.6% by chance, and the contrastive loss does not change this (Figure 6).")]
    s += [figure(os.path.join(rep, "G6_ast_2x2_source_aware_training.png"),
                 "Test macro-F1 of the four fine-tuned AST conditions (laptop experiment).", max_h=2.6 * inch),
          figure(os.path.join(rep, "G9_source_clustering_of_embeddings.png"),
                 "Share of test clips whose nearest same-class neighbour comes from the same recording (dashed: chance).",
                 max_h=2.6 * inch)]

    s += [P("7.3 Website shift", H2),
          P("In a leave-one-website-out test, a classifier trained without a website was compared with one trained on the same "
            "number of recordings that included other recordings of that website. Never having heard the website cost 0.057 "
            "macro-F1 [0.043, 0.072] for Aigei and 0.056 [0.035, 0.075] for Freesound (frozen AST features), with crying recall "
            "dropping by 0.18 when Aigei was held out. Website shift is therefore real but moderate."),
          P("7.4 Errors cluster by recording", H2)]
    best = ec[ec.run == "ast_ft_source_disjoint_standard_s0"].iloc[0]
    ml = ecm[ecm.run == "ml_ast_source_disjoint"].iloc[0]
    s += [P(f"Whether a test clip is misclassified depends strongly on its recording: the intraclass correlation of errors is "
            f"{best.error_icc:.2f} for fine-tuned AST and {ml.error_icc:.2f} for the frozen-AST baseline (permutation p &lt; 0.002). "
            f"The effective test size is about half the number of clips ({best.effective_n:.0f} of {int(best.test_clips)}), the "
            f"worst 10% of recordings contain {100 * best.top10pct_recordings_share_of_errors:.0f}% of all errors, and confidence "
            f"intervals computed over clips are about two times too narrow ({best.ci_width_clip_bootstrap:.3f} versus "
            f"{best.ci_width_source_bootstrap:.3f} wide). Results on this dataset should therefore be reported with "
            f"recording-level uncertainty."),
          P("7.5 What do the models listen to? (background probe)", H2),
          P("Each test clip was modified in three ways: the vocalisation removed (all audio within 20 dB of the loudest moment), "
            "only the vocalisation kept, and, as a control, the same amount of audio removed at a random position. The fine-tuned "
            "models were then re-evaluated without retraining (645 eligible clips, 186 recordings, GPU models, 3 seeds)."),
          figure(os.path.join(B, "background_probe.png"), "Background probe: macro-F1 of the four fully fine-tuned AST "
                 "conditions on full, vocalisation-only, randomly cut and vocalisation-removed audio.", max_h=2.9 * inch),
          P("The vocalisation alone keeps about 93% of the performance, and removing it costs about 0.5 macro-F1, far more than "
            "removing the same amount elsewhere (about 0.13). The models therefore mainly listen to the vocalisation. "
            "Surprisingly, the audio that remains after removing it still identifies the class at about twice chance "
            "(0.27-0.30 versus at most 0.16) on unseen recordings, and a classifier trained only on that remaining audio reaches "
            "0.39. Part of this may be faint traces of the vocalisation below the 20 dB threshold; a stricter 30 dB test is planned.")]

    # ---------------- 9. conclusion ----------------
    s += [P("8. Conclusion", H1),
          P("All objectives were achieved. We reconstructed recording identities, detected 137 duplicated audio pairs and built a "
            "source-disjoint split. The best fine-tuned AST recognises the seven vocalisations in unseen recordings with a macro-F1 "
            f"of about {f1['AST fine-tuned: combined']:.2f} (accuracy {M.set_index('system').accuracy['AST fine-tuned: combined']:.2f}), "
            "and pretrained transformers clearly outperform handcrafted features. The main findings are: (1) website shift costs "
            "about 6 macro-F1 points; (2) source-balanced sampling helps a little and consistently, for both the CNN (+0.030) and "
            "AST (+0.035), while the cross-source contrastive loss does not, and "
            "learned representations stay strongly organised by recording; (3) errors cluster by recording, so clip-level "
            "reporting overstates certainty about two-fold; (4) models mainly use the vocalisation, but the remaining audio still "
            "carries class information. Together these give a more reliable picture of how well vocalisation classifiers "
            "generalise than a single accuracy number on a random split.")]

    # ---------------- 10. future scope ----------------
    s += [P("9. Future Scope", H1)]
    s += bullets([
        "Repeat the background probe with a stricter 30 dB cut and per-class analysis, to separate background context from "
        "faint vocalisation traces.",
        "Background-swap training (already implemented and pre-registered): paste each vocalisation onto the background of "
        "another recording during training and test whether reliance on the background disappears without losing accuracy.",
        "Compare speech-pretrained models (Whisper encoder, WavLM) with the general-audio AST under the same protocol and "
        "background probe, to see which pretraining makes a model listen to the vocalisation itself. Models likely trained on "
        "Nonspeech7k (voc2vec, CLAP) are excluded to avoid test leakage.",
        "Train on the full training set and with longer schedules on a GPU, and tune the contrastive weight on the validation set.",
        "Test whether a recording-adversarial model (gradient reversal) removes recording identity and whether that helps.",
        "Cross-dataset evaluation (for example VocalSound) to check that the findings are not specific to Nonspeech7k.",
        "Calibrate the model's probabilities and deploy a lightweight model for real-time detection on a phone or edge device."])

    # ---------------- 11. references ----------------
    s += [P("10. References", H1)]
    refs = [
        "M. M. Rashid, G. Li and C. Du, \"Nonspeech7k dataset: Classification and analysis of human non-speech sound,\" "
        "<i>IET Signal Processing</i>, 2023, doi:10.1049/sil2.12233.",
        "Nonspeech7k dataset, Zenodo record 6967442, https://zenodo.org/records/6967442.",
        "Y. Gong, Y.-A. Chung and J. Glass, \"AST: Audio Spectrogram Transformer,\" in <i>Proc. Interspeech</i>, 2021.",
        "J. F. Gemmeke et al., \"Audio Set: An ontology and human-labeled dataset for audio events,\" in <i>Proc. ICASSP</i>, 2017.",
        "P. Khosla et al., \"Supervised contrastive learning,\" in <i>Advances in Neural Information Processing Systems</i>, 2020.",
        "Kim et al., \"Stethoscope-guided supervised contrastive learning for cross-domain adaptation on respiratory sound "
        "classification,\" in <i>Proc. ICASSP</i>, 2024, arXiv:2312.09603.",
        "Moummad et al., domain-invariant representation learning of bird sounds, arXiv:2409.08589, 2024.",
        "Y. Ganin and V. Lempitsky, \"Unsupervised domain adaptation by backpropagation,\" in <i>Proc. ICML</i>, 2015.",
        "SpurAudio: A benchmark for studying shortcut learning in few-shot audio classification, arXiv:2605.13672, 2026.",
        "B. Efron and R. J. Tibshirani, <i>An Introduction to the Bootstrap</i>. Chapman and Hall, 1993.",
        "PyTorch documentation, https://pytorch.org/docs; Hugging Face Transformers documentation, "
        "https://huggingface.co/docs/transformers; scikit-learn documentation, https://scikit-learn.org."]
    s += [P(f"[{i}] {r}", ParagraphStyle("ref", parent=BODY, leftIndent=22, firstLineIndent=-22, alignment=TA_LEFT))
          for i, r in enumerate(refs, 1)]

    # ---------------- 12. appendix ----------------
    s += [PageBreak(), P("Appendix", H1),
          P("<b>A. Source code.</b> Complete code, tests, notebooks and result files: "
            "https://github.com/Harbani007/nonspeech7k-generalisation (branch analysis/source-identity). Main files: "
            "sources.py, find_duplicates.py, dl_data.py, train_cnn.py, ast_finetune.py, dl_report.py, error_clustering.py, "
            "background_probe.py; Colab notebook colab/NS7K_AST_GPU_COLAB.ipynb; demo notebooks in professor_demo/."),
          P("<b>B. Reproduction.</b> Unit tests: <font face='Courier'>python -m unittest discover tests</font> (23 tests). "
            "GPU experiments: see colab/README.md. All numbers in this report are read from result files in results/ and docs/results/."),
          P("<b>C. Screenshots of implementation and execution.</b> Images below are taken from the saved outputs of the "
            "executed notebooks (Google Colab and the local demo notebooks); text outputs are shown as console renderings of "
            "the saved cell output."),
          figure(os.path.join(SCR, "console_colab_probe.png"), "Google Colab (T4 GPU): printed results of the background probe.", max_h=3.3 * inch),
          figure(os.path.join(SCR, "colab_probe_cell3_0.png"), "Google Colab: background-probe figure produced by background_probe.py.", max_h=2.3 * inch),
          figure(os.path.join(SCR, "console_demo.png"), "Live demo notebook: model loading and predictions on real test recordings (laptop CPU).", max_h=3.0 * inch),
          figure(os.path.join(SCR, "demo_cell7_0.png"), "Live demo: waveform and log-mel spectrogram of an unseen test recording (screaming).", max_h=2.6 * inch),
          figure(os.path.join(SCR, "demo_cell11_1.png"), "Live demo: raw softmax scores of the CNN and two fine-tuned AST models for that recording.", max_h=2.2 * inch),
          figure(os.path.join(SCR, "console_training.png"), "Training demo notebook: one manual step (forward, loss, backward, optimiser step) and a short training loop.", max_h=3.0 * inch),
          figure(os.path.join(rep, "G4_per_class_recall.png"), "Per-class recall of all systems on the source-disjoint test set.",
                 max_h=3.0 * inch)]

    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
                            title="Machine Learning Lab Project Report", author="Harbani Kaur, Ina Sharma")
    doc.build(s, onFirstPage=page_number, onLaterPages=page_number)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
