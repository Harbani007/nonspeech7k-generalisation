"""Builds Nonspeech7k_project_brief.pdf (run from the folder you want the PDF in)."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

OUT = "Nonspeech7k_project_brief.pdf"
INK, MUTED, ACCENT, TINT = colors.HexColor("#1f2933"), colors.HexColor("#5b6670"), colors.HexColor("#1d4e89"), colors.HexColor("#eaf1f8")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10, leading=14.5, textColor=INK, alignment=TA_LEFT, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.6, leading=11.5, textColor=MUTED, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=8.8, leading=11.5, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=ACCENT, spaceBefore=12, spaceAfter=5)
title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=3)
sub = ParagraphStyle("sub", parent=body, fontSize=11, leading=15, textColor=MUTED, spaceAfter=2)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)
box = ParagraphStyle("box", parent=body, fontName="Helvetica-Bold", fontSize=10.6, leading=15, spaceAfter=0)


def P(t, s=body):
    return Paragraph(t, s)


def B(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(c, cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d2da")),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), TINT))
    t.setStyle(TableStyle(st))
    return t


def callout(text):
    t = Table([[Paragraph(text, box)]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def link(url, text=None):
    return f'<link href="{url}" color="#1d4e89">{text or url}</link>'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Nonspeech7k project brief - draft, not yet run on real data")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


s = []
s += [P("Is Nonspeech7k measuring the sound or the recording?", title),
      P("Project brief: the problem, the question, and what the code does", sub),
      P("Draft of 27 October 2026. Written for a reader who knows basic machine learning but not this project.", small), Spacer(1, 4)]

s += [P("1. The dataset", h1),
      P("Nonspeech7k has 7,014 short audio clips (0.5 to 4 seconds) of human non-speech sounds in seven classes: breath, cough, cry, laugh, "
        "scream, sneeze and yawn. The clips come from Freesound, YouTube and Aigei. The official split is 6,289 training and 725 test clips. "
        "Published results include about 79% accuracy for CNNs in the original 2023 paper, about 96% for classical models on hand-made features "
        "in a 2025 paper (its split was not checked, so the numbers may not be comparable), and about 86% for audio-language models. "
        "Few papers use the dataset: a citation search found only about three classification studies.")]

s += [P("2. The problem", h1),
      P("The dataset was built by cutting longer recordings into short clips. Clips cut from the same recording share a microphone, a room, "
        "background noise and often the same person. If clips from one recording land in both the training set and the test set, a model can score "
        "well by recognising the recording instead of the sound. The accuracy then says little about how the model would do on a new recording."),
      P("This is a known problem in other audio datasets. Leakage studies report inflated accuracy of up to 25% on UrbanSound8K and 10 to 48 points on "
        "ship-noise datasets. For Nonspeech7k two things make it worth checking:"),
      *B(["The metadata gives every clip a <b>File_ID</b> naming the recording it was cut from, so we can group clips by recording.",
          "In the part of the training metadata we could read (only the first ~1,450 rows, all breath clips), a handful of YouTube videos supplied "
          "hundreds of clips each. This comes from a partial reading and still needs confirming.",
          "We have <b>not</b> yet checked whether the official train and test splits share recordings."])]

s += [P("3. Problem statement", h1),
      callout("With a small labelled budget (300 to 700 training clips) and frozen pretrained audio embeddings, how much of Nonspeech7k accuracy "
              "survives when the test clips come from recordings the model has never seen? And do features that describe a sound's "
              "temporal structure (bursts, rhythm, modulation) hold up better on unseen recordings than spectral features or embeddings?")]

s += [P("4. The hypothesis and why it might hold", h1),
      P("Body-produced sounds have characteristic timing. A cough is one to three sharp bursts. A laugh is a train of bursts about four to six per "
        "second. A breath is a slow swell. A sneeze is a build-up followed by one burst. A yawn is a long, slow swell. That timing comes from the body, "
        "not from the microphone. Spectral colour and the background noise level, by contrast, depend strongly on the recording."),
      P("<b>Hypothesis:</b> features describing temporal structure lose less accuracy than spectral features or embeddings when we move from a random clip "
        "split to a split that keeps recordings apart. <b>This may well be false.</b> Embeddings might win everywhere. Either outcome gets reported.")]

s += [P("5. What the code does", h1),
      P("The code lives in the <font name='Courier'>ns7k/</font> folder and runs in five steps."),
      *B(["<b>Audit</b> (a short separate script). Count how many recordings appear in both the official train and test sets, how many clips each "
          "recording contributes, and how much of each class comes from its single largest recording.",
          "<b>Features.</b> Compute a set of numbers for every clip (table below).",
          "<b>Splits.</b> Evaluate under three protocols (table below), with five random seeds each.",
          "<b>Budget.</b> From the training portion, draw 300, 500 or 700 clips, either at random or with at most three clips per recording.",
          "<b>Classifier.</b> Standardise each feature block and fit a class-balanced logistic regression. Report macro-F1 and balanced accuracy."]),
      Spacer(1, 4)]

s += [KeepTogether([P("Feature sets compared", cellb),
      table([["Set", "What it captures", "Size"],
             ["struct", "Burst count and rate, gaps between bursts, attack and decay time, rhythm strength, energy in 5 modulation ranges (0.5 to 16 Hz) across "
                        "4 frequency bands, duration and envelope shape. All measured relative to the loudest frame, so recording gain does not matter.", "38"],
             ["struct-nf", "The same without two columns that depend on background noise level (dynamic range and mean level), as an ablation.", "36"],
             ["spec", "Control: 13 MFCCs (mean and standard deviation) plus spectral centroid, bandwidth, rolloff, flatness and zero-crossing rate.", "35"],
             ["ast, clap", "Frozen embeddings from pretrained AST and CLAP models. Fusion with struct is also tested. <i>Code written, not run.</i>", "768 / 512"]],
            [24 * mm, 136 * mm, 14 * mm])]), Spacer(1, 8)]

s += [KeepTogether([P("Evaluation protocols compared", cellb),
      table([["Protocol", "How the split is made", "What it tells us"],
             ["official", "Train on the official train clips, test on the official test clips.", "What published papers (probably) report."],
             ["random", "Random stratified clip split of the pooled data. Recordings leak across the split.", "The optimistic number."],
             ["group", "Split by File_ID: test recordings never appear in training.", "Performance on genuinely unseen recordings."]],
            [24 * mm, 90 * mm, 60 * mm])]), Spacer(1, 6)]

s += [P("<b>The key output</b> is the <i>protocol gap</i>: macro-F1 under the random split minus macro-F1 under the group split, for each feature set. "
        "A small gap means the features depend less on recording identity. The hypothesis predicts a smaller gap for struct than for spec and the embeddings. "
        "A second table compares random sampling with recording-balanced sampling at each budget, which shows whether training on more distinct "
        "recordings helps. That result is expected to go the obvious way, so it is a supporting experiment rather than the main claim.")]

s += [P("6. What the code looks for in each sound", h1),
      P("The features are chosen from how each sound is produced. The right-hand column is what we <i>expect</i> the numbers to show. It has not been "
        "measured on the real data yet, and some of these expectations will probably turn out wrong."),
      table([["Class", "Typical temporal pattern", "Features that should reflect it"],
             ["breath", "Slow swells, no sharp attacks", "Few bursts, long active duration, modulation energy below 2 Hz"],
             ["cough", "One to three short, sharp bursts", "Fast attack, short decay, short active duration, modulation energy at 2 to 8 Hz"],
             ["cry", "Sustained voiced sound with a slow wobble (about 1 to 3 Hz)", "Long active duration, moderate burst count, modulation at 1 to 4 Hz"],
             ["laugh", "Repeating bursts, about 4 to 6 per second", "High burst rate, strong rhythm (autocorrelation peak near 0.2 s), modulation at 4 to 8 Hz"],
             ["scream", "Sustained, loud, steady", "Few bursts, flat envelope, high active fraction"],
             ["sneeze", "Build-up, then one dominant burst", "One burst, late energy centroid, fast attack"],
             ["yawn", "Long, slow rise and fall", "Long duration, very low burst rate, modulation below 1 Hz"]],
            [20 * mm, 68 * mm, 86 * mm]),
      Spacer(1, 4),
      P("The spectral control (MFCC and spectral statistics) describes what each sound <i>sounds like</i> instead. The experiment asks which description "
        "still works when the recording is new. Classes with similar timing, such as cry and scream, are where structure features are likely to struggle.")]

s += [P("7. Has this already been researched?", h1),
      P("A second literature check (27 October 2026) looked at each half of the problem statement. It found related work for both halves, "
        "but no study that tests either one on Nonspeech7k. Web searches cannot prove that nothing exists."),
      table([["Question", "Closest existing work", "What it shows", "Gap"],
             ["How much accuracy is lost when recordings are kept apart?",
              "UrbanSound8K leakage study (ICIIS 2025); UniqueShip (ship noise, 2026); infant-cry transfer audit (2026); cross-dataset tuberculosis cough study (2026)",
              "Random splits inflate accuracy (up to 25% on UrbanSound8K, 10 to 48 points on ship noise). Cross-corpus transfer is often poor.",
              "Not found for Nonspeech7k. It is also unknown whether the official split is already recording-disjoint. The published descriptions disagree: "
              "one says 725 test clips, another says 407 test clips from Freesound only."],
             ["Do temporal-structure features generalise better than spectral features or embeddings?",
              "ESC-50 rhythm-versus-spectral comparison (2025); tuberculosis cough study (2026); speech-versus-song modulation study (2024)",
              "Rhythm features did much worse than spectral on ESC-50 (one dataset, no shift). Cough representations cluster by recording device. "
              "Modulation features separate speech from song across 21 societies (a different task).",
              "No direct test found. The evidence is mixed, so the hypothesis could easily fail."],
             ["Do frozen embeddings work with a small labelled budget on non-speech human sounds?",
              "BLEMORE challenge entry (2026)",
              "Frozen features beat full fine-tuning on about 2,500 non-verbal emotion clips.",
              "Supports the frozen-embedding setup. It is not a recording-level test."]],
            [38 * mm, 42 * mm, 54 * mm, 40 * mm]),
      Spacer(1, 4),
      P("<b>Reading of this:</b> if the audit finds recording overlap, we have a real but incremental finding about the dataset. If the split is already clean, "
        "the first half mostly disappears and the project rests on the structure-feature question and the budget experiment, whose results cannot be predicted.")]

s += [P("8. Novelty, stated honestly", h1),
      *B(["<b>No new algorithm.</b> Recording-disjoint evaluation, frozen embeddings and hand-made burst features all exist already.",
          "<b>What we found no prior work on:</b> a recording-disjoint evaluation of Nonspeech7k, and a test of whether structure features generalise across "
          "recordings better than spectral ones on it. These are adjacent to published work on UrbanSound8K, ship noise and infant-cry corpora, not identical to any of it.",
          "<b>Limits of that claim.</b> It rests on web and citation searches, not an exhaustive review. One closely related paper (the 2025 IEEE Access ensemble paper) "
          "could not be opened and was not read.",
          "<b>The second half is the riskier one.</b> Nearby evidence (section 7) does not suggest that structure features will win, and a cough study found that "
          "learned audio representations cluster by recording device. We should expect that the hypothesis may fail.",
          "<b>Type of contribution.</b> Mainly an analysis of the dataset, with a small method component (the structure features). A negative result is possible and still reportable."])]

s += [P("9. Tricks planned for later", h1),
      P("None of these is built or tested, and none is known to help. They are candidates to add once the basic experiment works, and each one is a "
        "hypothesis, not a result."),
      table([["Trick", "What it is", "Status and uncertainty"],
             ["Frozen embeddings and fusion", "Add AST and CLAP embeddings, and fuse them with the structure features.",
              "Code written, never run. Do not use voc2vec for main claims: it was pretrained on Nonspeech7k."],
             ["Feature-group ablation", "Drop the modulation, burst or rhythm features one group at a time to see which ones matter.", "Easy. Not run."],
             ["Per-recording centering", "Subtract each recording's average feature vector, as channel normalisation does in speech processing.",
              "Needs to know which clips share a recording at test time, so it only applies in that setting."],
             ["Erasing recording identity", "Remove linear information about the recording from embeddings (LEACE, 2023).",
              "A 2026 analysis of HuBERT reports that LEACE can fail to generalise, so it may not help."],
             ["Group-robust classifier", "Train the classifier head with group-DRO or an adversarial recording classifier.",
              "The pieces exist in other audio settings. Low novelty."],
             ["Duplicate and label-conflict check", "Find near-identical clips and clips with conflicting labels using embeddings.",
              "An infant-cry study (2026) found such conflicts across corpora. Not checked on Nonspeech7k."],
             ["Cross-corpus test", "Train on Nonspeech7k and test on VocalSound for cough, laughter and sneeze.",
              "Needs duplicate removal first, since sources can overlap. Only three classes are shared."]],
            [38 * mm, 70 * mm, 66 * mm])]

s += [P("10. Status and caveats", h1),
      *B(["The feature extraction and experiment runner have been tested end to end only on <b>synthetic audio</b>. Results on that data mean nothing about Nonspeech7k.",
          "The AST and CLAP embedding code has never been run. The real dataset (about 2.5 GB) has not been downloaded, and the audit has not been run.",
          "The burst-detection thresholds are guesses. They should be fixed before looking at group-split test results, otherwise tuning them is cheating.",
          "The dataset page states two different licences (CC BY-NC-SA 4.0 and CC BY 4.0). Treat it as non-commercial academic use.",
          "The published descriptions of the official split disagree with each other (725 test clips in one, 407 Freesound-only test clips in another), "
          "and none we found says whether clips from one recording are kept on the same side. Only the audit can settle it.",
          "If the official split turns out to be recording-disjoint, the leakage story weakens, and the project becomes a robustness study."]),
      P("<b>Next steps:</b> download the data, run the audit, extract struct and spec features, run the experiments, then add the embeddings.")]

refs = [
    ("Rashid, Li, Du. Nonspeech7k dataset: Classification and analysis of human non-speech sound. IET Signal Processing, 2023.", "https://doi.org/10.1049/sil2.12233"),
    ("Nonspeech7k dataset on Zenodo.", "https://zenodo.org/records/6967442"),
    ("Exploring Human Non-Speech Sound Recognition: Insights from the Nonspeech7K Dataset. ICE 2025.", "https://doi.org/10.1109/ice63309.2025.10984139"),
    ("A Self-Improving Ensemble Learning Framework for Human Non-Speech Vocalization Classification. IEEE Access, 2025 (not read).", "https://doi.org/10.1109/access.2025.3639444"),
    ("Human Non-Speech Vocalization Recognition and Classification Based on Dual CNNs. FAIA, 2025.", "https://doi.org/10.3233/FAIA250740"),
    ("voc2vec: A Foundation Model for Non-Verbal Vocalization. ICASSP 2025 (pretrained on Nonspeech7k).", "https://arxiv.org/abs/2502.16298"),
    ("UniqueShip: Mitigating Data Leakage in Acoustic Ship Classification Benchmark Datasets. 2026.", "https://arxiv.org/abs/2609.13659"),
    ("Benchmarking Data Leakage and Generalization in Audio Classification: An Empirical Analysis. ICIIS 2025.", "https://researchers.westernsydney.edu.au/en/publications/benchmarking-data-leakage-and-generalization-in-audio-classificat/"),
    ("How Far Do Foundation Models Transfer to Infant Signals? A Cross-Dataset Transfer Audit. 2026.", "https://arxiv.org/abs/2608.08989"),
    ("VocalSound: A Dataset for Improving Human Vocal Sounds Recognition. 2022.", "https://arxiv.org/abs/2205.03433"),
    ("Detection of nonverbal vocalizations using Gaussian mixture models: looking for fillers and laughter in conversational speech.", "https://www.researchgate.net/publication/287863174"),
    ("Belrose et al. LEACE: Perfect linear concept erasure in closed form. NeurIPS 2023.", "https://arxiv.org/abs/2306.03819"),
    ("A framework for analyzing concept representations in neural models (reports LEACE generalisation limits on HuBERT). 2026.", "https://arxiv.org/abs/2605.01381"),
    ("Zhang et al. Why ML-based cough models do not generalize: a systematic cross-dataset evaluation for tuberculosis screening. 2026.", "https://arxiv.org/abs/2608.25846"),
    ("Spectral and Rhythm Feature Performance Evaluation for Category and Class Level Audio Classification with Deep CNNs. 2025 (seen in search summary only).", "https://arxiv.org/abs/2509.07756"),
    ("Spectro-temporal acoustical markers differentiate speech from song across cultures. Nature Communications, 2024 (seen in search summary only).", "https://www.nature.com/articles/s41467-024-49040-3"),
    ("Foundation Model Embeddings Meet Blended Emotions: A Multimodal Fusion Approach for the BLEMORE Challenge. 2026 (seen in search summary only).", "https://arxiv.org/abs/2603.23650"),
]
s += [P("11. References", h1)]
for i, (t, u) in enumerate(refs, 1):
    s.append(Paragraph(f"{t}<br/>{link(u)}", small, bulletText=f"{i}."))

repos = [
    ("patidarmonesh/audio-classification-pytorch", "Residual CNN on Nonspeech7k with the official 6,289/725 split and 3-fold CV. A baseline to compare against; it inherits any leakage in the official split."),
    ("YuanGongND/vocalsound", "VocalSound dataset (21,024 clips), baseline code and split files. Candidate for the cross-corpus test."),
    ("YuanGongND/ast", "Audio Spectrogram Transformer with pretrained AudioSet checkpoints (embedding extractor)."),
    ("qiuqiangkong/audioset_tagging_cnn", "PANNs, including pretrained CNN14 checkpoints (embedding extractor, not yet wired into the code)."),
    ("LAION-AI/CLAP", "CLAP audio-text models for extracting audio embeddings."),
    ("EleutherAI/concept-erasure", "Implementation of LEACE, for the planned recording-identity erasure experiment."),
]
s += [KeepTogether([P("12. Code repositories", h1),
      P("Existing GitHub repositories relevant to the project. Each was confirmed to exist on 27 October 2026. Our own code is local and not yet on GitHub.", small),
      table([["Repository (github.com/...)", "Use"]] + [[f"<font name='Courier'>{n}</font>", d] for n, d in repos], [66 * mm, 108 * mm])])]

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                  title="Is Nonspeech7k measuring the sound or the recording?", author="").build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
