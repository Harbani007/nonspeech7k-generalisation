"""Background-probe numbers for the report, slides and presentation folder, read from the A100 result files
(docs/results/a100_rerun/probe_db20_edgefix, probe_db30_edgefix; edge-fixed masks, 12 fully fine-tuned AST checkpoints)."""
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A100 = os.path.join(REPO, "docs", "results", "a100_rerun")
CONDITIONS = {"ast_ft_source_disjoint_standard": "ordinary", "ast_ft_source_disjoint_source": "source-balanced",
              "ast_ftsc_source_disjoint_standard": "contrastive", "ast_ftsc_source_disjoint_source": "combined"}
VARIANTS = {"full": "full clip", "foreground": "vocalisation only", "random": "random cut (control)",
            "background": "vocalisation removed"}


def load(db):
    """Fine-tuned AST rows (4 conditions x 4 variants) and the background-trained linear probe, one cut level."""
    r = pd.read_csv(os.path.join(A100, f"probe_db{db}_edgefix", "background_probe_results.csv"))
    ft = r[r.system.isin(CONDITIONS)].copy()
    ft["condition"] = ft.system.map(CONDITIONS)
    probe = r[(r.system == "frozen-AST probe (train background)") & (r.variant == "background")].iloc[0]
    return ft, probe


def summary():
    """Dict of the numbers quoted in text: ranges per cut level, probe values, sample sizes."""
    out = {}
    for db in (20, 30):
        ft, probe = load(db)
        bg = ft[ft.variant == "background"]
        fg = ft[ft.variant == "foreground"]
        full = ft[ft.variant == "full"]
        out[db] = dict(bg_min=bg.macro_f1.min(), bg_max=bg.macro_f1.max(), bg_lo_min=bg.ci_lo.min(), bg_lo_max=bg.ci_lo.max(),
                       chance_min=bg.chance_95.min(), chance_max=bg.chance_95.max(),
                       fg_min=fg.macro_f1.min(), fg_max=fg.macro_f1.max(), full_min=full.macro_f1.min(), full_max=full.macro_f1.max(),
                       probe=probe.macro_f1, probe_lo=probe.ci_lo, probe_hi=probe.ci_hi, probe_chance=probe.chance_95,
                       clips=int(bg.test_clips.iloc[0]), recordings=int(bg.test_sources.iloc[0]),
                       above_chance=int((bg.ci_lo > bg.chance_95).sum()))
    return out


def table():
    """Wide table: condition x (variant at 20 dB, background at 30 dB)."""
    rows = []
    for db in (20, 30):
        ft, probe = load(db)
        for _, x in ft.iterrows():
            rows.append(dict(cut=f"{db} dB", condition=x.condition, variant=VARIANTS[x.variant], macro_f1=round(x.macro_f1, 3),
                             ci=f"{x.ci_lo:.3f}-{x.ci_hi:.3f}", chance_95=round(x.chance_95, 3),
                             test_clips=int(x.test_clips), test_recordings=int(x.test_sources)))
        rows.append(dict(cut=f"{db} dB", condition="linear probe trained on background only", variant="vocalisation removed",
                         macro_f1=round(probe.macro_f1, 3), ci=f"{probe.ci_lo:.3f}-{probe.ci_hi:.3f}",
                         chance_95=round(probe.chance_95, 3), test_clips=int(probe.test_clips), test_recordings=int(probe.test_sources)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json
    print(json.dumps({k: {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} for k, v in summary().items()}, indent=1))
