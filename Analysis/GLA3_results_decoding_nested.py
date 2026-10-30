#!/usr/bin/env python3
"""
check_nested_decoding.py
========================

Group statistics for the nested voxel-count decoding (Reviewer 2, comment 1a),
using the statistics and Excel export of GLA3_results_decoding.py.

GLA3 cannot read the nested files directly (their per_pair sheet has one
accuracy per pair and ROI, with no 'voxels' column), so this script loads
them itself and then calls GLA3's functions:

    GLA3.subject_means          mean over decoding sets per subject
    GLA3.run_permutation_tests  sign-flip permutation p, Cohen's d, BF10
    GLA3.export_stats_excel     one styled xlsx per ROI (row "topnested")

With one accuracy per ROI, GLA3's FDR across voxel counts no longer applies
(p_fdr = p_raw). The overview adds BH-FDR across ROIs within each phase
(p_fdr_rois) and summarizes which voxel counts the inner loop chose.

Output (in OUT_DIR):
    decoding_stats_<analysis>_<roi>_<tag>.xlsx    GLA3 layout, one per ROI
    overview_<analysis>_<tag>.xlsx                all ROIs + voxel choices
"""

from __future__ import annotations

import importlib.util
import re
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# ============================================================================
# Settings
# ============================================================================
GLA3_PATH = Path("/home/annaz/PycharmProjects/Studie-BWM-1/Analysis/GLA3_results_decoding.py")
IN_DIR    = Path("/home/annaz/BindingWM/Studie-BWM-1_CP_revision1/decoding_nested")
OUT_DIR   = IN_DIR / "group_stats"

ANALYSIS = "binding"
PHASES   = ["del"]
ROIS     = ["V1", "V2", "V3", "V4", "early_visual", "pari", "dlpfc"]
SUBJECTS = [f"{i:02d}" for i in range(1, 21)]

# Which nested run to read: "nested-pooled", "nested-pooled_C1", "nested-per_pair"
RUN_TAG  = "nested-pooled"

CHANCE         = 0.5
N_PERMUTATIONS = 100_000
CONDITION      = "nested"   # fills GLA3's 'condition' (voxel-count) column


# ============================================================================
# Helpers
# ============================================================================
def load_gla3():
    spec = importlib.util.spec_from_file_location("gla3", GLA3_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gla3"] = mod   # lets GLA3's multiprocessing Pool pickle its functions
    spec.loader.exec_module(mod)
    return mod


def find_nested_file(sub: str, phase: str) -> Path | None:
    """Newest file for exactly this RUN_TAG (so 'nested-pooled' does not
    also match 'nested-pooled_C1')."""
    d = IN_DIR / f"sub-{sub}" / ANALYSIS
    if not d.exists():
        return None
    pat = re.compile(rf"decoding_{ANALYSIS}_{phase}_{re.escape(RUN_TAG)}_\d{{8}}\.xlsx$")
    hits = sorted(p for p in d.glob("*.xlsx") if pat.match(p.name))
    return hits[-1] if hits else None


def load_nested() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (accuracy rows in GLA3 format, voxel-choice rows)."""
    acc, choice = [], []
    for phase in PHASES:
        for sub in SUBJECTS:
            fn = find_nested_file(sub, phase)
            if fn is None:
                print(f"  [MISS] sub-{sub} / {phase}")
                continue
            pp = pd.read_excel(fn, sheet_name="per_pair")
            pp = pp[pp["ROI"].isin(ROIS)]
            if pp.empty:
                print(f"  [WARN] sub-{sub} / {phase}: none of the ROIs in {fn.name}")
                continue
            acc.append(pd.DataFrame({
                "subject":   sub,
                "phase":     phase,
                "area":      pp["ROI"].to_numpy(),
                "condition": CONDITION,
                "value":     pp["accuracy"].to_numpy(),
            }))
            vc = pd.read_excel(fn, sheet_name="voxel_choice")
            vc = vc[vc["ROI"].isin(ROIS)].assign(subject=sub, phase=phase)
            choice.append(vc)

    if not acc:
        raise RuntimeError(f"No '{RUN_TAG}' files found under {IN_DIR}.")
    df = pd.concat(acc, ignore_index=True)
    ch = pd.concat(choice, ignore_index=True)
    n_sub = df.groupby("phase")["subject"].nunique().to_dict()
    print(f"\nLoaded {len(df)} rows; subjects per phase: {n_sub}")
    return df, ch


def roi_order(s: pd.Series) -> pd.Series:
    return s.map({r: i for i, r in enumerate(ROIS)})


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    gla3 = load_gla3()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Nested decoding check | {ANALYSIS} | {RUN_TAG} | phases {PHASES}")
    df_pairs, choice = load_nested()
    df_subj = gla3.subject_means(df_pairs)

    # ---- GLA3 statistics -------------------------------------------------
    stats_df = gla3.run_permutation_tests(df_subj, chance=CHANCE,
                                          n_permutations=N_PERMUTATIONS)

    # BH-FDR across ROIs within each phase (the relevant family now).
    stats_df["p_fdr_rois"] = np.nan
    for phase, idx in stats_df.groupby("phase").groups.items():
        valid = stats_df.loc[idx, "p_raw"].dropna()
        if len(valid):
            stats_df.loc[valid.index, "p_fdr_rois"] = \
                multipletests(valid, method="fdr_bh")[1]
    stats_df = stats_df.sort_values(["phase", "area"], key=lambda s:
                                    roi_order(s) if s.name == "area" else s)

    # ---- voxel choices -----------------------------------------------------
    # share of folds per subject -> mean share across subjects
    choice_pivot = (choice.pivot_table(index=["phase", "ROI"], columns="voxels",
                                       values="share", aggfunc="sum",
                                       fill_value=0)
                    .div(choice.groupby(["phase", "ROI"])["subject"].nunique(),
                         axis=0)
                    .round(3)
                    .reset_index()
                    .sort_values(["phase", "ROI"], key=lambda s:
                                 roi_order(s) if s.name == "ROI" else s))

    subj_wide = (df_subj.pivot_table(index=["phase", "subject"], columns="area",
                                     values="value")
                 .reindex(columns=[r for r in ROIS if r in df_subj["area"].unique()])
                 .round(4).reset_index())

    # ---- print -------------------------------------------------------------
    cols = ["phase", "area", "n", "mean", "sem", "cohens_d",
            "p_raw", "p_fdr_rois", "bayes_H1"]
    print("\n" + "-" * 72)
    print(f"Group results ({RUN_TAG}); p_fdr_rois = BH across ROIs per phase")
    print("-" * 72)
    print(stats_df[cols].round(4).to_string(index=False))
    print("\nVoxel counts chosen in the inner loop (mean share of folds):")
    print(choice_pivot.to_string(index=False))

    # ---- GLA3-style workbook per ROI ---------------------------------------
    for roi in ROIS:
        if roi not in stats_df["area"].values:
            continue
        args = Namespace(roi=roi, analysis=ANALYSIS, phases=PHASES,
                         voxels=[CONDITION])
        gla3.export_stats_excel(
            stats_df, OUT_DIR / f"decoding_stats_{ANALYSIS}_{roi}_{RUN_TAG}.xlsx", args)

    # ---- overview workbook -------------------------------------------------
    overview = OUT_DIR / f"overview_{ANALYSIS}_{RUN_TAG}.xlsx"
    with pd.ExcelWriter(overview) as xl:
        stats_df.round(6).to_excel(xl, sheet_name="group_stats", index=False)
        choice_pivot.to_excel(xl, sheet_name="voxel_choice", index=False)
        subj_wide.to_excel(xl, sheet_name="subject_accuracy", index=False)
    print(f"\nOverview saved -> {overview}")
    print("Done.")


if __name__ == "__main__":   # required: GLA3's permutation test uses a Pool
    main()