#!/usr/bin/env python3
"""
analyze_decoding.py
===================

Group-level statistics on ROI-based SVM decoding accuracy.

For one (analysis, ROI) at a time, this script:

    1. Loads decoder-output xlsx files (one per subject x phase) from
       a directory tree produced by 08_decode_svm.py / decode_svm_loop.py.
    2. Filters to one ROI and a chosen list of voxel counts.
    3. Averages across pairs within each (subject, phase, voxels).
    4. Runs a subject-wise sign-flip permutation test against chance,
       computes Cohen's d and a Bayes factor (Cauchy prior, one-sided),
       and applies BH-FDR within each (phase x area) across voxel counts.
    5. Writes a styled .xlsx summary with one block per phase.

Output
------
    <output-dir>/decoding_stats_<analysis>_<roi>.xlsx

Installation
------------
    pip install numpy pandas scipy statsmodels openpyxl tqdm matplotlib
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.integrate import quad
from scipy.stats import sem
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# Stats primitives
# ============================================================================
def bayes_factor_point_null(data, null_value=0, prior_scale=1.0,
                            alternative="greater"):
    data = np.asarray(data)
    n = len(data)
    mean_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    se = std_data / np.sqrt(n)
    t_stat = (mean_data - null_value) / se
    r = prior_scale

    def integrand(delta):
        return (stats.nct.pdf(t_stat, df=n - 1, nc=np.sqrt(n) * delta)
                * stats.cauchy.pdf(delta, 0, r))

    if alternative == "greater":
        marginal_h1, _ = quad(integrand, 0, 10, limit=100)
        marginal_h1 *= 2
    else:
        marginal_h1, _ = quad(integrand, -10, 10, limit=100)

    marginal_h0 = stats.t.pdf(t_stat, df=n - 1)
    bf10 = marginal_h1 / marginal_h0 if marginal_h0 != 0 else np.nan
    return bf10, 1.0 / bf10 if bf10 else np.nan


def cohens_d_one_sample(data, null_value=0.5):
    data = np.asarray(data)
    return (np.mean(data) - null_value) / np.std(data, ddof=1)


def single_permutation(subject_vals):
    sample = [np.random.choice(subject_vals[subj], 1) for subj in subject_vals]
    return np.mean(sample)


def permutation_pval_subjectwise(values, subjects, n_permutations=10000,
                                 chance=0.5):
    subject_vals = {}
    for sub, val in zip(subjects, values):
        diff = val - chance
        subject_vals[sub] = [abs(diff), -abs(diff)]
    observed = np.mean(values - chance)
    with Pool() as pool:
        null_dist = pool.starmap(
            single_permutation,
            [(subject_vals,) for _ in range(int(n_permutations))],
        )
    return np.sum(np.array(null_dist) >= observed) / n_permutations


def run_permutation_tests(df, chance=0.5, n_permutations=10000):
    results = []
    grouped = df.groupby(["phase", "area", "condition"])
    for (phase, area, cond), group in tqdm(grouped, desc="Permutation Tests"):
        values   = group["value"].values
        subjects = group["subject"].values
        if len(values) < 2:
            mean_val = np.mean(values)
            results.append({
                "phase": phase, "area": area, "condition": cond,
                "mean": mean_val, "sem": 0.0,
                "ci_low": mean_val, "ci_high": mean_val,
                "cohens_d": np.nan, "bayes_H1": np.nan, "bayes_H0": np.nan,
                "p_raw": np.nan, "n": len(values),
            })
            continue

        mean_val = np.mean(values)
        sem_val  = sem(values)
        ci_low   = mean_val - 1.96 * sem_val
        ci_high  = mean_val + 1.96 * sem_val
        d        = cohens_d_one_sample(values, null_value=chance)
        bayes    = bayes_factor_point_null(values, null_value=chance,
                                           prior_scale=0.707,
                                           alternative="greater")
        p_val    = permutation_pval_subjectwise(values, subjects,
                                                n_permutations, chance)

        results.append({
            "phase": phase, "area": area, "condition": cond,
            "mean": mean_val, "sem": sem_val,
            "ci_low": ci_low, "ci_high": ci_high,
            "cohens_d": d, "bayes_H1": bayes[0], "bayes_H0": bayes[1],
            "p_raw": p_val, "n": len(values),
        })

    result_df = pd.DataFrame(results)
    result_df["p_fdr"] = np.nan

    # FDR within each phase x area, across conditions (voxel counts).
    for phase in result_df["phase"].unique():
        for area in result_df["area"].unique():
            mask  = (result_df["phase"] == phase) & (result_df["area"] == area)
            valid = mask & result_df["p_raw"].notna()
            if valid.sum() > 1:
                _, p_fdr, _, _ = multipletests(result_df.loc[valid, "p_raw"],
                                               method="fdr_bh")
                result_df.loc[valid, "p_fdr"] = p_fdr
            elif valid.sum() == 1:
                result_df.loc[valid, "p_fdr"] = result_df.loc[valid, "p_raw"]

    def star(p):
        if pd.isna(p):
            return ""
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

    result_df["signif"]     = result_df["p_fdr"].apply(star)
    result_df["signif_raw"] = result_df["p_raw"].apply(star)
    return result_df


# ============================================================================
# Display labels
# ============================================================================
PHASE_LABELS = {
    "enc": "Encoding Phase",
    "del": "Delay Phase",
    "res": "Response Phase",
}

ROI_LABELS = {
    "V1":           "V1",
    "V2":           "V2",
    "V3":           "V3",
    "V4":           "V4",
    "early_visual": "Early Visual Cortex",
    "pari":         "Parietal Cortex",
    "dlpfc":        "dlPFC",
}


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Group statistics on ROI-based SVM decoding accuracy.",
    )
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Decoding output root (the --output-dir you gave to "
                        "08_decode_svm.py / decode_svm_loop.py; contains "
                        "sub-XX/<analysis>/decoding_<analysis>_<phase>_*.xlsx).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where the stats summary xlsx will be written.")
    p.add_argument("--analysis", required=True,
                   choices=["binding", "color", "motion",
                            "temporal_order", "binding_interaction"],
                   help="Which decoding analysis to summarise.")
    p.add_argument("--roi", default="early_visual",
                   help="ROI label to analyse (e.g. 'early_visual', 'V1', "
                        "'pari', 'dlpfc'). One ROI per run.")
    p.add_argument("--phases", nargs="+", default=["del"],
                   choices=["enc", "del", "res"],
                   help="Phases to include (one block per phase in the output).")
    p.add_argument("--subjects", nargs="+",
                   default=[f"{i:02d}" for i in range(1, 21)],
                   help="Subject labels without 'sub-'.")
    p.add_argument("--voxels", nargs="+", type=int,
                   default=[200, 300, 400, 500, 600, 700, 800,
                            900, 1000, 2000, 3000],
                   help="Voxel-count levels to include (must match the "
                        "voxel counts produced by 07_create_top_voxel_rois.py).")
    p.add_argument("--chance", type=float, default=0.5,
                   help="Chance level for one-sample stats (default 0.5).")
    p.add_argument("--n-permutations", type=int, default=100_000,
                   help="Number of subject-wise sign-flip permutations.")
    return p.parse_args()


# ============================================================================
# Data loading
# ============================================================================
def find_file(input_dir: Path, sub: str, analysis: str, phase: str):
    """Return the most recent decoding xlsx for (sub, analysis, phase)."""
    pattern = str(input_dir / f"sub-{sub}" / analysis /
                  f"decoding_{analysis}_{phase}_*.xlsx")
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def load_all_data(args: argparse.Namespace) -> pd.DataFrame:
    all_rows = []
    for phase in args.phases:
        for sub in args.subjects:
            fn = find_file(args.input_dir, sub, args.analysis, phase)
            if fn is None:
                print(f"  [MISS] sub-{sub} / {phase}")
                continue
            try:
                raw = pd.read_excel(fn, sheet_name="per_pair")
            except Exception as e:
                print(f"  [WARN] sub-{sub} / {phase}: could not read {fn}: {e}")
                continue

            raw = raw[raw["ROI"] == args.roi]
            raw = raw[raw["voxels"].isin(args.voxels)]
            if raw.empty:
                print(f"  [WARN] sub-{sub} / {phase}: no rows for ROI={args.roi}")
                continue

            sub_df = raw.rename(columns={"ROI":      "area",
                                         "voxels":   "condition",
                                         "accuracy": "value"})
            sub_df["subject"] = sub
            sub_df["phase"]   = phase
            all_rows.append(sub_df[["subject", "phase", "area",
                                    "condition", "value"]])

    if not all_rows:
        raise RuntimeError("No data loaded -- check --input-dir, --analysis, "
                           "--phases, --roi, --subjects.")

    df = pd.concat(all_rows, axis=0, ignore_index=True)
    print(f"\nLoaded {len(df)} rows total")
    print(f"  phases     : {sorted(df['phase'].unique())}")
    print(f"  ROIs       : {sorted(df['area'].unique())}")
    print(f"  conditions : {sorted(df['condition'].unique())}")
    return df


def subject_means(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["subject", "phase", "area", "condition"])
              .agg(value=("value", "mean"))
              .reset_index())


# ============================================================================
# Styled Excel export
# ============================================================================
def export_stats_excel(stats_df: pd.DataFrame, out_path: Path,
                       args: argparse.Namespace) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    HDR_FILL  = PatternFill("solid", fgColor="1F3864")
    GRP_FILLS = {0: PatternFill("solid", fgColor="EBF3FB"),
                 1: PatternFill("solid", fgColor="FDFEFE")}
    GRP_HDR_FILLS = {0: PatternFill("solid", fgColor="BDD7EE"),
                     1: PatternFill("solid", fgColor="D9EAD3")}
    SIG_FILL = PatternFill("solid", fgColor="FFF2CC")

    def mk_font(bold=False, size=9, color="000000"):
        return Font(name="Calibri", bold=bold, size=size, color=color)

    CENTER = Alignment(horizontal="center", vertical="center")
    LEFT   = Alignment(horizontal="left",   vertical="center")
    thin   = Side(style="thin",   color="D0D0D0")
    med    = Side(style="medium", color="4472C4")
    BORD   = Border(left=thin, right=thin, top=thin, bottom=thin)
    BORD_L = Border(left=med,  right=thin, top=thin, bottom=thin)

    STAT_COLS = ["mean", "sem", "ci_low", "ci_high",
                 "cohens_d", "p_raw", "p_fdr", "bayes_H1"]
    STAT_HDRS = ["Mean", "SEM", "CI min", "CI max",
                 "Cohen's d", "p uncorr", "p FDR", "BF10"]
    NUM_FMTS  = ["0.0000", "0.0000", "0.0000", "0.0000",
                 "0.000", "0.0000", "0.0000", "0.00"]

    roi_label = ROI_LABELS.get(args.roi, args.roi)

    def write_sheet(ws, sheet_title, groups):
        # Title row
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=9)
        tc = ws.cell(row=1, column=1, value=sheet_title)
        tc.font = Font(name="Calibri", bold=True, size=13, color="FFFFFF")
        tc.fill = HDR_FILL
        tc.alignment = LEFT
        ws.row_dimensions[1].height = 22

        # Column-header row
        ws.cell(row=2, column=1, value="").border = BORD
        for ci, hdr in enumerate(STAT_HDRS, start=2):
            c = ws.cell(row=2, column=ci, value=hdr)
            c.font = mk_font(bold=True, size=9, color="FFFFFF")
            c.fill = HDR_FILL
            c.alignment = CENTER
            c.border = BORD
        ws.row_dimensions[2].height = 16

        cur_row = 3
        for gi, grp in enumerate(groups):
            fill     = GRP_FILLS[gi % 2]
            hdr_fill = GRP_HDR_FILLS[gi % 2]

            # Group header row
            ws.merge_cells(start_row=cur_row, start_column=1,
                           end_row=cur_row,   end_column=9)
            gc = ws.cell(row=cur_row, column=1, value=grp["label"])
            gc.font = mk_font(bold=True, size=10, color="1F3864")
            gc.fill = hdr_fill
            gc.alignment = LEFT
            gc.border = Border(left=med, right=med, top=med, bottom=med)
            ws.row_dimensions[cur_row].height = 15
            cur_row += 1

            for (row_label, phase, cond) in grp["rows"]:
                row_data = stats_df[
                    (stats_df["phase"]     == phase) &
                    (stats_df["area"]      == args.roi) &
                    (stats_df["condition"] == cond)
                ]

                lc = ws.cell(row=cur_row, column=1, value=row_label)
                lc.font = mk_font(size=9); lc.fill = fill
                lc.alignment = LEFT; lc.border = BORD_L

                if row_data.empty:
                    for ci in range(2, 10):
                        c = ws.cell(row=cur_row, column=ci, value="–")
                        c.font = mk_font(size=9); c.fill = fill
                        c.alignment = CENTER; c.border = BORD
                else:
                    rd = row_data.iloc[0]
                    p_fdr_val = rd.get("p_fdr", np.nan)
                    sig = (not pd.isna(p_fdr_val)) and (p_fdr_val < 0.05)

                    for ci, (col, fmt) in enumerate(zip(STAT_COLS, NUM_FMTS),
                                                    start=2):
                        val = rd.get(col, np.nan)
                        c = ws.cell(row=cur_row, column=ci,
                                    value=None if pd.isna(val)
                                    else round(float(val), 6))
                        c.font = mk_font(size=9, bold=sig)
                        c.fill = SIG_FILL if sig else fill
                        c.alignment = CENTER; c.border = BORD
                        if val is not None and not pd.isna(val):
                            c.number_format = fmt

                ws.row_dimensions[cur_row].height = 14
                cur_row += 1

            cur_row += 1   # blank spacer between blocks

        ws.column_dimensions["A"].width = 28
        for ci in range(len(STAT_HDRS)):
            ws.column_dimensions[get_column_letter(ci + 2)].width = 11
        ws.freeze_panes = "B3"

    # One block per phase
    groups = []
    for phase in args.phases:
        phase_label = PHASE_LABELS.get(phase, phase)
        groups.append({
            "label": f"{roi_label} — {phase_label}",
            "rows":  [(f"top{vox}", phase, vox) for vox in args.voxels],
        })

    wb = Workbook()
    ws = wb.active
    # Excel limits sheet names to 31 chars; truncate defensively.
    ws.title = f"{args.roi}_{args.analysis}"[:31]

    phases_pretty = ", ".join(PHASE_LABELS.get(p, p) for p in args.phases)
    write_sheet(ws,
                f"{roi_label} | {args.analysis} | {phases_pretty}",
                groups)

    wb.save(out_path)
    print(f"Stats Excel saved -> {out_path}")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()

    print("-" * 64)
    print("Decoding-results group statistics")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>16}: {v}")
    print("-" * 64)

    print("\nLoading data ...")
    df_pairs = load_all_data(args)
    df_subj  = subject_means(df_pairs)

    print("\nRunning permutation tests ...")
    stats_df = run_permutation_tests(df_subj,
                                     chance=args.chance,
                                     n_permutations=args.n_permutations)
    print(stats_df[["phase", "area", "condition",
                    "mean", "sem", "p_fdr", "signif"]].to_string())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = args.output_dir / f"decoding_stats_{args.analysis}_{args.roi}.xlsx"
    print("\nExporting stats summary ...")
    export_stats_excel(stats_df, out_xlsx, args)

    print("Done.")


if __name__ == "__main__":
    main()