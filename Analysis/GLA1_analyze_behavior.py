#!/usr/bin/env python3
"""
analyze_trials.py
=================

Compute group-level behavioural results from the per-trial TSV files
produced by the BindingWM pipeline (03_create_events_and_trials.py).

Usage
-----
    python analyze_trials.py \\
        --data_dir  /path/to/behav_data \\
        --output_dir /path/to/results \\
        --subjects 01 02 03 ... \\   # optional; defaults to all sub-XX folders found
        --sessions 01 02 03 04 \\    # optional; default: 01 02 03 04
        --n_runs 8                   # optional; default: 8

    # Minimal call — auto-discover subjects:
    python analyze_trials.py \\
        --data_dir /path/to/behav_data \\
        --output_dir /path/to/results

Input
-----
Per-trial TSV files at:
    <data_dir>/sub-XX/timing/ses-YY/trials/sub-XX_ses-YY_task-bwm_run-NN_trials.tsv

Required columns
----------------
    error_code          : int  0=correct, 1=feature error, 2=binding error, 3=no response
    misbinding_subtype  : str  'misbinding_fm' | 'misbinding_fnm' | 'n/a'
    trial_paper         : str  e.g. '1a', '1b', ... '18b'  (for staircase assignment)
    cue                 : str  'c' | 'm'

Output
------
Two parallel result sets are written to <output_dir>/:

    1. ALL trials included
       - behavresults_summary_all.xlsx         per subject x session table
       - behavresults_summary_stats_all.csv    group descriptives
       - behavresults_pie_all.pdf              pie chart of mean proportions

    2. No-response trials EXCLUDED from denominator
       - behavresults_summary_excl_nr.xlsx
       - behavresults_summary_stats_excl_nr.csv
       - behavresults_pie_excl_nr.pdf

Error categories
----------------
    correct             : error_code == 0
    feature_error       : error_code == 1
    binding_error_fm    : error_code == 2 AND misbinding_subtype == 'misbinding_fm'
                          (selected distractor whose features match a presented object)
    binding_error_fnm   : error_code == 2 AND misbinding_subtype == 'misbinding_fnm'
                          (selected distractor whose features do not match any presented object)
    no_response         : error_code == 3

Per-staircase proportions
--------------------------
    c1_p .. c3_p  : feature accuracy (correct / responded) per colour staircase
    m1_p .. m3_p  : feature accuracy per motion staircase
    c1_bp .. c3_bp: binding-error rate (binding errors / all trials) per colour staircase
    m1_bp .. m3_bp: binding-error rate per motion staircase

    Staircases are inferred from trial_paper + cue columns:
        cue=='c', trial_paper starts with '1' -> c1
        cue=='c', trial_paper starts with '2' -> c2
        cue=='c', trial_paper starts with '3' -> c3
        cue=='m', trial_paper starts with '1' -> m1  ... etc.
    (Matches the staircase numbering in Input_conditions.xlsx.)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute group-level behavioural results from per-trial TSV files "
            "produced by the BindingWM pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", required=True, type=Path,
        metavar="PATH",
        help=(
            "Root of the behav_data output tree, i.e. the directory that "
            "contains sub-XX/ folders with timing/ses-YY/trials/ inside."
        ),
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        metavar="PATH",
        help="Directory where result files (xlsx, csv, pdf) will be written.",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        metavar="XX",
        help=(
            "Zero-padded subject labels to process, e.g. '01 02 03'. "
            "If omitted, all sub-XX folders found under --data_dir are used."
        ),
    )
    parser.add_argument(
        "--sessions", nargs="+", default=["01", "02", "03", "04"],
        metavar="YY",
        help="Session labels to process (default: 01 02 03 04).",
    )
    parser.add_argument(
        "--n_runs", type=int, default=8,
        metavar="N",
        help="Number of binding runs per session (default: 8).",
    )
    return parser.parse_args()


# ============================================================================
# Subject discovery
# ============================================================================
def discover_subjects(data_dir: Path) -> list[str]:
    """Return sorted list of subject labels found as sub-XX folders."""
    pattern = re.compile(r"^sub-(\d+)$")
    subjects = []
    for p in sorted(data_dir.iterdir()):
        m = pattern.match(p.name)
        if m and p.is_dir():
            subjects.append(m.group(1))
    return subjects


# ============================================================================
# Data loading
# ============================================================================
def load_subject_session_trials(data_dir: Path,
                                 sub: str,
                                 session: str,
                                 n_runs: int) -> pd.DataFrame | None:
    """Load and concatenate all run-level trial TSVs for one sub x session."""
    trials_dir = data_dir / f"sub-{sub}" / "timing" / f"ses-{session}" / "trials"
    if not trials_dir.exists():
        return None

    dfs = []
    for run in range(1, n_runs + 1):
        fname = f"sub-{sub}_ses-{session}_task-bwm_run-{run:02d}_trials.tsv"
        p = trials_dir / fname
        if p.exists():
            dfs.append(pd.read_csv(p, sep="\t"))

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# ============================================================================
# Staircase mask helper
# ============================================================================
def staircase_mask(trials: pd.DataFrame, cue_type: str, number: str) -> pd.Series:
    """
    Return boolean mask for trials belonging to one staircase.

    Staircase identity is determined by:
        cue column == cue_type  ('c' or 'm')
        trial_paper starts with number  ('1', '2', or '3')

    This matches the logic of Input_conditions.xlsx where staircases are
    named c1/c2/c3/m1/m2/m3.
    """
    return (trials["cue"].str.lower() == cue_type) & \
           trials["trial_paper"].astype(str).str.startswith(number)


# ============================================================================
# Proportion computation
# ============================================================================
def compute_proportions(trials: pd.DataFrame,
                        exclude_no_response: bool) -> dict:
    """
    Compute error-category proportions for a set of trials.

    Parameters
    ----------
    trials : pd.DataFrame
        All trials for one subject x session (all runs concatenated).
    exclude_no_response : bool
        If True, no-response trials (error_code == 3) are excluded from the
        denominator used for all proportions except no_response_p itself.
        If False, denominator is all trials.

    Returns
    -------
    dict with proportion keys (see module docstring for full list).
    """
    n_total   = len(trials)
    nr_mask   = trials["error_code"] == 3
    n_no_resp = int(nr_mask.sum())

    base  = trials[~nr_mask].copy() if exclude_no_response else trials.copy()
    denom = len(base)
    if denom == 0:
        return {}

    n_correct = int((base["error_code"] == 0).sum())
    n_feature = int((base["error_code"] == 1).sum())
    n_fm      = int(((base["error_code"] == 2) &
                     (base["misbinding_subtype"] == "misbinding_fm")).sum())
    n_fnm     = int(((base["error_code"] == 2) &
                     (base["misbinding_subtype"] == "misbinding_fnm")).sum())

    result = {
        "n_total":          n_total,
        "n_answered":       n_total - n_no_resp,
        "n_correct":        n_correct,
        "n_feature":        n_feature,
        "n_misbinding_fm":  n_fm,
        "n_misbinding_fnm": n_fnm,
        "n_no_response":    n_no_resp,
        # proportions (out of denom)
        "correct_p":        round(n_correct / denom * 100, 2),
        "feature_p":        round(n_feature / denom * 100, 2),
        "misbinding_fm_p":  round(n_fm      / denom * 100, 2),
        "misbinding_fnm_p": round(n_fnm     / denom * 100, 2),
        # no_response always out of n_total regardless of exclude flag
        "no_response_p":    round(n_no_resp / n_total * 100, 2),
        "all_misbinding_p": round((n_fm + n_fnm) / denom * 100, 2),
    }

    # ---- per-staircase proportions -----------------------------------------
    for cue_type, num, label in [
        ("c", "1", "c1"), ("c", "2", "c2"), ("c", "3", "c3"),
        ("m", "1", "m1"), ("m", "2", "m2"), ("m", "3", "m3"),
    ]:
        sc_mask_all  = staircase_mask(trials, cue_type, num)
        sc_mask_base = staircase_mask(base,   cue_type, num)

        sc_all       = trials[sc_mask_all]
        sc_base      = base[sc_mask_base]
        sc_responded = sc_all[sc_all["error_code"] != 3]

        n_feat_denom = len(sc_responded)    # feature accuracy denominator: responded only
        n_sc_correct = int((sc_base["error_code"] == 0).sum())
        n_sc_binding = int((sc_base["error_code"] == 2).sum())
        n_sc_base    = len(sc_base)

        result[f"{label}_p"]  = (round(n_sc_correct / n_feat_denom * 100, 2)
                                  if n_feat_denom > 0 else np.nan)
        result[f"{label}_bp"] = (round(n_sc_binding / n_sc_base * 100, 2)
                                  if n_sc_base > 0 else np.nan)

    return result


# ============================================================================
# Analysis driver (one pass per exclude_no_response setting)
# ============================================================================
def run_analysis(subjects: list[str],
                 sessions: list[str],
                 data_dir: Path,
                 output_dir: Path,
                 n_runs: int,
                 exclude_no_response: bool) -> pd.DataFrame:
    tag   = "excl_nr" if exclude_no_response else "all"
    label = "no-response EXCLUDED" if exclude_no_response else "ALL trials (incl. no-response)"
    print(f"\n{'='*64}")
    print(f"Analysis: {label}")
    print(f"{'='*64}")

    rows = []
    for sub in subjects:
        for session in sessions:
            trials = load_subject_session_trials(data_dir, sub, session, n_runs)
            if trials is None or len(trials) == 0:
                print(f"  [skip] sub-{sub}, ses-{session}: no trial files found")
                continue

            props = compute_proportions(trials, exclude_no_response)
            if not props:
                print(f"  [skip] sub-{sub}, ses-{session}: empty after filtering")
                continue

            rows.append({"subject": sub, "session": session, **props})
            print(f"  sub-{sub} ses-{session}: "
                  f"correct={props['correct_p']:.2f}%  "
                  f"feat={props['feature_p']:.2f}%  "
                  f"bind_fm={props['misbinding_fm_p']:.2f}%  "
                  f"bind_fnm={props['misbinding_fnm_p']:.2f}%  "
                  f"no_resp={props['no_response_p']:.2f}%")

    if not rows:
        print("  No data found — check --data_dir and --subjects.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ---- append per-session group averages ---------------------------------
    all_rows = rows.copy()
    for ses in sessions:
        sub_df = df[df["session"] == ses]
        if sub_df.empty:
            continue
        avg_row = {"subject": f"avg_{ses}", "session": f"avg_{ses}"}
        for c in numeric_cols:
            avg_row[c] = round(sub_df[c].mean(), 2)
        all_rows.append(avg_row)

    df_full = pd.DataFrame(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = output_dir / f"behavresults_summary_{tag}.xlsx"
    df_full.to_excel(xlsx_path, index=False)
    print(f"\n  Saved summary      -> {xlsx_path}")

    # ---- group descriptive statistics --------------------------------------
    df_subs  = df_full[~df_full["subject"].astype(str).str.startswith("avg")]
    stat_cats = ["correct_p", "feature_p", "misbinding_fm_p", "misbinding_fnm_p",
                 "no_response_p", "all_misbinding_p"]
    stat_rows = []
    for cat in stat_cats:
        if cat not in df_subs.columns:
            continue
        vals = df_subs[cat].dropna()
        stat_rows.append({
            "category": cat,
            "mean":     round(vals.mean(),   2),
            "std":      round(vals.std(),    2),
            "sem":      round(vals.sem(),    2),
            "median":   round(vals.median(), 2),
            "min":      round(vals.min(),    2),
            "max":      round(vals.max(),    2),
            "n":        len(vals),
        })

    stats_df = pd.DataFrame(stat_rows).set_index("category")
    csv_path = output_dir / f"behavresults_summary_stats_{tag}.csv"
    stats_df.to_csv(csv_path)
    print(f"  Saved group stats  -> {csv_path}")
    print(f"\n  Descriptives ({tag}):\n{stats_df.to_string()}")

    # ---- pie chart ---------------------------------------------------------
    pie_cats   = ["correct_p", "feature_p", "misbinding_fm_p", "misbinding_fnm_p"]
    pie_labels = [
        "Correct binding",
        "Feature error (correct binding)",
        "Binding error - feature match",
        "Binding error - feature nonmatch",
    ]
    if not exclude_no_response:
        pie_cats.append("no_response_p")
        pie_labels.append("No response")

    pie_vals = [stats_df.loc[c, "mean"] if c in stats_df.index else 0.0
                for c in pie_cats]
    colors   = ["#00a67d", "#00a6c3", "#dc7562", "#d96ea6", "#b0b0b0"]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%",
           startangle=90, colors=colors[:len(pie_cats)])
    ax.axis("equal")
    suffix = "(no-response excluded)" if exclude_no_response else "(all trials)"
    ax.set_title(
        f"Response proportions - mean across participants {suffix}",
        fontsize=13, pad=20,
    )
    plt.tight_layout()
    pdf_path = output_dir / f"behavresults_pie_{tag}.pdf"
    plt.savefig(pdf_path)
    plt.close()
    print(f"  Saved pie chart    -> {pdf_path}")

    return df_full


# ============================================================================
# Entry point
# ============================================================================
def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: --data_dir does not exist: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    subjects = args.subjects
    if subjects is None:
        subjects = discover_subjects(args.data_dir)
        if not subjects:
            print(f"ERROR: no sub-XX folders found under {args.data_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-discovered {len(subjects)} subjects: {subjects}")

    print("=" * 64)
    print("Behavioural analysis from per-trial TSV files")
    print("=" * 64)
    print(f"  data_dir   : {args.data_dir}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  subjects   : {len(subjects)} ({subjects[0]} .. {subjects[-1]})")
    print(f"  sessions   : {args.sessions}")
    print(f"  n_runs     : {args.n_runs}")

    run_analysis(subjects, args.sessions, args.data_dir, args.output_dir,
                 args.n_runs, exclude_no_response=False)
    run_analysis(subjects, args.sessions, args.data_dir, args.output_dir,
                 args.n_runs, exclude_no_response=True)

    print("\nDone.")


if __name__ == "__main__":
    main()