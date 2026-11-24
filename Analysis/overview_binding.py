#!/usr/bin/env python3
"""
overview_binding.py
===================

Finds every BINDING decoding result on this PC and summarizes it, sorted by
input data:

    FIR original     phase del/enc/res (firestimates, per-run FIR GLM)
    LSS FIR 17       fir-<stat>-<window> WITHOUT z-scoring (old lss/ folder)
    LSS FIR 25       fir-<stat>-<window>_zrun / _zall (lss_fir25/; only it has z maps)
    LSS HRF          hrf-...  (listed under "other")
The 17/25 assignment is inferred from the file names (see above).

Searched: <root>/<folder>/sub-XX/binding/decoding_binding_*.xlsx and one
level deeper (<root>/<folder>/<folder>/sub-XX/...), newest file per subject.

Per run and ROI: mean accuracy per subject (across pairs), then group mean,
SEM and a one-sided t-test against 0.5 -- a quick overview; the GLA3 results
(permutation test, FDR) remain the numbers to report.
Runs with several voxel counts (old fixed mode, allcounts): additionally the
curve over voxel counts.

Output: <root>/overview_binding_<date>.xlsx (sheets runs, summary, curves)
        and a compact table in the terminal.

Usage:
    python3 overview_binding.py                       # default root below
    python3 overview_binding.py --root /other/folder
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/annaz/BindingWM/Studie-BWM-1_CP_revision1")
ROI_ORDER = ["V1", "V2", "V3", "V4", "early_visual", "pari", "dlpfc", "dlpfc_noSFL",
             "visual_all", "visual_locMT", "MT_complex"]
CATEGORIES = ["FIR original", "LSS FIR 17 (inferred)", "LSS FIR 25 (inferred)", "LSS HRF", "other"]
NAME_RE = re.compile(r"^decoding_binding_(?P<rest>.+)_(?P<date>\d{8})\.xlsx$")
TAG_RE = re.compile(r"^(?P<phase>.+?)(?:_(?P<tag>(?:nested|allcounts|pcavar|pca|mask)\S*))?$")


def classify(phase: str) -> str:
    if phase in ("del", "enc", "res"):
        return "FIR original"
    if phase.startswith("fir-"):
        return "LSS FIR 25 (inferred)" if re.search(r"_z(run|all)$", phase) else "LSS FIR 17 (inferred)"
    if phase.startswith("hrf-"):
        return "LSS HRF"
    return "other"


def describe(tag: str) -> tuple:
    """(method, C) from the run name."""
    if tag == "fixed":
        return "fixed: every voxel count (nilearn Decoder)", "tuned"
    c = "1" if "_C1" in tag else "tuned"
    if tag.startswith("nested"):
        m = "nested top-N"
    elif tag.startswith("allcounts"):
        m = "every voxel count (no selection)"
    elif tag.startswith("pcavar"):
        m = "PCA, variance fractions"
    elif tag.startswith("pca"):
        m = "PCA, fixed components"
    elif tag.startswith("mask"):
        m = "mask " + tag.split("_C")[0].replace("mask-", "")
    else:
        m = tag
    return m, c


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Overview of all binding decoding results.")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()

    # ---- find files: newest per (run folder, subject, phase, tag) ----------------
    files = {}
    for pattern in ("*/sub-*/binding/decoding_binding_*.xlsx",
                    "*/*/sub-*/binding/decoding_binding_*.xlsx"):
        for f in sorted(a.root.glob(pattern)):
            m = NAME_RE.match(f.name)
            if not m:
                continue
            t = TAG_RE.match(m.group("rest"))
            phase, tag = t.group("phase"), t.group("tag") or "fixed"
            run_dir = f.parents[2]
            key = (run_dir, f.parents[1].name, phase, tag)
            files[key] = f                                   # sorted -> newest date last
    if not files:
        raise SystemExit(f"No binding results found under {a.root}")

    runs = defaultdict(dict)
    for (run_dir, sub, phase, tag), f in files.items():
        runs[(run_dir, phase, tag)][sub] = f
    print(f"{len(files)} files in {len(runs)} runs under {a.root}\n")

    run_rows, summ_rows, curve_rows = [], [], []
    for (run_dir, phase, tag), subs in sorted(runs.items(), key=lambda x: (str(x[0][0]), x[0][1], x[0][2])):
        cat = classify(phase)
        method, c = describe(tag)
        folder = str(run_dir.relative_to(a.root))
        per_sub = []
        for sub, f in sorted(subs.items()):
            try:
                pp = pd.read_excel(f, sheet_name="per_pair")
            except Exception as e:
                print(f"  [skip] {f}: {e}")
                continue
            if "voxels" not in pp or pp["voxels"].astype(str).isin(["nested", "pca", "mask"]).all() \
                    or tag.startswith(("nested", "pca", "mask")):
                pp = pp.assign(voxels="selected")
            g = pp.groupby(["ROI", "voxels"])["accuracy"].mean().reset_index()
            g["subject"] = sub
            per_sub.append(g)
        if not per_sub:
            continue
        df = pd.concat(per_sub, ignore_index=True)
        multi = df["voxels"].nunique() > 1
        counts = sorted(df["voxels"].unique(), key=lambda v: (isinstance(v, str), v))
        run_rows.append({"category": cat, "folder": folder, "phase": phase, "run": tag,
                         "method": method, "C": c, "n_subjects": df["subject"].nunique(),
                         "voxel_counts": (f"{len(counts)} ({counts[0]}-{counts[-1]})" if multi
                                          else "selected per fold" if tag.startswith("nested")
                                          else "-"),
                         "rois": ", ".join(r for r in ROI_ORDER if r in df["ROI"].unique())})
        for (roi, vox), g in df.groupby(["ROI", "voxels"]):
            v = g["accuracy"].to_numpy()
            t, p2 = stats.ttest_1samp(v, 0.5) if len(v) > 1 else (np.nan, np.nan)
            p = p2 / 2 if t > 0 else 1 - p2 / 2
            row = {"category": cat, "folder": folder, "phase": phase, "run": tag, "ROI": roi,
                   "voxels": vox, "n": len(v), "mean": v.mean(),
                   "sem": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                   "t": t, "p_one_sided": p}
            curve_rows.append(row) if multi else summ_rows.append(row)
        if multi:   # one summary row per ROI for multi-count runs
            cur = pd.DataFrame([r for r in curve_rows if r["folder"] == folder and r["run"] == tag
                                and r["phase"] == phase])
            for roi, g in cur.groupby("ROI"):
                best = g.loc[g["mean"].idxmax()]
                summ_rows.append({"category": cat, "folder": folder, "phase": phase, "run": tag,
                                  "ROI": roi, "voxels": f"mean over {len(g)} counts",
                                  "n": int(g["n"].max()), "mean": g["mean"].mean(),
                                  "sem": np.nan, "t": np.nan, "p_one_sided": np.nan,
                                  "best_count": best["voxels"], "best_mean": best["mean"],
                                  "counts_p<.05": int((g["p_one_sided"] < .05).sum())})

    runs_df = pd.DataFrame(run_rows)
    summ = pd.DataFrame(summ_rows)
    curves = pd.DataFrame(curve_rows)
    roi_rank = {r: i for i, r in enumerate(ROI_ORDER)}
    cat_rank = {c: i for i, c in enumerate(CATEGORIES)}
    for d in (runs_df, summ, curves):
        if len(d):
            d.sort_values([c for c in ["category", "folder", "phase", "run", "ROI"] if c in d],
                          key=lambda s: s.map(cat_rank) if s.name == "category"
                          else s.map(roi_rank) if s.name == "ROI" else s, inplace=True)

    # ---- terminal overview -----------------------------------------------------------
    for cat in CATEGORIES:
        rr = runs_df[runs_df["category"] == cat]
        if rr.empty:
            continue
        print("=" * 100 + f"\n{cat}\n" + "=" * 100)
        for i, r in enumerate(rr.itertuples(), start=1):
            print(f"  [{i}] {r.folder} | {r.phase} | {r.method} | C={r.C} | voxels: {r.voxel_counts} "
                  f"| n={r.n_subjects}")
        tab = {}
        for i, r in enumerate(rr.itertuples(), start=1):
            s = summ[(summ["folder"] == r.folder) & (summ["phase"] == r.phase) & (summ["run"] == r.run)]
            for _, row in s.iterrows():
                if isinstance(row["voxels"], str) and row["voxels"].startswith("mean over"):
                    txt = (f"{row['mean']:.3f}~ best {row['best_mean']:.3f}@{int(row['best_count'])} "
                           f"({int(row['counts_p<.05'])} sig.)")
                else:
                    txt = f"{row['mean']:.3f}{star(row['p_one_sided'])}"
                tab.setdefault(row["ROI"], {})[f"[{i}]"] = txt
        t = pd.DataFrame(tab).T.reindex([r for r in ROI_ORDER if r in tab])
        print(t.fillna("").to_string())
        print()
    print("Mean accuracy per ROI (chance 0.5); * p<.05, ** p<.01, *** p<.001 (one-sided t-test,")
    print("uncorrected, quick overview). '~' = mean over all voxel counts, 'best X@N' = best count,")
    print("'(k sig.)' = number of voxel counts with p<.05.")

    out = a.root / f"overview_binding_{time.strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(out) as xl:
        runs_df.to_excel(xl, sheet_name="runs", index=False)
        summ.round(4).to_excel(xl, sheet_name="summary", index=False)
        if len(curves):
            curves.round(4).to_excel(xl, sheet_name="curves", index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()