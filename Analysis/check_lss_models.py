#!/usr/bin/env python3
"""
check_lss_models.py
===================

Label-free check that the LSS FIR estimates contain no data error (it does not
look at any decoding result):

1. Design QC (from the QC tables of 02b_trialestimates_lss.py), per LSS run:
       n_columns          confirms the number of FIR delays (17: ~50, 25: ~66)
       residual_df, condition_number, vif_del_max (precision of the delay window)
       assign_ok          the run's own events fit its BOLD run best
       motion, signal_scaling
2. Map agreement: for every trial, the correlation of its delay t map
   (delays 9-12) between FIR original (LSA), LSS 17 and LSS 25, inside the ROI
   voxels (union of the largest top-N masks of roi_func). All three estimate
   the same trial from the same data, so the maps must agree; a run where one
   model deviates clearly from the other two points to a data error there
   (wrong events, wrong BOLD file, ...).

Output: <out>_design.csv, <out>_maps.csv (per trial), <out>_runs.csv (per run)
and a summary with flagged runs.

    python check_lss_models.py --project /mnt/hpc/.../bindingwm-study1 --out qc/check_lss
"""

from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

ROIS = ["V1", "V2", "V3", "V4", "early_visual", "pari", "dlpfc"]
_KW = {k: True for k in ("force_resample", "copy_header")
       if k in inspect.signature(resample_to_img).parameters}
KEY = re.compile(r"_ses-(\d+)_(?:block-\d+_)?run-(\d+)_trial-(\w+)\.nii")


def trial_maps(project, model, sub):
    """{(ses, run, trial): path} of the delay t maps of one model."""
    if model == "orig":
        d = project / "firestimates" / f"sub-{sub}" / "model" / "fir" / "binding" / "tmap" / "del"
    elif model == "lss17":
        d = project / "lss" / f"sub-{sub}" / "fir" / "tmap" / "del" / "trials"
    else:
        d = project / "lss_fir25" / f"sub-{sub}" / "fir" / "tmap" / "del" / "trials"
    out = {}
    for f in d.glob(f"sub-{sub}_*.nii.gz") if d.exists() else []:
        m = KEY.search(f.name)
        if m:
            out[m.groups()] = f
    return out


def roi_index(project, sub, ref):
    union = None
    for roi in ROIS:
        files = sorted((project / "roi_func" / f"sub-{sub}" / roi).glob("top[0-9][0-9][0-9][0-9].nii.gz"))
        if files:
            m = np.asanyarray(resample_to_img(nib.load(str(files[-1])), ref,
                                              interpolation="nearest", **_KW).dataobj) > 0
            union = m if union is None else union | m
    return np.flatnonzero(union.reshape(-1)) if union is not None else None


def design_qc(project):
    rows = []
    for model, root in (("lss17", "lss"), ("lss25", "lss_fir25")):
        for f in sorted((project / root / "qc").glob("sub-*/sub-*_fir_design.csv")):
            q = pd.read_csv(f)
            q["model"] = model
            rows.append(q)
    if not rows:
        return pd.DataFrame()
    q = pd.concat(rows, ignore_index=True)
    q["subject"] = q["subject"].astype(str).str.zfill(2)
    agg = {c: ("first" if c in ("motion", "signal_scaling") else "median")
           for c in ("n_columns", "residual_df", "condition_number", "vif_del_max",
                     "motion", "signal_scaling") if c in q}
    if "assign_ok" in q:
        q["assign_ok"] = q["assign_ok"].astype(str).str.lower().isin(["true", "1"])
        agg["assign_ok"] = "min"
    return (q.groupby(["model", "subject", "session", "run"]).agg(agg).reset_index())


def main():
    p = argparse.ArgumentParser(description="Label-free check of the LSS FIR estimates.")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 21)])
    p.add_argument("--out", type=Path, default=Path("check_lss"))
    a = p.parse_args()
    pd.set_option("display.width", 220)

    # ---- 1. design QC ------------------------------------------------------------
    dq = design_qc(a.project)
    dq.to_csv(f"{a.out}_design.csv", index=False)
    if len(dq):
        print("=== 1. design per LSS model (median over runs; assign_ok = share of runs ok)")
        s = dq.groupby("model").agg(
            runs=("run", "size"), subjects=("subject", "nunique"),
            n_columns=("n_columns", "median"), residual_df=("residual_df", "median"),
            cond_median=("condition_number", "median"), cond_max=("condition_number", "max"),
            vif_del_median=("vif_del_max", "median"), vif_del_max=("vif_del_max", "max"),
            assign_ok=("assign_ok", "mean") if "assign_ok" in dq else ("run", "size"),
            motion=("motion", "first"), scaling=("signal_scaling", "first"))
        print(s.round(2).to_string())
        if "assign_ok" in dq and (~dq["assign_ok"]).any():
            print("\nruns whose events do NOT fit their own BOLD run best:")
            print(dq.loc[~dq["assign_ok"], ["model", "subject", "session", "run"]].to_string(index=False))
    else:
        print("=== 1. no QC tables found under lss/qc or lss_fir25/qc")

    # ---- 2. map agreement ------------------------------------------------------------
    rows = []
    for sub in [f"{int(s):02d}" for s in a.subjects]:
        maps = {m: trial_maps(a.project, m, sub) for m in ("orig", "lss17", "lss25")}
        keys = sorted(set(maps["lss17"]) & set(maps["lss25"]))
        if not keys:
            print(f"sub-{sub}: no common LSS trials, skipped", flush=True)
            continue
        ref = nib.load(str(maps["lss17"][keys[0]]))
        idx = roi_index(a.project, sub, ref)
        for k in keys:
            v = {}
            for m in ("orig", "lss17", "lss25"):
                f = maps[m].get(k)
                if f is None:
                    continue
                img = nib.load(str(f))
                if img.shape[:3] != ref.shape[:3]:
                    img = resample_to_img(img, ref, interpolation="continuous", **_KW)
                v[m] = img.get_fdata(dtype=np.float32).reshape(-1)[idx]
            ok = np.ones(idx.size, bool)
            for x in v.values():
                ok &= np.isfinite(x) & (x != 0)
            row = {"subject": sub, "session": k[0], "run": k[1], "trial": k[2], "voxels": int(ok.sum())}
            for a_, b_ in (("orig", "lss17"), ("orig", "lss25"), ("lss17", "lss25")):
                if a_ in v and b_ in v and ok.sum() > 10:
                    row[f"r_{a_}_{b_}"] = float(np.corrcoef(v[a_][ok], v[b_][ok])[0, 1])
            rows.append(row)
        print(f"sub-{sub}: {len(keys)} trials compared", flush=True)
    mp = pd.DataFrame(rows)
    mp.to_csv(f"{a.out}_maps.csv", index=False)
    rcols = [c for c in mp.columns if c.startswith("r_")]
    runs = mp.groupby(["subject", "session", "run"])[rcols].median().reset_index()
    runs.to_csv(f"{a.out}_runs.csv", index=False)

    print("\n=== 2. agreement of the per-trial delay t maps (Pearson r in ROI voxels)")
    print("median over trials, per subject:")
    print(mp.groupby("subject")[rcols].median().round(2).T.to_string())
    print("\nover all trials: median and 5th percentile")
    print(pd.DataFrame({"median": mp[rcols].median(), "p05": mp[rcols].quantile(.05)}).round(3).to_string())

    # flag runs: one model clearly off relative to the others in that run
    flags = []
    if "r_lss17_lss25" in runs:
        thr = runs["r_lss17_lss25"].median() - 3 * (runs["r_lss17_lss25"].quantile(.75)
                                                   - runs["r_lss17_lss25"].quantile(.25))
        for _, r in runs.iterrows():
            why = []
            if r["r_lss17_lss25"] < thr:
                why.append(f"LSS17 vs LSS25 r = {r['r_lss17_lss25']:.2f}")
            for c in ("r_orig_lss17", "r_orig_lss25"):
                if c in r and pd.notna(r[c]) and r[c] < runs[c].median() - 3 * (runs[c].quantile(.75) - runs[c].quantile(.25)):
                    why.append(f"{c[2:]} r = {r[c]:.2f}")
            if why:
                flags.append({"subject": r["subject"], "session": r["session"], "run": r["run"],
                              "why": "; ".join(why)})
    print(f"\n{len(flags)} runs with clearly deviating maps (more than 3 IQR below the median)"
          + ("" if not flags else ":\n" + pd.DataFrame(flags).to_string(index=False)))
    print(f"\nTables: {a.out}_design.csv, {a.out}_maps.csv, {a.out}_runs.csv")


if __name__ == "__main__":
    main()