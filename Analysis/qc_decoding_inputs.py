#!/usr/bin/env python3
"""
qc_decoding_inputs.py
=====================

Checks the binding pair maps that the decoding grid reads, per input and
subject, INSIDE the voxels the decoding uses (union of the largest top-N
masks of every ROI, resampled to the map grid as 07_decoding_pca.py does):

    files      number of pair maps (288 = 18 pairs x 16 blocks), labels (18),
               blocks per label (16)
    grid       every map on the grid of the first one
    nan_pct    % NaN/inf values (07_decoding_pca.py silently sets them to 0)
    zero_pct   % exact zeros (e.g. voxels outside the fitting mask)
    dead_vox   ROI voxels that are 0/NaN in EVERY map (carry no information)
    empty_maps maps that are 0/NaN in all ROI voxels
    dup_maps   maps identical to another map (copy errors)
    mean, sd   value distribution (z-scored then pair-averaged maps: mean ~0,
               sd somewhat below 1; plain t maps: larger, mean > 0 in active voxels)
Per ROI: share of the largest top-N mask with data (coverage).

Output: <out>.csv (one row per input x subject), <out>_roi.csv (coverage per
ROI) and a summary in the log; rows with a problem are marked in 'flags'.

Usage (HPC, as a job, see the message):
    python qc_decoding_inputs.py --project /mnt/hpc/.../bindingwm-study1 --out qc_inputs
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

INPUTS = [  # label, folder, layout, phase
    # FIR original: tmap_pair = z-scored per run (03_postprocess_tmaps.py), then pairs
    ("fir_orig", "firestimates", "fir", "del"),
    # LSS, z-scored per run with the sample SD, then pairs
    # (03_average.py --zscore run --ddof 1): processed exactly as fir_orig
    ("lss17_z", "lss", "lss", "fir/tmap/del_zrun_ddof1"),
    ("lss25_z", "lss_fir25", "lss", "fir/tmap/del_zrun_ddof1"),
    ("hrf_z", "lss", "lss", "hrf/tmap/del_zrun_ddof1"),
    # LSS, plain pairs (no z-scoring): the first grid run
    ("lss17", "lss", "lss", "fir/tmap/del"),
    ("lss25", "lss_fir25", "lss", "fir/tmap/del"),
    ("hrf", "lss", "lss", "hrf/tmap/del"),
]
ROIS = ["V1", "V2", "V3", "V4", "early_visual", "pari", "dlpfc"]
_KW = {k: True for k in ("force_resample", "copy_header")
       if k in inspect.signature(resample_to_img).parameters}
NAME = re.compile(r"_ses-(\d+)_block-(\d+)_.*pair-(\w+)\.nii(\.gz)?$")


def map_dir(project, folder, layout, phase, sub):
    if layout == "fir":
        return project / folder / f"sub-{sub}" / "model" / "fir" / "binding" / "tmap_pair" / phase
    return project / folder / f"sub-{sub}" / phase / "pairs"


def largest_mask(roi_dir, sub, roi):
    files = sorted((roi_dir / f"sub-{sub}" / roi).glob("top[0-9][0-9][0-9][0-9].nii.gz"))
    return files[-1] if files else None


def check(project, roi_dir, sub, inp):
    label, folder, layout, phase = inp
    d = map_dir(project, folder, layout, phase, sub)
    files = sorted(d.glob(f"sub-{sub}_ses-*_block-*_pair-*.nii.gz")) if d.exists() else []
    row = {"input": label, "subject": sub, "n_files": len(files)}
    roi_rows = []
    if not files:
        row["flags"] = "NO FILES"
        return row, roi_rows
    meta = [NAME.search(f.name).groups()[:3] for f in files]
    labs = pd.Series([m[2] for m in meta])
    row["n_labels"] = labs.nunique()
    row["min_blocks_per_label"] = int(labs.value_counts().min())
    ref = nib.load(str(files[0]))

    masks = {}
    for roi in ROIS:
        mf = largest_mask(roi_dir, sub, roi)
        if mf is not None:
            rs = resample_to_img(nib.load(str(mf)), ref, interpolation="nearest", **_KW)
            masks[roi] = np.asanyarray(rs.dataobj).reshape(-1) > 0
    union = np.zeros(int(np.prod(ref.shape[:3])), bool)
    for m in masks.values():
        union |= m
    idx = np.flatnonzero(union)

    X = np.empty((len(files), idx.size), np.float32)
    off_grid, hashes = 0, []
    for i, f in enumerate(files):
        img = nib.load(str(f))
        if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=1e-3):
            off_grid += 1
            X[i] = np.nan
            continue
        v = img.get_fdata(dtype=np.float32).reshape(-1)[idx]
        X[i] = v
        hashes.append(hashlib.md5(np.round(v, 5).tobytes()).hexdigest())
    bad = ~np.isfinite(X)
    zero = (X == 0)
    empty = bad | zero
    dead = empty.all(axis=0)
    fin = X[np.isfinite(X) & ~zero]
    row.update({
        "off_grid_maps": off_grid, "roi_voxels": int(idx.size),
        "nan_pct": round(100 * bad.mean(), 3), "zero_pct": round(100 * zero.mean(), 3),
        "dead_vox": int(dead.sum()), "empty_maps": int(empty.all(axis=1).sum()),
        "dup_maps": len(hashes) - len(set(hashes)),
        "mean": round(float(fin.mean()), 3) if fin.size else np.nan,
        "sd": round(float(fin.std()), 3) if fin.size else np.nan,
        "p01": round(float(np.percentile(fin, 1)), 2) if fin.size else np.nan,
        "p99": round(float(np.percentile(fin, 99)), 2) if fin.size else np.nan,
    })
    pos = {v: k for k, v in enumerate(idx)}
    for roi, m in masks.items():
        cols = [pos[v] for v in np.flatnonzero(m)]
        roi_rows.append({"input": label, "subject": sub, "ROI": roi, "voxels": len(cols),
                         "coverage_pct": round(100 * (1 - dead[cols].mean()), 2) if cols else np.nan})
    flags = []
    if len(files) != 288: flags.append(f"{len(files)} files")
    if row["n_labels"] != 18: flags.append(f"{row['n_labels']} labels")
    if row["min_blocks_per_label"] < 16: flags.append("label with <16 blocks")
    if off_grid: flags.append(f"{off_grid} maps off grid")
    if row["nan_pct"] > 0: flags.append(f"NaN {row['nan_pct']}%")
    if row["zero_pct"] > 1: flags.append(f"zeros {row['zero_pct']}%")
    if row["dead_vox"]: flags.append(f"{row['dead_vox']} dead ROI voxels")
    if row["empty_maps"]: flags.append(f"{row['empty_maps']} empty maps")
    if row["dup_maps"]: flags.append(f"{row['dup_maps']} duplicate maps")
    row["flags"] = "; ".join(flags) if flags else "ok"
    return row, roi_rows


def main():
    p = argparse.ArgumentParser(description="QC of the decoding input maps.")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--roi-dir", type=Path, default=None, help="default <project>/roi_func")
    p.add_argument("--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 21)])
    p.add_argument("--out", type=Path, default=Path("qc_inputs"))
    a = p.parse_args()
    roi_dir = a.roi_dir or a.project / "roi_func"
    rows, roi_rows = [], []
    for sub in [f"{int(s):02d}" for s in a.subjects]:
        for inp in INPUTS:
            r, rr = check(a.project, roi_dir, sub, inp)
            rows.append(r); roi_rows += rr
            print(f"sub-{sub} {inp[0]:<9} {r['flags']}", flush=True)
    df, roi = pd.DataFrame(rows), pd.DataFrame(roi_rows)
    df.to_csv(f"{a.out}.csv", index=False)
    roi.to_csv(f"{a.out}_roi.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== summary per input (over subjects)")
    agg = df.groupby("input", sort=False).agg(
        subjects_ok=("flags", lambda f: int((f == "ok").sum())), files_min=("n_files", "min"),
        nan_pct_max=("nan_pct", "max"), zero_pct_max=("zero_pct", "max"),
        dead_vox_max=("dead_vox", "max"), dup_maps_max=("dup_maps", "max"),
        mean=("mean", "mean"), sd=("sd", "mean"))
    print(agg.round(3).to_string())
    if len(roi):
        print("\n=== lowest coverage per input and ROI (% of the largest top-N mask with data)")
        print(roi.groupby(["input", "ROI"], sort=False)["coverage_pct"].min()
              .unstack().reindex(columns=ROIS).round(1).to_string())
    probs = df[df["flags"] != "ok"]
    print(f"\n{len(probs)} of {len(df)} input x subject rows with a problem" +
          ("" if probs.empty else ":\n" + probs[["input", "subject", "flags"]].to_string(index=False)))
    print(f"\nTables: {a.out}.csv, {a.out}_roi.csv")


if __name__ == "__main__":
    main()