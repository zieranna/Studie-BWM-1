#!/usr/bin/env python3
"""
07_create_responsive_rois.py
============================

ROIs after Rademaker, Chunharas & Serences (2019, Nat Neurosci): within each
anatomical ROI, ALL voxels that respond significantly to the stimuli -- no
fixed voxel count.

Two sources for "responsive":
  --map-pattern  any per-subject statistic map in TASK numbering, {sub} = subject,
                 e.g. the encoding > baseline t-map of the full FIR model:
                 .../firmodelcomplete/sub-{sub}/model/fir/binding/full/contrast/sub-{sub}_enc_t.nii.gz
                 (label-independent: does not use the decoded conditions)
  --localizer-dir + --mapping   the perception localizer (06b), described below
For a t-map give --stat t and, if known, --df (otherwise t is treated as z;
practically identical for large df).

Localizer: the perception localizer runs (06b_localizer_mt.py), contrast
stimGTbaseline = motion + color + control > baseline, all sessions, z-map:
    <localizer-dir>/sub-FF/sub-FF_task-perception_contrast-stimGTbaseline_stat-z.nii.gz
    (final numbering FF; mapped to task numbering via subject_mapping.csv)

Per subject:
    1. one-sided p of every brain voxel of the z-map (stimulus > baseline),
       Benjamini-Hochberg FDR across the brain at --q (default .05), as in
       Rademaker et al. ("p = 0.05; FDR corrected")
    2. the z-map is resampled onto the grid of each anatomical ROI mask
       (<roi-dir>/sub-TT/<ROI>_resampled.nii.gz, else <ROI>.nii.gz) and the ROI
       is restricted to voxels above the FDR threshold
    3. written as <out-dir>/sub-TT/<ROI>.nii.gz (same grid as the ROI mask)

Rademaker et al. used an ROI only if it had more than 20 responsive voxels
in every participant; the summary flags subjects/ROIs below --min-voxels.

Output:
    <out-dir>/sub-TT/<ROI>.nii.gz
    <out-dir>/responsive_voxels.csv     subject x ROI: ROI size, responsive voxels, share

Usage with the encoding maps:
    python 07_create_responsive_rois.py --stat t \
        --map-pattern '.../firmodelcomplete/sub-{sub}/model/fir/binding/full/contrast/sub-{sub}_enc_t.nii.gz' \
        --roi-dir .../roi_anat --out-dir .../roi_responsive_enc

Usage with the localizer (on the HPC, where roi_anat is):
    python 07_create_responsive_rois.py \\
        --localizer-dir .../localizer_maps --mapping .../subject_mapping.csv \\
        --roi-dir .../roi_anat --out-dir .../roi_responsive
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from scipy.stats import norm

ROIS = ["V1", "V2", "V3", "V4", "early_visual", "pari", "dlpfc"]
_KW = {k: True for k in ("force_resample", "copy_header")
       if k in inspect.signature(resample_to_img).parameters}


def fdr_threshold(v: np.ndarray, q: float, stat: str = "z", df=None) -> float:
    """Statistic threshold of a one-sided Benjamini-Hochberg FDR at q (inf if
    none). stat 't' with df uses the t distribution, otherwise the normal."""
    from scipy.stats import t as tdist
    sf, isf = ((lambda x: tdist.sf(x, df), lambda x: tdist.isf(x, df))
               if stat == "t" and df else (norm.sf, norm.isf))
    p = np.sort(sf(v))
    m = p.size
    ok = np.flatnonzero(p <= q * np.arange(1, m + 1) / m)
    return float(isf(p[ok[-1]])) if ok.size else np.inf


def roi_mask_file(roi_dir: Path, sub: str, roi: str):
    d = roi_dir / f"sub-{sub}"
    for name in (f"{roi}_resampled.nii.gz", f"{roi}.nii.gz"):
        if (d / name).exists():
            return d / name
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Localizer-responsive voxels within anatomical ROIs.")
    p.add_argument("--map-pattern", default=None,
                   help="per-subject statistic map, task numbering, {sub} = subject, e.g. "
                        "'/path/firmodelcomplete/sub-{sub}/model/fir/binding/full/contrast/"
                        "sub-{sub}_enc_t.nii.gz'. Replaces --localizer-dir/--mapping.")
    p.add_argument("--stat", choices=["z", "t"], default="z",
                   help="statistic in the map (default z)")
    p.add_argument("--df", type=float, default=None,
                   help="degrees of freedom for --stat t (default: treat t as z)")
    p.add_argument("--localizer-dir", type=Path, default=None,
                   help="folder with sub-FF/sub-FF_task-perception_contrast-<c>_stat-z.nii.gz")
    p.add_argument("--mapping", type=Path, default=None,
                   help="subject_mapping.csv (behav_subject = task numbering, final_subject)")
    p.add_argument("--roi-dir", type=Path, required=True,
                   help="anatomical ROIs, flat: <dir>/sub-TT/<ROI>[_resampled].nii.gz (roi_anat)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rois", nargs="+", default=ROIS)
    p.add_argument("--subjects", nargs="+", default=None, help="task numbering (default: all)")
    p.add_argument("--contrast", default="stimGTbaseline")
    p.add_argument("--q", type=float, default=0.05, help="FDR level (default .05)")
    p.add_argument("--min-voxels", type=int, default=20,
                   help="flag ROIs with fewer responsive voxels (Rademaker et al.: 20)")
    args = p.parse_args()
    if args.map_pattern is None and (args.localizer_dir is None or args.mapping is None):
        p.error("give --map-pattern, or --localizer-dir and --mapping")
    if args.stat == "t" and args.df is None:
        print("note: --stat t without --df: t is treated as z (fine for large df)")

    t2f = {}
    if args.mapping is not None:
        m = pd.read_csv(args.mapping, dtype=str)
        t2f = {f"{int(r.behav_subject):02d}": f"{int(r.final_subject):02d}" for r in m.itertuples()}
    if args.subjects:
        subs = [f"{int(s):02d}" for s in args.subjects]
    elif args.map_pattern:
        subs = [f"{i:02d}" for i in range(1, 21)]
    else:
        subs = sorted(t2f)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ts in subs:
        if args.map_pattern:
            f = Path(args.map_pattern.format(sub=ts))
            hits, src = ([f] if f.exists() else []), f"sub-{ts}"
        else:
            fs = t2f.get(ts)
            hits = sorted(args.localizer_dir.rglob(
                f"sub-{fs}_task-perception_contrast-{args.contrast}_stat-z.nii.gz")) if fs else []
            src = f"final sub-{fs}"
        if not hits:
            print(f"task sub-{ts}: no statistic map ({src}) -- skipped")
            rows += [{"subject": ts, "ROI": r, "note": "no map"} for r in args.rois]
            continue
        z_img = nib.load(str(hits[0]))
        z = z_img.get_fdata()
        brain = np.isfinite(z) & (z != 0)
        z_thr = fdr_threshold(z[brain], args.q, args.stat, args.df)
        n_sig = int((z[brain] > z_thr).sum())
        print(f"task sub-{ts} ({hits[0].name}): FDR q={args.q} -> {args.stat} > {z_thr:.2f}, "
              f"{n_sig} of {int(brain.sum())} brain voxels responsive", flush=True)
        for roi in args.rois:
            mf = roi_mask_file(args.roi_dir, ts, roi)
            if mf is None:
                rows.append({"subject": ts, "ROI": roi, "note": "no ROI mask"})
                continue
            roi_img = nib.load(str(mf))
            roi_mask = np.asanyarray(roi_img.dataobj) > 0
            z_on = resample_to_img(z_img, roi_img, interpolation="continuous", **_KW).get_fdata()
            resp = roi_mask & (z_on > z_thr)
            out = args.out_dir / f"sub-{ts}" / f"{roi}.nii.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.Nifti1Image(resp.astype(np.uint8), roi_img.affine).to_filename(str(out))
            rows.append({"subject": ts, "ROI": roi, "roi_voxels": int(roi_mask.sum()),
                         "responsive_voxels": int(resp.sum()),
                         "share": round(float(resp.sum() / max(roi_mask.sum(), 1)), 3),
                         "z_threshold": round(z_thr, 3), "roi_file": mf.name, "note": ""})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "responsive_voxels.csv", index=False)
    if "responsive_voxels" not in df:
        print("No ROIs written.")
        return
    tab = df.pivot_table(index="subject", columns="ROI", values="responsive_voxels",
                         aggfunc="first").reindex(columns=[r for r in args.rois if r in df["ROI"].values])
    print("\nResponsive voxels per subject and ROI:")
    print(tab.fillna(-1).astype(int).replace(-1, "-").to_string())
    low = df[df["responsive_voxels"].fillna(0) < args.min_voxels]
    if len(low):
        print(f"\nBelow {args.min_voxels} voxels (Rademaker et al. would drop the ROI if "
              f"this happens in any participant):")
        for _, r in low.iterrows():
            print(f"   sub-{r['subject']} {r['ROI']}: {r.get('responsive_voxels', 0):.0f} "
                  f"{r.get('note', '')}")
    ok = [r for r in tab.columns if (tab[r].fillna(0) >= args.min_voxels).all()]
    print(f"\nROIs with >= {args.min_voxels} responsive voxels in every subject: {ok}")
    print(f"Masks: {args.out_dir}/sub-XX/<ROI>.nii.gz; table: {args.out_dir}/responsive_voxels.csv")


if __name__ == "__main__":
    main()