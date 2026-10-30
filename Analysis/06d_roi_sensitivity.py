#!/usr/bin/env python3
"""
06d_roi_sensitivity.py
======================

How does the MT+ ROI change if the search region is made larger?

Uses the saved motion>control z-map of 06b (no refitting) and the SAME
functions as 06b (imported from it), and recomputes the ROI for several
margins around the HCP-MMP1 MT+ parcels (--dilations, mm). Per margin and
hemisphere:
    voxels, peak z, verdict, share inside the MT+ parcels, ARI lower bound,
    and the HCP-MMP1 areas of the voxels OUTSIDE the MT+ parcels
    (where does the ROI grow into?).

Reading:
    grows a little, extra voxels in MT+ neighbours (TPOJ2/3, LO2/3, PH,
    MST border)                                  -> larger margin is fine
    jumps in size, extra voxels in V1-V3, V3A/B, V6, LO1, or the peak moves
                                                 -> it merges with another
                                                    area; keep the margin
Pick ONE margin for all subjects (a per-subject choice is not defensible).

Each variant is written (for viewing and as specification-curve variants):
    <REV>/roi_localizer_sensitivity/sub-YY/<name>_dil05.nii.gz, ...
--name separates search-region variants, e.g. --name MTbroad together with
--search-labels MT MST FST V4t LO1 LO2 LO3 PH V3CD (Glasser's broader section).

Usage:
    python 06d_roi_sensitivity.py --task-subjects 01 --dilations 0 5 8 10 12 \
        --parcellation-dir .../hpcmmp1_output
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# 06b is looked for next to this script (either name), then in the Analysis folder
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [_HERE / "06_ROI_localizer.py", _HERE / "06b_localizer_mt.py",
               Path("/home/annaz/PycharmProjects/Studie-BWM-1/Analysis/06_ROI_localizer.py")]
MASK_RE = re.compile(r"^([LR])_(.+)_ROI\.(nii\.gz|nii|mgz)$")


def load_06b():
    script = next((c for c in _CANDIDATES if c.exists()), None)
    if script is None:
        sys.exit("06b not found; looked for:\n  " + "\n  ".join(str(c) for c in _CANDIDATES))
    print(f"using {script}", flush=True)
    spec = importlib.util.spec_from_file_location("loc06b", script)
    m = importlib.util.module_from_spec(spec)
    sys.modules["loc06b"] = m
    spec.loader.exec_module(m)
    return m


def label_counts(points_by_key: dict, parc_dir: Path, task_sub: str) -> dict:
    """{key: Counter(HCP label -> number of points inside)} via world coordinates."""
    out = {k: Counter() for k in points_by_key}
    files = {}
    for f in (parc_dir / f"sub-{task_sub}").rglob("*_ROI.*"):
        mm = MASK_RE.match(f.name)
        if mm:
            files.setdefault(f"{mm.group(1)}_{mm.group(2)}", f)
    for lab, f in sorted(files.items()):
        img = nib.load(str(f))
        data = np.asanyarray(img.dataobj)
        inv = np.linalg.inv(img.affine)
        for k, pts in points_by_key.items():
            if not len(pts):
                continue
            vox = np.rint(nib.affines.apply_affine(inv, pts)).astype(int)
            ok = np.all((vox >= 0) & (vox < np.array(data.shape[:3])), axis=1)
            n = int((data[tuple(vox[ok].T)] >= 0.5).sum()) if ok.any() else 0
            if n:
                out[k][lab] = n
    return out


def main() -> None:
    m = load_06b()
    p = argparse.ArgumentParser(description="MT+ ROI for several search margins.")
    p.add_argument("--task-subjects", nargs="+", required=True)
    p.add_argument("--parcellation-dir", "--parcellation_dir", type=Path, required=True)
    p.add_argument("--dilations", type=float, nargs="+", default=[0, 5, 8, 10, 12])
    p.add_argument("--z-threshold", type=float, default=3.09)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-voxels", type=int, default=10)
    p.add_argument("--search-labels", nargs="+", default=m.MT_LABELS)
    p.add_argument("--name", default="MT",
                   help="name of this variant in the output files, e.g. MTbroad -> "
                        "MTbroad_dil05.nii.gz (default MT)")
    args = p.parse_args()

    f2t = m.subject_mapping()
    t2f = {v: k for k, v in f2t.items()}
    out_root = m.REV / "roi_localizer_sensitivity"

    for ts in [f"{int(s):02d}" for s in args.task_subjects]:
        fs = t2f[ts]
        z_path = m.OUT_DIR / f"sub-{fs}" / f"sub-{fs}_task-perception_contrast-motionGTcontrol_stat-z.nii.gz"
        if not z_path.exists():
            print(f"task sub-{ts}: no z-map ({z_path}); run 06b first")
            continue
        print(f"\n== task sub-{ts} (final sub-{fs}) ==", flush=True)
        z_img = nib.load(str(z_path))
        z = z_img.get_fdata()
        zooms = z_img.header.get_zooms()
        search = {h: m.parcel_region(args.parcellation_dir, ts, args.search_labels, h, z_img)
                  for h in ("L", "R")}
        brain = z != 0
        rows, extra_pts, rois = [], {}, {}
        for dil in args.dilations:
            allowed = {h: m.dilate(search[h], dil, zooms) & ~search[o]
                       for h, o in (("L", "R"), ("R", "L"))}
            ari_mask = m.new_img_like(z_img, ((allowed["L"] | allowed["R"]) & brain).astype(np.uint8))
            tdp = m.cluster_level_inference(z_img, mask_img=ari_mask, threshold=args.z_threshold,
                                            alpha=args.alpha).get_fdata()
            roi = np.zeros(z.shape, bool)
            for h in ("L", "R"):
                vox, res = m.mt_hemisphere(z, search[h], allowed[h], zooms, args.z_threshold,
                                           args.min_voxels)
                r = {"margin_mm": dil, "hemi": h, "verdict": res["verdict"]}
                if vox is not None:
                    t = float(tdp[vox].mean())
                    r.update({"voxels": res["voxels"], "peak_z": res["peak_t"],
                              "peak_dist_mm": res["peak_dist_mm"], "in_parcel": res["in_parcel"],
                              "ari_tdp": round(t, 2)})
                    if t <= 0:
                        r["verdict"] = "not significant"
                    else:
                        roi |= vox
                    outside = np.argwhere(vox & ~search[h])
                    extra_pts[(dil, h)] = nib.affines.apply_affine(z_img.affine, outside)
                rows.append(r)
            rois[dil] = roi
            path = out_root / f"sub-{ts}" / f"{args.name}_dil{int(round(dil)):02d}.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            m.new_img_like(z_img, roi.astype(np.uint8)).to_filename(str(path))

        print("   reading HCP-MMP1 masks for the voxels outside the MT+ parcels ...", flush=True)
        labs = label_counts(extra_pts, args.parcellation_dir, ts)
        for r in rows:
            c = labs.get((r["margin_mm"], r["hemi"]), Counter())
            n_out = len(extra_pts.get((r["margin_mm"], r["hemi"]), []))
            r["voxels_outside_parcel"] = n_out
            r["outside_in_areas"] = ", ".join(f"{k} {v}" for k, v in c.most_common(4))
        df = pd.DataFrame(rows)
        cols = [c for c in ["hemi", "margin_mm", "verdict", "voxels", "peak_z", "in_parcel",
                            "ari_tdp", "voxels_outside_parcel", "outside_in_areas"] if c in df]
        for h in ("L", "R"):
            print(df[df["hemi"] == h][cols].to_string(index=False))
            print()
        df.insert(0, "search_labels", " ".join(args.search_labels))
        df.to_csv(out_root / f"sub-{ts}" / f"sensitivity_{args.name}.csv", index=False)
        print(f"   ROI variants: {out_root / f'sub-{ts}'}/{args.name}_dilXX.nii.gz")
    print("\nDone.")


if __name__ == "__main__":
    main()