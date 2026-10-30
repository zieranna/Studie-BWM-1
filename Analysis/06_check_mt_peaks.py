#!/usr/bin/env python3
"""
06c_check_mt_peaks.py
=====================

Is the MT+ ROI the right region? Uses the saved localizer t-map of
06b_localizer_mt.py (no refitting).

The motion>control map is thresholded WITHOUT the MT+ search region, and its
strongest clusters (26-connectivity) are listed with
    size, peak t, peak position (mm), the HCP-MMP1 area containing the peak
    (the most frequent label within 3 mm of the peak, so a peak in white
    matter still gets its nearest cortical area), the share of the cluster
    inside the MT+ search region (MT, MST, FST, V4t), and the distance of the
    peak to that search region.

Reading:
    strongest cluster in MT/MST/FST/V4t          -> MT+ ROI is the right region
    strongest cluster in a direct neighbour       -> MT+ may lie partly next to
      (LO1-3, PH, TPOJ2/3, V3CD, PHT)                the parcel in this subject
    strongest cluster in another motion area      -> parcel constraint correct;
      (V3A, V3B, V6, V6A, V7, IPS1, ...)             MT+ is just weaker here

Output: printed table and <localizer-dir>/sub-XX/mt_peak_check.csv
Usage:  python 06c_check_mt_peaks.py --task-subjects 01 --parcellation-dir .../hpcmmp1_output
"""

from __future__ import annotations

import argparse
import inspect
import re
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from scipy import ndimage

REV = Path("/home/annaz/BindingWM/Studie-BWM-1_CP_revision1")
LOC_DIR = REV / "localizer"
ROI_DIR = REV / "roi_localizer"
MAPPING = REV / "timing_check" / "subject_mapping.csv"
MT_COMPLEX = {"MT", "MST", "FST", "V4t"}
NEIGHBOURS = {"LO1", "LO2", "LO3", "PH", "TPOJ2", "TPOJ3", "V3CD", "PHT"}
MASK_RE = re.compile(r"^([LR])_(.+)_ROI\.(nii\.gz|nii|mgz)$")
_RESAMPLE_KW = {k: True for k in ("force_resample", "copy_header")
                if k in inspect.signature(resample_to_img).parameters}


def label_files(parc_dir: Path, task_sub: str) -> dict:
    """{'L_MT': path, ...} for every HCP-MMP1 volume mask below sub-XX/."""
    out = {}
    for f in (parc_dir / f"sub-{task_sub}").rglob("*_ROI.*"):
        m = MASK_RE.match(f.name)
        if m:
            out.setdefault(f"{m.group(1)}_{m.group(2)}", f)
    return out


def labels_near(points_mm: dict, files: dict) -> dict:
    """{cluster: Counter(label -> number of query points inside it)}."""
    hits = {k: Counter() for k in points_mm}
    for lab, f in sorted(files.items()):
        img = nib.load(str(f))
        data = np.asanyarray(img.dataobj)
        inv = np.linalg.inv(img.affine)
        for k, pts in points_mm.items():
            vox = np.rint(nib.affines.apply_affine(inv, pts)).astype(int)
            ok = np.all((vox >= 0) & (vox < np.array(data.shape[:3])), axis=1)
            n = int((data[tuple(vox[ok].T)] >= 0.5).sum()) if ok.any() else 0
            if n:
                hits[k][lab] = n
    return hits


def main() -> None:
    p = argparse.ArgumentParser(description="Where are the strongest motion>control clusters?")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--task-subjects", nargs="+")
    who.add_argument("--subjects", nargs="+", help="final numbering")
    p.add_argument("--parcellation-dir", "--parcellation_dir", type=Path, required=True)
    p.add_argument("--t-threshold", type=float, default=2.34)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--min-size", type=int, default=5, help="voxels")
    p.add_argument("--radius-mm", type=float, default=3.0)
    args = p.parse_args()

    m = pd.read_csv(MAPPING, dtype=str)
    f2t = {f"{int(r.final_subject):02d}": f"{int(r.behav_subject):02d}" for r in m.itertuples()}
    t2f = {v: k for k, v in f2t.items()}
    pairs = ([(t2f[f"{int(s):02d}"], f"{int(s):02d}") for s in args.task_subjects]
             if args.task_subjects else [(f"{int(s):02d}", f2t[f"{int(s):02d}"]) for s in args.subjects])

    for fin, task in pairs:
        print(f"\n== task sub-{task} (final sub-{fin}) ==")
        t_img = nib.load(str(LOC_DIR / f"sub-{fin}" /
                             f"sub-{fin}_task-perception_contrast-motionGTcontrol_stat-t.nii.gz"))
        t = t_img.get_fdata()
        search = np.zeros(t.shape, bool)
        for h in ("L", "R"):
            s_img = nib.load(str(ROI_DIR / f"sub-{task}" / "MT" / f"search_{h}.nii.gz"))
            if s_img.shape[:3] != t.shape[:3] or not np.allclose(s_img.affine, t_img.affine, atol=1e-3):
                # 06b writes the ROI on the task grid; bring it to the localizer grid
                s_img = resample_to_img(s_img, t_img, interpolation="nearest", **_RESAMPLE_KW)
            search |= np.asanyarray(s_img.dataobj) > 0

        lab, n = ndimage.label(t > args.t_threshold, structure=np.ones((3, 3, 3)))
        rows = []
        for c in range(1, n + 1):
            vox = np.argwhere(lab == c)
            if len(vox) < args.min_size:
                continue
            pk = tuple(vox[np.argmax(t[tuple(vox.T)])])
            rows.append({"cluster": c, "size": len(vox), "peak_t": float(t[pk]), "peak_vox": pk,
                         "in_mt_search": float(search[tuple(vox.T)].mean())})
        rows = sorted(rows, key=lambda r: -r["peak_t"])[:args.n_clusters]
        if not rows:
            print(f"   no cluster of >= {args.min_size} voxels with t > {args.t_threshold}: "
                  f"no motion response to localize MT+ from")
            pd.DataFrame(columns=["size", "peak_t", "hcp_area", "kind", "in_mt_search",
                                  "dist_to_mt_search_mm", "other_areas_3mm", "peak_mm"]).to_csv(
                LOC_DIR / f"sub-{fin}" / "mt_peak_check.csv", index=False)
            continue

        # distance from each peak to the MT+ search region (mm)
        zooms = np.array(t_img.header.get_zooms()[:3])
        dist = ndimage.distance_transform_edt(~search, sampling=zooms)
        # query points within radius_mm around each peak (1 mm grid)
        g = np.arange(-args.radius_mm, args.radius_mm + 1e-9, 1.0)
        off = np.array([(x, y, z) for x in g for y in g for z in g
                        if x * x + y * y + z * z <= args.radius_mm ** 2])
        pts = {r["cluster"]: nib.affines.apply_affine(t_img.affine, np.array(r["peak_vox"])) + off
               for r in rows}
        files = label_files(args.parcellation_dir, task)
        print(f"   reading {len(files)} HCP-MMP1 masks ...", flush=True)
        hits = labels_near(pts, files)

        for r in rows:
            mm = nib.affines.apply_affine(t_img.affine, np.array(r["peak_vox"]))
            top = hits[r["cluster"]].most_common(3)
            r.update({"peak_mm": tuple(np.round(mm, 1)), "dist_to_mt_search_mm": round(float(dist[r["peak_vox"]]), 1),
                      "hcp_area": top[0][0] if top else "none within 3 mm",
                      "other_areas_3mm": ", ".join(k for k, _ in top[1:])})
            area = r["hcp_area"].split("_", 1)[-1]
            r["kind"] = ("MT+ complex" if area in MT_COMPLEX else
                         "next to MT+" if area in NEIGHBOURS else "other")
        df = pd.DataFrame(rows)[["size", "peak_t", "hcp_area", "kind", "in_mt_search",
                                 "dist_to_mt_search_mm", "other_areas_3mm", "peak_mm"]]
        df["peak_t"] = df["peak_t"].round(2)
        df["in_mt_search"] = df["in_mt_search"].round(2)
        print(df.to_string(index=False))
        out = LOC_DIR / f"sub-{fin}" / "mt_peak_check.csv"
        df.to_csv(out, index=False)

        best = df.iloc[0]
        print(f"\n   strongest cluster: t = {best.peak_t} in {best.hcp_area} ({best.kind}), "
              f"{best.dist_to_mt_search_mm} mm from the MT+ search region")
        if best.kind == "MT+ complex":
            print("   -> the strongest motion response IS in the MT+ complex: ROI is the right region")
        elif best.kind == "next to MT+":
            print("   -> strongest response right next to the MT+ parcels: MT+ may lie partly "
                  "outside the parcel here (consider a larger --cluster-dilate-mm)")
        else:
            print("   -> strongest response in a different area: keep the parcel constraint; "
                  "check that a cluster in MT/MST/FST/V4t follows in the list")
        print(f"   table: {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()