#!/usr/bin/env python3
"""
06_ROI_selectvoxel.py
===========================

Create subject-specific functional ROI masks by selecting the most
active voxels (largest |t|) within each anatomical ROI, using the
encoding-phase whole-brain contrast map from 06_run_full_fir_glm.py.

Anatomical ROIs (resampled into BOLD space by 02_create_rois.py):

    V1, V2, V3, V4, early_visual, pari, dlpfc

Voxel selection
---------------
Top |t|-value voxels per ROI:

    V1 / V2 / V3 / V4    : 200, 300, 400, 500, 600, 700, 800, 900, 1000
    early_visual         : 200..1000 in 100 steps, plus 2000, 3000
    pari                 : 200..1000 in 100 steps, plus 2000, 3000
    dlpfc                : 200..1000 in 100 steps, plus 2000, 3000

Output layout
-------------
    <output-dir>/sub-XX/<ROI>/
        top0200.nii.gz
        top0300.nii.gz
        ...

Installation
------------
    pip install nibabel nilearn numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


# ============================================================================
# ROI definitions
# ============================================================================
_SMALL = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
_LARGE = _SMALL + [2000, 3000]

ROI_VOXELS = {
    "V1":           _SMALL,
    "V2":           _SMALL,
    "V3":           _SMALL,
    "V4":           _SMALL,
    "early_visual": _LARGE,
    "pari":         _LARGE,
    "dlpfc":        _LARGE,
}


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create top-voxel functional ROI masks from a whole-brain t-map.",
    )
    p.add_argument("--subject", required=True, type=str,
                   help="Participant label without 'sub-' (e.g. '01').")
    p.add_argument("--roi-dir", required=True, type=Path,
                   help="Output directory from 04_create_rois.py "
                        "(contains sub-XX/<ROI>_resampled.nii.gz).")
    p.add_argument("--tmap", required=True, type=Path,
                   help="Encoding-phase whole-brain t-map "
                        "(e.g. sub-01_enc.nii.gz from 04_fullmodel.py).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where the top-voxel masks will be written.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing masks.")
    return p.parse_args()


# ============================================================================
# Core
# ============================================================================
def create_top_voxel_mask(roi_path: Path, tmap_path: Path,
                          out_path: Path, n_voxels: int) -> None:
    """Write a binary mask of the top |t| voxels inside `roi_path`."""
    roi_img = nib.load(str(roi_path))


    tmap_img = nib.load(str(tmap_path))
    roi_img = resample_to_img(roi_img, tmap_img, interpolation="nearest")
    roi_data = roi_img.get_fdata() > 0
    tmap_data = tmap_img.get_fdata()

    masked = np.abs(tmap_data[roi_data])
    if len(masked) == 0:
        raise RuntimeError(f"No voxels inside ROI: {roi_path.name}")

    n_voxels = min(n_voxels, len(masked))
    flat_idx = np.flatnonzero(roi_data)

    vals = np.abs(tmap_data[roi_data])

    top_idx = flat_idx[np.argsort(vals)[-n_voxels:]]

    mask = np.zeros(tmap_data.shape, dtype=np.uint8)
    mask.flat[top_idx] = 1

    nib.Nifti1Image(mask, affine=tmap_img.affine, header=tmap_img.header
                    ).to_filename(str(out_path))
    print(f"  {out_path.name:<20}{int(mask.sum()):>6} voxels")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()

    print("-" * 64)
    print(f"Creating functional ROIs for sub-{args.subject}")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>14}: {v}")
    print("-" * 64)

    subj_roi_dir = args.roi_dir / f"sub-{args.subject}"
    out_root     = args.output_dir / f"sub-{args.subject}"
    out_root.mkdir(parents=True, exist_ok=True)

    for roi, voxel_counts in ROI_VOXELS.items():
        roi_path = subj_roi_dir / f"{roi}_resampled.nii.gz"
        if not roi_path.exists():
            print(f"[skip] missing ROI: {roi}")
            continue

        print(f"\n[{roi}]")
        roi_out = out_root / roi
        roi_out.mkdir(exist_ok=True)

        for n_vox in voxel_counts:
            out_path = roi_out / f"top{n_vox:04d}.nii.gz"
            if out_path.exists() and not args.force:
                print(f"  [cached] top{n_vox:04d}")
                continue
            create_top_voxel_mask(roi_path, args.tmap, out_path, n_vox)

    print("\nDone.")


if __name__ == "__main__":
    main()