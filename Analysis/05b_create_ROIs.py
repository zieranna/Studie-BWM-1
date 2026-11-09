#!/usr/bin/env python3
"""
05b_create_ROIs_visual.py
=========================

Same logic as 05_create_ROIs.py (HCP-MMP1 parcellation, L + R combined,
binarised >= 0.5, FLIRT -applyxfm -usesqform into BOLD space, re-binarised
>= 0.1), for the visual ROIs:

    * V1, V2, V3, V4          as in 05 (unchanged)
    * V3A, V3B                NEW, HCP-MMP1 labels V3A / V3B, same procedure
    * MT                      NEW, MT+ from the localizer (06b / 06d), copied
                              from <mt-dir>/sub-XX/<mt-file> (default MT_dil08)
    * visual_all              NEW, union of all of the above (like early_visual
                              in 05): V1-V4 + V3A + V3B combined in T1w
                              resolution and resampled as in 05, then MT+ added
                              in BOLD resolution

Each atlas mask is written twice, as in 05: <ROI>.nii.gz (T1w resolution) and
<ROI>_resampled.nii.gz (BOLD resolution). MT+ is already in BOLD resolution,
so only MT_resampled.nii.gz is written.

Prerequisites
-------------
    1. fMRIPrep has been run on the participant (01_run_fmriprep.py).
    2. The HCP-MMP1 atlas has been projected into the subject's native
       volume using ``create_subj_volume_parcellation.sh`` from
       https://figshare.com/articles/dataset/HCP-MMP1_0_volumetric_NIfTI_masks_in_native_structural_space/4249400
    3. The MT+ localizer ROI exists (06b / 06d): <mt-dir>/sub-XX/MT_dil08.nii.gz

Output layout
-------------
    <output-dir>/sub-XX/
        vol_ref.nii.gz
        <ROI>.nii.gz                       # T1w resolution
        <ROI>_resampled.nii.gz             # BOLD resolution
        ...                                # ROI = V1, V2, V3, V4, V3A, V3B
        MT_resampled.nii.gz                # localizer MT+, BOLD resolution
        visual_all.nii.gz                  # V1-V4 + V3A + V3B, T1w resolution (no MT+)
        visual_all_resampled.nii.gz        # V1-V4 + V3A + V3B + MT+, BOLD resolution

Installation
------------
External (must be on PATH):
    * FSL (``fslmaths``, ``flirt``)        https://fsl.fmrib.ox.ac.uk

Python:
    pip install numpy nibabel
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib


# ============================================================================
# ROI definitions (HCP-MMP1 labels, hemisphere-agnostic)
# ============================================================================
SINGLE_ROI_EXPORTS = ["V1", "V2", "V3", "V4", "V3A", "V3B"]   # exported individually
NETWORKS = {                                                   # exported as unions
    "visual_all": SINGLE_ROI_EXPORTS,                          # + MT+ (added in BOLD space)
}

MT_DIR  = Path("/home/annaz/BindingWM/Studie-BWM-1_CP_revision1/roi_localizer_sensitivity")
MT_FILE = "MT_dil08.nii.gz"


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visual ROI masks: HCP-MMP1 V1-V4, V3A, V3B + localizer MT+.",
    )
    p.add_argument("--subject", required=True, type=str,
                   help="Participant label without 'sub-' prefix (e.g. '01').")
    p.add_argument("--fmriprep-dir", required=True, type=Path,
                   help="Path to fMRIPrep derivatives directory.")
    p.add_argument("--parcellation-dir", required=True, type=Path,
                   help="Path to HCP-MMP1 subject-space parcellation output.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to save the ROI masks (one subdir per subject).")
    p.add_argument("--session", default="01", type=str,
                   help="Session label of the BOLD reference volume (default: '01').")
    p.add_argument("--run", default="01", type=str,
                   help="Run label of the BOLD reference volume (default: '01').")
    p.add_argument("--mt-dir", default=MT_DIR, type=Path,
                   help="Folder with sub-XX/<mt-file> from the localizer (task numbering).")
    p.add_argument("--mt-file", default=MT_FILE, type=str,
                   help=f"MT+ mask per subject (default: {MT_FILE}).")
    p.add_argument("--force", action="store_true",
                   help="Rebuild masks even if they already exist on disk.")
    return p.parse_args()


# ============================================================================
# Helpers
# ============================================================================
def run_cmd(cmd: list) -> None:
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def binarize_inplace(path: Path, threshold: float) -> None:
    """Threshold a NIfTI mask at `threshold` and overwrite it as uint8 0/1."""
    img = nib.load(str(path))
    mask = (img.get_fdata() >= threshold).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(path))


def resample_flirt(in_path: Path, out_path: Path, ref_path: Path) -> None:
    """Resample a binary mask into BOLD space.

    Input is binarised (>=0.5), FLIRT is applied with the stored
    sform/qform only (no realignment), then the output is re-binarised
    at the standard partial-volume threshold of 0.1.
    """
    binarize_inplace(in_path, threshold=0.5)
    run_cmd(["flirt",
             "-in",  str(in_path),
             "-ref", str(ref_path),
             "-out", str(out_path),
             "-applyxfm", "-usesqform"])
    binarize_inplace(out_path, threshold=0.1)
    n_vox = int(nib.load(str(out_path)).get_fdata().sum())
    print(f"  -> {out_path.name}: {n_vox} voxels")


def write_bold_reference(fmriprep_dir: Path, subject: str,
                         session: str, run: str, out_path: Path) -> Path:
    """Save the first volume of the preprocessed BOLD as a resampling reference."""
    bold_path = (fmriprep_dir / f"sub-{subject}" / f"ses-{session}" / "func" /
                 f"sub-{subject}_ses-{session}_task-bwm_acq-TR1200_run-{run}"
                 f"_space-T1w_desc-preproc_bold.nii.gz")
    if not bold_path.exists():
        raise FileNotFoundError(f"Preprocessed BOLD not found: {bold_path}")
    img = nib.load(str(bold_path))
    vol0 = img.get_fdata()[..., 0]
    nib.save(nib.Nifti1Image(vol0, img.affine, img.header), str(out_path))
    return out_path


def combine_hemispheres(roi: str, masks_dir: Path, out_path: Path) -> None:
    """fslmaths -add of the L and R hemispheric versions of a single ROI."""
    r = masks_dir / f"R_{roi}_ROI.nii.gz"
    l = masks_dir / f"L_{roi}_ROI.nii.gz"
    for p in (r, l):
        if not p.exists():
            raise FileNotFoundError(f"Missing HCP-MMP1 ROI mask: {p}")
    run_cmd(["fslmaths", str(r), "-add", str(l), str(out_path)])


# ============================================================================
# Pipeline steps
# ============================================================================
def build_single_roi(roi: str, masks_dir: Path, out_dir: Path,
                     ref_path: Path, force: bool) -> None:
    """Export one HCP-MMP1 region (L+R combined, resampled to BOLD space)."""
    roi_comb      = out_dir / f"{roi}.nii.gz"
    roi_resampled = out_dir / f"{roi}_resampled.nii.gz"
    if roi_resampled.exists() and not force:
        print(f"[{roi}] already exists, skipping")
        return
    print(f"\n[{roi}] single region")
    if not roi_comb.exists():
        combine_hemispheres(roi, masks_dir, roi_comb)
    resample_flirt(roi_comb, roi_resampled, ref_path)


def build_mt(mt_path: Path, out_dir: Path, ref_path: Path, force: bool) -> None:
    """MT+ from the localizer, put on the grid of vol_ref.

    The localizer ROI is already in BOLD resolution on the task grid, so it is
    copied as is. Only if the grid differs is it resampled with FLIRT
    (nearest neighbour; the 0.1 threshold of resample_flirt is meant for
    T1w -> BOLD and would inflate a BOLD-resolution mask).
    """
    out_path = out_dir / "MT_resampled.nii.gz"
    if out_path.exists() and not force:
        print("[MT] already exists, skipping")
        return
    if not mt_path.exists():
        print(f"[MT] missing localizer ROI {mt_path}, skipped")
        return
    print(f"\n[MT] localizer ROI {mt_path}")
    mt, ref = nib.load(str(mt_path)), nib.load(str(ref_path))
    if mt.shape[:3] == ref.shape[:3] and np.allclose(mt.affine, ref.affine, atol=1e-3):
        mask = (mt.get_fdata() > 0).astype(np.uint8)
        img = nib.Nifti1Image(mask, ref.affine, ref.header)
        img.set_data_dtype(np.uint8)
        nib.save(img, str(out_path))
    else:
        print("  grid differs from vol_ref -> FLIRT nearest neighbour")
        run_cmd(["flirt", "-in", str(mt_path), "-ref", str(ref_path), "-out", str(out_path),
                 "-applyxfm", "-usesqform", "-interp", "nearestneighbour"])
        binarize_inplace(out_path, threshold=0.5)
    n_vox = int(nib.load(str(out_path)).get_fdata().sum())
    print(f"  -> {out_path.name}: {n_vox} voxels")


def build_network(name: str, rois: list, masks_dir: Path,
                  out_dir: Path, ref_path: Path, force: bool) -> None:
    """Export the union of `rois` as a single network mask."""
    network_path      = out_dir / f"{name}.nii.gz"
    network_resampled = out_dir / f"{name}_resampled.nii.gz"
    if network_resampled.exists() and not force:
        print(f"[{name}] already exists, skipping")
        return
    if network_path.exists():                     # rebuild from scratch
        network_path.unlink()

    print(f"\n[{name}] union of {len(rois)} regions")
    for i, roi in enumerate(rois):
        roi_comb = out_dir / f"{roi}.nii.gz"
        if not roi_comb.exists():
            combine_hemispheres(roi, masks_dir, roi_comb)
        if i == 0:
            shutil.copy(str(roi_comb), str(network_path))
        else:
            run_cmd(["fslmaths", str(network_path),
                     "-add", str(roi_comb), str(network_path)])
    resample_flirt(network_path, network_resampled, ref_path)


def add_mt_to_network(name: str, out_dir: Path) -> None:
    """Add MT_resampled to <name>_resampled (both BOLD resolution, same grid)."""
    net_path, mt_path = out_dir / f"{name}_resampled.nii.gz", out_dir / "MT_resampled.nii.gz"
    if not mt_path.exists():
        print(f"[{name}] no MT_resampled.nii.gz -> union WITHOUT MT+")
        return
    net, mt = nib.load(str(net_path)), nib.load(str(mt_path))
    if net.shape[:3] != mt.shape[:3] or not np.allclose(net.affine, mt.affine, atol=1e-3):
        raise ValueError(f"{mt_path.name} and {net_path.name} are not on the same grid")
    union = ((net.get_fdata() > 0) | (mt.get_fdata() > 0)).astype(np.uint8)
    img = nib.Nifti1Image(union, net.affine, net.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(net_path))
    print(f"[{name}] + MT+ -> {net_path.name}: {int(union.sum())} voxels")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()

    print("-" * 64)
    print(f"Building ROI masks for sub-{args.subject}")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>18}: {v}")
    print("-" * 64)

    parc_subj = args.parcellation_dir / f"sub-{args.subject}"
    masks_dir = parc_subj / "masks"

    if not masks_dir.exists():
        raise FileNotFoundError(
            f"HCP-MMP1 parcellation masks not found at {masks_dir}. "
            f"Run create_subj_volume_parcellation.sh first."
        )

    out_dir = args.output_dir / f"sub-{args.subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = write_bold_reference(
        args.fmriprep_dir, args.subject, args.session, args.run,
        out_dir / "vol_ref.nii.gz",
    )

    # Individual visual ROIs (HCP-MMP1).
    for roi in SINGLE_ROI_EXPORTS:
        build_single_roi(roi, masks_dir, out_dir, ref_path, args.force)

    # MT+ from the localizer.
    build_mt(args.mt_dir / f"sub-{args.subject}" / args.mt_file, out_dir, ref_path, args.force)

    # Union of all visual ROIs (atlas areas as in 05, then + MT+).
    for name, rois in NETWORKS.items():
        build_network(name, rois, masks_dir, out_dir, ref_path, args.force)
        add_mt_to_network(name, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()