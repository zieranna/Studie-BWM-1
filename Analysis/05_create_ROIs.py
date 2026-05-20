#!/usr/bin/env python3
"""
05_create_rois.py
=================

Build subject-space ROI masks from the HCP-MMP1 parcellation, resampled
into the participant's preprocessed-BOLD (T1w) space.

For each participant, the script produces:

    * V1, V2, V3, V4          individual early-visual areas
    * early_visual            union of V1 + V2 + V3 + V4
    * pari                    parietal complex (IP0, IP1, IP2, AIP,
                              LIPv, LIPd, IPS1, MIP, 7Am, 7PL)
    * dlpfc                   dorsolateral PFC complex (46, 8Ad, 8Av,
                              8BL, 8C, 9-46d, 9a, 9p, a9-46v, i6-8,
                              p9-46v, s6-8, SFL)
    * hippocampus             L + R hippocampus from FreeSurfer aseg

Each mask is written twice: once in T1w-anatomical resolution
(e.g. ``early_visual.nii.gz``) and once resampled into BOLD resolution
(``early_visual_resampled.nii.gz``).

Prerequisites
-------------
    1. fMRIPrep has been run on the participant (01_run_fmriprep.py).
    2. The HCP-MMP1 atlas has been projected into the subject's native
       volume using ``create_subj_volume_parcellation.sh`` from
       https://figshare.com/articles/dataset/HCP-MMP1_0_volumetric_NIfTI_masks_in_native_structural_space/4249400
       run with the ``-s YES`` flag so aseg subcortical masks
       (incl. hippocampus) are also produced.

Output layout
-------------
    <output-dir>/sub-XX/
        vol_ref.nii.gz
        <ROI>.nii.gz                       # T1w resolution
        <ROI>_resampled.nii.gz             # BOLD resolution
        ...

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
EARLY_VISUAL = ["V1", "V2", "V3", "V4"]

PARI  = ["IP0", "IP1", "IP2", "AIP", "LIPv", "LIPd",
         "IPS1", "MIP", "7Am", "7PL"]

DLPFC = ["46", "8Ad", "8Av", "8BL", "8C", "9-46d", "9a", "9p",
         "a9-46v", "i6-8", "p9-46v", "s6-8", "SFL"]

SINGLE_ROI_EXPORTS = EARLY_VISUAL          # exported individually
NETWORKS = {                               # exported as unions
    "early_visual": EARLY_VISUAL,
    "pari":         PARI,
    "dlpfc":        DLPFC,
}


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build subject-space ROI masks from HCP-MMP1 parcellation.",
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


def build_hippocampus(aseg_dir: Path, out_dir: Path,
                      ref_path: Path, force: bool) -> None:
    """L + R hippocampus from FreeSurfer aseg, resampled into BOLD space."""
    comb      = out_dir / "hippocampus.nii.gz"
    resampled = out_dir / "hippocampus_resampled.nii.gz"
    if resampled.exists() and not force:
        print("[hippocampus] already exists, skipping")
        return
    r = aseg_dir / "Right-Hippocampus.nii.gz"
    l = aseg_dir / "Left-Hippocampus.nii.gz"
    for p in (r, l):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing aseg mask: {p}. Re-run "
                f"create_subj_volume_parcellation.sh with the -s YES flag."
            )
    print("\n[hippocampus] L + R from aseg masks")
    run_cmd(["fslmaths", str(r), "-add", str(l), str(comb)])
    resample_flirt(comb, resampled, ref_path)


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
    aseg_dir  = parc_subj / "aseg_masks"

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

    # Individual early-visual ROIs.
    for roi in SINGLE_ROI_EXPORTS:
        build_single_roi(roi, masks_dir, out_dir, ref_path, args.force)

    # Network unions.
    for name, rois in NETWORKS.items():
        build_network(name, rois, masks_dir, out_dir, ref_path, args.force)

    # Subcortical hippocampus (separate aseg source).
    build_hippocampus(aseg_dir, out_dir, ref_path, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()