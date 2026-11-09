#!/usr/bin/env python3
"""
05b_create_ROIs_revision.py
===========================

Same logic as 05_create_ROIs.py (HCP-MMP1 parcellation, L + R combined,
binarised >= 0.5, FLIRT -applyxfm -usesqform into BOLD space, re-binarised
>= 0.1). Areas are HCP-MMP1 (Glasser et al., 2016, Nature):

    single areas   V1, V2, V3, V4, V3A, V3B
    visual_all     V1, V2, V3, V4, V3A, V3B + MT from the atlas
                   (MT, MST, FST, V4t)
    visual_locMT   V1, V2, V3, V4, V3A, V3B + MT from the localizer, selected
                   within the MT+ complex areas (06d variant MTbroad_dil05:
                   MT, MST, FST, V4t, LO1, LO2, LO3, PH, V3CD + 5 mm)
    MT_complex     MT+ complex as defined by Glasser et al. (2016) / Huang et
                   al. (2022): MT, MST, FST, V4t, LO1, LO2, LO3, PH, V3CD
    dlpfc_noSFL    dlpfc of 05 without SFL (the other 12 areas unchanged)

Each atlas mask is written twice, as in 05: <ROI>.nii.gz (T1w resolution) and
<ROI>_resampled.nii.gz (BOLD resolution). The localizer MT+ is already in BOLD
resolution: it is copied to MTloc_resampled.nii.gz and added to visual_locMT
after resampling (visual_locMT.nii.gz in T1w resolution has no MT+).
Subjects without a localizer MT+ (file missing or empty) get no visual_locMT.

Prerequisites
-------------
    1. fMRIPrep has been run on the participant (01_run_fmriprep.py).
    2. The HCP-MMP1 atlas has been projected into the subject's native
       volume using ``create_subj_volume_parcellation.sh`` from
       https://figshare.com/articles/dataset/HCP-MMP1_0_volumetric_NIfTI_masks_in_native_structural_space/4249400
    3. For visual_locMT: the localizer MT+ of 06d,
       <mt-dir>/sub-XX/MTbroad_dil05.nii.gz (task numbering)

Output layout
-------------
    <output-dir>/sub-XX/
        vol_ref.nii.gz
        <ROI>.nii.gz                       # T1w resolution
        <ROI>_resampled.nii.gz             # BOLD resolution
        ...                                # single areas and the three unions
        MT.nii.gz, LO1.nii.gz, ...         # L+R areas used for the unions (T1w)

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

DLPFC = ["46", "8Ad", "8Av", "8BL", "8C", "9-46d", "9a", "9p",           # as in 05
         "a9-46v", "i6-8", "p9-46v", "s6-8", "SFL"]

MT_COMPLEX = ["MT", "MST", "FST", "V4t", "LO1", "LO2", "LO3", "PH", "V3CD"]

VISUAL = EARLY_VISUAL + ["V3A", "V3B"]

SINGLE_ROI_EXPORTS = VISUAL                                 # exported individually
NETWORKS = {                                                # exported as unions
    "visual_all":  VISUAL + ["MT", "MST", "FST", "V4t"],    # MT from the atlas
    "MT_complex":  MT_COMPLEX,
    "dlpfc_noSFL": [r for r in DLPFC if r != "SFL"],
}
LOC_NETWORKS = {                                            # + localizer MT+
    "visual_locMT": VISUAL,
}

MT_DIR  = Path("/home/annaz/BindingWM/Studie-BWM-1_CP_revision1/roi_localizer_sensitivity")
MT_FILE = "MTbroad_dil05.nii.gz"      # 06d: MT+ complex areas + 5 mm


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Revision ROI masks: visual_all, visual_locMT, MT_complex, dlpfc_noSFL.",
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
                   help="Folder with sub-XX/<mt-file> from 06d (task numbering).")
    p.add_argument("--mt-file", default=MT_FILE, type=str,
                   help=f"Localizer MT+ mask per subject (default: {MT_FILE}).")
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


def copy_localizer_mt(mt_path: Path, out_dir: Path, ref_path: Path) -> Path | None:
    """Localizer MT+ on the grid of vol_ref -> MTloc_resampled.nii.gz.

    The 06d ROI is written on the task grid, so it is normally copied as is;
    only if the grid differs it is resampled with FLIRT (nearest neighbour).
    Returns None if the file is missing or empty (-> no visual_locMT).
    """
    out_path = out_dir / "MTloc_resampled.nii.gz"
    if not mt_path.exists():
        print(f"[MTloc] no localizer MT+ ({mt_path}) -> no visual_locMT for this subject")
        return None
    mt, ref = nib.load(str(mt_path)), nib.load(str(ref_path))
    if mt.shape[:3] == ref.shape[:3] and np.allclose(mt.affine, ref.affine, atol=1e-3):
        img = nib.Nifti1Image((mt.get_fdata() > 0).astype(np.uint8), ref.affine, ref.header)
        img.set_data_dtype(np.uint8)
        nib.save(img, str(out_path))
    else:
        print("[MTloc] grid differs from vol_ref -> FLIRT nearest neighbour")
        run_cmd(["flirt", "-in", str(mt_path), "-ref", str(ref_path), "-out", str(out_path),
                 "-applyxfm", "-usesqform", "-interp", "nearestneighbour"])
        binarize_inplace(out_path, threshold=0.5)
    n_vox = int(nib.load(str(out_path)).get_fdata().sum())
    if n_vox == 0:
        print(f"[MTloc] {mt_path.name} is empty -> no visual_locMT for this subject")
        out_path.unlink()
        return None
    print(f"[MTloc] {mt_path.name} -> {out_path.name}: {n_vox} voxels")
    return out_path


def add_mask(net_path: Path, add_path: Path) -> None:
    """net_path |= add_path (both BOLD resolution, same grid), in place."""
    net, add = nib.load(str(net_path)), nib.load(str(add_path))
    if net.shape[:3] != add.shape[:3] or not np.allclose(net.affine, add.affine, atol=1e-3):
        raise ValueError(f"{add_path.name} and {net_path.name} are not on the same grid")
    union = ((net.get_fdata() > 0) | (add.get_fdata() > 0)).astype(np.uint8)
    img = nib.Nifti1Image(union, net.affine, net.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(net_path))
    print(f"  + {add_path.name} -> {net_path.name}: {int(union.sum())} voxels")


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

    # Individual ROIs.
    for roi in SINGLE_ROI_EXPORTS:
        build_single_roi(roi, masks_dir, out_dir, ref_path, args.force)

    # Unions.
    for name, rois in NETWORKS.items():
        build_network(name, rois, masks_dir, out_dir, ref_path, args.force)

    # Unions + localizer MT+.
    mt_loc = copy_localizer_mt(args.mt_dir / f"sub-{args.subject}" / args.mt_file,
                               out_dir, ref_path)
    for name, rois in LOC_NETWORKS.items():
        net_resampled = out_dir / f"{name}_resampled.nii.gz"
        if mt_loc is None:                        # never a version without MT+
            for f in (out_dir / f"{name}.nii.gz", net_resampled):
                f.unlink(missing_ok=True)
            continue
        if net_resampled.exists() and not args.force:
            print(f"[{name}] already exists, skipping")
            continue
        build_network(name, rois, masks_dir, out_dir, ref_path, force=True)
        add_mask(net_resampled, mt_loc)

    print("\nDone.")


if __name__ == "__main__":
    main()