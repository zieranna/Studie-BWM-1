#!/usr/bin/env python3
"""
03_postprocess_tmaps.py
=======================

Post-process the per-run FIR t-maps from ``04_run_fir_glm.py`` into the
two estimate forms used by the decoder.

Pipeline
--------
    1. Per-run z-scoring (within each original run, per voxel across
       trials), then the outputs are re-labelled as BLOCKS:
            block-01 <- run-01 + run-02
            block-02 <- run-03 + run-04
            block-03 <- run-05 + run-06
            block-04 <- run-07 + run-08
       -> tmap_z/{phase}/sub-XX_ses-YY_block-NN_trial-PP.nii.gz

    2. Pair-averaging within each block: 'Xa' and 'Xb' from the two
       runs of the block are averaged into one map.
       -> tmap_pair/{phase}/sub-XX_ses-YY_block-NN_pair-K.nii.gz

Z-scoring uses ``nilearn.image.clean_img(standardize="zscore_sample")``,
which normalises each voxel's series (here: across trials in one run)
to mean 0, std 1. Singletons (one of a/b missing in the block) are
kept in tmap_z but not pair-averaged.

Phases (enc, del, res) are processed independently.

Installation
------------
    pip install nilearn nibabel numpy tqdm
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import clean_img, concat_imgs
from tqdm import tqdm


PHASES = ["enc", "del", "res"]
RUNS_PER_BLOCK = 2


def run_to_block(run: str) -> str:
    """Original run -> block label. '01'/'02' -> '01', '03'/'04' -> '02', etc."""
    return f"{(int(run) - 1) // RUNS_PER_BLOCK + 1:02d}"


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-run z-score and pair-average FIR t-maps.")
    p.add_argument("--subject", required=True, type=str,
                   help="Participant label without 'sub-' (e.g. '01').")
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Root of 02_trialestimates.py output "
                        "(contains sub-XX/model/fir/binding/tmap/).")
    p.add_argument("--force", action="store_true",
                   help="Recompute outputs even if they already exist.")
    return p.parse_args()


# ============================================================================
# Filename parsing
# ============================================================================
def _strip_ext(name: str) -> str:
    return name.replace(".nii.gz", "").replace(".nii", "")


def parse_tmap_name(p: Path) -> dict:
    """Return a dict of BIDS entities from a filename
    (e.g. 'sub-01_ses-01_run-02_trial-4a' -> {'sub':'01','ses':'01','run':'02','trial':'4a'})."""
    out = {}
    for x in _strip_ext(p.name).split("_"):
        if "-" in x:
            k, v = x.split("-", 1)
            out[k] = v
    return out


def trial_to_pair(label: str):
    """'4a' -> ('4','a'); '13b' -> ('13','b'); other -> (None, None)."""
    if label and label[-1] in ("a", "b") and label[:-1].isdigit():
        return label[:-1], label[-1]
    return None, None


# ============================================================================
# 1. Per-run z-scoring
# ============================================================================
def zscore_per_run(sub: str, input_root: Path, force: bool) -> None:
    """Z-score within each original run, then rename outputs to block-NN."""
    tmap_dir = input_root / f"sub-{sub}" / "model" / "fir" / "binding" / "tmap"
    z_dir    = input_root / f"sub-{sub}" / "model" / "fir" / "binding" / "tmap_z"

    for phase in PHASES:
        # Group input maps by (ses, run) so z-scoring is per ORIGINAL run.
        groups: dict = defaultdict(list)
        for p in sorted((tmap_dir / phase).glob(
                f"sub-{sub}_ses-*_run-*_trial-*.nii.gz")):
            meta = parse_tmap_name(p)
            groups[(meta["ses"], meta["run"])].append(p)

        for (ses, run), files in tqdm(groups.items(),
                                       desc=f"  z-score {phase}",
                                       leave=False):
            # Output names carry block-NN rather than run-NN.
            block = run_to_block(run)
            out_files = [z_dir / phase /
                         f.name.replace(f"_run-{run}_", f"_block-{block}_")
                         for f in files]
            if all(o.exists() for o in out_files) and not force:
                continue
            (z_dir / phase).mkdir(parents=True, exist_ok=True)

            # Stack the run's trial maps along a "time" axis,
            # then z-score each voxel across that axis.
            concat = concat_imgs([str(f) for f in files])
            zd = clean_img(concat,
                           standardize="zscore_sample",
                           detrend=False, low_pass=None, high_pass=None,
                           t_r=1.0)

            zdata = zd.get_fdata()
            for idx, out in enumerate(out_files):
                img_3d = nib.Nifti1Image(zdata[..., idx], zd.affine, zd.header)
                img_3d.to_filename(str(out))


# ============================================================================
# 2. Per-block pair averaging (over z-scored maps)
# ============================================================================
def pair_average_per_block(sub: str, input_root: Path, force: bool) -> None:
    z_dir    = input_root / f"sub-{sub}" / "model" / "fir" / "binding" / "tmap_z"
    pair_dir = input_root / f"sub-{sub}" / "model" / "fir" / "binding" / "tmap_pair"

    for phase in PHASES:
        # Bucket z-scored maps as (ses, block, pair_number) -> {'a': path, 'b': path}.
        bucket: dict = defaultdict(dict)
        for p in sorted((z_dir / phase).glob(
                f"sub-{sub}_ses-*_block-*_trial-*.nii.gz")):
            meta = parse_tmap_name(p)
            pair_num, suffix = trial_to_pair(meta.get("trial", ""))
            if pair_num is not None:
                bucket[(meta["ses"], meta["block"], pair_num)][suffix] = p

        (pair_dir / phase).mkdir(parents=True, exist_ok=True)

        n_avg, n_skip = 0, 0
        for (ses, block, num), members in bucket.items():
            if "a" not in members or "b" not in members:
                n_skip += 1
                continue
            out_path = pair_dir / phase / (
                f"sub-{sub}_ses-{ses}_block-{block}_pair-{num}.nii.gz")
            if out_path.exists() and not force:
                n_avg += 1
                continue
            a_img = nib.load(str(members["a"]))
            b_img = nib.load(str(members["b"]))
            avg = (a_img.get_fdata() + b_img.get_fdata()) / 2.0
            nib.Nifti1Image(avg, a_img.affine, a_img.header
                            ).to_filename(str(out_path))
            n_avg += 1
        print(f"  pair-avg {phase}: {n_avg} pairs written, "
              f"{n_skip} singletons skipped")


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()
    print("-" * 64)
    print(f"Post-processing FIR t-maps for sub-{args.subject}")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>10}: {v}")
    print("-" * 64)

    print("\n[1/2] z-scoring within each original run (output relabelled as blocks)")
    zscore_per_run(args.subject, args.input_dir, args.force)

    print("\n[2/2] pair-averaging a + b within each block")
    pair_average_per_block(args.subject, args.input_dir, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()