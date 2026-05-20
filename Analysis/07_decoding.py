#!/usr/bin/env python3
"""
07_decoding.py
================

ROI-based SVM decoding for the binding WM task.

Five analyses are supported, each defined as a list of A-vs-B label
pairs. The decoder loads one input image per occurrence of A or B
across the experiment, runs Leave-One-Block-Out cross-validation with
nested hyperparameter search, and saves accuracy per (pair, ROI,
voxel-count) to an .xlsx table.

Parallelism
-----------
The (pair x ROI x voxel-count) loop is flattened into one job list and
spread across N_JOBS workers using threads. Each worker runs its outer
LOGO CV and inner grid search SERIALLY (no nested Parallel), which
avoids the CPU oversubscription that occurs when an outer thread pool
launches Decoders that each spawn their own loky pool. Threads are
fine here because LinearSVC.fit() releases the GIL.

Inputs
------
Two sources from ``05_postprocess_tmaps.py``:

    * Pair-averaged maps (tmap_pair/)  -- analyses whose unit of
      decoding is a paper PAIR (binding, color, motion).
    * Per-trial z-scored maps (tmap_z/) -- analyses whose unit is a
      single trial (temporal_order, binding_interaction).

ROI masks come from ``07_create_top_voxel_rois.py``.

Cross-validation
----------------
Group = (session, block). 4 sessions x 4 blocks = 16 groups, so
LeaveOneGroupOut gives 16 outer folds. Hyperparameter search runs as
nested LOGO inside each outer fold.

Output
------
    <output-dir>/sub-XX/<analysis>/decoding_<analysis>_<phase>_<date>.xlsx
      sheet 'per_pair' -- one row per (pair, ROI, voxels)
      sheet 'summary'  -- mean and std accuracy per (ROI, voxels)
    <output-dir>/sub-XX/<analysis>/decoding_script_<analysis>_<phase>_<date>.py
"""

from __future__ import annotations

import argparse
import shutil
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.decoding import Decoder
from nilearn.image import resample_img
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

# ============================================================================
# Constants
# ============================================================================
N_BLOCKS_PER_SESSION = 4

# Each analysis: (kind, [(A_label, B_label), ...]).
#   kind="pair"  -> reads tmap_pair/, labels are paper pair numbers ('4', '13')
#   kind="trial" -> reads tmap_z/,    labels are PAPER trial labels ('4a', '13b')
ANALYSES = {
    "binding": ("pair", [
        ("4", "3"),  ("6", "5"),  ("2", "1"),
        ("10", "9"), ("12", "11"), ("8", "7"),
        ("16", "15"), ("18", "17"), ("14", "13"),
    ]),
    "color": ("pair", [
        ("4", "6"),  ("2", "3"),  ("5", "1"),
        ("10", "12"), ("8", "9"), ("11", "7"),
        ("16", "18"), ("14", "15"), ("17", "13"),
        ("3", "5"),  ("6", "2"),  ("4", "1"),
        ("9", "11"), ("12", "8"), ("10", "7"),
        ("15", "17"), ("18", "14"), ("16", "13"),
    ]),
    "motion": ("pair", [
        ("4", "10"), ("6", "12"), ("2", "8"),
        ("3", "16"), ("5", "18"), ("1", "14"),
        ("9", "15"), ("11", "17"), ("7", "13"),
        ("3", "9"),  ("5", "11"), ("1", "7"),
        ("4", "15"), ("6", "17"), ("2", "13"),
        ("10", "16"), ("12", "18"), ("8", "14"),
    ]),
    "temporal_order": ("trial", [
        ("4a", "4b"),   ("6a", "6b"),   ("2a", "2b"),
        ("3a", "3b"),   ("5a", "5b"),   ("1a", "1b"),
        ("10a", "10b"), ("12a", "12b"), ("8a", "8b"),
        ("9a", "9b"),   ("11a", "11b"), ("7a", "7b"),
        ("16a", "16b"), ("18a", "18b"), ("14a", "14b"),
        ("15a", "15b"), ("17a", "17b"), ("13a", "13b"),
    ]),
    "binding_interaction": ("trial", [
        ("4a", "3b"),   ("4a", "3a"),   ("4b", "3b"),   ("4b", "3a"),
        ("6a", "5b"),   ("6a", "5a"),   ("6b", "5b"),   ("6b", "5a"),
        ("2a", "1b"),   ("2a", "1a"),   ("2b", "1b"),   ("2b", "1a"),
        ("10a", "9b"),  ("10a", "9a"),  ("10b", "9b"),  ("10b", "9a"),
        ("12a", "11b"), ("12a", "11a"), ("12b", "11b"), ("12b", "11a"),
        ("8a", "7b"),   ("8a", "7a"),   ("8b", "7b"),   ("8b", "7a"),
        ("16a", "15b"), ("16a", "15a"), ("16b", "15b"), ("16b", "15a"),
        ("18a", "17b"), ("18a", "17a"), ("18b", "17b"), ("18b", "17a"),
        ("14a", "13b"), ("14a", "13a"), ("14b", "13b"), ("14b", "13a"),
    ]),
}

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

# Default parallelism: 32 outer workers, decoder runs SERIALLY inside each.
DEFAULT_N_JOBS = 32


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ROI-based SVM decoding (LOGO over blocks).",
    )
    p.add_argument("--subject", required=True, type=str,
                   help="Participant label without 'sub-' (e.g. '01').")
    p.add_argument("--phase", required=True, choices=["enc", "del", "res"],
                   help="FIR phase to decode.")
    p.add_argument("--analysis", required=True, choices=list(ANALYSES.keys()),
                   help="Which A-vs-B label list to use.")
    p.add_argument("--tmap-dir", required=True, type=Path,
                   help="Root of 03_postprocess_tmaps.py output.")
    p.add_argument("--roi-dir", required=True, type=Path,
                   help="Root of 06_ROI_selectvoxel.py output.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where decoding tables will be written.")
    p.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                   help=f"Outer parallel workers (default {DEFAULT_N_JOBS}). "
                        "Each worker runs its decoder serially inside.")
    return p.parse_args()


# ============================================================================
# Path helpers
# ============================================================================
def tmap_phase_dir(tmap_root: Path, sub: str, phase: str, kind: str) -> Path:
    subdir = "tmap_pair" if kind == "pair" else "tmap_z"
    return tmap_root / f"sub-{sub}" / "model" / "fir" / "binding" / subdir / phase


def find_files(tmap_root: Path, sub: str, phase: str,
               kind: str, label: str) -> list:
    return sorted(tmap_phase_dir(tmap_root, sub, phase, kind).glob(
        f"sub-{sub}_ses-*_block-*_{kind}-{label}.nii.gz"))


def parse_block_id(p: Path) -> int:
    stem = p.name.replace(".nii.gz", "").replace(".nii", "")
    ses = block = None
    for x in stem.split("_"):
        if x.startswith("ses-"):
            ses = int(x.split("-", 1)[1])
        elif x.startswith("block-"):
            block = int(x.split("-", 1)[1])
    if ses is None or block is None:
        raise ValueError(f"Could not parse ses/block from {p.name}")
    return (ses - 1) * N_BLOCKS_PER_SESSION + block


def roi_mask_path(roi_dir: Path, sub: str, roi: str, n_vox: int) -> Path:
    return roi_dir / f"sub-{sub}" / roi / f"top{n_vox:04d}.nii.gz"


def resampled_mask_path(rp: Path) -> Path:
    return rp.with_name(rp.name.replace(".nii.gz", "_resampled.nii.gz"))


# ============================================================================
# Resampling
# ============================================================================
def ensure_resampled_roi(roi_path: Path, ref_path: Path) -> Path:
    """Cache a ROI mask resampled to the reference t-map grid."""
    rp_rs = resampled_mask_path(roi_path)
    if rp_rs.exists():
        return rp_rs
    ref = nib.load(str(ref_path))
    roi = nib.load(str(roi_path))
    roi_rs = resample_img(roi,
                          target_affine=ref.affine,
                          target_shape=ref.shape,
                          interpolation="nearest")
    nib.save(roi_rs, str(rp_rs))
    return rp_rs


# ============================================================================
# Single-combo decoding (one pair x ROI x voxel-count, SERIAL inside)
# ============================================================================
def decode_one_combo(
    tmap_root: Path,
    sub: str,
    phase: str,
    kind: str,
    a_label: str,
    b_label: str,
    mask_path: Path,
) -> float:

    print(f"Running pair {a_label} vs {b_label} - ROI {mask_path.parent.name} top{mask_path.stem.split('top')[-1]} ...")

    decoding = {
        "fmri_niiimgs": [],
        "conditions": [],
        "session": [],
    }

    imgs = []

    files_a = find_files(tmap_root, sub, phase, kind, a_label)
    files_b = find_files(tmap_root, sub, phase, kind, b_label)

    if len(files_a) == 0 or len(files_b) == 0:
        raise FileNotFoundError(
            f"Missing files for ({a_label}, {b_label})"
        )

    imgs.extend([str(f) for f in files_a])
    imgs.extend([str(f) for f in files_b])

    # EXACTLY like original code
    np.random.shuffle(imgs)

    # ------------------------------------------------------------------
    # Reconstruct labels + sessions exactly like old script
    # ------------------------------------------------------------------
    for fmri_niiimg in imgs:

        p = Path(fmri_niiimg)

        stem = p.name.replace(".nii.gz", "").replace(".nii", "")
        parts = stem.split("_")

        ses = None
        block = None

        for x in parts:

            if x.startswith("ses-"):
                ses = int(x.split("-")[1])

            elif x.startswith("block-"):
                block = int(x.split("-")[1])

        if ses is None or block is None:
            raise ValueError(f"Could not parse ses/block from {p.name}")

        run = f"{((ses - 1) * 4) + block}"

        prefix = "pair" if kind == "pair" else "trial"

        if f"{prefix}-{a_label}.nii.gz" in p.name:
            decoding["conditions"].append("A")
        elif f"{prefix}-{b_label}.nii.gz" in p.name:
            decoding["conditions"].append("B")
        else:
            raise ValueError(f"Could not assign label for {p.name}")

        decoding["fmri_niiimgs"].append(fmri_niiimg)
        decoding["session"].append(run)

    # ------------------------------------------------------------------
    # Convert to arrays
    # ------------------------------------------------------------------
    decoding_session = np.array(decoding["session"])
    decoding_conditions = np.array(decoding["conditions"])
    decoding_fmri_niiimgs = np.array(decoding["fmri_niiimgs"])

    run_lst = decoding["session"].copy()
    run_lst = list(dict.fromkeys(run_lst))

    # ------------------------------------------------------------------
    # Decoder setup
    # ------------------------------------------------------------------
    param_grid = [{
        "C": [0.001, 0.01, 1.0, 10, 100, 1000],
        "dual": [True],
        "loss": ["hinge", "squared_hinge"],
        "penalty": ["l2"],
        "tol": [1e-5],
        "max_iter": [1000000],
    }]


    cv = LeaveOneGroupOut()

    deco = Decoder(
        estimator="svc",
        mask=str(mask_path),
        standardize=False,
        cv=cv,
        screening_percentile=100,
        scoring="accuracy",
        n_jobs=1,
        param_grid=param_grid,
        smoothing_fwhm=None,
    )

    warnings.filterwarnings(
        "ignore",
        message="Use a custom estimator at your own risk"
    )

    # ------------------------------------------------------------------
    # EXACT old manual LOSO
    # ------------------------------------------------------------------
    accuracy_lst = []

    for run in run_lst:

        train_mask = decoding_session != run

        fmri_train = decoding_fmri_niiimgs[train_mask]
        y_train = decoding_conditions[train_mask]
        groups_train = decoding_session[train_mask]

        deco.fit(fmri_train, y_train, groups=groups_train)

        test_mask = decoding_session == run

        fmri_test = decoding_fmri_niiimgs[test_mask]
        y_test = decoding_conditions[test_mask]

        pred = deco.predict(fmri_test)

        acc = np.mean(pred == y_test)

        accuracy_lst.append(acc)

    return float(np.mean(accuracy_lst))


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()
    kind, pair_list = ANALYSES[args.analysis]

    print("-" * 64)
    print(f"Decoding sub-{args.subject} | {args.analysis} | phase {args.phase}")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>14}: {v}")
    print(f"  {'kind':>14}: {kind}")
    print(f"  {'n_pairs':>14}: {len(pair_list)}")
    print("-" * 64)

    out_dir = args.output_dir / f"sub-{args.subject}" / args.analysis
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Find one reference t-map for ROI resampling.
    # ------------------------------------------------------------------
    ref_path = None
    for a, _ in pair_list:
        ref_files = find_files(args.tmap_dir, args.subject, args.phase, kind, a)
        if ref_files:
            ref_path = ref_files[0]
            break
    if ref_path is None:
        print("ERROR: no t-map files found. Check --tmap-dir and --phase.")
        return

    # ------------------------------------------------------------------
    # Pre-resample every ROI mask once, before the parallel section.
    # Eliminates the race condition where multiple workers try to
    # write the same _resampled.nii.gz concurrently.
    # ------------------------------------------------------------------
    print(f"Pre-resampling ROI masks to grid of {ref_path.name} ...")
    n_resampled = 0
    for roi, voxel_list in ROI_VOXELS.items():
        for n_vox in voxel_list:
            rp = roi_mask_path(args.roi_dir, args.subject, roi, n_vox)
            if rp.exists():
                ensure_resampled_roi(rp, ref_path)
                n_resampled += 1
    print(f"  {n_resampled} ROI masks ready.")

    # ------------------------------------------------------------------
    # Flatten (pair x ROI x voxel) into one job list.
    # ------------------------------------------------------------------
    jobs = []
    for pair_idx, (a, b) in enumerate(pair_list, start=1):
        for roi, voxel_list in ROI_VOXELS.items():
            for n_vox in voxel_list:
                rp = roi_mask_path(args.roi_dir, args.subject, roi, n_vox)
                if not rp.exists():
                    continue
                jobs.append((pair_idx, a, b, roi, n_vox, resampled_mask_path(rp)))

    print(f"\nRunning {len(jobs)} decoder fits on {args.n_jobs} workers ...")
    t0 = time.time()

    def run_one(pair_idx, a, b, roi, n_vox, mask_path):
        try:
            acc = decode_one_combo(
                args.tmap_dir, args.subject, args.phase,
                kind, a, b, mask_path,
            )
        except Exception as e:
            print(f"  [error] pair {pair_idx} ({a} vs {b}) "
                  f"{roi} top{n_vox:04d}: {e}")
            return None
        return {
            "subject":  args.subject,
            "phase":    args.phase,
            "analysis": args.analysis,
            "pair":     pair_idx,
            "A_label":  a,
            "B_label":  b,
            "ROI":      roi,
            "voxels":   n_vox,
            "accuracy": acc,
        }

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(run_one)(*j) for j in jobs
    )
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows)

    elapsed = time.time() - t0
    n_ok = max(len(rows), 1)
    print(f"\nFinished {len(rows)}/{len(jobs)} jobs in {elapsed:.1f}s "
          f"({elapsed / n_ok:.2f}s avg per fit).")

    # ------------------------------------------------------------------
    # Save per_pair FIRST as a safety net (so a later failure doesn't
    # lose the decoder fits we just spent N minutes computing).
    # ------------------------------------------------------------------
    today = time.strftime("%Y%m%d")
    out_xlsx = out_dir / f"decoding_{args.analysis}_{args.phase}_{today}.xlsx"
    df.to_excel(out_xlsx, sheet_name="per_pair", index=False)
    print(f"\nSaved per_pair to {out_xlsx}")

    # ------------------------------------------------------------------
    # Summary across pairs.
    # ------------------------------------------------------------------
    summary = pd.DataFrame(columns=["ROI", "voxels", "mean_accuracy",
                                     "std_accuracy", "n_pairs"])
    if not df.empty:
        try:
            summary = (df.groupby(["ROI", "voxels"])
                         .agg(mean_accuracy=("accuracy", "mean"),
                              std_accuracy=("accuracy", "std"),
                              n_pairs=("accuracy", "size"))
                         .reset_index())
            summary["mean_accuracy"] = summary["mean_accuracy"].round(4)
            summary["std_accuracy"]  = summary["std_accuracy"].round(4)
            print("\n" + "-" * 64)
            print("Mean accuracy across pairs (per ROI x voxel-count):")
            print("-" * 64)
            print(summary.to_string(index=False))
        except Exception as e:
            print(f"  [warn] summary computation failed: {e}")

    # Append summary as a second sheet (replace the file with both sheets).
    with pd.ExcelWriter(out_xlsx) as xl:
        df.to_excel(xl, sheet_name="per_pair", index=False)
        summary.to_excel(xl, sheet_name="summary", index=False)
    print(f"Saved {out_xlsx}")

    snapshot = out_dir / f"decoding_script_{args.analysis}_{args.phase}_{today}.py"
    shutil.copy(__file__, str(snapshot))
    print(f"Snapshot at {snapshot}")

    print("Done.")


if __name__ == "__main__":
    main()