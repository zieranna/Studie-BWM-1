#!/usr/bin/env python3
"""
08_betacoefficients.py
======================

Single-ROI × single-voxel-count variant of 08_decode_svm.py that ALSO
saves the averaged SVM coefficient map per pair.

The CV logic is identical to 08_decode_svm.py.decode_one_combo() — same
shuffle (np.random.shuffle on the imgs list), same string-based group
IDs parsed from filenames, same insertion-order-preserving ``run_lst``
via ``dict.fromkeys``, same single reused ``deco`` instance with manual
outer LOGO. The only additions are (1) collecting
``deco.coef_img_["A"]`` after each fold's ``deco.fit`` and (2) averaging
across folds and writing one NIfTI per pair.

Output layout
-------------
    <output-dir>/sub-XX/<analysis>/<roi>/
        coef_top<NNNN>_<phase>_pair-<NN>_<A>vs<B>.nii.gz
        accuracy_top<NNNN>_<phase>_<YYYYMMDD>.csv
"""

from __future__ import annotations

import argparse
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
# Constants (identical to 08_decode_svm.py)
# ============================================================================
N_BLOCKS_PER_SESSION = 4

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

DEFAULT_ROI    = "early_visual"
DEFAULT_VOXELS = 700
DEFAULT_N_JOBS = 16


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SVM decoding with averaged-coefficient export "
                    "(single ROI × voxel-count).",
    )
    p.add_argument("--subject",  required=True, type=str)
    p.add_argument("--phase",    required=True, choices=["enc", "del", "res"])
    p.add_argument("--analysis", required=True, choices=list(ANALYSES.keys()))
    p.add_argument("--tmap-dir",   required=True, type=Path)
    p.add_argument("--roi-dir",    required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--roi",    default=DEFAULT_ROI,
                   help=f"ROI name (default: {DEFAULT_ROI}).")
    p.add_argument("--voxels", type=int, default=DEFAULT_VOXELS,
                   help=f"Top-K voxel count (default: {DEFAULT_VOXELS}).")
    p.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                   help=f"Parallel pairs (default: {DEFAULT_N_JOBS}). "
                        "Each worker runs its 16 folds serially.")
    return p.parse_args()


# ============================================================================
# Path helpers (identical to 08_decode_svm.py)
# ============================================================================
def tmap_phase_dir(tmap_root: Path, sub: str, phase: str, kind: str) -> Path:
    subdir = "tmap_pair" if kind == "pair" else "tmap_z"
    return tmap_root / f"sub-{sub}" / "model" / "fir" / "binding" / subdir / phase


def find_files(tmap_root: Path, sub: str, phase: str,
               kind: str, label: str) -> list:
    return sorted(tmap_phase_dir(tmap_root, sub, phase, kind).glob(
        f"sub-{sub}_ses-*_block-*_{kind}-{label}.nii.gz"))


def roi_mask_path(roi_dir: Path, sub: str, roi: str, n_vox: int) -> Path:
    return roi_dir / f"sub-{sub}" / roi / f"top{n_vox:04d}.nii.gz"


def resampled_mask_path(rp: Path) -> Path:
    return rp.with_name(rp.name.replace(".nii.gz", "_resampled.nii.gz"))


def ensure_resampled_roi(roi_path: Path, ref_path: Path) -> Path:
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
# Per-pair decoding: VERBATIM copy of 08_decode_svm.py.decode_one_combo,
# with coef collection added inside the existing fold loop.
# ============================================================================
def decode_one_combo_with_coef(
    tmap_root: Path,
    sub: str,
    phase: str,
    kind: str,
    a_label: str,
    b_label: str,
    mask_path: Path,
    coef_out_path: Path,
) -> dict:

    print(f"Running pair {a_label} vs {b_label} - ROI {mask_path.parent.name} "
          f"top{mask_path.stem.split('top')[-1]} ...")

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

        if f"pair-{a_label}.nii.gz" in p.name:
            decoding["conditions"].append("A")

        elif f"pair-{b_label}.nii.gz" in p.name:
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
    # EXACT old manual LOSO, with coef collection added
    # ------------------------------------------------------------------
    accuracy_lst = []
    coef_arrays = []
    last_coef_img = None

    for run in run_lst:

        train_mask = decoding_session != run

        fmri_train = decoding_fmri_niiimgs[train_mask]
        y_train = decoding_conditions[train_mask]
        groups_train = decoding_session[train_mask]

        deco.fit(fmri_train, y_train, groups=groups_train)

        # collect this fold's SVM weight image (class 'A')
        coef_img = deco.coef_img_["A"]
        coef_arrays.append(coef_img.get_fdata())
        last_coef_img = coef_img

        test_mask = decoding_session == run

        fmri_test = decoding_fmri_niiimgs[test_mask]
        y_test = decoding_conditions[test_mask]

        pred = deco.predict(fmri_test)

        acc = np.mean(pred == y_test)

        accuracy_lst.append(acc)

    # ------------------------------------------------------------------
    # Average coefs across folds and save NIfTI
    # ------------------------------------------------------------------
    mean_data = np.mean(np.stack(coef_arrays, axis=0), axis=0).astype(np.float32)
    mean_img = nib.Nifti1Image(
        mean_data,
        affine=last_coef_img.affine,
        header=last_coef_img.header,
    )
    coef_out_path.parent.mkdir(parents=True, exist_ok=True)
    mean_img.to_filename(str(coef_out_path))

    return {
        "accuracy": float(np.mean(accuracy_lst)),
        "n_folds":  int(len(accuracy_lst)),
    }


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()
    kind, pair_list = ANALYSES[args.analysis]

    print("-" * 64)
    print(f"Coef export sub-{args.subject} | {args.analysis} | phase {args.phase}"
          f" | ROI {args.roi} top{args.voxels:04d}")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>14}: {v}")
    print(f"  {'kind':>14}: {kind}")
    print(f"  {'n_pairs':>14}: {len(pair_list)}")
    print("-" * 64)

    # output: <output-dir>/sub-XX/<analysis>/<roi>/
    out_root = (args.output_dir / f"sub-{args.subject}" /
                args.analysis / f"svmcoef_{args.roi}")
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # reference t-map for ROI resampling
    # ------------------------------------------------------------------
    ref_path = None
    for a, _ in pair_list:
        ref_files = find_files(args.tmap_dir, args.subject, args.phase, kind, a)
        if ref_files:
            ref_path = ref_files[0]
            break
    if ref_path is None:
        print("ERROR: no t-map files found. Check --tmap-dir / --phase.")
        return

    # ------------------------------------------------------------------
    # resample mask once (single ROI × voxel-count, so no race condition)
    # ------------------------------------------------------------------
    rp = roi_mask_path(args.roi_dir, args.subject, args.roi, args.voxels)
    if not rp.exists():
        print(f"ERROR: ROI mask not found: {rp}")
        return
    mask_path = ensure_resampled_roi(rp, ref_path)
    print(f"Using mask: {mask_path}")

    # ------------------------------------------------------------------
    # job list: one per pair
    # ------------------------------------------------------------------
    def coef_path(pair_idx: int, a: str, b: str) -> Path:
        return out_root / (f"coef_top{args.voxels:04d}_{args.phase}_"
                           f"pair-{pair_idx:02d}_{a}vs{b}.nii.gz")

    jobs = [(i, a, b, coef_path(i, a, b))
            for i, (a, b) in enumerate(pair_list, start=1)]

    print(f"\nRunning {len(jobs)} pairs on {args.n_jobs} workers ...")
    t0 = time.time()

    def run_one(pair_idx, a, b, coef_out_path):
        try:
            res = decode_one_combo_with_coef(
                args.tmap_dir, args.subject, args.phase, kind,
                a, b, mask_path, coef_out_path,
            )
        except Exception as e:
            print(f"  [error] pair {pair_idx} ({a} vs {b}): {e}")
            return None
        return {
            "subject":   args.subject,
            "phase":     args.phase,
            "analysis":  args.analysis,
            "roi":       args.roi,
            "voxels":    args.voxels,
            "pair":      pair_idx,
            "A_label":   a,
            "B_label":   b,
            "n_folds":   res["n_folds"],
            "accuracy":  res["accuracy"],
            "coef_file": coef_out_path.name,
        }

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(run_one)(*j) for j in jobs
    )
    rows = [r for r in results if r is not None]

    elapsed = time.time() - t0
    print(f"\nFinished {len(rows)}/{len(jobs)} pairs in {elapsed:.1f}s "
          f"({elapsed / max(len(rows), 1):.2f}s avg).")

    # ------------------------------------------------------------------
    # accuracy table
    # ------------------------------------------------------------------
    today = time.strftime("%Y%m%d")
    csv_path = out_root / f"accuracy_top{args.voxels:04d}_{args.phase}_{today}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved accuracy table to {csv_path}")
    if not df.empty:
        print(f"  mean accuracy across pairs: {df['accuracy'].mean():.4f} "
              f"± {df['accuracy'].std():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()