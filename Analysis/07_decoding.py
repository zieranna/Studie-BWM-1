#!/usr/bin/env python3
"""
07_decoding.py
================

ROI-based SVM decoding for the binding WM task.

Five analyses are supported, each defined as a list of A-vs-B label
pairs. The decoder loads one input image per occurrence of A or B
across the experiment, runs Leave-One-Block-Out cross-validation with
nested hyperparameter search, and saves accuracy to an .xlsx table.

Voxel selection (--voxel-selection)
-----------------------------------
fixed  (default; original pipeline, unchanged)
    One decoder per (pair x ROI x voxel count). Accuracy is reported
    separately for every voxel count. nilearn's Decoder tunes C and
    loss with an inner LOGO and averages the inner-fold models.

nested (revision, Reviewer 2 comment 1a)
    The voxel count is treated as a hyperparameter and chosen INSIDE
    each outer training fold, so every (pair x ROI) yields ONE accuracy
    and no voxel count is picked post hoc. For each outer fold
    (held-out block):

      1. An inner LOGO over the 15 training blocks scores every
         candidate voxel count x (C, loss) combination.
      2. The voxel count with the best inner accuracy is selected.
         Default (--selection-scope pooled): one voxel count per ROI and
         fold, chosen on inner accuracy averaged across all decoding
         sets of the analysis. Alternative (per_pair): every set picks
         its own. C and loss are then chosen per set at that count.
         Ties between voxel counts are resolved by --tie-break
         (default: largest).
      3. A LinearSVC with the selected settings is refit on all 15
         training blocks and tested on the held-out block.

    The held-out block never enters any selection step. Patterns are
    loaded into memory once, and fits call scikit-learn's LinearSVC
    directly (same estimator, grid, tol and max_iter as the Decoder,
    but one refit model instead of nilearn's average of inner-fold
    models). Running nested mode with a single --voxel-counts value
    gives a fixed-count result from this engine, which isolates any
    difference between the two engines.

    Note: the top-N voxel RANKING (07_create_top_voxel_rois.py) still
    uses all runs. It is label-blind (Reviewer 2, minor comment 2).

Parallelism
-----------
fixed:  the (pair x ROI x voxel-count) loop is one flat job list.
nested: the (ROI x outer fold) loop is one flat job list.
Each worker runs its fits SERIALLY (no nested Parallel), which avoids
CPU oversubscription.

Resampled ROI masks are cached next to the ROI masks
(``topNNNN_resampled.nii.gz``). The cache is written atomically and
checked against the reference grid, so several jobs (e.g. SLURM array
tasks for the same subject) can run at the same time.

Inputs (--layout)
-----------------
fir (default): maps of the original FIR model (``05_postprocess_tmaps.py``)

    * Pair-averaged maps (tmap_pair/)  -- analyses whose unit of
      decoding is a paper PAIR (binding, color, motion).
    * Per-trial maps (tmap_z/, z-scored; or another folder via
      --trial-subdir, e.g. tmap_trial/ from 03b_average_runs.py) --
      analyses whose unit is a single trial (temporal_order,
      binding_interaction).

    <tmap-dir>/sub-XX/model/fir/binding/{tmap_pair,tmap_z}/<phase>/,
    --phase enc | del | res.

lss: single-trial estimates of ``02b_trialestimates_lss.py`` (pair maps
     from ``03b_average_runs.py``)

    <tmap-dir>/sub-XX/<model>/<stat>/<window>/pairs/    pair analyses
    <tmap-dir>/sub-XX/<model>/<stat>/<window>/trials/   trial analyses

    --phase gives <model>/<stat>/<window>, e.g. hrf/beta/del or
    fir/tmap/del_w10-13. Output file names use it with '-' instead of
    '/' (hrf-beta-del).

ROI masks come from ``07_create_top_voxel_rois.py``.

Cross-validation
----------------
Group = (session, block). 4 sessions x 4 blocks = 16 groups, so
LeaveOneGroupOut gives 16 outer folds. Hyperparameter search (and, in
nested mode, voxel-count selection) runs as nested LOGO inside each
outer fold.

Output
------
<phase> in file names is the phase tag (--layout lss: hrf-beta-del).
fixed:
    <output-dir>/sub-XX/<analysis>/decoding_<analysis>_<phase>_<date>.xlsx
      sheet 'per_pair' -- one row per (pair, ROI, voxels)
      sheet 'summary'  -- mean and std accuracy per (ROI, voxels)
nested:
    <output-dir>/sub-XX/<analysis>/decoding_<analysis>_<phase>_nested-<scope>[_C<c>]_<date>.xlsx
      sheet 'per_pair'     -- one row per (pair, ROI): accuracy across folds
      sheet 'summary'      -- mean and std accuracy per ROI (voxels = 'nested')
      sheet 'voxel_choice' -- how often each voxel count was selected
      sheet 'folds'        -- one row per (pair, ROI, fold): choice + accuracy
      sheet 'inner_curve'  -- mean inner-CV accuracy per (ROI, voxel count)
Both modes also save a copy of this script next to the table.
"""

from __future__ import annotations

import argparse
import os
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
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ============================================================================
# Constants
# ============================================================================
N_BLOCKS_PER_SESSION = 4
FIR_PHASES = ("enc", "del", "res")      # --phase values for --layout fir

# Each analysis: (kind, [(A_label, B_label), ...]).
#   kind="pair"  -> reads tmap_pair/ (lss: pairs/),  labels are paper pair numbers ('4', '13')
#   kind="trial" -> reads tmap_z/    (lss: trials/), labels are PAPER trial labels ('4a', '13b')
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
    "visual_all":   _LARGE,
    "visual_locMT": _LARGE,
    "MT_complex":   _SMALL,
    "dlpfc_noSFL":  _SMALL,
}

# SVM hyperparameters, shared by both modes (identical to the original grid).
C_GRID = [0.001, 0.01, 1.0, 10, 100, 1000]
LOSS_GRID = ["hinge", "squared_hinge"]
SVC_TOL = 1e-5
SVC_MAX_ITER = 1_000_000
PARAM_GRID = [{
    "C": C_GRID,
    "dual": [True],
    "loss": LOSS_GRID,
    "penalty": ["l2"],
    "tol": [SVC_TOL],
    "max_iter": [SVC_MAX_ITER],
}]

# Default parallelism: 32 outer workers, fits run SERIALLY inside each.
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
    p.add_argument("--phase", required=True,
                   help="--layout fir: FIR phase (enc, del, res). --layout lss: "
                        "<model>/<stat>/<window>, e.g. hrf/beta/del, "
                        "fir/tmap/del_w10-13.")
    p.add_argument("--analysis", required=True, choices=list(ANALYSES.keys()),
                   help="Which A-vs-B label list to use.")
    p.add_argument("--tmap-dir", required=True, type=Path,
                   help="--layout fir: root of 05_postprocess_tmaps.py output. "
                        "--layout lss: root of 02b_trialestimates_lss.py output. "
                        "In both cases the folder that contains sub-XX/.")
    p.add_argument("--roi-dir", required=True, type=Path,
                   help="Root of 07_create_top_voxel_rois.py output (the "
                        "folder that contains sub-XX/<ROI>/topNNNN.nii.gz).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where decoding tables will be written.")
    p.add_argument("--layout", choices=["fir", "lss"], default="fir",
                   help="'fir' (default): original FIR maps, "
                        "sub-XX/model/fir/binding/{tmap_pair,tmap_z}/<phase>/. "
                        "'lss': LSS estimates, "
                        "sub-XX/<model>/<stat>/<window>/{pairs,trials}/.")
    p.add_argument("--trial-subdir", default="tmap_z",
                   help="--layout fir only. Folder of the single-trial maps for "
                        "trial-level analyses (temporal_order, "
                        "binding_interaction): 'tmap_z' (z-scored, "
                        "05_postprocess_tmaps.py, default) or 'tmap_trial' "
                        "(not z-scored, 03b_average_runs.py).")
    p.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                   help=f"Outer parallel workers (default {DEFAULT_N_JOBS}). "
                        "Each worker runs its fits serially inside.")
    p.add_argument("--rois", nargs="+", choices=list(ROI_VOXELS), default=None,
                   help="Restrict to these ROIs (default: all).")

    # ---- voxel selection -------------------------------------------------
    p.add_argument("--voxel-selection", choices=["fixed", "nested"],
                   default="fixed",
                   help="'fixed' (default): one accuracy per voxel count, "
                        "original pipeline. 'nested': voxel count chosen "
                        "inside each training fold (Reviewer 2, comment 1a).")
    p.add_argument("--selection-scope", choices=["pooled", "per_pair"],
                   default="pooled",
                   help="nested only. 'pooled' (default): one voxel count per "
                        "ROI and fold, chosen on inner accuracy averaged "
                        "across all decoding sets. 'per_pair': each set "
                        "chooses its own voxel count.")
    p.add_argument("--tie-break", choices=["smallest", "median", "largest"],
                   default="largest",
                   help="nested only. Voxel count to take when several tie "
                        "on inner accuracy (default: largest).")
    p.add_argument("--fixed-c", type=float, default=None,
                   help="nested only. Fix C (loss = squared_hinge) and tune "
                        "only the voxel count.")
    p.add_argument("--voxel-counts", type=int, nargs="+", default=None,
                   help="nested only. Candidate voxel counts for every ROI "
                        "(overrides ROI_VOXELS). A single value gives a "
                        "fixed-count run with the nested engine.")
    p.add_argument("--seed", type=int, default=0,
                   help="nested only. random_state for LinearSVC.")
    args = p.parse_args()

    if args.layout == "fir":
        if args.phase not in FIR_PHASES:
            p.error(f"--layout fir: --phase must be one of {', '.join(FIR_PHASES)} "
                    f"(got '{args.phase}'). For LSS estimates add --layout lss.")
    else:
        args.phase = args.phase.strip("/")
        if len(args.phase.split("/")) != 3:
            p.error("--layout lss: --phase must be <model>/<stat>/<window>, "
                    f"e.g. hrf/beta/del (got '{args.phase}').")
        args.trial_subdir = "lss"      # routes all path look-ups to the LSS layout
    args.phase_tag = args.phase.replace("/", "-")   # for file names: hrf-beta-del
    return args


# ============================================================================
# Path helpers
# ============================================================================
def tmap_phase_dir(tmap_root: Path, sub: str, phase: str, kind: str,
                   trial_subdir: str = "tmap_z") -> Path:
    """Folder with the maps of one phase and kind.

    trial_subdir == "lss" selects the LSS layout of 02b_trialestimates_lss.py:
    <root>/sub-XX/<model>/<stat>/<window>/{pairs,trials}/, phase =
    "<model>/<stat>/<window>".
    """
    if trial_subdir == "lss":
        return (tmap_root / f"sub-{sub}" / phase /
                ("pairs" if kind == "pair" else "trials"))
    subdir = "tmap_pair" if kind == "pair" else trial_subdir
    return tmap_root / f"sub-{sub}" / "model" / "fir" / "binding" / subdir / phase


def find_files(tmap_root: Path, sub: str, phase: str,
               kind: str, label: str, trial_subdir: str = "tmap_z") -> list:
    # '*' after block- also covers the run entity of LSS trial maps
    # (sub-XX_ses-YY_block-BB_run-NN_trial-4a.nii.gz).
    return sorted(tmap_phase_dir(tmap_root, sub, phase, kind, trial_subdir).glob(
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
def _on_grid(img, ref) -> bool:
    """True if img has the same 3-D shape and affine as ref."""
    return (img.shape[:3] == ref.shape[:3]
            and np.allclose(img.affine, ref.affine, atol=1e-3))


def ensure_resampled_roi(roi_path: Path, ref_path: Path) -> Path:
    """Cache a ROI mask resampled to the reference t-map grid.

    The cached file is reused only if it is readable and on the reference
    grid. A new cache is written to a temporary file and moved into place
    with an atomic rename, so concurrent jobs never read a partial file.
    """
    rp_rs = resampled_mask_path(roi_path)
    ref = nib.load(str(ref_path))
    if rp_rs.exists():
        try:
            cached = nib.load(str(rp_rs))
            np.asanyarray(cached.dataobj)          # fails on a truncated file
            if _on_grid(cached, ref):
                return rp_rs
            print(f"  [info] {rp_rs.name} is on another grid; rebuilding.")
        except Exception:
            print(f"  [info] {rp_rs.name} is unreadable; rebuilding.")
    roi = nib.load(str(roi_path))
    roi_rs = resample_img(roi,
                          target_affine=ref.affine,
                          target_shape=ref.shape[:3],
                          interpolation="nearest")
    tmp = rp_rs.with_name(f".tmp_{os.getpid()}_{rp_rs.name}")
    nib.save(roi_rs, str(tmp))
    os.replace(tmp, rp_rs)                         # atomic on the same filesystem
    return rp_rs


# ============================================================================
# FIXED mode: single-combo decoding (one pair x ROI x voxel-count)
# (original code; only the parameter grid now comes from PARAM_GRID)
# ============================================================================
def decode_one_combo(
    tmap_root: Path,
    sub: str,
    phase: str,
    kind: str,
    a_label: str,
    b_label: str,
    mask_path: Path,
    trial_subdir: str = "tmap_z",
) -> float:

    print(f"Running pair {a_label} vs {b_label} - ROI {mask_path.parent.name} top{mask_path.stem.split('top')[-1]} ...")

    decoding = {
        "fmri_niiimgs": [],
        "conditions": [],
        "session": [],
    }

    imgs = []

    files_a = find_files(tmap_root, sub, phase, kind, a_label, trial_subdir)
    files_b = find_files(tmap_root, sub, phase, kind, b_label, trial_subdir)

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
    cv = LeaveOneGroupOut()

    deco = Decoder(
        estimator="svc",
        mask=str(mask_path),
        standardize=False,
        cv=cv,
        screening_percentile=100,
        scoring="accuracy",
        n_jobs=1,
        param_grid=PARAM_GRID,
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


def run_fixed(args, kind, pair_list, roi_voxels, ref_path, out_dir, today):
    """Original pipeline: one accuracy per (pair x ROI x voxel count)."""
    # ------------------------------------------------------------------
    # Pre-resample every ROI mask once, before the parallel section.
    # Eliminates the race condition where multiple workers try to
    # write the same _resampled.nii.gz concurrently.
    # ------------------------------------------------------------------
    print(f"Pre-resampling ROI masks to grid of {ref_path.name} ...")
    n_resampled = 0
    for roi, voxel_list in roi_voxels.items():
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
        for roi, voxel_list in roi_voxels.items():
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
                kind, a, b, mask_path, args.trial_subdir,
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
    out_xlsx = out_dir / f"decoding_{args.analysis}_{args.phase_tag}_{today}.xlsx"
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
    return f"decoding_script_{args.analysis}_{args.phase_tag}_{today}.py"


# ============================================================================
# NESTED mode: voxel count selected inside each outer training fold
# ============================================================================
def make_svc(C: float, loss: str, seed: int) -> LinearSVC:
    """Same estimator settings as the Decoder grid in fixed mode."""
    return LinearSVC(C=C, loss=loss, penalty="l2", dual=True,
                     tol=SVC_TOL, max_iter=SVC_MAX_ITER, random_state=seed)


def pick_index(scores: np.ndarray, tie_break: str) -> int:
    """Index of the best score; ties resolved by position (scores are
    ordered by ascending voxel count, or by grid order for C/loss)."""
    tied = np.flatnonzero(np.isclose(scores, np.nanmax(scores)))
    if tie_break == "smallest":
        return int(tied[0])
    if tie_break == "largest":
        return int(tied[-1])
    return int(tied[(len(tied) - 1) // 2])  # median (lower if even)


def load_roi_indices(roi_dir: Path, sub: str, roi_voxels: dict,
                     ref_path: Path) -> dict:
    """{roi: {n_vox: flat C-order voxel indices on the t-map grid}}."""
    ref = nib.load(str(ref_path))
    out = {}
    for roi, voxel_list in roi_voxels.items():
        idx = {}
        for n_vox in voxel_list:
            rp = roi_mask_path(roi_dir, sub, roi, n_vox)
            if not rp.exists():
                print(f"  [warn] missing mask {rp}")
                continue
            rs = nib.load(str(ensure_resampled_roi(rp, ref_path)))
            if not _on_grid(rs, ref):
                # Another job replaced the cache with a different grid
                # between the check and this load; never use wrong indices.
                raise RuntimeError(f"{resampled_mask_path(rp)} is not on the grid "
                                   f"of {ref_path.name}; rerun this job.")
            m = rs.get_fdata()
            idx[n_vox] = np.flatnonzero(m.reshape(-1) > 0)
        if not idx:
            continue
        counts = sorted(idx)
        nested = all(np.isin(idx[a], idx[b]).all()
                     for a, b in zip(counts[:-1], counts[1:]))
        print(f"  {roi:>13}: candidates {counts}; voxels after resampling "
              f"{[len(idx[n]) for n in counts]}; masks nested: {nested}")
        out[roi] = idx
    return out


def load_patterns(tmap_root: Path, sub: str, phase: str, kind: str,
                  labels: list, flat_idx: np.ndarray, ref_img,
                  trial_subdir: str = "tmap_z") -> dict:
    """Load every map once: {label: (X [n_blocks, n_vox], block ids)}."""
    data = {}
    for lab in labels:
        files = find_files(tmap_root, sub, phase, kind, lab, trial_subdir)
        if not files:
            raise FileNotFoundError(
                f"No maps for label {lab} in "
                f"{tmap_phase_dir(tmap_root, sub, phase, kind, trial_subdir)}")
        X = np.empty((len(files), flat_idx.size), dtype=np.float32)
        g = np.empty(len(files), dtype=int)
        for r, f in enumerate(files):
            img = nib.load(str(f))
            if not _on_grid(img, ref_img):
                raise ValueError(f"{f.name} is not on the reference grid")
            X[r] = img.get_fdata(dtype=np.float32).reshape(-1)[flat_idx]
            g[r] = parse_block_id(f)
        if len(np.unique(g)) != len(g):
            raise ValueError(f"Label {lab}: more than one map per block")
        data[lab] = (np.nan_to_num(X), g)
    return data


def nested_fold_job(roi, test_group, data, pos_by_n, pair_list, params,
                    scope, tie_break, seed):
    """One ROI x one held-out block, all decoding sets of the analysis.

    Returns (fold_rows, curve_rows, error_message_or_None).
    """
    warnings.filterwarnings("ignore")  # loky workers do not inherit filters
    try:
        counts = sorted(pos_by_n)                     # ascending voxel counts
        n_sets, n_v, n_p = len(pair_list), len(counts), len(params)
        inner = np.full((n_sets, n_v, n_p), np.nan)   # inner-CV accuracy
        splits = [None] * n_sets

        # ---- 1. inner LOGO on the training blocks only ------------------
        for k, (a, b) in enumerate(pair_list):
            (XA, gA), (XB, gB) = data[a], data[b]
            X = np.vstack([XA, XB])
            y = np.r_[np.zeros(len(gA), int), np.ones(len(gB), int)]
            g = np.r_[gA, gB]
            tr, te = g != test_group, g == test_group
            if not te.any():
                continue                              # set has no map in this block
            Xtr, ytr, gtr = X[tr], y[tr], g[tr]
            splits[k] = (Xtr, ytr, X[te], y[te])
            inner_cv = list(LeaveOneGroupOut().split(Xtr, ytr, gtr))
            for i, n in enumerate(counts):
                Xn = Xtr[:, pos_by_n[n]]
                for j, (C, loss) in enumerate(params):
                    accs = []
                    for itr, ite in inner_cv:
                        clf = make_svc(C, loss, seed).fit(Xn[itr], ytr[itr])
                        accs.append(np.mean(clf.predict(Xn[ite]) == ytr[ite]))
                    inner[k, i, j] = np.mean(accs)

        # ---- 2. select the voxel count -----------------------------------
        best_over_params = np.nanmax(inner, axis=2)   # (sets, voxel counts)
        if scope == "pooled":
            i_pooled = pick_index(np.nanmean(best_over_params, axis=0),
                                  tie_break)
            i_sel = [i_pooled] * n_sets
        else:
            i_sel = [pick_index(best_over_params[k], tie_break)
                     if splits[k] is not None else None
                     for k in range(n_sets)]

        # ---- 3. refit on all training blocks, test on held-out block -----
        rows, curve = [], []
        for k, (a, b) in enumerate(pair_list):
            if splits[k] is None:
                continue
            Xtr, ytr, Xte, yte = splits[k]
            i = i_sel[k]
            n = counts[i]
            j = pick_index(inner[k, i], "smallest")   # first in grid order
            C, loss = params[j]
            cols = pos_by_n[n]
            clf = make_svc(C, loss, seed).fit(Xtr[:, cols], ytr)
            acc = float(np.mean(clf.predict(Xte[:, cols]) == yte))
            rows.append({
                "pair": k + 1, "A_label": a, "B_label": b, "ROI": roi,
                "fold": int(test_group), "voxels": n,
                "voxels_actual": int(len(cols)), "C": C, "loss": loss,
                "inner_accuracy": float(inner[k, i, j]),
                "test_accuracy": acc,
            })
            for i2, n2 in enumerate(counts):
                curve.append({
                    "pair": k + 1, "ROI": roi, "fold": int(test_group),
                    "voxels": n2,
                    "inner_accuracy": float(best_over_params[k, i2]),
                })
        return rows, curve, None
    except Exception as e:
        return [], [], f"{roi} fold {test_group}: {e}"


def run_nested(args, kind, pair_list, roi_voxels, ref_path, out_dir, today):
    """Voxel count chosen inside each outer training fold (R2 comment 1a)."""
    if args.voxel_counts:
        roi_voxels = {r: sorted(set(args.voxel_counts)) for r in roi_voxels}

    if args.fixed_c is None:
        params = [(C, loss) for C in C_GRID for loss in LOSS_GRID]
    else:
        params = [(args.fixed_c, "squared_hinge")]

    print("Loading ROI masks ...")
    roi_idx = load_roi_indices(args.roi_dir, args.subject, roi_voxels, ref_path)
    if not roi_idx:
        raise FileNotFoundError("No ROI masks found. Check --roi-dir.")

    # Load each map once, restricted to the union of all ROI masks.
    all_idx = np.unique(np.concatenate(
        [v for d in roi_idx.values() for v in d.values()]))
    labels = list(dict.fromkeys(lab for pair in pair_list for lab in pair))
    print(f"Loading maps for {len(labels)} labels "
          f"({all_idx.size} voxels in the ROI union) ...")
    data_all = load_patterns(args.tmap_dir, args.subject, args.phase, kind,
                             labels, all_idx, nib.load(str(ref_path)),
                             args.trial_subdir)
    groups = np.unique(np.concatenate([g for _, g in data_all.values()]))
    print(f"  {sum(len(g) for _, g in data_all.values())} maps, "
          f"{len(groups)} blocks (outer folds).")

    jobs = []
    for roi, idx in roi_idx.items():
        roi_union = np.unique(np.concatenate(list(idx.values())))
        cols = np.searchsorted(all_idx, roi_union)
        pos_by_n = {n: np.searchsorted(roi_union, v) for n, v in idx.items()}
        data_roi = {lab: (X[:, cols], g) for lab, (X, g) in data_all.items()}
        for test_group in groups:
            jobs.append((roi, int(test_group), data_roi, pos_by_n))

    n_fits = sum(len(p) for *_, p in jobs) * len(pair_list) * len(params) \
        * (len(groups) - 1)
    print(f"\nRunning {len(jobs)} ROI x fold jobs (~{n_fits:,} inner fits) "
          f"on {args.n_jobs} workers ...")
    print(f"  scope={args.selection_scope}, tie-break={args.tie_break}, "
          f"C={'tuned' if args.fixed_c is None else args.fixed_c}")
    t0 = time.time()

    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(nested_fold_job)(roi, tg, d, pos, pair_list, params,
                                 args.selection_scope, args.tie_break,
                                 args.seed)
        for roi, tg, d, pos in jobs
    )
    errors = [e for *_, e in results if e is not None]
    for e in errors:
        print(f"  [error] {e}")
    folds = pd.DataFrame([r for rows, _, _ in results for r in rows])
    curve = pd.DataFrame([c for _, cs, _ in results for c in cs])
    print(f"\nFinished {len(jobs) - len(errors)}/{len(jobs)} jobs "
          f"in {time.time() - t0:.1f}s.")
    if folds.empty:
        raise RuntimeError("No results. See errors above.")

    tag = f"nested-{args.selection_scope}"
    if args.fixed_c is not None:
        tag += f"_C{args.fixed_c:g}"
    out_xlsx = out_dir / f"decoding_{args.analysis}_{args.phase_tag}_{tag}_{today}.xlsx"

    # Safety net: save the raw fold table before aggregating.
    folds.to_excel(out_xlsx, sheet_name="folds", index=False)

    meta = {
        "subject": args.subject, "phase": args.phase,
        "analysis": args.analysis, "selection": tag,
        "tie_break": args.tie_break,
    }

    per_pair = (folds.groupby(["pair", "A_label", "B_label", "ROI"], sort=False)
                     .agg(accuracy=("test_accuracy", "mean"),
                          n_folds=("fold", "size"),
                          median_voxels=("voxels", "median"),
                          modal_voxels=("voxels",
                                        lambda v: int(v.mode().min())))
                     .reset_index())
    for i, (k, v) in enumerate(meta.items()):
        per_pair.insert(i, k, v)

    summary = (per_pair.groupby("ROI", sort=False)
                       .agg(mean_accuracy=("accuracy", "mean"),
                            std_accuracy=("accuracy", "std"),
                            n_pairs=("accuracy", "size"))
                       .reset_index())
    summary.insert(1, "voxels", "nested")   # same columns as fixed mode
    summary[["mean_accuracy", "std_accuracy"]] = \
        summary[["mean_accuracy", "std_accuracy"]].round(4)

    choice = (folds.groupby(["ROI", "voxels"], sort=False).size()
                   .rename("n_chosen").reset_index()
                   .sort_values(["ROI", "voxels"]))
    choice["share"] = (choice["n_chosen"]
                       / choice.groupby("ROI")["n_chosen"].transform("sum")
                       ).round(4)

    inner_curve = (curve.groupby(["ROI", "voxels"], sort=False)
                        .agg(mean_inner_accuracy=("inner_accuracy", "mean"))
                        .reset_index().round(4))

    print("\n" + "-" * 64)
    print("Nested decoding: mean accuracy across pairs (per ROI):")
    print("-" * 64)
    print(summary.to_string(index=False))
    print("\nSelected voxel counts (share of folds):")
    print(choice.to_string(index=False))

    with pd.ExcelWriter(out_xlsx) as xl:
        per_pair.to_excel(xl, sheet_name="per_pair", index=False)
        summary.to_excel(xl, sheet_name="summary", index=False)
        choice.to_excel(xl, sheet_name="voxel_choice", index=False)
        folds.to_excel(xl, sheet_name="folds", index=False)
        inner_curve.to_excel(xl, sheet_name="inner_curve", index=False)
    print(f"Saved {out_xlsx}")
    return f"decoding_script_{args.analysis}_{args.phase_tag}_{tag}_{today}.py"


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()
    kind, pair_list = ANALYSES[args.analysis]
    roi_voxels = {r: v for r, v in ROI_VOXELS.items()
                  if args.rois is None or r in args.rois}

    print("-" * 64)
    print(f"Decoding sub-{args.subject} | {args.analysis} | phase {args.phase}")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>15}: {v}")
    print(f"  {'kind':>15}: {kind}")
    print(f"  {'n_pairs':>15}: {len(pair_list)}")
    print(f"  {'input':>15}: "
          f"{tmap_phase_dir(args.tmap_dir, args.subject, args.phase, kind, args.trial_subdir)}")
    print("-" * 64)

    out_dir = args.output_dir / f"sub-{args.subject}" / args.analysis
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Find one reference t-map for ROI resampling.
    # ------------------------------------------------------------------
    ref_path = None
    for a, _ in pair_list:
        ref_files = find_files(args.tmap_dir, args.subject, args.phase, kind, a,
                               args.trial_subdir)
        if ref_files:
            ref_path = ref_files[0]
            break
    if ref_path is None:
        # Non-zero exit so the SLURM task is marked FAILED, not COMPLETED.
        raise SystemExit(
            "ERROR: no maps found in "
            f"{tmap_phase_dir(args.tmap_dir, args.subject, args.phase, kind, args.trial_subdir)}. "
            "Check --tmap-dir, --phase and --layout.")

    today = time.strftime("%Y%m%d")
    if args.voxel_selection == "nested":
        snapshot_name = run_nested(args, kind, pair_list, roi_voxels,
                                   ref_path, out_dir, today)
    else:
        snapshot_name = run_fixed(args, kind, pair_list, roi_voxels,
                                  ref_path, out_dir, today)

    snapshot = out_dir / snapshot_name
    shutil.copy(__file__, str(snapshot))
    print(f"Snapshot at {snapshot}")

    print("Done.")


if __name__ == "__main__":
    main()