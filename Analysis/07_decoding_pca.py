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

allcounts (specification curve over voxel counts)
    Accuracy for EVERY candidate voxel count separately, no selection: the
    nested engine (refit LinearSVC on all training blocks) with C fixed
    (--fixed-c) or tuned per count in an inner LOGO. Output like the fixed
    mode (per_pair / summary per ROI x voxels), tag allcounts[_C<c>].

mask (Rademaker et al., 2019)
    All voxels of one mask per ROI from --mask-dir (sub-XX/<ROI>.nii.gz),
    e.g. localizer-responsive voxels (07_create_responsive_rois.py): no
    voxel count, no PCA. C fixed (--fixed-c) or tuned in an inner LOGO.
    ROIs below --min-voxels (default 20) are skipped. Tag mask-<folder>.

pca (alternative without voxel selection)
    Every voxel of the whole ROI is used. In each split the PCA is fitted
    on the TRAINING blocks only (--pca-fit analysis: all labels of the
    analysis; pair: the two labels of the set) and the held-out block is
    projected onto those components. The number of components is the
    hyperparameter, chosen in the inner LOGO exactly like the voxel count
    in nested mode (same --selection-scope, --tie-break, C grid).
    Whole-ROI masks: --pca-roi-dir (flat folder such as roi_anat:
    sub-XX/<ROI>_resampled.nii.gz if on the map grid, else <ROI>.nii.gz).
    Candidates (--n-components) are variance fractions by default (fewest
    components explaining >= x of the training variance, recomputed in every
    PCA fit), optionally fixed numbers; tag pcavar-... with fractions.
    Candidate 'all' keeps every component, which is equivalent to a
    linear SVM on all ROI voxels (up to liblinear's penalized intercept),
    so "no reduction" is always among the candidates. Output as in nested
    mode, tag pca-<fit>-<scope>[_C<c>][_zscore]; the 'voxels' columns then
    hold the number of components (column 'unit' = components).

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
    p.add_argument("--voxel-selection", choices=["fixed", "nested", "allcounts", "pca", "mask"],
                   default="fixed",
                   help="'fixed' (default): one accuracy per voxel count, "
                        "original pipeline. 'nested': voxel count chosen "
                        "inside each training fold (Reviewer 2, comment 1a). "
                        "'pca': whole ROI, PCA fitted on the training blocks, "
                        "number of components chosen inside each training fold. "
                        "'allcounts': accuracy for EVERY voxel count, no selection "
                        "(nested engine; C fixed with --fixed-c or tuned per count). "
                        "'mask': all voxels of one mask per ROI (--mask-dir), e.g. "
                        "localizer-responsive voxels after Rademaker et al. (2019).")
    p.add_argument("--mask-dir", type=Path, default=None,
                   help="mask only. Flat folder <dir>/sub-XX/<ROI>.nii.gz, e.g. the "
                        "output of 07_create_responsive_rois.py.")
    p.add_argument("--min-voxels", type=int, default=20,
                   help="mask only. Skip ROIs with fewer voxels (Rademaker et al.: 20).")
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
    p.add_argument("--n-components", nargs="+",
                   default=["0.5", "0.7", "0.8", "0.9", "0.95", "0.99", "all"],
                   help="pca only. Candidates, chosen inside each training fold like "
                        "the voxel count: variance fractions ('0.9' or '90%%' = fewest "
                        "components explaining >= 90 %% of the training variance, "
                        "recomputed for every PCA fit), fixed numbers ('20'), and "
                        "'all' (no reduction, equivalent to all ROI voxels). "
                        "Default: 0.5 0.7 0.8 0.9 0.95 0.99 all. Runs with variance "
                        "fractions are named pcavar-..., others pca-...")
    p.add_argument("--pca-fit", choices=["analysis", "pair"], default="analysis",
                   help="pca only. Maps the PCA is fitted on (training blocks only): "
                        "'analysis' (default) = all labels of the analysis (more maps, "
                        "more stable components; labels unused), 'pair' = the two "
                        "labels of each decoding set.")
    p.add_argument("--pca-roi-dir", type=Path, default=None,
                   help="pca only. Folder with the whole-ROI masks, flat per "
                        "subject: <dir>/sub-XX/<ROI>.nii.gz (e.g. roi_anat). Uses "
                        "<ROI>_resampled.nii.gz if it is on the map grid, else "
                        "<ROI>.nii.gz resampled to it. Without this option the "
                        "mask is searched in <roi-dir>/sub-XX/<ROI>/.")
    p.add_argument("--pca-roi-file", default=None,
                   help="pca only. File name of the whole-ROI mask inside "
                        "<roi-dir>/sub-XX/<ROI>/ ({roi} = ROI name), e.g. "
                        "'{roi}_mask.nii.gz'. Default: tries " + ", ".join(WHOLE_ROI_NAMES) +
                        ", else the largest top-N mask (with a warning).")
    p.add_argument("--seed", type=int, default=0,
                   help="nested only. random_state for LinearSVC.")
    p.add_argument("--standardize", choices=["none", "zscore_sample"], default="none",
                   help="nested only. 'zscore_sample': z-score every voxel with the mean and "
                        "SD (ddof=1) of the training data of each split, applied to its test "
                        "data (no leakage). Adds '_zscore' to the output name.")
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


def scale_split(Xa, Xb, standardize):
    """z-score every voxel with mean and SD of the training part Xa (ddof=1, as
    nilearn's 'zscore_sample') and apply them to Xa and Xb. Voxels without variance
    keep SD 1. 'none' returns the data unchanged."""
    if standardize == "none":
        return Xa, Xb
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (Xa - mu) / sd, (Xb - mu) / sd


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
                    scope, tie_break, seed, standardize="none"):
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
            # optional z-scoring with the training part of each inner split
            inner_sets = [scale_split(Xtr[itr], Xtr[ite], standardize) + (ytr[itr], ytr[ite])
                          for itr, ite in inner_cv]
            for i, n in enumerate(counts):
                cols = pos_by_n[n]
                for j, (C, loss) in enumerate(params):
                    accs = []
                    for Atr, Ate, ya, yb in inner_sets:
                        clf = make_svc(C, loss, seed).fit(Atr[:, cols], ya)
                        accs.append(np.mean(clf.predict(Ate[:, cols]) == yb))
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
            Atr, Ate = scale_split(Xtr, Xte, standardize)     # all training blocks -> held-out block
            clf = make_svc(C, loss, seed).fit(Atr[:, cols], ytr)
            acc = float(np.mean(clf.predict(Ate[:, cols]) == yte))
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


def write_nested_outputs(args, folds, curve, tag, out_dir, today, unit="voxels",
                         label=None):
    """Aggregate the fold table and write the workbook (nested and PCA modes).

    In PCA mode the numbers are PCA components, not voxels. The written
    columns are named accordingly (n_components_*, roi_voxels,
    variance_explained); only 'voxels' in the sheet voxel_choice keeps its
    name (read by the group-statistics script), with unit = 'components'.
    """
    pca = unit == "components"
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
    label = label or ("nested" if unit == "voxels" else "pca")
    summary.insert(1, "voxels", label)
    summary[["mean_accuracy", "std_accuracy"]] = \
        summary[["mean_accuracy", "std_accuracy"]].round(4)

    choice = (folds.groupby(["ROI", "voxels"], sort=False).size()
                   .rename("n_chosen").reset_index()
                   .sort_values(["ROI", "voxels"]))
    choice["share"] = (choice["n_chosen"]
                       / choice.groupby("ROI")["n_chosen"].transform("sum")
                       ).round(4)
    choice["unit"] = unit

    inner_curve = (curve.groupby(["ROI", "voxels"], sort=False)
                        .agg(mean_inner_accuracy=("inner_accuracy", "mean"))
                        .reset_index().round(4))
    inner_curve["unit"] = unit

    what = "voxel counts" if unit == "voxels" else "numbers of PCA components"
    print("\n" + "-" * 64)
    print(f"{label} decoding: mean accuracy across pairs (per ROI):")
    print("-" * 64)
    print(summary.to_string(index=False))
    print(f"\nSelected {what} (share of folds):")
    print(choice.to_string(index=False))

    if pca:
        # name the columns after what they hold (PCA components, not voxels)
        extra = (folds.groupby(["pair", "ROI"], sort=False)
                      .agg(roi_voxels=("roi_voxels", "first"),
                           median_variance_explained=("variance_explained", "median"))
                      .reset_index())
        per_pair = (per_pair.rename(columns={"median_voxels": "median_n_components",
                                             "modal_voxels": "modal_n_components"})
                            .merge(extra, on=["pair", "ROI"], how="left"))
        summary = summary.rename(columns={"voxels": "selection"})
        folds = folds.rename(columns={"voxels": "n_components_used",
                                      "n_components": "chosen_fraction"})
        inner_curve = inner_curve.rename(columns={"voxels": "candidate"})

    with pd.ExcelWriter(out_xlsx) as xl:
        per_pair.to_excel(xl, sheet_name="per_pair", index=False)
        summary.to_excel(xl, sheet_name="summary", index=False)
        choice.to_excel(xl, sheet_name="voxel_choice", index=False)
        folds.to_excel(xl, sheet_name="folds", index=False)
        inner_curve.to_excel(xl, sheet_name="inner_curve", index=False)
    print(f"Saved {out_xlsx}")
    return f"decoding_script_{args.analysis}_{args.phase_tag}_{tag}_{today}.py"


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
                                 args.seed, args.standardize)
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
    if args.standardize != "none":
        tag += "_zscore"
    return write_nested_outputs(args, folds, curve, tag, out_dir, today)


# ============================================================================
# PCA mode: whole ROI, number of PCA components selected inside each fold
# ============================================================================
# File names tried (in this order) for the whole-ROI mask in sub-XX/<ROI>/,
# unless --pca-roi-file is given. {roi} is replaced by the ROI name.
WHOLE_ROI_NAMES = ("{roi}.nii.gz", "mask.nii.gz", "full.nii.gz", "all.nii.gz")


def whole_roi_candidates(pca_roi_dir, sub: str, roi: str, roi_file) -> list:
    """Whole-ROI masks in a flat folder <pca_roi_dir>/sub-XX/ (e.g. roi_anat),
    in order of preference: --pca-roi-file, else {roi}_resampled.nii.gz
    (used only if it is on the map grid), then {roi}.nii.gz."""
    d = Path(pca_roi_dir) / f"sub-{sub}"
    names = [roi_file] if roi_file else ["{roi}_resampled.nii.gz", "{roi}.nii.gz"]
    return [d / n.format(roi=roi) for n in names if (d / n.format(roi=roi)).exists()]


def whole_roi_path(roi_dir: Path, sub: str, roi: str, roi_file, voxel_list):
    """(path, note) of the mask that defines the whole ROI."""
    d = roi_dir / f"sub-{sub}" / roi
    for name in ([roi_file] if roi_file else WHOLE_ROI_NAMES):
        p = d / name.format(roi=roi)
        if p.exists():
            return p, "whole ROI"
    if roi_file:
        return None, f"missing {d / roi_file.format(roi=roi)}"
    for n in sorted(voxel_list, reverse=True):
        p = roi_mask_path(roi_dir, sub, roi, n)
        if p.exists():
            return p, (f"WARNING: no whole-ROI mask in {d}, using the largest "
                       f"top-N mask top{n:04d} (set --pca-roi-file)")
    return None, f"no mask in {d}"


def parse_components(tokens) -> list:
    """Candidates for the number of PCA components.

    '0.9' or '90%'  -> ('var', 0.9): fewest components explaining >= 90 % of the
                       training variance (recomputed for every PCA fit)
    '20'            -> ('n', 20): a fixed number of components
    'all'           -> None: every component (no reduction)
    Order (relevant for --tie-break): variance fractions ascending, then fixed
    numbers ascending, then 'all'.
    """
    fracs, ints, has_all = set(), set(), False
    for t in map(str, tokens):
        t = t.strip().lower()
        if t == "all":
            has_all = True
        elif t.endswith("%"):
            fracs.add(float(t[:-1]) / 100)
        elif "." in t:
            fracs.add(float(t))
        else:
            ints.add(int(t))
    if any(not 0 < f < 1 for f in fracs):
        raise ValueError("variance fractions must lie between 0 and 1 (or 0 % and 100 %)")
    if any(i < 1 for i in ints):
        raise ValueError("numbers of components must be >= 1")
    return ([("var", f) for f in sorted(fracs)] + [("n", i) for i in sorted(ints)]
            + ([None] if has_all else []))


def comp_label(c) -> str:
    if c is None:
        return "all"
    kind, v = c
    return f"{v * 100:g}%" if kind == "var" else str(v)


def pca_fit(X: np.ndarray, standardize: str):
    """PCA model fitted on X (rows = maps): (mu, sd, mean, components, cumvar).

    Optional z-scoring per voxel first (training statistics only), then
    centering and SVD. Components with a singular value of ~0 are dropped,
    so at most n_maps - 1 components remain.
    """
    if standardize == "none":
        mu, sd = 0.0, 1.0
    else:
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
    Xs = (np.asarray(X, dtype=np.float64) - mu) / sd
    m = Xs.mean(axis=0)
    Xc = Xs - m
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # rank tolerance as in numpy.linalg.matrix_rank
    tol = S.max() * max(Xc.shape) * np.finfo(np.float32).eps if S.size else np.inf
    keep = S > tol
    var = S[keep] ** 2
    cumvar = np.cumsum(var) / var.sum() if var.size else np.zeros(0)
    return mu, sd, m, Vt[keep], cumvar


def pca_project(model, X: np.ndarray) -> np.ndarray:
    mu, sd, m, Vt, _ = model
    return ((X - mu) / sd - m) @ Vt.T


def pca_fold_job(roi, test_group, data, pair_list, params, cands, scope,
                 tie_break, seed, standardize="none", fit_on="analysis"):
    """One ROI x one held-out block: PCA on training blocks only, number of
    components chosen in an inner LOGO, refit, test on the held-out block.

    fit_on='analysis': PCA fitted on the training maps of ALL labels of the
    analysis (more maps, more stable components; labels are never used).
    fit_on='pair': PCA fitted on the training maps of each A-vs-B set.

    Returns (fold_rows, curve_rows, error_message_or_None).
    """
    warnings.filterwarnings("ignore")
    try:
        n_sets, n_c, n_p = len(pair_list), len(cands), len(params)
        inner = np.full((n_sets, n_c, n_p), np.nan)
        blocks = np.unique(np.concatenate([g for _, g in data.values()]))
        inner_blocks = [int(h) for h in blocks if h != test_group]

        proj, kmax, cumv = {}, {}, {}
        if fit_on == "analysis":
            for key, excl in [("outer", [test_group])] + \
                             [(h, [test_group, h]) for h in inner_blocks]:
                fit_X = np.vstack([X[~np.isin(g, excl)] for X, g in data.values()])
                mdl = pca_fit(fit_X, standardize)
                proj[key] = {lab: pca_project(mdl, X) for lab, (X, _) in data.items()}
                kmax[key] = mdl[3].shape[0]
                cumv[key] = mdl[4]

        def project_pair(key, a, b, X, tr):
            """Component scores of the pair's maps for one split, k max and
            cumulative explained variance of that PCA fit."""
            if fit_on == "analysis":
                return np.vstack([proj[key][a], proj[key][b]]), kmax[key], cumv[key]
            mdl = pca_fit(X[tr], standardize)
            return pca_project(mdl, X), mdl[3].shape[0], mdl[4]

        def n_comp(c, km, cv):
            if c is None:
                return km
            kind, v = c
            if kind == "n":
                return min(v, km)
            return int(min(np.searchsorted(cv, v - 1e-12) + 1, km))  # fewest comps with >= v

        # ---- 1. inner LOGO over the training blocks ------------------------
        splits = [None] * n_sets
        for k, (a, b) in enumerate(pair_list):
            (XA, gA), (XB, gB) = data[a], data[b]
            X = np.vstack([XA, XB])
            y = np.r_[np.zeros(len(gA), int), np.ones(len(gB), int)]
            g = np.r_[gA, gB]
            if not (g == test_group).any():
                continue
            splits[k] = (a, b, X, y, g)
            acc = np.zeros((n_c, n_p))
            n_in = 0
            for h in inner_blocks:
                tr = (g != test_group) & (g != h)
                te = g == h
                if not te.any() or len(np.unique(y[tr])) < 2:
                    continue
                Z, km, cv = project_pair(h, a, b, X, tr)
                for i, c in enumerate(cands):
                    kk = n_comp(c, km, cv)
                    for j, (C, loss) in enumerate(params):
                        clf = make_svc(C, loss, seed).fit(Z[tr][:, :kk], y[tr])
                        acc[i, j] += np.mean(clf.predict(Z[te][:, :kk]) == y[te])
                n_in += 1
            if n_in:
                inner[k] = acc / n_in

        # ---- 2. select the number of components ----------------------------
        best_over_params = np.nanmax(inner, axis=2)
        if scope == "pooled":
            i_pooled = pick_index(np.nanmean(best_over_params, axis=0), tie_break)
            i_sel = [i_pooled] * n_sets
        else:
            i_sel = [pick_index(best_over_params[k], tie_break)
                     if splits[k] is not None else None for k in range(n_sets)]

        # ---- 3. refit on all training blocks, test on the held-out block ---
        rows, curve = [], []
        for k in range(n_sets):
            if splits[k] is None:
                continue
            a, b, X, y, g = splits[k]
            tr, te = g != test_group, g == test_group
            Z, km, cv = project_pair("outer", a, b, X, tr)
            i = i_sel[k]
            kk = n_comp(cands[i], km, cv)
            j = pick_index(inner[k, i], "smallest")
            C, loss = params[j]
            clf = make_svc(C, loss, seed).fit(Z[tr][:, :kk], y[tr])
            rows.append({
                "pair": k + 1, "A_label": a, "B_label": b, "ROI": roi,
                "fold": int(test_group),
                "voxels": int(kk),                      # = number of components used
                "n_components": comp_label(cands[i]),
                "components_max": int(km), "roi_voxels": int(X.shape[1]),
                "variance_explained": round(float(cv[kk - 1]), 4) if kk else np.nan,
                "C": C, "loss": loss,
                "inner_accuracy": float(inner[k, i, j]),
                "test_accuracy": float(np.mean(clf.predict(Z[te][:, :kk]) == y[te])),
            })
            for i2, c2 in enumerate(cands):
                curve.append({
                    "pair": k + 1, "ROI": roi, "fold": int(test_group),
                    "voxels": comp_label(c2),
                    "inner_accuracy": float(best_over_params[k, i2]),
                })
        return rows, curve, None
    except Exception as e:
        return [], [], f"{roi} fold {test_group}: {e}"


def load_flat_masks(args, roi_voxels, ref_path, flat_dir, roi_file=None,
                    min_voxels=0) -> dict:
    """{roi: flat voxel indices on the map grid} for one mask per ROI.

    flat_dir given: masks in <flat_dir>/sub-XX/ (e.g. roi_anat, roi_responsive);
    a mask on the map grid is used as it is, otherwise it is resampled in
    memory. flat_dir None: <roi-dir>/sub-XX/<ROI>/ (PCA fallback). ROIs with
    fewer than min_voxels voxels are skipped.
    """
    ref = nib.load(str(ref_path))
    print("Loading whole-ROI masks ...")
    roi_cols = {}
    for roi, voxel_list in roi_voxels.items():
        rs = None
        if flat_dir:
            # flat folder (roi_anat): a mask already on the map grid is used
            # as it is; otherwise the unresampled mask is resampled here
            for cand in whole_roi_candidates(flat_dir, args.subject, roi,
                                             roi_file):
                img = nib.load(str(cand))
                if _on_grid(img, ref):
                    rs, p, note = img, cand, "whole ROI, already on the map grid"
                    break
                if not cand.name.endswith("_resampled.nii.gz"):
                    # resampled in memory only: nothing is written into this
                    # folder (a cache would overwrite <ROI>_resampled.nii.gz)
                    rs = resample_img(img, target_affine=ref.affine,
                                      target_shape=ref.shape[:3],
                                      interpolation="nearest")
                    p, note = cand, "whole ROI, resampled to the map grid in memory"
                    break
                print(f"  [info] {roi}: {cand.name} is on another grid; trying the next mask")
            if rs is None:
                print(f"  [warn] {roi}: no usable whole-ROI mask in "
                      f"{Path(flat_dir) / f'sub-{args.subject}'}")
                continue
        else:
            p, note = whole_roi_path(args.roi_dir, args.subject, roi,
                                     roi_file, voxel_list)
            if p is None:
                print(f"  [warn] {roi}: {note}")
                continue
            rs = nib.load(str(ensure_resampled_roi(p, ref_path)))
        if not _on_grid(rs, ref):
            raise RuntimeError(f"mask for {roi} is not on the grid "
                               f"of {ref_path.name}; rerun this job.")
        idx = np.flatnonzero(np.asanyarray(rs.dataobj).reshape(-1) > 0)
        print(f"  {roi:>13}: {idx.size} voxels from {p.name} ({note})")
        roi_cols[roi] = idx
        if idx.size < min_voxels:
            print(f"  [skip] {roi}: {idx.size} voxels < --min-voxels {min_voxels}")
            del roi_cols[roi]
    return roi_cols


def run_pca(args, kind, pair_list, roi_voxels, ref_path, out_dir, today):
    """Whole ROI, PCA fitted on training blocks, number of components nested."""
    cands = parse_components(args.n_components)
    if args.fixed_c is None:
        params = [(C, loss) for C in C_GRID for loss in LOSS_GRID]
    else:
        params = [(args.fixed_c, "squared_hinge")]

    ref = nib.load(str(ref_path))
    roi_cols = load_flat_masks(args, roi_voxels, ref_path, args.pca_roi_dir,
                               args.pca_roi_file)
    if not roi_cols:
        raise FileNotFoundError("No ROI masks found. Check --roi-dir / --pca-roi-file.")

    all_idx = np.unique(np.concatenate(list(roi_cols.values())))
    labels = list(dict.fromkeys(lab for pair in pair_list for lab in pair))
    print(f"Loading maps for {len(labels)} labels ({all_idx.size} voxels in the ROI union) ...")
    data_all = load_patterns(args.tmap_dir, args.subject, args.phase, kind,
                             labels, all_idx, ref, args.trial_subdir)
    groups = np.unique(np.concatenate([g for _, g in data_all.values()]))
    print(f"  {sum(len(g) for _, g in data_all.values())} maps, "
          f"{len(groups)} blocks (outer folds).")

    jobs = []
    for roi, idx in roi_cols.items():
        cols = np.searchsorted(all_idx, idx)
        data_roi = {lab: (X[:, cols], g) for lab, (X, g) in data_all.items()}
        for test_group in groups:
            jobs.append((roi, int(test_group), data_roi))

    print(f"\nRunning {len(jobs)} ROI x fold jobs on {args.n_jobs} workers ...")
    print(f"  PCA fitted on: {args.pca_fit} (training blocks only); components "
          f"{[comp_label(c) for c in cands]}; scope={args.selection_scope}, "
          f"tie-break={args.tie_break}, C={'tuned' if args.fixed_c is None else args.fixed_c}, "
          f"standardize={args.standardize}")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(pca_fold_job)(roi, tg, d, pair_list, params, cands,
                              args.selection_scope, args.tie_break, args.seed,
                              args.standardize, args.pca_fit)
        for roi, tg, d in jobs
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

    use_var = any(c is not None and c[0] == "var" for c in cands)
    tag = f"{'pcavar' if use_var else 'pca'}-{args.pca_fit}-{args.selection_scope}"
    if args.fixed_c is not None:
        tag += f"_C{args.fixed_c:g}"
    if args.standardize != "none":
        tag += "_zscore"
    return write_nested_outputs(args, folds, curve, tag, out_dir, today,
                                unit="components")


# ============================================================================
# ALLCOUNTS mode: one accuracy for EVERY candidate voxel count (no selection)
# ============================================================================
def allcounts_fold_job(roi, test_group, data, pos_by_n, pair_list, params, seed,
                       standardize="none"):
    """One ROI x one held-out block: an SVM per voxel count (top-N masks), each
    tested on the held-out block. C/loss tuned per count in an inner LOGO over
    the training blocks only if more than one setting is given."""
    warnings.filterwarnings("ignore")
    try:
        rows = []
        for k, (a, b) in enumerate(pair_list):
            (XA, gA), (XB, gB) = data[a], data[b]
            X = np.vstack([XA, XB])
            y = np.r_[np.zeros(len(gA), int), np.ones(len(gB), int)]
            g = np.r_[gA, gB]
            te = g == test_group
            if not te.any():
                continue
            tr = ~te
            inner_sets = []
            if len(params) > 1:
                inner_sets = [scale_split(X[tr][itr], X[tr][ite], standardize)
                              + (y[tr][itr], y[tr][ite])
                              for itr, ite in LeaveOneGroupOut().split(X[tr], y[tr], g[tr])]
            Atr, Ate = scale_split(X[tr], X[te], standardize)
            for n in sorted(pos_by_n):
                cols = pos_by_n[n]
                j, inner = 0, np.nan
                if inner_sets:
                    acc = np.zeros(len(params))
                    for Itr, Ite, ya, yb in inner_sets:
                        for jj, (C, loss) in enumerate(params):
                            clf = make_svc(C, loss, seed).fit(Itr[:, cols], ya)
                            acc[jj] += np.mean(clf.predict(Ite[:, cols]) == yb)
                    acc /= len(inner_sets)
                    j = pick_index(acc, "smallest")
                    inner = float(acc[j])
                C, loss = params[j]
                clf = make_svc(C, loss, seed).fit(Atr[:, cols], y[tr])
                rows.append({
                    "pair": k + 1, "A_label": a, "B_label": b, "ROI": roi,
                    "fold": int(test_group), "voxels": int(n),
                    "voxels_actual": int(len(cols)), "C": C, "loss": loss,
                    "inner_accuracy": inner,
                    "test_accuracy": float(np.mean(clf.predict(Ate[:, cols]) == y[te])),
                })
        return rows, [], None
    except Exception as e:
        return [], [], f"{roi} fold {test_group}: {e}"


def run_allcounts(args, kind, pair_list, roi_voxels, ref_path, out_dir, today):
    """Accuracy for every candidate voxel count separately (specification curve
    over voxel counts), with the nested engine (refit LinearSVC, C fixed or
    tuned per count). Output in the format of the fixed mode."""
    if args.voxel_counts:
        roi_voxels = {r: sorted(set(args.voxel_counts)) for r in roi_voxels}
    params = ([(args.fixed_c, "squared_hinge")] if args.fixed_c is not None
              else [(C, loss) for C in C_GRID for loss in LOSS_GRID])
    print("Loading ROI masks ...")
    roi_idx = load_roi_indices(args.roi_dir, args.subject, roi_voxels, ref_path)
    if not roi_idx:
        raise FileNotFoundError("No ROI masks found. Check --roi-dir.")
    all_idx = np.unique(np.concatenate([v for d in roi_idx.values() for v in d.values()]))
    labels = list(dict.fromkeys(lab for pair in pair_list for lab in pair))
    print(f"Loading maps for {len(labels)} labels ({all_idx.size} voxels in the ROI union) ...")
    data_all = load_patterns(args.tmap_dir, args.subject, args.phase, kind, labels,
                             all_idx, nib.load(str(ref_path)), args.trial_subdir)
    groups = np.unique(np.concatenate([g for _, g in data_all.values()]))
    jobs = []
    for roi, idx in roi_idx.items():
        roi_union = np.unique(np.concatenate(list(idx.values())))
        cols = np.searchsorted(all_idx, roi_union)
        pos_by_n = {n: np.searchsorted(roi_union, v) for n, v in idx.items()}
        data_roi = {lab: (X[:, cols], g) for lab, (X, g) in data_all.items()}
        for test_group in groups:
            jobs.append((roi, int(test_group), data_roi, pos_by_n))
    print(f"\nRunning {len(jobs)} ROI x fold jobs on {args.n_jobs} workers "
          f"(every voxel count; C={'tuned per count' if args.fixed_c is None else args.fixed_c}) ...")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(allcounts_fold_job)(roi, tg, d, pos, pair_list, params, args.seed,
                                    args.standardize)
        for roi, tg, d, pos in jobs)
    errors = [e for *_, e in results if e is not None]
    for e in errors:
        print(f"  [error] {e}")
    folds = pd.DataFrame([r for rows, _, _ in results for r in rows])
    print(f"\nFinished {len(jobs) - len(errors)}/{len(jobs)} jobs in {time.time() - t0:.1f}s.")
    if folds.empty:
        raise RuntimeError("No results. See errors above.")

    tag = "allcounts" + (f"_C{args.fixed_c:g}" if args.fixed_c is not None else "") \
        + ("_zscore" if args.standardize != "none" else "")
    out_xlsx = out_dir / f"decoding_{args.analysis}_{args.phase_tag}_{tag}_{today}.xlsx"
    folds.to_excel(out_xlsx, sheet_name="folds", index=False)          # safety net
    per_pair = (folds.groupby(["pair", "A_label", "B_label", "ROI", "voxels"], sort=False)
                     .agg(accuracy=("test_accuracy", "mean"), n_folds=("fold", "size"))
                     .reset_index())
    for i, (k, v) in enumerate({"subject": args.subject, "phase": args.phase,
                                "analysis": args.analysis, "selection": tag}.items()):
        per_pair.insert(i, k, v)
    summary = (per_pair.groupby(["ROI", "voxels"], sort=False)
                       .agg(mean_accuracy=("accuracy", "mean"),
                            std_accuracy=("accuracy", "std"), n_pairs=("accuracy", "size"))
                       .reset_index().round(4))
    print("\n" + "-" * 64 + "\nMean accuracy across pairs (per ROI x voxel count):\n" + "-" * 64)
    print(summary.to_string(index=False))
    with pd.ExcelWriter(out_xlsx) as xl:
        per_pair.to_excel(xl, sheet_name="per_pair", index=False)
        summary.to_excel(xl, sheet_name="summary", index=False)
        folds.to_excel(xl, sheet_name="folds", index=False)
    print(f"Saved {out_xlsx}")
    return f"decoding_script_{args.analysis}_{args.phase_tag}_{tag}_{today}.py"


# ============================================================================
# MASK mode: all voxels of a given mask per ROI (e.g. localizer-responsive
# voxels after Rademaker et al., 2019) -- no selection, no PCA
# ============================================================================
def mask_fold_job(roi, test_group, data, pair_list, params, seed, standardize="none"):
    """One ROI x one held-out block: SVM on all mask voxels. C/loss are tuned
    in an inner LOGO over the training blocks only if more than one setting
    is given (--fixed-c gives one)."""
    warnings.filterwarnings("ignore")
    try:
        rows, curve = [], []
        for k, (a, b) in enumerate(pair_list):
            (XA, gA), (XB, gB) = data[a], data[b]
            X = np.vstack([XA, XB])
            y = np.r_[np.zeros(len(gA), int), np.ones(len(gB), int)]
            g = np.r_[gA, gB]
            te = g == test_group
            if not te.any():
                continue
            tr = ~te
            inner_acc = np.full(len(params), np.nan)
            if len(params) > 1:
                acc = np.zeros(len(params))
                n_in = 0
                for itr, ite in LeaveOneGroupOut().split(X[tr], y[tr], g[tr]):
                    Atr, Ate = scale_split(X[tr][itr], X[tr][ite], standardize)
                    for j, (C, loss) in enumerate(params):
                        clf = make_svc(C, loss, seed).fit(Atr, y[tr][itr])
                        acc[j] += np.mean(clf.predict(Ate) == y[tr][ite])
                    n_in += 1
                inner_acc = acc / n_in
                j = pick_index(inner_acc, "smallest")
            else:
                j = 0
            C, loss = params[j]
            Atr, Ate = scale_split(X[tr], X[te], standardize)
            clf = make_svc(C, loss, seed).fit(Atr, y[tr])
            rows.append({
                "pair": k + 1, "A_label": a, "B_label": b, "ROI": roi,
                "fold": int(test_group), "voxels": int(X.shape[1]),
                "C": C, "loss": loss, "inner_accuracy": float(inner_acc[j]),
                "test_accuracy": float(np.mean(clf.predict(Ate) == y[te])),
            })
            curve.append({"pair": k + 1, "ROI": roi, "fold": int(test_group),
                          "voxels": int(X.shape[1]),
                          "inner_accuracy": float(inner_acc[j])})
        return rows, curve, None
    except Exception as e:
        return [], [], f"{roi} fold {test_group}: {e}"


def run_mask(args, kind, pair_list, roi_voxels, ref_path, out_dir, today):
    """All voxels of one mask per ROI (--mask-dir), no voxel selection."""
    if args.mask_dir is None:
        raise SystemExit("--voxel-selection mask needs --mask-dir")
    params = ([(args.fixed_c, "squared_hinge")] if args.fixed_c is not None
              else [(C, loss) for C in C_GRID for loss in LOSS_GRID])
    roi_cols = load_flat_masks(args, roi_voxels, ref_path, args.mask_dir,
                               args.pca_roi_file, min_voxels=args.min_voxels)
    if not roi_cols:
        raise FileNotFoundError(f"No usable ROI masks in {args.mask_dir}.")

    all_idx = np.unique(np.concatenate(list(roi_cols.values())))
    labels = list(dict.fromkeys(lab for pair in pair_list for lab in pair))
    print(f"Loading maps for {len(labels)} labels ({all_idx.size} voxels in the ROI union) ...")
    data_all = load_patterns(args.tmap_dir, args.subject, args.phase, kind,
                             labels, all_idx, nib.load(str(ref_path)), args.trial_subdir)
    groups = np.unique(np.concatenate([g for _, g in data_all.values()]))

    jobs = []
    for roi, idx in roi_cols.items():
        cols = np.searchsorted(all_idx, idx)
        data_roi = {lab: (X[:, cols], g) for lab, (X, g) in data_all.items()}
        for test_group in groups:
            jobs.append((roi, int(test_group), data_roi))
    print(f"\nRunning {len(jobs)} ROI x fold jobs on {args.n_jobs} workers "
          f"(masks from {args.mask_dir}; C={'tuned' if args.fixed_c is None else args.fixed_c}, "
          f"standardize={args.standardize}) ...")
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(mask_fold_job)(roi, tg, d, pair_list, params, args.seed, args.standardize)
        for roi, tg, d in jobs)
    errors = [e for *_, e in results if e is not None]
    for e in errors:
        print(f"  [error] {e}")
    folds = pd.DataFrame([r for rows, _, _ in results for r in rows])
    curve = pd.DataFrame([c for _, cs, _ in results for c in cs])
    print(f"\nFinished {len(jobs) - len(errors)}/{len(jobs)} jobs in {time.time() - t0:.1f}s.")
    if folds.empty:
        raise RuntimeError("No results. See errors above.")
    tag = f"mask-{Path(args.mask_dir).name}"
    if args.fixed_c is not None:
        tag += f"_C{args.fixed_c:g}"
    if args.standardize != "none":
        tag += "_zscore"
    return write_nested_outputs(args, folds, curve, tag, out_dir, today, label="mask")


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
    elif args.voxel_selection == "allcounts":
        snapshot_name = run_allcounts(args, kind, pair_list, roi_voxels,
                                      ref_path, out_dir, today)
    elif args.voxel_selection == "mask":
        snapshot_name = run_mask(args, kind, pair_list, roi_voxels,
                                 ref_path, out_dir, today)
    elif args.voxel_selection == "pca":
        snapshot_name = run_pca(args, kind, pair_list, roi_voxels,
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