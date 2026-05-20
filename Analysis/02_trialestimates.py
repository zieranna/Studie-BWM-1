#!/usr/bin/env python3
"""
02_trialestimates.py
=================

Per-run FIR-stick GLM for the binding WM task.

For each run, the script:

    1. Loads the preprocessed BOLD, the 6 motion regressors from
       fMRIPrep's confounds, and the BIDS events.tsv emitted by
       ``03_create_events.py``.
    2. Filters the events to encoding-phase onsets only and sets their
       duration to 0 (impulse / "stick" regressors). The FIR basis
       (17 delays) then captures the BOLD shape across the entire
       encoding -> delay -> response trial.
    3. Fits a nilearn ``FirstLevelModel`` (FIR HRF, cosine drift,
       AR(1) noise).
    4. Computes phase-averaged t-maps per trial:
            enc  = mean(delays 3, 4, 5, 6)
            del  = mean(delays 9, 10, 11, 12)
            res  = mean(delays 14, 15, 16, 17)

Output layout
-------------
    <output-dir>/sub-XX/model/fir/binding/
        glm/glm_ses-YY_run-NN.pkl
        tmap/enc/sub-XX_ses-YY_run-NN_trial-PP.nii.gz
        tmap/del/sub-XX_ses-YY_run-NN_trial-PP.nii.gz
        tmap/res/sub-XX_ses-YY_run-NN_trial-PP.nii.gz
    (PP is the PAPER label from the events file, e.g. 4a, 13b.)

Installation
------------
    pip install nilearn nibabel numpy pandas tqdm
"""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs
from tqdm import tqdm


# ============================================================================
# Constants
# ============================================================================
TR = 1.2
FIR_DELAYS = list(range(1, 18))                  # 17 delays * 1.2 s = 20.4 s

FIR_PHASES = {
    "enc": [3, 4, 5, 6],
    "del": [9, 10, 11, 12],
    "res": [14, 15, 16, 17],
}

N_JOBS_FIT      = 25
N_JOBS_CONTRAST = 60


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit FIR-stick GLMs and compute phase-averaged t-maps per trial.",
    )
    p.add_argument("--subject", required=True, type=str,
                   help="Participant label without 'sub-' (e.g. '01').")
    p.add_argument("--events-dir", required=True, type=Path,
                   help="Root of BIDS events derivatives")
    p.add_argument("--fmriprep-dir", required=True, type=Path,
                   help="fMRIPrep derivatives directory (provides BOLD + confounds).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where GLMs and t-maps will be written.")
    p.add_argument("--sessions", nargs="+",
                   default=["01", "02", "03", "04"])
    p.add_argument("--runs", nargs="+",
                   default=["01", "02", "03", "04",
                            "05", "06", "07", "08"])
    p.add_argument("--force", action="store_true",
                   help="Refit GLMs and overwrite t-maps even if outputs exist.")
    return p.parse_args()


# ============================================================================
# Path helpers
# ============================================================================
def events_path(imaging_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (imaging_dir / f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-bwm_run-{run}_events.tsv")


def bold_path(fmriprep_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (fmriprep_dir / f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}"
            f"_space-T1w_desc-preproc_bold.nii.gz")


def confounds_path(fmriprep_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (fmriprep_dir / f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}"
            f"_desc-confounds_timeseries.tsv")


# ============================================================================
# Loaders
# ============================================================================
def load_events(path: Path, frame_times: np.ndarray) -> pd.DataFrame:
    """events.tsv -> encoding sticks (duration=0), trial_type = 'trial_<PAPER>'."""
    ev = pd.read_csv(path, sep="\t")
    ev = ev[ev["trial_type"] == "encoding"].copy()

    # Snap to nearest TR boundary minus 50 ms.
    ev["onset"] = [min(frame_times, key=lambda x: abs(x - o)) - 0.05
                   for o in ev["onset"]]
    ev["duration"] = 0
    ev["trial_type"] = [f"trial_{p}" for p in ev["trial"]]
    return ev[["onset", "duration", "trial_type"]].reset_index(drop=True)


# ============================================================================
# GLM
# ============================================================================
def fit_or_load_glm(fmri_img, events: pd.DataFrame,
                    glm_path: Path, force: bool) -> FirstLevelModel:
    if glm_path.exists() and not force:
        print(f"  [cached] {glm_path.name}")
        with open(glm_path, "rb") as f:
            return pickle.load(f)

    glm = FirstLevelModel(
        t_r=TR, slice_time_ref=0.0,
        hrf_model="fir", fir_delays=FIR_DELAYS,
        drift_model="cosine", high_pass=0.01,
        noise_model="ar1",
        standardize=False, signal_scaling=False,
        mask_img=False,
        smoothing_fwhm=None,
        n_jobs=N_JOBS_FIT,
    )
    glm.fit(fmri_img, events=events)

    glm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(glm_path, "wb") as f:
        pickle.dump(glm, f)
    return glm


def build_phase_contrasts(design_columns) -> dict:
    cols = list(design_columns)
    col_idx = {c: i for i, c in enumerate(cols)}
    trial_labels = sorted({c.split("_delay_")[0] for c in cols
                           if c.startswith("trial_") and "_delay_" in c})

    contrasts = {}
    n = len(cols)
    for trial in trial_labels:
        for phase, delays in FIR_PHASES.items():
            vec = np.zeros(n)
            present = 0
            w = 1.0 / len(delays)
            for d in delays:
                col = f"{trial}_delay_{d}"
                if col in col_idx:
                    vec[col_idx[col]] = w
                    present += 1
            if present == len(delays):
                contrasts[f"{trial}_{phase}"] = vec
    return contrasts


def compute_tmap(glm: FirstLevelModel, name: str, vec: np.ndarray,
                 sub: str, ses: str, run: str, tmap_dir: Path) -> None:
    trial_part, phase = name.rsplit("_", 1)              # 'trial_4a', 'enc'
    trial_label = trial_part.replace("trial_", "trial-") # 'trial-4a'
    out_path = (tmap_dir / phase /
                f"sub-{sub}_ses-{ses}_run-{run}_{trial_label}.nii.gz")
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_img = glm.compute_contrast(vec, output_type="stat")
    t_img.to_filename(str(out_path))


# ============================================================================
# Per-run driver
# ============================================================================
def process_run(sub: str, ses: str, run: str,
                events_dir: Path, fmriprep_dir: Path,
                output_dir: Path, force: bool) -> None:
    print(f"\n== sub-{sub} ses-{ses} run-{run} ==")

    glm_path = (output_dir / f"sub-{sub}" / "model" / "fir" / "binding" /
                "glm" / f"glm_ses-{ses}_run-{run}.pkl")
    tmap_dir = (output_dir / f"sub-{sub}" / "model" / "fir" / "binding" /
                "tmap")

    fmri_img = concat_imgs(str(bold_path(fmriprep_dir, sub, ses, run)),
                           auto_resample=True)
    frame_times = np.arange(fmri_img.shape[-1]) * TR

    events = load_events(events_path(events_dir, sub, ses, run), frame_times)

    glm = fit_or_load_glm(fmri_img, events, glm_path, force)
    contrasts = build_phase_contrasts(glm.design_matrices_[0].columns)
    print(f"   {len(contrasts) // len(FIR_PHASES)} trials x "
          f"{len(FIR_PHASES)} phases = {len(contrasts)} t-maps")

    with ThreadPoolExecutor(max_workers=N_JOBS_CONTRAST) as pool:
        futs = [pool.submit(compute_tmap, glm, name, vec,
                            sub, ses, run, tmap_dir)
                for name, vec in contrasts.items()]
        for f in tqdm(as_completed(futs), total=len(futs), desc="  t-maps"):
            f.result()


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()
    print("-" * 64)
    print(f"FIR-stick GLM for sub-{args.subject}")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>14}: {v}")
    print("-" * 64)

    for ses in args.sessions:
        for run in args.runs:
            ev_p = events_path(args.events_dir, args.subject, ses, run)
            if not ev_p.exists():
                print(f"[skip] no events file: {ev_p}")
                continue
            process_run(args.subject, ses, run,
                        args.events_dir, args.fmriprep_dir,
                        args.output_dir, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()