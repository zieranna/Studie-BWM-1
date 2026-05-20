#!/usr/bin/env python3
"""
04_fullmodel.py
===============

Whole-subject FIR GLM producing one phase t-map per subject (used
downstream for voxel selection in 06_ROI_selectvoxel.py).

Unlike 02_trialestimates.py, which fits one GLM per run and outputs
per-trial t-maps, this script pools all runs across all sessions into
a single subject-level model. All encoding-phase trials are collapsed
into a single "trial" regressor — the FIR basis then captures the
average BOLD shape across the full trial epoch.

The fitted GLM is cached to disk; on re-runs the contrast computation
step is skipped straight to if the .pkl exists.

Steps
-----
    1. For each session × run, load the preprocessed BOLD
       (space-T1w), the BIDS events.tsv, and the 6 fMRIPrep motion
       regressors. Runs with missing events files are skipped silently.
    2. Build a per-run FIR design matrix (25 delays × 1.2 s = 30 s)
       with cosine drift and motion regressors. Within each session,
       run-level nuisance columns (motion, drift, constant) are
       disambiguated by a _runN suffix before concatenation.
    3. Add session-level intercept regressors (one binary column per
       session) to account for session-to-session mean-signal shifts.
    4. Fit a nilearn FirstLevelModel (FIR HRF, AR(1) noise, no
       smoothing) across the concatenated session images.
    5. Compute phase-averaged contrasts over the FIR delay set:
            enc  = mean(delays 3–5)
            del  = mean(delays 9–12)
            res  = mean(delays 15–19)
       Both t-stat and z-score images are written for each phase.
    6. Additionally save per-delay t-maps (delay 1–25) for inspecting
       the full FIR timecourse.

Output layout
-------------
    <output-dir>/sub-XX/model/fir/binding/full/
        glm/
            full_fir_model.pkl
        contrast/
            sub-XX_enc_t.nii.gz     ← used by 06_ROI_selectvoxel.py
            sub-XX_enc.nii.gz       ← z-score
            sub-XX_del_t.nii.gz
            sub-XX_del.nii.gz
            sub-XX_res_t.nii.gz
            sub-XX_res.nii.gz
            sub-XX_delay1_t.nii.gz  ← per-delay timecourse check
            …
            sub-XX_delay25_t.nii.gz

Usage
-----
    python 04_fullmodel.py \\
        --subject 01 \\
        --events-dir  /path/to/behav_data \\
        --fmriprep-dir /path/to/preprocessed \\
        --output-dir  /path/to/firmodelcomplete
"""

import argparse
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import concat_imgs

TR = 1.2

FIR_DELAYS = list(range(1, 26))

MOTION_COLS = ["trans_x", "trans_y", "trans_z",
               "rot_x", "rot_y", "rot_z"]

N_JOBS = 25


# ---------------------------------------------------------------------
# ARGPARSE
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit a whole-subject FIR GLM and compute phase t-maps "
                    "for voxel selection.",
    )
    p.add_argument("--subject", required=True, type=str,
                   help="Participant label without 'sub-' (e.g. '01').")
    p.add_argument("--events-dir", required=True, type=Path,
                   help="Root of BIDS behav_data directory (contains sub-XX/timing/).")
    p.add_argument("--fmriprep-dir", required=True, type=Path,
                   help="fMRIPrep derivatives directory (provides BOLD + confounds).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where the GLM cache and contrast maps will be written.")
    p.add_argument("--sessions", nargs="+",
                   default=["01", "02", "03", "04"],
                   help="Session labels to include (default: 01–04).")
    p.add_argument("--runs", nargs="+",
                   default=["01", "02", "03", "04",
                            "05", "06", "07", "08"],
                   help="Run labels to include (default: 01–08).")
    p.add_argument("--force", action="store_true",
                   help="Refit the GLM and overwrite contrasts even if the "
                        ".pkl cache exists.")
    return p.parse_args()

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
def bold_path(fmriprep_dir, sub, ses, run):
    return (fmriprep_dir /
            f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}"
            f"_space-T1w_desc-preproc_bold.nii.gz")


def events_path(imaging_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (imaging_dir / f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-bwm_run-{run}_events.tsv")


def confounds_path(fmriprep_dir, sub, ses, run):
    return (fmriprep_dir /
            f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}"
            f"_desc-confounds_timeseries.tsv")


# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------
def load_events(path, frame_times):
    ev = pd.read_csv(path, sep="\t")
    ev = ev[ev["trial_type"] == "encoding"].copy()

    ev["onset"] = [
        min(frame_times, key=lambda x: abs(x - o)) - 0.05
        for o in ev["onset"]
    ]

    ev["duration"] = 0
    ev["trial_type"] = "trial"

    return ev[["onset", "duration", "trial_type"]].reset_index(drop=True)


def load_motion(path):
    conf = pd.read_csv(path, sep="\t")
    return conf[MOTION_COLS].fillna(0).reset_index(drop=True)


# ---------------------------------------------------------------------
# COLLECT RUNS
# ---------------------------------------------------------------------
def collect_runs(args):
    imgs, events, motions, ses_ids = [], [], [], []

    for ses in args.sessions:
        for run in args.runs:

            ev_file = events_path(args.events_dir, args.subject, ses, run)
            img_file = bold_path(args.fmriprep_dir, args.subject, ses, run)
            mot_file = confounds_path(args.fmriprep_dir, args.subject, ses, run)

            if not ev_file.exists():
                continue

            img = concat_imgs(str(img_file), auto_resample=True)
            ft = np.arange(img.shape[-1]) * TR

            imgs.append(img)
            events.append(load_events(ev_file, ft))
            motions.append(load_motion(mot_file))
            ses_ids.append(ses)

    return imgs, events, motions, ses_ids


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    # =================================================================
    # PATHS  (matched to 06_run_full_fir_glm.py)
    # =================================================================
    outdir = (args.output_dir / f"sub-{args.subject}" /
              "model" / "fir" / "binding" / "full")
    glm_file = outdir / "glm" / "full_fir_model.pkl"
    out_con  = outdir / "contrast"

    print("Loading runs...")
    imgs, events, motions, ses_ids = collect_runs(args)
    print(f"Total runs: {len(imgs)}")

    # =================================================================
    # CASE 1: LOAD EXISTING MODEL
    # =================================================================
    if glm_file.exists():
        print(f"[CACHE] Loading GLM from {glm_file}")

        with open(glm_file, "rb") as f:
            glm = pickle.load(f)

    # =================================================================
    # STEP 1: BUILD RUN-LEVEL DESIGN MATRICES
    # =================================================================
    else:
        print("Building and fitting GLM...")

        # -------------------------------
        # build run-level design matrices
        # -------------------------------
        run_dms = []
        run_scans = []

        for img, ev, mot in zip(imgs, events, motions):
            n_scans = img.shape[-1]
            frame_times = np.arange(n_scans) * TR

            dm = make_first_level_design_matrix(
                frame_times=frame_times,
                events=ev,
                hrf_model="fir",
                fir_delays=FIR_DELAYS,
                drift_model="cosine",
                high_pass=0.01,
                add_regs=mot,
            )

            run_dms.append(dm)
            run_scans.append(n_scans)

        # -------------------------------
        # session grouping
        # -------------------------------
        unique_sessions = sorted(set(ses_ids))

        session_imgs = []
        session_dms = []

        for ses in unique_sessions:

            idx = [i for i, s in enumerate(ses_ids) if s == ses]

            ses_imgs = concat_imgs([imgs[i] for i in idx], auto_resample=True)

            ses_dms_raw = [run_dms[i] for i in idx]

            for j, dm in enumerate(ses_dms_raw):
                dm.columns = [
                    f"{c}_run{j + 1}"
                    if c.startswith(("trans", "rot", "drift", "constant"))
                    else c
                    for c in dm.columns
                ]

            ses_dm = pd.concat(
                ses_dms_raw,
                axis=0,
                join="outer",
                ignore_index=True,
            ).fillna(0)

            session_imgs.append(ses_imgs)
            session_dms.append(ses_dm)

        # -------------------------------
        # session regressors
        # -------------------------------
        n_sessions = len(session_imgs)

        session_dms_final = []
        for i, dm in enumerate(session_dms):
            sess_col = np.zeros((dm.shape[0], n_sessions))
            sess_col[:, i] = 1

            dm_final = pd.concat(
                [dm,
                 pd.DataFrame(sess_col,
                              columns=[f"Session_{j + 1}" for j in range(n_sessions)])],
                axis=1
            )

            session_dms_final.append(dm_final)

        # -------------------------------
        # FIT MODEL
        # -------------------------------
        glm = FirstLevelModel(
            t_r=TR,
            hrf_model="fir",
            fir_delays=FIR_DELAYS,
            drift_model="cosine",
            high_pass=0.01,
            noise_model="ar1",
            standardize=False,
            signal_scaling=False,
            mask_img=False,
            smoothing_fwhm=None,
            n_jobs=N_JOBS,
        )

        glm.fit(session_imgs, design_matrices=session_dms_final)

        # SAVE  (mkdir parent first so /glm/ exists)
        glm_file.parent.mkdir(parents=True, exist_ok=True)
        with open(glm_file, "wb") as f:
            pickle.dump(glm, f)

        print(f"[SAVE] GLM written to {glm_file}")

    # =================================================================
    # CONTRASTS
    # =================================================================
    FIR_PHASES = {
        "enc": [3, 4, 5],
        "del": [9, 10, 11, 12],
        "res": [15, 16, 17, 18, 19],
    }

    def build_contrasts(design_matrix):
        contrasts = {}
        n_cols = len(design_matrix.columns)
        for name, delays in FIR_PHASES.items():
            vec = np.zeros(n_cols)
            for d in delays:
                col = f"trial_delay_{d}"
                if col in design_matrix.columns:
                    vec[design_matrix.columns.get_loc(col)] = 1.0
            contrasts[name] = vec
        return contrasts

    print("Computing contrasts...")
    design_matrix = glm.design_matrices_[0]
    contrasts = build_contrasts(design_matrix)

    out_con.mkdir(parents=True, exist_ok=True)

    for name, vec in contrasts.items():
        # t-stat as the primary file (no suffix) -- this is what
        # 07_create_top_voxel_rois.py / the runner read for voxel selection.
        tmap = glm.compute_contrast(vec, output_type="stat")
        tmap.to_filename(out_con / f"sub-{args.subject}_{name}_t.nii.gz")

        # z-score saved alongside with explicit suffix.
        zmap = glm.compute_contrast(vec, output_type="z_score")
        zmap.to_filename(out_con / f"sub-{args.subject}_{name}.nii.gz")

        print(f"saved {name}")

    # =================================================================
    # t-stats for all FIR delays (to check timeline)
    for d in FIR_DELAYS:
        vec = np.zeros(len(design_matrix.columns))
        col = f"trial_delay_{d}"
        if col in design_matrix.columns:
            vec[design_matrix.columns.get_loc(col)] = 1.0
            tmap = glm.compute_contrast(vec, output_type="stat")
            tmap.to_filename(out_con / f"sub-{args.subject}_delay{d}_t.nii.gz")
            print(f"saved delay {d} t-map")


if __name__ == "__main__":
    main()