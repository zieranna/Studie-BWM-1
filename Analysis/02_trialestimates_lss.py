#!/usr/bin/env python3
"""
02b_trialestimates_lss.py
=========================

Least-squares-separate (LSS) single-trial estimates for the binding WM task
(Reviewer 2, comment 2b; Mumford et al., 2012, NeuroImage).

One GLM per trial. The target trial gets its own regressors; all other trials
of the run share one regressor set (label-blind, so no class information
enters the nuisance model). Two bases are available:

--basis hrf  (default; the conventional LSS the reviewer asks for)
    Canonical HRF convolved with four events per trial: encoding (2 TRs =
    2.4 s from encoding onset), early and late delay (the two halves of the
    interval from 2.4 s after encoding onset to the response onset, ~4.8 s
    each) and response (4.8 s), for the target trial (target_*) and for all
    other trials (other_*). Onsets come from events.tsv.
    Saved: enc, whole delay (average of early and late), res in hrf_<stat>/,
    and early and late delay in hrf_<stat>_early/ and hrf_<stat>_late/.

    Runs with corrupted timing (negative onsets, encoding events far shorter
    than logged, onsets after the last volume) are refused with an error.

Assignment check (both bases): per run, the mean ROI signal is fitted with
the encoding onsets of every events file of the session. The run's own
events must fit best (largest t, by ASSIGN_MARGIN); otherwise the run is
flagged in the log and the QC table (assign_best_run, assign_margin).

--basis fir  (the paper's time windows, but ~320 instead of ~46 residual df)
    Encoding-onset sticks with the 17 FIR delays, once for the target trial
    and once for all other trials (34 columns instead of 306). Phase
    estimates average the target's FIR delays exactly as in
    02_trialestimates.py (enc 3-6, del 9-12, res 14-17). Because the other
    trials now share columns, their volumes inform the baseline, drift and
    noise estimates, which answers the "implicit baseline" concern while
    keeping the original windows.

For every trial and phase, the beta (effect size) and the t map are saved.
A QC table per run lists residual df and the VIFs of the target regressors.

Specification curve (Reviewer 2, comment 1)
-------------------------------------------
With --basis fir, every window in FIR_WINDOWS is computed from the same fit
(one extra contrast each, so almost no extra time); each window is its own
estimate folder (e.g. fir/beta/del_w10-13). Fix the set of windows before
looking at any results. --windows main computes only enc, del and res.

FIR delay numbering: code delay d (nilearn fir_delays, 1-based) is the volume
acquired d * TR s after encoding onset. The paper counted from 0, so code
9-12 = paper 8-11 = volumes at 10.8, 12.0, 13.2 and 14.4 s. Report windows in
seconds to avoid the ambiguity.

Fitting
-------
Each trial GLM is a nilearn FirstLevelModel fitted with its LSS design
matrix (design_matrices=...), signal_scaling=0 by default (betas in percent
signal change). The run is loaded into memory once; the 18 trial GLMs of a
run are fitted in parallel threads.

Output layout
-------------
    <output-dir>/sub-XX/<model>/<stat>/<window>/trials/
        sub-XX_ses-YY_block-BB_run-NN_trial-PP.nii.gz
    <output-dir>/sub-XX/<model>/<stat>/<window>/pairs/      (03b_average_runs.py)
        sub-XX_ses-YY_block-BB_pair-K.nii.gz
    <output-dir>/qc/sub-XX/sub-XX_ses-YY_run-NN_<model>_design.csv

    model   hrf | fir
    stat    beta | tmap
    window  hrf: enc, del (whole delay), del_early, del_late, res
            fir: enc, del (paper window), res, del_w08-11, ..., pre_01-02

e.g. lss/sub-01/hrf/beta/del_early/trials/. Block = pair of runs (block-01 =
run-01 + run-02, ...). 07_decoding.py reads this layout with
--layout lss --phase <model>/<stat>/<window>, e.g. --phase hrf/beta/del.
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
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import (FirstLevelModel,
                                     make_first_level_design_matrix, run_glm)
import inspect

from nilearn.image import new_img_like, resample_to_img
try:                                    # nilearn >= 0.9
    from nilearn.maskers import NiftiMasker
except ImportError:                     # nilearn 0.8
    from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask

# ============================================================================
# Constants
# ============================================================================
TR = 1.2
FIR_DELAYS = list(range(1, 18))                  # 17 delays * 1.2 s = 20.4 s
# Time windows for --basis fir: {window: code delays}. Every window becomes an
# window folder fir/<stat>/<window>. Seconds after encoding onset = delay * 1.2.
# Windows without "_" (enc, del, res) are the main ones (--windows main).
FIR_WINDOWS = {
    "enc":        [3, 4, 5, 6],                  #  3.6-7.2 s
    "del":        [9, 10, 11, 12],               # 10.8-14.4 s (paper "8-11")
    "res":        [14, 15, 16, 17],              # 16.8-20.4 s
    # --- specification curve: shift, narrow, widen --------------------------
    "del_w08-11": [8, 9, 10, 11],                #  9.6-13.2 s
    "del_w10-13": [10, 11, 12, 13],              # 12.0-15.6 s
    "del_w09-11": [9, 10, 11],                   # 10.8-13.2 s
    "del_w10-12": [10, 11, 12],                  # 12.0-14.4 s
    "del_w08-13": [8, 9, 10, 11, 12, 13],        #  9.6-15.6 s
    # --- negative control: before the hemodynamic response rises -----------
    "pre_01-02":  [1, 2],                        #  1.2-2.4 s
}
RUNS_PER_BLOCK = 2      # block-01 = run-01 + run-02, ...

# --basis hrf: four regressors per trial, built from the logged onsets:
#   enc        encoding onset, lasting ENC_DUR = 2 TRs (2.4 s)
#   del_early  encoding onset + ENC_DUR -> midpoint of the delay
#   del_late   midpoint -> response onset
#   res        response onset, lasting RES_DUR
# With the logged timing (response ~11.98 s after encoding onset) each delay
# half lasts ~4.8 s. The 'delay' events in events.tsv are not used for the
# model; the delay starts where the encoding regressor ends.
ENC_DUR = 2 * TR
RES_DUR = 4.8
RES_AFTER_ENC = 11.98   # s; only for trials without a response event
MIN_DELAY = 6.0         # s; a shorter modeled delay means a timing problem
HRF_REGRESSORS = ["enc", "del_early", "del_late", "res"]

# Assignment check: all runs share the same structure (a trial every
# 20-25 s), so events of a wrong run still fit a BOLD run reasonably well and
# no fixed threshold separates right from wrong. Instead, the mean signal of
# the ROI voxels (--extra-mask-dir; else the brain) is fitted with the
# events of EVERY run of the session (all-trials encoding regressor). The
# run's own events must fit best, by at least ASSIGN_MARGIN in t; otherwise
# the run is flagged in the log and the QC table.
ASSIGN_MARGIN = 2.0

# --basis hrf outputs: {window: target regressors averaged}. Every window
# becomes a folder hrf/<stat>/<window>; the whole delay is the average of
# the early and late betas.
HRF_OUTPUTS = {
    "enc":       ["enc"],
    "del":       ["del_early", "del_late"],      # whole delay
    "del_early": ["del_early"],
    "del_late":  ["del_late"],
    "res":       ["res"],
}

# trial_type names of the three phases in events.tsv (03_create_events.py).
# CHECK these against your events files; the script lists the available
# names if one is missing. Only "encoding" is needed for --basis fir.
PHASE_EVENTS = {"enc": "encoding", "del": "delay", "res": "response"}

HRF_MODEL = "spm"
HIGH_PASS = 0.01                                 # Hz, as in 02_trialestimates.py
# nilearn warns that FIR sticks have duration 0; that is intended.
warnings.filterwarnings("ignore", message=".*null duration.*")
warnings.filterwarnings("ignore", message=".*boolean values for 'standardize'.*")

MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSS single-trial estimates.")
    p.add_argument("--subject", required=True, type=str,
                   help="Participant label without 'sub-' (e.g. '01').")
    p.add_argument("--events-dir", required=True, type=Path,
                   help="Root of BIDS events derivatives.")
    p.add_argument("--fmriprep-dir", required=True, type=Path,
                   help="fMRIPrep derivatives (BOLD, confounds, brain mask).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Root for <basis>_beta/, <basis>_t/ and qc/.")
    p.add_argument("--basis", choices=["hrf", "fir"], default="hrf",
                   help="'hrf' (default): canonical HRF, phase events. "
                        "'fir': the paper's FIR windows.")
    p.add_argument("--no-motion", action="store_true",
                   help="Leave out the 6 motion regressors (the original "
                        "FIR model had none).")
    p.add_argument("--noise-model", choices=["ar1", "ols"], default="ar1")
    p.add_argument("--no-signal-scaling", action="store_true",
                   help="Fit the raw signal. Default: signal_scaling=0, i.e. each "
                        "voxel's time series is divided by its mean, so betas are "
                        "in percent signal change.")
    p.add_argument("--windows", choices=["all", "main"], default="all",
                   help="--basis fir only. 'all' (default): main windows plus "
                        "every specification-curve window in FIR_WINDOWS. "
                        "'main': enc/del/res main windows only.")
    p.add_argument("--stats", nargs="+", choices=["beta", "t"],
                   default=["beta", "t"],
                   help="Which maps to save (default both).")
    p.add_argument("--extra-mask-dir", type=Path, default=None,
                   help="Folder whose .nii.gz masks (searched recursively, e.g. "
                        "roi_func/sub-XX) are added to the brain mask, so every "
                        "ROI voxel is fitted even at the brain edge.")
    p.add_argument("--sessions", nargs="+", default=["01", "02", "03", "04"])
    p.add_argument("--runs", nargs="+",
                   default=["01", "02", "03", "04", "05", "06", "07", "08"])
    p.add_argument("--n-jobs", type=int, default=18,
                   help="Parallel trial GLMs per run (threads).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing outputs.")
    return p.parse_args()


# ============================================================================
# Path helpers
# ============================================================================
def events_path(events_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (events_dir / f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-bwm_run-{run}_events.tsv")


def _func(fmriprep_dir: Path, sub: str, ses: str) -> Path:
    return fmriprep_dir / f"sub-{sub}" / f"ses-{ses}" / "func"


def bold_path(fmriprep_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (_func(fmriprep_dir, sub, ses) /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}"
            f"_space-T1w_desc-preproc_bold.nii.gz")


def mask_path(fmriprep_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (_func(fmriprep_dir, sub, ses) /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}"
            f"_space-T1w_desc-brain_mask.nii.gz")


def confounds_path(fmriprep_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (_func(fmriprep_dir, sub, ses) /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}"
            f"_desc-confounds_timeseries.tsv")


def run_to_block(run: str) -> str:
    return f"{(int(run) - 1) // RUNS_PER_BLOCK + 1:02d}"


STAT_DIRS = {"beta": "beta", "t": "tmap"}     # --stats value -> folder name


def estimate_dir(root: Path, sub: str, basis: str, stat: str, window: str) -> Path:
    return root / f"sub-{sub}" / basis / STAT_DIRS[stat] / window


def out_path(root: Path, basis: str, stat: str, window: str, sub: str,
             ses: str, run: str, trial: str) -> Path:
    return (estimate_dir(root, sub, basis, stat, window) / "trials" /
            f"sub-{sub}_ses-{ses}_block-{run_to_block(run)}_run-{run}"
            f"_trial-{trial}.nii.gz")


# ============================================================================
# Events and design
# ============================================================================
def read_events(path: Path) -> pd.DataFrame:
    """events.tsv with BIDS column names (accepts onset_s / duration_s)."""
    ev = pd.read_csv(path, sep="\t")
    ev = ev.rename(columns={"onset_s": "onset", "duration_s": "duration"})
    missing = {"onset", "duration", "trial_type", "trial"} - set(ev.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    ev["trial_type"] = ev["trial_type"].astype(str)
    return ev.sort_values("onset").reset_index(drop=True)


def validate_events(ev: pd.DataFrame, n_scans: int | None = None) -> list:
    """Timing problems that make a run unusable (empty list = fine)."""
    problems = []
    enc = ev[ev["trial_type"] == PHASE_EVENTS["enc"]]
    if enc.empty:
        return [f"no '{PHASE_EVENTS['enc']}' events"]
    if (ev["onset"] < 0).any():
        problems.append(f"{int((enc['onset'] < 0).sum())} of {len(enc)} "
                        "encoding onsets are negative")
    short = enc["duration"] < 0.75
    if short.any():
        problems.append(f"{int(short.sum())} encoding events last < 0.75 s")
    if n_scans is not None:
        late = enc["onset"] > n_scans * TR
        if late.any():
            problems.append(f"{int(late.sum())} encoding onsets after the "
                            "last volume")
    if enc["trial"].duplicated().any():
        problems.append("a trial label occurs twice in the run")
    return problems


def phase_onsets(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per trial: enc, del, res onsets (measured, filled if missing)."""
    rows = []
    for lab, g in ev.groupby("trial", sort=False):
        on = {ph: g.loc[g["trial_type"] == name, "onset"]
              for ph, name in PHASE_EVENTS.items()}
        if on["enc"].empty:
            continue
        enc = float(on["enc"].iloc[0])
        dl = float(on["del"].iloc[0]) if not on["del"].empty else np.nan
        res = float(on["res"].iloc[0]) if not on["res"].empty else enc + RES_AFTER_ENC
        rows.append({"trial": str(lab), "enc": enc, "del": dl, "res": res})
    return pd.DataFrame(rows).sort_values("enc").reset_index(drop=True)


def load_trial_events(path: Path, basis: str, frame_times: np.ndarray):
    """Returns (events: onset, duration, phase, trial; timing summary;
    per-trial phase onsets)."""
    ev = read_events(path)
    problems = validate_events(ev, len(frame_times))
    if problems:
        raise ValueError(f"{path.name}: corrupted timing: " + "; ".join(problems)
                         + ". Regenerate this events file from the raw logs.")
    on = phase_onsets(ev)
    timing = {"enc_to_delay_s": float((on["del"] - on["enc"]).median()),
              "enc_to_response_s": float((on["res"] - on["enc"]).median())}

    if basis == "fir":
        # Same stick construction as 02_trialestimates.py.
        out = pd.DataFrame({
            "onset": [min(frame_times, key=lambda x: abs(x - o)) - 0.05
                      for o in on["enc"]],
            "duration": 0.0, "phase": "stick", "trial": on["trial"]})
        return out, timing, on

    d_on = on["enc"] + ENC_DUR
    half = (on["res"] - d_on) / 2
    if (2 * half < MIN_DELAY).any():
        raise ValueError(f"{path.name}: modeled delay shorter than {MIN_DELAY} s "
                         f"in {int((2 * half < MIN_DELAY).sum())} trials")
    parts = [
        pd.DataFrame({"onset": on["enc"], "duration": ENC_DUR, "phase": "enc"}),
        pd.DataFrame({"onset": d_on, "duration": half, "phase": "del_early"}),
        pd.DataFrame({"onset": d_on + half, "duration": half,
                      "phase": "del_late"}),
        pd.DataFrame({"onset": on["res"], "duration": RES_DUR, "phase": "res"}),
    ]
    for p_ in parts:
        p_["trial"] = on["trial"].to_numpy()
    out = pd.concat(parts).sort_values("onset").reset_index(drop=True)
    return out[["onset", "duration", "phase", "trial"]], timing, on


def assignment_check(y: np.ndarray, frame_times: np.ndarray, on: pd.DataFrame,
                     confounds) -> float:
    """t of an all-trials encoding regressor on one (mean) time series."""
    ev = pd.concat([
        pd.DataFrame({"onset": on["enc"], "duration": ENC_DUR, "trial_type": "enc"}),
        pd.DataFrame({"onset": on["res"], "duration": RES_DUR, "trial_type": "res"})])
    kw = dict(hrf_model=HRF_MODEL, drift_model="cosine", high_pass=HIGH_PASS)
    if confounds is not None:
        kw.update(add_regs=confounds.to_numpy(), add_reg_names=list(confounds.columns))
    X = make_first_level_design_matrix(frame_times, ev, **kw)
    y = (y - y.mean()) / (y.std() + 1e-12)
    labels, results = run_glm(y[:, None], X.to_numpy(), noise_model="ols")
    con = t_contrast(labels, results, (X.columns == "enc").astype(float))
    return float(contrast_maps(con)["t"][0])


def lss_events(ev: pd.DataFrame, target: str, basis: str) -> pd.DataFrame:
    who = np.where(ev["trial"].to_numpy() == target, "target", "other")
    if basis == "fir":
        tt = who
    else:
        tt = [f"{w}_{p}" for w, p in zip(who, ev["phase"])]
    return pd.DataFrame({"onset": ev["onset"].to_numpy(),
                         "duration": ev["duration"].to_numpy(),
                         "trial_type": tt})


def make_design(frame_times, events, basis, confounds) -> pd.DataFrame:
    kw = dict(drift_model="cosine", high_pass=HIGH_PASS)
    if basis == "fir":
        kw.update(hrf_model="fir", fir_delays=FIR_DELAYS)
    else:
        kw.update(hrf_model=HRF_MODEL)
    if confounds is not None:
        kw.update(add_regs=confounds.to_numpy(),
                  add_reg_names=list(confounds.columns))
    return make_first_level_design_matrix(frame_times, events, **kw)


def window_keys(basis: str, which: str) -> list:
    """[(window, [target column names]), ...] to compute."""
    if basis == "hrf":
        return [(w, [f"target_{r}" for r in regs]) for w, regs in HRF_OUTPUTS.items()]
    return [(w, [f"target_delay_{d}" for d in delays])
            for w, delays in FIR_WINDOWS.items()
            if which == "all" or "_" not in w]


def window_contrasts(columns, keys: list) -> dict:
    """{window: contrast vector averaging the target columns}."""
    cols = list(columns)
    out = {}
    for w, names in keys:
        missing = [n for n in names if n not in cols]
        if missing:
            raise ValueError(f"Design has no column(s) {missing}")
        v = np.zeros(len(cols))
        for n in names:
            v[cols.index(n)] = 1.0 / len(names)
        out[w] = v
    return out


def design_qc(X: pd.DataFrame, basis: str) -> dict:
    """Residual df, condition number and VIFs of the target regressors."""
    n, p = X.shape
    rank = int(np.linalg.matrix_rank(X.to_numpy()))
    Xc = X.drop(columns=[c for c in X.columns if c == "constant"]).astype(float)
    Xc = Xc - Xc.mean()
    G = Xc.to_numpy().T @ Xc.to_numpy()
    vif_all = pd.Series(np.diag(np.linalg.pinv(G)) * np.diag(G),
                        index=Xc.columns)
    qc = {"n_scans": n, "n_columns": p, "rank": rank,
          "residual_df": n - rank,
          "condition_number": float(np.linalg.cond(X.to_numpy()))}
    if basis == "hrf":
        for r in HRF_REGRESSORS:
            if f"target_{r}" in vif_all.index:
                qc[f"vif_{r}"] = float(vif_all[f"target_{r}"])
    else:
        for w, names in window_keys(basis, "main"):
            names = [c for c in names if c in vif_all.index]
            if names:
                qc[f"vif_{w}_max"] = float(vif_all[names].max())
    return qc


# ============================================================================
# nilearn version compatibility (the analysis env runs Python 3.8, i.e. an
# older nilearn; arguments that only newer versions know are left out there)
# ============================================================================
_RESAMPLE_KW = {k: True for k in ("force_resample", "copy_header")
                if k in inspect.signature(resample_to_img).parameters}


# compute_contrast: 'contrast_type' up to nilearn 0.9, 'stat_type' later
_CONTRAST_KW = ("stat_type" if "stat_type" in inspect.signature(compute_contrast).parameters
                else "contrast_type")


def t_contrast(labels, results, vec):
    return compute_contrast(labels, results, vec, **{_CONTRAST_KW: "t"})


def resample_nearest(img, target):
    return resample_to_img(img, target, interpolation="nearest", **_RESAMPLE_KW)


def contrast_maps(con) -> dict:
    """Beta and t of a nilearn Contrast in any nilearn version."""
    eff = con.effect_size() if hasattr(con, "effect_size") else con.effect
    return {"beta": np.ravel(eff), "t": np.ravel(con.stat())}


# ============================================================================
# Fitting mask
# ============================================================================
_EXTRA_CACHE: dict = {}


def build_mask(mp: Path, img, extra_dir: Path | None):
    """fMRIPrep brain mask (or EPI mask), plus all masks in extra_dir."""
    if mp.exists():
        base = nib.load(str(mp))
    else:
        print(f"   [warn] no brain mask, using EPI mask instead of {mp.name}")
        base = compute_epi_mask(img)
    data = np.asanyarray(base.dataobj) > 0
    if extra_dir is None:
        return base, 0, None

    if extra_dir not in _EXTRA_CACHE:            # union of ROI masks, once
        files = sorted(extra_dir.rglob("*.nii.gz"))
        if not files:
            raise FileNotFoundError(f"No .nii.gz masks under {extra_dir}")
        union = None
        for f in files:
            m = resample_nearest(nib.load(str(f)), base)
            d = np.asanyarray(m.dataobj) > 0
            union = d if union is None else union | d
        _EXTRA_CACHE[extra_dir] = new_img_like(base, union.astype(np.uint8))
    extra = resample_nearest(_EXTRA_CACHE[extra_dir], base)
    extra_data = np.asanyarray(extra.dataobj) > 0
    n_added = int((extra_data & ~data).sum())
    return new_img_like(base, (data | extra_data).astype(np.uint8)), n_added, extra


# ============================================================================
# Per-run driver
# ============================================================================
def process_run(args, ses: str, run: str) -> None:
    sub = args.subject
    t0 = time.time()
    print(f"\n== sub-{sub} ses-{ses} run-{run} ({args.basis}) ==", flush=True)

    bp = bold_path(args.fmriprep_dir, sub, ses, run)
    img = nib.load(str(bp))
    n_scans = img.shape[-1]
    frame_times = np.arange(n_scans) * TR

    ev, timing, on = load_trial_events(events_path(args.events_dir, sub, ses, run),
                                       args.basis, frame_times)
    print(f"   logged timing (median): encoding->response "
          f"{timing['enc_to_response_s']:.2f} s; modeled: encoding {ENC_DUR:.1f} s, "
          f"delay halves {(timing['enc_to_response_s'] - ENC_DUR) / 2:.2f} s, "
          f"response {RES_DUR} s", flush=True)
    trials = list(dict.fromkeys(ev["trial"]))

    # Skip the run if every output already exists.
    keys = window_keys(args.basis, args.windows)
    targets = [out_path(args.output_dir, args.basis, s, w, sub, ses, run, t)
               for s in args.stats for w, _ in keys for t in trials]
    if not args.force and all(p.exists() for p in targets):
        print("   [done] all outputs exist")
        return

    confounds = None
    if not args.no_motion:
        confounds = (pd.read_csv(confounds_path(args.fmriprep_dir, sub, ses, run),
                                 sep="\t")[MOTION_COLS].fillna(0.0))

    mask_img, n_added, roi_img = build_mask(mask_path(args.fmriprep_dir, sub, ses, run),
                                            img, args.extra_mask_dir)
    masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()
    Y = masker.transform(img).astype(np.float64)
    extra = f" ({n_added} ROI voxels outside the brain mask added)" if n_added else ""
    print(f"   {len(trials)} trials, {Y.shape[1]} voxels{extra}, "
          f"loaded in {time.time() - t0:.0f} s", flush=True)

    # Do these events belong to this BOLD run?
    # ROI voxels in masker order (boolean indexing of the fitting mask, as
    # NiftiMasker does); read from the arrays to avoid version-specific
    # nilearn code paths for integer masks
    roi = None
    if roi_img is not None:
        in_mask = np.asanyarray(mask_img.dataobj) > 0
        roi = (np.asanyarray(roi_img.dataobj) > 0)[in_mask]
    signal = "roi" if roi is not None and roi.any() else "brain"
    y_mean = Y[:, roi].mean(axis=1) if signal == "roi" else Y.mean(axis=1)
    assign = {run: assignment_check(y_mean, frame_times, on, confounds)}
    pat = f"sub-{sub}_ses-{ses}_task-bwm_run-*_events.tsv"
    for p in sorted(events_path(args.events_dir, sub, ses, run).parent.glob(pat)):
        r_ = p.name.split("_run-")[1].split("_")[0]
        if r_ == run:
            continue
        try:
            ev_o = read_events(p)
            if not validate_events(ev_o):
                assign[r_] = assignment_check(y_mean, frame_times, phase_onsets(ev_o),
                                              confounds)
        except Exception:
            pass
    others = {k: v for k, v in assign.items() if k != run}
    best = max(assign, key=assign.get)
    margin = assign[run] - max(others.values()) if others else np.nan
    ok_assign = best == run and (not others or margin >= ASSIGN_MARGIN)
    second = f", best other run-{max(others, key=others.get)} t = {max(others.values()):.1f}" \
        if others else ""
    flag = "" if ok_assign else "   <-- CHECK: events may belong to another BOLD run"
    print(f"   assignment check: own events t = {assign[run]:.1f}{second} "
          f"({signal} mean){flag}", flush=True)
    timing.update({"assign_t_enc": round(assign[run], 2), "assign_best_run": best,
                   "assign_margin": round(float(margin), 2) if others else None,
                   "assign_ok": bool(ok_assign), "assign_signal": signal})

    del Y                                  # only needed for the check above
    # Load the run once into memory; every trial GLM masks from it.
    img = nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine, img.header)
    scaling = False if args.no_signal_scaling else 0

    def one_trial(trial):
        X = make_design(frame_times, lss_events(ev, trial, args.basis),
                        args.basis, confounds)
        glm = FirstLevelModel(t_r=TR, mask_img=mask_img, noise_model=args.noise_model,
                              signal_scaling=scaling, standardize=False,
                              smoothing_fwhm=None, minimize_memory=True, n_jobs=1)
        glm.fit(img, design_matrices=[X])
        out_type = {"beta": "effect_size", "t": "stat"}
        for window, vec in window_contrasts(X.columns, keys).items():
            for stat in args.stats:
                op = out_path(args.output_dir, args.basis, stat, window,
                              sub, ses, run, trial)
                if op.exists() and not args.force:
                    continue
                op.parent.mkdir(parents=True, exist_ok=True)
                glm.compute_contrast(vec, output_type=out_type[stat]).to_filename(str(op))
        return {"subject": sub, "session": ses, "run": run, "trial": trial,
                "basis": args.basis, "motion": not args.no_motion,
                "signal_scaling": not args.no_signal_scaling, **timing,
                **design_qc(X, args.basis)}

    rows = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(one_trial)(t) for t in trials)

    qc_dir = args.output_dir / "qc" / f"sub-{sub}"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc = pd.DataFrame(rows)
    qc.to_csv(qc_dir / f"sub-{sub}_ses-{ses}_run-{run}_{args.basis}_design.csv",
              index=False)
    vif_cols = [c for c in qc.columns if c.startswith("vif_")]
    print(f"   residual df {qc['residual_df'].min()}-{qc['residual_df'].max()}; "
          f"max VIF " + ", ".join(f"{c[4:]} {qc[c].max():.2f}" for c in vif_cols)
          + f"; {time.time() - t0:.0f} s", flush=True)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()
    print("-" * 64)
    import nilearn
    print(f"LSS single-trial estimates for sub-{args.subject} "
          f"(nilearn {nilearn.__version__})")
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
            process_run(args, ses, run)

    print("\nDone.")


if __name__ == "__main__":
    main()