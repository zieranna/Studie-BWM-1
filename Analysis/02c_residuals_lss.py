#!/usr/bin/env python3
"""
02c_residuals_lss.py
====================

GLM residual time series for the LSS FIR models (17 or 25 delays), for the
multivariate noise normalization in pcm_binding.py.

LSS fits one GLM per trial (target trial vs all other trials), so there is
no single residual time series per run. The equivalent used here is ONE GLM
per run with the same design as the LSS models of 02b_trialestimates_lss.py,
except that ALL trials share one FIR regressor set:

    encoding-onset sticks (snapped to the nearest volume - 0.05 s, as in
    02_trialestimates.py), N FIR delays (--n-delays 17 or 25), cosine drift
    (high-pass 0.01 Hz), the 6 motion regressors, signal_scaling=0 (percent
    signal change), the same fitting mask (brain mask + ROI masks)

An LSS model differs from this one only by splitting one trial off into its
own regressor set, so its residuals are practically identical. Residuals are
OLS (Y - X b): pcm_binding.py uses them for the SPATIAL noise correlation
between voxels, for which the temporal AR(1) model of the LSS fits makes no
practical difference.

The functions that read and check the events and build the design are taken
verbatim from 02b_trialestimates_lss.py (FIR branch).

Output (task numbering, 01-20):
    <output-dir>/sub-XX/fir/residuals/sub-XX_ses-YY_run-NN_residuals.nii.gz
        4D, one volume per scan, float32; values only in the saved voxels
        (default: union of the ROI masks in --save-mask-dirs), 0 elsewhere
    ... _residuals.json   residual df, design size, settings

In pcm_binding.py:
    RESID = '<output-dir>/sub-{sub:02d}/fir/residuals/sub-{sub:02d}_ses-*_run-*_residuals.nii.gz'
    RESID_SUB_OFFSET = 0
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import new_img_like, resample_to_img
try:                                    # nilearn >= 0.9
    from nilearn.maskers import NiftiMasker
except ImportError:                     # nilearn 0.8
    from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask

warnings.filterwarnings("ignore", message=".*null duration.*")
warnings.filterwarnings("ignore", category=FutureWarning)

TR = 1.2
HIGH_PASS = 0.01
RUNS_PER_BLOCK = 2
PHASE_EVENTS = {"enc": "encoding", "del": "delay", "res": "response"}
RES_AFTER_ENC = 11.98
MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
_RESAMPLE_KW = {k: True for k in ("force_resample", "copy_header")
                if k in inspect.signature(resample_to_img).parameters}


# ============================================================================
# From 02b_trialestimates_lss.py (verbatim, FIR branch)
# ============================================================================
def events_path(events_dir: Path, sub: str, ses: str, run: str) -> Path:
    return (events_dir / f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-bwm_run-{run}_events.tsv")


def _func(fmriprep_dir: Path, sub: str, ses: str) -> Path:
    return fmriprep_dir / f"sub-{sub}" / f"ses-{ses}" / "func"


def bold_path(fmriprep_dir, sub, ses, run):
    return (_func(fmriprep_dir, sub, ses) /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}_space-T1w_desc-preproc_bold.nii.gz")


def mask_path(fmriprep_dir, sub, ses, run):
    return (_func(fmriprep_dir, sub, ses) /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}_space-T1w_desc-brain_mask.nii.gz")


def confounds_path(fmriprep_dir, sub, ses, run):
    return (_func(fmriprep_dir, sub, ses) /
            f"sub-{sub}_ses-{ses}_task-bwm_acq-TR1200_run-{run}_desc-confounds_timeseries.tsv")


def read_events(path: Path) -> pd.DataFrame:
    ev = pd.read_csv(path, sep="\t")
    ev = ev.rename(columns={"onset_s": "onset", "duration_s": "duration"})
    missing = {"onset", "duration", "trial_type", "trial"} - set(ev.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    ev["trial_type"] = ev["trial_type"].astype(str)
    return ev.sort_values("onset").reset_index(drop=True)


def validate_events(ev: pd.DataFrame, n_scans=None) -> list:
    problems = []
    enc = ev[ev["trial_type"] == PHASE_EVENTS["enc"]]
    if enc.empty:
        return [f"no '{PHASE_EVENTS['enc']}' events"]
    if (ev["onset"] < 0).any():
        problems.append(f"{int((enc['onset'] < 0).sum())} of {len(enc)} encoding onsets are negative")
    short = enc["duration"] < 0.75
    if short.any():
        problems.append(f"{int(short.sum())} encoding events last < 0.75 s")
    if n_scans is not None:
        late = enc["onset"] > n_scans * TR
        if late.any():
            problems.append(f"{int(late.sum())} encoding onsets after the last volume")
    if enc["trial"].duplicated().any():
        problems.append("a trial label occurs twice in the run")
    return problems


def phase_onsets(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, g in ev.groupby("trial", sort=False):
        on = {ph: g.loc[g["trial_type"] == name, "onset"] for ph, name in PHASE_EVENTS.items()}
        if on["enc"].empty:
            continue
        enc = float(on["enc"].iloc[0])
        dl = float(on["del"].iloc[0]) if not on["del"].empty else np.nan
        res = float(on["res"].iloc[0]) if not on["res"].empty else enc + RES_AFTER_ENC
        rows.append({"trial": str(lab), "enc": enc, "del": dl, "res": res})
    return pd.DataFrame(rows).sort_values("enc").reset_index(drop=True)


def fir_sticks(path: Path, frame_times: np.ndarray) -> pd.DataFrame:
    """load_trial_events(..., basis='fir') of 02b: one stick per trial."""
    ev = read_events(path)
    problems = validate_events(ev, len(frame_times))
    if problems:
        raise ValueError(f"{path.name}: corrupted timing: " + "; ".join(problems))
    on = phase_onsets(ev)
    return pd.DataFrame({
        "onset": [min(frame_times, key=lambda x: abs(x - o)) - 0.05 for o in on["enc"]],
        "duration": 0.0, "trial_type": "trial"})          # ALL trials: one regressor set


def resample_nearest(img, target):
    return resample_to_img(img, target, interpolation="nearest", **_RESAMPLE_KW)


def union_mask(dirs, target):
    """Union of all .nii.gz masks below the given folders, on the target grid."""
    union = None
    for d in dirs:
        for f in sorted(Path(d).rglob("*.nii.gz")):
            m = np.asanyarray(resample_nearest(nib.load(str(f)), target).dataobj) > 0
            union = m if union is None else union | m
    return union


# ============================================================================
# Per run
# ============================================================================
def process_run(args, ses, run, fir_delays, save_cache):
    sub = args.subject
    out = (args.output_dir / f"sub-{sub}" / "fir" / "residuals" /
           f"sub-{sub}_ses-{ses}_run-{run}_residuals.nii.gz")
    if out.exists() and not args.force:
        print(f"  ses-{ses} run-{run}: exists, skipped")
        return
    t0 = time.time()
    img = nib.load(str(bold_path(args.fmriprep_dir, sub, ses, run)))
    n = img.shape[-1]
    frame_times = np.arange(n) * TR
    sticks = fir_sticks(events_path(args.events_dir, sub, ses, run), frame_times)
    conf = None
    if not args.no_motion:
        conf = pd.read_csv(confounds_path(args.fmriprep_dir, sub, ses, run),
                           sep="\t")[MOTION_COLS].fillna(0.0)
    kw = dict(drift_model="cosine", high_pass=HIGH_PASS, hrf_model="fir", fir_delays=fir_delays)
    if conf is not None:
        kw.update(add_regs=conf.to_numpy(), add_reg_names=list(conf.columns))
    X = make_first_level_design_matrix(frame_times, sticks, **kw)

    # fitting mask as in 02b: brain mask (or EPI mask) + ROI masks
    mp = mask_path(args.fmriprep_dir, sub, ses, run)
    base = nib.load(str(mp)) if mp.exists() else compute_epi_mask(img)
    fit = np.asanyarray(base.dataobj) > 0
    if args.extra_mask_dir is not None:
        fit |= union_mask([args.extra_mask_dir], base)
    # voxels written to the file
    if "save" not in save_cache:
        save_cache["save"] = union_mask(args.save_mask_dirs, base) if args.save_mask_dirs else None
    keep = fit & save_cache["save"] if save_cache["save"] is not None else fit

    masker = NiftiMasker(mask_img=new_img_like(base, keep.astype(np.uint8)), standardize=False).fit()
    Y = masker.transform(img).astype(np.float64)                # scans x voxels
    m = Y.mean(axis=0)
    ok = (Y.std(axis=0) > 0) & (m != 0)
    if not args.no_signal_scaling:                              # = nilearn signal_scaling=0
        Y[:, ok] = 100.0 * (Y[:, ok] / m[ok] - 1.0)
    Y[:, ~ok] = 0.0
    Xa = X.to_numpy()
    beta, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    R = (Y - Xa @ beta).astype(np.float32)
    R[:, ~ok] = 0.0
    rank = int(np.linalg.matrix_rank(Xa))

    vol = np.zeros(keep.shape + (n,), np.float32)
    vol[keep] = R.T
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(".tmp_" + out.name)
    nib.Nifti1Image(vol, base.affine).to_filename(str(tmp))
    tmp.replace(out)
    meta = {"subject": sub, "session": ses, "run": run, "n_scans": n,
            "n_columns": int(Xa.shape[1]), "rank": rank, "residual_df": n - rank,
            "fir_delays": len(fir_delays), "motion": not args.no_motion,
            "signal_scaling": not args.no_signal_scaling, "voxels_saved": int(ok.sum()),
            "model": "all trials, one FIR set; OLS residuals"}
    out.with_suffix("").with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"  ses-{ses} run-{run}: {n} scans, {Xa.shape[1]} columns, df {n - rank}, "
          f"{int(ok.sum())} voxels saved, {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Residuals of the all-trials FIR GLM per run (LSS FIR models).")
    p.add_argument("--subject", required=True)
    p.add_argument("--n-delays", type=int, required=True, help="17 or 25 (as the LSS model)")
    p.add_argument("--events-dir", required=True, type=Path)
    p.add_argument("--fmriprep-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path,
                   help="LSS root of that model (lss or lss_fir25)")
    p.add_argument("--extra-mask-dir", type=Path, default=None,
                   help="ROI masks added to the fitting mask (as --extra-mask-dir in 02b)")
    p.add_argument("--save-mask-dirs", type=Path, nargs="*", default=None,
                   help="save only voxels inside the union of these masks (default: all fitted voxels)")
    p.add_argument("--no-motion", action="store_true")
    p.add_argument("--no-signal-scaling", action="store_true")
    p.add_argument("--sessions", nargs="+", default=["01", "02", "03", "04"])
    p.add_argument("--runs", nargs="+", default=[f"{i:02d}" for i in range(1, 9)])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.subject = f"{int(args.subject):02d}"
    fir_delays = list(range(1, args.n_delays + 1))
    print(f"Residuals sub-{args.subject}: all-trials FIR GLM, {args.n_delays} delays -> "
          f"{args.output_dir}/sub-{args.subject}/fir/residuals/", flush=True)
    cache = {}
    for ses in args.sessions:
        for run in args.runs:
            if not events_path(args.events_dir, args.subject, ses, run).exists():
                continue
            try:
                process_run(args, ses, run, fir_delays, cache)
            except Exception as e:
                print(f"  ses-{ses} run-{run}: ERROR {e}", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()