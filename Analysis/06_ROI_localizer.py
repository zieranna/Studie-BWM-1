#!/usr/bin/env python3
"""
06b_localizer_mt.py
===================

MT+ ROI from the motion/color localizer, ALL sessions, one ROI per subject.

Localizer (per session one run): 6-s blocks of motion (moving white dots),
color (static colored dots) and control (static white dots), 1.2-7.2 s
fixation between blocks. MT+ = motion > control.

Per subject
-----------
1. Sessions: every ses-XX with a localizer BOLD in fmripost/ and a design
   matrix in final/sub-XX/duration_files/ses-XX/design_matrix_localizer.xlsx.
2. Per session checks (a failing session is excluded, the others are used):
   * design matrix vs raw log (T_stim - first trigger press): identical ->
     design matrix; constant offset (different scan-start reference) -> the
     version whose onsets fit the V1-V3 signal better; offsets that vary
     across blocks -> excluded.
   * offset profile: the chosen onsets shifted by -24..+24 s. Shifts up to
     +-MAX_LATENCY s are HRF latency (absorbed by the temporal derivative
     in the GLM). A clearly better fit beyond that (e.g. +12 s) is a timing
     error -> excluded. No visual response within +-MAX_LATENCY -> excluded.
   * space: brain mask Dice with the task data >= DICE_MIN.
   * blocks after the end of the scan are dropped (excluded if > 25 %).
   With --session-exclusion none, the timing and visual-response checks only
   warn: every session is used with the documented design-matrix timing
   (unless the raw log offset fits clearly better), and only structural
   problems exclude a session.
3. One GLM over all valid sessions: canonical HRF + temporal derivative,
   cosine drift (128 s), AR(1), signal_scaling=0, 6 motion regressors +
   non-steady-state volumes. Maps: motion>control (MT+), motion>color,
   color>control, stimulus>baseline (t, z, beta).
4. MT+ ROI, per hemisphere: within the HCP-MMP1 MT+ complex (MT, MST, FST,
   V4t; Glasser et al., 2016; --search-labels) dilated by --cluster-dilate-mm,
   the contiguous clusters (26-connectivity) of motion>control z >
   --z-threshold (3.09 = p < .001). A cluster counts only if its peak is a
   true local maximum of the whole z-map (not the flank of a peak outside the
   region, such as V6/DVT). Of those, the one with the highest peak is the
   hemisphere's MT+; its size is free.
   Cluster-level inference: All-Resolution Inference (Rosenblatt et al.,
   2018; nilearn.glm.cluster_level_inference) within the search region
   (both hemispheres, like a small-volume correction) gives a lower bound on
   the proportion of truly active voxels in each cluster, valid
   simultaneously for all clusters at --alpha. A hemisphere is used only if
   its cluster's bound is > 0 (cluster-level significant). Verdict:
       confirmed       significant, peak inside the MT+ parcels
       next to parcel  significant, peak within --cluster-dilate-mm of them
       not significant cluster found, but ARI bound = 0 -> left out
       not found       no such cluster (>= --min-voxels) -> left out

Output
------
    <roi-out>/sub-YY/MT.nii.gz        the ROI (valid hemispheres), task grid,
                                      task numbering -- the only ROI file
    <out-dir>/sub-XX/                 maps, events, designs, timing profiles,
                                      localizer_summary.csv (final numbering)
"""

from __future__ import annotations

import argparse
import inspect
import re
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm import cluster_level_inference
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import (FirstLevelModel,
                                     make_first_level_design_matrix, run_glm)
from nilearn.image import new_img_like, resample_to_img
from scipy import ndimage

warnings.filterwarnings("ignore", message=".*boolean values for 'standardize'.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Resampling binary images.*")
warnings.filterwarnings("ignore", message=".*Generation of a mask has been requested.*")
warnings.filterwarnings("ignore", message=".*null duration.*")

# ============================================================================
# Paths
# ============================================================================
STUDY        = Path("/home/annaz/BindingWM/Studie-BWM-1")
REV          = Path("/home/annaz/BindingWM/Studie-BWM-1_CP_revision1")
FINAL_DIR    = STUDY / "final"
FMRIPOST_DIR = STUDY / "fmripost"                    # localizer BOLD (final numbering)
TASK_DIR     = STUDY / "bindingWM" / "preprocessed"  # task BOLD (BIDS numbering)
MAPPING      = REV / "timing_check" / "subject_mapping.csv"
OUT_DIR      = REV / "localizer"
ROI_OUT_DIR  = REV / "roi_localizer"

# ============================================================================
# Settings
# ============================================================================
TR = 1.2
HRF = "spm + derivative"
CONDITIONS = ["motion", "color", "control"]
CONTRASTS = {
    "motionGTcontrol": "motion - control",      # MT+
    "motionGTcolor":   "motion - color",
    "colorGTcontrol":  "color - control",
    "stimGTbaseline":  "motion + color + control",
}
MT_LABELS = ["MT", "MST", "FST", "V4t"]          # HCP-MMP1 MT+ complex
VISUAL_LABELS = ["V1", "V2", "V3"]               # signal for the timing checks
MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
ONSET_TOL = 0.1          # s, design matrix vs raw log
MAX_LATENCY = 3.0        # s, shifts up to this are HRF latency, beyond: timing error
OFFSET_RANGE = 24.0      # s, offset profile
OFFSET_STEP = 0.3        # s
TIMING_MARGIN = 2.0      # t, "clearly better"
MIN_T_VISUAL = 3.0       # t, a session must evoke a visual response
DICE_MIN = 0.85
MAX_LOST_BLOCKS = 0.25

_RESAMPLE_KW = {k: True for k in ("force_resample", "copy_header")
                if k in inspect.signature(resample_to_img).parameters}
MASK_EXTS = (".nii.gz", ".nii", ".mgz")


# ============================================================================
# Small helpers
# ============================================================================
def resample(img, target, interpolation):
    return resample_to_img(img, target, interpolation=interpolation, **_RESAMPLE_KW)


def boolean(img) -> np.ndarray:
    return np.asanyarray(img.dataobj) > 0


def to_nifti(img):
    return img if isinstance(img, nib.Nifti1Image) else nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine)


def subject_mapping() -> dict:
    if not MAPPING.exists():
        return {}
    m = pd.read_csv(MAPPING, dtype=str)
    return {f"{int(r.final_subject):02d}": f"{int(r.behav_subject):02d}" for r in m.itertuples()}


def loc_file(root: Path, sub: str, ses: str, suffix: str) -> Path:
    return (root / f"sub-{sub}" / f"ses-{ses}" / "func" /
            f"sub-{sub}_ses-{ses}_task-perception_acq-TR1200_run-01_{suffix}")


def find_mask(parc_dir: Path, task_sub: str, hemi: str, label: str):
    for ext in MASK_EXTS:
        hits = sorted((parc_dir / f"sub-{task_sub}").rglob(f"{hemi}_{label}_ROI{ext}"))
        if hits:
            return hits[0]
    return None


def parcel_region(parc_dir: Path, task_sub: str, labels: list, hemi: str, ref_img):
    """Union of HCP-MMP1 masks on the grid of ref_img (as in 05: >=0.5, then
    continuous resampling, >=0.1). None if a mask is missing."""
    union = None
    for lab in labels:
        f = find_mask(parc_dir, task_sub, hemi, lab)
        if f is None:
            return None
        img = to_nifti(nib.load(str(f)))
        b = new_img_like(img, (img.get_fdata() >= 0.5).astype(np.float32))
        d = resample(b, ref_img, "continuous").get_fdata() >= 0.1
        union = d if union is None else union | d
    return union


def dilate(mask: np.ndarray, mm: float, zooms) -> np.ndarray:
    it = int(round(mm / float(min(zooms[:3]))))
    return mask if it <= 0 else ndimage.binary_dilation(
        mask, structure=ndimage.generate_binary_structure(3, 1), iterations=it)


# ============================================================================
# Events and timing
# ============================================================================
def load_events(sub: str, ses: str):
    """(design-matrix events, raw-log events or None)."""
    dm = FINAL_DIR / f"sub-{sub}" / "duration_files" / f"ses-{ses}" / "design_matrix_localizer.xlsx"
    ev = pd.read_excel(dm)[["onset", "duration", "trial_type"]]
    ev["trial_type"] = ev["trial_type"].astype(str).str.strip().str.lower()
    ev = ev.sort_values("onset").reset_index(drop=True)
    bad = set(ev["trial_type"]) - set(CONDITIONS)
    if bad:
        raise ValueError(f"unknown conditions {bad}")
    raw_dir = FINAL_DIR / f"sub-{sub}" / "behav" / f"ses-{ses}" / "Localizer"
    res, press = raw_dir / "Local_results.xlsx", raw_dir / "tr_5press.xlsx"
    if not (res.exists() and press.exists()):
        return ev, None
    r = pd.read_excel(res)
    t0 = pd.read_excel(press)["time"].iloc[0]
    on = ((r["T_stim"] - r["ExpTimeStart"]) - t0).to_numpy()
    if len(on) != len(ev) or not (r["Mode"].astype(str).str.lower().to_numpy() == ev["trial_type"].to_numpy()).all():
        return ev, "mismatch"
    raw = ev.copy()
    raw["onset"] = np.sort(on)
    return ev, raw


def fit_t(y: np.ndarray, n_vols: int, ev: pd.DataFrame, conf) -> float:
    """t of stimulus > baseline (main regressors) for one time series, OLS."""
    kw = dict(hrf_model=HRF, drift_model="cosine", high_pass=0.01)
    if conf is not None:
        kw.update(add_regs=conf.to_numpy(), add_reg_names=list(conf.columns))
    X = make_first_level_design_matrix(np.arange(n_vols) * TR, ev, **kw)
    y = (y - y.mean()) / (y.std() + 1e-12)
    labels, results = run_glm(y[:, None], X.to_numpy(), noise_model="ols")
    con = compute_contrast(labels, results, X.columns.isin(CONDITIONS).astype(float), "t")
    return float(np.ravel(con.stat())[0])


def profile(y, n_vols, ev, conf) -> pd.DataFrame:
    shifts = np.round(np.arange(-OFFSET_RANGE, OFFSET_RANGE + 1e-9, OFFSET_STEP), 2)
    ts = []
    for s in shifts:
        e = ev.copy()
        e["onset"] = e["onset"] + s
        ts.append(fit_t(y, n_vols, e, conf))
    return pd.DataFrame({"shift_s": shifts, "t": ts})


def near_far(prof: pd.DataFrame) -> dict:
    near = prof[prof["shift_s"].abs() <= MAX_LATENCY]
    far = prof[prof["shift_s"].abs() > MAX_LATENCY]
    i, j = near["t"].idxmax(), far["t"].idxmax()
    return {"t_near": float(prof.at[i, "t"]), "latency_s": float(prof.at[i, "shift_s"]),
            "t_far": float(prof.at[j, "t"]), "far_shift_s": float(prof.at[j, "shift_s"])}


def visual_signal(data, mask, img, task_sub, parc_dir) -> tuple:
    region = mask if mask is not None else data.std(axis=-1) > 0
    if parc_dir is not None:
        ref = img.slicer[..., 0]
        vis = None
        for h in ("L", "R"):
            v = parcel_region(parc_dir, task_sub, VISUAL_LABELS, h, ref)
            if v is not None:
                vis = v if vis is None else vis | v
        if vis is not None and (vis & region).any():
            return data[vis & region].mean(axis=0), "V1-V3"
    return data[region].mean(axis=0), "brain"


# ============================================================================
# One session
# ============================================================================
def check_session(sub, task_sub, ses, args, out, task_mask) -> dict:
    """Validate one localizer session; returns a dict with 'ok' and, if ok,
    the image, events and confounds for the GLM."""
    info = {"session": ses, "ok": False}
    if ses in args.exclude_sessions:
        info["reason"] = "excluded by --exclude-sessions"
        return info
    bold = loc_file(args.fmripost_dir, sub, ses, "space-T1w_desc-preproc_bold.nii.gz")
    dm_path = FINAL_DIR / f"sub-{sub}" / "duration_files" / f"ses-{ses}" / "design_matrix_localizer.xlsx"
    if not dm_path.exists():
        info["reason"] = "no design matrix"
        return info
    img = nib.load(str(bold))
    n_vols = img.shape[-1]
    mask_p = loc_file(args.fmripost_dir, sub, ses, "space-T1w_desc-brain_mask.nii.gz")
    mask_img = nib.load(str(mask_p)) if mask_p.exists() else None
    conf_p = loc_file(args.fmripost_dir, sub, ses, "desc-confounds_timeseries.tsv")
    conf = None
    if conf_p.exists():
        c = pd.read_csv(conf_p, sep="\t")
        conf = c[MOTION_COLS + [k for k in c.columns if k.startswith("non_steady_state")]].fillna(0.0)

    # space
    if task_mask is not None and mask_img is not None:
        same = task_mask.shape[:3] == mask_img.shape[:3] and np.allclose(task_mask.affine, mask_img.affine, atol=1e-3)
        a = boolean(task_mask)
        b = boolean(mask_img if same else resample(mask_img, task_mask, "nearest"))
        info["dice"] = round(float(2 * (a & b).sum() / (a.sum() + b.sum())), 3)
        if info["dice"] < DICE_MIN and not args.ignore_space_check:
            info["reason"] = f"space check failed (Dice {info['dice']})"
            return info

    # timing candidates
    ev_dm, ev_raw = load_events(sub, ses)
    if isinstance(ev_raw, str):
        info["reason"] = "raw log and design matrix list different blocks"
        return info
    cands = {"design matrix": ev_dm}
    warn = []                        # reasons that only warn with --session-exclusion none
    if ev_raw is not None and args.timing != "dm":
        d = ev_dm["onset"].to_numpy() - ev_raw["onset"].to_numpy()
        info["dm_minus_raw_s"] = round(float(np.median(d)), 2)
        if args.timing == "raw":
            cands = {"raw log": ev_raw}
        elif np.abs(d).max() > ONSET_TOL:
            if np.ptp(d) > ONSET_TOL:
                msg = f"design matrix and raw log differ by {d.min():+.1f} to {d.max():+.1f} s"
                if args.session_exclusion == "qc":
                    info["reason"] = msg
                    return info
                warn.append(msg + " -> design matrix used")
            else:
                cands["raw log"] = ev_raw
    shift = args.timing_shift.get(ses, 0.0)
    if shift:
        for k in cands:
            cands[k] = cands[k].assign(onset=cands[k]["onset"] + shift)
        info["manual_shift_s"] = shift

    data = img.get_fdata(dtype=np.float32)
    y, info["signal"] = visual_signal(data, boolean(mask_img) if mask_img is not None else None,
                                      img, task_sub, args.parcellation_dir)
    del data
    profs = {k: profile(y, n_vols, ev, conf) for k, ev in cands.items()}
    nf = {k: near_far(p) for k, p in profs.items()}
    best = max(nf, key=lambda k: nf[k]["t_near"])
    if len(nf) == 2:
        other = [k for k in nf if k != best][0]
        info["t_dm"], info["t_raw"] = round(nf["design matrix"]["t_near"], 2), round(nf["raw log"]["t_near"], 2)
        if nf[best]["t_near"] - nf[other]["t_near"] < TIMING_MARGIN:
            if args.session_exclusion == "qc":
                info["reason"] = "design matrix and raw log fit equally well (use --timing)"
                return info
            warn.append("design matrix and raw log fit equally well -> design matrix used")
            best = "design matrix"
    info["timing"] = best
    info.update({k: round(v, 2) for k, v in nf[best].items()})
    profs[best].to_csv(out / f"ses-{ses}_timing_offset_profile.csv", index=False)
    qc = []
    if nf[best]["t_far"] - nf[best]["t_near"] >= TIMING_MARGIN:      # timing error first
        qc.append(f"timing error: onsets shifted by {nf[best]['far_shift_s']:+.1f} s fit "
                  f"clearly better (t {nf[best]['t_far']:.1f} vs {nf[best]['t_near']:.1f})")
    elif nf[best]["t_near"] < MIN_T_VISUAL:
        qc.append(f"no visual response (t {nf[best]['t_near']:.1f})")
    if qc and args.session_exclusion == "qc":
        info["reason"] = qc[0]
        return info
    warn += qc

    ev = cands[best]
    lost = ev["onset"] >= n_vols * TR
    if lost.mean() > MAX_LOST_BLOCKS:
        info["reason"] = f"{int(lost.sum())} of {len(ev)} blocks after the end of the scan"
        return info
    info["blocks_dropped"] = int(lost.sum())
    ev = ev[~lost].reset_index(drop=True)
    ev.to_csv(out / f"ses-{ses}_events.tsv", sep="\t", index=False)
    info.update({"ok": True, "img": img, "events": ev, "confounds": conf, "mask": mask_img,
                 "n_blocks": len(ev), "warning": "; ".join(warn)})
    return info


# ============================================================================
# MT+ per hemisphere
# ============================================================================
def mt_hemisphere(t, search_h, allowed_h, zooms, thr, min_vox):
    comp, n = ndimage.label((t > thr) & allowed_h, structure=np.ones((3, 3, 3)))
    if n == 0:
        return None, {"verdict": "not found"}
    local_max = t >= ndimage.maximum_filter(t, size=3, mode="nearest")
    dist = ndimage.distance_transform_edt(~search_h, sampling=zooms)
    best = None
    for c in range(1, n + 1):
        vox = comp == c
        size = int(vox.sum())
        if size < min_vox:
            continue
        idx = np.flatnonzero(vox.ravel())
        pk = np.unravel_index(idx[np.argmax(t.ravel()[idx])], t.shape)
        if not local_max[pk]:
            continue                       # flank of a stronger peak outside the region
        if best is None or t[pk] > best[1]:
            best = (vox, float(t[pk]), pk, size)
    if best is None:
        return None, {"verdict": "not found"}
    vox, pt, pk, size = best
    d = float(dist[pk])
    return vox, {"verdict": "confirmed" if d == 0 else "next to parcel", "peak_t": round(pt, 2),
                 "voxels": size, "peak_dist_mm": round(d, 1),
                 "in_parcel": round(float((vox & search_h).sum() / size), 2)}


# ============================================================================
# One subject
# ============================================================================
def process(sub: str, task_sub: str, args, row: dict) -> None:
    row.update({"final_subject": sub, "task_subject": task_sub})
    out = args.out_dir / f"sub-{sub}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n== task sub-{task_sub} (final sub-{sub}) ==", flush=True)

    if args.parcellation_dir is not None and find_mask(args.parcellation_dir, task_sub, "L", "MT") is None:
        raise FileNotFoundError(f"no HCP-MMP1 masks for task sub-{task_sub} under {args.parcellation_dir}")
    task_mask_p = (TASK_DIR / f"sub-{task_sub}" / "ses-01" / "func" /
                   f"sub-{task_sub}_ses-01_task-bwm_acq-TR1200_run-01_space-T1w_desc-brain_mask.nii.gz")
    task_mask = nib.load(str(task_mask_p)) if task_mask_p.exists() else None

    sessions = sorted(p.name[4:] for p in (args.fmripost_dir / f"sub-{sub}").glob("ses-*")
                      if loc_file(args.fmripost_dir, sub, p.name[4:],
                                  "space-T1w_desc-preproc_bold.nii.gz").exists())
    if not sessions:
        raise FileNotFoundError(f"no localizer BOLD under {args.fmripost_dir / f'sub-{sub}'}")
    infos = []
    for ses in sessions:
        try:
            i = check_session(sub, task_sub, ses, args, out, task_mask)
        except Exception as e:
            i = {"session": ses, "ok": False, "reason": f"error: {e}"}
        infos.append(i)
        if i["ok"]:
            lat = f", latency {i['latency_s']:+.1f} s" if i.get("latency_s") else ""
            via = f" ({i['timing']}, design matrix - raw {i['dm_minus_raw_s']:+.1f} s)" \
                if i.get("dm_minus_raw_s") else f" ({i['timing']})"
            w = f"  [kept despite: {i['warning']}]" if i.get("warning") else ""
            print(f"   ses-{ses}: ok{via}, visual t {i['t_near']:.1f}{lat}, "
                  f"{i['n_blocks']} blocks{w}", flush=True)
        else:
            print(f"   ses-{ses}: EXCLUDED - {i['reason']}", flush=True)
    pd.DataFrame([{k: v for k, v in i.items() if k not in ("img", "events", "confounds", "mask")}
                  for i in infos]).to_csv(out / "sessions.csv", index=False)
    good = [i for i in infos if i["ok"]]
    row.update({"session_exclusion": args.session_exclusion,
                "sessions_warned": "; ".join(f"ses-{i['session']}: {i['warning']}"
                                             for i in good if i.get("warning")),
                "sessions_found": len(infos), "sessions_used": len(good),
                "sessions_excluded": "; ".join(f"ses-{i['session']}: {i['reason']}"
                                               for i in infos if not i["ok"])})
    if not good:
        raise ValueError("no valid localizer session")

    # GLM over all valid sessions
    ref = good[0]["img"]
    glm = FirstLevelModel(t_r=TR, slice_time_ref=0.0, hrf_model=HRF, drift_model="cosine",
                          high_pass=0.01, noise_model="ar1", signal_scaling=0, standardize=False,
                          mask_img=good[0]["mask"], smoothing_fwhm=args.smoothing,
                          minimize_memory=True)
    glm.fit([i["img"] for i in good], events=[i["events"] for i in good],
            confounds=[i["confounds"] for i in good])
    for i, X in zip(good, glm.design_matrices_):
        X.to_csv(out / f"ses-{i['session']}_design.csv")
    row["residual_df"] = int(sum(X.shape[0] - np.linalg.matrix_rank(X.to_numpy())
                                 for X in glm.design_matrices_))
    for c in CONDITIONS:
        row[f"n_{c}"] = int(sum((i["events"]["trial_type"] == c).sum() for i in good))
    t_img = None
    for name, expr in CONTRASTS.items():
        for out_type, tag in (("stat", "t"), ("z_score", "z"), ("effect_size", "beta")):
            m = glm.compute_contrast(expr, output_type=out_type)
            m.to_filename(str(out / f"sub-{sub}_task-perception_contrast-{name}_stat-{tag}.nii.gz"))
            if name == "motionGTcontrol" and tag == "z":
                t_img = m          # z-map: clusters and ARI work on z
    print(f"   GLM: {len(good)} session(s), {row['n_motion']} motion / {row['n_control']} "
          f"control blocks, residual df {row['residual_df']}", flush=True)

    # MT+ ROI
    if args.parcellation_dir is None:
        row["roi_status"] = "maps only (no --parcellation-dir)"
        return
    t = t_img.get_fdata()
    zooms = t_img.header.get_zooms()
    search = {h: parcel_region(args.parcellation_dir, task_sub, args.search_labels, h, t_img)
              for h in ("L", "R")}
    if any(v is None for v in search.values()):
        raise FileNotFoundError(f"missing HCP-MMP1 masks for {args.search_labels}")
    allowed = {h: dilate(search[h], args.cluster_dilate_mm, zooms) & ~search[o]
               for h, o in (("L", "R"), ("R", "L"))}
    brain = t != 0
    ari_mask = new_img_like(t_img, ((allowed["L"] | allowed["R"]) & brain).astype(np.uint8))
    tdp = cluster_level_inference(t_img, mask_img=ari_mask, threshold=args.t_threshold,
                                  alpha=args.alpha).get_fdata()
    row.update({"search_labels": " ".join(args.search_labels), "ari_alpha": args.alpha,
                "z_threshold": args.t_threshold})
    roi = np.zeros(t.shape, bool)
    for h in ("L", "R"):
        vox, res = mt_hemisphere(t, search[h], allowed[h], zooms, args.t_threshold, args.min_voxels)
        if vox is not None:
            # mean of nilearn's per-cluster bounds = valid (conservative) bound
            # for our 26-connected cluster, which consists of whole 6-connected ones
            res["tdp_lower_bound"] = round(float(tdp[vox].mean()), 3)
            if res["tdp_lower_bound"] <= 0:
                res["verdict"] = "not significant"
                vox = None
        row.update({f"{h}_{k}": v for k, v in res.items()})
        if vox is not None:
            roi |= vox
        if "peak_t" in res:
            desc = (f"{res['verdict']}, peak z {res['peak_t']}, {res['voxels']} voxels, "
                    f"{int(res['in_parcel'] * 100)} % in parcel, true-discovery proportion "
                    f">= {res['tdp_lower_bound']:.2f} (ARI, alpha {args.alpha})")
        else:
            desc = res["verdict"]
        print(f"   MT+ {h}: {desc}", flush=True)
    if not roi.any():
        row["roi_status"] = "no valid MT+"
        print("   no valid MT+ in either hemisphere: no ROI written", flush=True)
        return

    target = task_mask if task_mask is not None else t_img
    img_roi = new_img_like(t_img, roi.astype(np.uint8))
    if task_mask is not None and not (task_mask.shape[:3] == t.shape[:3]
                                      and np.allclose(task_mask.affine, t_img.affine, atol=1e-3)):
        img_roi = resample(img_roi, task_mask, "nearest")
    path = args.roi_out_dir / f"sub-{task_sub}" / "MT.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_img_like(target, boolean(img_roi).astype(np.uint8)).to_filename(str(path))
    row.update({"roi_voxels": int(boolean(img_roi).sum()), "roi_status": "ok"})
    print(f"   ROI: {path} ({row['roi_voxels']} voxels)", flush=True)


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="MT+ ROI from the localizer (all sessions).")
    who = p.add_mutually_exclusive_group()
    who.add_argument("--task-subjects", nargs="+", help="task (BIDS) numbering, e.g. 01")
    who.add_argument("--subjects", nargs="+", help="final numbering, e.g. 02")
    p.add_argument("--task-subject", default=None, help="task subject for a single --subjects entry")
    p.add_argument("--parcellation-dir", "--parcellation_dir", type=Path, default=None,
                   help="folder containing sub-YY/ (task numbering), e.g. .../hpcmmp1_output")
    p.add_argument("--fmripost-dir", type=Path, default=FMRIPOST_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--roi-out-dir", type=Path, default=ROI_OUT_DIR)
    p.add_argument("--smoothing", type=float, default=None)
    p.add_argument("--z-threshold", "--t-threshold", dest="t_threshold", type=float, default=3.09,
                   help="cluster-forming threshold on the motion>control z-map "
                        "(default 3.09 = p < .001 one-sided)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="error level of the cluster-level inference (ARI)")
    p.add_argument("--search-labels", nargs="+", default=MT_LABELS,
                   help="HCP-MMP1 areas of the search region (default: MT MST FST V4t)")
    p.add_argument("--cluster-dilate-mm", type=float, default=5.0)
    p.add_argument("--min-voxels", type=int, default=10)
    p.add_argument("--timing", choices=["auto", "dm", "raw"], default="auto")
    p.add_argument("--timing-shift", nargs="+", default=[], metavar="SES=SECONDS",
                   help="manual shift per session after inspecting its profile, e.g. 03=12.0")
    p.add_argument("--exclude-sessions", nargs="+", default=[], metavar="SES")
    p.add_argument("--session-exclusion", choices=["qc", "none"], default="qc",
                   help="'qc' (default): sessions failing the timing/visual-response checks "
                        "are excluded. 'none': all sessions are used (documented design-matrix "
                        "timing); those checks are only reported as warnings. Structural "
                        "problems (no design matrix, different blocks, space check, most "
                        "blocks after the scan) still exclude a session.")
    p.add_argument("--ignore-space-check", action="store_true")
    args = p.parse_args()
    args.timing_shift = {f"{int(k):02d}": float(v) for k, v in (x.split("=") for x in args.timing_shift)}
    args.exclude_sessions = [f"{int(s):02d}" for s in args.exclude_sessions]
    if args.parcellation_dir is not None and args.parcellation_dir.name in ("label", "masks"):
        raise SystemExit("--parcellation-dir must be the folder that CONTAINS sub-XX/")

    mapping = subject_mapping()
    to_final = {v: k for k, v in mapping.items()}
    if args.task_subjects:
        subs = []
        for t in args.task_subjects:
            t = f"{int(t):02d}"
            if t not in to_final:
                raise SystemExit(f"task sub-{t} is not in {MAPPING}")
            subs.append(to_final[t])
    else:
        subs = [f"{int(s):02d}" for s in (args.subjects or sorted(
            d.name[4:] for d in args.fmripost_dir.glob("sub-*") if d.is_dir()))]

    rows = []
    for sub in subs:
        task_sub = args.task_subject if (args.task_subject and len(subs) == 1) else mapping.get(sub)
        row = {}
        if task_sub is None:
            print(f"\n== final sub-{sub}: not in subject_mapping.csv; skipped")
            row.update({"final_subject": sub, "roi_status": "no task subject"})
        else:
            try:
                process(sub, f"{int(task_sub):02d}", args, row)
            except Exception as e:
                print(f"   [ERROR] {e}")
                row.update({"final_subject": sub, "task_subject": task_sub, "roi_status": f"error: {e}"})
        rows.append(row)
        o = args.out_dir / f"sub-{sub}"
        o.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(o / "localizer_summary.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()