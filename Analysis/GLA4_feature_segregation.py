#!/usr/bin/env python3
"""
09_voxel_segregation.py
=======================

Voxel-segregation analysis on the per-pair SVM coefficient maps written
by 08_betacoefficients.py.

Pipeline (per subject):

    1. For each analysis (default binding, color, motion): load every
       per-pair coef map, take voxel-wise ABS, average across pairs.
       Write that |·|-averaged map as a NIfTI and keep its within-ROI
       flat vector.
    2. For every pair of analyses (i, j):
         a. Compute a z_thresh-σ THRESHOLD per analysis vector. Because
            the abs-averaged values are non-negative, the empirical
            mean+z·std (the original 'naive' approach) is biased: the
            negative tail of the underlying signed distribution has been
            folded onto the positive side, deflating the std. Three
            options:
              - 'halfnorm'  : fit |N(0, σ²)|;     threshold = z · σ
         b. Per-vector top-σ mask, then union → combined selection.
         c. Angle (degrees) between v_i and v_j restricted to that
            combined mask.
         d. Permutation null: shuffle v_j voxel positions, recompute
            the angle. n_perm draws.
         e. Effect size = (observed − null_mean) / null_std.
         f. Percent overlap |mask1 ∩ mask2| / |mask1 ∪ mask2|.

Group:

    3. Paired one-sided t-test (angle > null_mean) across subjects.
    4. Heatmaps of mean observed angle and t-test p-values.
    5. Per-subject long-format CSV + pickled cache.

Expected input (output of 08_betacoefficients.py):

    <coef-dir>/sub-XX/<analysis>/<roi>/
        coef_top<NNNN>_<phase>_pair-<NN>_<A>vs<B>.nii.gz
"""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import halfnorm, ttest_rel
from tqdm import tqdm

DEFAULT_ANALYSES = ["binding", "color", "motion"]


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Voxel-segregation analysis: angles between "
                    "|·|-averaged SVM coefficient maps across analyses, "
                    "with combined z-σ voxel selection.",
    )
    p.add_argument("--coef-dir", required=True, type=Path,
                   help="Root of 08_betacoefficients.py output "
                        "(<coef-dir>/sub-XX/<analysis>/<roi>/coef_*.nii.gz).")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where per-subject |·|-mean maps and group "
                        "results are written.")
    p.add_argument("--subjects", nargs="+",
                   default=[f"{i:02d}" for i in range(1, 21)],
                   help="Subject labels without 'sub-' (default 01..20).")
    p.add_argument("--analyses", nargs="+", default=DEFAULT_ANALYSES,
                   help=f"Analyses to compare (default: "
                        f"{' '.join(DEFAULT_ANALYSES)}).")
    p.add_argument("--phase", default="del", choices=["enc", "del", "res"],
                   help="Phase tag in the coef filenames (default: del).")
    p.add_argument("--roi", default="early_visual",
                   help="ROI tag in the input path (default: early_visual).")
    p.add_argument("--voxels", type=int, default=700,
                   help="Voxel-count tag in the coef filenames (default: 700).")
    p.add_argument("--z-thresh", type=float, default=2.0,
                   help="σ multiplier for the voxel selection (default: 2.0).")
    p.add_argument("--n-perm", type=int, default=1000,
                   help="Permutations for the angle null (default 1000).")
    p.add_argument("--n-jobs", type=int, default=8,
                   help="Parallel subjects (default 8).")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if pickle cache exists.")
    return p.parse_args()


# ============================================================================
# Paths
# ============================================================================
def coef_dir_for(coef_root: Path, sub: str, analysis: str, roi: str) -> Path:
    return coef_root / f"sub-{sub}" / analysis / f"svmcoef_{roi}"


def find_coef_files(coef_root: Path, sub: str, analysis: str,
                    roi: str, voxels: int, phase: str) -> list[Path]:
    d = coef_dir_for(coef_root, sub, analysis, roi)
    return sorted(d.glob(f"coef_top{voxels:04d}_{phase}_pair-*.nii.gz"))


def absmean_path(output_dir: Path, sub: str, analysis: str,
                 roi: str, voxels: int, phase: str) -> Path:
    return (output_dir / f"sub-{sub}" /
            f"absmean_{analysis}_{roi}_top{voxels:04d}_{phase}.nii.gz")


# ============================================================================
# |·|-mean per analysis
# ============================================================================
def compute_abs_mean(paths: list[Path], out_path: Path
                     ) -> np.ndarray | None:
    if not paths:
        return None
    ref = nib.load(str(paths[0]))
    ref_data = ref.get_fdata()
    mask = ref_data != 0
    stack = np.empty((len(paths),) + ref_data.shape, dtype=np.float32)
    for i, p in enumerate(paths):
        stack[i] = np.abs(np.nan_to_num(nib.load(str(p)).get_fdata()))
    mean_data = stack.mean(axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mean_data, ref.affine, ref.header), str(out_path))
    return mean_data[mask]


# ============================================================================
# Threshold computation — three methods
# ============================================================================

def threshold_value(v: np.ndarray, z_thresh: float) -> float:
    """Top-percentile threshold. z_thresh interpreted as Gaussian z-score,
    mapped to the equivalent two-sided tail percentile (e.g. z=2 → top 2.275%)."""
    from scipy.stats import norm
    pct = norm.sf(z_thresh) * 100  # one-sided tail
    thr = float(np.percentile(v, 100 - pct))
    n_pass = int((v > thr).sum())
    print(f"  threshold_value: top {pct:.2f}% → thr={thr:.4f}  ({n_pass}/{len(v)} pass)")
    return thr

# ============================================================================
# Combined voxel selection
# ============================================================================
def threshold_by_reference(v1: np.ndarray, v2: np.ndarray,
                           thr1: float, thr2: float
                           ) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Union of the per-vector top-σ masks; both vectors restricted to it."""
    mask1 = v1 > thr1
    mask2 = v2 > thr2

    combined = mask1 | mask2
    overlap  = mask1 & mask2

    n_combined = int(combined.sum())
    pct = (100.0 * int(overlap.sum()) / n_combined) if n_combined > 0 else 0.0
    return v1[combined], v2[combined], pct, n_combined


# ============================================================================
# Angle + permutation null
# ============================================================================
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 ==  0:
        return float("nan")
    cos = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def null_angles(v1: np.ndarray, v2: np.ndarray, n_perm: int,
                rng: np.random.Generator) -> np.ndarray:
    """Vectorised permutation null: shuffle v2's voxel positions n_perm
    times and recompute the angle to v1."""
    n1 = np.linalg.norm(v1)
    if n1 == 0:
        return np.full(n_perm, np.nan)
    v1n = v1 / n1
    perms = np.argsort(rng.random((n_perm, len(v2))), axis=1)
    v2p = v2[perms]
    norms = np.linalg.norm(v2p, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v2n = v2p / norms
    cos = np.clip(v2n @ v1n, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# ============================================================================
# Per-subject driver
# ============================================================================
def compute_subject(sub: str, coef_root: Path, output_dir: Path,
                    analyses: list[str], roi: str, voxels: int,
                    phase: str, z_thresh: float,
                    n_perm: int) -> dict | None:
    # step 1: |·|-mean per analysis
    vectors: dict[str, np.ndarray] = {}
    for ana in analyses:
        paths = find_coef_files(coef_root, sub, ana, roi, voxels, phase)
        if not paths:
            print(f"  [skip] sub-{sub} {ana}: no files in "
                  f"{coef_dir_for(coef_root, sub, ana, roi)}")
            return None
        out_p = absmean_path(output_dir, sub, ana, roi, voxels, phase)
        v = compute_abs_mean(paths, out_p)
        if v is None or v.size == 0:
            return None
        vectors[ana] = v

    # step 1b: one threshold per analysis (cached so each ana is fit once)
    thresholds = {ana: threshold_value(v, z_thresh)
                  for ana, v in vectors.items()}

    # step 2: pairwise
    n = len(analyses)
    angle_mat   = np.full((n, n), np.nan)
    null_mean   = np.full((n, n), np.nan)
    effect_mat  = np.full((n, n), np.nan)
    overlap_mat = np.full((n, n), np.nan)
    n_sel_mat   = np.full((n, n), np.nan)

    rng = np.random.default_rng()
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = vectors[analyses[i]], vectors[analyses[j]]
            thr1, thr2 = thresholds[analyses[i]], thresholds[analyses[j]]

            v1s, v2s, pct_overlap, n_sel = threshold_by_reference(
                v1, v2, thr1, thr2)

            overlap_mat[i, j] = overlap_mat[j, i] = pct_overlap
            n_sel_mat[i, j]   = n_sel_mat[j, i]   = n_sel
            if v1s.size < 5:
                continue

            obs  = angle_between(v1s, v2s)
            null = null_angles(v1s, v2s, n_perm, rng)
            nm, ns = float(np.mean(null)), float(np.std(null))

            angle_mat[i, j] = angle_mat[j, i] = obs
            null_mean[i, j] = null_mean[j, i] = nm
            if ns > 0:
                effect_mat[i, j] = effect_mat[j, i] = (obs - nm) / ns

    return dict(
        subject=sub,
        thresholds=thresholds,
        angle=angle_mat,
        null_mean=null_mean,
        effect_size=effect_mat,
        overlap=overlap_mat,
        n_selected=n_sel_mat,
    )


# ============================================================================
# Group statistics + plotting
# ============================================================================
def paired_t_pvalues(angle_arr: np.ndarray, null_arr: np.ndarray) -> np.ndarray:
    n = angle_arr.shape[1]
    pmat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = angle_arr[:, i, j], null_arr[:, i, j]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() >= 5:
                _, p = ttest_rel(a[valid], b[valid], alternative="greater")
                pmat[i, j] = pmat[j, i] = float(p)
    return pmat


def plot_heatmap(mat: np.ndarray, labels: list[str], title: str,
                 cbar_label: str, fmt: str, out_path: Path,
                 vmin: float, vmax: float, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, xticklabels=labels, yticklabels=labels,
                annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax,
                cbar_kws={"label": cbar_label}, ax=ax, square=True)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>16}: {v}")
    print("-" * 64)

    suffix = (f"{args.roi}_top{args.voxels:04d}_{args.phase}"
              f"_z{args.z_thresh:g}_halfnorm")
    cache_path = args.output_dir / f"angles_{suffix}.pkl"

    if cache_path.exists() and not args.force:
        print(f"Loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        analyses = cache["analyses"]
        results  = cache["results"]
        if analyses != args.analyses:
            print(f"  [warn] cache analyses {analyses} != args.analyses "
                  f"{args.analyses}; using cache. Pass --force to recompute.")
    else:
        analyses = args.analyses
        results = []
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            futs = {pool.submit(compute_subject, s, args.coef_dir,
                                args.output_dir, analyses,
                                args.roi, args.voxels, args.phase,
                                args.z_thresh,
                                args.n_perm): s
                    for s in args.subjects}
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc="subjects"):
                r = fut.result()
                if r is not None:
                    results.append(r)
        results.sort(key=lambda r: r["subject"])
        with open(cache_path, "wb") as f:
            pickle.dump({"analyses": analyses, "results": results}, f)
        print(f"Saved cache: {cache_path}")

    if not results:
        print("No subject results; nothing to do.")
        return

    angle_arr   = np.stack([r["angle"]       for r in results])
    null_arr    = np.stack([r["null_mean"]   for r in results])
    effect_arr  = np.stack([r["effect_size"] for r in results])
    overlap_arr = np.stack([r["overlap"]     for r in results])
    n_sel_arr   = np.stack([r["n_selected"]  for r in results])

    # ---- per-subject CSV ----
    rows = []
    for r in results:
        for i, a in enumerate(analyses):
            for j in range(i + 1, len(analyses)):
                rows.append(dict(
                    subject=r["subject"], A=a, B=analyses[j],
                    threshold_A=r["thresholds"][a],
                    threshold_B=r["thresholds"][analyses[j]],
                    angle_deg=r["angle"][i, j],
                    null_mean_deg=r["null_mean"][i, j],
                    effect_size=r["effect_size"][i, j],
                    overlap_pct=r["overlap"][i, j],
                    n_selected=r["n_selected"][i, j],
                ))
    csv_path = args.output_dir / f"angles_{suffix}_per_subject.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved per-subject CSV: {csv_path}")

    mean_angle = np.nanmean(angle_arr, axis=0)
    pvalues = paired_t_pvalues(angle_arr, null_arr)

    plot_heatmap(
        mean_angle, analyses,
        title=f"Mean observed angle\n{suffix} (n={len(results)})",
        cbar_label="angle (deg)", fmt=".2f",
        vmin=float(np.nanmin(mean_angle)),
        vmax=float(np.nanmax(mean_angle)),
        out_path=args.output_dir / f"heatmap_angles_{suffix}.pdf",
        cmap="viridis",
    )
    plot_heatmap(
        pvalues, analyses,
        title=f"Paired t-test  p(angle > null mean)\n{suffix} (n={len(results)})",
        cbar_label="p-value", fmt=".3f",
        vmin=0, vmax=1,
        out_path=args.output_dir / f"heatmap_pvalues_{suffix}.pdf",
        cmap="coolwarm",
    )

    print(f"\nGroup summary (n={len(results)} subjects, "
          f"threshold = halfnorm:")
    for i, a in enumerate(analyses):
        for j in range(i + 1, len(analyses)):
            ang = angle_arr[:, i, j];
            nul = null_arr[:, i, j]
            eff = effect_arr[:, i, j];
            ovl = overlap_arr[:, i, j]
            nse = n_sel_arr[:, i, j]
            ang_v = ang[~np.isnan(ang)]
            nul_v = nul[~np.isnan(nul)]
            eff_v = eff[~np.isnan(eff)]
            ovl_v = ovl[~np.isnan(ovl)]
            nse_v = nse[~np.isnan(nse)]
            if ang_v.size == 0:
                continue
            print(f"  {a:>10} vs {analyses[j]:<10}  "
                  f"obs {ang_v.mean():.2f}±{ang_v.std():.2f}° "
                  f"[{ang_v.min():.2f}–{ang_v.max():.2f}]   "
                  f"null {nul_v.mean():.2f}±{nul_v.std():.2f}° "
                  f"[{nul_v.min():.2f}–{nul_v.max():.2f}]   "
                  f"overlap {ovl_v.mean():.2f}%±{ovl_v.std():.2f}% "
                  f"[{ovl_v.min():.2f}–{ovl_v.max():.2f}]   "
                  f"p {pvalues[i, j]:.4g}")
    print("Done.")

if __name__ == "__main__":
    main()