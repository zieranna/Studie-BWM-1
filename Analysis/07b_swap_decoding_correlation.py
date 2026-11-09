#!/usr/bin/env python3
"""
08_swap_decoding_correlation.py
===============================

Spearman correlation between swap errors and binding decoding.

Reads the NESTED output of 07_decoding.py (sheet 'folds', ideally with the
'signed_distance' column added in the revised nested_fold_job) and a
behavioral trial table, then runs three analyses:

  1. Within-subject (primary). Per subject and ROI, Spearman rho across
     units between binding evidence and mean swap probability.
       --unit block : 16 units (held-out blocks), permutation within session
       --unit set   : 9 units (binding sets), free permutation
     Optional partial Spearman: ranks of both variables are residualized
     on ranks of covariates (color / motion decoding of the same block,
     guess rate, selected voxel count, session).
     Group level: mean Fisher-z across subjects, tested against a null
     built from the same per-subject permutations. A max-statistic over
     ROIs gives family-wise corrected p-values.
  2. Specificity. Per subject, z(swap, binding) - z(swap, color/motion),
     tested across subjects.
  3. Between-subject (secondary). Subject swap rate vs subject binding
     evidence; permutation p, bootstrap CI, Holm across ROIs.
Plus split-half reliability of both block-level measures, which bounds
the correlation that can be observed (Spearman, 1904).

Behavioral input (CSV, one row per trial)
-----------------------------------------
Required: subject, session, block
          pair   (paper pair label, e.g. '4'; needed for --unit set)
--behavior-mode mixture (continuous report):
          response, target, and one or more non-target columns
          (--nontarget-cols). Units via --units / --period.
          A three-component mixture model (Bays, Catalao & Husain, 2009)
          is fit once per subject by EM; each trial gets posterior
          probabilities of target / swap / guess.
--behavior-mode precomputed:
          a column (--swap-col) with a per-trial swap indicator or
          probability (e.g. false alarm to a recombined probe).

Usage
-----
python 08_swap_decoding_correlation.py \
    --decoding-dir /path/to/decoding_out --behavior beh.csv \
    --subjects 01 02 03 --phase del --unit block \
    --covariates color motion --out swap_vs_binding_del.xlsx
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import i0e

N_BLOCKS_PER_SESSION = 4

# Binding sets in the same order as ANALYSES["binding"] in 07_decoding.py,
# so set k here is pair == k in the 'folds' sheet.
BINDING_SETS = [
    ("4", "3"), ("6", "5"), ("2", "1"),
    ("10", "9"), ("12", "11"), ("8", "7"),
    ("16", "15"), ("18", "17"), ("14", "13"),
]
LABEL_TO_SET = {lab: k + 1 for k, pair in enumerate(BINDING_SETS)
                for lab in pair}


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--decoding-dir", required=True, type=Path,
                   help="--output-dir of 07_decoding.py.")
    p.add_argument("--behavior", required=True, type=Path,
                   help="Trial-level behavioral CSV.")
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--phase", default="del", choices=["enc", "del", "res"])
    p.add_argument("--rois", nargs="+", default=None,
                   help="Default: all ROIs present in the binding output.")
    p.add_argument("--tag", default="nested-pooled",
                   help="Selection tag in the 07 file name.")
    p.add_argument("--metric", default="signed_distance",
                   choices=["signed_distance", "test_accuracy"])
    p.add_argument("--unit", default="block", choices=["block", "set"])
    p.add_argument("--covariates", nargs="*", default=["color", "motion"],
                   choices=["color", "motion", "guess", "voxels", "session"],
                   help="Partial-Spearman covariates (block unit only, "
                        "except 'guess'). Empty = plain Spearman.")
    p.add_argument("--alternative", default="less",
                   choices=["less", "greater", "two-sided"],
                   help="'less' = more binding information, fewer swaps.")
    # behavior
    p.add_argument("--behavior-mode", default="mixture",
                   choices=["mixture", "precomputed"])
    p.add_argument("--response-col", default="response")
    p.add_argument("--target-col", default="target")
    p.add_argument("--nontarget-cols", nargs="+", default=["nontarget"])
    p.add_argument("--swap-col", default="swap")
    p.add_argument("--units", default="deg", choices=["deg", "rad"])
    p.add_argument("--period", type=float, default=None,
                   help="Period of the feature space (default 360 deg / "
                        "2*pi rad; use 180 for orientation).")
    # inference
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--n-splits", type=int, default=200,
                   help="Random splits for split-half reliability.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


# ============================================================================
# Mixture model (Bays, Catalao & Husain, 2009), EM fit
# ============================================================================
def wrap(x):
    return np.angle(np.exp(1j * x))


def vm_pdf(x, k):
    """Von Mises density around 0, numerically stable for large kappa."""
    return np.exp(k * (np.cos(x) - 1.0)) / (2 * np.pi * i0e(k))


def a1inv(r):
    """Inverse of A1 = I1/I0 (Fisher, 1993, approximation)."""
    r = min(r, 1 - 1e-9)
    if r < 0.53:
        return 2 * r + r ** 3 + 5 * r ** 5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)


def _e_step(err, nt, K, pT, pN, pU):
    n_nt = np.sum(~np.isnan(nt), axis=1)
    share = np.divide(pN, n_nt, out=np.zeros(len(err)), where=n_nt > 0)
    Lt = pT * vm_pdf(err, K)
    Ln = np.where(np.isnan(nt), 0.0, share[:, None] * vm_pdf(np.nan_to_num(nt), K))
    Lu = np.full(len(err), pU / (2 * np.pi))
    L = Lt + Ln.sum(axis=1) + Lu
    return Lt / L, Ln / L[:, None], Lu / L, np.sum(np.log(L))


def fit_mixture(err, nt, max_iter=5000, tol=1e-7):
    """err: (n,) response - target; nt: (n, m) response - non-targets
    (radians, wrapped; NaN where a trial has fewer non-targets)."""
    best = None
    for K0 in (1.0, 10.0, 100.0):
        for pT0, pN0 in ((0.9, 0.05), (0.6, 0.2), (0.3, 0.4), (0.5, 0.05)):
            K, pT, pN, pU = K0, pT0, pN0, 1 - pT0 - pN0
            ll_old = -np.inf
            for _ in range(max_iter):
                Wt, Wn, Wu, ll = _e_step(err, nt, K, pT, pN, pU)
                pT, pN, pU = Wt.mean(), Wn.sum(axis=1).mean(), Wu.mean()
                S = np.sum(Wt * np.sin(err)) + np.nansum(Wn * np.sin(nt))
                C = np.sum(Wt * np.cos(err)) + np.nansum(Wn * np.cos(nt))
                W = Wt.sum() + Wn.sum()
                K = a1inv(np.hypot(S, C) / W) if W > 0 else 1e-3
                if abs(ll - ll_old) < tol:
                    break
                ll_old = ll
            if best is None or ll > best["ll"]:
                best = dict(K=K, pT=pT, pN=pN, pU=pU, ll=ll)
    Wt, Wn, Wu, _ = _e_step(err, nt, best["K"], best["pT"], best["pN"],
                            best["pU"])
    return best, Wt, Wn.sum(axis=1), Wu


# ============================================================================
# Loading
# ============================================================================
def load_behavior(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trial table with p_swap / p_guess per trial, plus subject params."""
    beh = pd.read_csv(args.behavior, dtype={"subject": str, "pair": str})
    beh["subject"] = beh["subject"].str.replace("sub-", "", regex=False)
    # tolerate lost zero-padding (e.g. '1' vs '01')
    key = {s.lstrip("0") or "0": s for s in args.subjects}
    beh["subject"] = beh["subject"].map(lambda s: key.get(s.lstrip("0") or "0", s))
    beh = beh[beh["subject"].isin(args.subjects)].copy()
    if beh.empty:
        raise ValueError("No behavioral rows match --subjects.")
    beh["fold"] = ((beh["session"].astype(int) - 1) * N_BLOCKS_PER_SESSION
                   + beh["block"].astype(int))
    if "pair" in beh:
        beh["set"] = beh["pair"].map(LABEL_TO_SET)

    params = []
    if args.behavior_mode == "precomputed":
        beh["p_swap"] = beh[args.swap_col].astype(float)
        beh["p_guess"] = np.nan
        for s, d in beh.groupby("subject"):
            params.append(dict(subject=s, pN=d["p_swap"].mean(),
                               n_trials=len(d)))
        return beh, pd.DataFrame(params)

    period = args.period or (360.0 if args.units == "deg" else 2 * np.pi)
    scale = 2 * np.pi / period
    beh["p_swap"] = np.nan
    beh["p_guess"] = np.nan
    for s, d in beh.groupby("subject"):
        d = d.dropna(subset=[args.response_col, args.target_col])
        resp = d[args.response_col].to_numpy(float)
        err = wrap(scale * (resp - d[args.target_col].to_numpy(float)))
        nt = wrap(scale * (resp[:, None]
                           - d[args.nontarget_cols].to_numpy(float)))
        fit, _, w_swap, w_guess = fit_mixture(err, nt)
        beh.loc[d.index, "p_swap"] = w_swap
        beh.loc[d.index, "p_guess"] = w_guess
        params.append(dict(subject=s, **fit, n_trials=len(d)))
        print(f"  sub-{s}: K={fit['K']:.2f} pT={fit['pT']:.3f} "
              f"pSwap={fit['pN']:.3f} pGuess={fit['pU']:.3f} (n={len(d)})")
    return beh, pd.DataFrame(params)


def find_nested_file(root: Path, sub: str, analysis: str, phase: str,
                     tag: str) -> Path | None:
    d = root / f"sub-{sub}" / analysis
    pat = re.compile(rf"decoding_{analysis}_{phase}_{re.escape(tag)}_(\d{{8}})\.xlsx$")
    hits = sorted((m.group(1), f) for f in d.glob("*.xlsx")
                  if (m := pat.match(f.name)))
    return hits[-1][1] if hits else None


def load_folds(args, sub: str, analysis: str) -> pd.DataFrame | None:
    f = find_nested_file(args.decoding_dir, sub, analysis, args.phase,
                         args.tag)
    if f is None:
        return None
    df = pd.read_excel(f, sheet_name="folds")
    metric = args.metric
    if metric not in df:
        warnings.warn(f"{f.name}: no '{metric}' column (re-run 07 with the "
                      "revised nested_fold_job); using test_accuracy.")
        metric = "test_accuracy"
    df = df.rename(columns={metric: "metric"})
    df["subject"] = sub
    return df[["subject", "ROI", "pair", "fold", "voxels", "metric"]]


# ============================================================================
# Statistics helpers
# ============================================================================
def _rank_resid(v, Z):
    r = stats.rankdata(v)
    if Z is None:
        return r - r.mean()
    return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]


def partial_ranks(x, y, covs):
    """Residualized ranks; Pearson of the outputs = (partial) Spearman."""
    Z = None
    if covs is not None and covs.shape[1] > 0:
        Z = np.column_stack([np.ones(len(x))]
                            + [stats.rankdata(c) for c in covs.T])
    return _rank_resid(x, Z), _rank_resid(y, Z)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return np.nan if den == 0 else float(np.sum(a * b) / den)


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def perm_p(obs, null, alternative):
    if alternative == "less":
        k = np.sum(null <= obs)
    elif alternative == "greater":
        k = np.sum(null >= obs)
    else:
        k = np.sum(np.abs(null) >= abs(obs))
    return (1 + k) / (1 + len(null))


def perm_indices(groups, n_perm, rng):
    """(n_perm, n) permutations that only shuffle within each group."""
    n = len(groups)
    out = np.tile(np.arange(n), (n_perm, 1))
    for g in np.unique(groups):
        pos = np.flatnonzero(groups == g)
        for p in range(n_perm):
            out[p, pos] = rng.permutation(pos)
    return out


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(p) - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


# ============================================================================
# Unit table
# ============================================================================
def build_unit_table(args, beh, rois) -> pd.DataFrame:
    unit = "fold" if args.unit == "block" else "set"
    rows = []
    for sub in args.subjects:
        b = load_folds(args, sub, "binding")
        if b is None:
            print(f"  [warn] sub-{sub}: no binding output, skipped")
            continue
        b = b[b["ROI"].isin(rois)]
        b = b.rename(columns={"pair": "set"})
        nb = (b.groupby(["ROI", unit])
               .agg(binding=("metric", "mean"), voxels=("voxels", "mean"))
               .reset_index())
        if args.unit == "block":
            for an in ("color", "motion"):
                f = load_folds(args, sub, an)
                if f is None:
                    nb[an] = np.nan
                    continue
                nb = nb.merge(f.groupby(["ROI", "fold"])["metric"].mean()
                               .rename(an).reset_index(),
                              on=["ROI", "fold"], how="left")
        bs = beh[beh["subject"] == sub].groupby(unit).agg(
            swap=("p_swap", "mean"), guess=("p_guess", "mean"),
            n_trials=("p_swap", "size")).reset_index()
        t = nb.merge(bs, on=unit, how="inner")
        t.insert(0, "subject", sub)
        rows.append(t)
    tab = pd.concat(rows, ignore_index=True)
    if args.unit == "block":
        tab["session"] = (tab["fold"] - 1) // N_BLOCKS_PER_SESSION + 1
    tab = tab.rename(columns={unit: "unit"})
    return tab


# ============================================================================
# 1. Within-subject
# ============================================================================
def within_subject(args, tab, rois, rng):
    covs = [c for c in args.covariates
            if args.unit == "block" or c == "guess"]
    subjects = sorted(tab["subject"].unique())
    perms = {}
    sub_rows, null_z = [], {r: [] for r in rois}

    for sub in subjects:
        d0 = tab[tab["subject"] == sub]
        units = np.sort(d0["unit"].unique())
        groups = (d0.drop_duplicates("unit").set_index("unit")
                    .loc[units, "session"].to_numpy()
                  if args.unit == "block" else np.ones(len(units)))
        perms[sub] = perm_indices(groups, args.n_perm, rng)
        for roi in rois:
            d = d0[d0["ROI"] == roi].set_index("unit").reindex(units)
            use = covs if all(d[c].notna().all() for c in covs) else []
            if d[["binding", "swap"]].isna().any().any():
                null_z[roi].append(np.full(args.n_perm, np.nan))
                continue
            C = d[use].to_numpy(float) if use else None
            rx, ry = partial_ranks(d["binding"].to_numpy(),
                                   d["swap"].to_numpy(), C)
            rho = pearson(rx, ry)
            null = np.array([pearson(rx, ry[pi]) for pi in perms[sub]])
            null_z[roi].append(fisher_z(null))
            raw = {}
            for an in ("binding", "color", "motion"):
                if an in d and d[an].notna().all():
                    raw[an] = stats.spearmanr(d[an], d["swap"])[0]
            sub_rows.append(dict(
                subject=sub, ROI=roi, n_units=len(units),
                covariates=",".join(use) or "none",
                rho=rho, z=fisher_z(rho),
                p_perm=perm_p(rho, null, args.alternative),
                rho_raw_binding=raw.get("binding"),
                rho_raw_color=raw.get("color"),
                rho_raw_motion=raw.get("motion")))

    per_sub = pd.DataFrame(sub_rows)

    # group level: mean z vs permutation null, max-statistic over ROIs
    obs = per_sub.groupby("ROI")["z"].mean().reindex(rois)
    null_mean = np.vstack([np.nanmean(np.vstack(null_z[r]), axis=0)
                           for r in rois])                # (rois, perms)
    if args.alternative == "less":
        fam = np.nanmin(null_mean, axis=0)
    elif args.alternative == "greater":
        fam = np.nanmax(null_mean, axis=0)
    else:
        fam = np.nanmax(np.abs(null_mean), axis=0)
    alt_scipy = args.alternative
    grp = []
    for i, roi in enumerate(rois):
        z = per_sub.loc[per_sub["ROI"] == roi, "z"].dropna().to_numpy()
        if len(z) == 0:
            continue
        row = dict(ROI=roi, n_subjects=len(z), mean_z=obs[roi],
                   mean_rho=np.tanh(obs[roi]),
                   p_perm=perm_p(obs[roi], null_mean[i], args.alternative),
                   p_perm_fwe=perm_p(obs[roi], fam, args.alternative))
        if len(z) >= 3:
            row["t"], row["p_t"] = stats.ttest_1samp(z, 0,
                                                     alternative=alt_scipy)
        if len(z) >= 6:
            row["p_wilcoxon"] = stats.wilcoxon(z, alternative=alt_scipy)[1]
        grp.append(row)
    return per_sub, pd.DataFrame(grp)


# ============================================================================
# 2. Specificity: binding vs feature correlations with swaps
# ============================================================================
def specificity(per_sub):
    rows = []
    for roi, d in per_sub.groupby("ROI"):
        for an in ("color", "motion"):
            col = f"rho_raw_{an}"
            dd = d[["rho_raw_binding", col]].dropna()
            if len(dd) < 3:
                continue
            diff = fisher_z(dd["rho_raw_binding"]) - fisher_z(dd[col])
            row = dict(ROI=roi, contrast=f"binding - {an}",
                       n_subjects=len(dd), mean_z_diff=diff.mean())
            row["t"], row["p_t"] = stats.ttest_1samp(diff, 0)
            if len(dd) >= 6:
                row["p_wilcoxon"] = stats.wilcoxon(diff)[1]
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# 3. Between-subject
# ============================================================================
def between_subject(args, tab, params, rois, rng):
    rows = []
    for roi in rois:
        d = (tab[tab["ROI"] == roi].groupby("subject")["binding"].mean()
             .rename("binding").reset_index()
             .merge(params[["subject", "pN"]], on="subject"))
        n = len(d)
        if n < 5:
            print(f"  [info] between-subject {roi}: n={n} < 5, skipped")
            continue
        x, y = d["binding"].to_numpy(), d["pN"].to_numpy()
        rho = stats.spearmanr(x, y)[0]
        null = np.array([stats.spearmanr(x, rng.permutation(y))[0]
                         for _ in range(args.n_perm)])
        boot = []
        for _ in range(args.n_boot):
            i = rng.integers(0, n, n)
            if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
                boot.append(stats.spearmanr(x[i], y[i])[0])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append(dict(ROI=roi, n_subjects=n, rho=rho,
                         p_perm=perm_p(rho, null, args.alternative),
                         ci95_low=lo, ci95_high=hi))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_perm"])
    return out


# ============================================================================
# Split-half reliability of the block-level measures
# ============================================================================
def reliability(args, beh, rois, rng):
    rows = []
    for sub in args.subjects:
        b = load_folds(args, sub, "binding")
        if b is None:
            continue
        bs = beh[beh["subject"] == sub]
        # behavior: split trials within block
        r_beh = []
        for _ in range(args.n_splits):
            h = bs.groupby("fold")["p_swap"].apply(
                lambda v: pd.Series(
                    [v.iloc[m].mean() for m in
                     np.array_split(rng.permutation(len(v)), 2)],
                    index=["h1", "h2"])).unstack()
            r_beh.append(stats.spearmanr(h["h1"], h["h2"])[0])
        rb = np.nanmean(r_beh)
        for roi in rois:
            d = b[b["ROI"] == roi].pivot_table(index="fold", columns="pair",
                                               values="metric")
            r_neu = []
            for _ in range(args.n_splits):
                cols = rng.permutation(d.columns.to_numpy())
                h1, h2 = np.array_split(cols, 2)
                r_neu.append(stats.spearmanr(d[h1].mean(axis=1),
                                             d[h2].mean(axis=1))[0])
            rn = np.nanmean(r_neu)
            sb = lambda r: 2 * r / (1 + r) if r > 0 else r  # SB only if r > 0
            rel_b, rel_n = sb(rb), sb(rn)
            ceil = (np.sqrt(rel_b * rel_n)
                    if rel_b > 0 and rel_n > 0 else np.nan)
            rows.append(dict(subject=sub, ROI=roi,
                             rel_swap_sb=rel_b, rel_binding_sb=rel_n,
                             max_observable_rho=ceil))
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    args.subjects = [s.replace("sub-", "") for s in args.subjects]
    rng = np.random.default_rng(args.seed)

    print("Behavior ...")
    beh, params = load_behavior(args)

    if args.rois is None:
        f0 = next((load_folds(args, s, "binding") for s in args.subjects
                   if find_nested_file(args.decoding_dir, s, "binding",
                                       args.phase, args.tag)), None)
        if f0 is None:
            raise FileNotFoundError("No nested binding output found.")
        rois = list(dict.fromkeys(f0["ROI"]))
    else:
        rois = args.rois

    print("Unit table ...")
    tab = build_unit_table(args, beh, rois)

    print(f"Within-subject ({args.unit}, {args.n_perm} permutations) ...")
    per_sub, grp = within_subject(args, tab, rois, rng)
    spec = specificity(per_sub) if args.unit == "block" else pd.DataFrame()

    print("Between-subject ...")
    betw = between_subject(args, tab, params, rois, rng)

    rel = pd.DataFrame()
    if args.unit == "block":
        print("Reliability ...")
        rel = reliability(args, beh, rois, rng)

    settings = pd.DataFrame([{k: str(v) for k, v in vars(args).items()}])
    with pd.ExcelWriter(args.out) as xl:
        grp.round(4).to_excel(xl, sheet_name="within_group", index=False)
        per_sub.round(4).to_excel(xl, sheet_name="within_subject", index=False)
        if not spec.empty:
            spec.round(4).to_excel(xl, sheet_name="specificity", index=False)
        betw.round(4).to_excel(xl, sheet_name="between_subject", index=False)
        if not rel.empty:
            rel.round(4).to_excel(xl, sheet_name="reliability", index=False)
        params.round(4).to_excel(xl, sheet_name="mixture_params", index=False)
        tab.to_excel(xl, sheet_name="unit_table", index=False)
        settings.to_excel(xl, sheet_name="settings", index=False)

    print("\nWithin-subject, group level:")
    print(grp.round(4).to_string(index=False))
    if not betw.empty:
        print("\nBetween-subject:")
        print(betw.round(4).to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()