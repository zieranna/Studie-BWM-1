#!/usr/bin/env python3
"""
summarize_grid.py
=================

Collects the GLA3 results of the decoding grid(s) (after gla3_grid.sh) into
one workbook, a compact terminal table and one figure.

Reads, for every <grid>/<input>/group_stats_<run>/:
    nested runs     overview_binding_<run>.xlsx  (group_stats, voxel_choice)
                    -> mean, SEM, d, p, p FDR across ROIs, BF10, chosen voxels
    all-count runs  stats_binding_<ROI>_<run>.csv  (written by gla3_grid.sh)
                    -> one row per voxel count (FDR across counts within the ROI)
                       and a summary per ROI: mean over counts, best count,
                       number of counts with p FDR < .05

Output (next to the first grid folder, or --out):
    grid_summary_<date>.xlsx   sheets nested, allcounts_summary, allcounts_curve, compact
    grid_summary_<date>.png    heatmap: accuracy - 50 %, stars = p FDR

Usage:
    python3 summarize_grid.py                      # the default grid folders below
    python3 summarize_grid.py --grids <folder> [<folder> ...]
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

CP = Path("/home/annaz/BindingWM/Studie-BWM-1_CP_revision1")
GRIDS = [CP / "decoding_grid_zrun", CP / "decoding_grid_zrun_step50"]
INPUTS = {"fir_orig": "FIR original", "lss17": "LSS FIR 17", "lss25": "LSS FIR 25", "hrf": "LSS HRF"}
VARIANTS = {"nested-pooled_C1": "nested, C = 1", "nested-pooled": "nested, C tuned",
            "allcounts_C1": "all counts, C = 1", "allcounts": "all counts, C tuned"}
ROI_ORDER = ["V1", "V2", "V3", "V4", "early_visual", "pari", "dlpfc", "dlpfc_noSFL",
             "visual_all", "visual_locMT", "MT_complex"]


def star(p):
    if p is None or pd.isna(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def variant_label(grid: Path, tag: str) -> str:
    lab = VARIANTS.get(tag, tag)
    if "step50" in grid.name and tag.startswith("nested"):
        lab = lab.replace("nested", "nested 50-steps")
    return lab


def read_nested(grid, inp, folder, tag):
    f = folder / f"overview_binding_{tag}.xlsx"
    if not f.exists():
        return []
    gs = pd.read_excel(f, sheet_name="group_stats")
    try:
        vc = pd.read_excel(f, sheet_name="voxel_choice")
    except ValueError:
        vc = pd.DataFrame()
    rows = []
    for _, r in gs.iterrows():
        row = {"grid": grid.name, "input": inp, "run": tag, "variant": variant_label(grid, tag),
               "phase": r.get("phase"), "ROI": r["area"], "n": r.get("n"),
               "mean": r["mean"], "sem": r.get("sem"), "cohens_d": r.get("cohens_d"),
               "p_raw": r.get("p_raw"), "p_fdr": r.get("p_fdr_rois", r.get("p_fdr")),
               "BF10": r.get("bayes_H1")}
        if len(vc) and "ROI" in vc:
            v = vc[(vc["ROI"] == r["area"]) & (vc.get("phase", r.get("phase")) == r.get("phase"))]
            if len(v):
                shares = v.drop(columns=[c for c in ("phase", "ROI") if c in v]).iloc[0]
                shares = shares[pd.to_numeric(shares.index, errors="coerce").notna()].astype(float)
                if shares.sum() > 0:
                    counts = np.asarray(shares.index, dtype=float)
                    row["voxels_modal"] = int(counts[shares.values.argmax()])
                    row["voxels_mean"] = round(float((counts * shares.values).sum() / shares.sum()))
        rows.append(row)
    return rows


def read_allcounts(grid, inp, folder, tag):
    rows = []
    for f in sorted(folder.glob(f"stats_binding_*_{tag}.csv")):
        st = pd.read_csv(f)
        for _, r in st.iterrows():
            rows.append({"grid": grid.name, "input": inp, "run": tag,
                         "variant": variant_label(grid, tag), "phase": r.get("phase"),
                         "ROI": r["area"], "voxels": int(r["condition"]), "n": r.get("n"),
                         "mean": r["mean"], "sem": r.get("sem"), "cohens_d": r.get("cohens_d"),
                         "p_raw": r.get("p_raw"), "p_fdr": r.get("p_fdr"),
                         "BF10": r.get("bayes_H1")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Summary of all GLA3 grid results.")
    ap.add_argument("--grids", nargs="+", type=Path, default=GRIDS)
    ap.add_argument("--out", type=Path, default=None, help="file stem (default next to the first grid)")
    a = ap.parse_args()

    nested, curve = [], []
    for grid in a.grids:
        if not grid.exists():
            print(f"[skip] {grid} does not exist")
            continue
        for d in sorted(p for p in grid.iterdir() if p.is_dir()):
            inp = d.name
            for gs in sorted(d.glob("group_stats_*")):
                tag = gs.name[len("group_stats_"):]
                if tag.startswith("nested"):
                    nested += read_nested(grid, inp, gs, tag)
                elif tag.startswith("allcounts"):
                    curve += read_allcounts(grid, inp, gs, tag)
    if not nested and not curve:
        raise SystemExit("No GLA3 results found. Run gla3_grid.sh first.")

    roi_rank = {r: i for i, r in enumerate(ROI_ORDER)}
    inp_rank = {k: i for i, k in enumerate(INPUTS)}
    var_rank = {v: i for i, v in enumerate(VARIANTS)}

    def order(df):
        return df.sort_values(
            ["input", "grid", "run", "ROI"] + (["voxels"] if "voxels" in df else []),
            key=lambda s: s.map(inp_rank).fillna(99) if s.name == "input"
            else s.map(var_rank).fillna(99) if s.name == "run"
            else s.map(roi_rank).fillna(99) if s.name == "ROI" else s).reset_index(drop=True)

    nest = order(pd.DataFrame(nested)) if nested else pd.DataFrame()
    cur = order(pd.DataFrame(curve)) if curve else pd.DataFrame()
    summ = pd.DataFrame()
    if len(cur):
        rows = []
        for (g, i, t, v, roi), x in cur.groupby(["grid", "input", "run", "variant", "ROI"], sort=False):
            b = x.loc[x["mean"].idxmax()]
            rows.append({"grid": g, "input": i, "run": t, "variant": v, "ROI": roi,
                         "n_counts": len(x), "mean_over_counts": x["mean"].mean(),
                         "best_count": int(b["voxels"]), "best_mean": b["mean"],
                         "best_p_fdr": b["p_fdr"], "counts_fdr_sig": int((x["p_fdr"] < .05).sum()),
                         "min_p_fdr": x["p_fdr"].min()})
        summ = order(pd.DataFrame(rows))

    # ---- compact table: one row per input x variant, one column per ROI ------------
    comp = []
    for _, r in nest.iterrows():
        comp.append({"input": INPUTS.get(r["input"], r["input"]), "variant": r["variant"],
                     "grid": r["grid"], "ROI": r["ROI"], "value": r["mean"] - .5,
                     "txt": f"{r['mean']:.3f}{star(r['p_fdr'])}"})
    for _, r in summ.iterrows():
        comp.append({"input": INPUTS.get(r["input"], r["input"]), "variant": r["variant"],
                     "grid": r["grid"], "ROI": r["ROI"], "value": r["mean_over_counts"] - .5,
                     "txt": f"{r['mean_over_counts']:.3f}~ ({r['counts_fdr_sig']}/{r['n_counts']})"})
    comp = pd.DataFrame(comp)
    rois = [r for r in ROI_ORDER if r in set(comp["ROI"])] + \
           sorted(set(comp["ROI"]) - set(ROI_ORDER))
    comp["row"] = comp["input"] + " | " + comp["variant"]
    rows_order = list(dict.fromkeys(
        comp.assign(i=comp["input"].map({v: k for k, v in enumerate(INPUTS.values())}).fillna(99),
                    v=comp["variant"].map(lambda s: next((k for k, x in enumerate(VARIANTS.values()) if x in s), 99)),
                    s=comp["variant"].str.contains("50-steps").astype(int))
        .sort_values(["i", "s", "v"])["row"]))
    tab = comp.pivot_table(index="row", columns="ROI", values="txt", aggfunc="first") \
              .reindex(index=rows_order, columns=rois)
    pd.set_option("display.width", 250)
    print("\nBinding decoding, GLA3 group results (chance 0.5)")
    print("nested: mean accuracy, * p FDR < .05, ** < .01, *** < .001 (FDR across ROIs)")
    print("all counts: mean over voxel counts '~', (k/n) = counts with p FDR < .05 (FDR across counts)\n")
    print(tab.fillna("").to_string())

    stem = a.out or (a.grids[0].parent / f"grid_summary_{time.strftime('%Y%m%d')}")
    with pd.ExcelWriter(f"{stem}.xlsx") as xl:
        tab.reset_index().to_excel(xl, sheet_name="compact", index=False)
        if len(nest):
            nest.round(5).to_excel(xl, sheet_name="nested", index=False)
        if len(summ):
            summ.round(5).to_excel(xl, sheet_name="allcounts_summary", index=False)
        if len(cur):
            cur.round(5).to_excel(xl, sheet_name="allcounts_curve", index=False)

    # ---- figure ------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    val = comp.pivot_table(index="row", columns="ROI", values="value", aggfunc="first") \
              .reindex(index=rows_order, columns=rois)
    lab = tab
    fig, ax = plt.subplots(figsize=(1.15 * len(rois) + 4.5, 0.42 * len(rows_order) + 1.8))
    lim = max(0.02, float(np.nanmax(np.abs(val.values))))
    im = ax.imshow(val.values * 100, cmap="RdBu_r", vmin=-lim * 100, vmax=lim * 100, aspect="auto")
    for i in range(val.shape[0]):
        for j in range(val.shape[1]):
            t = lab.iat[i, j]
            if isinstance(t, str):
                ax.text(j, i, t.replace(" (", "\n("), ha="center", va="center",
                        fontsize=7 if "(" in t else 7.5, linespacing=0.9)
    ax.set_xticks(range(len(rois)), rois, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows_order)), rows_order, fontsize=8.5)
    for k in range(1, len(rows_order)):
        if rows_order[k].split(" | ")[0] != rows_order[k - 1].split(" | ")[0]:
            ax.axhline(k - 0.5, color="black", lw=1)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("accuracy - 50 % (percentage points)")
    ax.set_title("Binding decoding, GLA3 group means. Nested: * p FDR < .05 (across ROIs).\n"
                 "All counts: ~ mean over voxel counts, (k/n) counts with p FDR < .05 (across counts)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200)
    print(f"\nSaved: {stem}.xlsx, {stem}.png")


if __name__ == "__main__":
    main()