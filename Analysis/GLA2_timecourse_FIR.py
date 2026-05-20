#!/usr/bin/env python3
"""
FIR timecourse within an anatomical ROI.

For each subject, the ROI mask is applied to every FIR delay map (1..N),
the within-ROI spatial mean is taken, and the resulting timecourse is
z-scored across delays. Across subjects, the script plots the group mean
with SEM and writes the per-subject z-scored values to CSV.

Expected layout
---------------
    <firmodel-dir>/sub-XX/model/fir/binding/full/contrast/
        sub-XX_delay1_t.nii.gz  ...  sub-XX_delay25_t.nii.gz
    <roi-dir>/sub-XX/{roi}_resampled.nii.gz

Outputs (written to --output-dir)
---------------------------------
    firtimecourse_<roi>.pdf            -- group mean ± SEM figure
    firtimecourse_<roi>_data.csv       -- per-subject z-scored timecourses

Multiple ROIs can be passed with --rois; each gets its own CSV and all
are overlaid on the same figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from tqdm import tqdm

# Delay-phase window from the FIR analysis (1-indexed TRs). For shading only.
DELAY_TRS = (9, 12)


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ROI-mean FIR timecourse, group mean ± SEM.",
    )
    p.add_argument("--firmodel-dir", type=Path,
                   help="Root containing sub-XX/model/fir/binding/full/contrast/.")
    p.add_argument("--roi-dir", type=Path,
                   help="Root containing sub-XX/{roi}_resampled.nii.gz.")
    p.add_argument("--rois", nargs="+", default=["early_visual"],
                   help="ROI base name(s) (default: early_visual). "
                        "Each becomes one line on the figure.")
    p.add_argument("--subjects", nargs="+",
                   default=[f"{i:02d}" for i in range(1, 21)],
                   help="Subject labels without 'sub-' (default 01..20).")
    p.add_argument("--n-delays", type=int, default=25)
    p.add_argument("--tr", type=float, default=1.2)
    p.add_argument("--output-dir", type=Path, default=Path("../figures_and_tables"),
                   help="Where the figure and CSV(s) are written.")
    return p.parse_args()


# ============================================================================
# Path helpers
# ============================================================================
def roi_path(roi_dir: Path, sub: str, roi: str) -> Path:
    return roi_dir / f"sub-{sub}" / f"{roi}_resampled.nii.gz"


def delay_path(firmodel_dir: Path, sub: str, delay: int) -> Path:
    return (firmodel_dir / f"sub-{sub}" / "model" / "fir" / "binding" /
            "full" / "contrast" / f"sub-{sub}_delay{delay}_t.nii.gz")


# ============================================================================
# Per-subject timecourse
# ============================================================================
def subject_timecourse(sub: str, roi: str,
                       firmodel_dir: Path, roi_dir: Path,
                       n_delays: int) -> np.ndarray | None:
    rp = roi_path(roi_dir, sub, roi)
    if not rp.exists():
        print(f"  [skip] no ROI for sub-{sub}: {rp}")
        return None

    masker = NiftiMasker(mask_img=str(rp), standardize=False)
    masker.fit()                       # fit once, reuse across delays

    tc = np.empty(n_delays, dtype=float)
    for d in range(1, n_delays + 1):
        dp = delay_path(firmodel_dir, sub, d)
        if not dp.exists():
            print(f"  [skip] sub-{sub} missing delay {d}: {dp}")
            return None
        data = masker.transform(str(dp))           # (1, n_voxels)
        tc[d - 1] = float(data.mean())

    # within-subject z-score across delays
    return (tc - tc.mean()) / tc.std()


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>14}: {v}")
    print("-" * 64)

    # ---- collect per-subject timecourses for each ROI -------------------
    all_data: dict[str, pd.DataFrame] = {}
    for roi in args.rois:
        print(f"\n== ROI: {roi} ==")
        rows = {}
        for sub in tqdm(args.subjects, desc="subjects"):
            tc = subject_timecourse(sub, roi,
                                    args.firmodel_dir, args.roi_dir,
                                    args.n_delays)
            if tc is not None:
                rows[sub] = tc

        if not rows:
            print(f"  no data for ROI {roi}; skipping.")
            continue

        df = pd.DataFrame.from_dict(
            rows, orient="index",
            columns=[f"delay_{i}" for i in range(1, args.n_delays + 1)],
        )
        df.index.name = "subject"
        csv_path = args.output_dir / f"firtimecourse_{roi}_data.csv"
        df.to_csv(csv_path)
        print(f"  wrote {csv_path}  ({df.shape[0]} subjects × {df.shape[1]} delays)")
        all_data[roi] = df

    if not all_data:
        print("\nNothing to plot.")
        return

    # ---- group plot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    time = np.arange(args.n_delays) * args.tr
    palette = plt.cm.tab10.colors

    for i, (roi, df) in enumerate(all_data.items()):
        mat = df.to_numpy()
        mean = mat.mean(axis=0)
        sem  = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        c = palette[i % len(palette)]
        ax.plot(time, mean, color=c, linewidth=2,
                label=f"{roi} (n={mat.shape[0]})")
        ax.fill_between(time, mean - sem, mean + sem, color=c, alpha=0.25)

    # shade delay window (subtract 1 from DELAY_TRS to convert from 1-indexed to 0-indexed, then multiply by TR to get seconds; add/subtract 0.6s to center the shading on the TRs)
    dly_start = ((DELAY_TRS[0] - 1) * args.tr ) -0.6 # subtract 0.6s to center the shading on the TR to make sure the TR at DELAY_TRS[0] is fully shaded and included
    dly_end   = ((DELAY_TRS[1] -1 ) * args.tr) + 0.6 # add 0.6s to center the shading on the TR to make sure the TR at DELAY_TRS[1] is fully shaded and included
    ax.axvspan(dly_start, dly_end, color="grey", alpha=0.15,
               label=f"delay (TRs {DELAY_TRS[0]}–{DELAY_TRS[1]})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("FIR β (z-scored within subject)")
    roi_tag = "_".join(all_data.keys())
    ax.set_title(f"FIR timecourse — {roi_tag}")
    ax.set_xticks(time)
    ax.set_xticklabels([f"{t:.1f}" for t in time], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    pdf_path = args.output_dir / f"firtimecourse_{roi_tag}.pdf"
    fig.savefig(pdf_path, dpi=300)
    print(f"\nWrote {pdf_path}")
    plt.show()


if __name__ == "__main__":
    main()