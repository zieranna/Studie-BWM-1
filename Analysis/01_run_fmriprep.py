#!/usr/bin/env python3
"""
01_run_fmriprep.py
==================

Run fMRIPrep preprocessing on a BIDS-formatted fMRI dataset and generate
motion-parameter diagnostic plots for each functional run.

This is the first preprocessing step in the BindingWM analysis pipeline. It
assumes that raw DICOMs have already been converted to NIfTI and organized
into a BIDS-compliant directory structure (see the accompanying data
documentation for the expected layout).

Installation requirements
-------------------------
External software (must be available on PATH):
    * Docker                  https://www.docker.com/
    * fmriprep-docker wrapper:  ``pip install fmriprep-docker``
    * A valid FreeSurfer license file (free registration at
      https://surfer.nmr.mgh.harvard.edu/registration.html)

Python packages:
    * numpy
    * pandas
    * matplotlib
    * mne-bids

Install Python deps via:
    pip install numpy pandas matplotlib mne-bids fmriprep-docker

Expected input layout
---------------------
    <bids-dir>/
        dataset_description.json        (created automatically if missing)
        sub-01/
            anat/  sub-01_ses-01_T1w.nii.gz ...
            ses-01/
                func/  sub-01_ses-01_task-bwm_run-01_bold.nii.gz ...
                fmap/  sub-01_ses-01_acq-AP_run-01_epi.nii.gz ...
            ses-02/ ...
        sub-02/ ...

Usage
-----
    python 01_run_fmriprep.py \\
        --bids-dir   /path/to/downloaded/bids_data \\
        --output-dir /path/to/derivatives \\
        --subject    01 \\
        --fs-license /path/to/freesurfer/license.txt \\
        --cpu        8

The script will:
    1. Ensure ``dataset_description.json`` exists at the BIDS root.
    2. Run fMRIPrep on the requested subject (all sessions auto-detected).
    3. For each functional run, write translation- and rotation-parameter
       plots (PNG) next to the confounds .tsv files in the derivatives.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne_bids


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run fMRIPrep on a BIDS dataset and plot motion parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bids-dir", required=True, type=Path,
                   help="Path to BIDS-formatted input dataset.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Path where fMRIPrep derivatives will be written.")
    p.add_argument("--subject", required=True, type=str,
                   help="Participant label without 'sub-' prefix (e.g. '01').")
    p.add_argument("--fs-license", required=True, type=Path,
                   help="Path to FreeSurfer license file.")
    p.add_argument("--cpu", default=4, type=int,
                   help="Number of CPU cores for fMRIPrep (default: 4).")
    p.add_argument("--skip-fmriprep", action="store_true",
                   help="Skip the fMRIPrep call and only (re)generate motion plots.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------------
def ensure_dataset_description(bids_dir: Path) -> None:
    """Create a minimal dataset_description.json at the BIDS root if missing."""
    desc = bids_dir / "dataset_description.json"
    if desc.exists():
        return
    print(f"Creating dataset description at {desc}")
    mne_bids.make_dataset_description(path=str(bids_dir),
                                      name="BindingWM",
                                      verbose=True)


def run_fmriprep(bids_dir: Path, output_dir: Path, subject: str,
                 fs_license: Path, cpu: int) -> None:
    """Invoke the fmriprep-docker wrapper for one participant."""
    work_dir = output_dir / "work" / subject
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "fmriprep-docker",
        str(bids_dir), str(output_dir), "participant",
        "--participant-label", subject,
        "--fs-license-file", str(fs_license),
        "--output-space", "T1w",
        "--write-graph",
        "--cifti-output",
        "--resource-monitor",
        "--stop-on-first-crash",
        "--ignore", "slicetiming",
        "--skip-bids-validation",
        "-w", str(work_dir),
        "--nthreads", str(cpu),
    ]
    print("Running fMRIPrep:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def plot_motion_parameters(output_dir: Path, subject: str) -> None:
    """Save translation and rotation PNGs next to each confounds .tsv."""
    sub_dir = output_dir / f"sub-{subject}"
    if not sub_dir.exists():
        print(f"[warn] No derivatives found at {sub_dir}; skipping motion plots.")
        return

    tsv_files = sorted(sub_dir.glob("ses-*/func/*confounds*timeseries*.tsv"))
    if not tsv_files:
        tsv_files = sorted(sub_dir.glob("ses-*/func/*.tsv"))

    required = {"trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"}
    n_plotted = 0

    for tsv_path in tsv_files:
        df = pd.read_csv(tsv_path, sep="\t")
        if not required.issubset(df.columns):
            continue

        # demean against the first usable volume so the plot starts at 0
        trans = df[["trans_x", "trans_y", "trans_z"]].iloc[1:] \
                - df[["trans_x", "trans_y", "trans_z"]].iloc[1]
        rot = df[["rot_x", "rot_y", "rot_z"]].iloc[1:] \
              - df[["rot_x", "rot_y", "rot_z"]].iloc[1]

        run_name = tsv_path.stem
        x = np.arange(len(trans))

        # --- translation -----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        for col, color in zip(["trans_x", "trans_y", "trans_z"],
                              ["#6A58C0", "#00B59E", "#DF8536"]):
            ax.plot(x, trans[col], alpha=0.7, color=color, label=col)
        ax.set(title=f"{run_name} - Translation",
               xlabel="Volume (TR)", ylabel="Translation (mm)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(tsv_path.with_name(f"{run_name}_translation.png"), dpi=120)
        plt.close(fig)

        # --- rotation --------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        for col, color in zip(["rot_x", "rot_y", "rot_z"],
                              ["#00B5D6", "#FF6699", "#0EB266"]):
            ax.plot(x, rot[col], alpha=0.7, color=color, label=col)
        ax.set(title=f"{run_name} - Rotation",
               xlabel="Volume (TR)", ylabel="Rotation (rad)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(tsv_path.with_name(f"{run_name}_rotation.png"), dpi=120)
        plt.close(fig)

        n_plotted += 1

    print(f"Saved motion plots for {n_plotted} run(s).")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print("-" * 64)
    print("fMRIPrep preprocessing")
    print("-" * 64)
    for k, v in vars(args).items():
        print(f"  {k:>15}: {v}")
    print("-" * 64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dataset_description(args.bids_dir)

    if not args.skip_fmriprep:
        run_fmriprep(args.bids_dir, args.output_dir, args.subject,
                     args.fs_license, args.cpu)
    else:
        print("Skipping fMRIPrep call (--skip-fmriprep set).")

    plot_motion_parameters(args.output_dir, args.subject)
    print("Done.")


if __name__ == "__main__":
    main()