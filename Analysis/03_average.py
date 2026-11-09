#!/usr/bin/env python3
"""
03b_average_runs.py
===================

Pair averaging of LSS single-trial estimates WITHOUT z-scoring, for every
window folder of one subject (layout of 02b_trialestimates_lss.py):

    <input-dir>/sub-XX/<model>/<stat>/<window>/trials/
        sub-XX_ses-YY_block-BB_run-NN_trial-PP.nii.gz
      -> <input-dir>/sub-XX/<model>/<stat>/<window>/pairs/
        sub-XX_ses-YY_block-BB_pair-K.nii.gz

'Ka' and 'Kb' of the same session and block (the two runs of the block) are
averaged into pair K. Pairs with a or b missing are skipped and counted.
The single-trial maps already carry block and run in their names, so the
trial-level analyses (temporal_order, binding_interaction) read them directly
from trials/ (07_decoding.py --layout lss).

Usage
-----
    python 03b_average_runs.py --subject 01 --input-dir .../lss
    (optionally --estimates hrf/beta/del fir/beta/del to restrict)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_name(p: Path) -> dict:
    out = {}
    for x in p.name.replace(".nii.gz", "").replace(".nii", "").split("_"):
        if "-" in x:
            k, v = x.split("-", 1)
            out[k] = v
    return out


def trial_to_pair(label: str):
    """'4a' -> ('4', 'a'); '13b' -> ('13', 'b'); other -> (None, None)."""
    if label and label[-1] in ("a", "b") and label[:-1].isdigit():
        return label[:-1], label[-1]
    return None, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pair-average LSS single-trial maps "
                                            "within blocks (no z-scoring).")
    p.add_argument("--subject", required=True, type=str)
    p.add_argument("--input-dir", required=True, type=Path,
                   help="LSS root (contains sub-XX/<model>/<stat>/<window>/trials/).")
    p.add_argument("--estimates", nargs="+", default=None,
                   help="Only these, as <model>/<stat>/<window> (default: all).")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def average_estimate(est_dir: Path, sub: str, force: bool) -> tuple:
    bucket = defaultdict(dict)
    for p in sorted((est_dir / "trials").glob(f"sub-{sub}_ses-*_block-*_run-*_trial-*.nii.gz")):
        meta = parse_name(p)
        num, suffix = trial_to_pair(meta.get("trial", ""))
        if num is not None:
            bucket[(meta["ses"], meta["block"], num)][suffix] = p
    pair_dir = est_dir / "pairs"
    pair_dir.mkdir(exist_ok=True)
    n_avg, n_skip = 0, 0
    for (ses, block, num), members in sorted(bucket.items()):
        if "a" not in members or "b" not in members:
            n_skip += 1
            continue
        out = pair_dir / f"sub-{sub}_ses-{ses}_block-{block}_pair-{num}.nii.gz"
        if out.exists() and not force:
            n_avg += 1
            continue
        a_img, b_img = nib.load(str(members["a"])), nib.load(str(members["b"]))
        avg = (a_img.get_fdata(dtype=np.float32) + b_img.get_fdata(dtype=np.float32)) / 2.0
        nib.Nifti1Image(avg.astype(np.float32), a_img.affine).to_filename(str(out))   # float32, not the trial header's dtype
        n_avg += 1
    return n_avg, n_skip


def main() -> None:
    args = parse_args()
    sub_dir = args.input_dir / f"sub-{args.subject}"
    ests = sorted(t.parent for t in sub_dir.glob("*/*/*/trials") if t.is_dir()) \
        if sub_dir.exists() else []
    rel = {d: d.relative_to(sub_dir).as_posix() for d in ests}
    if args.estimates:
        ests = [d for d in ests if rel[d] in args.estimates]
    if not ests:
        raise SystemExit(f"No <model>/<stat>/<window>/trials/ folders under {sub_dir}")
    print(f"Pair averaging (no z-scoring): sub-{args.subject}, {len(ests)} window folders")
    for d in ests:
        n_avg, n_skip = average_estimate(d, args.subject, args.force)
        print(f"  {rel[d]:<22} {n_avg} pair maps, {n_skip} incomplete pairs skipped")
    print("\nDone.")

if __name__ == "__main__":
    main()