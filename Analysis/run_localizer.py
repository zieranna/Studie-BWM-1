#!/usr/bin/env python3
"""
run_localizer_pipeline.py
=========================

MT+ ROI for all subjects (task numbering 01-20) with 06b_localizer_mt.py:
all localizer sessions per subject in one GLM, one ROI per subject
(roi_localizer/sub-XX/MT.nii.gz, valid hemispheres only).

Writes localizer/localizer_pipeline_summary.csv and prints a compact table.
Subjects that already have an ROI are skipped (FORCE = True redoes them).

Run:
    nohup python3 run_localizer_pipeline.py > run_localizer_pipeline.out 2>&1 &
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT    = Path("/home/annaz/PycharmProjects/Studie-BWM-1/Analysis/06_ROI_localizer.py")  # = 06b
PARC_DIR  = Path("/home/annaz/BindingWM/Studie-BWM-1/bindingWM/preprocessed/"
                 "sourcedata/freesurfer/hpcmmp1_output")
REV       = Path("/home/annaz/BindingWM/Studie-BWM-1_CP_revision1")
LOC_DIR   = REV / "localizer"
ROI_DIR   = REV / "roi_localizer"
MAPPING   = REV / "timing_check" / "subject_mapping.csv"
PYTHON    = sys.executable

TASK_SUBJECTS = [f"{i:02d}" for i in range(15, 21)]
EXTRA_ARGS = ["--session-exclusion", "none", "--smoothing", "6"]
FORCE = True


def summary_row(final: str) -> dict:
    f = LOC_DIR / f"sub-{final}" / "localizer_summary.csv"
    return pd.read_csv(f).iloc[0].to_dict() if f.exists() else {}


def short(x, n=55) -> str:
    s = "" if pd.isna(x) else str(x)
    return s if len(s) <= n else s[:n - 3] + "..."


def main() -> None:
    for p in (SCRIPT, PARC_DIR, MAPPING):
        if not p.exists():
            sys.exit(f"ERROR: not found: {p}")
    m = pd.read_csv(MAPPING, dtype=str)
    t2f = {f"{int(r.behav_subject):02d}": f"{int(r.final_subject):02d}" for r in m.itertuples()}
    (LOC_DIR / "logs").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for task in TASK_SUBJECTS:
        final = t2f.get(task)
        if final is None:
            print(f"[skip] task sub-{task}: not in subject_mapping.csv")
            continue
        if not FORCE and (ROI_DIR / f"sub-{task}" / "MT.nii.gz").exists():
            print(f"[skip] task sub-{task}: ROI exists")
            continue
        log = LOC_DIR / "logs" / f"sub-{task}.log"
        print(f"[{time.strftime('%H:%M:%S')}] task sub-{task} (final sub-{final}) ...", flush=True)
        cmd = [PYTHON, str(SCRIPT), "--task-subjects", task, "--parcellation-dir", str(PARC_DIR),
               *[str(a) for a in EXTRA_ARGS]]
        with open(log, "w") as fh:
            fh.write("# " + " ".join(cmd) + "\n")
            fh.flush()
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        # echo the per-session and ROI lines of the log
        for line in log.read_text(errors="ignore").splitlines():
            if line.strip().startswith(("ses-", "MT+", "ROI:", "GLM:", "[ERROR]", "no valid")):
                print("   " + line.strip(), flush=True)

    rows = []
    for task in TASK_SUBJECTS:
        final = t2f.get(task)
        r = summary_row(final) if final else {}
        rows.append({**r, "task_subject": task, "final_subject": final})
    df = pd.DataFrame(rows)
    df.to_csv(LOC_DIR / "localizer_pipeline_summary.csv", index=False)

    print("\n" + "=" * 100 + "\nMT+ ROI across subjects\n" + "=" * 100)
    table = pd.DataFrame({
        "sub": df["task_subject"],
        "sessions": [f"{int(u)}/{int(f)}" if pd.notna(u) and pd.notna(f) else ""
                     for u, f in zip(df.get("sessions_used", pd.Series(dtype=float)),
                                     df.get("sessions_found", pd.Series(dtype=float)))],
        "left": [short(f"{v} ({t})" if pd.notna(t) else v, 28) for v, t in
                 zip(df.get("L_verdict", ""), df.get("L_peak_t", pd.Series([None] * len(df))))],
        "right": [short(f"{v} ({t})" if pd.notna(t) else v, 28) for v, t in
                  zip(df.get("R_verdict", ""), df.get("R_peak_t", pd.Series([None] * len(df))))],
        "voxels": [int(v) if pd.notna(v) else "" for v in df.get("roi_voxels", pd.Series([None] * len(df)))],
        "status": [short(s) for s in df.get("roi_status", "")],
    })
    print(table.to_string(index=False))
    ok = df[df.get("roi_status") == "ok"]
    print(f"\n{len(ok)} of {len(df)} subjects with an MT+ ROI; both hemispheres valid: "
          f"{int(((ok.get('L_verdict') != 'not found') & (ok.get('R_verdict') != 'not found')).sum())}")
    exc = df[df.get("sessions_excluded").notna() & (df.get("sessions_excluded") != "")] \
        if "sessions_excluded" in df else pd.DataFrame()
    if len(exc):
        print("\nExcluded sessions:")
        for r in exc.itertuples():
            for part in str(r.sessions_excluded).split("; "):
                print(f"   sub-{r.task_subject} {short(part, 95)}")
    print(f"\nSummary: {LOC_DIR / 'localizer_pipeline_summary.csv'} "
          f"({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()