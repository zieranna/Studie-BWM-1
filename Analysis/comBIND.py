#!/usr/bin/env python3
"""
00_run_pipeline.py
==================

Master orchestration script for the BindingWM Studie-BWM-1 analysis
pipeline. Runs every per-subject step for the configured subject list.

Stage 1: per-subject preprocessing + GLMs
    01_run_fmriprep.py        preprocessing (Docker)
    02_trialestimates.py      per-run FIR-stick GLM (one t-map per trial)
    03_postprocess_tmaps.py   per-run z-scoring + pair-averaging
    04_fullmodel.py           full-subject FIR GLM (one t-map per phase)

Stage 2: one-time across all subjects
    create_subj_volume_parcellation.sh   HCP-MMP1 atlas in subject space, download from

Stage 3: per-subject ROIs + decoding
    05_create_ROIs.py         anatomical ROIs (HCP-MMP1 + aseg) in BOLD space
    06_ROI_selectvoxel.py     top-|t| voxel selection inside each anat. ROI
                              (uses the enc t-map from 04_fullmodel.py)
    07_decoding.py            SVM decoding for each analysis (--phase del)
    08_betacoefficients.py    extract SVM weight vectors for feature segregation analyses

Stage 4: group-level analyses
    GLA1_analyze_behavior.py     behavioral analyses (accuracy, RT, etc.)
    GLA2_timecourse_FIR.py       plot mean FIR timecourses per ROI
    GLA3_results_decoding.py     tables and plots of per-ROI decoding accuracies
    GLA4_feature_segregation.py plot of voxel selection overlap across ROIs and angles between weight vectors
"""

"""
Folder structure suggestion
----------------
Studie-BWM-1/
│
├── bindingwm_data/
│   ├── sub-XX/
│   │     └── events.tsv/               
│   ├── sub_lst.txt
│   └── summary/                        ← GLA1 behavioral output, e.g. accuracy and RT tables + plots
│
├── preprocessed/
│   ├── sub-XX/
│   └── sourcedata/
│       ├── freesurfer/
│       │   └── hpcmmp1_output/         ← HCP-MMP1 parcellation in subject space
│       └── create_subj_volume_parcellation.sh
│
├── firestimates/
│   └── sub-XX/
│       ├── run-XX/                     ← per-trial t-maps (02_trialestimates)
│       └── block-XX/                   ← z-scored + pair-averaged (03_postprocess)
│
├── firmodelcomplete/
│   └── sub-XX/
│       ├── model/fir/binding/full/
│       │   └── contrast/               ← whole-brain enc/del t-maps (04_fullmodel)
│       └── group_timecourses/          ← GLA2 FIR timecourse output
│
├── roi_anat/
│   └── sub-XX/
│       └── ROI_name.nii.gz             ← anatomical ROIs in BOLD space (05_create_ROIs)
│
├── roi_func/
│   └── sub-XX/
│       └── ROI_top-N.nii.gz            ← voxel-selected ROIs (06_ROI_selectvoxel)
│
└── decoding/
    ├── sub-XX/
    │   └── <analysis>/                 ← e.g. binding, color, motion …
    │       ├── accuracy.npy            ← per-fold decoding accuracies
    │       └── svmcoef_<ROI>           ← SVM weight vectors (07_decoding)
    ├── group_results/                  ← GLA3 decoding summary tables + plots
    └── feature_segregation/            ← GLA4 voxel overlap + weight-vector angles
"""

import os


# ============================================================================
# Paths
# ============================================================================
MAINdir     = "/home/annaz/BindingWM/Studie-BWM-1/bindingWM"                            #directory containing the raw BIDS data (behav_data + imaging_data)
scripts_dir = "/home/annaz/PycharmProjects/IMProjects/binding/Studie-BWM-1/Analysis"    # directory containing the analysis scripts (01_run_fmriprep.py, 02_trialestimates.py, etc.)

behav_dir      = os.path.join(MAINdir, "bindingwm_data")
fmri_dir       = os.path.join(MAINdir, "bindingwm_data")
preproc_dir    = os.path.join(MAINdir, "preprocessed")
firestimates   = os.path.join(MAINdir, "firestimates")
firmodels      = os.path.join(MAINdir, "firmodelcomplete")
roi_anat       = os.path.join(MAINdir, "roi_anat")
roi_func       = os.path.join(MAINdir, "roi_func")
decoding_dir   = os.path.join(MAINdir, "decoding")

fs_license     = "/home/annaz/freesurfer/freesurfer/license.txt" # to be filled in by user
hpcmmp1_output = os.path.join(preproc_dir, "sourcedata", "freesurfer", "hpcmmp1_output")
hpcmmp1_script = os.path.join(preproc_dir, "sourcedata", "create_subj_volume_parcellation.sh")
sub_lst_file   = os.path.join(fmri_dir, "sub_lst.txt")  # NB: no leading slash


# ============================================================================
# Config
# ============================================================================
subjects = [f"{i:02d}" for i in range(1, 21)]   # '01' .. '20'

ANALYSES = ["binding", "color", "motion", "temporal_order", "binding_interaction"]
ROIS = ["early_visual", "pari", "dlpfc", "V1", "V2", "V3", "V4"]
DECODING_PHASE     = "del"   # decoding focused on the delay phase
VOXEL_SELECT_PHASE = "enc"   # top-voxel selection uses the encoding t-map

N_CPU = 32


# ============================================================================
# Helpers
# ============================================================================
def run(cmd: str) -> None:
    """Echo and execute a shell command; warn on non-zero exit."""
    print(f"\n$ {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        print(f"  [warn] exit status {rc}")


# ============================================================================
# Stage 1: per-subject preprocessing + GLMs
# ============================================================================
for sub in subjects:
    print(f"\n{'='*64}\n  Stage 1: sub-{sub}\n{'='*64}")

    # 1a. fMRIPrep
    run(f"python {scripts_dir}/01_run_fmriprep.py "
        f"--bids-dir {fmri_dir} "
        f"--output-dir {preproc_dir} "
        f"--subject {sub} "
        f"--fs-license {fs_license} "
        f"--cpu {N_CPU}")

    # 1b. Per-run trial estimates (one t-map per trial per phase).
    run(f"python {scripts_dir}/02_trialestimates.py "
        f"--subject {sub} "
        f"--events-dir {behav_dir} "
        f"--fmriprep-dir {preproc_dir} "
        f"--output-dir {firestimates}")

    # 1c. Z-score per run, rename run-NN to block-NN, pair-average a+b.
    run(f"python {scripts_dir}/03_postprocess_tmaps.py "
        f"--subject {sub} "
        f"--input-dir {firestimates}")

    # 1d. Full-subject FIR model (whole-brain phase t-maps used downstream
    #     to select top voxels).
    run(f"python {scripts_dir}/04_fullmodel.py "
        f"--subject {sub} "
        f"--events-dir {behav_dir} "
        f"--fmriprep-dir {preproc_dir} "
        f"--output-dir {firmodels}")


# ============================================================================
# Stage 2: HCP-MMP1 parcellation (run ONCE, after FreeSurfer + sub_lst exists)
# ============================================================================
#print(f"\n{'='*64}\n  Stage 2: HCP-MMP1 parcellation\n{'='*64}")
run(f"{hpcmmp1_script} "
    f"-L {sub_lst_file} "
    f"-a HCPMMP1.annot "
    f"-d {hpcmmp1_output} "
    f"-s YES")


# ============================================================================
# Stage 3: per-subject ROIs + decoding
# ============================================================================
for sub in subjects:
    print(f"\n{'='*64}\n  Stage 3: sub-{sub}\n{'='*64}")

    # 3a. Anatomical ROIs (HCP-MMP1 unions + aseg hippocampus) in BOLD space.
    run(f"python {scripts_dir}/05_create_ROIs.py "
        f"--subject {sub} "
        f"--fmriprep-dir {preproc_dir} "
        f"--parcellation-dir {hpcmmp1_output} "
        f"--output-dir {roi_anat}")
    # 3b. Top-|t| voxel selection within each anatomical ROI, using the
    #     encoding-phase whole-brain t-map produced by 04_fullmodel.
    enc_tmap = f"{firmodels}/sub-{sub}/model/fir/binding/full/contrast/sub-{sub}_{VOXEL_SELECT_PHASE}.nii.gz"

    run(f"python {scripts_dir}/06_ROI_selectvoxel.py "
        f"--subject {sub} "
        f"--roi-dir {roi_anat} "
        f"--tmap {enc_tmap} "
        f"--output-dir {roi_func}")

    # 3c. SVM decoding for each analysis, on the delay-phase t-maps.
    for ana in ANALYSES:
        run(f"python {scripts_dir}/07_decoding.py "
            f"--subject {sub} "
            f"--phase {DECODING_PHASE} "
            f"--analysis {ana} "
            f"--tmap-dir {firestimates} "
            f"--roi-dir {roi_func} "
            f"--output-dir {decoding_dir}")

# ============================================================================
# Stage 4: group level analyses
# ============================================================================

print(f"\n{'='*64}\n  Stage 4: group-level analyses\n{'='*64}")

# 4.1 behavioral analyses (accuracy, RT, etc.)
run(f"python {scripts_dir}/GLA1_analyze_behavior.py "
    f"--data-dir {behav_dir} "
    f"--output-dir {behav_dir}/summary")

# 4.2 timecourse analyses (e.g. plot mean FIR timecourses per ROI)
run(f"python {scripts_dir}/GLA2_timecourse_FIR.py "
    f"--firmodel-dir {firmodels} "
    f"--roi-dir {roi_anat} "
    f"--output-dir {firmodels}/group_timecourses")

# 4.3 decoding analyses (e.g. tables and plots of per-ROI decoding accuracies)
for ana in ANALYSES:
    for roi in ROIS:
        run(f"python {scripts_dir}/GLA3_results_decoding.py "
            f"--input-dir {decoding_dir} "
            f"--output-dir {decoding_dir}/group_results"
            f"--analysis {ana} "
            f"--roi {roi}")

# 4.4 feature segregation analyses (e.g. plot of voxel selection overlap across ROIs and angles)
run(f"python {scripts_dir}/GLA4_feature_segregation.py "
    f"--coef-dir {decoding_dir} "
    f"--output-dir {decoding_dir}/feature_segregation")


print("\nAll done.")