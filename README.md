# Studie-BWM-1
scripts sorted to analyze the dataset of Studie-BWM-1, can be downloaded on openneuro.org

# BindingWM Studie-BWM-1 — Analysis Pipeline

fMRI decoding pipeline for the feature binding in visual working memory study.
The master script `comBIND.py` orchestrates all stages, from preprocessing through group-level analyses.

---

## Repository structure

```
Analysis/
├── comBIND.py                  ← master orchestration script 
│
├── 01_run_fmriprep.py                  ← Stage 1a: fMRIPrep preprocessing
├── 02_trialestimates.py                ← Stage 1b: per-trial FIR-stick GLM
├── 03_postprocess_tmaps.py             ← Stage 1c: z-scoring + pair-averaging
├── 04_fullmodel.py                     ← Stage 1d: full-subject FIR GLM
│
├── 05_create_ROIs.py                   ← Stage 3a: anatomical ROIs in BOLD space
├── 06_ROI_selectvoxel.py               ← Stage 3b: top-|t| voxel selection
├── 07_decoding.py                      ← Stage 3c: SVM decoding
├── 08_betacoefficients.py              ← Stage 3d: SVM weight vectors
│
├── GLA1_analyze_behavior.py            ← Stage 4: behavioral analyses
├── GLA2_timecourse_FIR.py              ← Stage 4: FIR timecourse plots
├── GLA3_results_decoding.py            ← Stage 4: decoding accuracy tables/plots
└── GLA4_feature_segregation.py         ← Stage 4: voxel overlap + weight-vector angles
```

---

## Data folder structure

```
Studie-BWM-1/bindingWM/
│
├── bindingwm_data/
│   ├── sub-XX/
│   │   └── events.tsv/                 ← behavioural event files (BIDS)
│   ├── sub_lst.txt                     ← newline-separated subject IDs
│   └── summary/                        ← GLA1 output (accuracy/RT tables + plots)
│
├── preprocessed/
│   ├── sub-XX/                         ← fMRIPrep derivatives
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
│       ├── model/fir/binding/full/contrast/   ← whole-brain enc/del t-maps (04_fullmodel)
│       └── group_timecourses/                 ← GLA2 FIR timecourse output
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
    │   └── <analysis>/                 ← binding | color | motion | …
    │       ├── accuracy.npy            ← per-fold decoding accuracies
    │       └── svmcoef_<ROI>           ← SVM weight vectors (08_betacoefficients)
    ├── group_results/                  ← GLA3 decoding summary tables + plots
    └── feature_segregation/            ← GLA4 voxel overlap + weight-vector angles
```

---

## Pipeline stages

### Stage 1 — Per-subject preprocessing and GLMs

Runs sequentially for each subject (`sub-01` … `sub-20`).

| Step | Script | Description                                                                           |
|------|--------|---------------------------------------------------------------------------------------|
| 1a | `01_run_fmriprep.py` | Runs fMRIPrep via Docker                                                              |
| 1b | `02_trialestimates.py` | Per-run FIR-stick GLM; produces one t-map per trial per phase                         |
| 1c | `03_postprocess_tmaps.py` | Z-scores t-maps per run, renames run-NN → block-NN, pair-averages a+b trials          |
| 1d | `04_fullmodel.py` | Full-subject FIR GLM; produces whole-brain encoding and delay phase t-maps and z-maps |

### Stage 2 — HCP-MMP1 parcellation (run once)

Runs the `create_subj_volume_parcellation.sh` script to project the HCP-MMP1 atlas into individual subject volume space. Requires FreeSurfer to be complete for all subjects and `sub_lst.txt` to exist.

### Stage 3 — Per-subject ROIs and SVM decoding

| Step | Script | Description |
|------|--------|-------------|
| 3a | `05_create_ROIs.py` | Creates anatomical ROIs (HCP-MMP1 unions + aseg hippocampus) in BOLD space |
| 3b | `06_ROI_selectvoxel.py` | Selects top-\|t\| voxels within each ROI using the encoding-phase t-map |
| 3c | `07_decoding.py` | Runs SVM decoding (delay phase) for each analysis × ROI combination |
| 3d | `08_betacoefficients.py` | Extracts SVM weight vectors for feature segregation analyses |

### Stage 4 — Group-level analyses

| Script | Description |
|--------|-------------|
| `GLA1_analyze_behavior.py` | Accuracy, RT, and error-type summaries |
| `GLA2_timecourse_FIR.py` | Mean FIR timecourses per ROI |
| `GLA3_results_decoding.py` | Per-ROI decoding accuracy tables and plots (runs for each analysis × ROI) |
| `GLA4_feature_segregation.py` | Voxel selection overlap (Jaccard) and angles between SVM weight vectors |

---

## Configuration

Edit the variables at the top of `comBIND.py` before running:

| Variable | Description |
|----------|-------------|
| `MAINdir` | Root directory of the BIDS dataset |
| `scripts_dir` | Directory containing the analysis scripts |
| `fs_license` | Path to your FreeSurfer license file |
| `subjects` | List of subject IDs to process (default: `01`–`20`) |
| `ANALYSES` | Decoding analyses to run |
| `ROIS` | ROIs to include in group-level decoding summaries |
| `DECODING_PHASE` | BOLD phase used for decoding (default: `del`) |
| `VOXEL_SELECT_PHASE` | BOLD phase used for voxel selection (default: `enc`) |
| `N_CPU` | Number of CPUs passed to fMRIPrep (default: `32`) |

**Analyses:** `binding`, `color`, `motion`, `temporal_order`, `binding_interaction`

**ROIs:** `early_visual`, `pari`, `dlpfc`, `V1`, `V2`, `V3`, `V4`

---

## Requirements

- Python environment as in the environment.yml 
- [fMRIPrep](23.0.2) (Docker)
- FreeSurfer (for HCP-MMP1 parcellation)
- A valid FreeSurfer license file

---

## Running the pipeline

```bash
python comBIND.py
```

The script prints each shell command before executing it and warns (but does not abort) on non-zero exit codes, so partial failures can be identified in the log without halting the full run.
