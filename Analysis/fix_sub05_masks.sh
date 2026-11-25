#!/bin/bash -l
# ---------------------------------------------------------------------------
# fix_sub05_masks.sh
#
# fMRIPrep wrote no brain mask for sub-05 ses-02 bwm runs 01 and 02; the LSS
# estimation then computed an EPI mask there. This job re-estimates exactly
# these two runs with the corrected 02_trialestimates_lss.py, which then fits
# them without any mask (all voxels; never a computed mask; brain voxels get
# the same values as with a brain mask), and rebuilds everything of sub-05
# that depends on them:
#   1. LSS estimates (--force, only ses-02 runs 01 02):
#        LSS FIR 17 -> lss        (--n-delays 17, ROI masks added as the original run)
#        LSS FIR 25 -> lss_fir25  (--n-delays 25)
#        LSS HRF    -> lss        (--basis hrf)
#   2. pairs of sub-05 (--force): plain pairs and z-scored per run (ddof 1)
#        lss: fir/tmap/del, hrf/tmap/del, hrf/tmap/del_late; lss_fir25: fir/tmap/del
#   3. residuals of the two runs (02c, --force) for FIR 17 and FIR 25
# The decoding of sub-05 is then rerun separately (see the message).
#
#   sbatch fix_sub05_masks.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=fix_sub05
#SBATCH --partition=32GBMx86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/hpc/home/ziera/scripts/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/hpc/home/ziera/scripts/slurm/logs/%x_%j.err

set -euo pipefail
A=/mnt/hpc/home/ziera/scripts/git/Studie-BWM-1/Analysis
P=/mnt/hpc/projects/g_study02_bindingwm/bindingwm-study1
EV=${P}/events_rebuilt
PREP=${P}/preprocessed
SUB=05
SEL=(--sessions 02 --runs 01 02 --force)

set +u
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate bindingwm_x86
set -u
unset PYTHONPATH
export PYTHONUNBUFFERED=1
NJ=${SLURM_CPUS_PER_TASK:-2}

grep -q "def fmriprep_mask" ${A}/02_trialestimates_lss.py || { echo "02_trialestimates_lss.py: old version"; exit 1; }
grep -q "def fmriprep_mask" ${A}/02c_residuals_lss.py || { echo "02c_residuals_lss.py: old version"; exit 1; }
grep -q -- "--ddof" ${A}/03_average.py || { echo "03_average.py: old version"; exit 1; }

echo "=== 1. LSS estimates, sub-${SUB} ses-02 runs 01 02 ($(date))"
python -u ${A}/02_trialestimates_lss.py --subject ${SUB} --basis fir --n-delays 17 \
    --events-dir ${EV} --fmriprep-dir ${PREP} --output-dir ${P}/lss \
    --extra-mask-dir ${P}/roi_func/sub-${SUB} --n-jobs ${NJ} "${SEL[@]}"
python -u ${A}/02_trialestimates_lss.py --subject ${SUB} --basis fir --n-delays 25 \
    --events-dir ${EV} --fmriprep-dir ${PREP} --output-dir ${P}/lss_fir25 \
    --n-jobs ${NJ} "${SEL[@]}"
python -u ${A}/02_trialestimates_lss.py --subject ${SUB} --basis hrf \
    --events-dir ${EV} --fmriprep-dir ${PREP} --output-dir ${P}/lss \
    --extra-mask-dir ${P}/roi_func/sub-${SUB} --n-jobs ${NJ} "${SEL[@]}"

echo "=== 2. pairs of sub-${SUB} ($(date))"
python -u ${A}/03_average.py --subject ${SUB} --input-dir ${P}/lss \
    --estimates fir/tmap/del hrf/tmap/del hrf/tmap/del_late --force
python -u ${A}/03_average.py --subject ${SUB} --input-dir ${P}/lss \
    --estimates fir/tmap/del hrf/tmap/del hrf/tmap/del_late --zscore run --ddof 1 --force
python -u ${A}/03_average.py --subject ${SUB} --input-dir ${P}/lss_fir25 \
    --estimates fir/tmap/del --force
python -u ${A}/03_average.py --subject ${SUB} --input-dir ${P}/lss_fir25 \
    --estimates fir/tmap/del --zscore run --ddof 1 --force

echo "=== 3. residuals, sub-${SUB} ses-02 runs 01 02 ($(date))"
for m in "lss 17" "lss_fir25 25"; do
    set -- $m
    python -u ${A}/02c_residuals_lss.py --subject ${SUB} --n-delays $2 \
        --events-dir ${EV} --fmriprep-dir ${PREP} --output-dir ${P}/$1 \
        --extra-mask-dir ${P}/roi_func/sub-${SUB} \
        --save-mask-dirs ${P}/roi_anat/sub-${SUB} ${P}/roi_func/sub-${SUB} "${SEL[@]}"
done

echo "=== check"
for d in lss/sub-${SUB}/fir/tmap/del lss/sub-${SUB}/hrf/tmap/del lss/sub-${SUB}/hrf/tmap/del_late lss_fir25/sub-${SUB}/fir/tmap/del; do
    for z in "" _zrun_ddof1; do
        echo "  ${d}${z}/pairs: $(ls ${P}/${d}${z}/pairs | wc -l) (288 expected)"
    done
done
echo "finished $(date)"