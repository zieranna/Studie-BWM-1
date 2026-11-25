#!/bin/bash
# ---------------------------------------------------------------------------
# gla3_grid.sh -- GLA3 group statistics for all runs of the decoding grid
#
#   bash gla3_grid.sh [grid folder]        (default: .../decoding_grid on this PC)
#   NP=10000 bash gla3_grid.sh             fewer permutations for a quick look
#   INPUTS=hrf_late bash gla3_grid.sh ...  only these inputs (others stay as they are)
#
# For every input (fir_orig, lss17, lss25, hrf, hrf_late; missing ones are skipped)
# and run name:
#   nested-pooled_C1, nested-pooled  -> GLA3_results_decoding_nested.py
#        one row per ROI, FDR across ROIs
#   allcounts_C1, allcounts          -> GLA3_results_decoding.py (original)
#        one row per voxel count, FDR across voxel counts within each ROI
# Output: <grid>/<input>/group_stats_<run>/   (log: gla3.log in that folder)
#   all counts additionally: stats_binding_<ROI>_<run>.csv (for summarize_grid.py)
# Each evaluation uses only the files of exactly its run name. ROIs and voxel
# counts are read from the result files (e.g. dlpfc_noSFL, 57 step-50 counts).
# ---------------------------------------------------------------------------
GRID=${1:-/home/annaz/BindingWM/Studie-BWM-1_CP_revision1/decoding_grid}
AN=/home/annaz/PycharmProjects/Studie-BWM-1/Analysis
G_ORIG=${G_ORIG:-$AN/GLA3_results_decoding.py}
G_NEST=${G_NEST:-$AN/GLA3_results_decoding_nested.py}
NP=${NP:-100000}
ROI_ORDER="V1 V2 V3 V4 early_visual pari dlpfc dlpfc_noSFL visual_all visual_locMT MT_complex"
INPUTS=${INPUTS:-"fir_orig lss17 lss25 hrf hrf_late"}    # e.g. INPUTS=hrf_late bash gla3_grid.sh ...
TAGS="nested-pooled_C1 nested-pooled allcounts_C1 allcounts"

for f in "$G_ORIG" "$G_NEST"; do [ -f "$f" ] || { echo "missing: $f"; exit 1; }; done
[ -d "$GRID" ] || { echo "missing grid folder: $GRID"; exit 1; }
echo "grid: $GRID | permutations: $NP | $(date)"

for IN in $INPUTS; do
    D=$GRID/$IN
    for T in $TAGS; do
        files=$(ls "$D"/sub-*/binding/decoding_binding_*_"${T}"_2*.xlsx 2>/dev/null)
        n=$(printf '%s\n' "$files" | grep -c "xlsx$")
        if [ "$n" -eq 0 ]; then echo "--- $IN / $T: no files, skipped"; continue; fi
        PH=$(printf '%s\n' "$files" | sed -E "s#.*/decoding_binding_(.*)_${T}_[0-9]{8}\.xlsx#\1#" | sort -u | tr '\n' ' ')
        O=$D/group_stats_$T
        rm -rf "$O"; mkdir -p "$O"
        first=$(printf '%s\n' "$files" | head -1)
        read -r ROIS VOX <<< "$(python3 -c "import pandas as pd; d=pd.read_excel('$first', sheet_name='per_pair'); o='$ROI_ORDER'.split(); r=list(dict.fromkeys(d['ROI'])); r=[x for x in o if x in r]+[x for x in r if x not in o]; v=sorted(int(x) for x in d['voxels'].unique()) if 'voxels' in d and str(d['voxels'].iloc[0]).isdigit() else []; print(','.join(r), ','.join(map(str, v)) or '-')")"
        ROIS=${ROIS//,/ }; VOX=${VOX//,/, }
        echo "=== $IN / $T / phase ${PH}($n files; ROIs: $ROIS) $(date +%H:%M)"
        if [[ $T == nested* ]]; then
            python3 -c "import importlib.util as u, sys; s=u.spec_from_file_location('nested_eval', '$G_NEST'); m=u.module_from_spec(s); sys.modules['nested_eval']=m; s.loader.exec_module(m); m.IN_DIR=m.Path('$D'); m.OUT_DIR=m.Path('$O'); m.ANALYSIS='binding'; m.RUN_TAG='$T'; m.PHASES='$PH'.split(); m.N_PERMUTATIONS=$NP; m.ROIS='$ROIS'.split(); sys.argv=['nested_eval', '--in-dir', '$D', '--out-dir', '$O', '--analysis', 'binding', '--run-tag', '$T', '--n-permutations', '$NP', '--phases'] + '$PH'.split() + ['--rois'] + '$ROIS'.split(); m.main()" > "$O/gla3.log" 2>&1 \
                && echo "    ok   -> $O" || echo "    FAILED (see $O/gla3.log)"
        else
            PY="import importlib.util as u, sys, glob; from argparse import Namespace; from pathlib import Path; s=u.spec_from_file_location('gla3', '$G_ORIG'); m=u.module_from_spec(s); sys.modules['gla3']=m; s.loader.exec_module(m); m.find_file=lambda d, sub, an, ph: (sorted(glob.glob(str(d / ('sub-' + sub) / an / ('decoding_' + an + '_' + ph + '_$T' + '_2*.xlsx')))) or [None])[-1]; a=Namespace(input_dir=Path('$D'), output_dir=Path('$O'), analysis='binding', roi=sys.argv[1], phases='$PH'.split(), subjects=['%02d' % i for i in range(1, 21)], voxels=[$VOX], chance=0.5, n_permutations=$NP); st=m.run_permutation_tests(m.subject_means(m.load_all_data(a)), chance=0.5, n_permutations=$NP); print(st[['area', 'condition', 'mean', 'sem', 'p_raw', 'p_fdr', 'signif']].round(4).to_string(index=False)); st.to_csv(a.output_dir / ('stats_binding_' + a.roi + '_$T.csv'), index=False); m.export_stats_excel(st, a.output_dir / ('decoding_stats_binding_' + a.roi + '_$T.xlsx'), a)"
            ok=1
            for r in $ROIS; do
                echo "=== $r" >> "$O/gla3.log"
                python3 -c "$PY" "$r" >> "$O/gla3.log" 2>&1 || ok=0
            done
            [ $ok -eq 1 ] && echo "    ok   -> $O" || echo "    FAILED for at least one ROI (see $O/gla3.log)"
        fi
    done
done
echo "Done. $(date)"