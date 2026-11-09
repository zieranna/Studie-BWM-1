"""
Pattern component modeling (PCM; Diedrichsen, Yokoi & Arbuckle, 2018, NeuroImage)
for the color-motion binding working-memory data (18 / 36 trial types).

Each trial shows two objects in sequence: (color1, motion1) then (color2, motion2),
with 3 color and 3 motion-direction categories and color1 != color2,
motion1 != motion2.

    36 conditions = ordered pairs of objects (with serial order)
    18 conditions = unordered pairs of objects (order collapsed)

The condition space is split into orthogonal subspaces; each component is the
second moment (G) of one subspace (component model, paper eq. 4):

    dims (18 / 36)   component       what it captures
    2 / 2            color           which two colors are held
    2 / 2            motion          which two directions are held
    4 / 4            set             color pair x direction pair (no binding)
    4 / 4            bind_conj       binding via conjunction patterns shared
                                     across sets
    5 / 5            bind_specific   binding specific to each set
    - / 2            color_order     color x serial position (additive)
    - / 2            motion_order    direction x serial position (additive)
    - / 4            object_order    object x serial position, beyond the two
                                     additive order codes
    - / 10           (not modeled)   remaining set-specific order variance

Analysis, following the paper:
  * Data: one delay-period FIR t-map per trial (tmap_z), all ROI voxels.
  * Fixed effects: one intercept per imaging run (section 2.3). Runs are
    recovered from the run-wise z-scoring of the t-maps (infer_runs).
  * Voxel independence: multivariate noise normalization (section 2.1;
    Walther et al., 2016) with the spatial noise correlation estimated from
    the GLM residual time series (shrinkage, true residual df).
  * Model evidence: every combination of components (model family), fitted
    with leave-one-participant-out group cross-validation (sections 2.7.2,
    3.1). Individual AIC fits are a quick check only.
  * Group inference: distribution, mean and t-test of individual log Bayes
    factors (section 2.7.3).
  * Effect sizes: unbiased component variances from the cross-validated
    second moment (eq. 14; the RSA / crossnobis view), tested against zero
    across participants with a sign-flip permutation test, uncorrected and
    FWE-corrected over components (section 2.7.1).
  * Figures: group cross-validated RDM, component RDMs, component variances.
  * Noise ceiling (section 2.8), approximated with the group-average
    cross-validated G instead of a free model (see noise_ceiling).
  * Binding null: block-wise random swaps of binding configurations.

Usage:
  python pcm_binding.py simulate              # recovery check on simulated data
  python pcm_binding.py real                  # quick check (AIC, no null)
  python pcm_binding.py real cv ceiling null=100   # analysis for the paper
  add "only18" to skip the 36-condition analysis (much faster with cv)
  add "nowhiten" to skip prewhitening, "whiten=maps" to estimate the noise
  from the t-maps instead of the residuals

Requires: pip install PcmPy numpy scipy scikit-learn nibabel pandas matplotlib
(works with pandas 2 and 3; see crossval_table)
"""

import numpy as np
import pandas as pd
import PcmPy as pcm

N_CAT = 3
PAIRS = [(0, 1), (0, 2), (1, 2)]


# ----------------------------------------------------------------------------
# 1. Conditions and trial labels
# ----------------------------------------------------------------------------
def condition_table(with_order=True):
    """One row per condition. Columns: color1, motion1, color2, motion2,
    set (0-8), bind (+1/-1 = binding class within set),
    color_order (+1 = lower color index first), motion_order (+1 = lower
    direction index first).
    Without order, each condition is stored with the lower color first."""
    rows = []
    for c1 in range(N_CAT):
        for c2 in range(N_CAT):
            if c1 == c2 or (not with_order and c1 > c2):
                continue
            for m1 in range(N_CAT):
                for m2 in range(N_CAT):
                    if m1 == m2:
                        continue
                    rows.append(dict(color1=c1, motion1=m1, color2=c2, motion2=m2))
    T = pd.DataFrame(rows)
    cpair = [PAIRS.index(tuple(sorted(p))) for p in zip(T.color1, T.color2)]
    mpair = [PAIRS.index(tuple(sorted(p))) for p in zip(T.motion1, T.motion2)]
    T['set'] = 3 * np.array(cpair) + np.array(mpair)
    # direction paired with the lower-index color
    m_low = np.where(T.color1 < T.color2, T.motion1, T.motion2)
    m_high = np.where(T.color1 < T.color2, T.motion2, T.motion1)
    T['bind'] = np.where(m_low < m_high, 1, -1)
    T['color_order'] = np.where(T.color1 < T.color2, 1, -1)
    T['motion_order'] = np.where(T.motion1 < T.motion2, 1, -1)
    T['cond'] = np.arange(len(T))
    return T


def label_trials(trials, with_order=True):
    """trials: DataFrame with columns color1, motion1, color2, motion2
    (category index 0-2 of first and second object). Returns condition index
    (0..35 or 0..17) for every trial."""
    T = condition_table(with_order)
    tr = trials[['color1', 'motion1', 'color2', 'motion2']].copy()
    if not with_order:  # put the lower color first
        swap = tr.color1 > tr.color2
        tr.loc[swap, ['color1', 'motion1', 'color2', 'motion2']] = \
            tr.loc[swap, ['color2', 'motion2', 'color1', 'motion1']].values
    key = ['color1', 'motion1', 'color2', 'motion2']
    out = tr.merge(T[key + ['cond']], on=key, how='left')
    if out.cond.isna().any():
        raise ValueError('Some trials do not match a condition; check coding.')
    return out.cond.values.astype(int)


# ----------------------------------------------------------------------------
# 2. Feature matrices and components
# ----------------------------------------------------------------------------
def _basis(X, tol=1e-8):
    """Orthonormal basis of the column space of X."""
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, S > tol * S.max()] if S.max() > 0 else U[:, :0]


def _proj(X):
    U = _basis(X)
    return U @ U.T


def features(T):
    n = len(T)
    F = {k: np.zeros((n, w)) for k, w in
         [('color', 3), ('motion', 3), ('conj', 9), ('set', 9), ('bind', 9),
          ('co', 9), ('mo', 9), ('color_pos', 3), ('motion_pos', 3), ('obj_pos', 18)]}
    for i, r in T.iterrows():
        c1, m1, c2, m2, s = r.color1, r.motion1, r.color2, r.motion2, r.set
        F['color'][i, [c1, c2]] = 1
        F['motion'][i, [m1, m2]] = 1
        F['conj'][i, [3 * c1 + m1, 3 * c2 + m2]] = 1
        F['set'][i, s] = 1
        F['bind'][i, s] = r.bind
        F['co'][i, s] = r.color_order
        F['mo'][i, s] = r.motion_order
        F['color_pos'][i, c1] += 1; F['color_pos'][i, c2] -= 1     # first minus second
        F['motion_pos'][i, m1] += 1; F['motion_pos'][i, m2] -= 1
        F['obj_pos'][i, [3 * c1 + m1, 9 + 3 * c2 + m2]] = 1
    return F


def build_components(with_order=True):
    """Returns (names, Gc [n_comp x n_cond x n_cond], condition table).
    Each G is the projection onto its subspace, scaled to a mean diagonal of 1.
    theta is then the pattern variance (per voxel) that the component adds to
    an average condition, which is comparable across components and between
    the 18- and 36-condition analyses."""
    T = condition_table(with_order)
    F = features(T)
    n = len(T)
    C = np.eye(n) - 1 / n                               # centering
    P_col = _proj(C @ F['color'])
    P_mot = _proj(C @ F['motion'])
    P_feat = _proj(np.c_[C @ F['color'], C @ F['motion']])
    P_set = _proj((np.eye(n) - P_feat) @ C @ F['set'])
    P_bind = _proj(F['bind'])                            # within-set binding, 9 dims
    P_bconj = _proj(P_bind @ F['conj'])                  # shared conjunctions, 4 dims
    comps = {'color': P_col, 'motion': P_mot, 'set': P_set,
             'bind_conj': P_bconj, 'bind_specific': P_bind - P_bconj}
    if with_order:
        P_co, P_mo = _proj(F['co']), _proj(F['mo'])      # order subspaces, 9 dims each
        P_cpos = _proj(F['color_pos'])                   # lies inside P_co, 2 dims
        P_mpos = _proj(F['motion_pos'])                  # lies inside P_mo, 2 dims
        R = P_co + P_mo - P_cpos - P_mpos
        P_obj = _proj(R @ F['obj_pos'])                  # 4 dims
        comps.update({'color_order': P_cpos, 'motion_order': P_mpos,
                      'object_order': P_obj})
    names = list(comps)
    Gc = np.stack([comps[k] * n / np.trace(comps[k]) for k in names])
    return names, Gc, T


def check_design(with_order=True):
    names, Gc, T = build_components(with_order)
    F = features(T)
    print(f'{len(T)} conditions')
    for k, G in zip(names, Gc):
        print(f'  {k:14s} dims = {np.linalg.matrix_rank(G)}')
    off = max(np.abs(Gc[i] @ Gc[j]).max()
              for i in range(len(names)) for j in range(i + 1, len(names)))
    print(f'  max overlap between components: {off:.1e}')
    if with_order:   # additive feature x order code (Schneegans & Bays type)
        n = len(T)
        Zadd = np.zeros((n, 12))
        for i, r in T.iterrows():
            Zadd[i, [r.color1, 3 + r.color2, 6 + r.motion1, 9 + r.motion2]] = 1
        P_bind = _proj(F['bind'])
        print(f'  additive feature x order code -> binding: {np.abs(P_bind @ Zadd).max():.1e}')


# ----------------------------------------------------------------------------
# 3. Your files and trial types
# ----------------------------------------------------------------------------
BASE = '/home/annaz/BindingWM/Studie-BWM-1/bindingWM'
FIR = BASE + '/firestimates/sub-{sub:02d}/model/fir/binding/{kind}/del'
ROI = BASE + '/roi_func/sub-{sub:02d}/early_visual/top0700_resampled.nii.gz'
# GLM residuals (time series) for prewhitening. These folders are numbered
# 02-21; bindingWM/sub-01 corresponds to final/sub-02.
RESID = ('/home/annaz/BindingWM/Studie-BWM-1/final/sub-{sub:02d}/model/fir/BindingExp/'
         'firsticks/residuals_single/sub-{sub:02d}_ses-*_run-*_residuals.nii.gz')
RESID_SUB_OFFSET = 1
RUNS_PER_BLOCK = 2
WHITEN = 'residuals'   # 'residuals' (paper / Walther et al.), 'maps' or None
SUBJECTS = None   # list of the included participants, e.g. [1, 2, 4];
                  # None = every sub-* folder in firestimates

# Only the structure matters (which trial types share a color or a direction),
# not the names, so the direction-label question (R2.3c) does not affect PCM.
# Derived from binding_exp_results.xlsx + trials_paper (see
# trial_types_from_results): colors are RGB clusters (green ~ 86/142/87,
# purple ~ 196/88/175, orange ~ 206/97/31); directions are angle ranges
# (55-90, 125-160, 245-280 deg; 0 deg = rightward, counterclockwise).
COLORS = {'purple': 0, 'green': 1, 'orange': 2}
MOTIONS = {'deg55-90': 0, 'deg125-160': 1, 'deg245-280': 2}

# Trial types 1-18 in their "a" order: (first object, second object).
# The "b" version shows the same two objects in reversed order.
TRIAL_TYPES = {
    1: (('purple', 'deg55-90'), ('green', 'deg125-160')),
    2: (('green', 'deg55-90'), ('purple', 'deg125-160')),
    3: (('green', 'deg55-90'), ('orange', 'deg125-160')),
    4: (('orange', 'deg55-90'), ('green', 'deg125-160')),
    5: (('purple', 'deg55-90'), ('orange', 'deg125-160')),
    6: (('orange', 'deg55-90'), ('purple', 'deg125-160')),
    7: (('purple', 'deg55-90'), ('green', 'deg245-280')),
    8: (('green', 'deg55-90'), ('purple', 'deg245-280')),
    9: (('green', 'deg55-90'), ('orange', 'deg245-280')),
    10: (('orange', 'deg55-90'), ('green', 'deg245-280')),
    11: (('purple', 'deg55-90'), ('orange', 'deg245-280')),
    12: (('orange', 'deg55-90'), ('purple', 'deg245-280')),
    13: (('purple', 'deg125-160'), ('green', 'deg245-280')),
    14: (('green', 'deg125-160'), ('purple', 'deg245-280')),
    15: (('green', 'deg125-160'), ('orange', 'deg245-280')),
    16: (('orange', 'deg125-160'), ('green', 'deg245-280')),
    17: (('purple', 'deg125-160'), ('orange', 'deg245-280')),
    18: (('orange', 'deg125-160'), ('purple', 'deg245-280')),
}


def trial_types_from_results(results_xlsx, trials_paper):
    """Derives TRIAL_TYPES from one behavioral results file (columns Set,
    Trial, Dir1/2, Col1.x/Col2.x) and the ORIG -> paper-label table.
    ORIG = (Set - 1) * 18 + Trial. Use it to check other participants."""
    d = pd.read_excel(results_xlsx)
    paper = pd.read_csv(trials_paper, sep='\t').set_index('ORIG').Paper

    def color(rgb):
        r, g, b = rgb.T
        return np.where(g > 125, 'green', np.where(b > 120, 'purple', 'orange'))

    def direction(a):
        return np.select([(a >= 50) & (a <= 95), (a >= 120) & (a <= 165),
                          (a >= 240) & (a <= 285)],
                         ['deg55-90', 'deg125-160', 'deg245-280'], 'NA')

    d['c1'] = color(d[['Col1.1', 'Col1.2', 'Col1.3']].values)
    d['c2'] = color(d[['Col2.1', 'Col2.2', 'Col2.3']].values)
    d['m1'], d['m2'] = direction(d.Dir1.values), direction(d.Dir2.values)
    if (d[['m1', 'm2']] == 'NA').any().any():
        raise ValueError('Direction outside the three categories')
    d['orig'] = (d.Set - 1) * 18 + d.Trial
    g = d.groupby('orig')[['c1', 'm1', 'c2', 'm2']]
    if (g.nunique() > 1).any().any():
        raise ValueError('An original trial number has varying categories')
    tt, rev = {}, {}
    for orig, r in g.first().iterrows():
        lab = paper[orig]
        num, ab = int(lab[:-1]), lab[-1]
        objs = ((r.c1, r.m1), (r.c2, r.m2))
        if ab == 'a':
            tt[num] = objs
        else:
            rev[num] = objs[::-1]
    if tt != rev:
        raise ValueError('A "b" trial type is not the reversed "a" trial type')
    return dict(sorted(tt.items()))


def check_trial_types():
    """Checks TRIAL_TYPES: 18 different pairs of objects, and trial types 2k-1
    and 2k form decoding set k (same two colors and directions, swapped
    pairing). The "b" = reversed-order assumption cannot be checked here."""
    missing = sorted(set(range(1, 19)) - set(TRIAL_TYPES))
    if missing:
        raise ValueError(f'TRIAL_TYPES is missing trial types {missing}')
    tr = pd.DataFrame([dict(trial_type=n, color1=COLORS[a[0]], motion1=MOTIONS[a[1]],
                            color2=COLORS[b[0]], motion2=MOTIONS[b[1]],
                            first=f'{a[0]}-{a[1]}', second=f'{b[0]}-{b[1]}')
                       for n, (a, b) in sorted(TRIAL_TYPES.items())])
    if (tr.color1 == tr.color2).any() or (tr.motion1 == tr.motion2).any():
        raise ValueError('A trial type has two equal colors or two equal directions')
    cond = label_trials(tr, with_order=False)
    if len(set(cond)) != 18:
        raise ValueError('Two trial types describe the same pair of objects')
    T = condition_table(False)
    tr['set'] = (tr.trial_type + 1) // 2
    tr['pcm_set'], tr['pcm_bind'] = T.set.values[cond], T.bind.values[cond]
    g = tr.groupby('set')
    if not (g.pcm_set.nunique().eq(1).all() and g.pcm_bind.nunique().eq(2).all()):
        print(tr[['trial_type', 'set', 'first', 'second', 'pcm_set']].to_string(index=False))
        raise ValueError('Trial types 2k-1 and 2k do not form a binding set')
    print(tr[['set', 'trial_type', 'first', 'second']].to_string(index=False))
    print('TRIAL_TYPES ok: 18 different pairs, 9 binding sets')
    return tr


# ----------------------------------------------------------------------------
# 4. Loading, run structure and prewhitening
# ----------------------------------------------------------------------------
def load_subject(sub, kind='tmap_z', roi=None):
    """Delay-period FIR t-maps of one participant inside the ROI mask.
    kind = 'tmap_z'    -> one map per trial type (1a ... 18b) and block
           'tmap_pair' -> one map per pair (1 ... 18) and block
    Returns Y (n_maps x n_voxels), the trial table (block_id = session x
    block) and the mask of kept ROI voxels (voxels with NaN are dropped)."""
    import glob, os, re
    import nibabel as nib
    folder = FIR.format(sub=sub, kind=kind)
    pat = re.compile(r'ses-(\d+)_block-(\d+)_(?:pair|trial)-(\d+)([ab]?)\.nii(?:\.gz)?$')
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, '*.nii*'))):
        m = pat.search(os.path.basename(f))
        if m is None:
            continue
        num, ab = int(m[3]), m[4]
        first, second = TRIAL_TYPES[num]
        if ab == 'b':
            first, second = second, first
        rows.append(dict(file=f, ses=int(m[1]), block=int(m[2]), trial_type=f'{num}{ab}',
                         color1=COLORS[first[0]], motion1=MOTIONS[first[1]],
                         color2=COLORS[second[0]], motion2=MOTIONS[second[1]]))
    if not rows:
        raise FileNotFoundError(f'No t-maps found in {folder}')
    trials = pd.DataFrame(rows)
    trials['block_id'] = trials.ses * 100 + trials.block

    mask_img = nib.load((roi or ROI).format(sub=sub))
    mask = np.nan_to_num(np.asarray(mask_img.dataobj).squeeze()) != 0
    Y = np.empty((len(trials), mask.sum()))
    for i, f in enumerate(trials.file):
        img = nib.load(f)
        if img.shape[:3] != mask.shape or not np.allclose(img.affine, mask_img.affine, atol=1e-3):
            raise ValueError(f'{f} is not in the space of the ROI mask')
        Y[i] = np.asarray(img.dataobj).squeeze()[mask]
    keep = np.isfinite(Y).all(axis=0)
    if not keep.all():
        print(f'  sub-{sub:02d}: dropping {(~keep).sum()} voxels with NaN/inf')
        Y = Y[:, keep]
    return Y, trials.drop(columns='file'), keep


def infer_runs(Y, trials, tol=1e-4):
    """Assigns each map to its imaging run (paper, section 2.3: one fixed
    effect per run). The FIR t-maps were z-scored across the trials of each
    run, so in every voxel the maps of one run sum to zero. Each run therefore
    adds one exact linear dependency to the maps of a block, and the null space
    of the block's maps shows which maps belong together. If no such structure
    is found, the block is used as the run."""
    run = np.zeros(len(trials), dtype=int)
    found = []
    for b, idx in trials.groupby('block_id').indices.items():
        U, S, _ = np.linalg.svd(Y[idx], full_matrices=False)
        null = U[:, S < tol * S[0]]
        k = null.shape[1]
        groups = np.zeros(len(idx), dtype=int)
        if k >= 2:
            Q = null @ null.T                       # 1/n_run within a run, 0 between
            same = Q > Q.max() / 2
            keys = {tuple(r): g for g, r in enumerate(np.unique(same, axis=0))}
            cand = np.array([keys[tuple(r)] for r in same])
            if len(keys) == k and np.bincount(cand).min() > 1:
                groups = cand
            else:
                k = 1
        run[idx] = b * 10 + groups
        found.append(k)
    trials = trials.copy()
    trials['run'] = run
    return trials, found


def prewhiten(Y, trials):
    """Multivariate noise normalization (Walther et al., 2016; paper, section
    2.1: PCM assumes independent voxels). The voxel-by-voxel covariance is
    estimated from the maps after removing run means, shrunk towards a
    diagonal (Ledoit-Wolf), and Y is multiplied by its inverse square root.
    Condition means are deliberately not removed: the residual covariance
    would then depend on the condition-mean noise, which biased the
    cross-validated variances upwards in simulation. Condition effects are
    about 1% of the variance, so they hardly affect the estimate. Variances
    after whitening are in whitened units."""
    from sklearn.covariance import ledoit_wolf
    D = pcm.matrix.indicator(trials.run.values)
    E = Y - D @ np.linalg.lstsq(D, Y, rcond=None)[0]
    Sigma, shrink = ledoit_wolf(E, assume_centered=True)
    w, V = np.linalg.eigh(Sigma)
    return Y @ (V / np.sqrt(w)) @ V.T, shrink


def noise_correlation_from_residuals(sub, keep, trials, roi=None, tol=1e-5,
                                     cache=True):
    """Spatial noise correlation of the ROI voxels from the GLM residual time
    series (Walther et al., 2016; paper, section 2.1), pooled over runs.

    * Each run's residuals span only n_volumes - n_regressors dimensions, so
      each run is reduced to its numerical rank (via SVD; the residual cross-
      product is unchanged). The shrinkage then uses the true residual
      degrees of freedom, not the number of volumes.
    * The t-maps are already divided by each voxel's noise SD, so the
      residuals are standardized per voxel and only the correlation between
      voxels is removed. Shrinkage: Ledoit-Wolf towards the identity.
    * The residuals are orthogonal to the trial estimates, so the whitening
      does not depend on the trial patterns (no bias in the cross-validated
      estimates).
    Returns R (n_voxels x n_voxels), shrinkage, df per run, file count."""
    import glob, os, re
    import nibabel as nib
    from sklearn.covariance import ledoit_wolf
    cfile = f'pcm_noisecorr_sub-{sub:02d}.npz'
    if cache and os.path.exists(cfile):
        z = np.load(cfile)
        if z['R'].shape[0] == keep.sum():
            return z['R'], float(z['shrink']), z['dfs'], int(z['n_files'])
    rs = sub + RESID_SUB_OFFSET
    files = sorted(glob.glob(RESID.format(sub=rs)))
    if not files:
        raise FileNotFoundError(f'No residuals for sub-{sub:02d}: {RESID.format(sub=rs)}')
    # sanity check: same sessions, RUNS_PER_BLOCK runs per block
    ses_res = pd.Series([int(re.search(r'ses-(\d+)', os.path.basename(f))[1]) for f in files])
    exp = trials.groupby('ses').block.nunique() * RUNS_PER_BLOCK
    got = ses_res.value_counts().sort_index()
    if exp.to_dict() != got.to_dict():
        print(f'  WARNING sub-{sub:02d}: residual runs per session {got.to_dict()} '
              f'do not match {RUNS_PER_BLOCK} x blocks {exp.to_dict()}; '
              f'check RESID_SUB_OFFSET')
    mask_img = nib.load((roi or ROI).format(sub=sub))
    mask = np.nan_to_num(np.asarray(mask_img.dataobj).squeeze()) != 0
    lo = [i.min() for i in np.nonzero(mask)]
    hi = [i.max() + 1 for i in np.nonzero(mask)]
    sub_mask = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    parts, dfs = [], []
    for f in files:
        img = nib.load(f)
        if img.shape[:3] != mask.shape or not np.allclose(img.affine, mask_img.affine, atol=1e-3):
            raise ValueError(f'{f} is not in the space of the ROI mask')
        slab = np.asarray(img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], :],
                          dtype=np.float64)
        E = slab[sub_mask][keep].T                       # volumes x voxels
        E = E[np.isfinite(E).all(axis=1)]
        _, S, Vt = np.linalg.svd(E, full_matrices=False)
        r = int((S > tol * S[0]).sum())
        parts.append(S[:r, None] * Vt[:r])
        dfs.append(r)
    X = np.vstack(parts)
    X = X / np.sqrt((X ** 2).mean(axis=0))              # unit variance per voxel
    R, shrink = ledoit_wolf(X, assume_centered=True)
    dfs = np.array(dfs)
    if cache:
        np.savez(cfile, R=R, shrink=shrink, dfs=dfs, n_files=len(files))
    return R, shrink, dfs, len(files)


def map_noise_correlation(Y, trials):
    """Mean |correlation| between voxels of the map residuals (after removing
    run and condition means). Should drop clearly after whitening if the
    residuals belong to the same participant and space."""
    cond = label_trials(trials, True)
    D = np.c_[pcm.matrix.indicator(cond), pcm.matrix.indicator(trials.run.values)]
    E = Y - D @ np.linalg.lstsq(D, Y, rcond=None)[0]
    C = np.corrcoef(E.T)
    return np.abs(C[np.triu_indices_from(C, 1)]).mean()


def whiten_with(Y, R):
    w, V = np.linalg.eigh(R)
    return Y @ (V / np.sqrt(w)) @ V.T


def make_dataset(Y, trials, with_order=True):
    """One PCM data set; partitions (fixed effects) = runs."""
    cond = label_trials(trials, with_order)
    return pcm.dataset.Dataset(np.asarray(Y, float),
                               obs_descriptors={'cond_vec': cond,
                                                'part_vec': trials['run'].values})


def prepare_subject(sub, kind='tmap_z', whiten=WHITEN):
    """Loads one participant, recovers the runs and prewhitens.
    whiten = 'residuals' (GLM residuals, recommended), 'maps' (noise estimated
    from the t-maps themselves) or None."""
    Y, trials, keep = load_subject(sub, kind)
    trials, k = infer_runs(Y, trials)
    msg = ''
    if whiten == 'residuals':
        R, shrink, dfs, nf = noise_correlation_from_residuals(sub, keep, trials)
        before = map_noise_correlation(Y, trials)
        Y = whiten_with(Y, R)
        after = map_noise_correlation(Y, trials)
        msg = (f', whitened with {nf} residual runs (df {dfs.min()}-{dfs.max()} '
               f'per run, shrinkage {shrink:.2f}); mean |r| between voxels '
               f'{before:.3f} -> {after:.3f}')
        if after >= before:
            print(f'  WARNING sub-{sub:02d}: whitening did not reduce the voxel '
                  f'correlations of the maps; check RESID_SUB_OFFSET and the space')
    elif whiten == 'maps':
        Y, shrink = prewhiten(Y, trials)
        msg = f', whitened from the maps (shrinkage {shrink:.2f})'
    print(f'  sub-{sub:02d} {kind}: {len(trials)} maps, {Y.shape[1]} voxels, '
          f'{trials.block_id.nunique()} blocks, {trials.run.nunique()} runs '
          f'(runs per block: {sorted(set(max(x, 1) for x in k))})' + msg)
    return Y, trials


# ----------------------------------------------------------------------------
# 5. Unbiased component variances from the cross-validated G (paper, eq. 14)
# ----------------------------------------------------------------------------
def cv_G(Y, trials, with_order=True):
    """Cross-validated second moment (paper, eq. 14) across blocks, centered.
    Every block contains every condition."""
    _, _, T = build_components(with_order)
    n = len(T)
    cond = label_trials(trials, with_order)
    blk = trials.block_id.values
    Ub = []
    for b in np.unique(blk):
        U = np.full((n, Y.shape[1]), np.nan)
        for c in range(n):
            sel = (blk == b) & (cond == c)
            if sel.any():
                U[c] = Y[sel].mean(0)
        if not np.isnan(U).any():
            Ub.append(U)
    Ub = np.array(Ub)
    M, P = len(Ub), Y.shape[1]
    tot = Ub.sum(0)
    G = sum(Ub[m] @ ((tot - Ub[m]) / (M - 1)).T for m in range(M)) / (M * P)
    C = np.eye(n) - 1 / n
    return C @ G @ C


def cv_component_variance(Y, trials, with_order=True, G=None):
    """PCM variance estimates are biased upwards, because G is constrained to
    be positive definite (paper, sections 2.7.1 and 4). The cross-validated
    second moment (eq. 14) is unbiased, so its projection on each component
    is an unbiased variance estimate that can be negative and can be tested
    against zero across participants. Units as theta in PCM: pattern variance
    per voxel added to an average condition."""
    names, Gc, _ = build_components(with_order)
    n = Gc.shape[1]
    G = cv_G(Y, trials, with_order) if G is None else G
    out = {}
    for k, Gk in zip(names, Gc):
        r = np.trace(Gk) ** 2 / np.trace(Gk @ Gk)          # dims of the subspace
        out[k] = np.trace(Gk * r / n @ G) / n
    return pd.Series(out)


def signflip_test(V, n_perm=100000, max_exact=20, seed=0):
    """Group-level permutation test (sign flipping) of the cross-validated
    component variances, one-sided (variance > 0).

    The per-participant estimates are unbiased, and because the components are
    orthogonal, an absent component's estimate is symmetric around zero even
    when other components are present. Under H0 the sign of each
    participant's estimate is therefore exchangeable (Nichols & Holmes, 2002).
    All 2^n sign patterns are used when n <= max_exact, otherwise n_perm
    random ones. The statistic is the t-value (with sign flips equivalent to
    the mean). The same flips are applied to all components, so the maximum
    t over components gives family-wise error corrected p-values.
    Returns t, p (uncorrected) and p (FWE) per component."""
    X = V.values.astype(float)
    n, k = X.shape
    ss = (X ** 2).sum(0)

    def tvals(S):                                      # S: patterns x n of +-1
        m = S @ X / n
        sd = np.sqrt(np.clip((ss - n * m ** 2) / (n - 1), 1e-300, None))
        return m / (sd / np.sqrt(n))

    t_obs = tvals(np.ones((1, n)))[0]
    tol = 1e-8 * np.maximum(1, np.abs(t_obs))    # rounding in the variance term
    ge, ge_max, total = np.zeros(k), np.zeros(k), 0
    if n <= max_exact:
        chunks = [np.arange(a, min(a + 2 ** 16, 2 ** n)) for a in range(0, 2 ** n, 2 ** 16)]
        exact = True
    else:
        rng = np.random.default_rng(seed)
        chunks = [None] * int(np.ceil(n_perm / 2 ** 14))
        exact = False
    for c in chunks:
        if exact:
            S = 1 - 2 * ((c[:, None] >> np.arange(n)) & 1)
        else:
            S = rng.choice([-1, 1], size=(2 ** 14, n))
        T = tvals(S)
        ge += (T >= t_obs - tol).sum(0)
        ge_max += (T.max(1)[:, None] >= t_obs - tol).sum(0)
        total += len(S)
    if exact:
        p, p_fwe = ge / total, ge_max / total
    else:
        p, p_fwe = (ge + 1) / (total + 1), (ge_max + 1) / (total + 1)
    return pd.DataFrame({'t': t_obs, 'p perm': p, 'p perm FWE': p_fwe},
                        index=V.columns)


def summarize_cv(V):
    """V: participants x components (cv variance). Mean, SE and the sign-flip
    permutation test (one-sided), uncorrected and FWE-corrected over
    components."""
    out = pd.DataFrame({'mean (x1e-3)': V.mean() * 1e3,
                        'SE (x1e-3)': V.sem() * 1e3,
                        'n > 0': (V > 0).sum()})
    out = out.join(signflip_test(V))
    return out[['mean (x1e-3)', 'SE (x1e-3)', 't', 'p perm', 'p perm FWE', 'n > 0']].round(4)


# ----------------------------------------------------------------------------
# 6. PCM model family, inference and noise ceiling
# ----------------------------------------------------------------------------
def crossval_table(data, models, theta, scale_prior=1e3):
    """Likelihood, scale and noise of each left-out participant.
    PcmPy 1.2 stores these with chained pandas assignment, which silently
    does nothing under pandas >= 3 (copy-on-write). The fitted parameters
    (theta) are returned correctly, so the left-out likelihoods are
    recomputed from them here; under pandas < 3 this gives the same numbers
    PcmPy stores."""
    from PcmPy.inference import set_up_fit_group, likelihood_individ
    Z, X, YY, n_channel, Noise, _ = set_up_fit_group(data, fixed_effect='block',
                                                     noise_cov=None)
    n_s, names = len(data), [m.name for m in models]
    LL, sc, no = (np.zeros((n_s, len(models))) for _ in range(3))
    for i, m in enumerate(models):
        for s in range(n_s):
            th = theta[i][:, s]
            nll = likelihood_individ(th, m, YY[s], Z[s], X=X[s], Noise=Noise[s],
                                     n_channel=n_channel[s], fit_scale=True,
                                     scale_prior=scale_prior, return_deriv=0)
            LL[s, i] = -np.ravel(nll)[0]
            sc[s, i], no[s, i] = np.exp(th[-2]), np.exp(th[-1])
    cols = pd.MultiIndex.from_product([['likelihood', 'noise', 'scale'], names],
                                      names=['variable', 'model'])
    return pd.DataFrame(np.c_[LL, no, sc], columns=cols)


def run_pcm(data, with_order=True, crossval=False, fixed=None, verbose=False):
    """Fits every combination of components (model family) to each participant.
    crossval=True : leave-one-participant-out group cross-validation (paper,
                    section 2.7.2 and Fig. 2b; used for component models in
                    section 3.1). Use this for the paper.
    crossval=False: individual fits with an AIC penalty (quick check only).
    fixed         : component names contained in every model; they get no
                    posterior or variance estimate."""
    import warnings
    warnings.filterwarnings('ignore', module='PcmPy')
    warnings.filterwarnings('ignore', category=FutureWarning)
    names, Gc, _ = build_components(with_order)
    fixed = fixed or []
    free = [i for i, k in enumerate(names) if k not in fixed]
    base = [i for i, k in enumerate(names) if k in fixed]
    MF = pcm.model.ModelFamily(Gc[free], comp_names=[names[i] for i in free],
                               basecomponents=Gc[base] if base else None)
    if crossval:
        _, theta = pcm.fit_model_group_crossval(data, MF.models, fit_scale=True,
                                                verbose=verbose)
        T = crossval_table(data, MF.models, theta)
        method = None
    else:
        T, theta = pcm.fit_model_individ(data, MF.models, verbose=verbose)
        method = 'AIC'
    LL = T['likelihood'].values
    post = MF.model_posterior(LL, method)
    exptheta = MF.component_exptheta(theta)
    if crossval:
        exptheta = exptheta * T['scale'].values[None]
    cn = list(MF.comp_names)
    return dict(
        posterior=MF.component_posterior(LL, method=method, format='DataFrame'),
        logBF=MF.component_bayesfactor(LL, method=method, format='DataFrame'),
        variance=pd.DataFrame((exptheta * post[None]).sum(axis=2).T, columns=cn),
        LL=pd.DataFrame(LL, columns=MF.model_names), family=MF,
        crossval=crossval)


def summarize_pcm(res):
    """Distribution of individual log Bayes factors per component (paper,
    section 2.7.3): mean (expected evidence for a new participant), a t-test
    of the individual log BFs against 0, and the number of participants with
    positive (> 1) and strong (> 3) evidence (Kass & Raftery, 1995).
    With AIC, log BF = -1 is the floor (component unused), so it cannot show
    evidence against a component. With group cross-validation the log BFs of
    different participants are not strictly independent. Log BFs above
    about 41 are capped by PcmPy (posterior floor 1e-18)."""
    from scipy import stats
    bf = res['logBF']
    t, p = stats.ttest_1samp(bf, 0)
    if not res['crossval']:      # AIC: floor at -1, a test against 0 is meaningless
        t, p = np.full(bf.shape[1], np.nan), np.full(bf.shape[1], np.nan)
    return pd.DataFrame({
        'mean log BF': bf.mean(), 'SE': bf.sem(),
        't': np.asarray(t), 'p (two-sided)': np.asarray(p),
        'median': bf.median(),
        'n > 1': (bf > 1).sum(), 'n > 3': (bf > 3).sum(),
        'mean posterior': res['posterior'].mean(),
        'variance (x1e-3)': res['variance'].mean() * 1e3}).round(3)


def noise_ceiling(subjects, data, res, with_order=False):
    """Noise ceiling (paper, section 2.8), as pseudo-R2 with 0 = null model
    (G = 0) and 1 = upper ceiling. The paper fits a free model (171 parameters
    for 18 conditions, 666 for 36), which was too slow here. Instead, as for
    RSA noise ceilings (Nili et al., 2014), the ceilings are fixed models with
    the group-average cross-validated G (eq. 14, made positive semi-definite)
    and free scale and noise per participant: upper ceiling = average over all
    participants (includes the evaluated one), lower ceiling = average over
    the other participants (same logic as the group cross-validation)."""
    import warnings
    warnings.filterwarnings('ignore', module='PcmPy')
    Gs = np.array([cv_G(Y, t, with_order) for Y, t in subjects])

    def psd(G):
        w, V = np.linalg.eigh((G + G.T) / 2)
        return (V * np.clip(w, 0, None)) @ V.T

    up, lo = [], []
    for i, d in enumerate(data):
        others = np.delete(np.arange(len(data)), i)
        for G, store in ((Gs.mean(0), up), (Gs[others].mean(0), lo)):
            M = pcm.model.FixedModel('ceiling', psd(G))
            T, _ = pcm.fit_model_individ([d], [M], fit_scale=True, verbose=False)
            store.append(T['likelihood'].values.ravel()[0])
    LL = res['LL']
    null, full = LL.iloc[:, 0].values, LL.iloc[:, -1].values
    up, lo = np.array(up), np.array(lo)
    r2 = lambda x: (x - null) / (up - null)
    return pd.DataFrame({'full model': r2(full), 'lower ceiling': r2(lo),
                         'upper ceiling': 1.0})


# ----------------------------------------------------------------------------
# 7. Permutation null for the binding components (18 conditions)
# ----------------------------------------------------------------------------
def swap_binding(trials, rng):
    """In each block and set, swap the two binding configurations with
    probability .5 (exchange the directions of the two objects). Color,
    motion and set labels stay intact; binding stops being consistent
    across blocks."""
    T = condition_table(False)
    s = T.set.values[label_trials(trials, False)]
    flip = rng.random((trials.block_id.max() + 1, 9)) < .5
    f = flip[trials.block_id.values, s]
    out = trials.copy()
    out.loc[f, 'motion1'] = trials.loc[f, 'motion2'].values
    out.loc[f, 'motion2'] = trials.loc[f, 'motion1'].values
    return out


def binding_null(subjects, n_perm=100, pcm_fits=True, seed=0):
    """subjects: list of (Y, trials). Returns observed statistics and their
    null distributions for bind_conj and bind_specific: mean cv variance and,
    if pcm_fits, mean log BF and n(log BF > 1) from individual fits of a
    reduced family (color, motion and set in every model; 4 models)."""
    rng = np.random.default_rng(seed)
    comps = ['bind_conj', 'bind_specific']

    def stats_for(subj):
        V = pd.DataFrame([cv_component_variance(Y, t, False) for Y, t in subj])
        row = {f'{k} cv variance': V[k].mean() for k in comps}
        if pcm_fits:
            data = [make_dataset(Y, t, False) for Y, t in subj]
            bf = run_pcm(data, False, fixed=['color', 'motion', 'set'])['logBF']
            row.update({f'{k} mean log BF': bf[k].mean() for k in comps})
            row.update({f'{k} n log BF > 1': (bf[k] > 1).sum() for k in comps})
        return row

    obs = pd.Series(stats_for(subjects))
    null = []
    for i in range(n_perm):
        null.append(stats_for([(Y, swap_binding(t, rng)) for Y, t in subjects]))
        if (i + 1) % 10 == 0:
            print(f'  permutation {i + 1}/{n_perm}')
    null = pd.DataFrame(null)
    p = {k: (1 + (null[k] >= obs[k]).sum()) / (1 + n_perm) for k in obs.index}
    summary = pd.DataFrame({'observed': obs, 'null mean': null.mean(),
                            'null 95%': null.quantile(.95), 'p': pd.Series(p)})
    return summary.round(4), null


# ----------------------------------------------------------------------------
# 8. Simulation (same structure as the real data, incl. run-wise z-scoring)
# ----------------------------------------------------------------------------
def simulate(theta, n_sub=6, n_block=8, n_vox=300, seed=1):
    """Each block = 2 runs; each run contains 18 of the 36 ordered trial types
    (one ordering of each pair of objects), and maps are z-scored within run,
    as in the FIR pipeline. Returns a list of (Y, trials)."""
    rng = np.random.default_rng(seed)
    names, Gc, T36 = build_components(True)
    G = sum(theta.get(k, 0) * Gc[i] for i, k in enumerate(names))
    w, V = np.linalg.eigh(G)
    A = V * np.sqrt(np.clip(w, 0, None))
    T18 = condition_table(False)
    out = []
    for s in range(n_sub):
        U = A @ rng.standard_normal((A.shape[1], n_vox))
        rows = []
        for b in range(n_block):
            order = rng.random(18) < .5
            for r in range(2):
                for j, c in T18.iterrows():
                    first = order[j] if r == 0 else not order[j]
                    o = (c.color1, c.motion1, c.color2, c.motion2)
                    o = o if first else (o[2], o[3], o[0], o[1])
                    rows.append(dict(color1=o[0], motion1=o[1], color2=o[2],
                                     motion2=o[3], block_id=101 + b, run_true=2 * b + r))
        trials = pd.DataFrame(rows)
        Y = U[label_trials(trials, True)] + rng.standard_normal((len(trials), n_vox))
        for r in np.unique(trials.run_true):
            m = trials.run_true.values == r
            Y[m] = (Y[m] - Y[m].mean(0)) / Y[m].std(0)
        out.append((Y, trials))
    return out


# ----------------------------------------------------------------------------
# 9. Figure of the component matrices
# ----------------------------------------------------------------------------
def condition_order(with_order=True):
    """Condition indices sorted by the paper's trial types (1a, 1b, 2a, ... or
    1, 2, ...), so that sets k = trial types 2k-1 and 2k are adjacent, and
    their labels."""
    rows = []
    for num, (a, b) in sorted(TRIAL_TYPES.items()):
        for ab, (f, sc) in (('a', (a, b)), ('b', (b, a))):
            rows.append(dict(color1=COLORS[f[0]], motion1=MOTIONS[f[1]],
                             color2=COLORS[sc[0]], motion2=MOTIONS[sc[1]],
                             label=f'{num}{ab}' if with_order else str(num)))
            if not with_order:
                break
    L = pd.DataFrame(rows)
    return label_trials(L, with_order), L.label.values


def _rdm(G):
    d = np.diag(G)
    return d[:, None] + d[None] - 2 * G


def plot_components(path, with_order=True):
    """Model RDMs of the components (distances implied by each G)."""
    import matplotlib.pyplot as plt
    names, Gc, T = build_components(with_order)
    idx, _ = condition_order(with_order)
    fig, ax = plt.subplots(1, len(names), figsize=(2.3 * len(names), 2.6))
    for a, k, G in zip(ax, names, Gc):
        D = _rdm(G)[np.ix_(idx, idx)]
        a.imshow(D, cmap='viridis')
        a.set_title(f'{k}\n({np.linalg.matrix_rank(G)} dims)', fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f'Component RDMs, {len(T)} conditions, sorted by trial type',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_results(Gs, V, stats, with_order, path):
    """Group-average cross-validated RDM, component RDMs and component
    variances (mean +- SE, participants as dots, sign-flip p-values).
    Gs: participants x n x n cross-validated G; V: participants x components."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    names, Gc, T = build_components(with_order)
    n, k = len(T), len(names)
    idx, labels = condition_order(with_order)
    per_set = 4 if with_order else 2
    fig = plt.figure(figsize=(max(2.1 * k, 12), 9.5))
    gs = GridSpec(2, k, height_ratios=[2.4, 1], figure=fig)

    a = fig.add_subplot(gs[0, :k // 2])
    D = _rdm(Gs.mean(0))[np.ix_(idx, idx)] * 1e3
    im = a.imshow(D, cmap='viridis')
    for b in np.arange(per_set, n, per_set) - .5:
        a.axhline(b, color='w', lw=.6); a.axvline(b, color='w', lw=.6)
    a.set_xticks(range(n)); a.set_yticks(range(n))
    a.set_xticklabels(labels, fontsize=5 if with_order else 7, rotation=90)
    a.set_yticklabels(labels, fontsize=5 if with_order else 7)
    a.set_title(f'Cross-validated RDM (squared distance x1e-3), '
                f'mean of {len(Gs)} participants\nlines separate binding sets',
                fontsize=9)
    fig.colorbar(im, ax=a, fraction=.046, pad=.03)

    a = fig.add_subplot(gs[0, k // 2:])
    m, se = V.mean() * 1e3, V.sem() * 1e3
    x = np.arange(k)
    a.bar(x, m, yerr=se, color='0.75', edgecolor='k', capsize=3)
    rng = np.random.default_rng(0)
    for i, c in enumerate(V.columns):
        a.scatter(i + rng.uniform(-.2, .2, len(V)), V[c] * 1e3, s=8, color='k',
                  alpha=.5, zorder=3)
    top = (V * 1e3).max().max()
    for i, c in enumerate(V.columns):
        p = stats.loc[c, 'p perm FWE']
        star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        pu = stats.loc[c, 'p perm']
        a.text(i, top * 1.05, ('p<.001' if pu < .001 else f'p={pu:.3f}') + f'\n{star}',
               ha='center', fontsize=7)
    a.axhline(0, color='k', lw=.8)
    a.set_xticks(x); a.set_xticklabels(V.columns, rotation=45, ha='right', fontsize=8)
    a.set_ylabel('variance per condition (x1e-3)')
    a.set_ylim(top=top * 1.25 if top > 0 else None)
    a.set_title('Component variance (cross-validated), mean +- SE\n'
                'p: sign-flip permutation, one-sided; stars: FWE-corrected',
                fontsize=9)

    for i, (c, G) in enumerate(zip(names, Gc)):
        a = fig.add_subplot(gs[1, i])
        a.imshow(_rdm(G)[np.ix_(idx, idx)], cmap='viridis')
        a.set_title(f'{c} ({np.linalg.matrix_rank(G)} dims)', fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyse(subjects, ids, crossval=False, ceiling=False, n_null=0, tag='',
            orders=(False, True)):
    """subjects: list of (Y, trials) after infer_runs / prewhiten.
    orders: (False,) = 18 conditions only, (False, True) = 18 and 36."""
    for wo in orders:
        n = 36 if wo else 18
        name = f'{n}{tag}{"_cv" if crossval else "_aic"}'
        Gs = np.array([cv_G(Y, t, wo) for Y, t in subjects])
        V = pd.DataFrame([cv_component_variance(Y, t, wo, G)
                          for (Y, t), G in zip(subjects, Gs)],
                         index=pd.Index(ids, name='sub'))
        cvs = summarize_cv(V)
        print(f'\n{n} conditions: cross-validated component variance (eq. 14)')
        print(cvs.to_string())
        V.to_csv(f'pcm_cvvariance_{n}{tag}.csv')
        cvs.to_csv(f'pcm_cvvariance_summary_{n}{tag}.csv')
        np.savez(f'pcm_cvG_{n}{tag}.npz', G=Gs, sub=np.array(ids))
        plot_results(Gs, V, cvs, wo, f'pcm_results_{n}{tag}.png')

        data = [make_dataset(Y, t, wo) for Y, t in subjects]
        res = run_pcm(data, wo, crossval)
        summ = summarize_pcm(res)
        print(f'\n{n} conditions: model family, '
              f'{"group cross-validation" if crossval else "individual fits, AIC"}')
        print(summ.to_string())
        summ.to_csv(f'pcm_summary_{name}.csv')
        res['logBF'].set_index(pd.Index(ids, name='sub')).to_csv(f'pcm_logBF_{name}.csv')
        res['variance'].set_index(pd.Index(ids, name='sub')).to_csv(f'pcm_variance_{name}.csv')

        if ceiling and crossval:
            nc = noise_ceiling(subjects, data, res, wo)
            print('\nPseudo-R2 (0 = null model, 1 = upper noise ceiling):')
            print(nc.describe().loc[['mean', 'std']].round(3).to_string())
            nc.set_index(pd.Index(ids, name='sub')).to_csv(f'pcm_noiseceiling_{name}.csv')

    if n_null:
        print(f'\nBinding null, 18 conditions, {n_null} permutations')
        summ, null = binding_null(subjects, n_null)
        print(summ.to_string())
        summ.to_csv(f'pcm_bindingnull_summary{tag}.csv')
        null.to_csv(f'pcm_bindingnull_distribution{tag}.csv')


if __name__ == '__main__':
    import sys, glob, os, re
    args = sys.argv[1:]
    cv, ceil = 'cv' in args, 'ceiling' in args
    whiten = (None if 'nowhiten' in args else 'maps' if 'whiten=maps' in args
              else WHITEN)
    n_null = next((int(a.split('=')[1]) for a in args if a.startswith('null=')), 0)
    orders = (False,) if 'only18' in args else (False, True)
    check_design(False)
    check_design(True)
    plot_components('pcm_components_18.png', False)
    plot_components('pcm_components_36.png', True)

    if 'simulate' in args:
        truth = {'color': .01, 'motion': .03, 'set': .005, 'bind_conj': .01,
                 'color_order': .01}
        print('\ntrue variance per condition:', truth)
        subjects = []
        for Y, t in simulate(truth):
            t, k = infer_runs(Y, t)
            ok = pd.crosstab(t.run, t.run_true).gt(0).sum(1).eq(1).all()
            print(f'  runs recovered: {ok} (runs per block: {sorted(set(k))})')
            if whiten:              # no residuals in the simulation
                Y, _ = prewhiten(Y, t)
            subjects.append((Y, t))
        analyse(subjects, list(range(1, len(subjects) + 1)), cv, ceil, n_null, '_sim',
                orders=orders)

    if 'real' in args:
        check_trial_types()
        subs = SUBJECTS or sorted(
            int(re.fullmatch(r'sub-(\d+)', os.path.basename(d))[1])
            for d in glob.glob(BASE + '/firestimates/sub-*'))
        print('\nLoading tmap_z')
        subjects = [prepare_subject(s, 'tmap_z', whiten) for s in subs]
        analyse(subjects, subs, cv, ceil, n_null, orders=orders)
