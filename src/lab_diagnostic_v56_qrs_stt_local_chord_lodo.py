r"""
lab_diagnostic_v56_qrs_stt_local_chord_lodo.py
==============================================
CONTROLLED 6-HOSPITAL LODO ABLATION OF PHYSIOLOGICAL QRS-vs-ST-T & CAUSAL LOCAL-CHORD GEOMETRY
(STRICTLY CAUSAL, ZERO STAGE-1 CHANGES, ZERO LTSTDB, SVM + LogR, R1 + R3_Joint2D_PatFA)

Tests the exact pre-registered physiological-geometric invariants verified in task-14281:
  Layer h (Local-Normal Drift & Causal Local Chord):
    - C_drift           = -|log((d_proj + 0.5) / (||y||_2 + 0.5))|         [bounded <= 0]
    - log_loc_chord_snr = log((d_full_loc + 0.5) / (d_seg_loc + 0.5))      [0.919-0.936 across ALL 5 DBs!]
    - bridge_loc_qrs    = min(lam_loc_qrs, 1.0 - lam_loc_qrs)              [bounded <= 0.5; 0.899 vs 0F_FP, 0.814 vs 0F_m05!]
  Layer w (Depolarization QRS [71:123] vs Repolarization ST-T [123:187] Decomposition):
    - tanh_d_qrs        = tanh((d_qrs_loc / sqrt(104)) / 0.75)             [45-deg saturated QRS deviation]
    - tanh_d_stt        = tanh((d_stt_loc / sqrt(128)) / 0.75)             [45-deg saturated ST-T deviation; 0.892 vs 0F_m05, 0.980 edb!]
    - log_qrs_vs_nonqrs = log((d_qrs_loc + 0.5) / (d_nonqrs_loc + 0.5))    [0.758 on F_TP vs V_FP AND 0.809 on F_TP vs 0F_FP!]
    - d_v_qrs_norm      = ||(x - b_V)_QRS|| / ||(b_V - b_N_loc)_QRS||      [0.881 vs 0F_FP, 0.879 vs 0F_m05]
  Layer Phi (Tangent-Subspace Orthogonality & Fixed Pure-Beat Poles):
    - log_tan_orth      = log((d_proj + 0.5) / (d_par_U + 0.5))
    - PB_4              = [log_lam_gap, log_1m_lam, log_eV_any, z_V_same_fix]

Configs (6-DB LODO, SubjectAware_251, SVM + LogR):
  - B0_Baseline_C3       : Exact v49 C3 (bit-exact R1 parity asserted!)
  - Q1_LocalChord4_NS    : B0 + [C_drift, log_tan_orth, log_loc_chord_snr, bridge_loc_qrs] in F:N/S; F:V LOCKED
  - Q2_QRS_STT_NS        : Q1 + [tanh_d_qrs, tanh_d_stt, log_qrs_vs_nonqrs, d_v_qrs_norm] in F:N/S; F:V LOCKED
  - Q3_QRS_STT_Both      : Q2 features in BOTH F:N/S and F:V (+ log_preRR in F:V)
  - Q4_Compact6_Both     : B0 + [C_drift, log_tan_orth, log_loc_chord_snr, bridge_loc_qrs, tanh_d_stt, log_qrs_vs_nonqrs] in BOTH
  - Q5_PureBeat_QRS_Both : PB_4 + BASE_REST_7 + [C_drift, log_tan_orth, log_loc_chord_snr, bridge_loc_qrs, tanh_d_qrs, tanh_d_stt, log_qrs_vs_nonqrs] in BOTH

Calibration Rules (strictly source-OOF, ZERO target peeking, applied identically to B0 and Q1..Q5):
  - R1              : Exact v46/v48/v49 PooledPriorF1
  - R3_Joint2D_PatFA: Exact vectorized 2D grid maximization of source-OOF patient-level MacroRec(F+)
                      (+ 0.05 * PooledRec - 0.002 * MeanFAPH_All tie-break) subject to
                      MeanPatFAPH_All_src <= 2.0 AND MeanPatFAPH_0F_src <= 1.0.
"""


import os as _os, sys as _sys
from pathlib import Path as _Path
_SRC_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_DATA_ROOT = _Path(_os.environ.get("ECG_DATA_ROOT", str(_REPO_ROOT / "data_public_ecg")))
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import os
import sys
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import lab_diagnostic_v46_two_branch_stage2_rr as v46
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_audit_v49_subject_unit_and_registry as v49
import lab_diagnostic_v53b_lodo_disentanglement_ablation as v53b

AUD_DIR = str(_REPO_ROOT / "audits")
CACHE_V37 = str(_DATA_ROOT / "cache_v37_waveforms_6db")
DB_ORDER = v53b.DB_ORDER
F2_IDX = list(v47.F2_IDX)        # [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]
IDX_C3_V = list(v47.IDX_C3)      # F2_IDX + [18]
BASE_REST_7 = [3, 4, 5, 8, 10, 13, 14]

SL_PRE = slice(0, 71)
SL_QRS = v33.QRS_CORE_SLICE      # slice(71, 123) -> 52 samples * 2 leads = 104
SL_STT = slice(123, 187)         # 64 samples * 2 leads = 128
R_QRS = float(np.sqrt(2.0 * 52.0))
R_STT = float(np.sqrt(2.0 * 64.0))


def extract_v37_qrs_local_db(db_name, items):
    meta = {m["rec_id"]: m for m in json.load(open(os.path.join(CACHE_V37, f"{db_name}_meta.json"), "r", encoding="utf-8"))}
    beats_mm = np.load(os.path.join(CACHE_V37, f"{db_name}_beats.npy"), mmap_mode="r")
    bn_mm = np.load(os.path.join(CACHE_V37, f"{db_name}_bn_sub.npy"), mmap_mode="r")
    bv_mm = np.load(os.path.join(CACHE_V37, f"{db_name}_bv_anchors.npy"), mmap_mode="r")
    bk_mm = np.load(os.path.join(CACHE_V37, f"{db_name}_best_k.npy"), mmap_mode="r")

    out = {}
    errs = []
    for it in items:
        rid = it["rec_id"]
        s2_idx = np.where(it["s2_mask"])[0]
        if len(s2_idx) == 0:
            out[rid] = None
            continue
        m = meta[rid]
        x = np.asarray(beats_mm[m["w0"] + 1 + s2_idx], dtype=np.float64)
        nbn = m["bn1"] - m["bn0"]
        bn_i = np.minimum(s2_idx // 16, nbn - 1)
        bN = np.asarray(bn_mm[m["bn0"] + bn_i], dtype=np.float64)
        bk = np.asarray(bk_mm[m["b0"] + s2_idx], dtype=np.int64)
        bV = np.asarray(bv_mm[m["bv0"] + bk], dtype=np.float64)

        # Alignment assertion on sample beats
        nchk = min(30, len(x))
        c_chk = 0.0
        for ch in (0, 1):
            a = x[:nchk, ch] - x[:nchk, ch].mean(axis=1, keepdims=True)
            b = bN[:nchk, ch] - bN[:nchk, ch].mean(axis=1, keepdims=True)
            c_chk += np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        errs.append(float(np.median(np.abs(0.5 * c_chk - it["X_s2"][:nchk, 4]))))

        dl = x - bN
        dv = bV - bN
        dx_v = x - bV

        d_full = np.linalg.norm(dl.reshape(len(x), -1), axis=1)
        d_pre = np.linalg.norm(dl[:, :, SL_PRE].reshape(len(x), -1), axis=1)
        d_qrs = np.linalg.norm(dl[:, :, SL_QRS].reshape(len(x), -1), axis=1)
        d_stt = np.linalg.norm(dl[:, :, SL_STT].reshape(len(x), -1), axis=1)
        d_nonq = np.sqrt(d_pre ** 2 + d_stt ** 2)

        dl_q = dl[:, :, SL_QRS].reshape(len(x), -1)
        dv_q = dv[:, :, SL_QRS].reshape(len(x), -1)
        nv_q = np.linalg.norm(dv_q, axis=1) + 1e-6
        lam_qrs = np.sum(dl_q * dv_q, axis=1) / (nv_q ** 2)
        bridge_loc_qrs = np.minimum(lam_qrs, 1.0 - lam_qrs)
        d_v_qrs_norm = np.linalg.norm(dx_v[:, :, SL_QRS].reshape(len(x), -1), axis=1) / nv_q

        dl_f = dl.reshape(len(x), -1)
        dv_f = dv.reshape(len(x), -1)
        nv_f = np.linalg.norm(dv_f, axis=1) + 1e-6
        lam_loc = np.sum(dl_f * dv_f, axis=1) / (nv_f ** 2)
        dseg_loc = np.linalg.norm(dl_f - np.clip(lam_loc, 0.0, 1.0)[:, None] * dv_f, axis=1)
        log_loc_chord_snr = np.log((d_full + 0.5) / (dseg_loc + 0.5))

        tanh_d_qrs = np.tanh((d_qrs / R_QRS) / 0.75)
        tanh_d_stt = np.tanh((d_stt / R_STT) / 0.75)
        log_qrs_vs_nonqrs = np.log((d_qrs + 0.5) / (d_nonq + 0.5))

        out[rid] = np.column_stack([
            log_loc_chord_snr,
            bridge_loc_qrs,
            tanh_d_qrs,
            tanh_d_stt,
            log_qrs_vs_nonqrs,
            np.log1p(np.maximum(0.0, d_v_qrs_norm)),
        ]).astype(np.float64)
    assert float(np.median(errs)) < 0.02, f"v37 alignment failed on {db_name}: {np.median(errs)}"
    print(f"  [v37 aligned OK] {db_name:8s}: median corr err = {np.median(errs):.5f}", flush=True)
    return out


def augment_dbs_v56(dbs_base, g16_by_db, q6_by_db):
    """
    Appends 12 columns (indices 24..35) to X_s2:
      24: log_lam_gap        = log1p(max(0, lam - mu_N) / 0.05)
      25: log_1m_lam         = log1p(max(0, 1 - lam) / 0.12)
      26: log_eV_any         = log1p(max(0, e_V_any) / 0.12)
      27: z_V_same_fix       = min(log1p(e_Vc / rad_V), log1p(e_V_same / rad_V))
      28: C_drift            = -|log((d_proj + 0.5) / (||y||_2 + 0.5))|
      29: log_tan_orth       = log((d_proj + 0.5) / (d_par_U + 0.5))
      30: log_loc_chord_snr  = log((d_full_loc + 0.5) / (d_seg_loc + 0.5))
      31: bridge_loc_qrs     = min(lam_loc_qrs, 1.0 - lam_loc_qrs)
      32: tanh_d_qrs         = tanh((d_qrs_loc / sqrt(104)) / 0.75)
      33: tanh_d_stt         = tanh((d_stt_loc / sqrt(128)) / 0.75)
      34: log_qrs_vs_nonqrs  = log((d_qrs_loc + 0.5) / (d_nonqrs_loc + 0.5))
      35: log_dv_qrs_norm    = log1p(d_v_qrs_norm)
    """
    out = {}
    for db in DB_ORDER:
        items = []
        for it in dbs_base[db]:
            c = dict(it)
            if len(it["X_s2"]) > 0:
                g = g16_by_db[db][it["rec_id"]].astype(np.float64)
                q6 = q6_by_db[db][it["rec_id"]]
                lam = g[:, 0]; muN = g[:, 1]; rV = g[:, 3]; nq = g[:, 5]
                eVc = g[:, 6]; eVs = g[:, 7]; eVa = g[:, 8]
                dp = g[:, 10]; df = g[:, 11]; ds = g[:, 12]
                dpar = np.sqrt(np.maximum(0.0, df ** 2 - dp ** 2))
                normy = np.sqrt((np.clip(lam, 0.0, 1.0) * nq) ** 2 + ds ** 2)
                log_rlp = np.log((dp + 0.5) / (normy + 0.5))

                f24 = np.log1p(np.maximum(0.0, lam - muN) / 0.05)
                f25 = np.log1p(np.maximum(0.0, 1.0 - lam) / 0.12)
                f26 = np.log1p(np.maximum(0.0, eVa) / 0.12)
                f27 = np.minimum(np.log1p(np.maximum(0.0, eVc) / np.maximum(rV, 1e-6)),
                                 np.log1p(np.maximum(0.0, eVs) / np.maximum(rV, 1e-6)))
                f28 = -np.abs(log_rlp)
                f29 = np.log((dp + 0.5) / (dpar + 0.5))

                extra = np.column_stack([f24, f25, f26, f27, f28, f29, q6])
                c["X_s2"] = np.column_stack([it["X_s2"].astype(np.float64), extra])
            else:
                c["X_s2"] = np.zeros((0, it["X_s2"].shape[1] + 12), dtype=np.float64)
            items.append(c)
        out[db] = items
    return out


PB_4 = [24, 25, 26, 27]
LC_4 = [28, 29, 30, 31]                     # C_drift, log_tan_orth, log_loc_chord_snr, bridge_loc_qrs
QS_8 = [28, 29, 30, 31, 32, 33, 34, 35]     # LC_4 + tanh_d_qrs, tanh_d_stt, log_qrs_vs_nonqrs, log_dv_qrs_norm
CP_6 = [28, 29, 30, 31, 33, 34]             # Compact 6: LC_4 + tanh_d_stt, log_qrs_vs_nonqrs
PB_Q = PB_4 + BASE_REST_7 + [28, 29, 30, 31, 32, 33, 34]

CONFIGS = {
    "B0_Baseline_C3":       (F2_IDX,         IDX_C3_V),
    "Q1_LocalChord4_NS":    (F2_IDX + LC_4,  IDX_C3_V),
    "Q2_QRS_STT8_NS":       (F2_IDX + QS_8,  IDX_C3_V),
    "Q3_QRS_STT8_Both":     (F2_IDX + QS_8,  IDX_C3_V + QS_8),
    "Q4_Compact6_Both":     (F2_IDX + CP_6,  IDX_C3_V + CP_6),
    "Q5_PureBeat_QRS_Both": (PB_Q,           PB_Q + [18]),
}
CFG_ORDER = list(CONFIGS.keys())
RULES = ("R1", "R3")


def solve_r3_joint2d_patfa(sN, sV, y_cat, pat_idx, src_pat_hours, src_pat_totF, b_tot=2.0, b_0f=1.0, n_grid=85):
    """
    Exact vectorized 2D grid solver on source OOF:
      Maximizes MacroPatRec(F+) + 0.05 * PooledRec - 0.002 * MeanPatFAPH_All
      subject to:
        MeanPatFAPH_All_src(tau_N, tau_V) <= b_tot
        MeanPatFAPH_0F_src(tau_N, tau_V)  <= b_0f
    Uses 2D reverse cumulative sums over per-patient binned counts -> exact & < 30 ms!
    """
    n_pats = len(src_pat_hours)
    is_f = (y_cat == 2)
    fpos_mask = (src_pat_totF > 0)
    zpos_mask = (src_pat_totF == 0)
    n_fpos = max(1, int(np.sum(fpos_mask)))
    n_zpos = max(1, int(np.sum(zpos_mask)))

    # Construct informative 1D grids covering F quantiles + high Non-F tail quantiles
    if np.sum(is_f) > 10:
        gN = np.unique(np.concatenate([
            np.quantile(sN[is_f], np.linspace(0.02, 0.99, n_grid // 2)),
            np.quantile(sN[~is_f], np.linspace(0.90, 0.9999, n_grid - n_grid // 2)),
        ]))
        gV = np.unique(np.concatenate([
            np.quantile(sV[is_f], np.linspace(0.02, 0.99, n_grid // 2)),
            np.quantile(sV[~is_f], np.linspace(0.70, 0.9995, n_grid - n_grid // 2)),
        ]))
    else:
        gN = np.unique(np.quantile(sN, np.linspace(0.5, 0.999, n_grid)))
        gV = np.unique(np.quantile(sV, np.linspace(0.5, 0.999, n_grid)))

    GN, GV = len(gN), len(gV)
    # Bin each beat to the largest grid index j such that score >= g[j] (i.e. searchsorted(..., side='right') - 1)
    # Beats below g[0] are excluded from grid>=0 counts (safe because g[0] is below 2nd percentile of F)
    bN = np.searchsorted(gN, sN, side="right") - 1
    bV = np.searchsorted(gV, sV, side="right") - 1
    in_g = (bN >= 0) & (bV >= 0)

    # Weight for each beat:
    #  - If F: contributes 1 / (totF[p] * n_fpos) to MacroRec, and 1 / tot_F_src to PooledRec
    #  - If Non-F: contributes 1 / (hours[p] * n_pats) to MeanFAPH_All,
    #              and (if pat is 0-F) 1 / (hours[p] * n_zpos) to MeanFAPH_0F
    bN_i = bN[in_g]; bV_i = bV[in_g]; y_i = y_cat[in_g]; p_i = pat_idx[in_g]
    f_m = (y_i == 2)
    nf_m = ~f_m

    tot_f_src = max(1.0, float(np.sum(src_pat_totF)))
    w_macro = np.zeros(len(y_i), dtype=np.float64)
    w_pool = np.zeros(len(y_i), dtype=np.float64)
    w_fall = np.zeros(len(y_i), dtype=np.float64)
    w_f0f = np.zeros(len(y_i), dtype=np.float64)

    w_macro[f_m] = 1.0 / (np.maximum(1.0, src_pat_totF[p_i[f_m]].astype(np.float64)) * float(n_fpos))
    w_pool[f_m] = 1.0 / tot_f_src
    w_fall[nf_m] = 1.0 / (np.maximum(1e-6, src_pat_hours[p_i[nf_m]]) * float(n_pats))
    z_nf = nf_m & zpos_mask[p_i]
    w_f0f[z_nf] = 1.0 / (np.maximum(1e-6, src_pat_hours[p_i[z_nf]]) * float(n_zpos))

    def grid_suffix_2d(weights):
        M = np.zeros((GN, GV), dtype=np.float64)
        np.add.at(M, (bN_i, bV_i), weights)
        return np.cumsum(np.cumsum(M[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]

    S_macro = grid_suffix_2d(w_macro)
    S_pool = grid_suffix_2d(w_pool)
    S_fall = grid_suffix_2d(w_fall)
    S_f0f = grid_suffix_2d(w_f0f)

    feas = (S_fall <= b_tot + 1e-12) & (S_f0f <= b_0f + 1e-12)
    if not np.any(feas):
        # Fallback: minimize excess FAPH
        bi, bj = np.unravel_index(np.argmin(S_fall + S_f0f), S_fall.shape)
        return float(gN[bi]), float(gV[bj])

    obj = np.where(feas, S_macro + 0.05 * S_pool - 0.002 * S_fall, -1e9)
    bi, bj = np.unravel_index(int(np.argmax(obj)), obj.shape)
    return float(gN[bi]), float(gV[bj])


def fit_fold_v56(tr_items, mtype, cap_pat_pos=100):
    pat_map = {}
    for it in tr_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: (-sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]), p))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    pat_to_idx = {p: i for i, p in enumerate(pat_list)}
    n_src_pats = len(pat_list)
    src_pat_hours = np.zeros(n_src_pats, dtype=np.float64)
    src_pat_totF = np.zeros(n_src_pats, dtype=np.int64)
    for p, items in pat_map.items():
        pi = pat_to_idx[p]
        src_pat_hours[pi] = sum(it["post_hours"] for it in items)
        src_pat_totF[pi] = sum(int(np.sum(it["y"] == 2)) for it in items)

    tot_b = sum(len(it["y"]) for it in tr_items)
    tot_f = int(np.sum(src_pat_totF))

    uniq_N = sorted({tuple(v[0]) for v in CONFIGS.values()})
    uniq_V = sorted({tuple(v[1]) for v in CONFIGS.values()})

    oof_sN = {k: [] for k in uniq_N}
    oof_sV = {k: [] for k in uniq_V}
    oof_y, oof_pi = [], []

    for fid in range(n_folds):
        tr_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] == fid]

        clfs_N = {}
        for fset in uniq_N:
            XN, yN, wN = v46.build_branch_xy(tr_sub, list(fset), {0, 3}, cap_pat_pos=cap_pat_pos)
            sc = StandardScaler(); clf = v46.make_clf(mtype); clf.fit(sc.fit_transform(XN), yN, sample_weight=wN)
            clfs_N[fset] = (sc, clf)

        clfs_V = {}
        for fset in uniq_V:
            XV, yV, wV = v46.build_branch_xy(tr_sub, list(fset), {1}, cap_pat_pos=cap_pat_pos)
            sc = StandardScaler(); clf = v46.make_clf(mtype); clf.fit(sc.fit_transform(XV), yV, sample_weight=wV)
            clfs_V[fset] = (sc, clf)

        for te in te_sub:
            if len(te["X_s2"]) > 0:
                for fset, (sc, clf) in clfs_N.items():
                    oof_sN[fset].append(clf.decision_function(sc.transform(te["X_s2"][:, list(fset)])))
                for fset, (sc, clf) in clfs_V.items():
                    oof_sV[fset].append(clf.decision_function(sc.transform(te["X_s2"][:, list(fset)])))
                oof_y.append(te["y"][te["s2_mask"]].astype(np.int8))
                oof_pi.append(np.full(len(te["X_s2"]), pat_to_idx[te["patient_id"]], dtype=np.int32))

    full_N, full_V, sig_N, sig_V = {}, {}, {}, {}
    for fset in uniq_N:
        XN, yN, wN = v46.build_branch_xy(tr_items, list(fset), {0, 3}, cap_pat_pos=cap_pat_pos)
        sc = StandardScaler(); clf = v46.make_clf(mtype); clf.fit(sc.fit_transform(XN), yN, sample_weight=wN)
        full_N[fset] = (sc, clf)
        oof_sN[fset] = np.concatenate(oof_sN[fset])
        sig_N[fset] = float(np.std(oof_sN[fset])) + 1e-9

    for fset in uniq_V:
        XV, yV, wV = v46.build_branch_xy(tr_items, list(fset), {1}, cap_pat_pos=cap_pat_pos)
        sc = StandardScaler(); clf = v46.make_clf(mtype); clf.fit(sc.fit_transform(XV), yV, sample_weight=wV)
        full_V[fset] = (sc, clf)
        oof_sV[fset] = np.concatenate(oof_sV[fset])
        sig_V[fset] = float(np.std(oof_sV[fset])) + 1e-9

    y_cat = np.concatenate(oof_y)
    y_bin = (y_cat == 2).astype(int)
    pi_cat = np.concatenate(oof_pi)

    out = {}
    for cfg in CFG_ORDER:
        fN = tuple(CONFIGS[cfg][0])
        fV = tuple(CONFIGS[cfg][1])
        sN = oof_sN[fN]; sV = oof_sV[fV]

        tN_r1, tV_r1, _ = v46.joint_oof_2d_threshold(sN, sV, y_bin, tot_f, tot_b, n_grid=140)
        tN_r3, tV_r3 = solve_r3_joint2d_patfa(sN, sV, y_cat, pi_cat, src_pat_hours, src_pat_totF, b_tot=2.0, b_0f=1.0, n_grid=85)

        out[cfg] = {
            "fN": list(fN), "fV": list(fV),
            "scN": full_N[fN][0], "clfN": full_N[fN][1], "sigN": sig_N[fN],
            "scV": full_V[fV][0], "clfV": full_V[fV][1], "sigV": sig_V[fV],
            "R1": (float(tN_r1), float(tV_r1)),
            "R3": (float(tN_r3), float(tV_r3)),
        }
    return out


def main():
    t0 = time.time()
    reg = json.load(open(v53b.REG_PATH, "r", encoding="utf-8"))
    rec_to_sid = {(r["db"], r["rec_id"]): r["verified_subject_id"] for r in reg["records"]}

    dbs_raw = {db: v47.build_or_load_v47_db(db) for db in DB_ORDER}
    dbs_base = v49.clone_dbs_with_mapping(dbs_raw, rec_to_sid)
    g16_by_db = v53b.build_or_load_v53_cache(dbs_base)
    q6_by_db = {db: extract_v37_qrs_local_db(db, dbs_base[db]) for db in DB_ORDER}
    dbs = augment_dbs_v56(dbs_base, g16_by_db, q6_by_db)

    audit = {
        "Audit_Version": "v56_qrs_stt_local_chord_lodo",
        "Date": "2026-11-23",
        "Scope": "6 source DBs (306 recs / 251 SubjectAware_251 units), strict 6-hospital LODO, SVM + LogR, R1 + R3_Joint2D_PatFA, Zero LTSTDB.",
        "Configs": {k: {"N": v[0], "V": v[1]} for k, v in CONFIGS.items()},
        "Models": {},
    }

    for mtype in ("SVM", "LogR"):
        print(f"\n==================== v56 QRS-vs-STT & LOCAL-CHORD LODO ({mtype}) ====================", flush=True)
        unit_data = {}
        for db in DB_ORDER:
            for it in dbs[db]:
                sid = it["patient_id"]
                if sid not in unit_data:
                    unit_data[sid] = {
                        "db": db, "tot_F": 0, "hours": 0.0, "ys2": [], "n_s2": 0,
                        **{f"mN_{rn}_{cfg}": [] for rn in RULES for cfg in CFG_ORDER},
                        **{f"mV_{rn}_{cfg}": [] for rn in RULES for cfg in CFG_ORDER},
                        **{f"Smin_{rn}_{cfg}": [] for rn in RULES for cfg in CFG_ORDER},
                    }
                ud = unit_data[sid]
                ud["tot_F"] += int(np.sum(it["y"] == 2))
                ud["hours"] += float(it["post_hours"])

        fold_meta = {}
        for tgt in DB_ORDER:
            tr_items = [it for db in DB_ORDER if db != tgt for it in dbs[db]]
            te_items = dbs[tgt]
            fitted = fit_fold_v56(tr_items, mtype)
            fold_meta[tgt] = {
                cfg: {
                    "R1": [round(x, 4) for x in fitted[cfg]["R1"]],
                    "R3": [round(x, 4) for x in fitted[cfg]["R3"]],
                    "beta_N": [round(float(x), 4) for x in fitted[cfg]["clfN"].coef_[0]],
                    "beta_V": [round(float(x), 4) for x in fitted[cfg]["clfV"].coef_[0]],
                }
                for cfg in CFG_ORDER
            }
            for it in te_items:
                if len(it["X_s2"]) == 0:
                    continue
                ud = unit_data[it["patient_id"]]
                ud["ys2"].append(it["y"][it["s2_mask"]].astype(np.int8))
                for cfg in CFG_ORDER:
                    cc = fitted[cfg]
                    sN = cc["clfN"].decision_function(cc["scN"].transform(it["X_s2"][:, cc["fN"]]))
                    sV = cc["clfV"].decision_function(cc["scV"].transform(it["X_s2"][:, cc["fV"]]))
                    for rn in RULES:
                        tN, tV = cc[rn]
                        mN = (sN - tN) / cc["sigN"]
                        mV = (sV - tV) / cc["sigV"]
                        ud[f"mN_{rn}_{cfg}"].append(mN)
                        ud[f"mV_{rn}_{cfg}"].append(mV)
                        ud[f"Smin_{rn}_{cfg}"].append(np.minimum(mN, mV))

        for sid, ud in unit_data.items():
            if ud["ys2"]:
                ud["ys2"] = np.concatenate(ud["ys2"])
                ud["n_s2"] = len(ud["ys2"])
                for cfg in CFG_ORDER:
                    for rn in RULES:
                        ud[f"mN_{rn}_{cfg}"] = np.concatenate(ud[f"mN_{rn}_{cfg}"])
                        ud[f"mV_{rn}_{cfg}"] = np.concatenate(ud[f"mV_{rn}_{cfg}"])
                        ud[f"Smin_{rn}_{cfg}"] = np.concatenate(ud[f"Smin_{rn}_{cfg}"])

        ev_b0_r1 = v53b.eval_at_score_threshold(unit_data, "Smin_R1_B0_Baseline_C3", 0.0)
        p2_b0_r1 = v53b.per_patient_patf2_macro(unit_data, "Smin_R1_B0_Baseline_C3")
        if mtype == "SVM":
            assert ev_b0_r1["TP"] == 1099 and ev_b0_r1["FP_NS"] == 2058 and ev_b0_r1["FP_V"] == 1406, ev_b0_r1
        else:
            assert ev_b0_r1["TP"] == 1169 and ev_b0_r1["FP_NS"] == 2330 and ev_b0_r1["FP_V"] == 1437, ev_b0_r1
        print(f"  [B0 R1 Parity OK] TP={ev_b0_r1['TP']}, FP_NS={ev_b0_r1['FP_NS']}, FP_V={ev_b0_r1['FP_V']}, MacroRec={ev_b0_r1['MacroRec']}%, FAPH={ev_b0_r1['MeanFAPH_All']}/{ev_b0_r1['MeanFAPH_0F']}", flush=True)

        model_res = {"fold_meta": fold_meta}
        for rn in RULES:
            model_res[rn] = {}
            ev_b0_same = v53b.eval_at_score_threshold(unit_data, f"Smin_{rn}_B0_Baseline_C3", 0.0)
            print(f"\n  === RULE {rn}: DEPLOYED T1 (WTL vs B0_R1 | also vs B0_{rn}) ===", flush=True)
            for cfg in CFG_ORDER:
                ev = v53b.eval_at_score_threshold(unit_data, f"Smin_{rn}_{cfg}", 0.0)
                wtl_r1 = v53b.compare_wtl(ev["per_unit"], ev_b0_r1["per_unit"])
                wtl_sm = v53b.compare_wtl(ev["per_unit"], ev_b0_same["per_unit"])
                all_y = np.concatenate([u["ys2"] for u in unit_data.values() if u["n_s2"] > 0])
                all_sc = np.concatenate([u[f"Smin_{rn}_{cfg}"] for u in unit_data.values() if u["n_s2"] > 0])
                ap_s2 = float(average_precision_score((all_y == 2).astype(int), all_sc))
                p2 = v53b.per_patient_patf2_macro(unit_data, f"Smin_{rn}_{cfg}")
                w2 = sum(1 for s in p2 if p2[s] > p2_b0_r1[s] + 1e-12)
                l2 = sum(1 for s in p2 if p2[s] < p2_b0_r1[s] - 1e-12)
                t2 = len(p2) - w2 - l2

                per_db = {}
                for db in DB_ORDER:
                    sub_u = {s: u for s, u in unit_data.items() if u["db"] == db}
                    edb = v53b.eval_at_score_threshold(sub_u, f"Smin_{rn}_{cfg}", 0.0)
                    ys_d = np.concatenate([u["ys2"] for u in sub_u.values() if u["n_s2"] > 0])
                    sc_d = np.concatenate([u[f"Smin_{rn}_{cfg}"] for u in sub_u.values() if u["n_s2"] > 0])
                    ap_d = float(average_precision_score((ys_d == 2).astype(int), sc_d)) if np.any(ys_d == 2) else 0.0
                    per_db[db] = {
                        "TP": edb["TP"], "FP_NS": edb["FP_NS"], "FP_V": edb["FP_V"],
                        "MacroRec": edb["MacroRec"], "MeanFAPH": edb["MeanFAPH_All"],
                        "MeanFAPH_0F": edb["MeanFAPH_0F"], "AP_s2": round(ap_d, 4),
                    }

                sw_all = v53b.matched_budget_sweep(unit_data, f"Smin_{rn}_{cfg}", None, [1.0, 1.5, 2.0, 2.5, 3.0], mode="all")
                sw_0f = v53b.matched_budget_sweep(unit_data, f"Smin_{rn}_{cfg}", None, [0.5, 1.0, 1.5], mode="0f")

                model_res[rn][cfg] = {
                    "T1": {k: v for k, v in ev.items() if k != "per_unit"},
                    "vs_B0_R1": wtl_r1,
                    f"vs_B0_{rn}": wtl_sm,
                    "AP_s2": round(ap_s2, 4),
                    "PatF2_MacroRec": round(100.0 * float(np.mean(list(p2.values()))), 2),
                    "PatF2_vs_B0R1_WTL": f"{w2}W/{t2}T/{l2}L",
                    "per_db_T1": per_db,
                    "matched_Smin_AllFAPH": sw_all,
                    "matched_Smin_0FFAPH": sw_0f,
                }
                print(
                    f"    {cfg:21s} | TP={ev['TP']:4d} NS={ev['FP_NS']:4d} V={ev['FP_V']:4d} | "
                    f"Rec={ev['MacroRec']:5.2f}% ({wtl_r1['Fpos_WTL']}, SD:{wtl_r1['StrictDom_Fpos']}) | "
                    f"FAPH={ev['MeanFAPH_All']:.3f}/{ev['MeanFAPH_0F']:.3f} (<=2:{ev['Pct_le_2FAPH']}%, p95:{ev['p95_FAPH_All']}) | "
                    f"0F:{wtl_r1['ZeroF_DecTieInc']} | AP={ap_s2:.4f} | P@F2={model_res[rn][cfg]['PatF2_MacroRec']}%({model_res[rn][cfg]['PatF2_vs_B0R1_WTL']})",
                    flush=True,
                )

            print(f"\n  --- {rn} MATCHED S_min SWEEP (MeanFAPH_All) ---", flush=True)
            for B in ["1.0", "1.5", "2.0", "2.5"]:
                line = f"    @All<={B}: "
                for cfg in CFG_ORDER:
                    r = model_res[rn][cfg]["matched_Smin_AllFAPH"][B]
                    line += f"{cfg[:2]}={r['MacroRec']:5.2f}%(TP={r['TP']:4d},0F={r['MeanFAPH_0F']:.2f},V={r['FP_V']:4d}) | "
                print(line, flush=True)

            print(f"\n  --- {rn} MATCHED S_min SWEEP (True-0F MeanFAPH) ---", flush=True)
            for B in ["0.5", "1.0", "1.5"]:
                line = f"    @0F <={B}: "
                for cfg in CFG_ORDER:
                    r = model_res[rn][cfg]["matched_Smin_0FFAPH"][B]
                    line += f"{cfg[:2]}={r['MacroRec']:5.2f}%(TP={r['TP']:4d},All={r['MeanFAPH_All']:.2f},V={r['FP_V']:4d}) | "
                print(line, flush=True)

            print(f"\n  --- {rn} PER-DB T1 BREAKDOWN ---", flush=True)
            for db in DB_ORDER:
                line = f"    {db:8s}: "
                for cfg in CFG_ORDER:
                    r = model_res[rn][cfg]["per_db_T1"][db]
                    line += f"{cfg[:2]}[TP={r['TP']:3d},NS={r['FP_NS']:4d},V={r['FP_V']:3d},R={r['MacroRec']:4.1f}%,F={r['MeanFAPH']:5.2f},AP={r['AP_s2']:.3f}] | "
                print(line, flush=True)

        audit["Models"][mtype] = model_res

    jpath = os.path.join(AUD_DIR, "V56_QRS_STT_LOCAL_CHORD_LODO_20261123.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[v56] Saved full JSON -> {jpath} (total {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
