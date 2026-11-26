r"""
lab_diagnostic_v43_hypB2_representation_vs_veto.py
==================================================
HYPOTHESIS B2 — SEPARATING REPRESENTATION RIGHT FROM HARD STAGE-1 VETO RIGHT
FOR NEW V-ANCHORS (k >= 5) AT K_max = 12 (ZERO NEW PHYSIOLOGICAL GATES!)

Controlled Single-Factor Experiment (Identical K=12 anchor library C00_K12_Base,
identical Static v36 N-chart, zero anchor guards, zero labels in anchor creation,
and Stage 2 RETRAINED + source-locked T1 recalibrated on each configuration's own
source candidate pool using the EXACT same classifier family & OOF threshold procedure!):

Configurations compared across all 6 hospitals (306 recs, 2.68M beats, 4,600 F):
  - K4_Base                   : Independent Static v36 baseline (K_max = 4) [reference]
  - M0_HardVeto_K12           : K_max = 12; new anchors (k >= 5) BOTH represent AND hard-veto at Stage 1 (S2 = S2_K12)
  - M1_SoftPass_Same18D       : Same K_max = 12 library; beats vetoed ONLY by k >= 5 (Fwd = S2_core4 \ S2_K12)
                                are FORWARDED to Stage 2 (S2_soft = S2_K12 U S2_core4) using exact same 18D K=12 features
                                (with pyth_loc & min_thales properly computed for all t in S2_soft)
  - M2_SoftPass_UnclippedGeom : Same K_max = 12 library & SAME S2_soft pool; prevents z_V=0 clipping & bridge overwrite
                                for forwarded beats by preserving core (k<=4) bridge coordinates on Fwd and adding
                                3 purely geometric O(4)-invariant k>=5 proximity scalars (new_dy_ratio, new_dlam_ratio,
                                new_veto_flag) — ZERO new physiological features or gates!

Mandatory Telemetry:
  1. Exact Stage-1 forwarding budget per DB & spotlight record: Fwd_F, Fwd_V, Fwd_N, Fwd_S, V:F ratio.
  2. Exact fate of forwarded beats at deployed T1: Fwd_to_TP (F rescued) vs Fwd_to_FPV (V leaked).
  3. Matched false-alarm recall BOTH pooled AND per-patient:
       - Pooled Rank_Rec@F2 (DB FAPH <= 2.0) & MacroPat_Rec@DB_F2
       - Per-patient matched FA recall: MacroPat_Rec@PatF2 (each F+ patient evaluated at <= 2.0 FAPH on that patient)
       - Per-patient recall at M0_HardVeto's exact T1 FP count per patient (MacroPat_Rec@M0_FP)
  4. Separability diagnostic directly INSIDE Fwd (AUROC of Stage-2 score & key single features on Fwd_F vs Fwd_V).
  5. Exact O(4) gauge invariance verified < 1e-12 for M0, M1, M2.
  6. Apples-to-apples PostBirth ("chi dung neo sau thoi diem sinh", end-of-tape V-radii) check alongside FullTape.
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
import glob
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ortho_group
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import lab_diagnostic_v36_6hospital_lodo_benchmark as v36
import lab_diagnostic_v42b_hypB1_anchor_guards_2x2 as v42b

AUD_DIR = str(_REPO_ROOT / "audits")
CACHE_V42B = str(_DATA_ROOT / "cache_v42b_guards_2x2_6db")
CACHE_V43 = str(_DATA_ROOT / "cache_v43_softpass_6db")
os.makedirs(AUD_DIR, exist_ok=True)
os.makedirs(CACHE_V43, exist_ok=True)

DB_ORDER = ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]
CORE_K = 4
K_CAP = 12

MODELS_18 = {
    "F2_RefFrame10D_SVM":   ([0, 1, 2, 3, 4, 5, 8, 10, 13, 14], "SVM"),
    "G3_FullCtx0b17D_LogR": (list(range(12)) + [13, 14, 15, 16, 17], "LogR"),
}
MODELS_21 = {
    "F2_RefFrame10D_SVM":   ([0, 1, 2, 3, 4, 5, 8, 10, 13, 14, 18, 19, 20], "SVM"),
    "G3_FullCtx0b17D_LogR": (list(range(12)) + [13, 14, 15, 16, 17, 18, 19, 20], "LogR"),
}

SPOTLIGHT_RECS = {
    "ltdb": ["14134", "14046", "15814", "14149"],
    "mitdb": ["233", "223", "213", "208"],
    "edb": ["e0605", "e0614", "e0602"],
    "sddb": ["31", "49", "45"],
    "incartdb": ["I75", "I62"],
}


def project_subset_features(
    post_joint, y_mat, b_N, U, b_N_loc_hist,
    loc_norm_corr_arr, loc_norm_dist_proj_arr,
    preqrs_corr_N, rr_dev_pre, sqi_hf_ratio, zeno_qiso_drop_all,
    pref_hf_p95, pref_bcorr_p05,
    anchors, anchor_meta, active_cols, eval_mask_override=None, post_birth=False,
):
    """
    Computes exact 18D features using ONLY anchors[active_cols].
    Crucially, if eval_mask_override is provided (e.g. S2_soft), computes
    pyth_margin_loc and min_lead_thales_loc for EVERY beat in eval_mask_override
    (avoiding dummy -1.0 values for forwarded beats!).
    Returns: ready, s2_native, X18_full_N (N x 18), lam_all (N x len(active_cols)), dy_all, rady_vec, radv_vec.
    """
    N = len(post_joint)
    bN_0, bN_1 = b_N[:187], b_N[187:]
    ready = np.zeros(N, dtype=bool)
    d_seg_arr = np.full(N, 9.99)
    z_N_arr = np.zeros(N)
    z_V_arr = np.zeros(N)
    m_bridge_arr = np.zeros(N)
    min_lead_bcorr_arr = np.zeros(N)
    loc_dist_norm_arr = np.zeros(N)
    pyth_margin_loc = np.full(N, -1.0)
    min_lead_thales_loc = np.full(N, -1.0)

    cols = list(active_cols)
    K_sub = len(cols)
    if K_sub > 0:
        sub_ancs = [anchors[c] for c in cols]
        sub_meta = [anchor_meta[c] for c in cols]
        Q_mat = np.vstack([a["q"] for a in sub_ancs])
        nq_vec = np.array([a["norm_q"] for a in sub_ancs], dtype=np.float64)
        muN_vec = np.array([a["mu_N"] for a in sub_ancs], dtype=np.float64)
        radN_vec = np.array([a["rad_N"] for a in sub_ancs], dtype=np.float64)
        radV_vec = np.array([a["rad_V"] for a in sub_ancs], dtype=np.float64)
        radyV_vec = np.array([a["rad_y_V"] for a in sub_ancs], dtype=np.float64)
        disc_vec = np.array([am["disc_idx"] for am in sub_meta], dtype=np.int64)
        bV_stack = np.vstack([a["b_V"] for a in sub_ancs])

        lam_all = (y_mat @ Q_mat.T) / (nq_vec[None, :] ** 2 + 1e-12)
        lam_c_all = np.clip(lam_all, 0.0, 1.0)
        es_all = np.full((N, K_sub), 1e9, dtype=np.float64)
        ej_all = np.full((N, K_sub), 1e9, dtype=np.float64)
        t_vec = np.arange(N)

        for k in range(K_sub):
            qk = Q_mat[k]
            es = np.linalg.norm(y_mat - lam_c_all[:, k:k+1] * qk[None, :], axis=1)
            dy_q = np.linalg.norm(y_mat - qk[None, :], axis=1)
            ej = np.maximum(0.0, (dy_q - radyV_vec[k]) / (nq_vec[k] + 1e-12))
            if post_birth:
                ok_k = (t_vec >= disc_vec[k])
                es_all[:, k] = np.where(ok_k, es, 1e9)
                ej_all[:, k] = np.where(ok_k, ej, 1e9)
            else:
                es_all[:, k] = es
                ej_all[:, k] = ej

        ready = (np.min(es_all, axis=1) < 1e8)
        if np.any(ready):
            bk = np.argmin(es_all, axis=1)
            idx_n = np.arange(N)
            bl = lam_all[idx_n, bk]
            blc = lam_c_all[idx_n, bk]
            d_seg_arr = np.where(ready, es_all[idx_n, bk] / (nq_vec[bk] + 1e-12), 9.99)
            e_V_any = np.min(ej_all, axis=1)
            e_N = np.maximum(0.0, (bl - muN_vec[bk]) - radN_vec[bk])
            e_Vc = np.maximum(0.0, (1.0 - bl) - radV_vec[bk])
            zN = np.log1p(e_N / radN_vec[bk])
            zV = np.minimum(np.log1p(e_Vc / radV_vec[bk]), np.log1p(e_V_any / radV_vec[bk]))
            z_N_arr = np.where(ready, zN, 0.0)
            z_V_arr = np.where(ready, zV, 0.0)
            m_bridge_arr = np.where(ready, np.minimum(zN, zV), 0.0)
            loc_dist_norm_arr = np.where(ready, loc_norm_dist_proj_arr / (nq_vec[bk] + 1e-12), 0.0)

            for k in range(K_sub):
                mk = (bk == k) & ready
                if not np.any(mk): continue
                bV0, bV1 = bV_stack[k, :187], bV_stack[k, 187:]
                lc_k = blc[mk, None]
                hat0 = bN_0[None, :] + lc_k * (bV0 - bN_0)[None, :]
                hat1 = bN_1[None, :] + lc_k * (bV1 - bN_1)[None, :]
                x0k, x1k = post_joint[mk, :187], post_joint[mk, 187:]
                x0c = x0k - x0k.mean(1, keepdims=True); h0c = hat0 - hat0.mean(1, keepdims=True)
                x1c = x1k - x1k.mean(1, keepdims=True); h1c = hat1 - hat1.mean(1, keepdims=True)
                c0 = np.sum(x0c * h0c, axis=1) / (np.linalg.norm(x0c, axis=1) * np.linalg.norm(h0c, axis=1) + 1e-12)
                c1 = np.sum(x1c * h1c, axis=1) / (np.linalg.norm(x1c, axis=1) * np.linalg.norm(h1c, axis=1) + 1e-12)
                min_lead_bcorr_arr[mk] = np.minimum(c0, c1)

            s2_native = ready & (m_bridge_arr > 0.0)
            eval_mask = (s2_native if eval_mask_override is None else (ready & eval_mask_override))
            s2_idx = np.where(eval_mask)[0]
            if len(s2_idx) > 0:
                diff_loc_s2 = b_N_loc_hist[s2_idx].astype(np.float64) - b_N[None, :]
                y_loc_s2 = diff_loc_s2 - (diff_loc_s2 @ U) @ U.T
                for j_pos, t in enumerate(s2_idx):
                    k = int(bk[t])
                    q = Q_mat[k]
                    dN_l = loc_norm_dist_proj_arr[t]
                    dV_l = float(np.linalg.norm(y_mat[t] - q))
                    dNV_l = float(np.linalg.norm(q - y_loc_s2[j_pos]))
                    _, pyth_margin_loc[t] = v33.thales_obt_and_pyth(dN_l, dV_l, dNV_l)

                    x0, x1 = post_joint[t, :187], post_joint[t, 187:]
                    bL0, bL1 = b_N_loc_hist[t, :187].astype(np.float64), b_N_loc_hist[t, 187:].astype(np.float64)
                    bV0, bV1 = bV_stack[k, :187], bV_stack[k, 187:]
                    obt_l0, _ = v33.thales_obt_and_pyth(float(np.linalg.norm(x0 - bL0)), float(np.linalg.norm(x0 - bV0)), float(np.linalg.norm(bV0 - bL0)))
                    obt_l1, _ = v33.thales_obt_and_pyth(float(np.linalg.norm(x1 - bL1)), float(np.linalg.norm(x1 - bV1)), float(np.linalg.norm(bV1 - bL1)))
                    min_lead_thales_loc[t] = min(obt_l0, obt_l1)
        else:
            s2_native = np.zeros(N, dtype=bool)
    else:
        s2_native = np.zeros(N, dtype=bool)

    rel_bcorr_drop = pref_bcorr_p05 - min_lead_bcorr_arr
    hf_excess = np.maximum(0.0, sqi_hf_ratio - max(pref_hf_p95, 0.70))
    bcorr_hinge_sq = np.maximum(0.0, rel_bcorr_drop - 0.10) ** 2
    hf_hinge_sq = hf_excess ** 2

    def shift_past(arr, k):
        out = np.empty_like(arr, dtype=np.float64)
        out[:k] = arr[0]; out[k:] = arr[:-k]
        return out

    nb_hf_past2 = 0.5 * (shift_past(sqi_hf_ratio, 1) + shift_past(sqi_hf_ratio, 2))
    nb_bcorr_drop_past2 = 0.5 * (shift_past(rel_bcorr_drop, 1) + shift_past(rel_bcorr_drop, 2))
    mb_ind = (m_bridge_arr > 0.0).astype(np.float64)
    nb_mbridge_frac_past3 = (shift_past(mb_ind, 1) + shift_past(mb_ind, 2) + shift_past(mb_ind, 3)) / 3.0

    X18 = np.column_stack([
        m_bridge_arr, z_N_arr, z_V_arr, d_seg_arr,
        loc_norm_corr_arr, loc_dist_norm_arr,
        preqrs_corr_N, rr_dev_pre, min_lead_bcorr_arr, sqi_hf_ratio,
        bcorr_hinge_sq, hf_hinge_sq,
        zeno_qiso_drop_all, pyth_margin_loc, min_lead_thales_loc,
        nb_hf_past2, nb_bcorr_drop_past2, nb_mbridge_frac_past3,
    ]).astype(np.float64)
    X18 = np.nan_to_num(X18, nan=0.0, posinf=0.0, neginf=0.0)
    return ready, s2_native, X18


def compute_new_anchor_geom_3d(y_mat, anchors, anchor_meta, fwd_mask, post_birth=False):
    """
    Pure geometric, 100% unlabeled, exact O(4)-invariant k>=5 proximity features:
      [0] new_dy_ratio   : min_{k>=5} (||y_t - q_k|| / rad_y_V_k) clipped to [0, 3]  (3.0 if no active k>=5)
      [1] new_dlam_ratio : min_{k>=5} (|1 - lam_k|   / rad_V_k)   clipped to [0, 3]  (3.0 if no active k>=5)
      [2] new_veto_flag  : 1.0 if beat is in Fwd (vetoed ONLY by k>=5), else 0.0
    ZERO physiological features or gates!
    """
    N = len(y_mat)
    dy_ratio = np.full(N, 3.0, dtype=np.float64)
    dl_ratio = np.full(N, 3.0, dtype=np.float64)
    if len(anchors) > CORE_K:
        t_vec = np.arange(N)
        for k in range(CORE_K, len(anchors)):
            anc = anchors[k]
            qk = anc["q"]
            nq = anc["norm_q"]
            lam_k = (y_mat @ qk) / (nq ** 2 + 1e-12)
            dy_k = np.linalg.norm(y_mat - qk[None, :], axis=1) / (anc["rad_y_V"] + 1e-6)
            dl_k = np.abs(1.0 - lam_k) / (anc["rad_V"] + 1e-6)
            if post_birth:
                ok_k = (t_vec >= anchor_meta[k]["disc_idx"])
                dy_k = np.where(ok_k, dy_k, 3.0)
                dl_k = np.where(ok_k, dl_k, 3.0)
            dy_ratio = np.minimum(dy_ratio, dy_k)
            dl_ratio = np.minimum(dl_ratio, dl_k)
    dy_ratio = np.clip(dy_ratio, 0.0, 3.0)
    dl_ratio = np.clip(dl_ratio, 0.0, 3.0)
    veto_flag = fwd_mask.astype(np.float64)
    return np.column_stack([dy_ratio, dl_ratio, veto_flag])


def extract_v43_record(shared):
    """
    Runs ONE exact C00_K12_Base discovery pass (zero guards, zero labels!)
    and builds exact M0_HardVeto_K12, M1_SoftPass_Same18D, and M2_SoftPass_UnclippedGeom
    in BOTH FullTape and PostBirth modes.
    """
    (
        post_joint, post_raw0, post_raw1, post_y, pre_rr_loc, post_r, fs, r0_time,
        joint_all, joint_50, norm_beats, final_norm_idx, ectopic_idx, labels_50,
        b_N, U, P, norm_y_prefix, eta_0,
        pre0_50, pre1_50, pN_0, pN_1, post_pre0, post_pre1,
        pref_hf_p95, pref_bcorr_p05,
        sqi_hf_ratio, zeno_qiso_drop_all, preqrs_corr_N, rr_dev_pre,
        y_mat, norm_y, norm_y_thresh,
    ) = shared

    N, D = post_joint.shape
    anchors = []
    anchor_meta = []
    if len(ectopic_idx) >= 4:
        anc = v33.make_anchor_obj(joint_50[ectopic_idx], norm_y_prefix, b_N, P, 0, "prefix")
        if anc is not None:
            ey = labels_50[ectopic_idx]
            anchors.append(anc)
            anchor_meta.append({
                "slot": 1, "src": "prefix", "disc_idx": 0, "pct_tape": 0.0,
                "v_pur": round(float(np.mean(ey == 1)), 3),
                "f_frac": round(float(np.mean(ey == 2)), 3),
            })

    loc_ring = np.zeros((v33.LOC_BUF_LEN, D), dtype=np.float64)
    init_nb = norm_beats[-v33.LOC_BUF_LEN:]
    loc_cnt = len(init_nb); loc_ring[:loc_cnt] = init_nb; loc_ptr = loc_cnt % v33.LOC_BUF_LEN
    b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
    b_N_loc_hist = np.zeros((N, D), dtype=np.float32)
    loc_norm_corr_arr = np.zeros(N, dtype=np.float64)
    loc_norm_dist_proj_arr = np.zeros(N, dtype=np.float64)
    prev1, prev2 = joint_all[49], joint_all[48]

    cand_clusters = []
    for t in range(N):
        x_t = post_joint[t]
        b_N_loc_hist[t] = b_N_loc
        c_loc = v33.mean_lead_corr(x_t, b_N_loc)
        loc_norm_corr_arr[t] = c_loc
        dl = x_t - b_N_loc
        dl_p = dl - U @ (U.T @ dl)
        loc_norm_dist_proj_arr[t] = float(np.linalg.norm(dl_p))

        if (norm_y[t] >= norm_y_thresh) and (pre_rr_loc[t] <= v33.CAND_PRE_RR_LOC_MAX) and (sqi_hf_ratio[t] <= v33.ONL_HF_MAX):
            matched_anc, best_adist = None, v33.CLUSTER_RADIUS
            for anc in anchors:
                d_a = float(np.linalg.norm(x_t - anc["b_V"]))
                if d_a <= best_adist:
                    best_adist, matched_anc = d_a, anc
            if matched_anc is not None:
                q = matched_anc["q"]; nq = matched_anc["norm_q"]
                matched_anc["v_lam_buf"].append(float(np.dot(y_mat[t], q) / (nq ** 2 + 1e-12)))
                matched_anc["v_dy_buf"].append(float(np.linalg.norm(y_mat[t] - q)))
                v33._refresh_v_radii(matched_anc)
            elif len(anchors) < K_CAP:
                best_c_idx, best_dist = -1, v33.CLUSTER_RADIUS
                for ci, cl in enumerate(cand_clusters):
                    dist = float(np.linalg.norm(x_t - cl["center"]))
                    if dist <= best_dist:
                        best_dist, best_c_idx = dist, ci
                if best_c_idx >= 0:
                    best_c = cand_clusters[best_c_idx]
                    best_c["beats"].append(x_t); best_c["ys"].append(int(post_y[t]))
                    best_c["center"] = np.median(best_c["beats"], axis=0)
                    if len(best_c["beats"]) >= v33.CONFIRM_MIN_BEATS:
                        arr_b = np.array(best_c["beats"]); med_b = np.median(arr_b, axis=0)
                        if float(np.median(np.linalg.norm(arr_b - med_b, axis=1))) <= v33.CONFIRM_MAX_COHESION:
                            anc = v33.make_anchor_obj(arr_b, norm_y_prefix, b_N, P, int(t + 1), "online")
                            if anc is not None:
                                cy = np.array(best_c["ys"])
                                anchors.append(anc)
                                anchor_meta.append({
                                    "slot": len(anchors), "src": "online", "disc_idx": int(t + 1),
                                    "pct_tape": round(100.0 * float(t + 1) / max(1, N), 1),
                                    "v_pur": round(float(np.mean(cy == 1)), 3),
                                    "f_frac": round(float(np.mean(cy == 2)), 3),
                                })
                                cand_clusters.pop(best_c_idx)
                else:
                    if len(cand_clusters) >= v33.MAX_CAND_CLUSTERS:
                        cand_clusters.sort(key=lambda c: len(c["beats"]))
                        cand_clusters.pop(0)
                    cand_clusters.append({"beats": [x_t], "ys": [int(post_y[t])], "center": x_t.copy()})

        reg_rr = (v33.LOC_RR_MIN <= pre_rr_loc[t] <= v33.LOC_RR_MAX)
        far_from_v = all(float(np.linalg.norm(x_t - anc["b_V"])) >= v33.LOC_V_EXCLUSION_DIST for anc in anchors)
        if reg_rr and far_from_v:
            c_prev1 = v33.mean_lead_corr(x_t, prev1)
            c_prev12 = v33.mean_lead_corr(prev1, prev2)
            if (c_loc >= v33.LOC_ADMIT_CORR_BN and c_prev1 >= 0.70) or (c_prev1 >= v33.LOC_ADMIT_RUN_CORR and c_prev12 >= v33.LOC_ADMIT_RUN_CORR):
                loc_ring[loc_ptr] = x_t
                loc_ptr = (loc_ptr + 1) % v33.LOC_BUF_LEN
                if loc_cnt < v33.LOC_BUF_LEN: loc_cnt += 1
                b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
        prev2, prev1 = prev1, x_t

    K = len(anchors)
    core_cols = list(range(min(K, CORE_K)))
    all_cols = list(range(K))

    def pack_mode(post_birth=False):
        # 1. Core (k <= 4) Stage-1 mask & features
        _, s2_core4, X18_core4 = project_subset_features(
            post_joint, y_mat, b_N, U, b_N_loc_hist,
            loc_norm_corr_arr, loc_norm_dist_proj_arr,
            preqrs_corr_N, rr_dev_pre, sqi_hf_ratio, zeno_qiso_drop_all,
            pref_hf_p95, pref_bcorr_p05,
            anchors, anchor_meta, core_cols, eval_mask_override=None, post_birth=post_birth,
        )
        # 2. Full K=12 native Stage-1 mask (M0_HardVeto_K12)
        _, s2_k12, _ = project_subset_features(
            post_joint, y_mat, b_N, U, b_N_loc_hist,
            loc_norm_corr_arr, loc_norm_dist_proj_arr,
            preqrs_corr_N, rr_dev_pre, sqi_hf_ratio, zeno_qiso_drop_all,
            pref_hf_p95, pref_bcorr_p05,
            anchors, anchor_meta, all_cols, eval_mask_override=None, post_birth=post_birth,
        )
        # 3. Exact Soft-Pass union & Forwarded set: beats vetoed ONLY by new anchors (k >= 5)
        fwd_mask = s2_core4 & (~s2_k12)
        s2_soft = s2_k12 | s2_core4

        # 4. Compute exact K=12 18D features with pyth_loc & min_thales properly populated for ALL t in s2_soft!
        _, _, X18_k12_soft = project_subset_features(
            post_joint, y_mat, b_N, U, b_N_loc_hist,
            loc_norm_corr_arr, loc_norm_dist_proj_arr,
            preqrs_corr_N, rr_dev_pre, sqi_hf_ratio, zeno_qiso_drop_all,
            pref_hf_p95, pref_bcorr_p05,
            anchors, anchor_meta, all_cols, eval_mask_override=s2_soft, post_birth=post_birth,
        )
        # Note: for beats in s2_k12, X18_k12_soft[s2_k12] is BIT-FOR-BIT identical to native M0_HardVeto_K12!
        X18_m0 = X18_k12_soft[s2_k12].astype(np.float32)
        X18_m1 = X18_k12_soft[s2_soft].astype(np.float32)

        # 5. M2_SoftPass_UnclippedGeom: preserve core-bridge X18 on Fwd + append 3 pure geometric k>=5 scalars!
        X18_hybrid = np.where(fwd_mask[:, None], X18_core4, X18_k12_soft)
        G3_new = compute_new_anchor_geom_3d(y_mat, anchors, anchor_meta, fwd_mask, post_birth=post_birth)
        X21_m2 = np.hstack([X18_hybrid[s2_soft], G3_new[s2_soft]]).astype(np.float32)

        return {
            "s2_k12": s2_k12, "s2_soft": s2_soft, "fwd_mask": fwd_mask,
            "X18_m0": X18_m0, "X18_m1": X18_m1, "X21_m2": X21_m2,
        }

    return {
        "K": K, "anchor_meta": anchor_meta,
        "full": pack_mode(post_birth=False),
        "pb": pack_mode(post_birth=True),
    }


def _worker_v43(args):
    db_name, rec_dir, rec_id = args
    loaded = v42b.load_raw_record_arrays(db_name, rec_dir, rec_id)
    if loaded is None:
        return None
    beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, patient_id, post_hours = loaded
    shared = v42b.prepare_shared_prefix(beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, R_rot=None)
    out = extract_v43_record(shared)
    return {
        "db": db_name, "rec_id": rec_id, "patient_id": patient_id,
        "post_hours": post_hours, "y": shared[3], **out,
    }


def verify_o4_invariance_v43():
    R = ortho_group.rvs(4, random_state=20261122)
    diffs = {}
    for rid in ("208", "233"):
        loaded = v42b.load_raw_record_arrays("mitdb", v36.DB_DIRS["mitdb"], rid)
        beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, _, _ = loaded
        sh0 = v42b.prepare_shared_prefix(beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, R_rot=None)
        shR = v42b.prepare_shared_prefix(beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, R_rot=R)
        o0 = extract_v43_record(sh0)
        oR = extract_v43_record(shR)
        assert o0["K"] == oR["K"]
        assert np.array_equal(o0["full"]["s2_k12"], oR["full"]["s2_k12"])
        assert np.array_equal(o0["full"]["s2_soft"], oR["full"]["s2_soft"])
        assert np.array_equal(o0["full"]["fwd_mask"], oR["full"]["fwd_mask"])
        for key in ("X18_m0", "X18_m1", "X21_m2"):
            d = float(np.max(np.abs(o0["full"][key].astype(np.float64) - oR["full"][key].astype(np.float64)))) if len(o0["full"][key]) else 0.0
            diffs[f"{rid}_{key}"] = d
    return diffs


def build_or_load_v43_db(db_name):
    meta_p = os.path.join(CACHE_V43, f"{db_name}_v43_meta.json")
    npz_p = os.path.join(CACHE_V43, f"{db_name}_v43.npz")
    if os.path.exists(meta_p) and os.path.exists(npz_p):
        meta = json.load(open(meta_p, "r", encoding="utf-8"))
        dat = np.load(npz_p)
        res = {mode: {"M0": [], "M1": [], "M2": []} for mode in ("full", "pb")}
        for m in meta:
            y = dat["y_all"][m["b0"]:m["b1"]]
            for mode in ("full", "pb"):
                mm = m[mode]
                base = {
                    "db": m["db"], "rec_id": m["rec_id"], "patient_id": m["patient_id"],
                    "post_hours": m["post_hours"], "y": y, "K": m["K"], "anchor_meta": m["anchor_meta"],
                    "fwd_mask": dat[f"{mode}_fwd"][m["b0"]:m["b1"]],
                }
                res[mode]["M0"].append({
                    **base,
                    "s2_mask": dat[f"{mode}_s2_k12"][m["b0"]:m["b1"]],
                    "X_s2": dat[f"{mode}_X18_m0"][mm["m0_0"]:mm["m0_1"]],
                })
                res[mode]["M1"].append({
                    **base,
                    "s2_mask": dat[f"{mode}_s2_soft"][m["b0"]:m["b1"]],
                    "X_s2": dat[f"{mode}_X18_m1"][mm["ms_0"]:mm["ms_1"]],
                })
                res[mode]["M2"].append({
                    **base,
                    "s2_mask": dat[f"{mode}_s2_soft"][m["b0"]:m["b1"]],
                    "X_s2": dat[f"{mode}_X21_m2"][mm["ms_0"]:mm["ms_1"]],
                })
        return res

    d = v36.DB_DIRS[db_name]
    recs = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(d, "*.atr"))))
    if db_name == "mitdb":
        recs = [r for r in recs if r not in {"102", "104", "107", "217"}]
    nw = 4 if db_name in ("ltdb", "sddb") else 6
    t0 = time.time()
    print(f"Extracting v43 SoftPass (M0/M1/M2) on {db_name:8s} ({len(recs)} recs, {nw} workers)...", flush=True)
    raw = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for r in ex.map(_worker_v43, [(db_name, d, rid) for rid in recs]):
            if r is not None:
                raw.append(r)

    y_l = []
    arrs = {f"{m}_{k}": [] for m in ("full", "pb") for k in ("s2_k12", "s2_soft", "fwd", "X18_m0", "X18_m1", "X21_m2")}
    bc = 0
    cur = {m: {"m0": 0, "ms": 0} for m in ("full", "pb")}
    meta = []
    for it in raw:
        nb = len(it["y"])
        y_l.append(it["y"])
        md = {"db": it["db"], "rec_id": it["rec_id"], "patient_id": it["patient_id"],
              "post_hours": it["post_hours"], "K": it["K"], "anchor_meta": it["anchor_meta"],
              "b0": bc, "b1": bc + nb}
        for m in ("full", "pb"):
            pm = it[m]
            nm0, nms = len(pm["X18_m0"]), len(pm["X18_m1"])
            arrs[f"{m}_s2_k12"].append(pm["s2_k12"]); arrs[f"{m}_s2_soft"].append(pm["s2_soft"]); arrs[f"{m}_fwd"].append(pm["fwd_mask"])
            arrs[f"{m}_X18_m0"].append(pm["X18_m0"]); arrs[f"{m}_X18_m1"].append(pm["X18_m1"]); arrs[f"{m}_X21_m2"].append(pm["X21_m2"])
            md[m] = {"m0_0": cur[m]["m0"], "m0_1": cur[m]["m0"] + nm0, "ms_0": cur[m]["ms"], "ms_1": cur[m]["ms"] + nms}
            cur[m]["m0"] += nm0; cur[m]["ms"] += nms
        meta.append(md)
        bc += nb

    save_d = {"y_all": np.concatenate(y_l)}
    for k, v in arrs.items():
        if "_X" in k:
            dim = 21 if "X21" in k else 18
            save_d[k] = np.vstack([a for a in v if len(a) > 0]) if any(len(a) > 0 for a in v) else np.zeros((0, dim), dtype=np.float32)
        else:
            save_d[k] = np.concatenate(v)
    np.savez_compressed(npz_p, **save_d)
    json.dump(meta, open(meta_p, "w", encoding="utf-8"))
    print(f"  DONE {db_name:8s}: {bc:,} beats in {time.time()-t0:.1f}s", flush=True)
    return build_or_load_v43_db(db_name)


def load_k4_ref_from_v42b(db_name):
    meta = json.load(open(os.path.join(CACHE_V42B, f"{db_name}_v42b_meta.json"), "r", encoding="utf-8"))
    dat = np.load(os.path.join(CACHE_V42B, f"{db_name}_v42b.npz"))
    out = {"full": [], "pb": []}
    for m in meta:
        y = dat["y_all"][m["b0"]:m["b1"]]
        cm = m["cfgs"]["K4_Base"]
        base = {"db": m["db"], "rec_id": m["rec_id"], "patient_id": m["patient_id"],
                "post_hours": m["post_hours"], "y": y, "K": cm["K"], "fwd_mask": np.zeros(len(y), dtype=bool)}
        out["full"].append({**base, "s2_mask": dat["sf_K4_Base"][m["b0"]:m["b1"]], "X_s2": dat["xf_K4_Base"][cm["sf0"]:cm["sf1"]]})
        out["pb"].append({**base, "s2_mask": dat["sp_K4_Base"][m["b0"]:m["b1"]], "X_s2": dat["xp_K4_Base"][cm["sp0"]:cm["sp1"]]})
    return out


def fit_lodo_on_pool(tr_items, feat_idx, mtype, cap_pat_pos=100):
    """
    EXACT same LODO training & OOF source-locked T1 selection procedure as v36/v42/v42b,
    trained cleanly on each configuration's OWN source Stage-2 candidate pool!
    """
    pat_map = {}
    for it in tr_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: -sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    def build_xy(sub_items):
        Xs, ys, ws = [], [], []
        pf_c = Counter()
        for it in sub_items:
            if len(it["X_s2"]) > 0:
                pf_c[it["patient_id"]] += int(np.sum(it["y"][it["s2_mask"]] == 2))
        for it in sub_items:
            if len(it["X_s2"]) == 0: continue
            Xm = it["X_s2"][:, feat_idx].astype(np.float64)
            ym = (it["y"][it["s2_mask"]] == 2).astype(int)
            wm = np.ones(len(ym), dtype=np.float64)
            pf = pf_c[it["patient_id"]]
            if cap_pat_pos is not None and pf > cap_pat_pos:
                wm[ym == 1] = float(cap_pat_pos) / float(pf)
            Xs.append(Xm); ys.append(ym); ws.append(wm)
        X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
        sp, sn = float(np.sum(w[y == 1])), float(np.sum(w[y == 0]))
        w[y == 1] *= (0.5 * (sp + sn) / sp); w[y == 0] *= (0.5 * (sp + sn) / sn)
        return X, y, w

    oof_sc, oof_y = [], []
    tot_b = sum(len(it["y"]) for it in tr_items)
    tot_f = sum(int(np.sum(it["y"] == 2)) for it in tr_items)
    for fid in range(n_folds):
        tr_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] == fid]
        Xtr, ytr, wtr = build_xy(tr_sub)
        if np.sum(ytr == 1) == 0: continue
        sc = StandardScaler()
        clf = LogisticRegression(C=0.3, max_iter=500, random_state=42) if mtype == "LogR" else LinearSVC(C=0.3, dual="auto", max_iter=3000, random_state=42)
        clf.fit(sc.fit_transform(Xtr), ytr, sample_weight=wtr)
        for te in te_sub:
            if len(te["X_s2"]) > 0:
                oof_sc.append(clf.decision_function(sc.transform(te["X_s2"][:, feat_idx].astype(np.float64))))
                oof_y.append((te["y"][te["s2_mask"]] == 2).astype(int))
    s2_sc = np.concatenate(oof_sc); s2_y = np.concatenate(oof_y)
    order = np.argsort(s2_sc)[::-1]
    ys_o, ss_o = s2_y[order], s2_sc[order]
    tp_c = np.cumsum(ys_o == 1).astype(float); fp_c = np.cumsum(ys_o == 0).astype(float)
    pi_src = tot_f / max(1.0, float(tot_b))
    neg_mult = (pi_src / max(1e-9, 1.0 - pi_src)) / (0.001 / 0.999)
    tau1 = float(ss_o[int(np.argmax(2.0 * tp_c / (float(tot_f) + tp_c + fp_c * neg_mult + 1e-12)))])

    X_all, y_all, w_all = build_xy(tr_items)
    scaler = StandardScaler()
    clf = LogisticRegression(C=0.3, max_iter=500, random_state=42) if mtype == "LogR" else LinearSVC(C=0.3, dual="auto", max_iter=3000, random_state=42)
    clf.fit(scaler.fit_transform(X_all), y_all, sample_weight=w_all)
    return scaler, clf, tau1


def recall_at_fp_budget(scores, y_bin, max_fp):
    """Exact recall achieved when allowing at most max_fp false positives among valid Stage-2 beats."""
    tot_p = int(np.sum(y_bin == 1))
    if tot_p == 0:
        return 0.0
    valid = (scores > -1e5)
    if not np.any(valid):
        return 0.0
    sv = scores[valid]; yv = (y_bin[valid] == 1)
    ord_v = np.argsort(sv)[::-1]
    ok = np.where(np.cumsum(~yv[ord_v]) <= int(max_fp))[0]
    if len(ok) == 0:
        return 0.0
    tau = float(sv[ord_v[ok[-1]]])
    tp = int(np.sum((scores >= tau) & (y_bin == 1)))
    return 100.0 * tp / tot_p


def score_and_eval_v43(te_items, feat_idx, scaler, clf, tau1, m0_pat_fp_map=None):
    tot_hours = sum(it["post_hours"] for it in te_items)
    sc_l, y_l, fwd_l, rec_sc = [], [], [], []
    for it in te_items:
        sc = np.full(len(it["y"]), -1e6, dtype=np.float64)
        if len(it["X_s2"]) > 0:
            sc[it["s2_mask"]] = clf.decision_function(scaler.transform(it["X_s2"][:, feat_idx].astype(np.float64)))
        sc_l.append(sc); y_l.append(it["y"]); fwd_l.append(it["fwd_mask"]); rec_sc.append((it, sc))

    sc_cat = np.concatenate(sc_l); y_cat = np.concatenate(y_l); fwd_cat = np.concatenate(fwd_l)
    y_bin = (y_cat == 2).astype(int)
    tot_ft = int(np.sum(y_bin == 1))
    ap = float(average_precision_score(y_bin, sc_cat)) if tot_ft > 0 else 0.0

    # Pooled oracle-threshold ranking @ DB FAPH <= 2.0 and <= 5.0
    valid = (sc_cat > -1e5)
    sv = sc_cat[valid]; yv = (y_cat[valid] == 2)
    ord_v = np.argsort(sv)[::-1]

    def db_rank_at_faph(faph_cap):
        ok = np.where(np.cumsum(~yv[ord_v]) <= int(np.floor(faph_cap * tot_hours)))[0]
        tau = float(sv[ord_v[ok[-1]]]) if len(ok) > 0 else float(sv[ord_v[0]]) + 1.0
        tp = int(np.sum((sc_cat >= tau) & (y_cat == 2)))
        pf_tot, pf_tp = Counter(), Counter()
        for it, sc in rec_sc:
            nf = int(np.sum(it["y"] == 2))
            if nf > 0:
                pf_tot[it["patient_id"]] += nf
                pf_tp[it["patient_id"]] += int(np.sum((sc >= tau) & (it["y"] == 2)))
        mp = float(np.mean([100.0 * pf_tp[p] / pf_tot[p] for p in pf_tot])) if pf_tot else 0.0
        return round(100.0 * tp / max(1, tot_ft), 2), round(mp, 2)

    rec_f2, mp_db_f2 = db_rank_at_faph(2.0)
    rec_f5, mp_db_f5 = db_rank_at_faph(5.0)

    # Per-patient matched FA budgets!
    pat_data = {}
    for it, sc in rec_sc:
        p = it["patient_id"]
        d = pat_data.setdefault(p, {"sc": [], "y": [], "h": 0.0})
        d["sc"].append(sc); d["y"].append(it["y"]); d["h"] += it["post_hours"]

    pat_fp_t1 = {}
    pat_rec_at_pat_f2 = []
    pat_rec_at_m0_fp = []
    for p, pd in pat_data.items():
        psc = np.concatenate(pd["sc"]); py = np.concatenate(pd["y"]); pyb = (py == 2).astype(int)
        pat_fp_t1[p] = int(np.sum((psc >= tau1) & (py != 2)))
        if int(np.sum(pyb)) > 0:
            pat_rec_at_pat_f2.append(recall_at_fp_budget(psc, pyb, int(np.floor(2.0 * pd["h"]))))
            if m0_pat_fp_map is not None and p in m0_pat_fp_map:
                pat_rec_at_m0_fp.append(recall_at_fp_budget(psc, pyb, m0_pat_fp_map[p]))

    mp_pat_f2 = round(float(np.mean(pat_rec_at_pat_f2)), 2) if pat_rec_at_pat_f2 else 0.0
    mp_pat_m0fp = round(float(np.mean(pat_rec_at_m0_fp)), 2) if pat_rec_at_m0_fp else 0.0

    # Deployed source-locked T1
    tp1 = int(np.sum((sc_cat >= tau1) & (y_cat == 2)))
    fpn1 = int(np.sum((sc_cat >= tau1) & (y_cat == 0)))
    fpv1 = int(np.sum((sc_cat >= tau1) & (y_cat == 1)))
    fps1 = int(np.sum((sc_cat >= tau1) & (y_cat == 3)))
    fp1 = fpn1 + fpv1 + fps1
    f1_t1 = 2.0 * tp1 / max(1, 2 * tp1 + fp1 + (tot_ft - tp1))
    prec1 = 100.0 * tp1 / max(1, tp1 + fp1)
    rec1 = 100.0 * tp1 / max(1, tot_ft)
    faph1 = fp1 / max(1e-6, tot_hours)
    zf_tot = sum(len(it["y"]) for it, _ in rec_sc if int(np.sum(it["y"] == 2)) == 0)
    zf_fp = sum(int(np.sum((sc >= tau1) & (it["y"] != 2))) for it, sc in rec_sc if int(np.sum(it["y"] == 2)) == 0)
    spec = 100.0 * (1.0 - zf_fp / max(1, zf_tot)) if zf_tot > 0 else 100.0 * (1.0 - fp1 / max(1, len(y_cat) - tot_ft))

    # Forwarding & Separability inside Fwd!
    s2_f = sum(int(np.sum((it["y"] == 2) & it["s2_mask"])) for it, _ in rec_sc)
    s2_v = sum(int(np.sum((it["y"] == 1) & it["s2_mask"])) for it, _ in rec_sc)
    fwd_f = int(np.sum(fwd_cat & (y_cat == 2)))
    fwd_v = int(np.sum(fwd_cat & (y_cat == 1)))
    fwd_n = int(np.sum(fwd_cat & (y_cat == 0)))
    fwd_to_tp = int(np.sum(fwd_cat & (y_cat == 2) & (sc_cat >= tau1)))
    fwd_to_fpv = int(np.sum(fwd_cat & (y_cat == 1) & (sc_cat >= tau1)))
    fwd_to_fpn = int(np.sum(fwd_cat & (y_cat == 0) & (sc_cat >= tau1)))

    # Stage-2 score AUROC strictly inside Fwd (F vs V)!
    fwd_vf_mask = fwd_cat & ((y_cat == 2) | (y_cat == 1)) & (sc_cat > -1e5)
    if np.sum(fwd_vf_mask & (y_cat == 2)) > 0 and np.sum(fwd_vf_mask & (y_cat == 1)) > 0:
        fwd_auroc_fv = round(float(roc_auc_score((y_cat[fwd_vf_mask] == 2).astype(int), sc_cat[fwd_vf_mask])), 4)
    else:
        fwd_auroc_fv = None

    ev = {
        "AP": round(ap, 4),
        "Rank_Rec@F2_%": rec_f2, "MacroPat_DB@F2_%": mp_db_f2,
        "Rank_Rec@F5_%": rec_f5, "MacroPat_DB@F5_%": mp_db_f5,
        "MacroPat_Pat@F2_%": mp_pat_f2, "MacroPat_AtM0FP_%": mp_pat_m0fp,
        "T1_F1": round(f1_t1, 4), "T1_Prec_%": round(prec1, 2), "T1_Rec_%": round(rec1, 2),
        "T1_Spec_%": round(spec, 3), "T1_FAPH": round(faph1, 2),
        "T1_TP": tp1, "T1_FPN": fpn1, "T1_FPV": fpv1, "T1_FPS": fps1,
        "T1_Counts": f"{tp1}/{fpn1}/{fpv1}/{fps1}S",
        "S2_F": s2_f, "Tot_F": tot_ft, "S2_V": s2_v,
        "Fwd_F": fwd_f, "Fwd_V": fwd_v, "Fwd_N": fwd_n,
        "Fwd_to_TP": fwd_to_tp, "Fwd_to_FPV": fwd_to_fpv, "Fwd_to_FPN": fwd_to_fpn,
        "Fwd_AUROC_FvsV": fwd_auroc_fv,
    }
    return ev, rec_sc, pat_fp_t1


def inspect_spotlights(rec_sc_map, tau_map, db_name):
    rids = SPOTLIGHT_RECS.get(db_name, [])
    if not rids:
        return {}
    out = {}
    for rid in rids:
        rec_entry = {}
        # Lookup M0's T1 FP count on this record for matched-FP recall!
        m0_map = {it["rec_id"]: (it, sc) for it, sc in rec_sc_map["M0_HardVeto_K12"]}
        if rid not in m0_map:
            continue
        it_m0, sc_m0 = m0_map[rid]
        m0_fp = int(np.sum((sc_m0 >= tau_map["M0_HardVeto_K12"]) & (it_m0["y"] != 2)))
        pat_f2_cap = int(np.floor(2.0 * it_m0["post_hours"]))

        for cname, rsc_list in rec_sc_map.items():
            cmap = {it["rec_id"]: (it, sc) for it, sc in rsc_list}
            it, sc = cmap[rid]
            y = it["y"]; yb = (y == 2).astype(int)
            tau = tau_map[cname]
            p = (sc >= tau)
            fwd = it["fwd_mask"]
            fwd_vf = fwd & ((y == 2) | (y == 1)) & (sc > -1e5)
            auroc_fwd = round(float(roc_auc_score((y[fwd_vf] == 2).astype(int), sc[fwd_vf])), 3) if (np.sum(fwd_vf & (y == 2)) > 0 and np.sum(fwd_vf & (y == 1)) > 0) else None
            rec_entry[cname] = {
                "K": it["K"], "tot_F": int(np.sum(yb)),
                "S2_F": int(np.sum((y == 2) & it["s2_mask"])),
                "S2_V": int(np.sum((y == 1) & it["s2_mask"])),
                "Fwd_F": int(np.sum(fwd & (y == 2))),
                "Fwd_V": int(np.sum(fwd & (y == 1))),
                "T1_TP": int(np.sum((y == 2) & p)),
                "T1_FPV": int(np.sum((y == 1) & p)),
                "T1_FPN": int(np.sum((y == 0) & p)),
                "Fwd_to_TP": int(np.sum(fwd & (y == 2) & p)),
                "Fwd_to_FPV": int(np.sum(fwd & (y == 1) & p)),
                "Rec@PatF2_%": round(recall_at_fp_budget(sc, yb, pat_f2_cap), 1),
                "Rec@M0FP_%": round(recall_at_fp_budget(sc, yb, m0_fp), 1),
                "Fwd_AUROC_FvV": auroc_fwd,
            }
        # Also add pure single-feature separability inside Fwd from M2's 21D features (0=mb4, 6=preqrs, 7=rr_dev, 8=bcorr4, 13=pyth4, 18=new_dy_ratio, 19=new_dlam_ratio)
        it2, _ = {it["rec_id"]: (it, sc) for it, sc in rec_sc_map["M2_SoftPass_UnclippedGeom"]}[rid]
        fwd2 = it2["fwd_mask"]; y2 = it2["y"]
        if np.sum(fwd2 & (y2 == 2)) > 0 and np.sum(fwd2 & (y2 == 1)) > 0:
            X_full = np.zeros((len(y2), 21), dtype=np.float64)
            X_full[it2["s2_mask"]] = it2["X_s2"]
            f_sub = X_full[fwd2 & (y2 == 2)]; v_sub = X_full[fwd2 & (y2 == 1)]
            feat_diag = {}
            for fi, fn in [(0, "mb_core4"), (6, "preqrs_c"), (7, "rr_dev"), (8, "bcorr_core4"), (13, "pyth_core4"), (18, "new_dy_ratio"), (19, "new_dlam_ratio")]:
                vals = np.concatenate([f_sub[:, fi], v_sub[:, fi]])
                labs = np.concatenate([np.ones(len(f_sub)), np.zeros(len(v_sub))])
                auc = float(roc_auc_score(labs, vals))
                feat_diag[fn] = {
                    "F_med": round(float(np.median(f_sub[:, fi])), 3),
                    "V_med": round(float(np.median(v_sub[:, fi])), 3),
                    "AUROC_dir": round(auc, 3),
                }
            rec_entry["Fwd_Feature_Separability"] = feat_diag
        out[rid] = rec_entry
    return out


def main():
    t0 = time.time()
    print("=== Step 1: Verifying Exact O(4) Gauge Invariance (MIT-208 & MIT-233, M0/M1/M2) ===", flush=True)
    o4_diffs = verify_o4_invariance_v43()
    max_o4 = max(o4_diffs.values())
    print(f"  O(4) max abs diff: {max_o4:.3e} ({o4_diffs})", flush=True)
    assert max_o4 < 1e-11, f"O(4) invariance violated: {max_o4}"

    print("\n=== Step 2: Extracting/Loading v43 SoftPass Caches (6 Hospitals, 2.68M Beats) ===", flush=True)
    dbs = {mode: {c: {} for c in ("K4_Base", "M0_HardVeto_K12", "M1_SoftPass_Same18D", "M2_SoftPass_UnclippedGeom")} for mode in ("full", "pb")}
    for db in DB_ORDER:
        k4 = load_k4_ref_from_v42b(db)
        v43 = build_or_load_v43_db(db)
        for mode in ("full", "pb"):
            dbs[mode]["K4_Base"][db] = k4[mode]
            dbs[mode]["M0_HardVeto_K12"][db] = v43[mode]["M0"]
            dbs[mode]["M1_SoftPass_Same18D"][db] = v43[mode]["M1"]
            dbs[mode]["M2_SoftPass_UnclippedGeom"][db] = v43[mode]["M2"]

    # Sanity-check M0_HardVeto_K12 bit-for-bit parity vs v42b C00_K12_Base!
    print("\n=== Step 3: Sanity-Checking M0_HardVeto_K12 Parity vs v42b C00_K12_Base ===", flush=True)
    for db in DB_ORDER:
        m42 = json.load(open(os.path.join(CACHE_V42B, f"{db}_v42b_meta.json"), "r", encoding="utf-8"))
        d42 = np.load(os.path.join(CACHE_V42B, f"{db}_v42b.npz"))
        m_old = {m["rec_id"]: (m["cfgs"]["C00_K12_Base"]["K"], int(np.sum(d42["sf_C00_K12_Base"][m["b0"]:m["b1"]]))) for m in m42}
        k_diff = sum(1 for it in dbs["full"]["M0_HardVeto_K12"][db] if it["K"] != m_old[it["rec_id"]][0])
        s2_diff = sum(abs(int(np.sum(it["s2_mask"])) - m_old[it["rec_id"]][1]) for it in dbs["full"]["M0_HardVeto_K12"][db])
        fwd_f = sum(int(np.sum(it["fwd_mask"] & (it["y"] == 2))) for it in dbs["full"]["M1_SoftPass_Same18D"][db])
        fwd_v = sum(int(np.sum(it["fwd_mask"] & (it["y"] == 1))) for it in dbs["full"]["M1_SoftPass_Same18D"][db])
        print(f"  Parity {db:8s}: K_diffs={k_diff}, S2_diffs={s2_diff} | FullTape Fwd: F={fwd_f:4d}, V={fwd_v:5d}", flush=True)

    summary = {"O4_Max_Diff": max_o4, "O4_Details": o4_diffs, "Models": {}}
    cfg_order = ["K4_Base", "M0_HardVeto_K12", "M1_SoftPass_Same18D", "M2_SoftPass_UnclippedGeom"]

    for mname in MODELS_18.keys():
        summary["Models"][mname] = {}
        f18, mtype = MODELS_18[mname]
        f21, _ = MODELS_21[mname]
        print(f"\n==================== {mname}: REPRESENTATION vs HARD VETO (RETRAINED PER CONFIG) ====================", flush=True)

        for tgt in DB_ORDER:
            summary["Models"][mname][tgt] = {}
            for mode in ("full", "pb"):
                ev_map = {}; rsc_map = {}; tau_map = {}; m2_new_weights = {}
                # First fit M0 so we have m0_pat_fp_map for exact per-patient matched-FP recall!
                fitted = {}
                for cname in cfg_order:
                    fidx = f21 if cname == "M2_SoftPass_UnclippedGeom" else f18
                    tr_items = [it for db in DB_ORDER if db != tgt for it in dbs[mode][cname][db]]
                    sc, clf, tau1 = fit_lodo_on_pool(tr_items, fidx, mtype, cap_pat_pos=100)
                    fitted[cname] = (fidx, sc, clf, tau1)
                    tau_map[cname] = tau1
                    if cname == "M2_SoftPass_UnclippedGeom":
                        w = clf.coef_.ravel()
                        m2_new_weights = {"new_dy_ratio": round(float(w[-3]), 3), "new_dlam_ratio": round(float(w[-2]), 3), "new_veto_flag": round(float(w[-1]), 3)}

                # Evaluate M0 first to obtain per-patient T1 FP counts
                fidx0, sc0, clf0, tau0 = fitted["M0_HardVeto_K12"]
                ev0, rsc0, m0_pat_fp = score_and_eval_v43(dbs[mode]["M0_HardVeto_K12"][tgt], fidx0, sc0, clf0, tau0, m0_pat_fp_map=None)
                # Re-score M0 with its own pat_fp_map so MacroPat_AtM0FP is populated symmetrically
                ev0, rsc0, _ = score_and_eval_v43(dbs[mode]["M0_HardVeto_K12"][tgt], fidx0, sc0, clf0, tau0, m0_pat_fp_map=m0_pat_fp)
                ev_map["M0_HardVeto_K12"] = ev0; rsc_map["M0_HardVeto_K12"] = rsc0

                for cname in ("K4_Base", "M1_SoftPass_Same18D", "M2_SoftPass_UnclippedGeom"):
                    fidx, sc, clf, tau1 = fitted[cname]
                    ev, rsc, _ = score_and_eval_v43(dbs[mode][cname][tgt], fidx, sc, clf, tau1, m0_pat_fp_map=m0_pat_fp)
                    ev_map[cname] = ev; rsc_map[cname] = rsc

                spot = inspect_spotlights(rsc_map, tau_map, tgt) if mode == "full" else {}
                summary["Models"][mname][tgt][mode] = {
                    "Eval": ev_map,
                    "M2_NewGeom_Weights": m2_new_weights,
                    "Spotlights": spot,
                }

            # Print concise FullTape summary
            em = summary["Models"][mname][tgt]["full"]["Eval"]
            print(f"\n  [{tgt}] FullTape (Stage 2 RETRAINED on each config's own 5 source DBs) | M2 weights={summary['Models'][mname][tgt]['full']['M2_NewGeom_Weights']}", flush=True)
            for cname in cfg_order:
                ev = em[cname]
                print(
                    f"    {cname:25s} | S2_F={ev['S2_F']:4d} S2_V={ev['S2_V']:5d} (Fwd:F={ev['Fwd_F']:3d},V={ev['Fwd_V']:4d} -> TP={ev['Fwd_to_TP']:3d},FPV={ev['Fwd_to_FPV']:4d} AUC={ev['Fwd_AUROC_FvsV']}) | "
                    f"AP={ev['AP']:.4f} DB@F2={ev['Rank_Rec@F2_%']:4.1f}% Pat@F2={ev['MacroPat_Pat@F2_%']:4.1f}% @M0FP={ev['MacroPat_AtM0FP_%']:4.1f}% | "
                    f"T1: F1={ev['T1_F1']:.4f} [{ev['T1_Counts']:16s}] FAPH={ev['T1_FAPH']:5.2f}",
                    flush=True,
                )

    out_json = os.path.join(AUD_DIR, "V43_HYPB2_REPRESENTATION_VS_VETO_AUDIT_20261122.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_md = os.path.join(AUD_DIR, "V43_HYPB2_REPRESENTATION_VS_VETO_AUDIT_20261122.md")
    lines = [
        "# v43 Hypothesis B2: Separating V-Anchor Representation Right vs. Hard Stage-1 Veto Right (`K_max = 12`)",
        "",
        "- **Strict Single-Factor Control**: Identical raw `K=12` anchor library (`C00_K12_Base`, zero anchor guards, zero new physiological gates), identical Static `v36` N-chart, AND **Stage 2 retrained + source-locked `T1` recalibrated independently on each configuration's own 5-hospital source candidate pool** (identical classifier family & OOF thresholding).",
        f"- **Exact $O(4)$ Gauge Invariance**: Verified on `MIT-208` & `MIT-233` across `M0`, `M1`, `M2` (`max |diff| = {max_o4:.3e}`).",
        "- **Configurations**:",
        "  - `K4_Base`: Independent `K_max = 4` baseline reference.",
        "  - `M0_HardVeto_K12`: New anchors (`k >= 5`) BOTH represent AND hard-veto at Stage 1 (`S2 = S2_K12`).",
        "  - `M1_SoftPass_Same18D`: Same `K=12` library; beats vetoed ONLY by `k >= 5` (`Fwd = S2_core4 \\ S2_K12`) are forwarded to Stage 2 using the exact same 18D `K=12` vector.",
        "  - `M2_SoftPass_UnclippedGeom`: Same `K=12` library & same `S2_soft` pool; preserves core (`k <= 4`) bridge geometry on `Fwd` and adds 3 pure geometric $O(4)$-invariant `k >= 5` proximity scalars (`new_dy_ratio`, `new_dlam_ratio`, `new_veto_flag`).",
        "",
    ]

    for mname in MODELS_18.keys():
        lines.extend([
            f"## 1. FullTape Comparison (`{mname}`, Stage 2 Retrained per Config)", "",
            "| DB | Config | S2_F / S2_V | Forwarded (`Fwd_F` / `Fwd_V`) | Fwd Fate (`->TP` / `->FPV` / AUROC) | AP | DB Rank@F2 (%) | MacroPat @PatF2 (%) | MacroPat @M0_FP (%) | Deployed `T1` F1 | TP/FP_N/FP_V/FP_S | FAPH |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
        ])
        for tgt in DB_ORDER:
            em = summary["Models"][mname][tgt]["full"]["Eval"]
            for cname in cfg_order:
                ev = em[cname]
                auc_s = f"{ev['Fwd_AUROC_FvsV']:.3f}" if ev["Fwd_AUROC_FvsV"] is not None else "—"
                lines.append(
                    f"| `{tgt}` | `{cname}` | {ev['S2_F']}/{ev['S2_V']} | {ev['Fwd_F']} / {ev['Fwd_V']} | "
                    f"+{ev['Fwd_to_TP']} TP / +{ev['Fwd_to_FPV']} FPV ({auc_s}) | **{ev['AP']:.4f}** | "
                    f"**{ev['Rank_Rec@F2_%']:.1f}** | **{ev['MacroPat_Pat@F2_%']:.1f}** | **{ev['MacroPat_AtM0FP_%']:.1f}** | "
                    f"**{ev['T1_F1']:.4f}** | {ev['T1_Counts']} | {ev['T1_FAPH']:.2f} |"
                )

        lines.extend([
            "", f"## 2. Apples-to-Apples `PostBirth` (*chi dung neo sau thoi diem sinh*, `{mname}`, Retrained per Config)", "",
            "| DB | Config | S2_F / S2_V | Forwarded (`Fwd_F` / `Fwd_V`) | Fwd Fate (`->TP` / `->FPV`) | AP | DB Rank@F2 (%) | MacroPat @PatF2 (%) | MacroPat @M0_FP (%) | Deployed `T1` F1 | TP/FP_N/FP_V/FP_S | FAPH |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
        ])
        for tgt in DB_ORDER:
            em = summary["Models"][mname][tgt]["pb"]["Eval"]
            for cname in cfg_order:
                ev = em[cname]
                lines.append(
                    f"| `{tgt}` | `{cname}` | {ev['S2_F']}/{ev['S2_V']} | {ev['Fwd_F']} / {ev['Fwd_V']} | "
                    f"+{ev['Fwd_to_TP']} TP / +{ev['Fwd_to_FPV']} FPV | **{ev['AP']:.4f}** | "
                    f"**{ev['Rank_Rec@F2_%']:.1f}** | **{ev['MacroPat_Pat@F2_%']:.1f}** | **{ev['MacroPat_AtM0FP_%']:.1f}** | "
                    f"**{ev['T1_F1']:.4f}** | {ev['T1_Counts']} | {ev['T1_FAPH']:.2f} |"
                )

        lines.extend([
            "", f"## 3. Target Record Spotlights (`{mname}`, FullTape)", "",
            "| DB | Record | Config | K | S2_F / S2_V | Fwd (`F`/`V`) | Fwd Fate (`TP`/`FPV`) | `T1` TP / FP_V / FP_N | Rec@PatF2 (%) | Rec@M0FP (%) | Fwd AUROC (F vs V) |",
            "|---|---|---|---:|---|---|---|---|---:|---:|---:|",
        ])
        for tgt in ("ltdb", "mitdb", "edb", "sddb", "incartdb"):
            spots = summary["Models"][mname][tgt]["full"]["Spotlights"]
            for rid, rdat in spots.items():
                for cname in cfg_order:
                    rd = rdat[cname]
                    auc_s = f"{rd['Fwd_AUROC_FvV']:.3f}" if rd["Fwd_AUROC_FvV"] is not None else "—"
                    lines.append(
                        f"| `{tgt}` | `{rid}` | `{cname}` | {rd['K']} | {rd['S2_F']}/{rd['S2_V']} | "
                        f"{rd['Fwd_F']}/{rd['Fwd_V']} | +{rd['Fwd_to_TP']}/+{rd['Fwd_to_FPV']} | "
                        f"{rd['T1_TP']}/{rd['T1_FPV']}/{rd['T1_FPN']} | {rd['Rec@PatF2_%']:.1f} | {rd['Rec@M0FP_%']:.1f} | {auc_s} |"
                    )
        lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nALL DONE in {time.time()-t0:.1f}s!\nSaved: {out_json}\nSaved: {out_md}", flush=True)


if __name__ == "__main__":
    main()
