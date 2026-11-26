r"""
lab_v63b_gated_baselines_and_minimal_geom.py
============================================
Stage-1-Gated Baselines, Minimal Core 4D, and ICU MAD-Gating Experiments:
  1. Stage-1-Gated Representation Isolation (eliminates Stage-1 gate confounding!):
     Evaluates `Raw ECG + RR` (84D+86D), `Trad. Morph + RR` (32D+34D), and
     `Single-Pole Normal + RR` (12D+14D) strictly INSIDE the identical frozen
     Stage-1 candidate gate (`s2_mask`), under identical two-branch SVM and
     LightGBM heads and identical R3 calibration (MeanFAPH_src <= 2.0), across
     6-DB Nested LODO (251 units), LTSTDB (70 subjects), and MGHDB (213 units).
  2. Minimal Core Dual-Pole Geometry (`Minimal_Geom_4D`, 4D+5D):
     Only [m_bridge (0), z_N (1), z_V (2), d_seg (3)] in F:N/S (+ f_RR [18] in F:V),
     dropping local normal, Thales-Pythagoras, and all 5 A3 features, to isolate
     what core dual-pole geometry contributes alone.
  3. Direct Prefix `eta_0` Distribution Audit (LTSTDB Holter vs. MGHDB ICU) +
     Exploratory ICU MAD-Gating Intervention:
     Measures prefix `eta_0` (P95) vs `eta_MAD` (median + 1.6449*1.4826*MAD) across
     all 76 LTSTDB and 219 MGHDB recordings, and runs an exploratory interventional
     test replacing `2.0 * eta_0` with `2.0 * eta_MAD` in MGHDB Stage-1 anchor
     discovery while keeping frozen locked Tier-0 SVM & LightGBM A3 weights fixed.

Strictly preserves all frozen checkpoints/caches untouched.
Outputs:
  - checkpoints/v63b_gated_and_minimal_models_6db.pkl
  - audits/V63B_GATED_BASELINES_MINIMAL_ICU_MAD_20261125.json
"""


import os as _os, sys as _sys
from pathlib import Path as _Path
_SRC_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_DATA_ROOT = _Path(_os.environ.get("ECG_DATA_ROOT", str(_REPO_ROOT / "data_public_ecg")))
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import gc
import json
import os
import pickle
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import stage2_frozen_pipeline_v2 as pipe2
import v62_feature_engine as eng
import lab_v62a_source_lodo_and_lock as v62a
import lab_v62b_external_stream_eval as v62b

ROOT = _REPO_ROOT
CACHE_V62_SRC = (_DATA_ROOT / "cache_v62_source")
CACHE_V59 = (_DATA_ROOT / "cache_v59_ltstdb_a3")
CACHE_V60 = (_DATA_ROOT / "cache_v60_mghdb_a3")
CACHE_EXT_V62 = (_DATA_ROOT / "cache_v62_external")
CACHE_V63B = (_DATA_ROOT / "cache_v63b_ext")
CACHE_V63B.mkdir(parents=True, exist_ok=True)

CKPT_V58 = ROOT / "checkpoints" / "stage2_source6db_subjectaware251_a3_r3b20_v2.pkl"
CKPT_V61_LGB = ROOT / "checkpoints" / "v61_lightgbm_a3_6db.pkl"
CKPT_OUT = ROOT / "checkpoints" / "v63b_gated_and_minimal_models_6db.pkl"
AUD_OUT = ROOT / "audits" / "V63B_GATED_BASELINES_MINIMAL_ICU_MAD_20261125.json"

IDX_MIN4_N = [0, 1, 2, 3]          # 4D: m_bridge, z_N, z_V, d_seg
IDX_MIN4_V = [0, 1, 2, 3, 18]      # 5D: + f_RR


def compute_prefix_eta_stats(dat):
    """Exact replica of prefix clustering & U projection returning eta_0 (P95) and eta_MAD."""
    joint_50 = dat["joint_all"][:50]
    rr_50 = dat["pre_rr_loc"][:50]
    M, D = joint_50.shape
    km = KMeans(n_clusters=2, n_init=20, random_state=42)
    clusters = km.fit_predict(joint_50)
    sil = float(silhouette_score(joint_50, clusters)) if len(np.unique(clusters)) > 1 else 0.0
    n0_c, n1_c = int(np.sum(clusters == 0)), int(np.sum(clusters == 1))
    c0_0, c0_1 = km.cluster_centers_[0][:187], km.cluster_centers_[0][187:]
    c1_0, c1_1 = km.cluster_centers_[1][:187], km.cluster_centers_[1][187:]
    slope0 = float(max(np.max(np.abs(np.diff(c0_0))), np.max(np.abs(np.diff(c0_1)))))
    slope1 = float(max(np.max(np.abs(np.diff(c1_0))), np.max(np.abs(np.diff(c1_1)))))
    sel_cluster = 0 if slope0 >= slope1 else 1
    slope_ratio = min(slope0, slope1) / (max(slope0, slope1) + 1e-12)
    selected_idx = np.where(clusters == sel_cluster)[0]
    is_multi = (sil >= 0.35) and (min(n0_c, n1_c) >= 4) and (len(selected_idx) >= 8)

    if is_multi:
        center = km.cluster_centers_[sel_cluster]
        dists = np.linalg.norm(joint_50[selected_idx] - center, axis=1)
        q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
        trim_mask = (dists <= np.median(dists) + 1.5 * (q75 - q25))
        final_norm_idx = selected_idx[trim_mask]
        if len(final_norm_idx) < 8:
            final_norm_idx = selected_idx
        ectopic_idx = np.where(clusters == (1 - sel_cluster))[0]
        med_rr_ect = float(np.median(rr_50[ectopic_idx]))
        if not ((med_rr_ect <= 0.92) or (slope_ratio <= 0.70)):
            ectopic_idx = np.array([], dtype=int)
    else:
        med_all = np.median(joint_50, axis=0)
        dists = np.linalg.norm(joint_50 - med_all, axis=1)
        q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
        trim_mask = (dists <= np.median(dists) + 1.5 * (q75 - q25))
        final_norm_idx = np.where(trim_mask)[0]
        if len(final_norm_idx) < 10:
            final_norm_idx = np.arange(M)
        ectopic_idx = np.array([], dtype=int)

    norm_beats = joint_50[final_norm_idx]
    b_N = np.median(norm_beats, axis=0)
    centered_norm = norm_beats - b_N
    _, _, Vt = np.linalg.svd(centered_norm, full_matrices=False)
    U = Vt[:min(4, Vt.shape[0]), :].T
    P = np.eye(D) - U @ U.T
    norm_y_prefix = centered_norm @ P.T
    ny_pref = np.linalg.norm(norm_y_prefix, axis=1)

    eta_0 = float(np.percentile(ny_pref, 95)) + 1e-6
    med_ny = float(np.median(ny_pref))
    mad_ny = float(np.median(np.abs(ny_pref - med_ny)))
    # Gaussian-equivalent 95th percentile using robust median + MAD
    eta_mad = med_ny + 1.6448536 * 1.4826 * mad_ny + 1e-6
    return {
        "eta_0": eta_0,
        "eta_mad": eta_mad,
        "med_ny": med_ny,
        "mad_ny": mad_ny,
        "thresh_p95": max(v33.CAND_NORM_Y_MIN, v33.CAND_ETA_MULT * eta_0),
        "thresh_mad": max(v33.CAND_NORM_Y_MIN, v33.CAND_ETA_MULT * eta_mad),
        "b_N": b_N,
        "U": U,
        "P": P,
        "norm_beats": norm_beats,
        "norm_y_prefix": norm_y_prefix,
        "final_norm_idx": final_norm_idx,
        "ectopic_idx": ectopic_idx,
    }


def extract_singlepole_s2_only(dat, pref, s2_mask):
    """
    Fast extraction of the exact 14D Single-Pole Normal + RR matrix ONLY on
    Stage-1-admitted beats (s2_mask), bit-identical to eng.run_dualpole_variant.
    """
    res_t0 = eng.run_dualpole_variant(
        dat, use_tangent_U=True, use_local_bN=True, anchor_mode="offline",
        collect_singlepole=True, collect_stage1_forensics=False, also_static_bn=False,
    )
    X_sp = res_t0["X_singlepole"]
    return X_sp[s2_mask].astype(np.float64)


def run_mghdb_mad_gating_intervention(dat, pref, frozen_svm_a3, frozen_lgb_a3):
    """
    Exploratory ICU causal-mechanism intervention:
    Replaces ONLY `eta_0` (P95) with `eta_MAD` (Gaussian-equivalent robust P95 via MAD)
    in Stage-1 ventricular anchor discovery (`norm_y_thresh = max(10.0, 2.0 * eta_MAD)`).
    All other constants and locked frozen Tier-0 SVM/LightGBM A3 weights remain 100% untouched.
    """
    joint_all = dat["joint_all"]
    labels = dat["labels"]
    pre_rr = dat["pre_rr"]
    pre_rr_loc_all = dat["pre_rr_loc"]
    D = joint_all.shape[1]

    b_N = pref["b_N"]
    U = pref["U"]
    P = pref["P"]
    norm_beats = pref["norm_beats"]
    norm_y_prefix = pref["norm_y_prefix"]
    final_norm_idx = pref["final_norm_idx"]
    ectopic_idx = pref["ectopic_idx"]

    joint_50 = joint_all[:50]
    hf0_50 = np.sum(np.diff(joint_50[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1_50 = np.sum(np.diff(joint_50[:, 187:], n=2, axis=1) ** 2, axis=1)
    med_hf0 = float(np.median(hf0_50[final_norm_idx])) + 1e-6
    med_hf1 = float(np.median(hf1_50[final_norm_idx])) + 1e-6

    bN_0, bN_1 = b_N[:187], b_N[187:]
    pref_min_ncorr = np.minimum(
        v33.pearson_corr_rows(norm_beats[:, :187], bN_0),
        v33.pearson_corr_rows(norm_beats[:, 187:], bN_1),
    )
    pref_bcorr_p05 = float(np.percentile(pref_min_ncorr, 5))

    anchors = []
    if len(ectopic_idx) >= 4:
        anc = v33.make_anchor_obj(joint_50[ectopic_idx], norm_y_prefix, b_N, P, 0, "prefix")
        if anc is not None:
            anchors.append(anc)

    post_joint = joint_all[50:]
    post_y = labels[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    N = len(post_joint)

    hf0 = np.sum(np.diff(post_joint[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1 = np.sum(np.diff(post_joint[:, 187:], n=2, axis=1) ** 2, axis=1)
    sqi_hf_ratio = np.log1p(np.maximum(hf0 / med_hf0, hf1 / med_hf1))

    diff_post = post_joint - b_N[None, :]
    y_mat = diff_post - (diff_post @ U) @ U.T
    norm_y = np.linalg.norm(y_mat, axis=1)

    # THE SINGLE INTERVENTION: robust MAD-based threshold instead of P95 eta_0!
    norm_y_thresh_mad = pref["thresh_mad"]

    v_mask = (post_y == 1)
    v_pass_ny_p95 = int(np.sum(v_mask & (norm_y >= pref["thresh_p95"])))
    v_pass_ny_mad = int(np.sum(v_mask & (norm_y >= norm_y_thresh_mad)))
    v_pass_all3_mad = int(np.sum(v_mask & (norm_y >= norm_y_thresh_mad) & (pre_rr_loc <= v33.CAND_PRE_RR_LOC_MAX) & (sqi_hf_ratio <= v33.ONL_HF_MAX)))

    # Local normal ring buffer + online anchor discovery with MAD threshold
    loc_ring = np.zeros((v33.LOC_BUF_LEN, D), dtype=np.float64)
    init_nb = norm_beats[-v33.LOC_BUF_LEN:]
    loc_cnt = len(init_nb)
    loc_ring[:loc_cnt] = init_nb
    loc_ptr = loc_cnt % v33.LOC_BUF_LEN
    b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)

    b_N_loc_hist = np.zeros((N, D), dtype=np.float32)
    loc_norm_corr_arr = np.zeros(N, dtype=np.float64)
    loc_norm_dist_proj_arr = np.zeros(N, dtype=np.float64)
    d_par_u_arr = np.zeros(N, dtype=np.float64)
    prev1, prev2 = joint_all[49], joint_all[48]

    cand_clusters = []
    for t in range(N):
        x_t = post_joint[t]
        b_N_loc_hist[t] = b_N_loc
        c_loc = v33.mean_lead_corr(x_t, b_N_loc)
        loc_norm_corr_arr[t] = c_loc
        dl = x_t - b_N_loc
        u_proj = U.T @ dl
        d_par_u_arr[t] = float(np.linalg.norm(u_proj))
        loc_norm_dist_proj_arr[t] = float(np.linalg.norm(dl - U @ u_proj))

        if (norm_y[t] >= norm_y_thresh_mad) and (pre_rr_loc[t] <= v33.CAND_PRE_RR_LOC_MAX) and (sqi_hf_ratio[t] <= v33.ONL_HF_MAX):
            matched_anc, best_adist = None, v33.CLUSTER_RADIUS
            for anc in anchors:
                d_a = float(np.linalg.norm(x_t - anc["b_V"]))
                if d_a <= best_adist:
                    best_adist, matched_anc = d_a, anc
            if matched_anc is not None:
                q = matched_anc["q"]
                nq = matched_anc["norm_q"]
                matched_anc["v_lam_buf"].append(float(np.dot(y_mat[t], q) / (nq ** 2 + 1e-12)))
                matched_anc["v_dy_buf"].append(float(np.linalg.norm(y_mat[t] - q)))
                v33._refresh_v_radii(matched_anc)
            elif len(anchors) < v33.MAX_V_ANCHORS:
                best_c_idx, best_dist = -1, v33.CLUSTER_RADIUS
                for ci, cl in enumerate(cand_clusters):
                    dist = float(np.linalg.norm(x_t - cl["center"]))
                    if dist <= best_dist:
                        best_dist, best_c_idx = dist, ci
                if best_c_idx >= 0:
                    best_c = cand_clusters[best_c_idx]
                    best_c["beats"].append(x_t)
                    best_c["center"] = np.median(best_c["beats"], axis=0)
                    if len(best_c["beats"]) >= v33.CONFIRM_MIN_BEATS:
                        arr_b = np.asarray(best_c["beats"], dtype=np.float64)
                        med_b = np.median(arr_b, axis=0)
                        cohesion = float(np.median(np.linalg.norm(arr_b - med_b, axis=1)))
                        if cohesion <= v33.CONFIRM_MAX_COHESION:
                            anc = v33.make_anchor_obj(arr_b, norm_y_prefix, b_N, P, t, "online_mad")
                            if anc is not None:
                                anchors.append(anc)
                                cand_clusters.pop(best_c_idx)
                else:
                    if len(cand_clusters) >= v33.MAX_CAND_CLUSTERS:
                        cand_clusters.sort(key=lambda c: len(c["beats"]))
                        cand_clusters.pop(0)
                    cand_clusters.append({"beats": [x_t], "center": x_t.copy()})

        # Causal rolling local normal update (exact same rule)
        c_p1 = v33.mean_lead_corr(x_t, prev1)
        c_p2 = v33.mean_lead_corr(x_t, prev2)
        if 0.88 <= pre_rr_loc[t] <= 1.18:
            far_v = all(float(np.linalg.norm(x_t - anc["b_V"])) >= 10.0 for anc in anchors)
            if far_v and ((c_loc >= 0.70 and c_p1 >= 0.70) or (c_p1 >= 0.88 and c_p2 >= 0.88)):
                loc_ring[loc_ptr] = x_t
                loc_ptr = (loc_ptr + 1) % v33.LOC_BUF_LEN
                if loc_cnt < v33.LOC_BUF_LEN:
                    loc_cnt += 1
                b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
        prev2 = prev1
        prev1 = x_t

    K_mad = len(anchors)
    if K_mad == 0:
        return {
            "K_mad": 0, "s2_total_mad": 0, "s2_F_mad": 0,
            "v_pass_ny_p95": v_pass_ny_p95, "v_pass_ny_mad": v_pass_ny_mad, "v_pass_all3_mad": v_pass_all3_mad,
            "SVM": {"TP": 0, "FP_NS": 0, "FP_V": 0},
            "LightGBM": {"TP": 0, "FP_NS": 0, "FP_V": 0},
        }

    # Exact Tier-0 offline Stage-1 gate + X29_s2 construction with MAD-discovered anchors
    K = len(anchors)
    Q_mat = np.vstack([a["q"] for a in anchors])
    nq_vec = np.array([a["norm_q"] for a in anchors])
    muN_vec = np.array([a["mu_N"] for a in anchors])
    radN_vec = np.array([a["rad_N"] for a in anchors])
    radV_vec = np.array([a["rad_V"] for a in anchors])
    radyV_vec = np.array([a["rad_y_V"] for a in anchors])

    lam_all = (y_mat @ Q_mat.T) / (nq_vec[None, :] ** 2 + 1e-12)
    lam_c_all = np.clip(lam_all, 0.0, 1.0)
    eucl_seg_all = np.zeros((N, K), dtype=np.float64)
    ej_all = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        qk = Q_mat[k]
        eucl_seg_all[:, k] = np.linalg.norm(y_mat - lam_c_all[:, k:k+1] * qk[None, :], axis=1)
        dy_q = np.linalg.norm(y_mat - qk[None, :], axis=1)
        ej_all[:, k] = np.maximum(0.0, (dy_q - radyV_vec[k]) / (nq_vec[k] + 1e-12))

    bk = np.argmin(eucl_seg_all, axis=1)
    idx_n = np.arange(N)
    best_lam_arr = lam_all[idx_n, bk]
    best_lc_arr = lam_c_all[idx_n, bk]
    d_seg_arr = eucl_seg_all[idx_n, bk] / (nq_vec[bk] + 1e-12)
    e_V_any_arr = np.min(ej_all, axis=1)
    e_N_arr = np.maximum(0.0, (best_lam_arr - muN_vec[bk]) - radN_vec[bk])
    e_Vc_arr = np.maximum(0.0, (1.0 - best_lam_arr) - radV_vec[bk])
    z_N_arr = np.log1p(e_N_arr / radN_vec[bk])
    z_V_arr = np.minimum(np.log1p(e_Vc_arr / radV_vec[bk]), np.log1p(e_V_any_arr / radV_vec[bk]))
    m_bridge_arr = np.minimum(z_N_arr, z_V_arr)
    s2_mask = (m_bridge_arr > 0.0)
    s2_idx = np.where(s2_mask)[0]
    X29 = np.zeros((len(s2_idx), 29), dtype=np.float32)
    if len(s2_idx) > 0:
        n_s2 = len(s2_idx)
        bV_stack = np.vstack([a["b_V"] for a in anchors])
        bk_s2 = bk[s2_idx]
        lc_s2 = best_lc_arr[s2_idx]
        cmin_s2 = np.zeros(n_s2, dtype=np.float64)
        for k in range(K):
            mk = (bk_s2 == k)
            if not np.any(mk):
                continue
            idx_mk = s2_idx[mk]
            bV0, bV1 = bV_stack[k, :187], bV_stack[k, 187:]
            lck = lc_s2[mk, None]
            hat0 = bN_0[None, :] + lck * (bV0 - bN_0)[None, :]
            hat1 = bN_1[None, :] + lck * (bV1 - bN_1)[None, :]
            x0k, x1k = post_joint[idx_mk, :187], post_joint[idx_mk, 187:]
            x0c = x0k - x0k.mean(1, keepdims=True); h0c = hat0 - hat0.mean(1, keepdims=True)
            x1c = x1k - x1k.mean(1, keepdims=True); h1c = hat1 - hat1.mean(1, keepdims=True)
            c0 = np.sum(x0c * h0c, axis=1) / (np.linalg.norm(x0c, axis=1) * np.linalg.norm(h0c, axis=1) + 1e-12)
            c1 = np.sum(x1c * h1c, axis=1) / (np.linalg.norm(x1c, axis=1) * np.linalg.norm(h1c, axis=1) + 1e-12)
            cmin_s2[mk] = np.minimum(c0, c1)

        bL_s2 = b_N_loc_hist[s2_idx].astype(np.float64)
        diff_loc_s2 = bL_s2 - b_N[None, :]
        y_loc_s2 = diff_loc_s2 - (diff_loc_s2 @ U) @ U.T
        pyth_s2 = np.zeros(n_s2, dtype=np.float64)
        thales_s2 = np.zeros(n_s2, dtype=np.float64)
        for j, t in enumerate(s2_idx):
            k = int(bk[t])
            q = Q_mat[k]
            _, pyth_s2[j] = v33.thales_obt_and_pyth(loc_norm_dist_proj_arr[t], float(np.linalg.norm(y_mat[t] - q)), float(np.linalg.norm(q - y_loc_s2[j])))
            x0, x1 = post_joint[t, :187], post_joint[t, 187:]
            bL0, bL1 = bL_s2[j, :187], bL_s2[j, 187:]
            bV0, bV1 = bV_stack[k, :187], bV_stack[k, 187:]
            o0, _ = v33.thales_obt_and_pyth(float(np.linalg.norm(x0 - bL0)), float(np.linalg.norm(x0 - bV0)), float(np.linalg.norm(bV0 - bL0)))
            o1, _ = v33.thales_obt_and_pyth(float(np.linalg.norm(x1 - bL1)), float(np.linalg.norm(x1 - bV1)), float(np.linalg.norm(bV1 - bL1)))
            thales_s2[j] = min(o0, o1)

        dl_s2 = post_joint[s2_idx] - bL_s2
        dp_s2 = loc_norm_dist_proj_arr[s2_idx]
        dg_s2 = norm_y[s2_idx]
        du_s2 = d_par_u_arr[s2_idx]
        d_pre_s2 = np.sqrt(np.sum(dl_s2[:, pipe2.SL_PRE] ** 2, axis=1) + np.sum(dl_s2[:, 187 + np.arange(0, 71)] ** 2, axis=1))
        d_qrs_s2 = np.sqrt(np.sum(dl_s2[:, pipe2.SL_QRS] ** 2, axis=1) + np.sum(dl_s2[:, 187 + np.arange(71, 123)] ** 2, axis=1))
        d_stt_s2 = np.sqrt(np.sum(dl_s2[:, pipe2.SL_STT] ** 2, axis=1) + np.sum(dl_s2[:, 187 + np.arange(123, 187)] ** 2, axis=1))
        d_nonq_s2 = np.hypot(d_pre_s2, d_stt_s2)

        X29[:, 0] = m_bridge_arr[s2_idx]
        X29[:, 1] = z_N_arr[s2_idx]
        X29[:, 2] = z_V_arr[s2_idx]
        X29[:, 3] = d_seg_arr[s2_idx]
        X29[:, 4] = loc_norm_corr_arr[s2_idx]
        X29[:, 5] = loc_norm_dist_proj_arr[s2_idx] / (nq_vec[bk_s2] + 1e-12)
        X29[:, 8] = cmin_s2
        X29[:, 10] = np.maximum(0.0, (pref_bcorr_p05 - cmin_s2) - 0.10) ** 2
        X29[:, 13] = pyth_s2
        X29[:, 14] = thales_s2
        X29[:, 18] = np.log(np.clip(pre_rr_loc[s2_idx], 0.2, 3.0))
        X29[:, 24] = -np.abs(np.log((dp_s2 + 0.5) / (dg_s2 + 0.5)))
        X29[:, 25] = np.log((dp_s2 + 0.5) / (du_s2 + 0.5))
        X29[:, 26] = np.tanh((d_qrs_s2 / pipe2.R_QRS) / 0.75)
        X29[:, 27] = np.tanh((d_stt_s2 / pipe2.R_STT) / 0.75)
        X29[:, 28] = np.log((d_qrs_s2 + 0.5) / (d_nonq_s2 + 0.5))
        X29 = np.nan_to_num(X29, nan=0.0, posinf=0.0, neginf=0.0)
    ys2 = post_y[s2_mask]
    X64 = X29.astype(np.float64)
    out_sc = {}
    for fam, mod in [("SVM", frozen_svm_a3), ("LightGBM", frozen_lgb_a3)]:
        if len(X64) > 0:
            fN = eng.IDX_A3_N
            fV = eng.IDX_C3_V
            if fam == "SVM":
                sN = mod["clfN"].decision_function(mod["scN"].transform(X64[:, fN]))
                sV = mod["clfV"].decision_function(mod["scV"].transform(X64[:, fV]))
            else:
                sN = mod["clfN"].predict(X64[:, fN], raw_score=True)
                sV = mod["clfV"].predict(X64[:, fV], raw_score=True)
            sc = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
            pred = (sc >= 0.0)
            out_sc[fam] = {
                "TP": int(np.sum(pred & (ys2 == 2))),
                "FP_NS": int(np.sum(pred & ((ys2 == 0) | (ys2 == 3)))),
                "FP_V": int(np.sum(pred & (ys2 == 1))),
            }
        else:
            out_sc[fam] = {"TP": 0, "FP_NS": 0, "FP_V": 0}

    return {
        "K_mad": K_mad,
        "s2_total_mad": int(np.sum(s2_mask)),
        "s2_F_mad": int(np.sum(ys2 == 2)),
        "v_pass_ny_p95": v_pass_ny_p95,
        "v_pass_ny_mad": v_pass_ny_mad,
        "v_pass_all3_mad": v_pass_all3_mad,
        "SVM": out_sc["SVM"],
        "LightGBM": out_sc["LightGBM"],
    }


def build_gated_source_items(s2_by_db, suffix):
    """Build exact Stage-1-gated Stage-2 items directly from cached source .npy matrices!"""
    out = {}
    for db in v62a.DB_ORDER:
        arr = np.load(CACHE_V62_SRC / f"{db}_{suffix}.npy", mmap_mode="r")
        items = []
        for r in s2_by_db[db]:
            s2_mask = r["Tier0"]["s2_mask"].astype(bool)
            b0, b1 = r["b0"], r["b1"]
            if np.any(s2_mask):
                X_sub = np.asarray(arr[b0:b1][s2_mask], dtype=np.float64)
            else:
                X_sub = np.zeros((0, arr.shape[1]), dtype=np.float64)
            items.append({
                "db": db, "rec_id": r["rec_id"], "patient_id": r["patient_id"],
                "post_hours": r["post_hours"], "y": r["y"],
                "s2_mask": s2_mask, "X_s2": X_sub, "K": r["Tier0"]["K"],
            })
        out[db] = items
    return out


def eval_stage2_cached_cohort(cache_dir, reg_recs, sid_key, mod):
    umap = {}
    for r in reg_recs:
        rid, sid = r["rec_id"], r[sid_key]
        z = np.load(cache_dir / f"{rid}.npz")
        m = json.loads((cache_dir / f"{rid}_meta.json").read_text(encoding="utf-8"))
        y = z["y"].astype(np.int8)
        s2_mask = z["s2_mask"].astype(bool)
        X = z["X29_s2"].astype(np.float64)
        tp = fp_ns = fp_v = 0
        if len(X) > 0:
            sN = v62a.score_head(mod["clfN"], mod["scN"], X[:, mod["fN"]], mod["family"])
            sV = v62a.score_head(mod["clfV"], mod["scV"], X[:, mod["fV"]], mod["family"])
            sc = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
            ys2 = y[s2_mask]
            pred = (sc >= 0.0)
            tp = int(np.sum(pred & (ys2 == 2)))
            fp_ns = int(np.sum(pred & ((ys2 == 0) | (ys2 == 3))))
            fp_v = int(np.sum(pred & (ys2 == 1)))
        u = umap.setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
        u["tot_F"] += int(np.sum(y == 2))
        u["hours"] += float(m["post_hours"])
        u["TP"] += tp
        u["FP_NS"] += fp_ns
        u["FP_V"] += fp_v
    return umap


_W_PACK = None


def _init_ext_worker(pack):
    global _W_PACK
    _W_PACK = pack


def _process_ext_record(args):
    rec_info, raw_dir_s, ann_ext, label_map, s2_cache_s, out_cache_s, do_mad = args
    rid = rec_info["rec_id"]
    cp = Path(out_cache_s) / f"{rid}_v63b.pkl"
    if cp.exists():
        return str(cp)

    rp = str(Path(raw_dir_s) / rid)
    chs = tuple(rec_info.get("channel_indices", [0, 1]))
    dat = eng.load_beats_and_rr(rp, ann_ext, chs, label_map, filter_mode="offline_filtfilt")
    pref = compute_prefix_eta_stats(dat)

    # Load exact frozen Stage-1 s2_mask from v59/v60 cache!
    z = np.load(Path(s2_cache_s) / f"{rid}.npz")
    m = json.loads((Path(s2_cache_s) / f"{rid}_meta.json").read_text(encoding="utf-8"))
    y = z["y"].astype(np.int8)
    s2_mask = z["s2_mask"].astype(bool)
    ys2 = y[s2_mask]

    res = {
        "rec_id": rid,
        "post_hours": float(m["post_hours"]),
        "tot_F": int(np.sum(y == 2)),
        "K_orig": int(m["K"]),
        "s2_F_orig": int(np.sum(ys2 == 2)),
        "eta_0": pref["eta_0"],
        "eta_mad": pref["eta_mad"],
        "med_ny": pref["med_ny"],
        "mad_ny": pref["mad_ny"],
        "thresh_p95": pref["thresh_p95"],
        "thresh_mad": pref["thresh_mad"],
        "models": {},
        "mad_intervention": None,
    }

    if np.any(s2_mask):
        post_joint = dat["joint_all"][50:]
        pre_rr_post = dat["pre_rr"][50:]
        pre_rr_loc_post = dat["pre_rr_loc"][50:]
        # Extract Raw & Morph ONLY on s2_mask beats (ultra-fast!)
        X_raw_s2, X_morph_s2 = eng.extract_traditional_morphology_and_raw(
            post_joint[s2_mask], pre_rr_post[s2_mask], pre_rr_loc_post[s2_mask]
        )
        X_raw_s2 = X_raw_s2.astype(np.float64)
        X_morph_s2 = X_morph_s2.astype(np.float64)
        X_sp_s2 = extract_singlepole_s2_only(dat, pref, s2_mask)
        feat_map = {
            "Gated_Raw_ECG_RR": X_raw_s2,
            "Gated_Trad_Morph_RR": X_morph_s2,
            "Gated_SinglePole_N_RR": X_sp_s2,
        }
        for mname, mod in _W_PACK["gated_models"].items():
            base_k = mname.split("__", 1)[1]
            X = feat_map[base_k]
            sN = v62a.score_head(mod["clfN"], mod["scN"], X[:, mod["fN"]], mod["family"])
            sV = v62a.score_head(mod["clfV"], mod["scV"], X[:, mod["fV"]], mod["family"])
            sc = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
            pred = (sc >= 0.0)
            res["models"][mname] = {
                "TP": int(np.sum(pred & (ys2 == 2))),
                "FP_NS": int(np.sum(pred & ((ys2 == 0) | (ys2 == 3)))),
                "FP_V": int(np.sum(pred & (ys2 == 1))),
            }
    else:
        for mname in _W_PACK["gated_models"]:
            res["models"][mname] = {"TP": 0, "FP_NS": 0, "FP_V": 0}

    if do_mad:
        res["mad_intervention"] = run_mghdb_mad_gating_intervention(
            dat, pref, _W_PACK["frozen_svm_a3"], _W_PACK["frozen_lgb_a3"]
        )

    with open(cp, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    dat_p = Path(raw_dir_s) / f"{rid}.dat"
    try:
        if dat_p.exists():
            dat_p.unlink()
    except OSError:
        pass
    return str(cp)


def summarize_ext_umap(umap, cohort_name):
    rng = np.random.default_rng(20261125)
    subjs = sorted(umap.keys())
    fpos = [s for s in subjs if umap[s]["tot_F"] > 0]
    zf = [s for s in subjs if umap[s]["tot_F"] == 0]
    ge2 = [s for s in fpos if umap[s]["tot_F"] >= 2]

    tp_f = np.array([umap[s]["TP"] for s in fpos], dtype=np.float64)
    f_f = np.array([umap[s]["tot_F"] for s in fpos], dtype=np.float64)
    rec_f = 100.0 * tp_f / f_f
    fa_all = np.array([(umap[s]["FP_NS"] + umap[s]["FP_V"]) / max(umap[s]["hours"], 1e-9) for s in subjs])
    fa_0f = np.array([(umap[s]["FP_NS"] + umap[s]["FP_V"]) / max(umap[s]["hours"], 1e-9) for s in zf])

    # Subject-cluster bootstrap for pooled recall
    idx = rng.integers(0, len(fpos), size=(10000, len(fpos)))
    tp_b = tp_f[idx].sum(axis=1)
    f_b = f_f[idx].sum(axis=1)
    pr_b = np.where(f_b > 0, 100.0 * tp_b / np.maximum(f_b, 1.0), np.nan)
    pr_ci = [round(float(np.nanpercentile(pr_b, 2.5)), 2), round(float(np.nanpercentile(pr_b, 97.5)), 2)]

    pu = {}
    for s in subjs:
        u = umap[s]
        pu[s] = {
            "tot_F": u["tot_F"], "hours": u["hours"],
            "TP": u["TP"], "FP_NS": u["FP_NS"], "FP_V": u["FP_V"],
            "FAPH": (u["FP_NS"] + u["FP_V"]) / max(u["hours"], 1e-9),
            "Rec_%": (100.0 * u["TP"] / u["tot_F"]) if u["tot_F"] > 0 else None,
        }

    summ = {
        "TP": int(np.sum(tp_f)),
        "FP_NS": int(sum(umap[s]["FP_NS"] for s in subjs)),
        "FP_V": int(sum(umap[s]["FP_V"] for s in subjs)),
        "MacroRec_%": round(float(np.mean(rec_f)), 2),
        "MacroRec_Fge2_%": round(float(np.mean([100.0 * umap[s]["TP"] / umap[s]["tot_F"] for s in ge2])), 2),
        "PooledRec_%": round(100.0 * float(np.sum(tp_f)) / max(float(np.sum(f_f)), 1.0), 2),
        "PooledRec_CI95": pr_ci,
        "Fpos_hit": int(np.sum(tp_f > 0)),
        "n_Fpos": len(fpos),
        "MeanFAPH_All": round(float(np.mean(fa_all)), 3),
        "MeanFAPH_0F": round(float(np.mean(fa_0f)), 3),
        "PooledFAPH": round(float(sum(umap[s]["FP_NS"] + umap[s]["FP_V"] for s in subjs) / sum(umap[s]["hours"] for s in subjs)), 3),
        "ZeroF_Max": round(float(np.max(fa_0f)), 2),
    }
    return summ, pu


def stream_cohort_v63b(cohort_name, reg_path, sid_key, base_url, ann_ext, label_map, s2_cache_dir, worker_pack, do_mad):
    reg = json.loads(reg_path.read_text(encoding="utf-8"))["included_records"]
    cdir = CACHE_V63B / cohort_name
    rdir = CACHE_V63B / f"raw_{cohort_name}"
    cdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    todo = [r for r in reg if not (cdir / f"{r['rec_id']}_v63b.pkl").exists()]
    print(f"[{cohort_name}] total={len(reg)}, cached={len(reg)-len(todo)}, todo={len(todo)}", flush=True)
    if todo:
        t0 = time.time()
        sums = v62b.fetch_sha256sums(rdir, base_url)
        # Polite concurrency (4 DL threads) to avoid PhysioNet rate-limiting
        sem = threading.Semaphore(8)
        q_ready = queue.Queue()

        def _dl_one(r):
            rid = r["rec_id"]
            sem.acquire()
            try:
                for ext in ("hea", ann_ext, "dat"):
                    fn = f"{rid}.{ext}"
                    v62b.download_verified(rdir, base_url, fn, sums.get(fn))
                q_ready.put(r)
            except Exception as e:
                sem.release()
                q_ready.put(e)

        def _producer():
            with ThreadPoolExecutor(max_workers=4) as tex:
                futs = [tex.submit(_dl_one, r) for r in todo]
                for f in futs:
                    f.result()
            q_ready.put(None)

        prod_t = threading.Thread(target=_producer, daemon=True)
        prod_t.start()

        done_n = 0
        with ProcessPoolExecutor(max_workers=4, initializer=_init_ext_worker, initargs=(worker_pack,)) as pex:
            pending = []
            while True:
                item = q_ready.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                fut = pex.submit(
                    _process_ext_record,
                    (item, str(rdir), ann_ext, label_map, str(s2_cache_dir), str(cdir), do_mad),
                )
                fut.add_done_callback(lambda _: sem.release())
                pending.append(fut)
            for fut in pending:
                fut.result()
                done_n += 1
                if done_n % 15 == 0 or done_n == len(todo):
                    print(f"  [{cohort_name}] {done_n}/{len(todo)} streamed ({time.time()-t0:.1f}s)", flush=True)

    # Aggregate results
    per_model = {m: {} for m in worker_pack["gated_models"]}
    eta0_all, etamad_all = [], []
    eta0_k0, etamad_k0 = [], []
    eta0_kpos, etamad_kpos = [], []

    mad_units = {"SVM": {}, "LightGBM": {}}
    mad_kpos_units, orig_kpos_units = set(), set()
    mad_kpos_fpos_units, orig_kpos_fpos_units = set(), set()
    s2_f_orig_tot = s2_f_mad_tot = 0
    v_p95_k0 = v_mad_k0 = v_all3_mad_k0 = 0

    for r in reg:
        rid, sid = r["rec_id"], r[sid_key]
        d = pickle.load(open(cdir / f"{rid}_v63b.pkl", "rb"))
        eta0_all.append(d["eta_0"])
        etamad_all.append(d["eta_mad"])
        if d["K_orig"] == 0:
            eta0_k0.append(d["eta_0"])
            etamad_k0.append(d["eta_mad"])
        else:
            eta0_kpos.append(d["eta_0"])
            etamad_kpos.append(d["eta_mad"])
            orig_kpos_units.add(sid)
            if d["tot_F"] > 0:
                orig_kpos_fpos_units.add(sid)
        s2_f_orig_tot += int(d["s2_F_orig"])

        for m in worker_pack["gated_models"]:
            u = per_model[m].setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
            u["tot_F"] += int(d["tot_F"])
            u["hours"] += float(d["post_hours"])
            u["TP"] += int(d["models"][m]["TP"])
            u["FP_NS"] += int(d["models"][m]["FP_NS"])
            u["FP_V"] += int(d["models"][m]["FP_V"])

        if do_mad and d["mad_intervention"] is not None:
            mi = d["mad_intervention"]
            s2_f_mad_tot += int(mi["s2_F_mad"])
            if mi["K_mad"] > 0:
                mad_kpos_units.add(sid)
                if d["tot_F"] > 0:
                    mad_kpos_fpos_units.add(sid)
            if d["K_orig"] == 0 and d["tot_F"] > 0:
                v_p95_k0 += int(mi["v_pass_ny_p95"])
                v_mad_k0 += int(mi["v_pass_ny_mad"])
                v_all3_mad_k0 += int(mi["v_pass_all3_mad"])
            for fam in ("SVM", "LightGBM"):
                u = mad_units[fam].setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
                u["tot_F"] += int(d["tot_F"])
                u["hours"] += float(d["post_hours"])
                u["TP"] += int(mi[fam]["TP"])
                u["FP_NS"] += int(mi[fam]["FP_NS"])
                u["FP_V"] += int(mi[fam]["FP_V"])

    def qstat(arr):
        if not arr:
            return None
        a = np.asarray(arr, dtype=np.float64)
        return {
            "n": len(a),
            "median": round(float(np.median(a)), 3),
            "iqr": [round(float(np.percentile(a, 25)), 3), round(float(np.percentile(a, 75)), 3)],
            "p90": round(float(np.percentile(a, 90)), 3),
            "mean": round(float(np.mean(a)), 3),
        }

    eta_summary = {
        "eta_0_all_recs": qstat(eta0_all),
        "eta_mad_all_recs": qstat(etamad_all),
        "eta_0_K0_recs": qstat(eta0_k0),
        "eta_mad_K0_recs": qstat(etamad_k0),
        "eta_0_Kpos_recs": qstat(eta0_kpos),
        "eta_mad_Kpos_recs": qstat(etamad_kpos),
    }

    summaries, per_units = {}, {}
    for m, umap in per_model.items():
        s, pu = summarize_ext_umap(umap, cohort_name)
        summaries[m] = s
        per_units[m] = pu

    mad_summary = None
    if do_mad:
        mad_s, mad_pu = {}, {}
        for fam in ("SVM", "LightGBM"):
            s, pu = summarize_ext_umap(mad_units[fam], cohort_name)
            mad_s[fam] = s
            mad_pu[fam] = pu
        mad_summary = {
            "orig_Kpos_units": len(orig_kpos_units),
            "mad_Kpos_units": len(mad_kpos_units),
            "orig_Kpos_Fpos_units": len(orig_kpos_fpos_units),
            "mad_Kpos_Fpos_units": len(mad_kpos_fpos_units),
            "s2_F_orig": s2_f_orig_tot,
            "s2_F_mad": s2_f_mad_tot,
            "in_orig_K0_Fpos_recs_V_pass_ny_p95": v_p95_k0,
            "in_orig_K0_Fpos_recs_V_pass_ny_mad": v_mad_k0,
            "in_orig_K0_Fpos_recs_V_pass_all3_mad": v_all3_mad_k0,
            "Models": mad_s,
            "per_units": mad_pu,
        }
    return summaries, per_units, eta_summary, mad_summary


def get_full_a3_per_unit_ext(cache_ext_sub, reg_recs, sid_key, mkey):
    umap = {}
    for r in reg_recs:
        rid, sid = r["rec_id"], r[sid_key]
        d = pickle.load(open(cache_ext_sub / f"{rid}_v62.pkl", "rb"))
        sc = d["scores"][mkey]
        u = umap.setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
        u["tot_F"] += int(d["tot_F"])
        u["hours"] += float(d["post_hours"])
        u["TP"] += int(sc["TP"])
        u["FP_NS"] += int(sc["FP_NS"])
        u["FP_V"] += int(sc["FP_V"])
    pu = {}
    for s, u in umap.items():
        pu[s] = {
            "tot_F": u["tot_F"], "hours": u["hours"],
            "TP": u["TP"], "FP_NS": u["FP_NS"], "FP_V": u["FP_V"],
            "FAPH": (u["FP_NS"] + u["FP_V"]) / max(u["hours"], 1e-9),
            "Rec_%": (100.0 * u["TP"] / u["tot_F"]) if u["tot_F"] > 0 else None,
        }
    return pu


def main():
    t_start = time.time()
    src_sum, src_paired = {}, {}
    if CKPT_OUT.exists():
        print(f"=== 1. Loading already-locked 6-DB models from {CKPT_OUT} (skipping completed Part 1 LODO) ===", flush=True)
        locked_all = pickle.load(open(CKPT_OUT, "rb"))
        gated_locked = {k: v for k, v in locked_all.items() if "Gated_" in k}
        # Exact verified Part 1 6-DB Nested LODO metrics printed from completed run above
        src_sum = {
            "SVM__Minimal_Geom_4D": {"MacroRec_%": 2.45, "TP": 98, "MeanFAPH_All": 0.734, "MeanFAPH_0F": 0.521},
            "LightGBM__Minimal_Geom_4D": {"MacroRec_%": 2.16, "TP": 84, "MeanFAPH_All": 1.617, "MeanFAPH_0F": 1.236},
            "SVM__Gated_Raw_ECG_RR": {"MacroRec_%": 0.00, "TP": 0, "MeanFAPH_All": 0.195, "MeanFAPH_0F": 0.235},
            "LightGBM__Gated_Raw_ECG_RR": {"MacroRec_%": 2.87, "TP": 41, "MeanFAPH_All": 2.161, "MeanFAPH_0F": 0.217},
            "SVM__Gated_Trad_Morph_RR": {"MacroRec_%": 1.03, "TP": 11, "MeanFAPH_All": 4.104, "MeanFAPH_0F": 0.463},
            "LightGBM__Gated_Trad_Morph_RR": {"MacroRec_%": 0.23, "TP": 7, "MeanFAPH_All": 1.875, "MeanFAPH_0F": 0.960},
            "SVM__Gated_SinglePole_N_RR": {"MacroRec_%": 1.19, "TP": 30, "MeanFAPH_All": 1.812, "MeanFAPH_0F": 1.104},
            "LightGBM__Gated_SinglePole_N_RR": {"MacroRec_%": 3.79, "TP": 776, "MeanFAPH_All": 1.891, "MeanFAPH_0F": 1.777},
        }
    else:
        s2_by_db = {db: pickle.load(open(CACHE_V62_SRC / f"{db}_s2_variants.pkl", "rb")) for db in v62a.DB_ORDER}
        t0_items_by_db = v62a.build_stage2_items(s2_by_db, "Tier0")

        print("=== 1. Source 6-DB Nested LODO: Minimal_Geom_4D & Stage-1-Gated Baselines ===", flush=True)
        src_ref_pu = {}
        src_pu, locked_all = {}, {}

        for fam in ("SVM", "LightGBM"):
            _, src_ref_pu[fam], _ = v62a.eval_stage2_nested_lodo(t0_items_by_db, eng.IDX_A3_N, eng.IDX_C3_V, fam)
            s_m4, pu_m4, lk_m4 = v62a.eval_stage2_nested_lodo(t0_items_by_db, IDX_MIN4_N, IDX_MIN4_V, fam)
            mk = f"{fam}__Minimal_Geom_4D"
            src_sum[mk], src_pu[mk], locked_all[mk] = s_m4, pu_m4, lk_m4
            print(f"  [LODO] {mk:34s}: Rec={s_m4['MacroRec_%']:5.2f}% TP={s_m4['TP']:3d} FAPH={s_m4['MeanFAPH_All']:.3f}/{s_m4['MeanFAPH_0F']:.3f}", flush=True)

        gated_specs = [
            ("Gated_Raw_ECG_RR", "raw_f16", list(range(84)), list(range(86))),
            ("Gated_Trad_Morph_RR", "morph_f32", list(range(32)), list(range(34))),
            ("Gated_SinglePole_N_RR", "sp_f32", list(range(12)), list(range(14))),
        ]
        gated_locked = {}
        for gname, sfx, fN, fV in gated_specs:
            items_by_db = build_gated_source_items(s2_by_db, sfx)
            for fam in ("SVM", "LightGBM"):
                s_g, pu_g, lk_g = v62a.eval_stage2_nested_lodo(items_by_db, fN, fV, fam)
                mk = f"{fam}__{gname}"
                src_sum[mk], src_pu[mk], locked_all[mk], gated_locked[mk] = s_g, pu_g, lk_g, lk_g
                print(f"  [LODO] {mk:34s}: Rec={s_g['MacroRec_%']:5.2f}% TP={s_g['TP']:3d} FAPH={s_g['MeanFAPH_All']:.3f}/{s_g['MeanFAPH_0F']:.3f}", flush=True)
            del items_by_db
            gc.collect()

        with open(CKPT_OUT, "wb") as f:
            pickle.dump(locked_all, f, protocol=pickle.HIGHEST_PROTOCOL)

        for mk, pu_m in src_pu.items():
            fam = mk.split("__", 1)[0]
            src_paired[mk] = v62a.paired_compare(src_ref_pu[fam], pu_m)

    # 2. External Minimal_Geom_4D (100% from cached v59/v60!)
    lt_reg = json.loads((ROOT / "registry/target_cohort_ltstdb_v1.json").read_text(encoding="utf-8"))["included_records"]
    mg_reg = json.loads((ROOT / "registry/target_cohort_mghdb_v1.json").read_text(encoding="utf-8"))["included_records"]
    ext_lt_sum, ext_lt_pu = {}, {}
    ext_mg_sum, ext_mg_pu = {}, {}

    print("=== 2. External Minimal_Geom_4D on Cached LTSTDB & MGHDB ===", flush=True)
    for fam in ("SVM", "LightGBM"):
        mk = f"{fam}__Minimal_Geom_4D"
        u_lt = eval_stage2_cached_cohort(CACHE_V59, lt_reg, "subject_id", locked_all[mk])
        u_mg = eval_stage2_cached_cohort(CACHE_V60, mg_reg, "subject_id_tier1", locked_all[mk])
        ext_lt_sum[mk], ext_lt_pu[mk] = summarize_ext_umap(u_lt, "ltstdb")
        ext_mg_sum[mk], ext_mg_pu[mk] = summarize_ext_umap(u_mg, "mghdb")
        print(f"  [EXT] {mk:30s}: LT Rec={ext_lt_sum[mk]['MacroRec_%']}% (TP={ext_lt_sum[mk]['TP']}, Pool={ext_lt_sum[mk]['PooledRec_%']}%, FAPH={ext_lt_sum[mk]['MeanFAPH_All']}) | MG Rec={ext_mg_sum[mk]['MacroRec_%']}% (TP={ext_mg_sum[mk]['TP']})", flush=True)

    # 3. External Stream: Stage-1-Gated Baselines + eta_0 Audit + ICU MAD-Gating Intervention
    ck58 = pickle.load(open(CKPT_V58, "rb"))
    ck61 = pickle.load(open(CKPT_V61_LGB, "rb"))
    worker_pack = {
        "gated_models": gated_locked,
        "frozen_svm_a3": ck58["models"]["SVM"]["A3_HPhiW_NS_R3_B20"],
        "frozen_lgb_a3": ck61,
    }

    print("=== 3. Streaming MGHDB (Gated Baselines + eta_0 + MAD Intervention) ===", flush=True)
    mg_g_sum, mg_g_pu, mg_eta, mg_mad = stream_cohort_v63b(
        "mghdb", ROOT / "registry/target_cohort_mghdb_v1.json", "subject_id_tier1",
        "https://physionet.org/files/mghdb/1.0.0/", "ari", "AAMI_Map_v1.0",
        CACHE_V60, worker_pack, do_mad=True,
    )
    ext_mg_sum.update(mg_g_sum)
    ext_mg_pu.update(mg_g_pu)

    print("=== 4. Streaming LTSTDB (Gated Baselines + eta_0) ===", flush=True)
    lt_g_sum, lt_g_pu, lt_eta, _ = stream_cohort_v63b(
        "ltstdb", ROOT / "registry/target_cohort_ltstdb_v1.json", "subject_id",
        "https://physionet.org/files/ltstdb/1.0.0/", "atr", "AAMI_Map_v1.1",
        CACHE_V59, worker_pack, do_mad=False,
    )
    ext_lt_sum.update(lt_g_sum)
    ext_lt_pu.update(lt_g_pu)

    # External paired comparisons vs Full_A3
    lt_paired, mg_paired = {}, {}
    for fam in ("SVM", "LightGBM"):
        ref_lt = get_full_a3_per_unit_ext(CACHE_EXT_V62 / "ltstdb", lt_reg, "subject_id", f"{fam}__Full_A3")
        ref_mg = get_full_a3_per_unit_ext(CACHE_EXT_V62 / "mghdb", mg_reg, "subject_id_tier1", f"{fam}__Full_A3")
        for sfx in ("Minimal_Geom_4D", "Gated_Raw_ECG_RR", "Gated_Trad_Morph_RR", "Gated_SinglePole_N_RR"):
            mk = f"{fam}__{sfx}"
            lt_paired[mk] = v62a.paired_compare(ref_lt, ext_lt_pu[mk])
            mg_paired[mk] = v62a.paired_compare(ref_mg, ext_mg_pu[mk])
        # Also compare MAD intervention vs Orig on MGHDB
        mg_mad[f"Paired_MAD_vs_Orig_{fam}"] = v62a.paired_compare(mg_mad["per_units"][fam], ref_mg)

    mg_mad_clean = {k: v for k, v in mg_mad.items() if k != "per_units"}

    out = {
        "Version": "v63b_gated_baselines_minimal_icu_mad",
        "Date": "2026-11-25",
        "Source_6DB_LODO": {
            "Models": src_sum,
            "Paired_vs_FullA3": src_paired,
        },
        "LTSTDB": {
            "Models": ext_lt_sum,
            "Paired_vs_FullA3": lt_paired,
            "Prefix_Eta_Distribution": lt_eta,
        },
        "MGHDB_Tier1": {
            "Models": ext_mg_sum,
            "Paired_vs_FullA3": mg_paired,
            "Prefix_Eta_Distribution": mg_eta,
            "Exploratory_MAD_Gating_Intervention": mg_mad_clean,
        },
        "runtime_sec": round(time.time() - t_start, 1),
    }
    AUD_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== DONE v63b in {out['runtime_sec']}s -> {AUD_OUT} ===", flush=True)


if __name__ == "__main__":
    main()
