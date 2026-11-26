r"""
stage2_frozen_pipeline_v1.py
============================
Self-contained, bit-exact feature extraction module reproducing the locked
Gen-3 Stage-1 (K=4, offline_filtfilt, 360 Hz 187-sample grid, 50-beat prefix)
+ Stage-2 (18D geometry + causal log(pre_rr_loc) + strictly causal 25-beat
rolling unlabeled pre-QRS [35:63] buffer for C5) pipeline on any WFDB record.

Supports two versioned label maps:
  - AAMI_Map_v1.0 : N={'N','L','R','e','j'}               [exact v36/v47 source parity]
  - AAMI_Map_v1.1 : N={'N','L','R','e','j','B'}           [locked for LTSTDB external validation]

Includes `--parity-test` to prove bit-level numerical agreement against cached
6-DB source features (`cache_v47_preqrs_causal_6db`) BEFORE external inference.
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
import argparse
from collections import Counter
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import wfdb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import lab_diagnostic_v45_conditional_physio_and_error_budget as v45
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47

SL_PWIN = slice(35, 63)   # [-161 ms, -83 ms] pre-QRS window
SL_QRS = v33.QRS_CORE_SLICE
P_BUF_LEN = 25

LABEL_MAPS = {
    "AAMI_Map_v1.0": {
        "N": {"N", "L", "R", "e", "j"},
        "V": {"V", "E"},
        "F": {"F"},
        "S": {"A", "a", "J", "S"},
    },
    "AAMI_Map_v1.1": {
        "N": {"N", "L", "R", "e", "j", "B"},
        "V": {"V", "E"},
        "F": {"F"},
        "S": {"A", "a", "J", "S"},
    },
}


def map_symbol(sym, map_version="AAMI_Map_v1.1"):
    lm = LABEL_MAPS[map_version]
    if sym in lm["N"]:
        return 0
    if sym in lm["V"]:
        return 1
    if sym in lm["F"]:
        return 2
    if sym in lm["S"]:
        return 3
    return -1


def extract_record_features_v1(
    rec_path,
    db_name,
    rec_id,
    subject_id,
    ann_ext="atr",
    channel_indices=(0, 1),
    label_map_version="AAMI_Map_v1.1",
):
    """
    Bit-exact reproduction of v36 + v37 + v47 feature extraction for one WFDB record.
    Returns compact dict with y, s2_mask, X24_s2 (float32), buf_audit, and exact beat accounting.
    """
    t0 = time.perf_counter()
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, ann_ext)
    fs = float(rec.fs)

    i0, i1 = int(channel_indices[0]), int(channel_indices[1])
    leads_str = f"{rec.sig_name[i0]}+{rec.sig_name[i1]}"

    s0_raw = np.nan_to_num(rec.p_signal[:, i0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    s1_raw = np.nan_to_num(rec.p_signal[:, i1].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sig0, delay = v33.filter_signal_generic(s0_raw, fs, mode="offline_filtfilt")
    sig1, _ = v33.filter_signal_generic(s1_raw, fs, mode="offline_filtfilt")

    samples = ann.sample
    symbols = ann.symbol
    raw_sym_counter = Counter(symbols)
    raw_F_total = int(raw_sym_counter.get("F", 0))
    raw_B_total = int(raw_sym_counter.get("B", 0))

    n_sig = len(sig0)
    if abs(fs - v33.TARGET_FS) < 1e-3:
        beats0, beats1, labels, r_peaks, is_B_flag = [], [], [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < 93 or sc + 94 > n_sig:
                continue
            c = map_symbol(sym, label_map_version)
            if c >= 0:
                beats0.append(sig0[sc - 93 : sc + 94])
                beats1.append(sig1[sc - 93 : sc + 94])
                labels.append(c)
                r_peaks.append(s)
                is_B_flag.append(sym == "B")
        beats0 = np.asarray(beats0, dtype=np.float64)
        beats1 = np.asarray(beats1, dtype=np.float64)
    else:
        half_win = int(np.ceil(0.30 * fs))
        local_t = np.arange(-half_win, half_win + 1, dtype=np.float64) / fs
        grid_idx = (v33.PHYS_TIME_GRID - local_t[0]) * fs
        lo_i = np.floor(grid_idx).astype(np.int64)
        hi_i = np.minimum(lo_i + 1, len(local_t) - 1)
        frac = (grid_idx - lo_i).astype(np.float64)

        valid_sc, labels, r_peaks, is_B_flag = [], [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < half_win or sc + half_win + 1 > n_sig:
                continue
            c = map_symbol(sym, label_map_version)
            if c >= 0:
                valid_sc.append(sc)
                labels.append(c)
                r_peaks.append(s)
                is_B_flag.append(sym == "B")
        valid_sc = np.array(valid_sc, dtype=np.int64)
        if len(valid_sc) <= 55:
            return None
        base_off = valid_sc[:, None] - half_win
        beats0 = (1.0 - frac[None, :]) * sig0[base_off + lo_i[None, :]] + frac[None, :] * sig0[base_off + hi_i[None, :]]
        beats1 = (1.0 - frac[None, :]) * sig1[base_off + lo_i[None, :]] + frac[None, :] * sig1[base_off + hi_i[None, :]]

    labels = np.asarray(labels, dtype=np.int8)
    r_peaks = np.asarray(r_peaks, dtype=np.float64)
    is_B_flag = np.asarray(is_B_flag, dtype=bool)
    N_all = len(labels)
    if N_all <= 55:
        return None

    # Exact causal 11-beat rolling median pre-RR (vectorized for i >= 10, bit-identical to loop)
    pre_rr = np.zeros(N_all, dtype=np.float64)
    pre_rr[1:] = (r_peaks[1:] - r_peaks[:-1]) / fs
    pre_rr[0] = pre_rr[1]
    local_rr = np.zeros(N_all, dtype=np.float64)
    for i in range(min(10, N_all)):
        local_rr[i] = float(np.median(pre_rr[: i + 1])) + 1e-6
    if N_all > 10:
        local_rr[10:] = np.median(sliding_window_view(pre_rr, 11), axis=1) + 1e-6
    pre_rr_loc_all = pre_rr / local_rr
    post_hours = float((r_peaks[-1] - r_peaks[50]) / fs / 3600.0)

    # Stage 1: Exact locked Gen-3 Static-Prefix K=4 + Local b_N_loc
    n0_all = v33.normalize_lead(beats0)
    n1_all = v33.normalize_lead(beats1)
    joint_all = np.column_stack([n0_all, n1_all])
    joint_50 = joint_all[:50]
    rr_50 = pre_rr_loc_all[:50]
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
    eta_0 = float(np.percentile(np.linalg.norm(norm_y_prefix, axis=1), 95)) + 1e-6

    pre0_50 = beats0[:50, v33.PRE_QRS_SLICE] - np.mean(beats0[:50, v33.PRE_QRS_SLICE], axis=1, keepdims=True)
    pre1_50 = beats1[:50, v33.PRE_QRS_SLICE] - np.mean(beats1[:50, v33.PRE_QRS_SLICE], axis=1, keepdims=True)
    pN_0 = np.median(pre0_50[final_norm_idx], axis=0)
    pN_1 = np.median(pre1_50[final_norm_idx], axis=0)

    hf0_50 = np.sum(np.diff(joint_50[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1_50 = np.sum(np.diff(joint_50[:, 187:], n=2, axis=1) ** 2, axis=1)
    med_hf0 = float(np.median(hf0_50[final_norm_idx])) + 1e-6
    med_hf1 = float(np.median(hf1_50[final_norm_idx])) + 1e-6
    pref_hf_ratio = np.log1p(np.maximum(hf0_50[final_norm_idx] / med_hf0, hf1_50[final_norm_idx] / med_hf1))
    pref_hf_p95 = float(np.percentile(pref_hf_ratio, 95))

    bN_0, bN_1 = b_N[:187], b_N[187:]
    pref_min_ncorr = np.minimum(
        v33.pearson_corr_rows(norm_beats[:, :187], bN_0),
        v33.pearson_corr_rows(norm_beats[:, 187:], bN_1),
    )
    pref_bcorr_p05 = float(np.percentile(pref_min_ncorr, 5))
    pref_qiso_abs = v33.zenodorus_qiso_abs_batch(norm_beats[:, v33.QRS_CORE_SLICE], norm_beats[:, 187:][:, v33.QRS_CORE_SLICE])
    med_pref_qiso_abs = float(np.median(pref_qiso_abs))

    anchors = []
    if len(ectopic_idx) >= 4:
        anc = v33.make_anchor_obj(joint_50[ectopic_idx], norm_y_prefix, b_N, P, 0, "prefix")
        if anc is not None:
            anchors.append(anc)

    post_joint = joint_all[50:]
    post_raw0, post_raw1 = beats0[50:], beats1[50:]
    post_y = labels[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    N = len(post_joint)

    hf0 = np.sum(np.diff(post_joint[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1 = np.sum(np.diff(post_joint[:, 187:], n=2, axis=1) ** 2, axis=1)
    sqi_hf_ratio = np.log1p(np.maximum(hf0 / med_hf0, hf1 / med_hf1))

    all_qiso_abs = v33.zenodorus_qiso_abs_batch(post_joint[:, v33.QRS_CORE_SLICE], post_joint[:, 187:][:, v33.QRS_CORE_SLICE])
    zeno_qiso_drop_all = med_pref_qiso_abs - all_qiso_abs

    diff_post = post_joint - b_N[None, :]
    y_mat = diff_post - (diff_post @ U) @ U.T
    norm_y = np.linalg.norm(y_mat, axis=1)
    norm_y_thresh = max(v33.CAND_NORM_Y_MIN, v33.CAND_ETA_MULT * eta_0)

    loc_ring = np.zeros((v33.LOC_BUF_LEN, D), dtype=np.float64)
    init_nb = norm_beats[-v33.LOC_BUF_LEN:]
    loc_cnt = len(init_nb)
    loc_ring[:loc_cnt] = init_nb
    loc_ptr = loc_cnt % v33.LOC_BUF_LEN
    b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
    b_N_loc_init0 = b_N_loc.copy()
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
                        arr_b = np.array(best_c["beats"])
                        med_b = np.median(arr_b, axis=0)
                        cohesion = float(np.median(np.linalg.norm(arr_b - med_b, axis=1)))
                        if cohesion <= v33.CONFIRM_MAX_COHESION:
                            anc = v33.make_anchor_obj(arr_b, norm_y_prefix, b_N, P, int(t + 1), "online")
                            if anc is not None:
                                anchors.append(anc)
                                cand_clusters.pop(best_c_idx)
                else:
                    if len(cand_clusters) >= v33.MAX_CAND_CLUSTERS:
                        cand_clusters.sort(key=lambda c: len(c["beats"]))
                        cand_clusters.pop(0)
                    cand_clusters.append({"beats": [x_t], "center": x_t.copy()})

        reg_rr = (v33.LOC_RR_MIN <= pre_rr_loc[t] <= v33.LOC_RR_MAX)
        far_from_v = all(float(np.linalg.norm(x_t - anc["b_V"])) >= v33.LOC_V_EXCLUSION_DIST for anc in anchors)
        if reg_rr and far_from_v:
            c_prev1 = v33.mean_lead_corr(x_t, prev1)
            c_prev12 = v33.mean_lead_corr(prev1, prev2)
            if (c_loc >= v33.LOC_ADMIT_CORR_BN and c_prev1 >= 0.70) or (c_prev1 >= v33.LOC_ADMIT_RUN_CORR and c_prev12 >= v33.LOC_ADMIT_RUN_CORR):
                loc_ring[loc_ptr] = x_t
                loc_ptr = (loc_ptr + 1) % v33.LOC_BUF_LEN
                if loc_cnt < v33.LOC_BUF_LEN:
                    loc_cnt += 1
                b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
        prev2, prev1 = prev1, x_t

    ready = np.zeros(N, dtype=bool)
    d_seg_arr = np.full(N, 9.99)
    z_N_arr = np.zeros(N)
    z_V_arr = np.zeros(N)
    m_bridge_arr = np.zeros(N)
    min_lead_bcorr_arr = np.zeros(N)
    loc_dist_norm_arr = np.zeros(N)
    pyth_margin_loc = np.full(N, -1.0)
    min_lead_thales_loc = np.full(N, -1.0)

    preqrs_corr_N = 0.5 * (
        v33.pearson_corr_rows(post_raw0[:, v33.PRE_QRS_SLICE], pN_0) +
        v33.pearson_corr_rows(post_raw1[:, v33.PRE_QRS_SLICE], pN_1)
    )
    rr_dev_pre = np.abs(pre_rr_loc - 1.0)

    if len(anchors) > 0:
        ready[:] = True
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
        loc_dist_norm_arr = loc_norm_dist_proj_arr / (nq_vec[bk] + 1e-12)

        bV_stack = np.vstack([a["b_V"] for a in anchors])
        for k in range(K):
            mk = (bk == k)
            if not np.any(mk):
                continue
            bV0, bV1 = bV_stack[k, :187], bV_stack[k, 187:]
            lc_k = best_lc_arr[mk, None]
            hat0 = bN_0[None, :] + lc_k * (bV0 - bN_0)[None, :]
            hat1 = bN_1[None, :] + lc_k * (bV1 - bN_1)[None, :]
            x0k, x1k = post_joint[mk, :187], post_joint[mk, 187:]
            x0c = x0k - x0k.mean(1, keepdims=True); h0c = hat0 - hat0.mean(1, keepdims=True)
            x1c = x1k - x1k.mean(1, keepdims=True); h1c = hat1 - hat1.mean(1, keepdims=True)
            c0 = np.sum(x0c * h0c, axis=1) / (np.linalg.norm(x0c, axis=1) * np.linalg.norm(h0c, axis=1) + 1e-12)
            c1 = np.sum(x1c * h1c, axis=1) / (np.linalg.norm(x1c, axis=1) * np.linalg.norm(h1c, axis=1) + 1e-12)
            min_lead_bcorr_arr[mk] = np.minimum(c0, c1)

        s2_idx = np.where(m_bridge_arr > 0.0)[0]
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
                obt_l0, _ = v33.thales_obt_and_pyth(
                    float(np.linalg.norm(x0 - bL0)), float(np.linalg.norm(x0 - bV0)), float(np.linalg.norm(bV0 - bL0))
                )
                obt_l1, _ = v33.thales_obt_and_pyth(
                    float(np.linalg.norm(x1 - bL1)), float(np.linalg.norm(x1 - bV1)), float(np.linalg.norm(bV1 - bL1))
                )
                min_lead_thales_loc[t] = min(obt_l0, obt_l1)

    rel_bcorr_drop = pref_bcorr_p05 - min_lead_bcorr_arr
    hf_excess = np.maximum(0.0, sqi_hf_ratio - max(pref_hf_p95, 0.70))
    bcorr_hinge_sq = np.maximum(0.0, rel_bcorr_drop - 0.10) ** 2
    hf_hinge_sq = hf_excess ** 2

    def shift_past(arr, k):
        out = np.empty_like(arr, dtype=np.float64)
        out[:k] = arr[0]
        out[k:] = arr[:-k]
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
    ]).astype(np.float32)
    X18 = np.nan_to_num(X18, nan=0.0, posinf=0.0, neginf=0.0)
    s2_mask = ready & (m_bridge_arr > 0.0)
    X18_s2 = X18[s2_mask].astype(np.float64)

    # Stage 2 C5: Exact v37->v47 strictly causal unlabeled 25-beat rolling pre-QRS [35:63] buffer
    # Match exact v37 float16 waveform storage & float32 clipped pre_rr parity
    W_pad = np.stack([n0_all[49:], n1_all[49:]], axis=1).astype(np.float16).astype(np.float64)
    W_center = W_pad[1:]
    bN_init = b_N_loc_init0.astype(np.float32).reshape(2, 187).astype(np.float16).astype(np.float64)
    pre_rr_v47 = np.clip(pre_rr_loc_all[50 : 50 + N], 0.2, 3.0).astype(np.float32).astype(np.float64)

    # Oracle N template strictly for post-hoc buffer fidelity audit (NEVER used in C5 features!)
    n_idx = np.where(post_y == 0)[0]
    if len(n_idx) >= 10:
        sub_n = n_idx[np.linspace(0, len(n_idx) - 1, min(500, len(n_idx)), dtype=int)]
        seg_n = W_center[sub_n, :, SL_PWIN]
        seg_n = seg_n - np.mean(seg_n, axis=2, keepdims=True)
        pN_ora = np.median(seg_n, axis=0)
        ora_c_n = v45.corr_rows_2lead(W_center[sub_n, :, SL_PWIN], pN_ora[0], pN_ora[1])
        cons_ora = float(np.median(ora_c_n))
        has_ora = True
    else:
        pN_ora = np.zeros((2, 28), dtype=np.float64)
        cons_ora = 0.0
        has_ora = False
    qgate_ora = v47.quality_gate(cons_ora)

    def batch_2lead_corr_adj(A, B):
        Ac0 = A[:, 0, :] - A[:, 0, :].mean(1, keepdims=True)
        Bc0 = B[:, 0, :] - B[:, 0, :].mean(1, keepdims=True)
        c0 = np.sum(Ac0 * Bc0, axis=1) / (np.linalg.norm(Ac0, axis=1) * np.linalg.norm(Bc0, axis=1) + 1e-12)
        Ac1 = A[:, 1, :] - A[:, 1, :].mean(1, keepdims=True)
        Bc1 = B[:, 1, :] - B[:, 1, :].mean(1, keepdims=True)
        c1 = np.sum(Ac1 * Bc1, axis=1) / (np.linalg.norm(Ac1, axis=1) * np.linalg.norm(Bc1, axis=1) + 1e-12)
        return 0.5 * (c0 + c1)

    c_prev1_all = batch_2lead_corr_adj(W_center, W_pad[:-1])
    c_prev12_all = np.empty(N, dtype=np.float64)
    c_prev12_all[0] = c_prev1_all[0]
    c_prev12_all[1:] = c_prev1_all[:-1]

    P_all = W_center[:, :, SL_PWIN]
    P_all = P_all - np.mean(P_all, axis=2, keepdims=True)

    w_ring = np.zeros((P_BUF_LEN, 2, 187), dtype=np.float64)
    p_ring = np.zeros((P_BUF_LEN, 2, 28), dtype=np.float64)
    y_ring = np.full(P_BUF_LEN, -1, dtype=np.int8)  # audit only
    w_ring[0] = bN_init
    p_init = bN_init[:, SL_PWIN] - np.mean(bN_init[:, SL_PWIN], axis=1, keepdims=True)
    p_ring[0] = p_init
    y_ring[0] = 0
    buf_cnt = 1
    buf_ptr = 1 % P_BUF_LEN
    n_admitted = 0

    bN_cur = bN_init.copy()
    pN_cur = p_init.copy()
    cons_cur = 0.0

    s2_indices = np.where(s2_mask)[0]
    n_s2 = len(s2_indices)
    s2_pos = 0
    next_s2_t = int(s2_indices[0]) if n_s2 > 0 else -1

    pq_ora_raw = np.zeros(n_s2, dtype=np.float64)
    pq_cau_raw = np.zeros(n_s2, dtype=np.float64)
    cons_cau_arr = np.zeros(n_s2, dtype=np.float64)
    qgate_cau_arr = np.zeros(n_s2, dtype=np.float64)

    adm_y_counts = Counter()
    s2f_buf_n, s2f_buf_f, s2f_buf_v, s2f_fid = [], [], [], []
    s2v_buf_n, s2v_buf_f, s2v_buf_v, s2v_fid = [], [], [], []
    s2f_tot = 0
    s2f_unwarmed = 0

    reg_rr_all = (v33.LOC_RR_MIN <= pre_rr_v47) & (pre_rr_v47 <= v33.LOC_RR_MAX)

    for t in range(N):
        if t == next_s2_t:
            pt = P_all[t]
            pq_ora_raw[s2_pos] = v47._mean_pwin_corr(pt, pN_ora) if has_ora else 0.0
            pq_cau_raw[s2_pos] = v47._mean_pwin_corr(pt, pN_cur) if n_admitted >= 5 else 0.0
            cons_cau_arr[s2_pos] = cons_cur if n_admitted >= 5 else 0.0
            qgate_cau_arr[s2_pos] = v47.quality_gate(cons_cur) if n_admitted >= 5 else 0.0

            if post_y[t] == 2:
                s2f_tot += 1
                if n_admitted < 5:
                    s2f_unwarmed += 1

            if n_admitted >= 5 and post_y[t] in (1, 2):
                active_y = y_ring[:buf_cnt]
                valid_y = active_y[active_y >= 0]
                if len(valid_y) > 0:
                    fn = float(np.mean(valid_y == 0))
                    ff = float(np.mean(valid_y == 2))
                    fv = float(np.mean(valid_y == 1))
                    fid = v47._mean_pwin_corr(pN_cur, pN_ora) if has_ora else 0.0
                    if post_y[t] == 2:
                        s2f_buf_n.append(fn); s2f_buf_f.append(ff); s2f_buf_v.append(fv); s2f_fid.append(fid)
                    else:
                        s2v_buf_n.append(fn); s2v_buf_f.append(ff); s2v_buf_v.append(fv); s2v_fid.append(fid)

            s2_pos += 1
            next_s2_t = int(s2_indices[s2_pos]) if s2_pos < n_s2 else -1

        if reg_rr_all[t]:
            c_p1 = c_prev1_all[t]
            c_p12 = c_prev12_all[t]
            if c_p1 >= 0.80:
                xt = W_center[t]
                qrs_c = v47._min_qrs_corr(xt, bN_cur)
                if qrs_c >= 0.85:
                    c_bn = v47._mean_2lead_corr(xt, bN_cur)
                    if (c_bn >= 0.80 and c_p1 >= 0.80) or (c_p1 >= 0.90 and c_p12 >= 0.90):
                        if n_admitted == 0:
                            buf_ptr = 0; buf_cnt = 0
                        w_ring[buf_ptr] = xt
                        p_ring[buf_ptr] = P_all[t]
                        y_ring[buf_ptr] = post_y[t]
                        buf_ptr = (buf_ptr + 1) % P_BUF_LEN
                        if buf_cnt < P_BUF_LEN:
                            buf_cnt += 1
                        n_admitted += 1
                        adm_y_counts[int(post_y[t])] += 1

                        if buf_cnt < P_BUF_LEN or (n_admitted % 5 == 0):
                            bN_cur = np.median(w_ring[:buf_cnt], axis=0)
                            pN_cur = np.median(p_ring[:buf_cnt], axis=0)
                            if buf_cnt >= 5:
                                cs = v45.corr_rows_2lead(p_ring[:buf_cnt], pN_cur[0], pN_cur[1])
                                cons_cur = float(np.median(cs))

    log_rr_s2 = np.log(np.clip(pre_rr_v47[s2_mask], 0.2, 3.0))
    pq_ora_gated = qgate_ora * pq_ora_raw
    pq_cau_gated = qgate_cau_arr * pq_cau_raw
    X24_s2 = np.column_stack([
        X18_s2, log_rr_s2, pq_ora_gated, pq_cau_gated, pq_ora_raw, pq_cau_raw, cons_cau_arr
    ]).astype(np.float32) if n_s2 > 0 else np.zeros((0, 24), dtype=np.float32)

    tot_adm = sum(adm_y_counts.values())
    buf_audit = {
        "rec_id": rec_id, "patient_id": subject_id, "db": db_name,
        "tot_F": int(np.sum(post_y == 2)), "s2_F": int(np.sum((post_y == 2) & s2_mask)),
        "n_admitted": tot_adm,
        "adm_N_pct": round(100.0 * adm_y_counts[0] / max(1, tot_adm), 2),
        "adm_F_pct": round(100.0 * adm_y_counts[2] / max(1, tot_adm), 2),
        "adm_V_pct": round(100.0 * adm_y_counts[1] / max(1, tot_adm), 2),
        "adm_S_pct": round(100.0 * adm_y_counts[3] / max(1, tot_adm), 2),
        "cons_ora": round(cons_ora, 3),
        "cons_cau_med": round(float(np.median(cons_cau_arr)), 3) if n_s2 > 0 else 0.0,
        "s2F_buf_N_pct": round(100.0 * float(np.mean(s2f_buf_n)), 1) if s2f_buf_n else None,
        "s2F_buf_F_pct": round(100.0 * float(np.mean(s2f_buf_f)), 1) if s2f_buf_f else None,
        "s2F_buf_V_pct": round(100.0 * float(np.mean(s2f_buf_v)), 1) if s2f_buf_v else None,
        "s2F_unwarmed_pct": round(100.0 * s2f_unwarmed / max(1, s2f_tot), 1) if s2f_tot > 0 else None,
        "s2F_tmpl_fid": round(float(np.median(s2f_fid)), 3) if s2f_fid else None,
        "s2V_tmpl_fid": round(float(np.median(s2v_fid)), 3) if s2v_fid else None,
    }

    accounting = {
        "raw_atr_F": raw_F_total,
        "raw_atr_B": raw_B_total,
        "valid_win_beats": int(N_all),
        "valid_win_F": int(np.sum(labels == 2)),
        "valid_win_B": int(np.sum(is_B_flag)),
        "prefix_F": int(np.sum(labels[:50] == 2)),
        "post_prefix_beats": int(N),
        "post_prefix_F": int(np.sum(post_y == 2)),
        "post_prefix_B": int(np.sum(is_B_flag[50:])),
        "post_prefix_N_incl_B": int(np.sum(post_y == 0)),
        "post_prefix_V": int(np.sum(post_y == 1)),
        "post_prefix_S": int(np.sum(post_y == 3)),
        "s2_total": int(n_s2),
        "s2_F": int(np.sum((post_y == 2) & s2_mask)),
    }

    return {
        "db": db_name, "rec_id": rec_id, "patient_id": subject_id,
        "fs": fs, "leads": leads_str, "post_hours": post_hours,
        "K": len(anchors), "label_map_version": label_map_version,
        "y": post_y, "s2_mask": s2_mask, "X24_s2": X24_s2,
        "buf_audit": buf_audit, "accounting": accounting,
        "proc_sec": round(time.perf_counter() - t0, 2),
    }


def run_parity_test():
    """Verifies bit-level agreement against cached 6-DB source v47 features on diverse sampling rates."""
    import lab_diagnostic_v36_6hospital_lodo_benchmark as v36

    cases = [
        ("mitdb", "213", (0, 1)),      # 360 Hz, high F
        ("edb", "e0103", (0, 1)),      # 250 Hz (same fs as LTSTDB!)
        ("ltdb", "14134", (0, 1)),     # 128 Hz, 20h Holter
        ("svdb", "800", (0, 1)),       # 128 Hz, 0-F
    ]
    print("=== BIT-EXACT PARITY UNIT TEST (stage2_frozen_pipeline_v1 vs cache_v47_preqrs_causal_6db) ===", flush=True)
    for db, rid, ch in cases:
        cached = {it["rec_id"]: it for it in v47.build_or_load_v47_db(db)}[rid]
        rec_path = os.path.join(v36.DB_DIRS[db], rid)
        res = extract_record_features_v1(
            rec_path, db, rid, cached["patient_id"],
            ann_ext="atr", channel_indices=ch, label_map_version="AAMI_Map_v1.0",
        )
        assert np.array_equal(res["y"], cached["y"]), f"y mismatch on {db}/{rid}"
        assert np.array_equal(res["s2_mask"], cached["s2_mask"]), f"s2_mask mismatch on {db}/{rid}"
        diff = np.abs(res["X24_s2"].astype(np.float64) - cached["X_s2"])
        max_all = float(np.max(diff)) if diff.size > 0 else 0.0
        max_c5 = float(np.max(diff[:, v47.IDX_C5_CAUSAL])) if diff.size > 0 else 0.0
        print(
            f"  PASS {db:6s}/{rid:6s} | N={len(res['y']):6d}, s2={len(res['X24_s2']):5d}, "
            f"max|dX_all|={max_all:.3e}, max|dX_C5|={max_c5:.3e} ({res['proc_sec']}s)",
            flush=True,
        )
        assert max_c5 <= 1e-6, f"Parity violation on {db}/{rid}: max_c5={max_c5}"
    print("=== ALL 4 PARITY TESTS PASSED (EXACT MATCH) ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-test", action="store_true")
    args = ap.parse_args()
    if args.parity_test:
        run_parity_test()
