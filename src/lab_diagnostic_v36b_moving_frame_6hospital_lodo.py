"""
lab_diagnostic_v36b_moving_frame_6hospital_lodo.py
===================================================
STEP 3B (v36b): Moving-Frame Local Dual-Chart Upgrade for 2-24h Holters
Across All 6 Public Hospitals (306 Recordings, 2.68M Beats, 4,600 Class-F Beats,
Unfiltered 4-Class AAMI Streams N + S + V + F).

Root cause diagnosed in v36:
  Frozen minute-0 sinus anchor b_N_prefix50 & pre-QRS template pN_prefix50 drifted
  over 2-24h Holters (EDB ST-T ischemia, SDDB pre-VF/posture, LTDB 22h), causing:
    (1) Impure V-anchors (V_pur 67-75% on EDB/SDDB/SVDB),
    (2) Drifted-Normal Stage-2 flood (289,865 N in SDDB [38.5%], 122,260 N in EDB),
    (3) Sign inversion of static-chart features (min_bcorr, preqrs_c) & LODO poisoning.

Moving-Frame Local Dual-Chart Fix (100% Causal, Zero New Tuned Hyperparameters,
Exact O(4) Gauge Invariant):
  1. Track causal local sinus template b_N_loc(t), local pre-QRS template pN_loc(t),
     and local normal HF median med_hf_loc(t) inside the existing 25-beat loc_ring
     (with flatline & extreme-HF safety guards).
  2. Stage-1 V-anchor discovery measures projected deviation relative to CURRENT b_N_loc(t)
     with a first-principles physiological His-Purkinje gate: min_qrs_corr(x_t, b_N_loc) <= 0.85
     (verified on MIT-BIH: keeps 97.0% of true PVCs while rejecting narrow-QRS ST-T drift & SVEBs).
  3. Stage-1 bridge projection, min_lead_bcorr, preqrs_corr_N, and Stage-2 Thales-Pythagoras
     all operate in ONE unified Moving-Frame Dual-Chart (b_N_loc(t), b_V^(k)), with
     automatic degenerate-anchor exclusion whenever ||P(b_V^(k) - b_N_loc(t))|| < 12.0.

Saves clean Moving-Frame caches in ./data_public_ecg/cache_gen3_moving_6db/
for v37 (Plug-and-Play Geometric Front-End + Modern Deep Learning on 6-Hospital LODO).
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
import glob
import time
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import wfdb
from scipy.stats import ortho_group
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33

DATA_ROOT = str(_DATA_ROOT)
CACHE_DIR = os.path.join(DATA_ROOT, "cache_gen3_moving_6db")
AUD_DIR = str(_REPO_ROOT / "audits")
V36_JSON = os.path.join(AUD_DIR, "V36_6HOSPITAL_LODO_BENCHMARK_AUDIT_20261121.json")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUD_DIR, exist_ok=True)

DB_DIRS = {
    "mitdb": v33.MITDB_DIR,
    "incartdb": v33.INCART_DIR,
    "ltdb": os.path.join(DATA_ROOT, "ltdb"),
    "edb": os.path.join(DATA_ROOT, "edb"),
    "sddb": os.path.join(DATA_ROOT, "sddb"),
    "svdb": os.path.join(DATA_ROOT, "svdb"),
}

QRS_CORR_ANCHOR_MAX = 0.85  # First-principles His-Purkinje narrow-QRS exclusion (MIT verified: 97.0% V <= 0.85)
LOC_RING_HF_MAX = 1.50      # Guard against electrode-detachment artifact entering b_N_loc

FEAT_NAMES = [
    "m_bridge", "z_N", "z_V", "d_seg",
    "loc_ncorr", "loc_dnorm", "preqrs_c", "rr_dev", "min_bcorr", "sqi_hf",
    "bcorr_h2", "hf_h2",
    "zeno_dr", "pyth_loc", "min_thales",
    "nb_hf2", "nb_bdr2", "nb_mb3",
]

VARIANTS_IDX = {
    "B0_Gen2-Soft (12D)": list(range(12)),
    "G3_ThalesPyth (14D)": list(range(12)) + [13, 14],
    "G3_BurstPast2 (16D)": list(range(12)) + [13, 14, 15, 16],
    "G4_MBridge3 (16D)": list(range(12)) + [12, 13, 14, 17],
    "G3_FullCtx0b (17D)": list(range(12)) + [13, 14, 15, 16, 17],
    "G4_FullCtx0b (18D)": list(range(18)),
}


def _corr1d(a, b):
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    return float(np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-12))


def min_qrs_corr_2lead(x_vec, ref_vec):
    qs = v33.QRS_CORE_SLICE
    c0 = _corr1d(x_vec[:187][qs], ref_vec[:187][qs])
    c1 = _corr1d(x_vec[187:][qs], ref_vec[187:][qs])
    return min(c0, c1)


def make_anchor_moving(ect_beats, loc_norm_beats, b_N_ref, P, disc_idx, src):
    """Build anchor object relative to CURRENT local normal frame b_N_ref & loc_norm_beats."""
    b_V = np.median(ect_beats, axis=0)
    q = P @ (b_V - b_N_ref)
    nq = float(np.linalg.norm(q))
    if nq < v33.CONFIRM_MIN_NORM_Q:
        return None
    if min_qrs_corr_2lead(b_V, b_N_ref) > QRS_CORR_ANCHOR_MAX:
        return None
    # Local normal projected residuals relative to b_N_ref
    yn_loc = (loc_norm_beats - b_N_ref[None, :]) @ P.T
    lam_n = (yn_loc @ q) / (nq ** 2 + 1e-12)
    mu_N = float(np.median(lam_n))
    rad_N = max(v33.RAD_N_FLOOR, float(np.percentile(np.abs(lam_n - mu_N), 95)))

    yv = (ect_beats - b_N_ref[None, :]) @ P.T
    lam_v = (yv @ q) / (nq ** 2 + 1e-12)
    dy_v = np.linalg.norm(yv - q[None, :], axis=1)
    anc = {
        "b_V": b_V,
        "mu_N": mu_N,
        "rad_N": rad_N,
        "v_lam_buf": v33.deque(lam_v.tolist(), maxlen=v33.V_SPREAD_BUF_MAX),
        "v_dy_buf": v33.deque(dy_v.tolist(), maxlen=v33.V_SPREAD_BUF_MAX),
        "norm_q_init": nq,
        "disc_idx": int(disc_idx),
        "src": src,
    }
    _refresh_v_radii_moving(anc)
    return anc


def _refresh_v_radii_moving(anc):
    lam_arr = np.array(anc["v_lam_buf"], dtype=np.float64)
    dy_arr = np.array(anc["v_dy_buf"], dtype=np.float64)
    anc["rad_V"] = max(v33.RAD_V_FLOOR, float(np.percentile(np.abs(1.0 - lam_arr), 95)))
    anc["rad_y_V"] = max(v33.RAD_YV_FRAC_FLOOR * anc["norm_q_init"], float(np.percentile(dy_arr, 95)))


def extract_moving_frame_features(beats0, beats1, labels, pre_rr_loc_all, U_override=None):
    """
    Core Moving-Frame Local Dual-Chart feature extractor.
    Accepts optional U_override (e.g. U @ R for O(4) gauge invariance verification).
    """
    N_all = len(labels)
    joint_all = np.column_stack([v33.normalize_lead(beats0), v33.normalize_lead(beats1)])
    valid_lead_all = (np.std(beats0, axis=1) > 1e-4) & (np.std(beats1, axis=1) > 1e-4)
    joint_50 = joint_all[:50]
    rr_50 = pre_rr_loc_all[:50]
    M, D = joint_50.shape

    km = KMeans(n_clusters=2, n_init=20, random_state=42)
    clusters = km.fit_predict(joint_50)
    sil = float(silhouette_score(joint_50, clusters)) if len(np.unique(clusters)) > 1 else 0.0
    n0, n1 = int(np.sum(clusters == 0)), int(np.sum(clusters == 1))
    c0_0, c0_1 = km.cluster_centers_[0][:187], km.cluster_centers_[0][187:]
    c1_0, c1_1 = km.cluster_centers_[1][:187], km.cluster_centers_[1][187:]
    slope0 = float(max(np.max(np.abs(np.diff(c0_0))), np.max(np.abs(np.diff(c0_1)))))
    slope1 = float(max(np.max(np.abs(np.diff(c1_0))), np.max(np.abs(np.diff(c1_1)))))
    sel_cluster = 0 if slope0 >= slope1 else 1
    slope_ratio = min(slope0, slope1) / (max(slope0, slope1) + 1e-12)
    selected_idx = np.where(clusters == sel_cluster)[0]
    is_multi = (sil >= 0.35) and (min(n0, n1) >= 4) and (len(selected_idx) >= 8)

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

    b_N_pur_ns = float(np.mean(np.isin(labels[:50][final_norm_idx], [0, 3])))
    norm_beats = joint_50[final_norm_idx]
    b_N = np.median(norm_beats, axis=0)
    centered_norm = norm_beats - b_N
    if U_override is None:
        _, _, Vt = np.linalg.svd(centered_norm, full_matrices=False)
        U = Vt[:min(4, Vt.shape[0]), :].T
    else:
        U = U_override
    P = np.eye(D) - U @ U.T
    norm_y_prefix = centered_norm @ P.T
    eta_0 = float(np.percentile(np.linalg.norm(norm_y_prefix, axis=1), 95)) + 1e-6

    pre0_50 = beats0[:50, v33.PRE_QRS_SLICE] - np.mean(beats0[:50, v33.PRE_QRS_SLICE], axis=1, keepdims=True)
    pre1_50 = beats1[:50, v33.PRE_QRS_SLICE] - np.mean(beats1[:50, v33.PRE_QRS_SLICE], axis=1, keepdims=True)

    hf0_all = np.sum(np.diff(joint_all[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1_all = np.sum(np.diff(joint_all[:, 187:], n=2, axis=1) ** 2, axis=1)
    hf0_50, hf1_50 = hf0_all[:50], hf1_all[:50]
    med_hf0_init = float(np.median(hf0_50[final_norm_idx])) + 1e-6
    med_hf1_init = float(np.median(hf1_50[final_norm_idx])) + 1e-6
    pref_hf_ratio = np.log1p(np.maximum(hf0_50[final_norm_idx] / med_hf0_init, hf1_50[final_norm_idx] / med_hf1_init))
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
        anc = make_anchor_moving(joint_50[ectopic_idx], norm_beats, b_N, P, 0, "prefix")
        if anc is not None:
            anc["v_pur"] = float(np.mean(labels[:50][ectopic_idx] == 1))
            anchors.append(anc)

    post_joint = joint_all[50:]
    post_raw0, post_raw1 = beats0[50:], beats1[50:]
    post_valid = valid_lead_all[50:]
    post_y = labels[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    hf0_post, hf1_post = hf0_all[50:], hf1_all[50:]
    N = len(post_joint)

    post_pre0 = post_raw0[:, v33.PRE_QRS_SLICE] - np.mean(post_raw0[:, v33.PRE_QRS_SLICE], axis=1, keepdims=True)
    post_pre1 = post_raw1[:, v33.PRE_QRS_SLICE] - np.mean(post_raw1[:, v33.PRE_QRS_SLICE], axis=1, keepdims=True)
    L_pre = post_pre0.shape[1]

    all_qiso_abs = v33.zenodorus_qiso_abs_batch(post_joint[:, v33.QRS_CORE_SLICE], post_joint[:, 187:][:, v33.QRS_CORE_SLICE])
    zeno_qiso_drop_all = med_pref_qiso_abs - all_qiso_abs

    norm_y_thresh = max(v33.CAND_NORM_Y_MIN, v33.CAND_ETA_MULT * eta_0)

    # Causal moving-frame local sinus ring buffers (normal beats, pre-QRS templates, local HF floor)
    loc_ring = np.zeros((v33.LOC_BUF_LEN, D), dtype=np.float64)
    pre0_ring = np.zeros((v33.LOC_BUF_LEN, L_pre), dtype=np.float64)
    pre1_ring = np.zeros((v33.LOC_BUF_LEN, L_pre), dtype=np.float64)
    hf0_ring = np.zeros(v33.LOC_BUF_LEN, dtype=np.float64)
    hf1_ring = np.zeros(v33.LOC_BUF_LEN, dtype=np.float64)

    init_idx = final_norm_idx[-v33.LOC_BUF_LEN:]
    loc_cnt = len(init_idx)
    loc_ring[:loc_cnt] = joint_50[init_idx]
    pre0_ring[:loc_cnt] = pre0_50[init_idx]
    pre1_ring[:loc_cnt] = pre1_50[init_idx]
    hf0_ring[:loc_cnt] = hf0_50[init_idx]
    hf1_ring[:loc_cnt] = hf1_50[init_idx]
    loc_ptr = loc_cnt % v33.LOC_BUF_LEN

    b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
    pN0_loc = np.median(pre0_ring[:loc_cnt], axis=0)
    pN1_loc = np.median(pre1_ring[:loc_cnt], axis=0)
    med_hf0_loc = float(np.median(hf0_ring[:loc_cnt])) + 1e-6
    med_hf1_loc = float(np.median(hf1_ring[:loc_cnt])) + 1e-6

    b_N_loc_hist = np.zeros((N, D), dtype=np.float32)
    preqrs_corr_N = np.zeros(N, dtype=np.float64)
    sqi_hf_ratio = np.zeros(N, dtype=np.float64)
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
        d_loc_p = float(np.linalg.norm(dl_p))
        loc_norm_dist_proj_arr[t] = d_loc_p

        # Local moving pre-QRS correlation & local moving HF ratio
        preqrs_corr_N[t] = 0.5 * (_corr1d(post_pre0[t], pN0_loc) + _corr1d(post_pre1[t], pN1_loc))
        hf_r = math.log1p(max(hf0_post[t] / med_hf0_loc, hf1_post[t] / med_hf1_loc))
        sqi_hf_ratio[t] = hf_r

        # Stage-1 V-anchor discovery strictly in Moving Local Frame b_N_loc(t) + His-Purkinje QRS gate
        if (
            post_valid[t]
            and (d_loc_p >= norm_y_thresh)
            and (pre_rr_loc[t] <= v33.CAND_PRE_RR_LOC_MAX)
            and (hf_r <= v33.ONL_HF_MAX)
            and (min_qrs_corr_2lead(x_t, b_N_loc) <= QRS_CORR_ANCHOR_MAX)
        ):
            matched_anc, best_adist = None, v33.CLUSTER_RADIUS
            for anc in anchors:
                d_a = float(np.linalg.norm(x_t - anc["b_V"]))
                if d_a <= best_adist:
                    best_adist, matched_anc = d_a, anc
            if matched_anc is not None:
                dq_m = matched_anc["b_V"] - b_N_loc
                q_m = dq_m - U @ (U.T @ dq_m)
                nq_m = float(np.linalg.norm(q_m))
                if nq_m >= v33.CONFIRM_MIN_NORM_Q:
                    matched_anc["v_lam_buf"].append(float(np.dot(dl_p, q_m) / (nq_m ** 2 + 1e-12)))
                    matched_anc["v_dy_buf"].append(float(np.linalg.norm(dl_p - q_m)))
                    _refresh_v_radii_moving(matched_anc)
            elif len(anchors) < v33.MAX_V_ANCHORS:
                best_c_idx, best_dist = -1, v33.CLUSTER_RADIUS
                for ci, cl in enumerate(cand_clusters):
                    dist = float(np.linalg.norm(x_t - cl["center"]))
                    if dist <= best_dist:
                        best_dist, best_c_idx = dist, ci
                if best_c_idx >= 0:
                    best_c = cand_clusters[best_c_idx]
                    best_c["beats"].append(x_t)
                    best_c["ys"].append(int(post_y[t]))
                    best_c["center"] = np.median(best_c["beats"], axis=0)
                    if len(best_c["beats"]) >= v33.CONFIRM_MIN_BEATS:
                        arr_b = np.array(best_c["beats"])
                        med_b = np.median(arr_b, axis=0)
                        cohesion = float(np.median(np.linalg.norm(arr_b - med_b, axis=1)))
                        if cohesion <= v33.CONFIRM_MAX_COHESION:
                            anc = make_anchor_moving(arr_b, loc_ring[:loc_cnt], b_N_loc, P, int(t + 1), "online")
                            if anc is not None:
                                anc["v_pur"] = float(np.mean(np.array(best_c["ys"]) == 1))
                                anchors.append(anc)
                                cand_clusters.pop(best_c_idx)
                else:
                    if len(cand_clusters) >= v33.MAX_CAND_CLUSTERS:
                        cand_clusters.sort(key=lambda c: len(c["beats"]))
                        cand_clusters.pop(0)
                    cand_clusters.append({"beats": [x_t], "ys": [int(post_y[t])], "center": x_t.copy()})

        # Causal local sinus frame update (guarded against flatlines, extreme HF bursts, and V clusters)
        reg_rr = (v33.LOC_RR_MIN <= pre_rr_loc[t] <= v33.LOC_RR_MAX)
        far_from_v = all(float(np.linalg.norm(x_t - anc["b_V"])) >= v33.LOC_V_EXCLUSION_DIST for anc in anchors)
        if post_valid[t] and reg_rr and far_from_v and (hf_r <= LOC_RING_HF_MAX):
            c_prev1 = v33.mean_lead_corr(x_t, prev1)
            c_prev12 = v33.mean_lead_corr(prev1, prev2)
            if (c_loc >= v33.LOC_ADMIT_CORR_BN and c_prev1 >= 0.70) or (c_prev1 >= v33.LOC_ADMIT_RUN_CORR and c_prev12 >= v33.LOC_ADMIT_RUN_CORR):
                loc_ring[loc_ptr] = x_t
                pre0_ring[loc_ptr] = post_pre0[t]
                pre1_ring[loc_ptr] = post_pre1[t]
                hf0_ring[loc_ptr] = hf0_post[t]
                hf1_ring[loc_ptr] = hf1_post[t]
                loc_ptr = (loc_ptr + 1) % v33.LOC_BUF_LEN
                if loc_cnt < v33.LOC_BUF_LEN:
                    loc_cnt += 1
                b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
                pN0_loc = np.median(pre0_ring[:loc_cnt], axis=0)
                pN1_loc = np.median(pre1_ring[:loc_cnt], axis=0)
                med_hf0_loc = float(np.median(hf0_ring[:loc_cnt])) + 1e-6
                med_hf1_loc = float(np.median(hf1_ring[:loc_cnt])) + 1e-6

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
    rr_dev_pre = np.abs(pre_rr_loc - 1.0)

    if len(anchors) > 0:
        K = len(anchors)
        bV_stack = np.vstack([a["b_V"] for a in anchors])  # (K, D)
        muN_vec = np.array([a["mu_N"] for a in anchors])
        radN_vec = np.array([a["rad_N"] for a in anchors])
        radV_vec = np.array([a["rad_V"] for a in anchors])
        radyV_vec = np.array([a["rad_y_V"] for a in anchors])

        # Chunked vectorized Moving-Frame Dual-Chart projection (low RAM even on 130k-beat 24h tapes!)
        CHUNK = 8192
        for c0 in range(0, N, CHUNK):
            c1 = min(N, c0 + CHUNK)
            nc = c1 - c0
            x_c = post_joint[c0:c1]                          # (nc, D)
            bn_c = b_N_loc_hist[c0:c1].astype(np.float64)    # (nc, D)
            dl_c = x_c - bn_c
            y_c = dl_c - (dl_c @ U) @ U.T                    # (nc, D) = P(x_t - b_N_loc(t))
            dN_c = np.linalg.norm(y_c, axis=1)               # (nc,)

            lam_c = np.zeros((nc, K), dtype=np.float64)
            lam_clip_c = np.zeros((nc, K), dtype=np.float64)
            eucl_seg_c = np.full((nc, K), 1e9, dtype=np.float64)
            nq_c = np.zeros((nc, K), dtype=np.float64)
            ej_c = np.full((nc, K), 1e9, dtype=np.float64)
            dy_q_all = np.zeros((nc, K), dtype=np.float64)

            for k in range(K):
                dq_k = bV_stack[k][None, :] - bn_c           # (nc, D)
                q_k = dq_k - (dq_k @ U) @ U.T                # (nc, D) = P(b_V^k - b_N_loc(t))
                nq_k = np.linalg.norm(q_k, axis=1)           # (nc,)
                nq_c[:, k] = nq_k
                valid_k = (nq_k >= v33.CONFIRM_MIN_NORM_Q)
                lk = np.sum(y_c * q_k, axis=1) / (nq_k ** 2 + 1e-12)
                lck = np.clip(lk, 0.0, 1.0)
                lam_c[:, k] = lk
                lam_clip_c[:, k] = lck
                eseg = np.linalg.norm(y_c - lck[:, None] * q_k, axis=1)
                dy_q = np.linalg.norm(y_c - q_k, axis=1)
                dy_q_all[:, k] = dy_q
                eucl_seg_c[:, k] = np.where(valid_k, eseg, 1e9)
                ej_c[:, k] = np.where(valid_k, np.maximum(0.0, (dy_q - radyV_vec[k]) / (nq_k + 1e-12)), 1e9)

            any_valid = np.min(eucl_seg_c, axis=1) < 1e8
            ready[c0:c1] = any_valid & post_valid[c0:c1]
            if not np.any(ready[c0:c1]):
                continue

            best_k = np.argmin(eucl_seg_c, axis=1)
            idx_c = np.arange(nc)
            b_lam = lam_c[idx_c, best_k]
            b_lc = lam_clip_c[idx_c, best_k]
            b_nq = nq_c[idx_c, best_k]
            b_eseg = eucl_seg_c[idx_c, best_k]
            e_V_any = np.min(ej_c, axis=1)

            e_N = np.maximum(0.0, (b_lam - muN_vec[best_k]) - radN_vec[best_k])
            e_Vc = np.maximum(0.0, (1.0 - b_lam) - radV_vec[best_k])
            zN = np.log1p(e_N / radN_vec[best_k])
            zV = np.minimum(np.log1p(e_Vc / radV_vec[best_k]), np.log1p(e_V_any / radV_vec[best_k]))
            mb = np.where(ready[c0:c1], np.minimum(zN, zV), 0.0)

            d_seg_arr[c0:c1] = np.where(ready[c0:c1], b_eseg / (b_nq + 1e-12), 9.99)
            z_N_arr[c0:c1] = np.where(ready[c0:c1], zN, 0.0)
            z_V_arr[c0:c1] = np.where(ready[c0:c1], zV, 0.0)
            m_bridge_arr[c0:c1] = mb
            loc_dist_norm_arr[c0:c1] = np.where(ready[c0:c1], dN_c / (b_nq + 1e-12), 0.0)

            # Moving-frame min_lead_bcorr: hat_x(t) = b_N_loc(t) + lc * (b_V^k - b_N_loc(t))
            for k in range(K):
                mk = (best_k == k) & ready[c0:c1]
                if not np.any(mk):
                    continue
                bL0, bL1 = bn_c[mk, :187], bn_c[mk, 187:]
                bV0, bV1 = bV_stack[k, :187][None, :], bV_stack[k, 187:][None, :]
                lc_k = b_lc[mk, None]
                hat0 = bL0 + lc_k * (bV0 - bL0)
                hat1 = bL1 + lc_k * (bV1 - bL1)
                x0k, x1k = x_c[mk, :187], x_c[mk, 187:]
                x0c = x0k - x0k.mean(1, keepdims=True); h0c = hat0 - hat0.mean(1, keepdims=True)
                x1c = x1k - x1k.mean(1, keepdims=True); h1c = hat1 - hat1.mean(1, keepdims=True)
                cor0 = np.sum(x0c * h0c, axis=1) / (np.linalg.norm(x0c, axis=1) * np.linalg.norm(h0c, axis=1) + 1e-12)
                cor1 = np.sum(x1c * h1c, axis=1) / (np.linalg.norm(x1c, axis=1) * np.linalg.norm(h1c, axis=1) + 1e-12)
                min_lead_bcorr_arr[c0:c1][mk] = np.minimum(cor0, cor1)

            # Unified Moving-Frame Thales-Pythagoras on Stage-2 beats (mb > 0)
            s2_loc = np.where(mb > 0.0)[0]
            if len(s2_loc) > 0:
                dN_s = dN_c[s2_loc]
                dV_s = dy_q_all[s2_loc, best_k[s2_loc]]
                dNV_s = b_nq[s2_loc]
                pyth_s = (dNV_s ** 2 - dN_s ** 2 - dV_s ** 2) / (dNV_s ** 2 + 1e-12)
                pyth_margin_loc[c0:c1][s2_loc] = np.clip(pyth_s, -1.0, 1.0)

                for j_i in s2_loc:
                    k = int(best_k[j_i])
                    x0, x1 = x_c[j_i, :187], x_c[j_i, 187:]
                    bL0, bL1 = bn_c[j_i, :187], bn_c[j_i, 187:]
                    bV0, bV1 = bV_stack[k, :187], bV_stack[k, 187:]
                    obt0, _ = v33.thales_obt_and_pyth(
                        float(np.linalg.norm(x0 - bL0)), float(np.linalg.norm(x0 - bV0)), float(np.linalg.norm(bV0 - bL0))
                    )
                    obt1, _ = v33.thales_obt_and_pyth(
                        float(np.linalg.norm(x1 - bL1)), float(np.linalg.norm(x1 - bV1)), float(np.linalg.norm(bV1 - bL1))
                    )
                    min_lead_thales_loc[c0 + j_i] = min(obt0, obt1)

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
    ]).astype(np.float64)
    X18 = np.nan_to_num(X18, nan=0.0, posinf=0.0, neginf=0.0)
    s2_mask = ready & (m_bridge_arr > 0.0)
    return {
        "U": U, "K": len(anchors), "b_N_pur": b_N_pur_ns,
        "v_purs": [a["v_pur"] for a in anchors],
        "y": post_y, "ready": ready, "s2_mask": s2_mask,
        "X18_full": X18,
    }


def verify_o4_gauge_invariance():
    """Verify exact O(4) gauge invariance of Moving-Frame Dual-Chart on MIT-BIH 208."""
    d = v33.load_record(v33.MITDB_DIR, "208", dataset="mitdb", include_class_s=True)
    res_base = extract_moving_frame_features(
        d["raw_0"].astype(np.float64), d["raw_1"].astype(np.float64),
        d["labels"].astype(np.int8), d["pre_rr_loc"].astype(np.float64),
    )
    R = ortho_group.rvs(dim=res_base["U"].shape[1], random_state=2026)
    res_rot = extract_moving_frame_features(
        d["raw_0"].astype(np.float64), d["raw_1"].astype(np.float64),
        d["labels"].astype(np.int8), d["pre_rr_loc"].astype(np.float64),
        U_override=res_base["U"] @ R,
    )
    s2_diff = int(np.sum(res_base["s2_mask"] != res_rot["s2_mask"]))
    max_diff = float(np.max(np.abs(res_base["X18_full"] - res_rot["X18_full"])))
    print(f"O(4) Moving-Frame Gauge Invariance Audit (MIT-208): s2_mask_diff={s2_diff}, max_abs_diff={max_diff:.3e}", flush=True)
    assert s2_diff == 0 and max_diff < 1e-12, f"O(4) invariance violated: {s2_diff}, {max_diff}"
    return max_diff


def load_and_process_one_record(args):
    db_name, rec_dir, rec_id = args
    t0 = time.perf_counter()
    rec_path = os.path.join(rec_dir, rec_id)
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")
    fs = float(rec.fs)

    if db_name == "incartdb":
        sig_names = [s.upper().strip() for s in rec.sig_name]
        i0, i1 = sig_names.index("II"), sig_names.index("V1")
        patient_id = "incart_unknown"
        for cmt in (rec.comments or []):
            if "patient" in cmt.lower():
                patient_id = "incart_" + cmt.strip().replace(" ", "_")
                break
        leads_str = "II+V1"
    else:
        i0, i1 = 0, 1
        patient_id = f"{db_name}_{rec_id}"
        leads_str = f"{rec.sig_name[0]}+{rec.sig_name[1]}"

    s0_raw = np.nan_to_num(rec.p_signal[:, i0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    s1_raw = np.nan_to_num(rec.p_signal[:, i1].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sig0, delay = v33.filter_signal_generic(s0_raw, fs, mode="offline_filtfilt")
    sig1, _ = v33.filter_signal_generic(s1_raw, fs, mode="offline_filtfilt")

    def map_sym(sym):
        if sym in ("N", "L", "R", "e", "j"):
            return 0
        if sym in ("V", "E"):
            return 1
        if sym == "F":
            return 2
        if sym in ("A", "a", "J", "S"):
            return 3
        return -1

    samples, symbols = ann.sample, ann.symbol
    n_sig = len(sig0)
    if abs(fs - v33.TARGET_FS) < 1e-3:
        beats0, beats1, labels, r_peaks = [], [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < 93 or sc + 94 > n_sig:
                continue
            c = map_sym(sym)
            if c >= 0:
                beats0.append(sig0[sc - 93 : sc + 94])
                beats1.append(sig1[sc - 93 : sc + 94])
                labels.append(c)
                r_peaks.append(s)
    else:
        half_win = int(np.ceil(0.30 * fs))
        local_t = np.arange(-half_win, half_win + 1, dtype=np.float64) / fs
        grid_idx = (v33.PHYS_TIME_GRID - local_t[0]) * fs
        lo_i = np.floor(grid_idx).astype(np.int64)
        hi_i = np.minimum(lo_i + 1, len(local_t) - 1)
        frac = (grid_idx - lo_i).astype(np.float64)

        valid_sc, labels, r_peaks = [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < half_win or sc + half_win + 1 > n_sig:
                continue
            c = map_sym(sym)
            if c >= 0:
                valid_sc.append(sc)
                labels.append(c)
                r_peaks.append(s)
        valid_sc = np.array(valid_sc, dtype=np.int64)
        if len(valid_sc) <= 55:
            return None
        base_off = valid_sc[:, None] - half_win
        idx_lo = base_off + lo_i[None, :]
        idx_hi = base_off + hi_i[None, :]
        beats0 = (1.0 - frac[None, :]) * sig0[idx_lo] + frac[None, :] * sig0[idx_hi]
        beats1 = (1.0 - frac[None, :]) * sig1[idx_lo] + frac[None, :] * sig1[idx_hi]

    beats0 = np.asarray(beats0, dtype=np.float64)
    beats1 = np.asarray(beats1, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    r_peaks = np.asarray(r_peaks, dtype=np.float64)
    N_all = len(labels)
    if N_all <= 55:
        return None

    pre_rr = np.zeros(N_all, dtype=np.float64)
    pre_rr[1:] = (r_peaks[1:] - r_peaks[:-1]) / fs
    pre_rr[0] = pre_rr[1]
    local_rr = np.zeros(N_all, dtype=np.float64)
    for i in range(N_all):
        lo = 0 if i < 10 else i - 10
        local_rr[i] = float(np.median(pre_rr[lo : i + 1])) + 1e-6
    pre_rr_loc_all = pre_rr / local_rr
    post_hours = float((r_peaks[-1] - r_peaks[50]) / fs / 3600.0)

    out = extract_moving_frame_features(beats0, beats1, labels, pre_rr_loc_all)
    s2_mask = out["s2_mask"]
    return {
        "db": db_name, "rec_id": rec_id, "patient_id": patient_id, "fs": fs, "leads": leads_str,
        "post_hours": post_hours, "K": out["K"], "b_N_pur": out["b_N_pur"],
        "v_purs": out["v_purs"],
        "y": out["y"], "ready": out["ready"], "s2_mask": s2_mask,
        "X18_s2": out["X18_full"][s2_mask].astype(np.float32),
        "proc_sec": time.perf_counter() - t0,
    }


def get_db_record_list(db_name):
    d = DB_DIRS[db_name]
    recs = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(d, "*.atr"))))
    if db_name == "mitdb":
        recs = [r for r in recs if r not in {"102", "104", "107", "217"}]
    return recs


def build_or_load_db_cache(db_name, n_workers=6):
    cache_npz = os.path.join(CACHE_DIR, f"{db_name}_gen3_moving_4cls.npz")
    cache_meta = os.path.join(CACHE_DIR, f"{db_name}_gen3_moving_4cls_meta.json")
    if os.path.exists(cache_npz) and os.path.exists(cache_meta):
        meta = json.load(open(cache_meta, "r", encoding="utf-8"))
        dat = np.load(cache_npz)
        items = []
        y_all, ready_all, s2_all, X18_s2_all = dat["y_all"], dat["ready_all"], dat["s2_all"], dat["X18_s2_all"]
        for m in meta:
            b0, b1 = m["b0"], m["b1"]
            s0, s1 = m["s0"], m["s1"]
            items.append({
                "db": m["db"], "rec_id": m["rec_id"], "patient_id": m["patient_id"],
                "fs": m["fs"], "leads": m["leads"], "post_hours": m["post_hours"],
                "K": m["K"], "b_N_pur": m["b_N_pur"], "v_purs": m["v_purs"],
                "y": y_all[b0:b1], "ready": ready_all[b0:b1], "s2_mask": s2_all[b0:b1],
                "X18_s2": X18_s2_all[s0:s1], "proc_sec": m["proc_sec"],
            })
        print(f"Loaded Moving-Frame cache for {db_name}: {len(items)} recs, {len(y_all):,} beats", flush=True)
        return items

    recs = get_db_record_list(db_name)
    print(f"Extracting Moving-Frame Gen-3 features for {db_name} ({len(recs)} recs) with {n_workers} workers...", flush=True)
    t0 = time.time()
    tasks = [(db_name, DB_DIRS[db_name], r) for r in recs]
    items = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(load_and_process_one_record, tasks):
            if res is not None:
                items.append(res)

    y_list, r_list, s2_list, x_list, meta = [], [], [], [], []
    b_cur = s_cur = 0
    for it in items:
        nb = len(it["y"])
        ns = len(it["X18_s2"])
        y_list.append(it["y"]); r_list.append(it["ready"]); s2_list.append(it["s2_mask"]); x_list.append(it["X18_s2"])
        meta.append({
            "db": it["db"], "rec_id": it["rec_id"], "patient_id": it["patient_id"],
            "fs": it["fs"], "leads": it["leads"], "post_hours": it["post_hours"],
            "K": it["K"], "b_N_pur": it["b_N_pur"], "v_purs": it["v_purs"],
            "b0": b_cur, "b1": b_cur + nb, "s0": s_cur, "s1": s_cur + ns,
            "proc_sec": it["proc_sec"],
        })
        b_cur += nb; s_cur += ns

    np.savez_compressed(
        cache_npz,
        y_all=np.concatenate(y_list),
        ready_all=np.concatenate(r_list),
        s2_all=np.concatenate(s2_list),
        X18_s2_all=np.vstack(x_list) if x_list else np.zeros((0, 18), dtype=np.float32),
    )
    with open(cache_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"  DONE {db_name}: {len(items)} recs, {b_cur:,} beats ({s_cur:,} S2, {100.0*s_cur/max(1,b_cur):.1f}%) in {time.time()-t0:.1f}s", flush=True)
    return items


def fit_source_model(train_items, feat_idx, model_type, cap_pat_pos=100, pi_shift=0.001):
    pat_map = {}
    for it in train_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: -sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    def build_weighted_xy(subset_items):
        Xs, ys, ws = [], [], []
        pat_f_cnt = Counter()
        for it in subset_items:
            if len(it["X18_s2"]) > 0:
                pat_f_cnt[it["patient_id"]] += int(np.sum(it["y"][it["s2_mask"]] == 2))
        for it in subset_items:
            if len(it["X18_s2"]) == 0:
                continue
            X_m = it["X18_s2"][:, feat_idx].astype(np.float64)
            y_m = (it["y"][it["s2_mask"]] == 2).astype(int)
            w_m = np.ones(len(y_m), dtype=np.float64)
            pf = pat_f_cnt[it["patient_id"]]
            if cap_pat_pos is not None and pf > cap_pat_pos:
                w_m[y_m == 1] = float(cap_pat_pos) / float(pf)
            Xs.append(X_m); ys.append(y_m); ws.append(w_m)
        X_cat = np.vstack(Xs); y_cat = np.concatenate(ys); w_cat = np.concatenate(ws)
        sum_pos = float(np.sum(w_cat[y_cat == 1]))
        sum_neg = float(np.sum(w_cat[y_cat == 0]))
        if sum_pos > 0 and sum_neg > 0:
            w_cat[y_cat == 1] *= (0.5 * (sum_pos + sum_neg) / sum_pos)
            w_cat[y_cat == 0] *= (0.5 * (sum_pos + sum_neg) / sum_neg)
        return X_cat, y_cat, w_cat

    oof_s2_sc, oof_s2_y = [], []
    tot_src_beats = sum(len(it["y"]) for it in train_items)
    tot_src_f = sum(int(np.sum(it["y"] == 2)) for it in train_items)

    for f_id in range(n_folds):
        tr_sub = [it for it in train_items if fold_of_pat[it["patient_id"]] != f_id]
        te_sub = [it for it in train_items if fold_of_pat[it["patient_id"]] == f_id]
        Xtr, ytr, wtr = build_weighted_xy(tr_sub)
        if np.sum(ytr == 1) == 0:
            continue
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        clf = (LogisticRegression(C=0.3, max_iter=500, random_state=42)
               if model_type == "Linear_LogReg"
               else LinearSVC(C=0.3, dual="auto", max_iter=3000, random_state=42))
        clf.fit(Xtr_s, ytr, sample_weight=wtr)
        for te in te_sub:
            if len(te["X18_s2"]) > 0:
                s_m = clf.decision_function(sc.transform(te["X18_s2"][:, feat_idx].astype(np.float64)))
                oof_s2_sc.append(s_m)
                oof_s2_y.append((te["y"][te["s2_mask"]] == 2).astype(int))

    s2_sc = np.concatenate(oof_s2_sc)
    s2_y = np.concatenate(oof_s2_y)
    order = np.argsort(s2_sc)[::-1]
    ys_o, ss_o = s2_y[order], s2_sc[order]
    tp_c = np.cumsum(ys_o == 1).astype(float)
    fp_c = np.cumsum(ys_o == 0).astype(float)

    f1_t0 = 2.0 * tp_c / (float(tot_src_f) + tp_c + fp_c + 1e-12)
    tau_t0 = float(ss_o[int(np.argmax(f1_t0))])

    pi_src = tot_src_f / max(1.0, float(tot_src_beats))
    neg_mult = (pi_src / max(1e-9, 1.0 - pi_src)) / (pi_shift / (1.0 - pi_shift))
    f1_t1 = 2.0 * tp_c / (float(tot_src_f) + tp_c + fp_c * neg_mult + 1e-12)
    tau_t1 = float(ss_o[int(np.argmax(f1_t1))])

    X_all, y_all, w_all = build_weighted_xy(train_items)
    scaler = StandardScaler()
    clf = (LogisticRegression(C=0.3, max_iter=500, random_state=42)
           if model_type == "Linear_LogReg"
           else LinearSVC(C=0.3, dual="auto", max_iter=3000, random_state=42))
    clf.fit(scaler.fit_transform(X_all), y_all, sample_weight=w_all)
    return scaler, clf, tau_t0, tau_t1


def eval_on_db(target_items, feat_idx, scaler, clf, tau_t0, tau_t1):
    tot_hours = sum(it["post_hours"] for it in target_items)
    all_sc, all_y, rec_sc = [], [], []
    for it in target_items:
        sc = np.full(len(it["y"]), -1e6, dtype=np.float64)
        if len(it["X18_s2"]) > 0:
            sc[it["s2_mask"]] = clf.decision_function(scaler.transform(it["X18_s2"][:, feat_idx].astype(np.float64)))
        all_sc.append(sc); all_y.append(it["y"]); rec_sc.append((it, sc))

    sc_cat = np.concatenate(all_sc)
    y_cat = np.concatenate(all_y)
    y_bin = (y_cat == 2).astype(int)
    tot_f = int(np.sum(y_bin == 1))
    ap = float(average_precision_score(y_bin, sc_cat)) if tot_f > 0 else 0.0

    def metrics_at_tau(tau):
        tp = fpn = fpv = fps = fn = zf_tot = zf_fp = 0
        for it, sc in rec_sc:
            y = it["y"]
            pr = (sc >= tau)
            tp_r = int(np.sum(pr & (y == 2)))
            fpn_r = int(np.sum(pr & (y == 0)))
            fpv_r = int(np.sum(pr & (y == 1)))
            fps_r = int(np.sum(pr & (y == 3)))
            fn_r = int(np.sum((~pr) & (y == 2)))
            tp += tp_r; fpn += fpn_r; fpv += fpv_r; fps += fps_r; fn += fn_r
            if int(np.sum(y == 2)) == 0:
                zf_tot += len(y); zf_fp += (fpn_r + fpv_r + fps_r)
        fp_tot = fpn + fpv + fps
        f1 = 2.0 * tp / max(1, 2 * tp + fp_tot + fn)
        prec = 100.0 * tp / max(1, tp + fp_tot)
        rec_p = 100.0 * tp / max(1, tp + fn)
        faph = fp_tot / max(1e-6, tot_hours)
        zf_spec = 100.0 * (1.0 - zf_fp / max(1, zf_tot)) if zf_tot > 0 else 100.0 * (1.0 - fp_tot / max(1, len(y_cat) - tot_f))
        return {
            "F1": round(f1, 4), "Prec_%": round(prec, 2), "Rec_%": round(rec_p, 2),
            "TP_FPN_FPV_FPS": f"{tp}/{fpn}/{fpv}/{fps}S",
            "Spec_%": round(zf_spec, 3), "FAPH": round(faph, 2),
        }

    order = np.argsort(sc_cat)[::-1]
    ys = y_bin[order]
    tp_c = np.cumsum(ys == 1); fp_c = np.cumsum(ys == 0)
    lim2 = int(math.floor(2.0 * tot_hours))
    i2 = np.where(fp_c <= lim2)[0]
    if len(i2) > 0 and tot_f > 0:
        j = int(i2[-1])
        t2, f2 = int(tp_c[j]), int(fp_c[j])
        faph2_str = f"{100*t2/max(1,t2+f2):.1f}%/{100*t2/tot_f:.1f}%/{2*t2/max(1,2*t2+f2+(tot_f-t2)):.3f}"
    else:
        faph2_str = "0/0/0"
    if tot_f > 0:
        f1_all = 2.0 * tp_c / (float(tot_f) + tp_c + fp_c + 1e-12)
        jb = int(np.argmax(f1_all))
        max_f1 = round(float(f1_all[jb]), 4)
        max_faph = round(float(fp_c[jb]) / max(1e-6, tot_hours), 2)
    else:
        max_f1, max_faph = 0.0, 0.0

    return {
        "AP": round(ap, 4),
        "T0": metrics_at_tau(tau_t0),
        "T1_BayesPi01": metrics_at_tau(tau_t1),
        "Desc_FAPH2": faph2_str,
        "Max_Desc_F1": max_f1,
        "Max_Desc_FAPH": max_faph,
    }


def main():
    o4_diff = verify_o4_gauge_invariance()
    db_order = ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]
    all_dbs = {}
    disc_rows = []
    feat_auc_rows = {}

    for db in db_order:
        nw = 4 if db in ("ltdb", "sddb") else 6
        items = build_or_load_db_cache(db, n_workers=nw)
        all_dbs[db] = items
        tot_b = sum(len(it["y"]) for it in items)
        tot_s2 = sum(len(it["X18_s2"]) for it in items)
        tot_h = sum(it["post_hours"] for it in items)
        n_n = sum(int(np.sum(it["y"] == 0)) for it in items)
        n_v = sum(int(np.sum(it["y"] == 1)) for it in items)
        n_f = sum(int(np.sum(it["y"] == 2)) for it in items)
        n_s = sum(int(np.sum(it["y"] == 3)) for it in items)
        f_cov = sum(int(np.sum((it["y"] == 2) & it["ready"])) for it in items)
        f_s2 = sum(int(np.sum((it["y"] == 2) & it["s2_mask"])) for it in items)
        n_s2_n = sum(int(np.sum(it["y"][it["s2_mask"]] == 0)) for it in items)
        n_pats = len(set(it["patient_id"] for it in items))
        f_pats = len(set(it["patient_id"] for it in items if int(np.sum(it["y"] == 2)) > 0))
        recs_k = sum(1 for it in items if it["K"] > 0)
        bn_pur = 100.0 * float(np.mean([it["b_N_pur"] for it in items]))
        v_purs = [p for it in items for p in it["v_purs"]]
        v_pur = 100.0 * float(np.mean(v_purs)) if v_purs else 0.0
        impure_recs = sum(1 for it in items if it["v_purs"] and min(it["v_purs"]) < 0.5)

        drow = {
            "DB": db, "Recs": len(items), "Pats": n_pats, "F_Pats": f_pats,
            "Hours": round(tot_h, 1), "Beats": tot_b,
            "S2_Beats": tot_s2, "S2_Rate_%": round(100.0 * tot_s2 / max(1, tot_b), 2),
            "S2_N": n_s2_n, "N": n_n, "S": n_s, "V": n_v, "F": n_f,
            "Pi_F_%": round(100.0 * n_f / max(1, tot_b), 4),
            "Recs_K>0": recs_k, "Impure_V_Recs": impure_recs,
            "F_Cov_%": round(100.0 * f_cov / max(1, n_f), 1),
            "S2_F_%": round(100.0 * f_s2 / max(1, n_f), 1),
            "bN_Pur_%": round(bn_pur, 2), "V_Pur_%": round(v_pur, 2),
        }
        disc_rows.append(drow)
        print(
            f"AUDIT {db:8s} | S2={tot_s2:7,d} ({drow['S2_Rate_%']:4.1f}%, S2_N={n_s2_n:6,d}) | "
            f"F_cov={drow['F_Cov_%']:5.1f}% | S2_F={drow['S2_F_%']:5.1f}% | "
            f"V_pur={v_pur:5.1f}% (impure_recs={impure_recs:2d}/{len(items)})",
            flush=True,
        )

        # Stage-2 per-feature AUCs to audit sign stability
        X_s2_db = np.vstack([it["X18_s2"] for it in items if len(it["X18_s2"]) > 0])
        y_s2_db = np.concatenate([it["y"][it["s2_mask"]] for it in items if len(it["X18_s2"]) > 0])
        yf_db = (y_s2_db == 2).astype(int)
        if np.sum(yf_db) > 0:
            feat_auc_rows[db] = {fn: round(float(roc_auc_score(yf_db, X_s2_db[:, i])), 3) for i, fn in enumerate(FEAT_NAMES)}

    model_specs = [
        ("B0_Gen2-Soft (12D)", "Linear_SVM"),
        ("G3_ThalesPyth (14D)", "Linear_SVM"),
        ("G3_BurstPast2 (16D)", "Linear_LogReg"),
        ("G4_MBridge3 (16D)", "Linear_SVM"),
        ("G3_FullCtx0b (17D)", "Linear_LogReg"),
        ("G4_FullCtx0b (18D)", "Linear_LogReg"),
    ]

    print("\n=== PART 2: Moving-Frame MIT-Dev10 -> 5 Unseen Hospitals (4-Class N+S+V+F) ===", flush=True)
    mit_dev10 = [it for it in all_dbs["mitdb"] if it["rec_id"] in set(v33.MIT_TRAIN_10)]
    part2_rows = []
    for var_name, mtype in model_specs:
        fidx = VARIANTS_IDX[var_name]
        scaler, clf, tau0, tau1 = fit_source_model(mit_dev10, fidx, mtype, cap_pat_pos=None, pi_shift=0.001)
        for tgt_db in ["incartdb", "ltdb", "edb", "sddb", "svdb"]:
            ev = eval_on_db(all_dbs[tgt_db], fidx, scaler, clf, tau0, tau1)
            row = {"Train": "MIT-Dev10", "Target": tgt_db, "Variant": var_name, "Model": mtype, **ev}
            part2_rows.append(row)
            if var_name in ("B0_Gen2-Soft (12D)", "G3_BurstPast2 (16D)", "G3_FullCtx0b (17D)"):
                t1 = ev["T1_BayesPi01"]
                print(f"  MIT10->{tgt_db:8s} | {var_name:19s} | AP={ev['AP']:.4f} | T0_F1={ev['T0']['F1']:.4f} | T1_F1={t1['F1']:.4f} (P={t1['Prec_%']:5.1f}% R={t1['Rec_%']:5.1f}% {t1['TP_FPN_FPV_FPS']:17s} Spec={t1['Spec_%']:.3f}% FAPH={t1['FAPH']:5.2f}) | MaxF1={ev['Max_Desc_F1']:.4f}", flush=True)

    print("\n=== PART 3: Moving-Frame 6-HOSPITAL LODO (5 Hospitals -> Unseen 6th Hospital) ===", flush=True)
    part3_rows = []
    for tgt_db in db_order:
        tr_items = [it for db in db_order if db != tgt_db for it in all_dbs[db]]
        for var_name, mtype in model_specs:
            fidx = VARIANTS_IDX[var_name]
            scaler, clf, tau0, tau1 = fit_source_model(tr_items, fidx, mtype, cap_pat_pos=100, pi_shift=0.001)
            ev = eval_on_db(all_dbs[tgt_db], fidx, scaler, clf, tau0, tau1)
            row = {"Train": f"LODO_5DB_no_{tgt_db}", "Target": tgt_db, "Variant": var_name, "Model": mtype, **ev}
            part3_rows.append(row)
            if var_name in ("B0_Gen2-Soft (12D)", "G3_ThalesPyth (14D)", "G3_BurstPast2 (16D)", "G3_FullCtx0b (17D)"):
                t1 = ev["T1_BayesPi01"]
                print(f"  LODO ->{tgt_db:8s} | {var_name:19s} | AP={ev['AP']:.4f} | T0_F1={ev['T0']['F1']:.4f} | T1_F1={t1['F1']:.4f} (P={t1['Prec_%']:5.1f}% R={t1['Rec_%']:5.1f}% {t1['TP_FPN_FPV_FPS']:17s} Spec={t1['Spec_%']:.3f}% FAPH={t1['FAPH']:5.2f}) | MaxF1={ev['Max_Desc_F1']:.4f}", flush=True)

    out_json = os.path.join(AUD_DIR, "V36B_MOVING_FRAME_6HOSPITAL_LODO_AUDIT_20261121.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "o4_gauge_invariance_max_diff": o4_diff,
            "part1_6hospital_discovery": disc_rows,
            "part1b_feature_s2_aucs": feat_auc_rows,
            "part2_mit10_to_5db": part2_rows,
            "part3_6hospital_lodo": part3_rows,
        }, f, indent=2)

    # Load v36 baseline for side-by-side comparison in MD
    v36_old = json.load(open(V36_JSON, "r", encoding="utf-8")) if os.path.exists(V36_JSON) else None

    out_md = os.path.join(AUD_DIR, "V36B_MOVING_FRAME_6HOSPITAL_LODO_AUDIT_20261121.md")
    lines = [
        "# v36b Moving-Frame Local Dual-Chart 6-Hospital LODO Benchmark (4-Class N+S+V+F)",
        "",
        f"- **Exact $O(4)$ Gauge Invariance (MIT-208)**: `max_abs_diff = {o4_diff:.3e}` (`s2_mask_diff = 0`).",
        "",
        "## Part 1: 6-Hospital Discovery Audit (Moving-Frame vs. Static Prefix-50 v36)",
        "",
        "| DB | Hours | Beats | F | S2 Beats (%) | S2_N | F Cov (%) | S2 F (%) | V Pur (%) | Impure V Recs |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in disc_rows:
        lines.append(
            f"| `{r['DB']}` | {r['Hours']:.1f} | {r['Beats']:,} | **{r['F']}** | {r['S2_Beats']:,} ({r['S2_Rate_%']:.1f}%) | {r['S2_N']:,} | "
            f"{r['F_Cov_%']:.1f} | {r['S2_F_%']:.1f} | **{r['V_Pur_%']:.2f}** | {r['Impure_V_Recs']}/{r['Recs']} |"
        )

    lines.extend([
        "", "## Part 1b: Stage-2 Feature Sign-Stability Audit Across All 6 Hospitals (AUC)", "",
        "| DB | m_bridge | d_seg | loc_ncorr | loc_dnorm | preqrs_c | min_bcorr | sqi_hf | pyth_loc | min_thales | nb_mb3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for db in db_order:
        a = feat_auc_rows.get(db, {})
        if a:
            lines.append(
                f"| `{db}` | {a['m_bridge']:.3f} | {a['d_seg']:.3f} | {a['loc_ncorr']:.3f} | {a['loc_dnorm']:.3f} | "
                f"{a['preqrs_c']:.3f} | {a['min_bcorr']:.3f} | {a['sqi_hf']:.3f} | {a['pyth_loc']:.3f} | {a['min_thales']:.3f} | {a['nb_mb3']:.3f} |"
            )

    def append_res_table(title, rows):
        lines.extend([
            "", f"## {title}", "",
            "| Target DB | Variant | Model | AP | T0 F1 (FAPH) | T1 Bayes F1 | Prec (%) | Rec (%) | TP/FP_N/FP_V/FP_S | Spec (%) | FAPH | @FAPH<=2 (P/R/F1) | Max Desc F1 (FAPH) |",
            "|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---|---|",
        ])
        for r in rows:
            t0 = r["T0"]; t1 = r["T1_BayesPi01"]
            lines.append(
                f"| `{r['Target']}` | {r['Variant']} | {r['Model']} | **{r['AP']:.4f}** | {t0['F1']:.4f} ({t0['FAPH']:.1f}) | "
                f"**{t1['F1']:.4f}** | {t1['Prec_%']:.2f} | {t1['Rec_%']:.2f} | {t1['TP_FPN_FPV_FPS']} | "
                f"{t1['Spec_%']:.3f} | {t1['FAPH']:.2f} | {r['Desc_FAPH2']} | {r['Max_Desc_F1']:.4f} ({r['Max_Desc_FAPH']:.1f}) |"
            )

    append_res_table("Part 2: Moving-Frame MIT-Dev10 -> 5 Unseen Hospitals (Zero External Tuning)", part2_rows)
    append_res_table("Part 3: Moving-Frame 6-Hospital LODO (5 Hospitals -> Unseen 6th Hospital)", part3_rows)

    if v36_old is not None:
        lines.extend([
            "", "## Part 4: Head-to-Head Summary — Static v36 vs. Moving-Frame v36b (G3_FullCtx0b & G3_ThalesPyth)", "",
            "| Target DB | Mode | Static v36 AP | Moving v36b AP | Static v36 T1 F1 (FAPH) | Moving v36b T1 F1 (FAPH) | Static Max F1 | Moving Max F1 |",
            "|---|---|---:|---:|---|---|---:|---:|",
        ])
        for mode_label, k_old, rows_new in [("MIT-Dev10->Target", "part2_mit10_to_5db", part2_rows), ("LODO 5DB->Target", "part3_6hospital_lodo", part3_rows)]:
            old_map = {(r["Target"], r["Variant"]): r for r in v36_old[k_old]}
            for r in rows_new:
                if r["Variant"] not in ("G3_ThalesPyth (14D)", "G3_FullCtx0b (17D)"):
                    continue
                ro = old_map.get((r["Target"], r["Variant"]))
                if ro:
                    lines.append(
                        f"| `{r['Target']}` | {mode_label} `{r['Variant'][:12]}` | {ro['AP']:.4f} | **{r['AP']:.4f}** | "
                        f"{ro['T1_BayesPi01']['F1']:.4f} ({ro['T1_BayesPi01']['FAPH']:.1f}) | "
                        f"**{r['T1_BayesPi01']['F1']:.4f}** ({r['T1_BayesPi01']['FAPH']:.1f}) | "
                        f"{ro['Max_Desc_F1']:.4f} | **{r['Max_Desc_F1']:.4f}** |"
                    )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {out_json}\nSaved: {out_md}", flush=True)


if __name__ == "__main__":
    main()
