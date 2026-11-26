"""
lab_diagnostic_v42b_hypB1_anchor_guards_2x2.py
==============================================
HYPOTHESIS B1 — 2x2 FACTORIAL ON ONLINE V-ANCHOR ADMISSION/CONFIRMATION GUARDS AT K_max = 12
(Separating Anti-N candidate blocking from Anti-F candidate protection, keeping EVERYTHING else identical!)

Exact 2x2 Factorial Matrix (at K_max = 12, Static v36 N-chart, MAX_CAND_CLUSTERS = 8, ZERO labels in anchor creation,
identical 18D features, AND strict FROZEN K=4 classifier weights + FROZEN K=4 source-locked T1 threshold):
  - K4_Base      : Static v36 baseline (K_max = 4, No Guards)
  - C00_K12_Base : K_max = 12, Anti-N = OFF, Anti-F = OFF  (exact v42 K=12 baseline)
  - C10_AntiN    : K_max = 12, Anti-N = ON,  Anti-F = OFF
  - C01_AntiF    : K_max = 12, Anti-N = OFF, Anti-F = ON
  - C11_Both     : K_max = 12, Anti-N = ON,  Anti-F = ON

Locked Guards (Thresholds selected STRICTLY on DEV = mitdb + edb; ZERO tuning on ltdb/incartdb/sddb/svdb;
100% unlabeled; exact O(4) gauge invariant verified < 1e-12):
  1. Anti-N Guard (Pre-existing v36b His-Purkinje Narrow-QRS Gate, ZERO new hyperparameters):
       Reject beat from candidate-cluster entry if min_qrs_corr_2lead(x_t, b_N_loc) > 0.85,
       and reject cluster promotion if min_qrs_corr_2lead(med_b, b_N_loc) > 0.85.
  2. Anti-F Guard (Dual-Chart Fusion-Evidence Protection, locked strictly on DEV mitdb+edb):
       At cluster confirmation (n >= 6, cohesion <= 8.0), reject promotion to V-anchor if:
         (med_rr > 0.80 and (mb_med > 0.0 or med_pq_loc >= 0.40))
         OR (mb_med > 0.0 and pyth_med > 0.0 and med_pq_loc >= 0.40)
       where mb_med and pyth_med are computed against existing anchors (Proposition 3 bridge interiority)
       and med_pq_loc is median local pre-QRS sinus P-wave correlation against causal loc_ring template pN_loc.

Key Telemetry & Methodological Discipline:
  - Per-Anchor Scope-of-Impact (Leave-One-Anchor-Out counterfactual): exact (Delta_V_S1, Delta_F_S1, Delta_FPV_T1,
    Delta_TPF_T1) for EVERY anchor (and specifically every k >= 5 anchor) — directly answering:
    "Each anchor rejects how many additional V beats and how many additional F beats?"
  - Apples-to-apples comparisons in BOTH:
      (a) FullTape (K4_Full vs C00..C11_Full, exact v36 Holter parity)
      (b) PostBirth (K4_PostBirth vs C00..C11_PostBirth: each anchor active only for t >= disc_idx,
          explicitly labeled "post-birth anchor availability with end-of-tape V-radii", NOT full causal).
  - Explicit spotlights on:
      * LTDB-14134 (explicitly framed as known-error fix check / bang chung sua loi da biet)
      * LTDB-14046 & LTDB-15814 (and LTDB-14149)
      * MIT-233 (testing whether clean pure-V anchors STILL reject F due to exclusion-zone overlap!)
      * Plus MIT-213, MIT-223, EDB-e0605, INCART-I75.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import wfdb

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import lab_diagnostic_v36_6hospital_lodo_benchmark as v36
import lab_diagnostic_v36b_moving_frame_6hospital_lodo as v36b

AUD_DIR = str(_REPO_ROOT / "audits")
CACHE_K4 = str(_DATA_ROOT / "cache_gen3_6db")
CACHE_V42B = str(_DATA_ROOT / "cache_v42b_guards_2x2_6db")
os.makedirs(AUD_DIR, exist_ok=True)
os.makedirs(CACHE_V42B, exist_ok=True)

DB_ORDER = ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]

# Locked DEV-only guard constants (ZERO tuning on ltdb, incartdb, sddb, svdb!)
ANTI_N_QRS_CORR_MAX = 0.85   # Exact v36b His-Purkinje gate (MIT verified)
ANTI_F_LATE_RR_MIN = 0.80    # Late-diastolic coupling threshold (DEV mitdb+edb)
ANTI_F_PREQRS_LOC_MIN = 0.40 # Local sinus P-wave correlation threshold (DEV mitdb+edb)

CONFIGS = [
    ("K4_Base",      4,  False, False),
    ("C00_K12_Base", 12, False, False),
    ("C10_AntiN",    12, True,  False),
    ("C01_AntiF",    12, False, True),
    ("C11_Both",     12, True,  True),
]
CFG_NAMES = [c[0] for c in CONFIGS]

MODELS = {
    "F2_RefFrame10D_SVM":   ([0, 1, 2, 3, 4, 5, 8, 10, 13, 14], "SVM"),
    "G3_FullCtx0b17D_LogR": (list(range(12)) + [13, 14, 15, 16, 17], "LogR"),
}

SPOTLIGHT_RECS = {
    "mitdb": ["233", "223", "213", "208"],
    "ltdb": ["14134", "14046", "15814", "14149"],
    "edb": ["e0605", "e0602", "e0614"],
    "incartdb": ["I75", "I62"],
}


def compute_s2_and_loao_s1(y_mat, post_y, anchors, anchor_meta, post_birth=False):
    """
    Vectorized Stage-1 projection + EXACT Leave-One-Anchor-Out (LOAO) counterfactual
    Stage-1 scope-of-impact for every anchor k in {0..K-1}:
      - s1_v_excl_net: net true-V beats removed from Stage-2 by adding anchor k
      - s1_f_excl_net: net true-F beats removed from Stage-2 by adding anchor k
      - s1_v_excl_raw: true-V beats in S2 without k, rejected when k is added
      - s1_f_excl_raw: true-F beats in S2 without k, rejected when k is added
      - s1_f_gain_raw: true-F beats NOT in S2 without k, admitted when k is added
    """
    N = len(post_y)
    K = len(anchors)
    if K == 0:
        return (
            np.zeros(N, dtype=bool), np.zeros(N, dtype=bool),
            np.zeros(N, dtype=np.int32), np.zeros(N), np.zeros(N),
            np.full(N, 9.99), np.zeros(N), np.zeros(N), np.zeros(N),
            [],
        )

    Q_mat = np.vstack([a["q"] for a in anchors])
    nq_vec = np.array([a["norm_q"] for a in anchors], dtype=np.float64)
    muN_vec = np.array([a["mu_N"] for a in anchors], dtype=np.float64)
    radN_vec = np.array([a["rad_N"] for a in anchors], dtype=np.float64)
    radV_vec = np.array([a["rad_V"] for a in anchors], dtype=np.float64)
    radyV_vec = np.array([a["rad_y_V"] for a in anchors], dtype=np.float64)
    disc_vec = np.array([am["disc_idx"] for am in anchor_meta], dtype=np.int64)

    lam_all = (y_mat @ Q_mat.T) / (nq_vec[None, :] ** 2 + 1e-12)
    lam_c_all = np.clip(lam_all, 0.0, 1.0)
    es_all = np.full((N, K), 1e9, dtype=np.float64)
    ej_all = np.full((N, K), 1e9, dtype=np.float64)
    t_vec = np.arange(N)

    for k in range(K):
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

    def _eval_subset(cols):
        if len(cols) == 0:
            z = np.zeros(N, dtype=bool)
            return z, z, np.zeros(N, dtype=np.int32), np.zeros(N), np.zeros(N), np.full(N, 9.99), np.zeros(N), np.zeros(N), np.zeros(N)
        es_s = es_all[:, cols]
        ej_s = ej_all[:, cols]
        rd = (np.min(es_s, axis=1) < 1e8)
        if not np.any(rd):
            z = np.zeros(N, dtype=bool)
            return rd, z, np.zeros(N, dtype=np.int32), np.zeros(N), np.zeros(N), np.full(N, 9.99), np.zeros(N), np.zeros(N), np.zeros(N)
        loc_k = np.argmin(es_s, axis=1)
        bk = cols[loc_k]
        idx_n = np.arange(N)
        bl = lam_all[idx_n, bk]
        blc = lam_c_all[idx_n, bk]
        d_seg = np.where(rd, es_all[idx_n, bk] / (nq_vec[bk] + 1e-12), 9.99)
        e_V_any = np.min(ej_s, axis=1)
        e_N = np.maximum(0.0, (bl - muN_vec[bk]) - radN_vec[bk])
        e_Vc = np.maximum(0.0, (1.0 - bl) - radV_vec[bk])
        zN = np.where(rd, np.log1p(e_N / radN_vec[bk]), 0.0)
        zV = np.where(rd, np.minimum(np.log1p(e_Vc / radV_vec[bk]), np.log1p(e_V_any / radV_vec[bk])), 0.0)
        mb = np.where(rd, np.minimum(zN, zV), 0.0)
        s2 = rd & (mb > 0.0)
        return rd, s2, bk, bl, blc, d_seg, zN, zV, mb

    all_cols = np.arange(K)
    rd_all, s2_all, bk_all, bl_all, blc_all, dseg_all, zN_all, zV_all, mb_all = _eval_subset(all_cols)

    # Exact Leave-One-Anchor-Out (LOAO) Stage-1 scope of impact for EVERY anchor k!
    is_v = (post_y == 1)
    is_f = (post_y == 2)
    is_n = (post_y == 0)
    loao_s1 = []
    for k in range(K):
        other = np.array([j for j in range(K) if j != k], dtype=np.int64)
        _, s2_wo, *_ = _eval_subset(other)
        v_excl = int(np.sum(is_v & s2_wo & (~s2_all)))
        v_gain = int(np.sum(is_v & (~s2_wo) & s2_all))
        f_excl = int(np.sum(is_f & s2_wo & (~s2_all)))
        f_gain = int(np.sum(is_f & (~s2_wo) & s2_all))
        n_excl = int(np.sum(is_n & s2_wo & (~s2_all)))
        n_gain = int(np.sum(is_n & (~s2_wo) & s2_all))
        loao_s1.append({
            "slot": int(anchor_meta[k]["slot"]),
            "v_excl_s1": v_excl, "v_gain_s1": v_gain, "v_net_rej_s1": v_excl - v_gain,
            "f_excl_s1": f_excl, "f_gain_s1": f_gain, "f_net_rej_s1": f_excl - f_gain,
            "n_net_rej_s1": n_excl - n_gain,
        })

    return rd_all, s2_all, bk_all, bl_all, blc_all, dseg_all, zN_all, zV_all, mb_all, loao_s1


def extract_one_config(
    post_joint, post_raw0, post_raw1, post_y, pre_rr_loc, post_r, fs, r0_time,
    joint_all, joint_50, norm_beats, final_norm_idx, ectopic_idx, labels_50,
    b_N, U, P, norm_y_prefix, eta_0,
    pre0_50, pre1_50, pN_0, pN_1, post_pre0, post_pre1,
    pref_hf_p95, pref_bcorr_p05,
    sqi_hf_ratio, zeno_qiso_drop_all, preqrs_corr_N, rr_dev_pre,
    y_mat, norm_y, norm_y_thresh,
    k_max=12, anti_n=False, anti_f=False,
):
    """
    Runs ONE exact Static-v36 discovery configuration (k_max, anti_n, anti_f)
    and returns FullTape + PostBirth features + per-anchor metadata + LOAO Stage-1 impact!
    ZERO labels are ever read by anchor discovery or feature projection!
    """
    N, D = post_joint.shape
    bN_0, bN_1 = b_N[:187], b_N[187:]

    anchors = []
    anchor_meta = []
    guard_stats = {"blocked_anti_n_beats": 0, "blocked_anti_n_clusters": 0, "blocked_anti_f_clusters": 0,
                   "blocked_anti_f_true_f": 0, "blocked_anti_f_true_v": 0,
                   "blocked_anti_n_true_n": 0, "blocked_anti_n_true_v": 0}

    if len(ectopic_idx) >= 4:
        anc = v33.make_anchor_obj(joint_50[ectopic_idx], norm_y_prefix, b_N, P, 0, "prefix")
        if anc is not None:
            ey = labels_50[ectopic_idx]
            anc["v_pur"] = float(np.mean(ey == 1))
            anchors.append(anc)
            anchor_meta.append({
                "slot": 1, "src": "prefix", "disc_idx": 0, "time_min": 0.0, "pct_tape": 0.0,
                "v_cnt": int(np.sum(ey == 1)), "f_cnt": int(np.sum(ey == 2)), "n_cnt": int(np.sum(ey == 0)), "s_cnt": int(np.sum(ey == 3)),
                "v_pur": round(float(np.mean(ey == 1)), 3),
                "f_frac": round(float(np.mean(ey == 2)), 3),
                "n_frac": round(float(np.mean(ey == 0)), 3),
                "norm_q": round(float(anc["norm_q"]), 2),
            })

    loc_ring = np.zeros((v33.LOC_BUF_LEN, D), dtype=np.float64)
    init_nb = norm_beats[-v33.LOC_BUF_LEN:]
    loc_cnt = len(init_nb)
    loc_ring[:loc_cnt] = init_nb
    loc_ptr = loc_cnt % v33.LOC_BUF_LEN
    b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)

    # Local pre-QRS sinus template ring (only used by anti_f; tracks exact same loc_ring admissions!)
    if anti_f:
        L_pre = post_pre0.shape[1]
        pre0_ring = np.zeros((v33.LOC_BUF_LEN, L_pre), dtype=np.float64)
        pre1_ring = np.zeros((v33.LOC_BUF_LEN, L_pre), dtype=np.float64)
        init_idx = final_norm_idx[-v33.LOC_BUF_LEN:]
        pre0_ring[:loc_cnt] = pre0_50[init_idx]
        pre1_ring[:loc_cnt] = pre1_50[init_idx]
        pN0_loc = np.median(pre0_ring[:loc_cnt], axis=0)
        pN1_loc = np.median(pre1_ring[:loc_cnt], axis=0)

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
            elif len(anchors) < k_max:
                # GUARD 1 (Anti-N, beat entry): His-Purkinje narrow-QRS exclusion (exact v36b threshold 0.85)
                if anti_n and (v36b.min_qrs_corr_2lead(x_t, b_N_loc) > ANTI_N_QRS_CORR_MAX):
                    guard_stats["blocked_anti_n_beats"] += 1
                else:
                    pq_loc_t = 0.5 * (v36b._corr1d(post_pre0[t], pN0_loc) + v36b._corr1d(post_pre1[t], pN1_loc)) if anti_f else 0.0
                    best_c_idx, best_dist = -1, v33.CLUSTER_RADIUS
                    for ci, cl in enumerate(cand_clusters):
                        dist = float(np.linalg.norm(x_t - cl["center"]))
                        if dist <= best_dist:
                            best_dist, best_c_idx = dist, ci
                    if best_c_idx >= 0:
                        best_c = cand_clusters[best_c_idx]
                        best_c["beats"].append(x_t)
                        best_c["ys"].append(int(post_y[t]))
                        if anti_f:
                            best_c["pq_loc"].append(pq_loc_t)
                            best_c["rrs"].append(float(pre_rr_loc[t]))
                        best_c["center"] = np.median(best_c["beats"], axis=0)
                        if len(best_c["beats"]) >= v33.CONFIRM_MIN_BEATS:
                            arr_b = np.array(best_c["beats"])
                            med_b = np.median(arr_b, axis=0)
                            cohesion = float(np.median(np.linalg.norm(arr_b - med_b, axis=1)))
                            if cohesion <= v33.CONFIRM_MAX_COHESION:
                                cy = np.array(best_c["ys"])
                                reject_n = False
                                reject_f = False

                                # GUARD 1 (Anti-N, cluster medoid check)
                                if anti_n and (v36b.min_qrs_corr_2lead(med_b, b_N_loc) > ANTI_N_QRS_CORR_MAX):
                                    reject_n = True
                                    guard_stats["blocked_anti_n_clusters"] += 1
                                    if np.mean(cy == 0) >= 0.5: guard_stats["blocked_anti_n_true_n"] += 1
                                    if np.mean(cy == 1) >= 0.5: guard_stats["blocked_anti_n_true_v"] += 1

                                # GUARD 2 (Anti-F, cluster F-evidence protection — locked strictly on DEV!)
                                if (not reject_n) and anti_f:
                                    med_rr = float(np.median(best_c["rrs"]))
                                    med_pq = float(np.median(best_c["pq_loc"]))
                                    mb_m = 0.0
                                    pyth_m = -1.0
                                    if len(anchors) > 0:
                                        y_med = P @ (med_b - b_N)
                                        best_es = 1e18
                                        best_a = None
                                        best_la = 0.0
                                        for a in anchors:
                                            la = float(np.dot(y_med, a["q"]) / (a["norm_q"] ** 2 + 1e-12))
                                            lc = float(np.clip(la, 0.0, 1.0))
                                            es = float(np.linalg.norm(y_med - lc * a["q"]))
                                            if es < best_es:
                                                best_es, best_a, best_la = es, a, la
                                        eN = max(0.0, (best_la - best_a["mu_N"]) - best_a["rad_N"])
                                        eVc = max(0.0, (1.0 - best_la) - best_a["rad_V"])
                                        eV_any = min(
                                            max(0.0, (float(np.linalg.norm(y_med - aj["q"])) - aj["rad_y_V"]) / (aj["norm_q"] + 1e-12))
                                            for aj in anchors
                                        )
                                        zN_m = float(np.log1p(eN / best_a["rad_N"]))
                                        zV_m = min(float(np.log1p(eVc / best_a["rad_V"])), float(np.log1p(eV_any / best_a["rad_V"])))
                                        mb_m = min(zN_m, zV_m)
                                        _, pyth_m = v33.thales_obt_and_pyth(
                                            float(np.linalg.norm(y_med)),
                                            float(np.linalg.norm(y_med - best_a["q"])),
                                            float(best_a["norm_q"]),
                                        )
                                    is_f_evid = (
                                        (med_rr > ANTI_F_LATE_RR_MIN and (mb_m > 0.0 or med_pq >= ANTI_F_PREQRS_LOC_MIN))
                                        or (mb_m > 0.0 and pyth_m > 0.0 and med_pq >= ANTI_F_PREQRS_LOC_MIN)
                                    )
                                    if is_f_evid:
                                        reject_f = True
                                        guard_stats["blocked_anti_f_clusters"] += 1
                                        if np.mean(cy == 2) > 0.0: guard_stats["blocked_anti_f_true_f"] += 1
                                        if np.mean(cy == 1) >= 0.5: guard_stats["blocked_anti_f_true_v"] += 1

                                if reject_n or reject_f:
                                    cand_clusters.pop(best_c_idx)
                                else:
                                    anc = v33.make_anchor_obj(arr_b, norm_y_prefix, b_N, P, int(t + 1), "online")
                                    if anc is not None:
                                        anc["v_pur"] = float(np.mean(cy == 1))
                                        anchors.append(anc)
                                        anchor_meta.append({
                                            "slot": len(anchors), "src": "online",
                                            "disc_idx": int(t + 1),
                                            "time_min": round(float((post_r[t] - r0_time) / fs / 60.0), 2),
                                            "pct_tape": round(100.0 * float(t + 1) / max(1, N), 1),
                                            "v_cnt": int(np.sum(cy == 1)), "f_cnt": int(np.sum(cy == 2)),
                                            "n_cnt": int(np.sum(cy == 0)), "s_cnt": int(np.sum(cy == 3)),
                                            "v_pur": round(float(np.mean(cy == 1)), 3),
                                            "f_frac": round(float(np.mean(cy == 2)), 3),
                                            "n_frac": round(float(np.mean(cy == 0)), 3),
                                            "norm_q": round(float(anc["norm_q"]), 2),
                                        })
                                        cand_clusters.pop(best_c_idx)
                    else:
                        if len(cand_clusters) >= v33.MAX_CAND_CLUSTERS:
                            cand_clusters.sort(key=lambda c: len(c["beats"]))
                            cand_clusters.pop(0)
                        nc = {"beats": [x_t], "ys": [int(post_y[t])], "center": x_t.copy()}
                        if anti_f:
                            nc["pq_loc"] = [pq_loc_t]
                            nc["rrs"] = [float(pre_rr_loc[t])]
                        cand_clusters.append(nc)

        reg_rr = (v33.LOC_RR_MIN <= pre_rr_loc[t] <= v33.LOC_RR_MAX)
        far_from_v = all(float(np.linalg.norm(x_t - anc["b_V"])) >= v33.LOC_V_EXCLUSION_DIST for anc in anchors)
        if reg_rr and far_from_v:
            c_prev1 = v33.mean_lead_corr(x_t, prev1)
            c_prev12 = v33.mean_lead_corr(prev1, prev2)
            if (c_loc >= v33.LOC_ADMIT_CORR_BN and c_prev1 >= 0.70) or (c_prev1 >= v33.LOC_ADMIT_RUN_CORR and c_prev12 >= v33.LOC_ADMIT_RUN_CORR):
                loc_ring[loc_ptr] = x_t
                if anti_f:
                    pre0_ring[loc_ptr] = post_pre0[t]
                    pre1_ring[loc_ptr] = post_pre1[t]
                loc_ptr = (loc_ptr + 1) % v33.LOC_BUF_LEN
                if loc_cnt < v33.LOC_BUF_LEN:
                    loc_cnt += 1
                b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
                if anti_f:
                    pN0_loc = np.median(pre0_ring[:loc_cnt], axis=0)
                    pN1_loc = np.median(pre1_ring[:loc_cnt], axis=0)
        prev2, prev1 = prev1, x_t

    def build_features(post_birth=False):
        rd, s2_mask, bk, bl, blc, d_seg_arr, z_N_arr, z_V_arr, m_bridge_arr, loao_s1 = compute_s2_and_loao_s1(
            y_mat, post_y, anchors, anchor_meta, post_birth=post_birth
        )
        min_lead_bcorr_arr = np.zeros(N)
        loc_dist_norm_arr = np.zeros(N)
        pyth_margin_loc = np.full(N, -1.0)
        min_lead_thales_loc = np.full(N, -1.0)

        if len(anchors) > 0 and np.any(rd):
            K = len(anchors)
            Q_mat = np.vstack([a["q"] for a in anchors])
            nq_vec = np.array([a["norm_q"] for a in anchors])
            bV_stack = np.vstack([a["b_V"] for a in anchors])
            loc_dist_norm_arr = np.where(rd, loc_norm_dist_proj_arr / (nq_vec[bk] + 1e-12), 0.0)

            for k in range(K):
                mk = (bk == k) & rd
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

            s2_idx = np.where(s2_mask)[0]
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
        return rd, s2_mask, X18[s2_mask].astype(np.float32), loao_s1

    rd_f, s2_f, x_f, loao_f = build_features(post_birth=False)
    rd_p, s2_p, x_p, loao_p = build_features(post_birth=True)

    # Merge LOAO Stage-1 impact directly into anchor_meta!
    for k in range(len(anchor_meta)):
        anchor_meta[k]["loao_full_s1"] = loao_f[k]
        anchor_meta[k]["loao_postbirth_s1"] = loao_p[k]

    return {
        "K": len(anchors), "anchor_meta": anchor_meta, "guard_stats": guard_stats,
        "ready_full": rd_f, "s2_full": s2_f, "X18_full": x_f,
        "ready_pb": rd_p, "s2_pb": s2_p, "X18_pb": x_p,
    }


def load_raw_record_arrays(db_name, rec_dir, rec_id):
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
    else:
        i0, i1 = 0, 1
        patient_id = f"{db_name}_{rec_id}"

    s0_raw = np.nan_to_num(rec.p_signal[:, i0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    s1_raw = np.nan_to_num(rec.p_signal[:, i1].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sig0, delay = v33.filter_signal_generic(s0_raw, fs, mode="offline_filtfilt")
    sig1, _ = v33.filter_signal_generic(s1_raw, fs, mode="offline_filtfilt")

    def map_sym(sym):
        if sym in ("N", "L", "R", "e", "j"): return 0
        if sym in ("V", "E"): return 1
        if sym == "F": return 2
        if sym in ("A", "a", "J", "S"): return 3
        return -1

    samples, symbols = ann.sample, ann.symbol
    n_sig = len(sig0)
    if abs(fs - v33.TARGET_FS) < 1e-3:
        beats0, beats1, labels, r_peaks = [], [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < 93 or sc + 94 > n_sig: continue
            c = map_sym(sym)
            if c >= 0:
                beats0.append(sig0[sc - 93 : sc + 94]); beats1.append(sig1[sc - 93 : sc + 94])
                labels.append(c); r_peaks.append(s)
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
            if sc < half_win or sc + half_win + 1 > n_sig: continue
            c = map_sym(sym)
            if c >= 0:
                valid_sc.append(sc); labels.append(c); r_peaks.append(s)
        valid_sc = np.array(valid_sc, dtype=np.int64)
        if len(valid_sc) <= 55: return None
        base_off = valid_sc[:, None] - half_win
        beats0 = (1.0 - frac[None, :]) * sig0[base_off + lo_i[None, :]] + frac[None, :] * sig0[base_off + hi_i[None, :]]
        beats1 = (1.0 - frac[None, :]) * sig1[base_off + lo_i[None, :]] + frac[None, :] * sig1[base_off + hi_i[None, :]]

    beats0 = np.asarray(beats0, dtype=np.float64)
    beats1 = np.asarray(beats1, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    r_peaks = np.asarray(r_peaks, dtype=np.float64)
    N_all = len(labels)
    if N_all <= 55: return None

    pre_rr = np.zeros(N_all, dtype=np.float64)
    pre_rr[1:] = (r_peaks[1:] - r_peaks[:-1]) / fs
    pre_rr[0] = pre_rr[1]
    local_rr = np.zeros(N_all, dtype=np.float64)
    for i in range(N_all):
        lo = 0 if i < 10 else i - 10
        local_rr[i] = float(np.median(pre_rr[lo : i + 1])) + 1e-6
    pre_rr_loc_all = pre_rr / local_rr
    post_hours = float((r_peaks[-1] - r_peaks[50]) / fs / 3600.0)
    return beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, patient_id, post_hours


def prepare_shared_prefix(beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, R_rot=None):
    joint_all = np.column_stack([v33.normalize_lead(beats0), v33.normalize_lead(beats1)])
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
        final_norm_idx = selected_idx[trim_mask] if np.sum(trim_mask) >= 8 else selected_idx
        ectopic_idx = np.where(clusters == (1 - sel_cluster))[0]
        if not ((float(np.median(rr_50[ectopic_idx])) <= 0.92) or (slope_ratio <= 0.70)):
            ectopic_idx = np.array([], dtype=int)
    else:
        med_all = np.median(joint_50, axis=0)
        dists = np.linalg.norm(joint_50 - med_all, axis=1)
        q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
        trim_mask = (dists <= np.median(dists) + 1.5 * (q75 - q25))
        final_norm_idx = np.where(trim_mask)[0] if np.sum(trim_mask) >= 10 else np.arange(M)
        ectopic_idx = np.array([], dtype=int)

    norm_beats = joint_50[final_norm_idx]
    b_N = np.median(norm_beats, axis=0)
    centered_norm = norm_beats - b_N
    _, _, Vt = np.linalg.svd(centered_norm, full_matrices=False)
    U = Vt[:min(4, Vt.shape[0]), :].T
    if R_rot is not None:
        U = U @ R_rot
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

    post_joint = joint_all[50:]
    post_raw0, post_raw1 = beats0[50:], beats1[50:]
    post_y = labels[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    post_r = r_peaks[50:]
    r0_time = float(r_peaks[50])

    post_pre0 = post_raw0[:, v33.PRE_QRS_SLICE] - np.mean(post_raw0[:, v33.PRE_QRS_SLICE], axis=1, keepdims=True)
    post_pre1 = post_raw1[:, v33.PRE_QRS_SLICE] - np.mean(post_raw1[:, v33.PRE_QRS_SLICE], axis=1, keepdims=True)
    hf0 = np.sum(np.diff(post_joint[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1 = np.sum(np.diff(post_joint[:, 187:], n=2, axis=1) ** 2, axis=1)
    sqi_hf_ratio = np.log1p(np.maximum(hf0 / med_hf0, hf1 / med_hf1))
    all_qiso_abs = v33.zenodorus_qiso_abs_batch(post_joint[:, v33.QRS_CORE_SLICE], post_joint[:, 187:][:, v33.QRS_CORE_SLICE])
    zeno_qiso_drop_all = med_pref_qiso_abs - all_qiso_abs
    preqrs_corr_N = 0.5 * (
        v33.pearson_corr_rows(post_raw0[:, v33.PRE_QRS_SLICE], pN_0) +
        v33.pearson_corr_rows(post_raw1[:, v33.PRE_QRS_SLICE], pN_1)
    )
    rr_dev_pre = np.abs(pre_rr_loc - 1.0)
    diff_post = post_joint - b_N[None, :]
    y_mat = diff_post - (diff_post @ U) @ U.T
    norm_y = np.linalg.norm(y_mat, axis=1)
    norm_y_thresh = max(v33.CAND_NORM_Y_MIN, v33.CAND_ETA_MULT * eta_0)

    return (
        post_joint, post_raw0, post_raw1, post_y, pre_rr_loc, post_r, fs, r0_time,
        joint_all, joint_50, norm_beats, final_norm_idx, ectopic_idx, labels[:50],
        b_N, U, P, norm_y_prefix, eta_0,
        pre0_50, pre1_50, pN_0, pN_1, post_pre0, post_pre1,
        pref_hf_p95, pref_bcorr_p05,
        sqi_hf_ratio, zeno_qiso_drop_all, preqrs_corr_N, rr_dev_pre,
        y_mat, norm_y, norm_y_thresh,
    )


def _worker_rec_2x2(args):
    db_name, rec_dir, rec_id = args
    loaded = load_raw_record_arrays(db_name, rec_dir, rec_id)
    if loaded is None:
        return None
    beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, patient_id, post_hours = loaded
    shared = prepare_shared_prefix(beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, R_rot=None)
    out_cfgs = {}
    for cname, k_max, anti_n, anti_f in CONFIGS:
        out_cfgs[cname] = extract_one_config(*shared, k_max=k_max, anti_n=anti_n, anti_f=anti_f)
    return {
        "db": db_name, "rec_id": rec_id, "patient_id": patient_id,
        "post_hours": post_hours, "y": shared[3], "cfgs": out_cfgs,
    }


def verify_o4_invariance():
    R = ortho_group.rvs(4, random_state=20261122)
    max_diffs = {}
    for rid in ("208", "233"):
        loaded = load_raw_record_arrays("mitdb", v36.DB_DIRS["mitdb"], rid)
        beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, _, _ = loaded
        sh0 = prepare_shared_prefix(beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, R_rot=None)
        shR = prepare_shared_prefix(beats0, beats1, labels, pre_rr_loc_all, r_peaks, fs, R_rot=R)
        for cname, k_max, anti_n, anti_f in CONFIGS[1:]:
            o0 = extract_one_config(*sh0, k_max=k_max, anti_n=anti_n, anti_f=anti_f)
            oR = extract_one_config(*shR, k_max=k_max, anti_n=anti_n, anti_f=anti_f)
            assert o0["K"] == oR["K"] and np.array_equal(o0["s2_full"], oR["s2_full"]), f"O(4) mask mismatch on {rid} {cname}"
            d_f = float(np.max(np.abs(o0["X18_full"].astype(np.float64) - oR["X18_full"].astype(np.float64)))) if len(o0["X18_full"]) else 0.0
            max_diffs[f"{rid}_{cname}"] = d_f
    return max_diffs


def build_or_load_db_2x2(db_name):
    meta_p = os.path.join(CACHE_V42B, f"{db_name}_v42b_meta.json")
    npz_p = os.path.join(CACHE_V42B, f"{db_name}_v42b.npz")
    if os.path.exists(meta_p) and os.path.exists(npz_p):
        meta = json.load(open(meta_p, "r", encoding="utf-8"))
        dat = np.load(npz_p)
        out_full = {c: [] for c in CFG_NAMES}
        out_pb = {c: [] for c in CFG_NAMES}
        for m in meta:
            y = dat["y_all"][m["b0"]:m["b1"]]
            for c in CFG_NAMES:
                cm = m["cfgs"][c]
                base = {
                    "db": m["db"], "rec_id": m["rec_id"], "patient_id": m["patient_id"],
                    "post_hours": m["post_hours"], "y": y,
                    "K": cm["K"], "anchor_meta": cm["anchor_meta"], "guard_stats": cm["guard_stats"],
                }
                out_full[c].append({
                    **base,
                    "ready": dat[f"rf_{c}"][m["b0"]:m["b1"]],
                    "s2_mask": dat[f"sf_{c}"][m["b0"]:m["b1"]],
                    "X18_s2": dat[f"xf_{c}"][cm["sf0"]:cm["sf1"]],
                })
                out_pb[c].append({
                    **base,
                    "ready": dat[f"rp_{c}"][m["b0"]:m["b1"]],
                    "s2_mask": dat[f"sp_{c}"][m["b0"]:m["b1"]],
                    "X18_s2": dat[f"xp_{c}"][cm["sp0"]:cm["sp1"]],
                })
        return out_full, out_pb

    d = v36.DB_DIRS[db_name]
    recs = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(d, "*.atr"))))
    if db_name == "mitdb":
        recs = [r for r in recs if r not in {"102", "104", "107", "217"}]
    nw = 4 if db_name in ("ltdb", "sddb") else 6
    t0 = time.time()
    print(f"Extracting v42b 2x2 (5 configs) on {db_name:8s} ({len(recs)} recs, {nw} workers)...", flush=True)
    raw_items = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for res in ex.map(_worker_rec_2x2, [(db_name, d, r) for r in recs]):
            if res is not None:
                raw_items.append(res)

    y_l = []
    arrs = {f"{p}_{c}": [] for c in CFG_NAMES for p in ("rf", "sf", "xf", "rp", "sp", "xp")}
    sf_cur = {c: 0 for c in CFG_NAMES}
    sp_cur = {c: 0 for c in CFG_NAMES}
    bc = 0
    meta = []
    for it in raw_items:
        nb = len(it["y"])
        y_l.append(it["y"])
        cfg_meta = {}
        for c in CFG_NAMES:
            co = it["cfgs"][c]
            nsf, nsp = len(co["X18_full"]), len(co["X18_pb"])
            arrs[f"rf_{c}"].append(co["ready_full"]); arrs[f"sf_{c}"].append(co["s2_full"]); arrs[f"xf_{c}"].append(co["X18_full"])
            arrs[f"rp_{c}"].append(co["ready_pb"]);   arrs[f"sp_{c}"].append(co["s2_pb"]);   arrs[f"xp_{c}"].append(co["X18_pb"])
            cfg_meta[c] = {
                "K": co["K"], "anchor_meta": co["anchor_meta"], "guard_stats": co["guard_stats"],
                "sf0": sf_cur[c], "sf1": sf_cur[c] + nsf, "sp0": sp_cur[c], "sp1": sp_cur[c] + nsp,
            }
            sf_cur[c] += nsf; sp_cur[c] += nsp
        meta.append({
            "db": it["db"], "rec_id": it["rec_id"], "patient_id": it["patient_id"],
            "post_hours": it["post_hours"], "b0": bc, "b1": bc + nb, "cfgs": cfg_meta,
        })
        bc += nb

    save_dict = {"y_all": np.concatenate(y_l)}
    for k, v in arrs.items():
        if k.startswith("xf_") or k.startswith("xp_"):
            save_dict[k] = np.vstack([a for a in v if len(a) > 0]) if any(len(a) > 0 for a in v) else np.zeros((0, 18), dtype=np.float32)
        else:
            save_dict[k] = np.concatenate(v)
    np.savez_compressed(npz_p, **save_dict)
    json.dump(meta, open(meta_p, "w", encoding="utf-8"))
    print(f"  DONE {db_name:8s}: {bc:,} beats in {time.time()-t0:.1f}s", flush=True)
    return build_or_load_db_2x2(db_name)


def fit_lodo_model(tr_items, feat_idx, mtype, cap_pat_pos=100):
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
            if len(it["X18_s2"]) > 0:
                pf_c[it["patient_id"]] += int(np.sum(it["y"][it["s2_mask"]] == 2))
        for it in sub_items:
            if len(it["X18_s2"]) == 0: continue
            Xm = it["X18_s2"][:, feat_idx].astype(np.float64)
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
            if len(te["X18_s2"]) > 0:
                oof_sc.append(clf.decision_function(sc.transform(te["X18_s2"][:, feat_idx].astype(np.float64))))
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


def score_and_eval(te_items, feat_idx, scaler, clf, tau1):
    tot_hours = sum(it["post_hours"] for it in te_items)
    sc_l, y_l, rec_sc = [], [], []
    for it in te_items:
        sc = np.full(len(it["y"]), -1e6, dtype=np.float64)
        if len(it["X18_s2"]) > 0:
            sc[it["s2_mask"]] = clf.decision_function(scaler.transform(it["X18_s2"][:, feat_idx].astype(np.float64)))
        sc_l.append(sc); y_l.append(it["y"]); rec_sc.append((it, sc))
    sc_cat = np.concatenate(sc_l); y_cat = np.concatenate(y_l)
    y_bin = (y_cat == 2).astype(int)
    tot_ft = int(np.sum(y_bin == 1))
    ap = float(average_precision_score(y_bin, sc_cat)) if tot_ft > 0 else 0.0

    # Post-hoc oracle-threshold ranking @ FAPH<=2.0
    valid = (sc_cat > -1e5)
    sv = sc_cat[valid]; yv = (y_cat[valid] == 2)
    ord_v = np.argsort(sv)[::-1]
    ok2 = np.where(np.cumsum(~yv[ord_v]) <= int(np.floor(2.0 * tot_hours)))[0]
    tau_f2 = float(sv[ord_v[ok2[-1]]]) if len(ok2) > 0 else float(sv[ord_v[0]]) + 1.0
    tp2 = int(np.sum((sc_cat >= tau_f2) & (y_cat == 2)))
    fp2 = int(np.sum((sc_cat >= tau_f2) & (y_cat != 2)))
    f1_f2 = 2.0 * tp2 / max(1, 2 * tp2 + fp2 + (tot_ft - tp2))

    # Source-locked deployed T1
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

    pat_f = Counter(); pat_tp2 = Counter()
    for it, sc in rec_sc:
        nf = int(np.sum(it["y"] == 2))
        if nf > 0:
            pat_f[it["patient_id"]] += nf
            pat_tp2[it["patient_id"]] += int(np.sum((sc >= tau_f2) & (it["y"] == 2)))
    mp_rec2 = round(float(np.mean([100.0 * pat_tp2[p] / pat_f[p] for p in pat_f])), 2) if pat_f else 0.0

    # Stage-1 survival + Anchor purity + Per-anchor LOAO summary
    s1_f = sum(int(np.sum((it["y"] == 2) & it["s2_mask"])) for it, _ in rec_sc)
    s1_v = sum(int(np.sum((it["y"] == 1) & it["s2_mask"])) for it, _ in rec_sc)
    all_ancs = [a for it, _ in rec_sc for a in it["anchor_meta"]]
    new_ancs = [a for a in all_ancs if a["slot"] >= 5]
    n_false_n = sum(1 for a in all_ancs if a["n_frac"] >= 0.5)
    n_false_f = sum(1 for a in all_ancs if a["f_frac"] > 0.0)
    new_false_n = sum(1 for a in new_ancs if a["n_frac"] >= 0.5)
    new_false_f = sum(1 for a in new_ancs if a["f_frac"] > 0.0)

    ev = {
        "AP": round(ap, 4),
        "Rank_Rec@F2_%": round(100.0 * tp2 / max(1, tot_ft), 2), "Rank_F1@F2": round(f1_f2, 4),
        "Rank_MacroPat@F2_%": mp_rec2,
        "T1_F1": round(f1_t1, 4), "T1_Prec_%": round(prec1, 2), "T1_Rec_%": round(rec1, 2),
        "T1_Spec_%": round(spec, 3), "T1_FAPH": round(faph1, 2),
        "T1_TP": tp1, "T1_FPN": fpn1, "T1_FPV": fpv1, "T1_FPS": fps1,
        "T1_Counts": f"{tp1}/{fpn1}/{fpv1}/{fps1}S",
        "S1_F_Surv": s1_f, "Tot_F": tot_ft, "S1_V_Surv": s1_v,
        "Tot_Anc": len(all_ancs), "New_Anc_k5p": len(new_ancs),
        "All_VPur_%": round(100.0 * float(np.mean([a["v_pur"] for a in all_ancs])), 1) if all_ancs else 0.0,
        "New_VPur_%": round(100.0 * float(np.mean([a["v_pur"] for a in new_ancs])), 1) if new_ancs else 0.0,
        "All_FalseN": n_false_n, "All_AnyF": n_false_f,
        "New_FalseN": new_false_n, "New_AnyF": new_false_f,
    }
    return ev, rec_sc


def summarize_per_anchor_scope(te_items, mode_key="loao_full_s1"):
    """
    Aggregates the user's requested per-anchor scope-of-impact metric:
      "Each new anchor (k >= 5) rejects how many additional V beats and how many additional F beats?"
    """
    rows = []
    for it in te_items:
        for a in it["anchor_meta"]:
            if a["slot"] >= 5:
                lo = a[mode_key]
                rows.append({
                    "rec_id": it["rec_id"], "slot": a["slot"],
                    "v_cnt": a["v_cnt"], "f_cnt": a["f_cnt"], "n_cnt": a["n_cnt"],
                    "v_pur": a["v_pur"], "f_frac": a["f_frac"], "pct_tape": a["pct_tape"],
                    **lo,
                })
    n_tot = len(rows)
    n_clean_gain = sum(1 for r in rows if r["v_excl_s1"] > 0 and r["f_excl_s1"] == 0)
    n_tradeoff = sum(1 for r in rows if r["v_excl_s1"] > 0 and r["f_excl_s1"] > 0)
    n_harm_only = sum(1 for r in rows if r["v_excl_s1"] == 0 and r["f_excl_s1"] > 0)
    n_zero_v = sum(1 for r in rows if r["v_excl_s1"] == 0 and r["f_excl_s1"] == 0)
    rows_worst_f = sorted([r for r in rows if r["f_excl_s1"] > 0], key=lambda r: -r["f_excl_s1"])[:8]
    rows_best_v = sorted(rows, key=lambda r: -r["v_excl_s1"])[:8]
    return {
        "n_new_anc": n_tot,
        "sum_v_excl_s1": sum(r["v_excl_s1"] for r in rows),
        "sum_f_excl_s1": sum(r["f_excl_s1"] for r in rows),
        "sum_f_gain_s1": sum(r["f_gain_s1"] for r in rows),
        "n_clean_v_gain_only": n_clean_gain,
        "n_tradeoff_v_and_f": n_tradeoff,
        "n_f_harm_only": n_harm_only,
        "n_zero_impact": n_zero_v,
        "top_f_killers": rows_worst_f,
        "top_v_eliminators": rows_best_v,
    }


def extract_spotlight_records(rec_sc_by_cfg, tau1_k4, db_name):
    rids = SPOTLIGHT_RECS.get(db_name, [])
    if not rids:
        return {}
    out = {}
    for rid in rids:
        rec_out = {}
        # Lookup K4 baseline for deltas
        m_k4 = {it["rec_id"]: (it, sc) for it, sc in rec_sc_by_cfg["K4_Base"]}
        if rid not in m_k4:
            continue
        it4, sc4 = m_k4[rid]
        y = it4["y"]
        p4 = (sc4 >= tau1_k4)
        s2_4 = it4["s2_mask"]
        for cname in CFG_NAMES:
            mc = {it["rec_id"]: (it, sc) for it, sc in rec_sc_by_cfg[cname]}
            itc, scc = mc[rid]
            pc = (scc >= tau1_k4)
            s2_c = itc["s2_mask"]
            anc_rows = []
            for a in itc["anchor_meta"]:
                lo = a["loao_full_s1"]
                anc_rows.append({
                    "slot": a["slot"], "src": a["src"], "pct": a["pct_tape"],
                    "seed": f"V{a['v_cnt']}F{a['f_cnt']}N{a['n_cnt']}S{a['s_cnt']}",
                    "v_pur": a["v_pur"], "f_frac": a["f_frac"],
                    "loao_v_excl_s1": lo["v_excl_s1"], "loao_f_excl_s1": lo["f_excl_s1"],
                    "loao_f_gain_s1": lo["f_gain_s1"],
                })
            rec_out[cname] = {
                "K": itc["K"], "tot_F": int(np.sum(y == 2)), "tot_V": int(np.sum(y == 1)),
                "S1_F": int(np.sum((y == 2) & s2_c)), "S1_V": int(np.sum((y == 1) & s2_c)),
                "T1_TP": int(np.sum((y == 2) & pc)),
                "T1_FPN": int(np.sum((y == 0) & pc)),
                "T1_FPV": int(np.sum((y == 1) & pc)),
                "dS1_F_vs_K4": int(np.sum((y == 2) & s2_c)) - int(np.sum((y == 2) & s2_4)),
                "dT1_TP_vs_K4": int(np.sum((y == 2) & pc)) - int(np.sum((y == 2) & p4)),
                "dT1_FPV_vs_K4": int(np.sum((y == 1) & pc)) - int(np.sum((y == 1) & p4)),
                "guard_stats": itc["guard_stats"],
                "anchors": anc_rows,
            }
        out[rid] = rec_out
    return out


def main():
    t0 = time.time()
    print("=== Step 1: Verifying Exact O(4) Gauge Invariance (MIT-208 & MIT-233, all 4 K=12 configs) ===", flush=True)
    o4_diffs = verify_o4_invariance()
    max_o4 = max(o4_diffs.values())
    print(f"  O(4) max abs diff across all configs: {max_o4:.3e} ({o4_diffs})", flush=True)
    assert max_o4 < 1e-11, f"O(4) invariance violated: {max_o4}"

    print("\n=== Step 2: Extracting/Loading 6-Hospital 2x2 Guard Caches ===", flush=True)
    full_dbs = {c: {} for c in CFG_NAMES}
    pb_dbs = {c: {} for c in CFG_NAMES}
    for db in DB_ORDER:
        of, op = build_or_load_db_2x2(db)
        for c in CFG_NAMES:
            full_dbs[c][db] = of[c]
            pb_dbs[c][db] = op[c]

    # Sanity-check exact parity between our freshly extracted K4_Base and v36 cache_gen3_6db!
    print("\n=== Step 3: Sanity-Checking K4_Base Parity vs cache_gen3_6db ===", flush=True)
    for db in DB_ORDER:
        m36 = json.load(open(os.path.join(CACHE_K4, f"{db}_gen3_4cls_meta.json"), "r", encoding="utf-8"))
        d36 = np.load(os.path.join(CACHE_K4, f"{db}_gen3_4cls.npz"))
        m_old = {m["rec_id"]: (m["K"], int(np.sum(d36["s2_all"][m["b0"]:m["b1"]]))) for m in m36}
        k_diff = sum(1 for it in full_dbs["K4_Base"][db] if it["K"] != m_old[it["rec_id"]][0])
        s2_diff = sum(abs(int(np.sum(it["s2_mask"])) - m_old[it["rec_id"]][1]) for it in full_dbs["K4_Base"][db])
        print(f"  Parity {db:8s}: K_diffs={k_diff}, S2_beat_diffs={s2_diff}", flush=True)

    summary = {"O4_Max_Diff": max_o4, "O4_Details": o4_diffs, "Models": {}}

    for mname, (fidx, mtype) in MODELS.items():
        summary["Models"][mname] = {}
        print(f"\n==================== {mname}: 2x2 FACTORIAL AT K=12 (FROZEN K4 HEAD & T1) ====================", flush=True)
        for tgt in DB_ORDER:
            # Train & FREEZE exact K=4 LODO model + threshold tau1_k4 (FullTape!)
            tr_k4 = [it for db in DB_ORDER if db != tgt for it in full_dbs["K4_Base"][db]]
            sc_k4, clf_k4, tau1_k4 = fit_lodo_model(tr_k4, fidx, mtype, cap_pat_pos=100)

            res_full, rsc_full = {}, {}
            res_pb, rsc_pb = {}, {}
            scope_full, scope_pb = {}, {}
            for cname in CFG_NAMES:
                ev_f, r_f = score_and_eval(full_dbs[cname][tgt], fidx, sc_k4, clf_k4, tau1_k4)
                ev_p, r_p = score_and_eval(pb_dbs[cname][tgt], fidx, sc_k4, clf_k4, tau1_k4)
                res_full[cname] = ev_f; rsc_full[cname] = r_f
                res_pb[cname] = ev_p;   rsc_pb[cname] = r_p
                scope_full[cname] = summarize_per_anchor_scope(full_dbs[cname][tgt], "loao_full_s1")
                scope_pb[cname] = summarize_per_anchor_scope(pb_dbs[cname][tgt], "loao_postbirth_s1")

            # Also fit Retrained LODO on C11_Both (reference only!)
            tr_c11 = [it for db in DB_ORDER if db != tgt for it in full_dbs["C11_Both"][db]]
            sc_c11, clf_c11, tau1_c11 = fit_lodo_model(tr_c11, fidx, mtype, cap_pat_pos=100)
            ev_c11_ret, _ = score_and_eval(full_dbs["C11_Both"][tgt], fidx, sc_c11, clf_c11, tau1_c11)

            spot_full = extract_spotlight_records(rsc_full, tau1_k4, tgt)
            spot_pb = extract_spotlight_records(rsc_pb, tau1_k4, tgt)

            summary["Models"][mname][tgt] = {
                "FullTape_FrozenK4Head": res_full,
                "PostBirth_FrozenK4Head": res_pb,
                "C11_FullTape_RetrainedHead": ev_c11_ret,
                "PerAnchor_Scope_FullTape": scope_full,
                "PerAnchor_Scope_PostBirth": scope_pb,
                "Spotlights_FullTape": spot_full,
                "Spotlights_PostBirth": spot_pb,
            }

            print(f"\n  [{tgt}] FullTape (Frozen K=4 Head & T1):", flush=True)
            for cname in CFG_NAMES:
                ev = res_full[cname]
                sc_a = scope_full[cname]
                print(
                    f"    {cname:13s} | Anc={ev['Tot_Anc']:3d}(k5+:{ev['New_Anc_k5p']:2d}, Vp={ev['New_VPur_%']:5.1f}%, FN={ev['All_FalseN']:2d}, FF={ev['All_AnyF']:2d}) | "
                    f"S1_F={ev['S1_F_Surv']:4d}/{ev['Tot_F']:4d} | AP={ev['AP']:.4f} Rank@F2={ev['Rank_Rec@F2_%']:5.1f}% | "
                    f"T1: F1={ev['T1_F1']:.4f} [{ev['T1_Counts']:16s}] FAPH={ev['T1_FAPH']:5.2f} | "
                    f"LOAO(k5+): -V={sc_a['sum_v_excl_s1']:4d} -F={sc_a['sum_f_excl_s1']:3d} +F={sc_a['sum_f_gain_s1']:3d}",
                    flush=True,
                )
            if spot_full:
                for rid, rdat in spot_full.items():
                    parts = []
                    for cname in CFG_NAMES:
                        rd = rdat[cname]
                        parts.append(f"{cname[:3]}:K={rd['K']},S1F={rd['S1_F']},TP={rd['T1_TP']},FPV={rd['T1_FPV']}")
                    print(f"      SPOT {rid:6s}: " + " | ".join(parts), flush=True)

    out_json = os.path.join(AUD_DIR, "V42B_HYPB1_ANCHOR_GUARDS_2X2_AUDIT_20261122.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write clean publication-grade Markdown audit
    out_md = os.path.join(AUD_DIR, "V42B_HYPB1_ANCHOR_GUARDS_2X2_AUDIT_20261122.md")
    lines = [
        "# v42b Hypothesis B1: 2x2 Factorial on V-Anchor Confirmation Guards at `K_max = 12`",
        "",
        "- **Strict Control**: Identical Static `v36` N-chart, `MAX_CAND_CLUSTERS = 8`, ZERO labels in anchor creation, identical 18D features, AND **Frozen `K=4` classifier weights + Frozen `K=4` source-locked `T1` threshold**.",
        f"- **Exact $O(4)$ Gauge Invariance**: Verified on `MIT-208` & `MIT-233` across all 4 `K=12` configs (`max |diff| = {max_o4:.3e}`).",
        "- **Locked DEV-Only Guard Rules** (chosen strictly on `mitdb + edb`; zero tuning on `ltdb/incartdb/sddb/svdb`; `LTDB-14134` explicitly treated as a *known-error fix check*):",
        "  - **Anti-N (`C10`, `C11`)**: Pre-existing `v36b` His-Purkinje narrow-QRS gate `min_qrs_corr_2lead(x, b_N_loc) <= 0.85` (entry & cluster medoid).",
        "  - **Anti-F (`C01`, `C11`)**: Reject cluster confirmation if `(med_rr > 0.80 and (m_bridge(med_b) > 0 or med_preqrs_loc >= 0.40)) or (m_bridge(med_b) > 0 and pyth_margin(med_b) > 0 and med_preqrs_loc >= 0.40)`.",
        "- **Online Terminology**: `PostBirth` masks every anchor strictly to `t >= disc_idx` using end-of-tape V-radii (*chi dung neo sau thoi diem sinh*, NOT full causal), evaluated apples-to-apples against `K4_PostBirth`.",
        "",
    ]

    for mname in MODELS.keys():
        lines.extend([
            f"## 1. FullTape 2x2 Factorial (`{mname}`, Frozen `K=4` Head & `T1`)", "",
            "| DB | Config | Anc (k>=5) | New V_pur (%) | FalseN / AnyF (All) | S1 F Surv | AP | Rank@F2 (%) | Deployed `T1` F1 | TP/FP_N/FP_V/FP_S | FAPH | LOAO k>=5 (`-V_S1` / `-F_S1` / `+F_S1`) |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|",
        ])
        for tgt in DB_ORDER:
            s = summary["Models"][mname][tgt]
            for cname in CFG_NAMES:
                ev = s["FullTape_FrozenK4Head"][cname]
                sc = s["PerAnchor_Scope_FullTape"][cname]
                lines.append(
                    f"| `{tgt}` | `{cname}` | {ev['Tot_Anc']} ({ev['New_Anc_k5p']}) | {ev['New_VPur_%']:.1f} | "
                    f"{ev['All_FalseN']} / {ev['All_AnyF']} | {ev['S1_F_Surv']}/{ev['Tot_F']} | **{ev['AP']:.4f}** | "
                    f"**{ev['Rank_Rec@F2_%']:.1f}** | **{ev['T1_F1']:.4f}** | {ev['T1_Counts']} | {ev['T1_FAPH']:.2f} | "
                    f"-{sc['sum_v_excl_s1']}V / -{sc['sum_f_excl_s1']}F / +{sc['sum_f_gain_s1']}F |"
                )

        lines.extend([
            "", f"## 2. Apples-to-Apples `PostBirth` (*chi dung neo sau thoi diem sinh*, `{mname}`, Frozen `K=4` Head & `T1`)", "",
            "| DB | Config | S1 F Surv | AP | Rank@F2 (%) | Deployed `T1` F1 | TP/FP_N/FP_V/FP_S | FAPH | LOAO k>=5 (`-V_S1` / `-F_S1`) |",
            "|---|---|---:|---:|---:|---:|---|---:|---|",
        ])
        for tgt in DB_ORDER:
            s = summary["Models"][mname][tgt]
            for cname in CFG_NAMES:
                ev = s["PostBirth_FrozenK4Head"][cname]
                sc = s["PerAnchor_Scope_PostBirth"][cname]
                lines.append(
                    f"| `{tgt}` | `{cname}` | {ev['S1_F_Surv']}/{ev['Tot_F']} | **{ev['AP']:.4f}** | "
                    f"**{ev['Rank_Rec@F2_%']:.1f}** | **{ev['T1_F1']:.4f}** | {ev['T1_Counts']} | {ev['T1_FAPH']:.2f} | "
                    f"-{sc['sum_v_excl_s1']}V / -{sc['sum_f_excl_s1']}F |"
                )

        lines.extend([
            "", f"## 3. Target Record Spotlights (`{mname}`, FullTape, Frozen `K=4` Head & `T1`)", "",
            "| DB | Record | Config | K | S1_F / Tot_F | T1_TP | T1_FP_V | T1_FP_N | Per-Anchor LOAO (`slot: seed -> -V_S1, -F_S1, +F_S1`) |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for tgt in ("ltdb", "mitdb", "edb", "incartdb"):
            spots = summary["Models"][mname][tgt]["Spotlights_FullTape"]
            for rid, rdat in spots.items():
                for cname in CFG_NAMES:
                    rd = rdat[cname]
                    anc_str = "; ".join(
                        f"k{a['slot']}({a['seed']}: -{a['loao_v_excl_s1']}V,-{a['loao_f_excl_s1']}F,+{a['loao_f_gain_s1']}F)"
                        for a in rd["anchors"] if a["slot"] >= 5 or rd["K"] <= 4
                    )
                    lines.append(
                        f"| `{tgt}` | `{rid}` | `{cname}` | {rd['K']} | {rd['S1_F']}/{rd['tot_F']} | "
                        f"{rd['T1_TP']} | {rd['T1_FPV']} | {rd['T1_FPN']} | {anc_str} |"
                    )
        lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nALL DONE in {time.time()-t0:.1f}s!\nSaved: {out_json}\nSaved: {out_md}", flush=True)


if __name__ == "__main__":
    main()
