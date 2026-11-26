r"""
lab_audit_v58_exact_beatwise_m3_and_ablation.py
===============================================
BIT-EXACT BEAT-BY-BEAT CAUSAL EXTRACTION & COMPLETE MECHANISTIC ABLATION FOR M3 (v58)
(ZERO stride=16 APPROXIMATION, ZERO STAGE-1 CHANGES, ZERO NEW STAGE-1 HYPERPARAMETERS, ZERO LTSTDB)

1. Builds & verifies a bit-exact beat-by-beat causal cache (`cache_v58_exact_m3_6db`) directly
   from the 306 WFDB source records using exact b_N_loc[t] at every beat t (no stride=16),
   asserting bit-exact parity (< 1e-4) against cache_v47 on z_N, z_V, loc_norm_corr, loc_dist_norm.
   Extracts ONLY 6 clean, dimensionless, scale-invariant causal geometric primitives:
     - Layer h/Phi (Prefix Drift Consistency & Normal-Tangent Orthogonality):
         [24] C_drift           = -|log((d_proj(t) + 0.5) / (||y(t)||_2 + 0.5))|      <= 0
         [25] log_tan_orth      = log((d_proj(t) + 0.5) / (d_par_U(t) + 0.5))
     - Layer w (Depolarization QRS [71:123] vs Repolarization ST-T [123:187] Residual Decomposition):
         [26] tanh_d_qrs        = tanh((||(x_t - b_N_loc[t])_QRS||_2 / sqrt(104)) / 0.75)
         [27] tanh_d_stt        = tanh((||(x_t - b_N_loc[t])_STT||_2 / sqrt(128)) / 0.75)
         [28] log_qrs_vs_nonqrs = log((d_qrs(t) + 0.5) / (d_nonqrs(t) + 0.5))
         [29] log_dv_qrs_norm   = log1p(||(x_t - b_V[k])_QRS||_2 / (||(b_V[k] - b_N_loc[t])_QRS||_2 + 1e-6))

2. Runs strict 6-hospital LODO (SubjectAware_251: 66 F+ / 185 True 0-F) on BOTH SVM (primary)
   and LogR (supporting check) across the complete mechanistic decomposition ladder:
     - A0_B0_R1           : Exact v49/v50 C3 baseline under R1 (bit-exact parity asserted!)
     - A1_B0_R3           : Exact C3 features under R3_All2 (calibration alone)
     - A2_HPhi_NS_R3      : B0 + Layer h/Phi [C_drift, log_tan_orth] in F:N/S; F:V locked
     - A3_HPhiW_NS_R3     : B0 + [C_drift, log_tan_orth, tanh_d_qrs, tanh_d_stt, log_qrs_vs_nonqrs] in F:N/S; F:V locked
     - A4_W_FV_Only_R3    : F:N/S locked at B0; F:V + [tanh_d_qrs, tanh_d_stt, log_qrs_vs_nonqrs, log_dv_qrs_norm]
     - A5_Full_M3_Both_R3 : Full 6-primitive specialized model (F:N/S 15D, F:V 15D) under R3_All2
                            (plus sensitivity at B_tot in {1.5, 2.0, 2.5} and R3_All2_0F1!)

3. Reports:
   - Exact deployed T1 metrics, True 0-F tail quantiles (p50/p90/p95/p99/max, %==0, %<=2, worst-5 0-F units)
   - Exact matched-FA curves (S_min @ AllFAPH & @ 0FFAPH, plus pure m_N @ fixed B0 F:V gate)
   - Exact per-DB breakdown across all 6 DBs
   - Paired Wilcoxon signed-rank & binomial sign tests on 66 F+ Pat@F2 and deployed T1
   - Full helped/harmed F+ unit table for A5 vs A0_B0_R1
   - Writes complete audit JSON + publication-grade Markdown report.
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
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import wilcoxon, binomtest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import wfdb

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import lab_diagnostic_v36_6hospital_lodo_benchmark as v36
import lab_diagnostic_v46_two_branch_stage2_rr as v46
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_audit_v49_subject_unit_and_registry as v49
import lab_diagnostic_v53b_lodo_disentanglement_ablation as v53b
import lab_diagnostic_v56_qrs_stt_local_chord_lodo as v56

AUD_DIR = str(_REPO_ROOT / "audits")
CACHE_V58 = str(_DATA_ROOT / "cache_v58_exact_m3_6db")
os.makedirs(AUD_DIR, exist_ok=True)
os.makedirs(CACHE_V58, exist_ok=True)

DB_ORDER = v53b.DB_ORDER
F2_IDX = list(v47.F2_IDX)        # [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]
IDX_C3_V = list(v47.IDX_C3)      # F2_IDX + [18]

SL_PRE = slice(0, 71)
SL_QRS = v33.QRS_CORE_SLICE      # slice(71, 123)
SL_STT = slice(123, 187)
R_QRS = float(np.sqrt(2.0 * 52.0))
R_STT = float(np.sqrt(2.0 * 64.0))


def _extract_exact_m3_one_record(args):
    """Bit-exact Gen-3 Stage-1 replay extracting the 6 exact beat-by-beat M3 primitives."""
    db_name, rec_id = args
    rec_path = os.path.join(v36.DB_DIRS[db_name], rec_id)
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")
    fs = float(rec.fs)

    if db_name == "incartdb":
        sig_names = [s.upper().strip() for s in rec.sig_name]
        i0, i1 = sig_names.index("II"), sig_names.index("V1")
    else:
        i0, i1 = 0, 1
    s0_raw = np.nan_to_num(rec.p_signal[:, i0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    s1_raw = np.nan_to_num(rec.p_signal[:, i1].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sig0, delay = v33.filter_signal_generic(s0_raw, fs, mode="offline_filtfilt")
    sig1, _ = v33.filter_signal_generic(s1_raw, fs, mode="offline_filtfilt")

    n_sig = len(sig0)
    samples, symbols = ann.sample, ann.symbol
    if abs(fs - v33.TARGET_FS) < 1e-3:
        beats0, beats1, labels, r_peaks = [], [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < 93 or sc + 94 > n_sig:
                continue
            c = v53b._map_sym_v10(sym)
            if c >= 0:
                beats0.append(sig0[sc - 93 : sc + 94])
                beats1.append(sig1[sc - 93 : sc + 94])
                labels.append(c)
                r_peaks.append(s)
        beats0 = np.asarray(beats0, dtype=np.float64)
        beats1 = np.asarray(beats1, dtype=np.float64)
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
            c = v53b._map_sym_v10(sym)
            if c >= 0:
                valid_sc.append(sc)
                labels.append(c)
                r_peaks.append(s)
        valid_sc = np.array(valid_sc, dtype=np.int64)
        base_off = valid_sc[:, None] - half_win
        beats0 = (1.0 - frac[None, :]) * sig0[base_off + lo_i[None, :]] + frac[None, :] * sig0[base_off + hi_i[None, :]]
        beats1 = (1.0 - frac[None, :]) * sig1[base_off + lo_i[None, :]] + frac[None, :] * sig1[base_off + hi_i[None, :]]

    labels = np.asarray(labels, dtype=np.int8)
    r_peaks = np.asarray(r_peaks, dtype=np.float64)
    N_all = len(labels)
    pre_rr = np.zeros(N_all, dtype=np.float64)
    pre_rr[1:] = (r_peaks[1:] - r_peaks[:-1]) / fs
    pre_rr[0] = pre_rr[1]
    local_rr = np.zeros(N_all, dtype=np.float64)
    for i in range(min(10, N_all)):
        local_rr[i] = float(np.median(pre_rr[: i + 1])) + 1e-6
    if N_all > 10:
        local_rr[10:] = np.median(sliding_window_view(pre_rr, 11), axis=1) + 1e-6
    pre_rr_loc_all = pre_rr / local_rr

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

    hf0_50 = np.sum(np.diff(joint_50[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1_50 = np.sum(np.diff(joint_50[:, 187:], n=2, axis=1) ** 2, axis=1)
    med_hf0 = float(np.median(hf0_50[final_norm_idx])) + 1e-6
    med_hf1 = float(np.median(hf1_50[final_norm_idx])) + 1e-6

    anchors = []
    if len(ectopic_idx) >= 4:
        anc = v33.make_anchor_obj(joint_50[ectopic_idx], norm_y_prefix, b_N, P, 0, "prefix")
        if anc is not None:
            anchors.append(anc)

    post_joint = joint_all[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    N = len(post_joint)

    hf0 = np.sum(np.diff(post_joint[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1 = np.sum(np.diff(post_joint[:, 187:], n=2, axis=1) ** 2, axis=1)
    sqi_hf_ratio = np.log1p(np.maximum(hf0 / med_hf0, hf1 / med_hf1))

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

    b_N_loc_hist = np.zeros((N, D), dtype=np.float32)
    d_proj_arr = np.zeros(N, dtype=np.float64)
    d_full_arr = np.zeros(N, dtype=np.float64)
    loc_corr_arr = np.zeros(N, dtype=np.float64)
    prev1, prev2 = joint_all[49], joint_all[48]
    cand_clusters = []

    for t in range(N):
        x_t = post_joint[t]
        b_N_loc_hist[t] = b_N_loc
        c_loc = v33.mean_lead_corr(x_t, b_N_loc)
        loc_corr_arr[t] = c_loc
        dl = x_t - b_N_loc
        d_full_arr[t] = float(np.linalg.norm(dl))
        dl_p = dl - U @ (U.T @ dl)
        d_proj_arr[t] = float(np.linalg.norm(dl_p))

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

    K = len(anchors)
    if K == 0:
        return (db_name, rec_id, None)

    Q_mat = np.vstack([a["q"] for a in anchors])
    BV_mat = np.vstack([a["b_V"] for a in anchors])
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
    best_lam = lam_all[idx_n, bk]
    e_V_any = np.min(ej_all, axis=1)
    e_N = np.maximum(0.0, (best_lam - muN_vec[bk]) - radN_vec[bk])
    e_Vc = np.maximum(0.0, (1.0 - best_lam) - radV_vec[bk])
    z_N = np.log1p(e_N / radN_vec[bk])
    z_V = np.minimum(np.log1p(e_Vc / radV_vec[bk]), np.log1p(e_V_any / radV_vec[bk]))
    s2 = (z_N > 0.0) & (z_V > 0.0)
    if not np.any(s2):
        return (db_name, rec_id, None)

    # Exact beat-by-beat M3 6 primitives on Stage-2 beats!
    x_s2 = post_joint[s2].reshape(-1, 2, 187)
    bN_s2 = b_N_loc_hist[s2].astype(np.float64).reshape(-1, 2, 187)
    bV_s2 = BV_mat[bk[s2]].reshape(-1, 2, 187)

    dp = d_proj_arr[s2]
    df = d_full_arr[s2]
    ny = norm_y[s2]
    dpar = np.sqrt(np.maximum(0.0, df ** 2 - dp ** 2))

    # Exact Layer h/Phi:
    C_drift = -np.abs(np.log((dp + 0.5) / (ny + 0.5)))
    log_tan_orth = np.log((dp + 0.5) / (dpar + 0.5))

    # Exact Layer w (beat-by-beat b_N_loc[t] QRS vs ST-T):
    dl = x_s2 - bN_s2
    dv = bV_s2 - bN_s2
    dx_v = x_s2 - bV_s2

    d_pre = np.linalg.norm(dl[:, :, SL_PRE].reshape(len(x_s2), -1), axis=1)
    d_qrs = np.linalg.norm(dl[:, :, SL_QRS].reshape(len(x_s2), -1), axis=1)
    d_stt = np.linalg.norm(dl[:, :, SL_STT].reshape(len(x_s2), -1), axis=1)
    d_nonq = np.sqrt(d_pre ** 2 + d_stt ** 2)

    tanh_d_qrs = np.tanh((d_qrs / R_QRS) / 0.75)
    tanh_d_stt = np.tanh((d_stt / R_STT) / 0.75)
    log_qrs_vs_nonqrs = np.log((d_qrs + 0.5) / (d_nonq + 0.5))
    nv_q = np.linalg.norm(dv[:, :, SL_QRS].reshape(len(x_s2), -1), axis=1) + 1e-6
    dv_qrs_norm = np.linalg.norm(dx_v[:, :, SL_QRS].reshape(len(x_s2), -1), axis=1) / nv_q
    log_dv_qrs_norm = np.log1p(np.maximum(0.0, dv_qrs_norm))

    M6 = np.column_stack([
        C_drift, log_tan_orth, tanh_d_qrs, tanh_d_stt, log_qrs_vs_nonqrs, log_dv_qrs_norm
    ]).astype(np.float32)

    chk = np.column_stack([
        z_N[s2], z_V[s2], loc_corr_arr[s2], dp / (nq_vec[bk[s2]] + 1e-12)
    ]).astype(np.float32)

    return (db_name, rec_id, M6, chk)


def build_or_load_v58_exact_cache(dbs):
    ready = all(
        os.path.exists(os.path.join(CACHE_V58, f"{db}_m6.npz"))
        and os.path.exists(os.path.join(CACHE_V58, f"{db}_meta.json"))
        for db in DB_ORDER
    )
    if ready:
        print("[v58] Loading verified bit-exact M3 beat-by-beat cache from disk...", flush=True)
        out = {}
        for db in DB_ORDER:
            meta = json.load(open(os.path.join(CACHE_V58, f"{db}_meta.json"), "r", encoding="utf-8"))
            arr = np.load(os.path.join(CACHE_V58, f"{db}_m6.npz"))["m6"]
            out[db] = {m["rec_id"]: (arr[m["i0"]:m["i1"]] if m["i1"] > m["i0"] else None) for m in meta}
        return out

    t0 = time.time()
    print("[v58] Building ONE-TIME bit-exact beat-by-beat M3 cache across all 306 WFDB records...", flush=True)
    tasks = []
    it_map = {}
    for db in DB_ORDER:
        for it in dbs[db]:
            it_map[(db, it["rec_id"])] = it
            if len(it["X_s2"]) == 0:
                continue
            tasks.append((db, it["rec_id"]))

    per_db = {db: {} for db in DB_ORDER}
    max_err = {"z_N": 0.0, "z_V": 0.0, "lnc": 0.0, "ldn": 0.0}
    n_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for done, res in enumerate(ex.map(_extract_exact_m3_one_record, tasks), 1):
            db, rid = res[0], res[1]
            it = it_map[(db, rid)]
            assert res[2] is not None and len(res[2]) == len(it["X_s2"]), f"Len mismatch {db}/{rid}"
            M6, chk = res[2], res[3]
            X = it["X_s2"]
            e_zn = float(np.max(np.abs(chk[:, 0] - X[:, 1])))
            e_zv = float(np.max(np.abs(chk[:, 1] - X[:, 2])))
            e_lc = float(np.max(np.abs(chk[:, 2] - X[:, 4])))
            e_ld = float(np.max(np.abs(chk[:, 3] - X[:, 5])))
            max_err["z_N"] = max(max_err["z_N"], e_zn)
            max_err["z_V"] = max(max_err["z_V"], e_zv)
            max_err["lnc"] = max(max_err["lnc"], e_lc)
            max_err["ldn"] = max(max_err["ldn"], e_ld)
            assert max(e_zn, e_zv, e_lc, e_ld) < 1e-3, f"Parity FAIL on {db}/{rid}: {e_zn},{e_zv},{e_lc},{e_ld}"
            per_db[db][rid] = M6
            if done % 40 == 0 or done == len(tasks):
                print(f"  [v58] {done}/{len(tasks)} records exact-verified ({time.time() - t0:.1f}s) | max_err={max_err}", flush=True)

    for db in DB_ORDER:
        chunks, meta, cur = [], [], 0
        for it in dbs[db]:
            rid = it["rec_id"]
            m6 = per_db[db].get(rid)
            if m6 is not None:
                chunks.append(m6)
                meta.append({"rec_id": rid, "i0": cur, "i1": cur + len(m6)})
                cur += len(m6)
            else:
                meta.append({"rec_id": rid, "i0": cur, "i1": cur})
        arr = np.vstack(chunks) if chunks else np.zeros((0, 6), dtype=np.float32)
        np.savez_compressed(os.path.join(CACHE_V58, f"{db}_m6.npz"), m6=arr)
        with open(os.path.join(CACHE_V58, f"{db}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    print(f"[v58] Exact cache saved & verified bit-exact ({time.time() - t0:.1f}s): max_err={max_err}", flush=True)
    return {db: {it["rec_id"]: per_db[db].get(it["rec_id"]) for it in dbs[db]} for db in DB_ORDER}


def augment_dbs_v58(dbs_base, m6_by_db):
    out = {}
    for db in DB_ORDER:
        items = []
        for it in dbs_base[db]:
            c = dict(it)
            if len(it["X_s2"]) > 0:
                c["X_s2"] = np.column_stack([it["X_s2"].astype(np.float64), m6_by_db[db][it["rec_id"]].astype(np.float64)])
            else:
                c["X_s2"] = np.zeros((0, it["X_s2"].shape[1] + 6), dtype=np.float64)
            items.append(c)
        out[db] = items
    return out


# Columns 24..29:
#  24: C_drift
#  25: log_tan_orth
#  26: tanh_d_qrs
#  27: tanh_d_stt
#  28: log_qrs_vs_nonqrs
#  29: log_dv_qrs_norm
CONFIGS = {
    "A0_B0_Baseline":   (F2_IDX,                             IDX_C3_V),
    "A2_HPhi_NS":       (F2_IDX + [24, 25],                  IDX_C3_V),
    "A3_HPhiW_NS_Only": (F2_IDX + [24, 25, 26, 27, 28],      IDX_C3_V),
    "A4_W_FV_Only":     (F2_IDX,                             IDX_C3_V + [26, 27, 28, 29]),
    "A5_Full_M3_Both":  (F2_IDX + [24, 25, 26, 27, 28],      IDX_C3_V + [26, 27, 28, 29]),
}
CFG_ORDER = list(CONFIGS.keys())
RULES = ("R1", "R3_B15", "R3_B20", "R3_B25", "R3_B20_0F1")


def fit_fold_v58(tr_items, mtype, cap_pat_pos=100):
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
        tN_15, tV_15 = v56.solve_r3_joint2d_patfa(sN, sV, y_cat, pi_cat, src_pat_hours, src_pat_totF, b_tot=1.5, b_0f=1e9, n_grid=85)
        tN_20, tV_20 = v56.solve_r3_joint2d_patfa(sN, sV, y_cat, pi_cat, src_pat_hours, src_pat_totF, b_tot=2.0, b_0f=1e9, n_grid=85)
        tN_25, tV_25 = v56.solve_r3_joint2d_patfa(sN, sV, y_cat, pi_cat, src_pat_hours, src_pat_totF, b_tot=2.5, b_0f=1e9, n_grid=85)
        tN_20_1, tV_20_1 = v56.solve_r3_joint2d_patfa(sN, sV, y_cat, pi_cat, src_pat_hours, src_pat_totF, b_tot=2.0, b_0f=1.0, n_grid=85)

        out[cfg] = {
            "fN": list(fN), "fV": list(fV),
            "scN": full_N[fN][0], "clfN": full_N[fN][1], "sigN": sig_N[fN],
            "scV": full_V[fV][0], "clfV": full_V[fV][1], "sigV": sig_V[fV],
            "R1": (float(tN_r1), float(tV_r1)),
            "R3_B15": (float(tN_15), float(tV_15)),
            "R3_B20": (float(tN_20), float(tV_20)),
            "R3_B25": (float(tN_25), float(tV_25)),
            "R3_B20_0F1": (float(tN_20_1), float(tV_20_1)),
        }
    return out


def paired_stats(a_dict, b_dict):
    keys = sorted(b_dict.keys())
    da = np.array([a_dict[k] for k in keys], dtype=np.float64)
    db = np.array([b_dict[k] for k in keys], dtype=np.float64)
    diff = da - db
    nz = diff[np.abs(diff) > 1e-12]
    w = int(np.sum(nz > 0)); l = int(np.sum(nz < 0)); t = len(keys) - w - l
    p_sign = float(binomtest(w, w + l, 0.5).pvalue) if (w + l) > 0 else 1.0
    try:
        p_wilc = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
    except Exception:
        p_wilc = 1.0
    return {
        "WTL": f"{w}W/{t}T/{l}L",
        "mean_diff": round(float(np.mean(diff)), 4),
        "p_wilcoxon": round(p_wilc, 5),
        "p_sign": round(p_sign, 5),
    }


def zero_f_tail_audit(per_unit):
    z = [(sid, u["faph"], u["fp_ns"], u["fp_v"], u["db"]) for sid, u in per_unit.items() if u["tot_F"] == 0]
    arr = np.array([x[1] for x in z], dtype=np.float64)
    z_sorted = sorted(z, key=lambda x: -x[1])
    return {
        "n_0F": len(z),
        "mean": round(float(np.mean(arr)), 3),
        "pct_zero": round(100.0 * float(np.mean(arr == 0.0)), 1),
        "pct_le_2": round(100.0 * float(np.mean(arr <= 2.0)), 1),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(np.max(arr)), 3),
        "top5_worst": [{"unit": x[0], "db": x[4], "faph": round(x[1], 2), "fp_ns": x[2], "fp_v": x[3]} for x in z_sorted[:5]],
    }


def main():
    t0 = time.time()
    reg = json.load(open(v53b.REG_PATH, "r", encoding="utf-8"))
    rec_to_sid = {(r["db"], r["rec_id"]): r["verified_subject_id"] for r in reg["records"]}

    dbs_raw = {db: v47.build_or_load_v47_db(db) for db in DB_ORDER}
    dbs_base = v49.clone_dbs_with_mapping(dbs_raw, rec_to_sid)
    m6_by_db = build_or_load_v58_exact_cache(dbs_base)
    dbs = augment_dbs_v58(dbs_base, m6_by_db)

    audit = {
        "Audit_Version": "v58_exact_beatwise_m3_and_ablation",
        "Date": "2026-11-23",
        "Scope": "6 source DBs (306 recs / 251 SubjectAware_251 units), BIT-EXACT beat-by-beat causal b_N_loc[t], strict 6-hospital LODO, SVM + LogR, Zero LTSTDB.",
        "Configs": {k: {"N": v[0], "V": v[1]} for k, v in CONFIGS.items()},
        "Models": {},
    }

    for mtype in ("SVM", "LogR"):
        print(f"\n==================== v58 BIT-EXACT M3 MECHANISTIC ABLATION ({mtype}) ====================", flush=True)
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
            fitted = fit_fold_v58(tr_items, mtype)
            fold_meta[tgt] = {
                cfg: {
                    **{rn: [round(x, 4) for x in fitted[cfg][rn]] for rn in RULES},
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

        # Exact B0 R1 parity check!
        ev_b0_r1 = v53b.eval_at_score_threshold(unit_data, "Smin_R1_A0_B0_Baseline", 0.0)
        p2_b0_r1 = v53b.per_patient_patf2_macro(unit_data, "Smin_R1_A0_B0_Baseline")
        if mtype == "SVM":
            assert ev_b0_r1["TP"] == 1099 and ev_b0_r1["FP_NS"] == 2058 and ev_b0_r1["FP_V"] == 1406, ev_b0_r1
        else:
            assert ev_b0_r1["TP"] == 1169 and ev_b0_r1["FP_NS"] == 2330 and ev_b0_r1["FP_V"] == 1437, ev_b0_r1
        print(f"  [B0 R1 Parity OK] TP={ev_b0_r1['TP']}, FP_NS={ev_b0_r1['FP_NS']}, FP_V={ev_b0_r1['FP_V']}, Rec={ev_b0_r1['MacroRec']}%, FAPH={ev_b0_r1['MeanFAPH_All']}/{ev_b0_r1['MeanFAPH_0F']}", flush=True)

        # Evaluate complete ladder + sensitivity rules
        ladder = [
            ("A0_B0_R1",           "A0_B0_Baseline",   "R1"),
            ("A1_B0_R3_B20",       "A0_B0_Baseline",   "R3_B20"),
            ("A2_HPhi_NS_R3_B20",  "A2_HPhi_NS",       "R3_B20"),
            ("A3_HPhiW_NS_R3_B20", "A3_HPhiW_NS_Only", "R3_B20"),
            ("A4_W_FV_Only_R3_B20","A4_W_FV_Only",     "R3_B20"),
            ("A5_Full_M3_R3_B20",  "A5_Full_M3_Both",  "R3_B20"),
            ("A5b_Full_M3_R3_B15", "A5_Full_M3_Both",  "R3_B15"),
            ("A5c_Full_M3_R3_B25", "A5_Full_M3_Both",  "R3_B25"),
            ("A5d_Full_M3_R3_0F1", "A5_Full_M3_Both",  "R3_B20_0F1"),
            ("A1b_B0_R3_B15",      "A0_B0_Baseline",   "R3_B15"),
            ("A1c_B0_R3_B25",      "A0_B0_Baseline",   "R3_B25"),
        ]

        ev_b0_r3 = v53b.eval_at_score_threshold(unit_data, "Smin_R3_B20_A0_B0_Baseline", 0.0)
        p2_b0_r3 = v53b.per_patient_patf2_macro(unit_data, "Smin_R3_B20_A0_B0_Baseline")

        res_ladder = {}
        print("\n  === 1. COMPLETE MECHANISTIC DECOMPOSITION LADDER (BIT-EXACT b_N_loc[t]) ===", flush=True)
        for label, cfg, rn in ladder:
            sk = f"Smin_{rn}_{cfg}"
            ev = v53b.eval_at_score_threshold(unit_data, sk, 0.0)
            wtl_r1 = v53b.compare_wtl(ev["per_unit"], ev_b0_r1["per_unit"])
            wtl_r3 = v53b.compare_wtl(ev["per_unit"], ev_b0_r3["per_unit"])
            all_y = np.concatenate([u["ys2"] for u in unit_data.values() if u["n_s2"] > 0])
            all_sc = np.concatenate([u[sk] for u in unit_data.values() if u["n_s2"] > 0])
            ap_s2 = float(average_precision_score((all_y == 2).astype(int), all_sc))
            p2 = v53b.per_patient_patf2_macro(unit_data, sk)
            st_p2_r1 = paired_stats(p2, p2_b0_r1)
            st_p2_r3 = paired_stats(p2, p2_b0_r3)
            rec_dict = {s: u["rec"] for s, u in ev["per_unit"].items() if u["tot_F"] > 0}
            rec_b0r1 = {s: u["rec"] for s, u in ev_b0_r1["per_unit"].items() if u["tot_F"] > 0}
            rec_b0r3 = {s: u["rec"] for s, u in ev_b0_r3["per_unit"].items() if u["tot_F"] > 0}
            st_t1_r1 = paired_stats(rec_dict, rec_b0r1)
            st_t1_r3 = paired_stats(rec_dict, rec_b0r3)
            z0f = zero_f_tail_audit(ev["per_unit"])

            per_db = {}
            for db in DB_ORDER:
                sub_u = {s: u for s, u in unit_data.items() if u["db"] == db}
                edb = v53b.eval_at_score_threshold(sub_u, sk, 0.0)
                ys_d = np.concatenate([u["ys2"] for u in sub_u.values() if u["n_s2"] > 0])
                sc_d = np.concatenate([u[sk] for u in sub_u.values() if u["n_s2"] > 0])
                ap_d = float(average_precision_score((ys_d == 2).astype(int), sc_d)) if np.any(ys_d == 2) else 0.0
                per_db[db] = {
                    "TP": edb["TP"], "FP_NS": edb["FP_NS"], "FP_V": edb["FP_V"],
                    "MacroRec": edb["MacroRec"], "MeanFAPH": edb["MeanFAPH_All"],
                    "MeanFAPH_0F": edb["MeanFAPH_0F"], "AP_s2": round(ap_d, 4),
                }

            sw_all = v53b.matched_budget_sweep(unit_data, sk, None, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], mode="all")
            sw_0f = v53b.matched_budget_sweep(unit_data, sk, None, [0.25, 0.5, 1.0, 1.5], mode="0f")

            res_ladder[label] = {
                "cfg": cfg, "rule": rn,
                "T1": {k: v for k, v in ev.items() if k != "per_unit"},
                "vs_B0_R1": wtl_r1, "vs_B0_R3": wtl_r3,
                "T1_Rec_stats_vs_B0R1": st_t1_r1, "T1_Rec_stats_vs_B0R3": st_t1_r3,
                "AP_s2": round(ap_s2, 4),
                "PatF2_MacroRec": round(100.0 * float(np.mean(list(p2.values()))), 2),
                "PatF2_stats_vs_B0R1": st_p2_r1, "PatF2_stats_vs_B0R3": st_p2_r3,
                "ZeroF_tail": z0f,
                "per_db_T1": per_db,
                "matched_Smin_AllFAPH": sw_all,
                "matched_Smin_0FFAPH": sw_0f,
            }
            print(
                f"    {label:20s} | TP={ev['TP']:4d} NS={ev['FP_NS']:4d} V={ev['FP_V']:4d} | "
                f"Rec={ev['MacroRec']:5.2f}% (vsR1:{wtl_r1['Fpos_WTL']} SD:{wtl_r1['StrictDom_Fpos']}) | "
                f"FAPH={ev['MeanFAPH_All']:.3f}/{ev['MeanFAPH_0F']:.3f} (<=2:{ev['Pct_le_2FAPH']}%, p95:{ev['p95_FAPH_All']}) | "
                f"AP={ap_s2:.4f} | P@F2={res_ladder[label]['PatF2_MacroRec']}%({st_p2_r1['WTL']}, pW={st_p2_r1['p_wilcoxon']:.4f})",
                flush=True,
            )

        # Pure m_N sweep under FIXED B0_R1 F:V gate for A0, A2, A3 (directly quantifies R_{Phi,w}^* - R_{current} on Branch F:N/S!)
        pure_mN = {}
        for cfg in ("A0_B0_Baseline", "A2_HPhi_NS", "A3_HPhiW_NS_Only"):
            pure_mN[cfg] = {
                "AllFAPH": v53b.matched_budget_sweep(unit_data, f"mN_R3_B20_{cfg}", "mV_R1_A0_B0_Baseline", [1.0, 1.5, 2.0, 2.5], mode="all"),
                "0FFAPH": v53b.matched_budget_sweep(unit_data, f"mN_R3_B20_{cfg}", "mV_R1_A0_B0_Baseline", [0.5, 1.0, 1.5], mode="0f"),
            }

        print("\n  === 2. MATCHED S_min CURVE (AllFAPH): A0_B0_R1 vs A1..A5 ===", flush=True)
        for B in ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]:
            line = f"    @All<={B:3s}: "
            for lbl in ("A0_B0_R1", "A1_B0_R3_B20", "A2_HPhi_NS_R3_B20", "A3_HPhiW_NS_R3_B20", "A4_W_FV_Only_R3_B20", "A5_Full_M3_R3_B20"):
                r = res_ladder[lbl]["matched_Smin_AllFAPH"][B]
                line += f"{lbl[:2]}={r['MacroRec']:5.2f}%(TP={r['TP']:4d},0F={r['MeanFAPH_0F']:.2f},V={r['FP_V']:4d}) | "
            print(line, flush=True)

        print("\n  === 3. MATCHED S_min CURVE (True-0F FAPH): A0_B0_R1 vs A1..A5 ===", flush=True)
        for B in ["0.25", "0.5", "1.0", "1.5"]:
            line = f"    @0F <={B:4s}: "
            for lbl in ("A0_B0_R1", "A1_B0_R3_B20", "A2_HPhi_NS_R3_B20", "A3_HPhiW_NS_R3_B20", "A4_W_FV_Only_R3_B20", "A5_Full_M3_R3_B20"):
                r = res_ladder[lbl]["matched_Smin_0FFAPH"][B]
                line += f"{lbl[:2]}={r['MacroRec']:5.2f}%(TP={r['TP']:4d},All={r['MeanFAPH_All']:.2f},V={r['FP_V']:4d}) | "
            print(line, flush=True)

        print("\n  === 4. PER-DB BREAKDOWN: A0_B0_R1 vs A1_B0_R3 vs A5_Full_M3_R3_B20 ===", flush=True)
        for db in DB_ORDER:
            line = f"    {db:8s}: "
            for lbl in ("A0_B0_R1", "A1_B0_R3_B20", "A3_HPhiW_NS_R3_B20", "A5_Full_M3_R3_B20", "A5d_Full_M3_R3_0F1"):
                r = res_ladder[lbl]["per_db_T1"][db]
                line += f"{lbl[:3]}[TP={r['TP']:3d},NS={r['FP_NS']:4d},V={r['FP_V']:3d},R={r['MacroRec']:4.1f}%,F={r['MeanFAPH']:5.2f},AP={r['AP_s2']:.3f}] | "
            print(line, flush=True)

        # Helped / Harmed F+ unit details for A5 vs A0_B0_R1
        ev_a5 = v53b.eval_at_score_threshold(unit_data, "Smin_R3_B20_A5_Full_M3_Both", 0.0)
        hh = []
        for sid in sorted(ev_b0_r1["per_unit"].keys()):
            b = ev_b0_r1["per_unit"][sid]; a = ev_a5["per_unit"][sid]
            if b["tot_F"] == 0:
                continue
            if abs(a["rec"] - b["rec"]) > 1e-12 or abs(a["faph"] - b["faph"]) > 0.5:
                hh.append({
                    "unit": sid, "db": b["db"], "tot_F": b["tot_F"],
                    "B0_rec_%": round(100 * b["rec"], 1), "A5_rec_%": round(100 * a["rec"], 1),
                    "dRec_pp": round(100 * (a["rec"] - b["rec"]), 1),
                    "B0_faph": round(b["faph"], 2), "A5_faph": round(a["faph"], 2),
                    "B0_NS_V": f"{b['fp_ns']}/{b['fp_v']}", "A5_NS_V": f"{a['fp_ns']}/{a['fp_v']}",
                })
        hh.sort(key=lambda x: -x["dRec_pp"])

        print(f"\n  === 5. ZERO-F TAIL AUDIT: A0_B0_R1 vs A5_Full_M3_R3_B20 vs A5d_0F1 ===", flush=True)
        for lbl in ("A0_B0_R1", "A1_B0_R3_B20", "A5_Full_M3_R3_B20", "A5d_Full_M3_R3_0F1"):
            zt = res_ladder[lbl]["ZeroF_tail"]
            print(f"    {lbl:20s}: mean={zt['mean']:.3f}, %==0:{zt['pct_zero']}%, %<=2:{zt['pct_le_2']}%, p90={zt['p90']}, p95={zt['p95']}, p99={zt['p99']}, max={zt['max']}", flush=True)

        audit["Models"][mtype] = {
            "fold_meta": fold_meta,
            "ladder": res_ladder,
            "pure_mN_fixed_B0_FV": pure_mN,
            "helped_harmed_Fpos_A5_vs_B0R1": hh,
        }

    jpath = os.path.join(AUD_DIR, "V58_EXACT_BEATWISE_M3_ABLATION_20261123.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[v58] Saved full JSON -> {jpath} (total {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
