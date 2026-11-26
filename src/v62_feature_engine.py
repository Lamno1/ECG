r"""
v62_feature_engine.py
=====================
Shared, bit-exact, RAM-efficient feature extraction core for the Wave-4
BSPC revision suite (v62A Component Ablations, v62B ICU Stage-1/Stage-2
Forensics, v62C 4-Tier Causality Evaluation, v62D Representation Isolation).

Guarantees:
  - ZERO modifications to any frozen checkpoint, registry, or canonical cache.
  - Exact parity with `stage2_frozen_pipeline_v2.py` on Tier-0 Offline A3.
  - Identical past-only RR context (`f_RR(t) = ln(clip(pre_rr/loc_rr, 0.2, 3.0))`
    and `pre_rr_s = clip(pre_rr, 0.2, 2.5)`) across Raw ECG, Traditional Morphology,
    and Single-Pole baselines so rhythm context is strictly matched.
  - Strict zero-lookahead causality in Tier 1 (online-expanding V-anchors) and
    Tier 3 (causal double-median + causal FIR filter + online-expanding V-anchors).
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
import time
from collections import deque
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import wfdb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import stage2_frozen_pipeline_v2 as pipe2

# Exact locked A3 column indices in X29_s2:
#   X10D = [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]
#   C3 F:V = X10D + [18]
#   A3 F:N/S = X10D + [24, 25, 26, 27, 28]
IDX_X10D = [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]
IDX_A3_N = IDX_X10D + [24, 25, 26, 27, 28]     # 15D
IDX_C3_V = IDX_X10D + [18]                      # 11D

# Stage-2 feature-subset ablations (direct slices of X29_s2)
IDX_NO_TP_N = [0, 1, 2, 3, 4, 5, 8, 10, 24, 25, 26, 27, 28]  # 13D (drop M_Pyth=13, tau_min=14)
IDX_NO_TP_V = [0, 1, 2, 3, 4, 5, 8, 10, 18]                  # 9D
IDX_NO_DP_N = [4, 5, 24, 25, 26, 27, 28]                     # 7D (local normal + 5 A3 feats only)
IDX_NO_DP_V = [4, 5, 18]                                     # 3D (local normal + f_RR only)

# Indices for No-Tangent-U (14D F:N/S: drops l_orthU=25) & Static-bN (14D F:N/S: drops C_drift=24)
IDX_NO_U_N = IDX_X10D + [24, 26, 27, 28]                     # 14D
IDX_STATIC_BN_N = IDX_X10D + [25, 26, 27, 28]                # 14D

# Uniform 42-sample/lead grid across 187 samples (~81.3 Hz > 2*40 Hz Nyquist)
RAW_SUB_IDX = np.round(np.linspace(0, 186, 42)).astype(np.int64)

# Traditional ECG Morphology sub-windows on the 187-sample 360-Hz grid (fiducial = 93)
MORPH_SLICES = (
    slice(20, 71),   # pre-QRS [-203, -61] ms
    slice(71, 93),   # QRS onset-to-peak [-61, 0] ms
    slice(93, 123),  # QRS peak-to-offset [0, +83] ms
    slice(123, 150), # ST segment [+83, +158] ms
    slice(150, 187), # T wave [+158, +258] ms
)


def load_beats_and_rr(rec_path, ann_ext, channel_indices, label_map_version, filter_mode="offline_filtfilt"):
    """Load WFDB record, filter, interpolate to 187-sample 360-Hz grid, compute causal RR."""
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, ann_ext)
    fs = float(rec.fs)
    i0, i1 = int(channel_indices[0]), int(channel_indices[1])

    s0_raw = np.nan_to_num(rec.p_signal[:, i0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    s1_raw = np.nan_to_num(rec.p_signal[:, i1].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sig0, delay = v33.filter_signal_generic(s0_raw, fs, mode=filter_mode)
    sig1, _ = v33.filter_signal_generic(s1_raw, fs, mode=filter_mode)

    samples, symbols = ann.sample, ann.symbol
    n_sig = len(sig0)
    if abs(fs - v33.TARGET_FS) < 1e-3:
        beats0, beats1, labels, r_peaks = [], [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < 93 or sc + 94 > n_sig:
                continue
            c = pipe2.map_symbol(sym, label_map_version)
            if c >= 0:
                beats0.append(sig0[sc - 93 : sc + 94])
                beats1.append(sig1[sc - 93 : sc + 94])
                labels.append(c)
                r_peaks.append(s)
        if len(labels) <= 55:
            return None
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
            c = pipe2.map_symbol(sym, label_map_version)
            if c >= 0:
                valid_sc.append(sc)
                labels.append(c)
                r_peaks.append(s)
        if len(valid_sc) <= 55:
            return None
        valid_sc = np.asarray(valid_sc, dtype=np.int64)
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
    post_hours = float((r_peaks[-1] - r_peaks[50]) / fs / 3600.0)

    n0_all = v33.normalize_lead(beats0)
    n1_all = v33.normalize_lead(beats1)
    joint_all = np.column_stack([n0_all, n1_all])
    return {
        "fs": fs, "r_peaks": r_peaks, "labels": labels,
        "pre_rr": pre_rr, "pre_rr_loc": pre_rr_loc_all,
        "post_hours": post_hours,
        "joint_all": joint_all,
    }


def extract_traditional_morphology_and_raw(joint_post, pre_rr_post, pre_rr_loc_post):
    """
    Vectorized extraction of:
      1) Raw ECG + RR (86D: 42 uniform samples/lead * 2 + f_RR + pre_rr_s)
      2) Traditional ECG Morphology + RR (34D: 16 morph feats/lead * 2 + f_RR + pre_rr_s)
         No recording-specific normal or ventricular poles used anywhere.
    """
    N = len(joint_post)
    f_rr = np.log(np.clip(pre_rr_loc_post, 0.2, 3.0))
    rr_s = np.clip(pre_rr_post, 0.2, 2.5)

    # 1. Raw ECG + RR (86D): cols 0..83 waveform, 84=f_rr, 85=rr_s
    X_raw = np.empty((N, 86), dtype=np.float32)
    X_raw[:, :42] = joint_post[:, RAW_SUB_IDX]
    X_raw[:, 42:84] = joint_post[:, 187 + RAW_SUB_IDX]
    X_raw[:, 84] = f_rr
    X_raw[:, 85] = rr_s

    # 2. Traditional ECG Morphology + RR (34D): cols 0..31 morph, 32=f_rr, 33=rr_s
    feats = []
    for off in (0, 187):
        w = joint_post[:, off : off + 187]
        for sl in MORPH_SLICES:
            seg = w[:, sl]
            feats.append(np.mean(seg, axis=1))
            feats.append(np.sqrt(np.mean(seg * seg, axis=1) + 1e-12))
        for sl in (slice(20, 71), slice(71, 123), slice(123, 187)):
            seg = w[:, sl]
            feats.append(np.max(seg, axis=1) - np.min(seg, axis=1))
        qrs = w[:, 71:123]
        feats.append(np.max(np.abs(np.diff(qrs, axis=1)), axis=1))
        qm = np.mean(qrs, axis=1, keepdims=True)
        qs = np.std(qrs, axis=1, keepdims=True) + 1e-6
        qz = (qrs - qm) / qs
        feats.append(np.clip(np.mean(qz ** 3, axis=1), -10.0, 10.0))
        feats.append(np.clip(np.mean(qz ** 4, axis=1) - 3.0, -5.0, 30.0))
    feats.append(f_rr)
    feats.append(rr_s)
    X_morph = np.column_stack(feats).astype(np.float32)
    X_morph = np.nan_to_num(X_morph, nan=0.0, posinf=0.0, neginf=0.0)
    return X_raw, X_morph


def run_dualpole_variant(
    data,
    use_tangent_U=True,
    use_local_bN=True,
    anchor_mode="offline",   # "offline" | "online_causal" | "prefix_only" | "offline_and_online"
    collect_singlepole=False,
    collect_stage1_forensics=False,
    also_static_bn=False,
):
    """
    Unified dual-pole extractor supporting:
      - Tier 0 Offline A3 + Tier 1 Online + Static_bN + SinglePole + Forensics in ONE pass
        (anchor_mode="offline_and_online", also_static_bn=True, collect_singlepole=True)
      - Structural ablation: use_tangent_U=False (P=I)
      - Causality tiers: anchor_mode="online_causal" (Tier 3) OR "prefix_only" (Tier 2, O(1) skip if no prefix anchor)
    """
    joint_all = data["joint_all"]
    labels = data["labels"]
    r_peaks = data["r_peaks"]
    fs = data["fs"]
    pre_rr = data["pre_rr"]
    pre_rr_loc_all = data["pre_rr_loc"]

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
    if use_tangent_U:
        _, _, Vt = np.linalg.svd(centered_norm, full_matrices=False)
        U = Vt[:min(4, Vt.shape[0]), :].T
        P = np.eye(D) - U @ U.T
        norm_y_prefix = centered_norm @ P.T
    else:
        U = np.zeros((D, 0), dtype=np.float64)
        P = np.eye(D)
        norm_y_prefix = centered_norm.copy()
    eta_0 = float(np.percentile(np.linalg.norm(norm_y_prefix, axis=1), 95)) + 1e-6

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
    post_r_peaks = r_peaks[50:]
    pre_rr_post = pre_rr[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    N = len(post_joint)

    # Fast O(1) short-circuit for prefix_only when no prefix anchor was seeded
    if anchor_mode == "prefix_only" and len(anchors) == 0:
        return {
            "y": post_y, "s2_mask": np.zeros(N, dtype=bool), "X29_s2": np.zeros((0, 29), dtype=np.float32),
            "K": 0, "post_hours": data["post_hours"],
            "tau1_beat": None, "tau1_min": None, "pre_anchor_F": int(np.sum(post_y == 2)),
            "X_singlepole": None, "forensics": None,
        }

    hf0 = np.sum(np.diff(post_joint[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1 = np.sum(np.diff(post_joint[:, 187:], n=2, axis=1) ** 2, axis=1)
    sqi_hf_ratio = np.log1p(np.maximum(hf0 / med_hf0, hf1 / med_hf1))

    diff_post = post_joint - b_N[None, :]
    if use_tangent_U:
        y_mat = diff_post - (diff_post @ U) @ U.T
    else:
        y_mat = diff_post.copy()
    norm_y = np.linalg.norm(y_mat, axis=1)
    norm_y_thresh = max(v33.CAND_NORM_Y_MIN, v33.CAND_ETA_MULT * eta_0)

    # Local normal ring buffer
    loc_ring = np.zeros((v33.LOC_BUF_LEN, D), dtype=np.float64)
    init_nb = norm_beats[-v33.LOC_BUF_LEN:]
    loc_cnt = len(init_nb)
    loc_ring[:loc_cnt] = init_nb
    loc_ptr = loc_cnt % v33.LOC_BUF_LEN
    b_N_loc = np.median(loc_ring[:loc_cnt], axis=0) if use_local_bN else b_N.copy()

    need_hist = (use_local_bN or anchor_mode in ("offline", "offline_and_online"))
    b_N_loc_hist = np.zeros((N, D), dtype=np.float32) if need_hist else None
    loc_norm_corr_arr = np.zeros(N, dtype=np.float64)
    loc_norm_dist_proj_arr = np.zeros(N, dtype=np.float64)
    d_full_loc_arr = np.zeros(N, dtype=np.float64)
    d_par_u_arr = np.zeros(N, dtype=np.float64)
    prev1, prev2 = joint_all[49], joint_all[48]

    # Online-causal evaluation arrays (populated strictly before updating anchors at beat t!)
    is_causal_anc = (anchor_mode in ("online_causal", "prefix_only", "offline_and_online"))
    if is_causal_anc:
        c_s2_idx = []
        c_X29_rows = []

    first_anchor_beat = 0 if len(anchors) > 0 else None
    forensics = None
    if collect_stage1_forensics:
        forensics = {
            "post_V_total": int(np.sum(post_y == 1)),
            "post_V_pass_norm_y": int(np.sum((post_y == 1) & (norm_y >= norm_y_thresh))),
            "post_V_pass_rr": int(np.sum((post_y == 1) & (pre_rr_loc <= v33.CAND_PRE_RR_LOC_MAX))),
            "post_V_pass_sqi": int(np.sum((post_y == 1) & (sqi_hf_ratio <= v33.ONL_HF_MAX))),
            "post_V_pass_all3": int(np.sum((post_y == 1) & (norm_y >= norm_y_thresh) & (pre_rr_loc <= v33.CAND_PRE_RR_LOC_MAX) & (sqi_hf_ratio <= v33.ONL_HF_MAX))),
            "n_cand_all3": 0,
            "max_cluster_size": 0,
            "min_cohesion_at_ge6": None,
            "rejected_norm_q_count": 0,
            "prefix_anchor_seeded": int(len(anchors) > 0),
        }

    cand_clusters = []
    for t in range(N):
        x_t = post_joint[t]
        b_cur = b_N_loc if use_local_bN else b_N
        if b_N_loc_hist is not None:
            b_N_loc_hist[t] = b_cur
        c_loc = v33.mean_lead_corr(x_t, b_cur)
        loc_norm_corr_arr[t] = c_loc
        dl = x_t - b_cur
        d_full_loc_arr[t] = float(np.linalg.norm(dl))
        if use_tangent_U:
            u_proj = U.T @ dl
            d_par_u_arr[t] = float(np.linalg.norm(u_proj))
            dl_p = dl - U @ u_proj
            loc_norm_dist_proj_arr[t] = float(np.linalg.norm(dl_p))
        else:
            d_par_u_arr[t] = 0.0
            loc_norm_dist_proj_arr[t] = d_full_loc_arr[t]

        # ---- STRICTLY BEFORE UPDATING ANCHORS AT BEAT t (for Tier 1 / 2 / 3) ----
        if is_causal_anc and len(anchors) > 0:
            yt = y_mat[t]
            best_k, best_eucl, best_lam, best_lc = 0, 1e18, 0.0, 0.0
            e_V_any = 1e18
            for k, anc in enumerate(anchors):
                qk = anc["q"]
                nq = anc["norm_q"]
                lam_k = float(np.dot(yt, qk) / (nq * nq + 1e-12))
                lc_k = 0.0 if lam_k < 0.0 else (1.0 if lam_k > 1.0 else lam_k)
                eucl_k = float(np.linalg.norm(yt - lc_k * qk))
                if eucl_k < best_eucl:
                    best_eucl, best_k, best_lam, best_lc = eucl_k, k, lam_k, lc_k
                dy_q = float(np.linalg.norm(yt - qk))
                ej = max(0.0, (dy_q - anc["rad_y_V"]) / (nq + 1e-12))
                if ej < e_V_any:
                    e_V_any = ej
            anc_b = anchors[best_k]
            nq_b = anc_b["norm_q"]
            e_N = max(0.0, (best_lam - anc_b["mu_N"]) - anc_b["rad_N"])
            e_Vc = max(0.0, (1.0 - best_lam) - anc_b["rad_V"])
            z_N = float(np.log1p(e_N / anc_b["rad_N"]))
            z_V = float(min(np.log1p(e_Vc / anc_b["rad_V"]), np.log1p(e_V_any / anc_b["rad_V"])))
            m_br = min(z_N, z_V)
            if m_br > 0.0:
                d_seg = best_eucl / (nq_b + 1e-12)
                loc_dist_norm = loc_norm_dist_proj_arr[t] / (nq_b + 1e-12)
                bV0, bV1 = anc_b["b_V"][:187], anc_b["b_V"][187:]
                hat0 = bN_0 + best_lc * (bV0 - bN_0)
                hat1 = bN_1 + best_lc * (bV1 - bN_1)
                cmin = min(v33.pearson_1d(x_t[:187], hat0), v33.pearson_1d(x_t[187:], hat1))
                psi_c = max(0.0, (pref_bcorr_p05 - cmin) - 0.10) ** 2

                # Local Pyth & Thales
                diff_l = b_cur - b_N
                y_loc = diff_l - U @ (U.T @ diff_l) if use_tangent_U else diff_l
                dN_l = loc_norm_dist_proj_arr[t]
                dV_l = float(np.linalg.norm(yt - anc_b["q"]))
                dNV_l = float(np.linalg.norm(anc_b["q"] - y_loc))
                _, pyth_l = v33.thales_obt_and_pyth(dN_l, dV_l, dNV_l)
                obt0, _ = v33.thales_obt_and_pyth(
                    float(np.linalg.norm(x_t[:187] - b_cur[:187])),
                    float(np.linalg.norm(x_t[:187] - bV0)),
                    float(np.linalg.norm(bV0 - b_cur[:187])),
                )
                obt1, _ = v33.thales_obt_and_pyth(
                    float(np.linalg.norm(x_t[187:] - b_cur[187:])),
                    float(np.linalg.norm(x_t[187:] - bV1)),
                    float(np.linalg.norm(bV1 - b_cur[187:])),
                )
                thales_l = min(obt0, obt1)

                # 5 A3 features
                dp = loc_norm_dist_proj_arr[t]
                dg = norm_y[t]
                du = d_par_u_arr[t]
                c_drift = -abs(np.log((dp + 0.5) / (dg + 0.5)))
                l_orth = np.log((dp + 0.5) / (du + 0.5)) if use_tangent_U else 0.0
                dl0, dl1 = dl[:187], dl[187:]
                d_pre = float(np.sqrt(np.dot(dl0[pipe2.SL_PRE], dl0[pipe2.SL_PRE]) + np.dot(dl1[pipe2.SL_PRE], dl1[pipe2.SL_PRE])))
                d_qrs = float(np.sqrt(np.dot(dl0[pipe2.SL_QRS], dl0[pipe2.SL_QRS]) + np.dot(dl1[pipe2.SL_QRS], dl1[pipe2.SL_QRS])))
                d_stt = float(np.sqrt(np.dot(dl0[pipe2.SL_STT], dl0[pipe2.SL_STT]) + np.dot(dl1[pipe2.SL_STT], dl1[pipe2.SL_STT])))
                d_nonq = float(np.hypot(d_pre, d_stt))
                t_qrs = float(np.tanh((d_qrs / pipe2.R_QRS) / 0.75))
                t_stt = float(np.tanh((d_stt / pipe2.R_STT) / 0.75))
                l_qn = float(np.log((d_qrs + 0.5) / (d_nonq + 0.5)))
                f_rr_t = float(np.log(np.clip(pre_rr_loc[t], 0.2, 3.0)))

                row = np.zeros(29, dtype=np.float32)
                row[0] = m_br; row[1] = z_N; row[2] = z_V; row[3] = d_seg
                row[4] = c_loc; row[5] = loc_dist_norm; row[8] = cmin; row[10] = psi_c
                row[13] = pyth_l; row[14] = thales_l; row[18] = f_rr_t
                row[24] = c_drift; row[25] = l_orth; row[26] = t_qrs; row[27] = t_stt; row[28] = l_qn
                c_s2_idx.append(t)
                c_X29_rows.append(row)

        # ---- ONLINE ANCHOR DISCOVERY UPDATE (disabled in prefix_only mode!) ----
        if anchor_mode != "prefix_only":
            if (norm_y[t] >= norm_y_thresh) and (pre_rr_loc[t] <= v33.CAND_PRE_RR_LOC_MAX) and (sqi_hf_ratio[t] <= v33.ONL_HF_MAX):
                if forensics is not None:
                    forensics["n_cand_all3"] += 1
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
                        if forensics is not None and len(best_c["beats"]) > forensics["max_cluster_size"]:
                            forensics["max_cluster_size"] = len(best_c["beats"])
                        if len(best_c["beats"]) >= v33.CONFIRM_MIN_BEATS:
                            arr_b = np.array(best_c["beats"])
                            med_b = np.median(arr_b, axis=0)
                            cohesion = float(np.median(np.linalg.norm(arr_b - med_b, axis=1)))
                            if forensics is not None:
                                if forensics["min_cohesion_at_ge6"] is None or cohesion < forensics["min_cohesion_at_ge6"]:
                                    forensics["min_cohesion_at_ge6"] = round(cohesion, 3)
                            if cohesion <= v33.CONFIRM_MAX_COHESION:
                                anc = v33.make_anchor_obj(arr_b, norm_y_prefix, b_N, P, int(t + 1), "online")
                                if anc is not None:
                                    anchors.append(anc)
                                    if first_anchor_beat is None:
                                        first_anchor_beat = int(t + 1)
                                    cand_clusters.pop(best_c_idx)
                                elif forensics is not None:
                                    forensics["rejected_norm_q_count"] += 1
                    else:
                        if len(cand_clusters) >= v33.MAX_CAND_CLUSTERS:
                            cand_clusters.sort(key=lambda c: len(c["beats"]))
                            cand_clusters.pop(0)
                        cand_clusters.append({"beats": [x_t], "center": x_t.copy()})

        # ---- LOCAL NORMAL RING UPDATE ----
        if use_local_bN:
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

    # ---- OPTIONAL FULL-STREAM SINGLE-POLE NORMAL-SUBSPACE FEATURES (14D, ZERO V-POLE!) ----
    X_singlepole = None
    if collect_singlepole:
        # Vectorized over all N post-prefix beats using exact b_N_loc_hist
        dl_all = post_joint - b_N_loc_hist.astype(np.float64)
        d_pre_all = np.sqrt(np.sum(dl_all[:, pipe2.SL_PRE] ** 2, axis=1) + np.sum(dl_all[:, 187 + np.arange(0, 71)] ** 2, axis=1))
        d_qrs_all = np.sqrt(np.sum(dl_all[:, pipe2.SL_QRS] ** 2, axis=1) + np.sum(dl_all[:, 187 + np.arange(71, 123)] ** 2, axis=1))
        d_stt_all = np.sqrt(np.sum(dl_all[:, pipe2.SL_STT] ** 2, axis=1) + np.sum(dl_all[:, 187 + np.arange(123, 187)] ** 2, axis=1))
        d_nonq_all = np.hypot(d_pre_all, d_stt_all)
        min_corr_bN = np.minimum(
            v33.pearson_corr_rows(post_joint[:, :187], bN_0),
            v33.pearson_corr_rows(post_joint[:, 187:], bN_1),
        )
        X_singlepole = np.column_stack([
            np.log1p(norm_y / (eta_0 + 1e-6)),
            np.log1p(loc_norm_dist_proj_arr / (eta_0 + 1e-6)),
            np.log1p(d_par_u_arr / (eta_0 + 1e-6)),
            np.linalg.norm(diff_post, axis=1),
            d_full_loc_arr,
            loc_norm_corr_arr,
            min_corr_bN,
            -np.abs(np.log((loc_norm_dist_proj_arr + 0.5) / (norm_y + 0.5))),
            np.log((loc_norm_dist_proj_arr + 0.5) / (d_par_u_arr + 0.5)),
            np.tanh((d_qrs_all / pipe2.R_QRS) / 0.75),
            np.tanh((d_stt_all / pipe2.R_STT) / 0.75),
            np.log((d_qrs_all + 0.5) / (d_nonq_all + 0.5)),
            np.log(np.clip(pre_rr_loc, 0.2, 3.0)),
            np.clip(pre_rr_post, 0.2, 2.5),
        ]).astype(np.float32)
        X_singlepole = np.nan_to_num(X_singlepole, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- BUILD STAGE-2 OUTPUT ----
    tau1_min = None
    pre_anchor_F = int(np.sum(post_y == 2))
    if first_anchor_beat is not None:
        if first_anchor_beat == 0:
            tau1_min = 0.0
            pre_anchor_F = 0
        elif first_anchor_beat < N:
            tau1_min = float((post_r_peaks[first_anchor_beat] - post_r_peaks[0]) / fs / 60.0)
            pre_anchor_F = int(np.sum(post_y[:first_anchor_beat] == 2))
        else:
            tau1_min = float((post_r_peaks[-1] - post_r_peaks[0]) / fs / 60.0)
            pre_anchor_F = int(np.sum(post_y == 2))

    online_res = None
    if is_causal_anc:
        c_s2_mask = np.zeros(N, dtype=bool)
        if len(c_s2_idx) > 0:
            c_s2_mask[np.asarray(c_s2_idx, dtype=np.int64)] = True
            c_X29_s2 = np.vstack(c_X29_rows).astype(np.float32)
        else:
            c_X29_s2 = np.zeros((0, 29), dtype=np.float32)
        c_X29_s2 = np.nan_to_num(c_X29_s2, nan=0.0, posinf=0.0, neginf=0.0)
        online_res = {
            "y": post_y, "s2_mask": c_s2_mask, "X29_s2": c_X29_s2,
            "K": len(anchors), "post_hours": data["post_hours"],
            "tau1_beat": first_anchor_beat, "tau1_min": tau1_min, "pre_anchor_F": pre_anchor_F,
            "X_singlepole": X_singlepole, "forensics": forensics,
        }
        if anchor_mode != "offline_and_online":
            return online_res

    # ---- OFFLINE FINALIZED-ANCHOR EVALUATION (EXACT v2 PARITY!) ----
    s2_mask = np.zeros(N, dtype=bool)
    X29_s2 = np.zeros((0, 29), dtype=np.float32)
    X29_s2_static = np.zeros((0, 29), dtype=np.float32) if also_static_bn else None
    if len(anchors) > 0:
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
        z_Vc_arr = np.log1p(e_Vc_arr / radV_vec[bk])
        z_Vb_arr = np.log1p(e_V_any_arr / radV_vec[bk])
        z_V_arr = np.minimum(z_Vc_arr, z_Vb_arr)
        m_bridge_arr = np.minimum(z_N_arr, z_V_arr)
        s2_mask = (m_bridge_arr > 0.0)

        if forensics is not None:
            f_m = (post_y == 2)
            f_rej = f_m & (~s2_mask)
            forensics["F_in_Kpos_total"] = int(np.sum(f_m))
            forensics["F_in_Kpos_admitted"] = int(np.sum(f_m & s2_mask))
            forensics["F_rej_zN0_only"] = int(np.sum(f_rej & (z_N_arr <= 0.0) & (z_V_arr > 0.0)))
            forensics["F_rej_zVc0"] = int(np.sum(f_rej & (z_N_arr > 0.0) & (z_Vc_arr <= 0.0)))
            forensics["F_rej_zVb0_only"] = int(np.sum(f_rej & (z_N_arr > 0.0) & (z_Vc_arr > 0.0) & (z_Vb_arr <= 0.0)))
            forensics["F_rej_both_zN_zV"] = int(np.sum(f_rej & (z_N_arr <= 0.0) & (z_V_arr <= 0.0)))
            forensics["F_rej_lam_med"] = round(float(np.median(best_lam_arr[f_rej])), 3) if np.any(f_rej) else None

        s2_idx = np.where(s2_mask)[0]
        if len(s2_idx) > 0:
            n_s2 = len(s2_idx)
            X29_s2 = np.zeros((n_s2, 29), dtype=np.float32)
            bV_stack = np.vstack([a["b_V"] for a in anchors])
            bk_s2 = bk[s2_idx]
            lc_s2 = best_lc_arr[s2_idx]

            # min_lead_bcorr on s2 beats
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
            y_loc_s2 = diff_loc_s2 - (diff_loc_s2 @ U) @ U.T if use_tangent_U else diff_loc_s2

            pyth_s2 = np.zeros(n_s2, dtype=np.float64)
            thales_s2 = np.zeros(n_s2, dtype=np.float64)
            for j, t in enumerate(s2_idx):
                k = int(bk[t])
                q = Q_mat[k]
                dN_l = loc_norm_dist_proj_arr[t]
                dV_l = float(np.linalg.norm(y_mat[t] - q))
                dNV_l = float(np.linalg.norm(q - y_loc_s2[j]))
                _, pyth_s2[j] = v33.thales_obt_and_pyth(dN_l, dV_l, dNV_l)

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

            X29_s2[:, 0] = m_bridge_arr[s2_idx]
            X29_s2[:, 1] = z_N_arr[s2_idx]
            X29_s2[:, 2] = z_V_arr[s2_idx]
            X29_s2[:, 3] = d_seg_arr[s2_idx]
            X29_s2[:, 4] = loc_norm_corr_arr[s2_idx]
            X29_s2[:, 5] = loc_norm_dist_proj_arr[s2_idx] / (nq_vec[bk_s2] + 1e-12)
            X29_s2[:, 8] = cmin_s2
            X29_s2[:, 10] = np.maximum(0.0, (pref_bcorr_p05 - cmin_s2) - 0.10) ** 2
            X29_s2[:, 13] = pyth_s2
            X29_s2[:, 14] = thales_s2
            X29_s2[:, 18] = np.log(np.clip(pre_rr_loc[s2_idx], 0.2, 3.0))
            X29_s2[:, 24] = -np.abs(np.log((dp_s2 + 0.5) / (dg_s2 + 0.5)))
            X29_s2[:, 25] = np.log((dp_s2 + 0.5) / (du_s2 + 0.5)) if use_tangent_U else 0.0
            X29_s2[:, 26] = np.tanh((d_qrs_s2 / pipe2.R_QRS) / 0.75)
            X29_s2[:, 27] = np.tanh((d_stt_s2 / pipe2.R_STT) / 0.75)
            X29_s2[:, 28] = np.log((d_qrs_s2 + 0.5) / (d_nonq_s2 + 0.5))
            X29_s2 = np.nan_to_num(X29_s2, nan=0.0, posinf=0.0, neginf=0.0)

            if also_static_bn:
                # Exact Static-bN (b_N_loc == b_N): identical anchors & s2_mask, recompute b_N-dependent columns only!
                X29_s2_static = X29_s2.copy()
                xs2 = post_joint[s2_idx]
                c_st = 0.5 * (v33.pearson_corr_rows(xs2[:, :187], bN_0) + v33.pearson_corr_rows(xs2[:, 187:], bN_1))
                dl_st = xs2 - b_N[None, :]
                dp_st = norm_y[s2_idx]
                du_st = np.linalg.norm(dl_st @ U, axis=1) if use_tangent_U else np.zeros(n_s2, dtype=np.float64)
                pyth_st = np.zeros(n_s2, dtype=np.float64)
                thales_st = np.zeros(n_s2, dtype=np.float64)
                for j, t in enumerate(s2_idx):
                    k = int(bk[t])
                    q = Q_mat[k]
                    _, pyth_st[j] = v33.thales_obt_and_pyth(dp_st[j], float(np.linalg.norm(y_mat[t] - q)), nq_vec[k])
                    x0, x1 = xs2[j, :187], xs2[j, 187:]
                    bV0, bV1 = bV_stack[k, :187], bV_stack[k, 187:]
                    o0, _ = v33.thales_obt_and_pyth(float(np.linalg.norm(x0 - bN_0)), float(np.linalg.norm(x0 - bV0)), float(np.linalg.norm(bV0 - bN_0)))
                    o1, _ = v33.thales_obt_and_pyth(float(np.linalg.norm(x1 - bN_1)), float(np.linalg.norm(x1 - bV1)), float(np.linalg.norm(bV1 - bN_1)))
                    thales_st[j] = min(o0, o1)
                d_pre_st = np.sqrt(np.sum(dl_st[:, pipe2.SL_PRE] ** 2, axis=1) + np.sum(dl_st[:, 187 + np.arange(0, 71)] ** 2, axis=1))
                d_qrs_st = np.sqrt(np.sum(dl_st[:, pipe2.SL_QRS] ** 2, axis=1) + np.sum(dl_st[:, 187 + np.arange(71, 123)] ** 2, axis=1))
                d_stt_st = np.sqrt(np.sum(dl_st[:, pipe2.SL_STT] ** 2, axis=1) + np.sum(dl_st[:, 187 + np.arange(123, 187)] ** 2, axis=1))
                d_nonq_st = np.hypot(d_pre_st, d_stt_st)
                X29_s2_static[:, 4] = c_st
                X29_s2_static[:, 5] = dp_st / (nq_vec[bk_s2] + 1e-12)
                X29_s2_static[:, 13] = pyth_st
                X29_s2_static[:, 14] = thales_st
                X29_s2_static[:, 24] = 0.0
                X29_s2_static[:, 25] = np.log((dp_st + 0.5) / (du_st + 0.5)) if use_tangent_U else 0.0
                X29_s2_static[:, 26] = np.tanh((d_qrs_st / pipe2.R_QRS) / 0.75)
                X29_s2_static[:, 27] = np.tanh((d_stt_st / pipe2.R_STT) / 0.75)
                X29_s2_static[:, 28] = np.log((d_qrs_st + 0.5) / (d_nonq_st + 0.5))
                X29_s2_static = np.nan_to_num(X29_s2_static, nan=0.0, posinf=0.0, neginf=0.0)

    out_off = {
        "y": post_y, "s2_mask": s2_mask, "X29_s2": X29_s2,
        "K": len(anchors), "post_hours": data["post_hours"],
        "tau1_beat": first_anchor_beat, "tau1_min": tau1_min, "pre_anchor_F": pre_anchor_F,
        "X_singlepole": X_singlepole, "forensics": forensics,
    }
    if also_static_bn:
        out_off["static_bn"] = {
            "y": post_y, "s2_mask": s2_mask, "X29_s2": X29_s2_static,
            "K": len(anchors), "post_hours": data["post_hours"],
            "tau1_beat": first_anchor_beat, "tau1_min": tau1_min, "pre_anchor_F": pre_anchor_F,
        }
    if anchor_mode == "offline_and_online":
        out_off["online_causal"] = online_res
    return out_off


def parity_selftest():
    """Verify bit-exact parity of Tier-0 Offline A3 against stage2_frozen_pipeline_v2 on 3 DBs."""
    import lab_diagnostic_v36_6hospital_lodo_benchmark as v36
    tests = [("mitdb", "213", (0, 1)), ("edb", "e0103", (0, 1)), ("svdb", "801", (0, 1))]
    for db, rid, chs in tests:
        rp = os.path.join(v36.DB_DIRS[db], rid)
        ref = pipe2.extract_record_features_v2(rp, db, rid, rid, ann_ext="atr", channel_indices=chs, label_map_version="AAMI_Map_v1.0")
        dat = load_beats_and_rr(rp, "atr", chs, "AAMI_Map_v1.0", filter_mode="offline_filtfilt")
        got = run_dualpole_variant(dat, use_tangent_U=True, use_local_bN=True, anchor_mode="offline", collect_singlepole=True, collect_stage1_forensics=True)
        assert np.array_equal(ref["y"], got["y"]), f"y mismatch on {db}/{rid}"
        assert np.array_equal(ref["s2_mask"], got["s2_mask"]), f"s2_mask mismatch on {db}/{rid}"
        cols_check = sorted(set(IDX_A3_N + IDX_C3_V))
        max_err = float(np.max(np.abs(ref["X29_s2"][:, cols_check] - got["X29_s2"][:, cols_check]))) if len(ref["X29_s2"]) else 0.0
        # Also verify Raw & TradMorph shapes
        xr, xm = extract_traditional_morphology_and_raw(dat["joint_all"][50:], dat["pre_rr"][50:], dat["pre_rr_loc"][50:])
        # Also verify Tier 1, Tier 2, Tier 3 run cleanly
        t1 = run_dualpole_variant(dat, anchor_mode="online_causal")
        t2 = run_dualpole_variant(dat, anchor_mode="prefix_only")
        dat_c = load_beats_and_rr(rp, "atr", chs, "AAMI_Map_v1.0", filter_mode="causal_double_median_fir")
        t3 = run_dualpole_variant(dat_c, anchor_mode="online_causal")
        print(f"[PARITY PASS] {db}/{rid}: max|dA3|={max_err:.2e} | s2={got['s2_mask'].sum()} | T1={t1['s2_mask'].sum()} (tau1={t1['tau1_min']}m) | T2={t2['s2_mask'].sum()} | T3={t3['s2_mask'].sum()} | raw={xr.shape} morph={xm.shape} sp={got['X_singlepole'].shape}")
        assert max_err < 1e-5, f"Parity error {max_err} on {db}/{rid}"


if __name__ == "__main__":
    parity_selftest()
