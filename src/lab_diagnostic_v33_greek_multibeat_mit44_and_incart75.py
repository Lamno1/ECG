"""
lab_diagnostic_v33_greek_multibeat_mit44_and_incart75.py
========================================================
THỬ NGHIỆM v32b (MIT-BIH 44/44) -> v33 (KIỂM CHỨNG NGOẠI SUY SẠCH TRÊN TOÀN BỘ 75 BẢN GHI INCART):
Kiểm chứng cụm Ngữ cảnh Đa nhịp 0-beat nhân quả thuần túy (KHÔNG cần nhịp tương lai,
đã loại post-RR do xung đột 208/213 ở v32) đặt chồng lên Hình học Hy Lạp cổ G3 & G4:
  - BurstPast2      : nb_hf_past2, nb_bcorr_drop_past2 (chặn chùm nhiễu burst như Rec 105)
  - MBridgePast3    : nb_mbridge_frac_past3 (chặn cả chùm nhiễu lẫn dải trôi nền Drift3; đồng nhất 0.333 trên 208 & 213!)
  - FullCtx (0-beat): BurstPast2 + nb_mbridge_frac_past3

Tất cả đặc trưng bất biến trực giao O(4) chính xác (<= 1e-14).
Stage A: Ablation trên MIT-BIH 44/44 (Dev10 LOPO -> Hold34, offline & causal FIR).
Stage B: Khóa cứng từ MIT-Dev10 (ZERO TUNING trên INCART), đánh giá 1 lần trên toàn bộ
         75 bản ghi INCART (32 bệnh nhân, 36.57 giờ) ở cả Offline, Causal FIR, và 4 lớp N+S+V+F.
"""


import os as _os, sys as _sys
from pathlib import Path as _Path
_SRC_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_DATA_ROOT = _Path(_os.environ.get("ECG_DATA_ROOT", str(_REPO_ROOT / "data_public_ecg")))
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import os
import json
import time
from collections import deque
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import ortho_group
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import wfdb

MITDB_DIR = str(_DATA_ROOT / "mitdb") + "/"
INCART_DIR = str(_DATA_ROOT / "incartdb") + "/"
MIT_TRAIN_10 = ["208", "213", "223", "233", "205", "210", "200", "201", "203", "221"]
MINOR_F_RECS = {"223", "233", "205", "210"}
DRIFT3_RECS = {"124", "207", "106"}

TARGET_FS = 360.0
PHYS_TIME_GRID = np.arange(-93, 94, dtype=float) / TARGET_FS
PRE_QRS_SLICE = slice(7, 73)
QRS_CORE_SLICE = slice(71, 123)

# Exact locked Gen-2 constants (ZERO change!)
CAND_NORM_Y_MIN = 10.0
CAND_ETA_MULT = 2.0
CAND_PRE_RR_LOC_MAX = 0.88
CLUSTER_RADIUS = 12.0
CONFIRM_MIN_BEATS = 6
CONFIRM_MAX_COHESION = 8.0
CONFIRM_MIN_NORM_Q = 12.0
MAX_V_ANCHORS = 4
MAX_CAND_CLUSTERS = 8
RAD_N_FLOOR = 0.05
RAD_V_FLOOR = 0.12
RAD_YV_FRAC_FLOOR = 0.25
V_SPREAD_BUF_MAX = 50
LOC_BUF_LEN = 25
LOC_RR_MIN = 0.88
LOC_RR_MAX = 1.18
LOC_ADMIT_CORR_BN = 0.70
LOC_ADMIT_RUN_CORR = 0.88
LOC_V_EXCLUSION_DIST = 10.0
ONL_HF_MAX = 1.20


def filter_signal_generic(sig, fs, mode="offline_filtfilt"):
    nyq = fs / 2.0
    if mode == "offline_filtfilt":
        b, a = signal.butter(3, [0.5 / nyq, 40.0 / nyq], btype="bandpass")
        return signal.filtfilt(b, a, sig), 0
    elif mode == "causal_double_median_fir":
        w200 = int(round(0.200 * fs)) | 1
        w600 = int(round(0.600 * fs)) | 1
        s = pd.Series(sig)
        bw = s.rolling(w200, min_periods=1).median().rolling(w600, min_periods=1).median().to_numpy()
        detrended = sig - bw
        ntaps = int(round(0.084 * fs)) | 1
        b_fir = signal.firwin(ntaps, cutoff=40.0, fs=fs)
        zi = signal.lfilter_zi(b_fir, [1.0]) * float(detrended[0])
        out, _ = signal.lfilter(b_fir, [1.0], detrended, zi=zi)
        return out, (ntaps - 1) // 2
    raise ValueError(mode)


def load_record(rec_dir, rec_id, dataset="mitdb", filter_mode="offline_filtfilt", include_class_s=False):
    rec_path = os.path.join(rec_dir, rec_id)
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")
    fs = float(rec.fs)

    if dataset == "incart":
        sig_names = [s.upper().strip() for s in rec.sig_name]
        i0, i1 = sig_names.index("II"), sig_names.index("V1")
        patient_id = "unknown"
        for cmt in (rec.comments or []):
            if "patient" in cmt.lower():
                patient_id = cmt.strip()
                break
    else:
        i0, i1 = 0, 1
        patient_id = f"mit_{rec_id}"

    sig0, delay = filter_signal_generic(rec.p_signal[:, i0], fs, mode=filter_mode)
    sig1, _ = filter_signal_generic(rec.p_signal[:, i1], fs, mode=filter_mode)

    def map_sym(sym):
        if sym in ["N", "L", "R", "e", "j"]:
            return 0
        if sym in ["V", "E"]:
            return 1
        if sym == "F":
            return 2
        if include_class_s and sym in ["A", "a", "J", "S"]:
            return 3
        return -1

    beats0, beats1, labels, r_peaks = [], [], [], []
    if abs(fs - TARGET_FS) < 1e-3:
        for s, sym in zip(ann.sample, ann.symbol):
            sc = s + delay
            if sc < 93 or sc + 94 > len(sig0):
                continue
            c = map_sym(sym)
            if c >= 0:
                beats0.append(sig0[sc - 93 : sc + 94])
                beats1.append(sig1[sc - 93 : sc + 94])
                labels.append(c)
                r_peaks.append(s)
    else:
        half_win = int(np.ceil(0.30 * fs))
        local_t = np.arange(-half_win, half_win + 1, dtype=float) / fs
        for s, sym in zip(ann.sample, ann.symbol):
            sc = s + delay
            if sc < half_win or sc + half_win + 1 > len(sig0):
                continue
            c = map_sym(sym)
            if c >= 0:
                w0 = sig0[sc - half_win : sc + half_win + 1]
                w1 = sig1[sc - half_win : sc + half_win + 1]
                beats0.append(np.interp(PHYS_TIME_GRID, local_t, w0))
                beats1.append(np.interp(PHYS_TIME_GRID, local_t, w1))
                labels.append(c)
                r_peaks.append(s)

    beats0 = np.array(beats0, dtype=np.float32)
    beats1 = np.array(beats1, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    r_peaks = np.array(r_peaks, dtype=np.float64)
    N = len(labels)

    pre_rr = np.zeros(N, dtype=np.float64)
    if N > 1:
        pre_rr[1:] = (r_peaks[1:] - r_peaks[:-1]) / fs
        pre_rr[0] = pre_rr[1]
    else:
        pre_rr[:] = 0.8

    local_rr = np.zeros(N, dtype=np.float64)
    for i in range(N):
        lo = max(0, i - 10)
        local_rr[i] = float(np.median(pre_rr[lo : i + 1])) + 1e-6

    post_hours = float((r_peaks[-1] - r_peaks[50]) / fs / 3600.0) if N > 51 else 0.0
    return {
        "rec_id": rec_id, "patient_id": patient_id,
        "raw_0": beats0, "raw_1": beats1,
        "labels": labels,
        "pre_rr_loc": (pre_rr / local_rr).astype(np.float32),
        "post_hours": post_hours,
    }


def normalize_lead(beats):
    m = np.mean(beats, axis=1, keepdims=True)
    s = np.std(beats, axis=1, keepdims=True) + 1e-6
    return (beats - m) / s


def pearson_1d(a, b):
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    return float(np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-12))


def mean_lead_corr(x_vec, ref_vec):
    return 0.5 * (pearson_1d(x_vec[:187], ref_vec[:187]) + pearson_1d(x_vec[187:], ref_vec[187:]))


def pearson_corr_rows(X, ref):
    r_c = ref - np.mean(ref)
    r_n = np.linalg.norm(r_c) + 1e-12
    X_c = X - np.mean(X, axis=1, keepdims=True)
    X_n = np.linalg.norm(X_c, axis=1) + 1e-12
    return (X_c @ r_c) / (X_n * r_n)


def thales_obt_and_pyth(dA, dB, dAB):
    dAB_s = dAB + 1e-12
    num = dAB * dAB - dA * dA - dB * dB
    obt = float(np.clip(num / (2.0 * dA * dB + 1e-12), -1.0, 1.0))
    pyth = float(num / (dAB_s * dAB_s))
    return obt, pyth


def zenodorus_qiso_abs_batch(X0_qrs, X1_qrs):
    u = X0_qrs - np.mean(X0_qrs, axis=1, keepdims=True)
    v = X1_qrs - np.mean(X1_qrs, axis=1, keepdims=True)
    un = np.roll(u, -1, axis=1)
    vn = np.roll(v, -1, axis=1)
    area = 0.5 * np.sum(u * vn - un * v, axis=1)
    du = un - u
    dv = vn - v
    perim = np.sum(np.sqrt(du * du + dv * dv), axis=1)
    return np.abs((4.0 * np.pi * area) / (perim * perim + 1e-12))


def _refresh_v_radii(anc):
    lam_arr = np.array(anc["v_lam_buf"], dtype=float)
    dy_arr = np.array(anc["v_dy_buf"], dtype=float)
    anc["rad_V"] = max(float(np.percentile(np.abs(1.0 - lam_arr), 90)), RAD_V_FLOOR)
    anc["rad_y_V"] = max(float(np.percentile(dy_arr, 90)), RAD_YV_FRAC_FLOOR * anc["norm_q"])


def make_anchor_obj(ect_beats, norm_y_prefix, b_N, P, confirm_beat, source):
    b_V = np.median(ect_beats, axis=0)
    q = P @ (b_V - b_N)
    nq = float(np.linalg.norm(q))
    if nq < CONFIRM_MIN_NORM_Q:
        return None
    lam_N_pref = (norm_y_prefix @ q) / (nq ** 2 + 1e-12)
    mu_N = float(np.median(lam_N_pref))
    rad_N = max(float(np.percentile(np.abs(lam_N_pref - mu_N), 95)), RAD_N_FLOOR)

    y_ect = (ect_beats - b_N[None, :]) @ P.T
    lam_V_seed = (y_ect @ q) / (nq ** 2 + 1e-12)
    dy_V_seed = np.linalg.norm(y_ect - q[None, :], axis=1)
    anc = {
        "b_V": b_V, "q": q, "norm_q": nq, "mu_N": mu_N, "rad_N": rad_N,
        "v_lam_buf": deque(lam_V_seed.tolist()[-V_SPREAD_BUF_MAX:], maxlen=V_SPREAD_BUF_MAX),
        "v_dy_buf": deque(dy_V_seed.tolist()[-V_SPREAD_BUF_MAX:], maxlen=V_SPREAD_BUF_MAX),
        "confirm_beat": confirm_beat, "source": source,
    }
    _refresh_v_radii(anc)
    return anc


def process_record_v33(rd, R_rot=None, compute_all=False):
    t0 = time.perf_counter()
    raw0 = rd["raw_0"].astype(np.float64)
    raw1 = rd["raw_1"].astype(np.float64)
    y_all = rd["labels"]
    pre_rr_loc_all = rd["pre_rr_loc"].astype(np.float64)
    joint_all = np.column_stack([normalize_lead(raw0), normalize_lead(raw1)])

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

    pre0_50 = raw0[:50, PRE_QRS_SLICE] - np.mean(raw0[:50, PRE_QRS_SLICE], axis=1, keepdims=True)
    pre1_50 = raw1[:50, PRE_QRS_SLICE] - np.mean(raw1[:50, PRE_QRS_SLICE], axis=1, keepdims=True)
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
        pearson_corr_rows(norm_beats[:, :187], bN_0),
        pearson_corr_rows(norm_beats[:, 187:], bN_1),
    )
    pref_bcorr_p05 = float(np.percentile(pref_min_ncorr, 5))

    pref_qiso_abs = zenodorus_qiso_abs_batch(norm_beats[:, QRS_CORE_SLICE], norm_beats[:, 187:][:, QRS_CORE_SLICE])
    med_pref_qiso_abs = float(np.median(pref_qiso_abs))

    anchors = []
    if len(ectopic_idx) >= 4:
        anc = make_anchor_obj(joint_50[ectopic_idx], norm_y_prefix, b_N, P, 0, "prefix")
        if anc is not None:
            anchors.append(anc)

    post_joint = joint_all[50:]
    post_raw0, post_raw1 = raw0[50:], raw1[50:]
    post_y = y_all[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    N = len(post_joint)

    hf0 = np.sum(np.diff(post_joint[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1 = np.sum(np.diff(post_joint[:, 187:], n=2, axis=1) ** 2, axis=1)
    sqi_hf_ratio = np.log1p(np.maximum(hf0 / med_hf0, hf1 / med_hf1))

    all_qiso_abs = zenodorus_qiso_abs_batch(post_joint[:, QRS_CORE_SLICE], post_joint[:, 187:][:, QRS_CORE_SLICE])
    zeno_qiso_drop_all = med_pref_qiso_abs - all_qiso_abs

    y_mat = (post_joint - b_N[None, :]) @ P.T
    norm_y = np.linalg.norm(y_mat, axis=1)
    norm_y_thresh = max(CAND_NORM_Y_MIN, CAND_ETA_MULT * eta_0)

    loc_buf = deque([b.copy() for b in norm_beats[-LOC_BUF_LEN:]], maxlen=LOC_BUF_LEN)
    b_N_loc = np.median(np.array(loc_buf), axis=0)
    b_N_loc_hist = np.zeros((N, D))
    loc_norm_corr_arr = np.zeros(N)
    loc_norm_dist_proj_arr = np.zeros(N)
    prev1, prev2 = joint_all[49], joint_all[48]

    cand_clusters = []
    for t in range(N):
        x_t = post_joint[t]
        b_N_loc_hist[t] = b_N_loc
        c_loc = mean_lead_corr(x_t, b_N_loc)
        loc_norm_corr_arr[t] = c_loc
        loc_norm_dist_proj_arr[t] = float(np.linalg.norm(P @ (x_t - b_N_loc)))

        if (norm_y[t] >= norm_y_thresh) and (pre_rr_loc[t] <= CAND_PRE_RR_LOC_MAX) and (sqi_hf_ratio[t] <= ONL_HF_MAX):
            matched_anc, best_adist = None, CLUSTER_RADIUS
            for anc in anchors:
                d_a = float(np.linalg.norm(x_t - anc["b_V"]))
                if d_a <= best_adist:
                    best_adist, matched_anc = d_a, anc
            if matched_anc is not None:
                q = matched_anc["q"]
                nq = matched_anc["norm_q"]
                matched_anc["v_lam_buf"].append(float(np.dot(y_mat[t], q) / (nq ** 2 + 1e-12)))
                matched_anc["v_dy_buf"].append(float(np.linalg.norm(y_mat[t] - q)))
                _refresh_v_radii(matched_anc)
            elif len(anchors) < MAX_V_ANCHORS:
                best_c, best_dist = None, CLUSTER_RADIUS
                for cl in cand_clusters:
                    dist = float(np.linalg.norm(x_t - cl["center"]))
                    if dist <= best_dist:
                        best_dist, best_c = dist, cl
                if best_c is not None:
                    best_c["beats"].append(x_t)
                    best_c["center"] = np.median(best_c["beats"], axis=0)
                    if len(best_c["beats"]) >= CONFIRM_MIN_BEATS:
                        arr_b = np.array(best_c["beats"])
                        med_b = np.median(arr_b, axis=0)
                        cohesion = float(np.median(np.linalg.norm(arr_b - med_b, axis=1)))
                        if cohesion <= CONFIRM_MAX_COHESION:
                            anc = make_anchor_obj(arr_b, norm_y_prefix, b_N, P, int(t + 1), "online")
                            if anc is not None:
                                anchors.append(anc)
                                cand_clusters.remove(best_c)
                else:
                    if len(cand_clusters) >= MAX_CAND_CLUSTERS:
                        cand_clusters.sort(key=lambda c: len(c["beats"]))
                        cand_clusters.pop(0)
                    cand_clusters.append({"beats": [x_t], "center": x_t.copy()})

        reg_rr = (LOC_RR_MIN <= pre_rr_loc[t] <= LOC_RR_MAX)
        far_from_v = all(float(np.linalg.norm(x_t - anc["b_V"])) >= LOC_V_EXCLUSION_DIST for anc in anchors)
        if reg_rr and far_from_v:
            c_prev1 = mean_lead_corr(x_t, prev1)
            c_prev12 = mean_lead_corr(prev1, prev2)
            if (c_loc >= LOC_ADMIT_CORR_BN and c_prev1 >= 0.70) or (c_prev1 >= LOC_ADMIT_RUN_CORR and c_prev12 >= LOC_ADMIT_RUN_CORR):
                loc_buf.append(x_t)
                b_N_loc = np.median(np.array(loc_buf), axis=0)

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
        pearson_corr_rows(post_raw0[:, PRE_QRS_SLICE], pN_0) +
        pearson_corr_rows(post_raw1[:, PRE_QRS_SLICE], pN_1)
    )
    rr_dev_pre = np.abs(pre_rr_loc - 1.0)

    if len(anchors) > 0:
        ready[:] = True
        y_loc_pole_mat = (b_N_loc_hist - b_N[None, :]) @ P.T
        for t in range(N):
            yv = y_mat[t]
            best_eucl_seg, best_k, best_lam, best_dseg, e_V_any = 1e9, 0, 0.0, 9.99, 1e9
            for idx, anc in enumerate(anchors):
                q = anc["q"]
                nq = anc["norm_q"]
                lam = float(np.dot(yv, q) / (nq ** 2 + 1e-12))
                lam_c = float(np.clip(lam, 0.0, 1.0))
                eucl_seg = float(np.linalg.norm(yv - lam_c * q))
                ej = max(0.0, (float(np.linalg.norm(yv - q)) - anc["rad_y_V"]) / (nq + 1e-12))
                if ej < e_V_any:
                    e_V_any = ej
                if eucl_seg < best_eucl_seg:
                    best_eucl_seg, best_k, best_lam, best_dseg = eucl_seg, idx, lam, eucl_seg / (nq + 1e-12)

            anc_s = anchors[best_k]
            e_N = max(0.0, (best_lam - anc_s["mu_N"]) - anc_s["rad_N"])
            e_Vc = max(0.0, (1.0 - best_lam) - anc_s["rad_V"])
            zN = float(np.log1p(e_N / anc_s["rad_N"]))
            zV = min(float(np.log1p(e_Vc / anc_s["rad_V"])), float(np.log1p(e_V_any / anc_s["rad_V"])))
            d_seg_arr[t] = best_dseg
            z_N_arr[t] = zN
            z_V_arr[t] = zV
            m_b = min(zN, zV)
            m_bridge_arr[t] = m_b

            x0, x1 = post_joint[t, :187], post_joint[t, 187:]
            bV0, bV1 = anc_s["b_V"][:187], anc_s["b_V"][187:]
            lc = float(np.clip(best_lam, 0.0, 1.0))
            min_lead_bcorr_arr[t] = min(pearson_1d(x0, bN_0 + lc * (bV0 - bN_0)),
                                        pearson_1d(x1, bN_1 + lc * (bV1 - bN_1)))
            loc_dist_norm_arr[t] = loc_norm_dist_proj_arr[t] / (anc_s["norm_q"] + 1e-12)

            if m_b > 0.0 or compute_all:
                q = anc_s["q"]
                dN_l = loc_norm_dist_proj_arr[t]
                dV_l = float(np.linalg.norm(yv - q))
                dNV_l = float(np.linalg.norm(q - y_loc_pole_mat[t]))
                _, pyth_margin_loc[t] = thales_obt_and_pyth(dN_l, dV_l, dNV_l)

                bL0, bL1 = b_N_loc_hist[t, :187], b_N_loc_hist[t, 187:]
                obt_l0, _ = thales_obt_and_pyth(
                    float(np.linalg.norm(x0 - bL0)), float(np.linalg.norm(x0 - bV0)), float(np.linalg.norm(bV0 - bL0))
                )
                obt_l1, _ = thales_obt_and_pyth(
                    float(np.linalg.norm(x1 - bL1)), float(np.linalg.norm(x1 - bV1)), float(np.linalg.norm(bV1 - bL1))
                )
                min_lead_thales_loc[t] = min(obt_l0, obt_l1)

    rel_bcorr_drop = pref_bcorr_p05 - min_lead_bcorr_arr
    hf_excess = np.maximum(0.0, sqi_hf_ratio - max(pref_hf_p95, 0.70))
    bcorr_hinge_sq = np.maximum(0.0, rel_bcorr_drop - 0.10) ** 2
    hf_hinge_sq = hf_excess ** 2

    def shift_past(arr, k):
        out = np.empty_like(arr, dtype=float)
        out[:k] = arr[0]
        out[k:] = arr[:-k]
        return out

    # Strictly 0-beat causal neighborhood features
    nb_hf_past2 = 0.5 * (shift_past(sqi_hf_ratio, 1) + shift_past(sqi_hf_ratio, 2))
    nb_bcorr_drop_past2 = 0.5 * (shift_past(rel_bcorr_drop, 1) + shift_past(rel_bcorr_drop, 2))
    mb_ind = (m_bridge_arr > 0.0).astype(float)
    nb_mbridge_frac_past3 = (shift_past(mb_ind, 1) + shift_past(mb_ind, 2) + shift_past(mb_ind, 3)) / 3.0

    X12 = np.column_stack([
        m_bridge_arr, z_N_arr, z_V_arr, d_seg_arr,
        loc_norm_corr_arr, loc_dist_norm_arr,
        preqrs_corr_N, rr_dev_pre, min_lead_bcorr_arr, sqi_hf_ratio,
        bcorr_hinge_sq, hf_hinge_sq,
    ])
    extras = {
        "zeno_qiso_drop": zeno_qiso_drop_all,
        "pyth_margin_loc": pyth_margin_loc,
        "min_lead_thales_loc": min_lead_thales_loc,
        "nb_hf_past2": nb_hf_past2,
        "nb_bcorr_drop_past2": nb_bcorr_drop_past2,
        "nb_mbridge_frac_past3": nb_mbridge_frac_past3,
    }
    return {
        "rec_id": rd["rec_id"],
        "patient_id": rd["patient_id"],
        "post_hours": rd["post_hours"],
        "y": post_y,
        "ready": ready,
        "m_bridge": m_bridge_arr,
        "X12": X12,
        "extras": extras,
        "proc_sec": time.perf_counter() - t0,
    }


def select_oof_f1_threshold(oof_scores, oof_y_bin):
    if int(np.sum(oof_y_bin == 1)) == 0:
        return 0.0
    prec, rec, thresholds = precision_recall_curve(oof_y_bin, oof_scores)
    if len(thresholds) == 0:
        return 0.0
    p, r = prec[:-1], rec[:-1]
    denom = p + r
    f1s = np.zeros_like(denom)
    nz = denom > 0
    f1s[nz] = 2.0 * p[nz] * r[nz] / denom[nz]
    if float(np.max(f1s)) <= 0.0:
        return float(np.max(oof_scores)) + 1.0
    return float(thresholds[int(np.argmax(f1s))])


def build_X(it, keys):
    if not keys:
        return it["X12"]
    return np.column_stack([it["X12"]] + [it["extras"][k] for k in keys])


def fit_locked_on_mit10(dev_items, keys, model_type):
    gate_fn = lambda it: it["ready"] & (it["m_bridge"] > 0.0)
    oof_scores, oof_y, oof_recs = [], [], []
    for i, te_it in enumerate(dev_items):
        tr_X = np.vstack([build_X(it, keys)[gate_fn(it)] for j, it in enumerate(dev_items) if j != i and np.any(gate_fn(it))])
        tr_y = np.concatenate([(it["y"][gate_fn(it)] == 2).astype(int) for j, it in enumerate(dev_items) if j != i and np.any(gate_fn(it))])
        sc_te = np.full(len(te_it["y"]), -1e6)
        m_te = gate_fn(te_it)
        if np.any(m_te) and np.sum(tr_y == 1) > 0:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(tr_X)
            Xte_s = sc.transform(build_X(te_it, keys)[m_te])
            clf = (LogisticRegression(class_weight="balanced", C=0.3, max_iter=500, random_state=42)
                   if model_type == "Linear_LogReg"
                   else LinearSVC(class_weight="balanced", C=0.3, dual="auto", max_iter=3000, random_state=42))
            clf.fit(Xtr_s, tr_y)
            sc_te[m_te] = clf.decision_function(Xte_s)
        oof_scores.append(sc_te)
        oof_y.append(te_it["y"])
        oof_recs.append(np.full(len(te_it["y"]), te_it["rec_id"]))

    oof_sc = np.concatenate(oof_scores)
    oof_y_all = np.concatenate(oof_y)
    oof_recs_all = np.concatenate(oof_recs)
    oof_bin = (oof_y_all == 2).astype(int)
    dev_ap = float(average_precision_score(oof_bin, oof_sc))
    tau = select_oof_f1_threshold(oof_sc, oof_bin)
    pr = (oof_sc >= tau).astype(int)
    tp = int(np.sum((pr == 1) & (oof_y_all == 2)))
    fp = int(np.sum((pr == 1) & (oof_y_all != 2)))
    fn = int(np.sum((pr == 0) & (oof_y_all == 2)))
    mit_f1 = float(2.0 * tp / (2.0 * tp + fp + fn + 1e-12))

    def rf1(rid):
        rm = (oof_recs_all == rid)
        rtp = int(np.sum((pr[rm] == 1) & (oof_y_all[rm] == 2)))
        rfp = int(np.sum((pr[rm] == 1) & (oof_y_all[rm] != 2)))
        rfn = int(np.sum((pr[rm] == 0) & (oof_y_all[rm] == 2)))
        return float(2.0 * rtp / (2.0 * rtp + rfp + rfn + 1e-12)), rtp, int(np.sum(oof_y_all[rm] == 2))

    f1_208, _, _ = rf1("208")
    f1_213, _, _ = rf1("213")
    min_tp = sum(rf1(mr)[1] for mr in sorted(MINOR_F_RECS))
    min_tot = sum(rf1(mr)[2] for mr in sorted(MINOR_F_RECS))

    X_all = np.vstack([build_X(it, keys)[gate_fn(it)] for it in dev_items if np.any(gate_fn(it))])
    y_all = np.concatenate([(it["y"][gate_fn(it)] == 2).astype(int) for it in dev_items if np.any(gate_fn(it))])
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    clf = (LogisticRegression(class_weight="balanced", C=0.3, max_iter=500, random_state=42)
           if model_type == "Linear_LogReg"
           else LinearSVC(class_weight="balanced", C=0.3, dual="auto", max_iter=3000, random_state=42))
    clf.fit(X_all_s, y_all)
    meta = {
        "Params": int(X_all.shape[1] + 1),
        "Dev10_AP": round(dev_ap, 4),
        "Dev10_OOF_F1": round(mit_f1, 4),
        "F1_208": round(f1_208, 4),
        "F1_213": round(f1_213, 4),
        "MinorF_Rec": f"{min_tp}/{min_tot}",
        "Tau_MIT": round(tau, 4),
    }
    return scaler, clf, tau, meta


def eval_hold34(hold_items, keys, scaler, clf, tau):
    gate_fn = lambda it: it["ready"] & (it["m_bridge"] > 0.0)
    h_tp = h_fpn = h_fpv = fp_105 = fp_dr3 = fp_108 = fp_230 = zf_tot = zf_fp = 0
    for hit in hold_items:
        hy = hit["y"]
        sc = np.full(len(hy), -1e6)
        mh = gate_fn(hit)
        if np.any(mh):
            sc[mh] = clf.decision_function(scaler.transform(build_X(hit, keys)[mh]))
        pr = (sc >= tau).astype(int)
        tp_r = int(np.sum((pr == 1) & (hy == 2)))
        fpn_r = int(np.sum((pr == 1) & (hy == 0)))
        fpv_r = int(np.sum((pr == 1) & (hy == 1)))
        h_tp += tp_r; h_fpn += fpn_r; h_fpv += fpv_r
        rid = hit["rec_id"]
        if rid == "105": fp_105 = fpn_r
        if rid in DRIFT3_RECS: fp_dr3 += fpn_r
        if rid == "108": fp_108 = fpn_r
        if rid == "230": fp_230 = fpn_r
        if int(np.sum(hy == 2)) == 0:
            zf_tot += len(hy); zf_fp += (fpn_r + fpv_r)
    return {
        "Hold34_FP_N": h_fpn, "Hold34_FP_V": h_fpv,
        "FPN_108_230_105_Dr3": f"{fp_108}/{fp_230}/{fp_105}/{fp_dr3}",
        "Hold34_ZeroF_Spec_%": round(100.0 * (1.0 - zf_fp / max(1, zf_tot)), 3),
    }


def eval_incart75(inc_items, keys, scaler, clf, tau, meta, include_s=False):
    gate_fn = lambda it: it["ready"] & (it["m_bridge"] > 0.0)
    all_scores, all_y, all_pats = [], [], []
    tot_hours = sum(it["post_hours"] for it in inc_items)
    zf_tot = zf_fp = 0
    rec_detail = []
    for it in inc_items:
        y = it["y"]
        sc = np.full(len(y), -1e6)
        m = gate_fn(it)
        if np.any(m):
            sc[m] = clf.decision_function(scaler.transform(build_X(it, keys)[m]))
        all_scores.append(sc); all_y.append(y); all_pats.append(np.full(len(y), it["patient_id"]))
        pr = (sc >= tau).astype(int)
        n_f = int(np.sum(y == 2))
        tp_r = int(np.sum((pr == 1) & (y == 2)))
        fpn_r = int(np.sum((pr == 1) & (y == 0)))
        fpv_r = int(np.sum((pr == 1) & (y == 1)))
        fps_r = int(np.sum((pr == 1) & (y == 3)))
        if n_f == 0:
            zf_tot += len(y); zf_fp += (fpn_r + fpv_r + fps_r)
        else:
            f1_r = float(2.0 * tp_r / (2.0 * tp_r + fpn_r + fpv_r + fps_r + (n_f - tp_r) + 1e-12))
            rec_detail.append({"Rec": it["rec_id"], "True_F": n_f, "TP": tp_r, "FP_N": fpn_r, "FP_V": fpv_r, "F1": round(f1_r, 4)})

    sc_all = np.concatenate(all_scores)
    y_all = np.concatenate(all_y)
    pats_all = np.concatenate(all_pats)
    y_bin = (y_all == 2).astype(int)
    pooled_ap = float(average_precision_score(y_bin, sc_all))
    pr_all = (sc_all >= tau).astype(int)
    tp = int(np.sum((pr_all == 1) & (y_all == 2)))
    fpn = int(np.sum((pr_all == 1) & (y_all == 0)))
    fpv = int(np.sum((pr_all == 1) & (y_all == 1)))
    fps = int(np.sum((pr_all == 1) & (y_all == 3)))
    tot_s = int(np.sum(y_all == 3))
    fn = int(np.sum((pr_all == 0) & (y_all == 2)))
    tot_fp = fpn + fpv + fps
    prec = 100.0 * tp / max(1, tp + tot_fp)
    rec = 100.0 * tp / max(1, tp + fn)
    f1 = 2.0 * tp / max(1e-12, 2.0 * tp + tot_fp + fn)
    faph = tot_fp / max(1e-6, tot_hours)
    zf_spec = 100.0 * (1.0 - zf_fp / max(1, zf_tot))

    pat_aps, pat_f1s = [], []
    for p in sorted(set(pats_all)):
        pm = (pats_all == p)
        if int(np.sum(y_all[pm] == 2)) > 0:
            pat_aps.append(float(average_precision_score(y_bin[pm], sc_all[pm])))
            ptp = int(np.sum((pr_all[pm] == 1) & (y_all[pm] == 2)))
            pfp = int(np.sum((pr_all[pm] == 1) & (y_all[pm] != 2)))
            pfn = int(np.sum((pr_all[pm] == 0) & (y_all[pm] == 2)))
            pat_f1s.append(float(2.0 * ptp / max(1e-12, 2.0 * ptp + pfp + pfn)))

    # Descriptive frontier
    order = np.argsort(-sc_all)
    y_sorted, sc_sorted = y_bin[order], sc_all[order]
    y_v = y_sorted[sc_sorted > -1e5]
    cum_tp, cum_fp = np.cumsum(y_v), np.cumsum(1 - y_v)
    n_pos = int(np.sum(y_bin))
    frontier = {}
    for target_faph in [2.0, 5.0, 10.0]:
        idx = np.searchsorted(cum_fp, int(np.floor(target_faph * tot_hours)), side="right") - 1
        if idx >= 0:
            ftp, ffp = int(cum_tp[idx]), int(cum_fp[idx])
            frontier[f"FAPH<={int(target_faph)}"] = {
                "TP": ftp, "FP": ffp,
                "Prec_%": round(100.0 * ftp / max(1, ftp + ffp), 2),
                "Rec_%": round(100.0 * ftp / max(1, n_pos), 2),
                "F1": round(2.0 * ftp / max(1e-12, 2.0 * ftp + ffp + (n_pos - ftp)), 4),
            }
        else:
            frontier[f"FAPH<={int(target_faph)}"] = {"TP": 0, "FP": 0, "Prec_%": 0.0, "Rec_%": 0.0, "F1": 0.0}
    denom = 2.0 * cum_tp + cum_fp + (n_pos - cum_tp)
    f1_c = np.where(denom > 0, 2.0 * cum_tp / denom, 0.0)
    bi = int(np.argmax(f1_c)) if len(f1_c) > 0 else 0
    if len(f1_c) > 0:
        btp, bfp = int(cum_tp[bi]), int(cum_fp[bi])
        frontier["Max_Desc_F1"] = {
            "F1": round(float(f1_c[bi]), 4),
            "Prec_%": round(100.0 * btp / max(1, btp + bfp), 2),
            "Rec_%": round(100.0 * btp / max(1, n_pos), 2),
            "FAPH": round(bfp / max(1e-6, tot_hours), 2),
        }

    row = {
        "Params": meta["Params"], "MIT_OOF_F1": meta["Dev10_OOF_F1"],
        "Pooled_AP": round(pooled_ap, 4), "Pooled_F1": round(f1, 4),
        "Pooled_Prec_%": round(prec, 2), "Pooled_Rec_%": round(rec, 2),
        "TP_FPN_FPV": f"{tp}/{fpn}/{fpv}" + (f"/{fps}S" if include_s else ""),
        "Pat_Mean_AP": round(float(np.mean(pat_aps)), 4),
        "Pat_Mean_F1": round(float(np.mean(pat_f1s)), 4),
        "ZeroF_Spec_%": round(zf_spec, 3),
        "FAPH": round(faph, 2),
    }
    if include_s:
        row["INCART_FP_S"] = f"{fps}/{tot_s} ({100.0*fps/max(1,tot_s):.2f}%)"
    return row, frontier, rec_detail


def main():
    all_mit = sorted([f.replace(".hea", "") for f in os.listdir(MITDB_DIR)
                      if f.endswith(".hea") and f.replace(".hea", "") not in ["102", "104", "107", "217"]])
    hold_mit = [r for r in all_mit if r not in MIT_TRAIN_10]
    inc_recs = sorted([f.replace(".hea", "") for f in os.listdir(INCART_DIR) if f.endswith(".hea")])

    # Exact O(4) Gauge Check
    rd208 = load_record(MITDB_DIR, "208", "mitdb", "offline_filtfilt", False)
    b208 = process_record_v33(rd208, R_rot=None, compute_all=True)
    R = ortho_group.rvs(4, random_state=33)
    r208 = process_record_v33(rd208, R_rot=R, compute_all=True)
    max_diff = max(
        float(np.max(np.abs(b208["X12"] - r208["X12"]))),
        max(float(np.max(np.abs(b208["extras"][k] - r208["extras"][k]))) for k in b208["extras"]),
    )
    print(f"=== O(4) GAUGE INVARIANCE CHECK (v33 all 0-beat features): {max_diff:.3e} ===")
    assert max_diff < 1e-12

    G3 = ["pyth_margin_loc", "min_lead_thales_loc"]
    G4 = ["zeno_qiso_drop", "pyth_margin_loc", "min_lead_thales_loc"]
    B2 = ["nb_hf_past2", "nb_bcorr_drop_past2"]
    MB3 = ["nb_mbridge_frac_past3"]

    variants = [
        ("B0_Gen2-Soft (12D)", []),
        ("G3_ThalesPyth (14D)", G3),
        ("G4_GreekTriad (15D)", G4),
        ("G3_BurstPast2 (16D)", G3 + B2),
        ("G4_BurstPast2 (17D)", G4 + B2),
        ("G4_MBridge3 (16D)", G4 + MB3),
        ("G3_FullCtx0b (17D)", G3 + B2 + MB3),
        ("G4_FullCtx0b (18D)", G4 + B2 + MB3),
    ]

    stage_a_rows = []
    stage_b_rows = []
    frontiers = {}
    rec22 = {}
    cpu_incart = {}

    for mode_tag, fmode, inc_s in [
        ("offline_NVF", "offline_filtfilt", False),
        ("causal_NVF", "causal_double_median_fir", False),
        ("offline_4cls_NSVF", "offline_filtfilt", True),
    ]:
        print(f"\n==================== [{mode_tag}] ====================")
        mit_dev = [process_record_v33(load_record(MITDB_DIR, r, "mitdb", fmode, inc_s)) for r in MIT_TRAIN_10]
        mit_hold = [process_record_v33(load_record(MITDB_DIR, r, "mitdb", fmode, inc_s)) for r in hold_mit] if not inc_s else None
        inc_items = [process_record_v33(load_record(INCART_DIR, r, "incart", fmode, inc_s)) for r in inc_recs]
        cpu_incart[mode_tag] = round(sum(it["proc_sec"] for it in inc_items) * 1000.0 / sum(len(it["y"]) for it in inc_items), 4)

        if not inc_s:
            print(f"\n--- STAGE A: MIT-BIH 44/44 [{mode_tag}] ---")
            print(f"{'Variant':24s} | {'Model':13s} | {'P':>2s} | {'DevAP':>6s} {'DevF1':>6s} {'208/213':>11s} {'MinF':>5s} | "
                  f"{'H34_FPN':>7s} {'FPV':>4s} {'108/230/105/Dr3':>15s} {'0F_Spec':>8s}")
            print("-" * 124)

        locked_models = []
        for vname, fkeys in variants:
            for mtype in ["Linear_SVM", "Linear_LogReg"]:
                scaler, clf, tau, meta = fit_locked_on_mit10(mit_dev, fkeys, mtype)
                locked_models.append((vname, fkeys, mtype, scaler, clf, tau, meta))
                if not inc_s:
                    hres = eval_hold34(mit_hold, fkeys, scaler, clf, tau)
                    arow = {"Mode": mode_tag, "Variant": vname, "Model": mtype, **meta, **hres}
                    stage_a_rows.append(arow)
                    print(f"{vname:24s} | {mtype:13s} | {arow['Params']:2d} | {arow['Dev10_AP']:6.4f} {arow['Dev10_OOF_F1']:6.4f} "
                          f"{arow['F1_208']:.3f}/{arow['F1_213']:.3f} {arow['MinorF_Rec']:>5s} | "
                          f"{arow['Hold34_FP_N']:7d} {arow['Hold34_FP_V']:4d} {arow['FPN_108_230_105_Dr3']:>15s} "
                          f"{arow['Hold34_ZeroF_Spec_%']:7.3f}%")

        print(f"\n--- STAGE B: INCART-75 ZERO-TUNING [{mode_tag}] (CPU={cpu_incart[mode_tag]} ms/beat) ---")
        print(f"{'Variant':24s} | {'Model':13s} | {'P':>2s} | {'MIT_F1':>6s} | {'PoolAP':>6s} {'PoolF1':>6s} {'Prec%':>6s} {'Rec%':>6s} | "
              f"{'TP/FPN/FPV':>16s} | {'PatAP':>6s} {'PatF1':>6s} | {'0F_Spec':>7s} {'FAPH':>6s}")
        print("-" * 141)
        for vname, fkeys, mtype, scaler, clf, tau, meta in locked_models:
            brow, front, rdet = eval_incart75(inc_items, fkeys, scaler, clf, tau, meta, include_s=inc_s)
            brow["Mode"] = mode_tag
            brow["Variant"] = vname
            brow["Model"] = mtype
            stage_b_rows.append(brow)
            print(f"{vname:24s} | {mtype:13s} | {brow['Params']:2d} | {brow['MIT_OOF_F1']:6.4f} | "
                  f"{brow['Pooled_AP']:6.4f} {brow['Pooled_F1']:6.4f} {brow['Pooled_Prec_%']:6.2f} {brow['Pooled_Rec_%']:6.2f} | "
                  f"{brow['TP_FPN_FPV']:>16s} | {brow['Pat_Mean_AP']:6.4f} {brow['Pat_Mean_F1']:6.4f} | "
                  f"{brow['ZeroF_Spec_%']:7.3f} {brow['FAPH']:6.2f}")
            if mode_tag == "offline_NVF":
                frontiers[f"{vname}__{mtype}"] = front
                if mtype == "Linear_SVM":
                    rec22[vname] = rdet

    print("\n=== DESCRIPTIVE CLINICAL FRONTIER ON INCART-75 (Offline N/V/F) ===")
    for k, fr in frontiers.items():
        if "Linear_SVM" in k:
            f2, f5, f10, mx = fr["FAPH<=2"], fr["FAPH<=5"], fr["FAPH<=10"], fr["Max_Desc_F1"]
            print(f"  {k:38s} | @FAPH<=2: P={f2['Prec_%']:5.1f}% R={f2['Rec_%']:4.1f}% F1={f2['F1']:.3f} | "
                  f"@FAPH<=5: P={f5['Prec_%']:5.1f}% R={f5['Rec_%']:4.1f}% F1={f5['F1']:.3f} | "
                  f"@FAPH<=10: P={f10['Prec_%']:5.1f}% R={f10['Rec_%']:4.1f}% F1={f10['F1']:.3f} | "
                  f"MaxDescF1={mx['F1']:.4f} (P={mx['Prec_%']:.1f}%, R={mx['Rec_%']:.1f}%, FAPH={mx['FAPH']:.1f})")

    out_json = str(_REPO_ROOT / "audits/V33_GREEK_MULTIBEAT_MIT44_INCART75_AUDIT_20261121.json")
    out_md = str(_REPO_ROOT / "audits/V33_GREEK_MULTIBEAT_MIT44_INCART75_AUDIT_20261121.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "gauge_invariance_max_diff": max_diff,
            "cpu_ms_per_beat_incart75": cpu_incart,
            "stage_a_mit44": stage_a_rows,
            "stage_b_incart75": stage_b_rows,
            "offline_frontiers": frontiers,
            "offline_22_f_records_svm": rec22,
        }, f, indent=2)

    lines = [
        "# v33 Classical Greek Geometry + 0-Beat Multi-Beat Context Audit (MIT-BIH 44/44 -> INCART-75 Zero-Tuning)",
        "",
        f"- **Exact O(4) Gauge Invariance:** `{max_diff:.3e}`",
        f"- **INCART-75 CPU Timing (pure Python):** `{cpu_incart}` ms/beat",
        "",
        "## Stage A: MIT-BIH 44/44 (Dev10 LOPO -> Zero-Tuning Hold34)",
        "",
        "| Mode | Variant | Model | P | DevAP | DevF1 | 208/213 | MinorF | H34 FP_N | FP_V | 108/230/105/Dr3 | 0-F Spec |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for r in stage_a_rows:
        lines.append(
            f"| `{r['Mode']}` | {r['Variant']} | {r['Model']} | {r['Params']} | {r['Dev10_AP']:.4f} | "
            f"{r['Dev10_OOF_F1']:.4f} | {r['F1_208']:.3f}/{r['F1_213']:.3f} | {r['MinorF_Rec']} | "
            f"{r['Hold34_FP_N']} | {r['Hold34_FP_V']} | {r['FPN_108_230_105_Dr3']} | {r['Hold34_ZeroF_Spec_%']:.3f}% |"
        )
    lines += [
        "",
        "## Stage B: INCART-75 Zero-Tuning External Validation (32 Patients, 36.57 hrs, 218 True F)",
        "",
        "| Mode | Variant | Model | P | MIT F1 | Pooled AP | Pooled F1 | Prec (%) | Rec (%) | TP/FP_N/FP_V | Pat AP | Pat F1 | 0-F Spec (%) | FAPH |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in stage_b_rows:
        lines.append(
            f"| `{r['Mode']}` | {r['Variant']} | {r['Model']} | {r['Params']} | {r['MIT_OOF_F1']:.4f} | "
            f"**{r['Pooled_AP']:.4f}** | **{r['Pooled_F1']:.4f}** | {r['Pooled_Prec_%']:.2f} | {r['Pooled_Rec_%']:.2f} | "
            f"{r['TP_FPN_FPV']} | {r['Pat_Mean_AP']:.4f} | {r['Pat_Mean_F1']:.4f} | {r['ZeroF_Spec_%']:.3f} | {r['FAPH']:.2f} |"
        )
    lines += [
        "",
        "## Descriptive Clinical Frontier on INCART-75 (Offline N/V/F)",
        "",
        "| Model | @FAPH<=2 (Prec / Rec / F1) | @FAPH<=5 (Prec / Rec / F1) | @FAPH<=10 (Prec / Rec / F1) | Max Desc F1 (Prec / Rec / FAPH) |",
        "|---|---|---|---|---|",
    ]
    for k, fr in frontiers.items():
        f2, f5, f10, mx = fr["FAPH<=2"], fr["FAPH<=5"], fr["FAPH<=10"], fr["Max_Desc_F1"]
        lines.append(
            f"| `{k}` | {f2['Prec_%']:.1f}% / {f2['Rec_%']:.1f}% / {f2['F1']:.3f} | "
            f"{f5['Prec_%']:.1f}% / {f5['Rec_%']:.1f}% / {f5['F1']:.3f} | "
            f"{f10['Prec_%']:.1f}% / {f10['Rec_%']:.1f}% / {f10['F1']:.3f} | "
            f"**{mx['F1']:.4f}** ({mx['Prec_%']:.1f}% / {mx['Rec_%']:.1f}% / {mx['FAPH']:.1f}) |"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {out_json} & {out_md}")


if __name__ == "__main__":
    main()
