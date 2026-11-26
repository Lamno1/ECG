r"""
lab_diagnostic_v53b_lodo_disentanglement_ablation.py
====================================================
STEP 2 CONTROLLED 6-HOSPITAL LODO ABLATION (v53b) ON 6 SOURCE DATABASES
(ZERO LTSTDB PEEKING; SAME STAGE-1; SAME LINEAR CLF FAMILIES SVM & LogR; FIXED F:V BRANCH)

Caches bit-exact Stage-1 internals + d_proj_raw/d_seg_raw ONCE into
`./data_public_ecg/cache_v53_geom_internals_6db` (verified max |diff| < 1e-4 against
cache_gen3_6db), then evaluates the exact 4-way controlled Branch-F:N/S ablation under strict
6-Hospital LODO (251 SubjectAware_251 units: 66 F+ / 185 True 0-F):

  - B0_Baseline_C3      (10D): exact v49 C3 F:N/S feature set (F2_IDX)
  - B1_CompA_Disentangle(13D): replaces confounded (m_bridge, z_N, z_V) with 6 decoupled
                               primitives [log(lam-mu_N), log(1-lam), log1p(e_V_same/0.12),
                               -log(rad_N), -log(rad_V), log(rho_yV)] + 7 unchanged features
  - B2_Add_dProjRaw     (11D): B0 + d_proj_raw = ||P(x - b_N_loc)||_2 (anchor-independent!)
  - B3_A_plus_dProjRaw  (14D): B1 + d_proj_raw

Strict Central Controls:
  1. Deployed T1 under exact R1 source-OOF calibration (no target peeking):
     TP, FP_NS, FP_V, F1, MacroRec (66 F+), W/T/L & StrictDom vs B0,
     MeanFAPH (All / 0-F / % <= 2 / p95), 0-F Dec/Tie/Inc, AP(S_min), per-DB breakdown.
  2. Pure Branch-N threshold-sliding matched-budget curve (holding B0's exact F:V gate
     m_V^(B0) >= 0 FIXED across all 4 configs!) at matched MeanFAPH in {1.0, 1.5, 2.0, 2.5, 3.0}
     AND matched True-0F MeanFAPH in {0.5, 1.0, 1.5}, PLUS per-patient @PatF2 (66 F+ WTL).
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
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import wfdb

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import lab_diagnostic_v36_6hospital_lodo_benchmark as v36
import lab_diagnostic_v46_two_branch_stage2_rr as v46
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_diagnostic_v48_stage2_operating_point_calibration as v48
import lab_audit_v49_subject_unit_and_registry as v49

REG_PATH = str(_REPO_ROOT / "registry/subject_mapping_6db_v1.json")
CACHE_V53 = str(_DATA_ROOT / "cache_v53_geom_internals_6db")
AUD_DIR = str(_REPO_ROOT / "audits")
os.makedirs(CACHE_V53, exist_ok=True)
os.makedirs(AUD_DIR, exist_ok=True)

DB_ORDER = ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]
F2_IDX = v47.F2_IDX       # [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]
IDX_C3_V = v47.IDX_C3     # F2_IDX + [18]


def _map_sym_v10(sym):
    if sym in ("N", "L", "R", "e", "j"):
        return 0
    if sym in ("V", "E"):
        return 1
    if sym == "F":
        return 2
    if sym in ("A", "a", "J", "S"):
        return 3
    return -1


def _extract_cache_worker(args):
    """Bit-exact Stage-1 + online loc_ring internals on one K > 0 source record."""
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
    if abs(fs - v33.TARGET_FS) < 1e-3:
        beats0, beats1, labels, r_peaks = [], [], [], []
        for s, sym in zip(ann.sample, ann.symbol):
            sc = int(s) + delay
            if sc < 93 or sc + 94 > n_sig:
                continue
            c = _map_sym_v10(sym)
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
        for s, sym in zip(ann.sample, ann.symbol):
            sc = int(s) + delay
            if sc < half_win or sc + half_win + 1 > n_sig:
                continue
            c = _map_sym_v10(sym)
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

    d_proj_raw_arr = np.zeros(N, dtype=np.float64)
    d_full_raw_arr = np.zeros(N, dtype=np.float64)
    d_corr_raw_arr = np.zeros(N, dtype=np.float64)
    bn_drift_raw_arr = np.zeros(N, dtype=np.float64)
    prev1, prev2 = joint_all[49], joint_all[48]
    cand_clusters = []

    for t in range(N):
        x_t = post_joint[t]
        c_loc = v33.mean_lead_corr(x_t, b_N_loc)
        d_corr_raw_arr[t] = max(0.0, 1.0 - c_loc)
        dl = x_t - b_N_loc
        d_full_raw_arr[t] = float(np.linalg.norm(dl))
        dl_p = dl - U @ (U.T @ dl)
        d_proj_raw_arr[t] = float(np.linalg.norm(dl_p))
        bn_drift_raw_arr[t] = float(np.linalg.norm(b_N_loc - b_N))

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
    d_seg_raw = eucl_seg_all[idx_n, bk]
    e_V_same = ej_all[idx_n, bk]
    e_V_any = np.min(ej_all, axis=1)
    if K > 1:
        ej_other = ej_all.copy()
        ej_other[idx_n, bk] = np.inf
        e_V_other = np.min(ej_other, axis=1)
    else:
        e_V_other = e_V_same.copy()

    e_N = np.maximum(0.0, (best_lam - muN_vec[bk]) - radN_vec[bk])
    e_Vc = np.maximum(0.0, (1.0 - best_lam) - radV_vec[bk])
    z_N = np.log1p(e_N / radN_vec[bk])
    z_V = np.minimum(np.log1p(e_Vc / radV_vec[bk]), np.log1p(e_V_any / radV_vec[bk]))
    s2 = (z_N > 0.0) & (z_V > 0.0)

    # Compact 16-col float32 matrix per Stage-2 beat:
    #  0: best_lam
    #  1: mu_N
    #  2: rad_N
    #  3: rad_V
    #  4: rho_yV = rad_y_V / norm_q
    #  5: norm_q
    #  6: e_Vc
    #  7: e_V_same
    #  8: e_V_any
    #  9: e_V_other
    # 10: d_proj_raw
    # 11: d_full_raw
    # 12: d_seg_raw
    # 13: d_corr_raw
    # 14: bn_drift_raw
    # 15: eta_0
    G16 = np.column_stack([
        best_lam[s2], muN_vec[bk[s2]], radN_vec[bk[s2]], radV_vec[bk[s2]],
        radyV_vec[bk[s2]] / (nq_vec[bk[s2]] + 1e-12), nq_vec[bk[s2]],
        e_Vc[s2], e_V_same[s2], e_V_any[s2], e_V_other[s2],
        d_proj_raw_arr[s2], d_full_raw_arr[s2], d_seg_raw[s2],
        d_corr_raw_arr[s2], bn_drift_raw_arr[s2], np.full(int(np.sum(s2)), eta_0),
    ]).astype(np.float32)

    return (db_name, rec_id, G16, z_N[s2].astype(np.float32), z_V[s2].astype(np.float32))


def build_or_load_v53_cache(dbs):
    """Builds or loads bit-exact Stage-1 internals cache for all 6 DBs."""
    ready_all = all(
        os.path.exists(os.path.join(CACHE_V53, f"{db}_g16.npz"))
        and os.path.exists(os.path.join(CACHE_V53, f"{db}_meta.json"))
        for db in DB_ORDER
    )
    if not ready_all:
        tasks, rec_lookup = [], {}
        for db in DB_ORDER:
            for it in dbs[db]:
                if len(it["X_s2"]) > 0:
                    tasks.append((db, it["rec_id"]))
                    rec_lookup[(db, it["rec_id"])] = it
        print(f"[v53b] Building one-time bit-exact Stage-1 internals cache for {len(tasks)} K>0 records...", flush=True)
        t0 = time.time()
        by_db = {db: {} for db in DB_ORDER}
        n_workers = min(8, os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for db, rid, G16, zN_chk, zV_chk in ex.map(_extract_cache_worker, tasks):
                it = rec_lookup[(db, rid)]
                assert G16 is not None and len(G16) == len(it["X_s2"])
                assert float(np.max(np.abs(zN_chk - it["X_s2"][:, 1]))) < 1e-4
                assert float(np.max(np.abs(zV_chk - it["X_s2"][:, 2]))) < 1e-4
                assert float(np.max(np.abs(G16[:, 10] / (G16[:, 5] + 1e-12) - it["X_s2"][:, 5]))) < 1e-4
                assert float(np.max(np.abs(G16[:, 12] / (G16[:, 5] + 1e-12) - it["X_s2"][:, 3]))) < 1e-4
                by_db[db][rid] = G16
        for db in DB_ORDER:
            arrs, meta = [], []
            pos = 0
            for it in dbs[db]:
                rid = it["rec_id"]
                if rid in by_db[db]:
                    g = by_db[db][rid]
                    arrs.append(g)
                    meta.append({"rec_id": rid, "s0": pos, "s1": pos + len(g)})
                    pos += len(g)
                else:
                    meta.append({"rec_id": rid, "s0": pos, "s1": pos})
            G_all = np.vstack(arrs) if arrs else np.zeros((0, 16), dtype=np.float32)
            np.savez(os.path.join(CACHE_V53, f"{db}_g16.npz"), G16=G_all)
            with open(os.path.join(CACHE_V53, f"{db}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        print(f"[v53b] Cache built & verified bit-exact in {time.time() - t0:.1f}s -> {CACHE_V53}", flush=True)

    out = {}
    for db in DB_ORDER:
        G_all = np.load(os.path.join(CACHE_V53, f"{db}_g16.npz"))["G16"]
        meta = json.load(open(os.path.join(CACHE_V53, f"{db}_meta.json"), "r", encoding="utf-8"))
        out[db] = {m["rec_id"]: G_all[m["s0"]:m["s1"]].astype(np.float64) for m in meta}
    return out


def augment_dbs_with_v53_features(dbs, g16_by_db):
    """
    Appends 7 disentangled / raw-projection columns to X_s2 (cols 24..30) for clean indexing:
      24: log_lam_gap   = log(max(1e-3, best_lam - mu_N))
      25: log_1m_lam    = log(max(1e-3, 1.0 - best_lam))
      26: log_eV_same   = log1p(e_V_same / 0.12)
      27: neg_log_rad_N = -log(max(1e-6, rad_N))
      28: neg_log_rad_V = -log(max(1e-6, rad_V))
      29: log_rho_yV    = log(max(1e-6, rho_yV))
      30: d_proj_raw    = ||P(x - b_N_loc)||_2
      31: log_norm_q    = log(max(1.0, norm_q))
    """
    out = {}
    for db in DB_ORDER:
        items = []
        for it in dbs[db]:
            c = dict(it)
            G = g16_by_db[db][it["rec_id"]]
            if len(it["X_s2"]) > 0:
                extra = np.column_stack([
                    np.log(np.maximum(1e-3, G[:, 0] - G[:, 1])),
                    np.log(np.maximum(1e-3, 1.0 - G[:, 0])),
                    np.log1p(np.maximum(0.0, G[:, 7]) / 0.12),
                    -np.log(np.maximum(1e-6, G[:, 2])),
                    -np.log(np.maximum(1e-6, G[:, 3])),
                    np.log(np.maximum(1e-6, G[:, 4])),
                    G[:, 10],
                    np.log(np.maximum(1.0, G[:, 5])),
                ])
                c["X_s2"] = np.column_stack([it["X_s2"].astype(np.float64), extra])
            else:
                c["X_s2"] = np.zeros((0, it["X_s2"].shape[1] + 8), dtype=np.float64)
            items.append(c)
        out[db] = items
    return out


# Feature sets for Branch F:N/S (Branch F:V stays FIXED IDENTICAL = IDX_C3_V across ALL configs!)
BASE_REST_7 = [3, 4, 5, 8, 10, 13, 14]  # d_seg, loc_norm_corr, loc_dist_norm, min_lead_bcorr, bcorr_hinge_sq, pyth_margin_loc, min_lead_thales_loc
COMP_A_6 = [24, 25, 26, 27, 28, 29]     # log_lam_gap, log_1m_lam, log_eV_same, neg_log_rad_N, neg_log_rad_V, log_rho_yV

CONFIGS = {
    "B0_Baseline_C3":      F2_IDX,                      # 10D exact v49 C3
    "B1_CompA_Disentangle": COMP_A_6 + BASE_REST_7,     # 13D replaces (m_bridge, z_N, z_V) with 6 decoupled primitives
    "B2_Add_dProjRaw":     F2_IDX + [30],               # 11D B0 + d_proj_raw
    "B3_A_plus_dProjRaw":  COMP_A_6 + BASE_REST_7 + [30], # 14D B1 + d_proj_raw
}
CFG_ORDER = list(CONFIGS.keys())


def fit_lodo_fold_configs(tr_items, mtype, cap_pat_pos=100):
    """
    Fits exact v49 deterministic 5-fold patient-grouped source OOF and full-source classifiers
    for the SHARED F:V branch (C3) and ALL 4 Branch-F:N/S configs (B0..B3), then selects
    joint R1 (tau_N*, tau_V*) strictly on source OOF (exact v48 R1_PooledPriorF1 rule).
    """
    pat_map = {}
    for it in tr_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: (-sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]), p))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    tot_b = sum(len(it["y"]) for it in tr_items)
    tot_f = sum(int(np.sum(it["y"] == 2)) for it in tr_items)

    oof_sV = []
    oof_sN = {cfg: [] for cfg in CFG_ORDER}
    oof_y = []

    for fid in range(n_folds):
        tr_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] == fid]

        XV, yV, wV = v46.build_branch_xy(tr_sub, IDX_C3_V, {1}, cap_pat_pos=cap_pat_pos)
        scV = StandardScaler(); clfV = v46.make_clf(mtype); clfV.fit(scV.fit_transform(XV), yV, sample_weight=wV)

        clfs_N = {}
        for cfg in CFG_ORDER:
            XN, yN, wN = v46.build_branch_xy(tr_sub, CONFIGS[cfg], {0, 3}, cap_pat_pos=cap_pat_pos)
            scN = StandardScaler(); clfN = v46.make_clf(mtype); clfN.fit(scN.fit_transform(XN), yN, sample_weight=wN)
            clfs_N[cfg] = (scN, clfN)

        for te in te_sub:
            if len(te["X_s2"]) > 0:
                oof_sV.append(clfV.decision_function(scV.transform(te["X_s2"][:, IDX_C3_V])))
                for cfg in CFG_ORDER:
                    scN, clfN = clfs_N[cfg]
                    oof_sN[cfg].append(clfN.decision_function(scN.transform(te["X_s2"][:, CONFIGS[cfg]])))
                oof_y.append(te["y"][te["s2_mask"]].astype(np.int8))

    # Full-source fits
    XVa, yVa, wVa = v46.build_branch_xy(tr_items, IDX_C3_V, {1}, cap_pat_pos=cap_pat_pos)
    scV_f = StandardScaler(); clfV_f = v46.make_clf(mtype); clfV_f.fit(scV_f.fit_transform(XVa), yVa, sample_weight=wVa)

    oof_sV_cat = np.concatenate(oof_sV)
    oof_y_cat = np.concatenate(oof_y)
    sigV = float(np.std(oof_sV_cat)) + 1e-9

    out = {"scV": scV_f, "clfV": clfV_f, "sigV": sigV, "configs": {}}
    for cfg in CFG_ORDER:
        XNa, yNa, wNa = v46.build_branch_xy(tr_items, CONFIGS[cfg], {0, 3}, cap_pat_pos=cap_pat_pos)
        scN_f = StandardScaler(); clfN_f = v46.make_clf(mtype); clfN_f.fit(scN_f.fit_transform(XNa), yNa, sample_weight=wNa)
        oof_sN_cat = np.concatenate(oof_sN[cfg])
        sigN = float(np.std(oof_sN_cat)) + 1e-9
        tau_N, tau_V, _ = v46.joint_oof_2d_threshold(oof_sN_cat, oof_sV_cat, (oof_y_cat == 2).astype(int), tot_f, tot_b, n_grid=140)
        out["configs"][cfg] = {
            "scN": scN_f, "clfN": clfN_f,
            "sigN": sigN, "tau_N": float(tau_N), "tau_V": float(tau_V),
            "feat_idx_N": CONFIGS[cfg],
        }
    return out


def eval_at_score_threshold(unit_data, score_key, thr, gate_key=None):
    """Evaluates cohort metrics across all 251 units when alarming iff (score >= thr) [& gate >= 0]."""
    tp_tot, fp_ns_tot, fp_v_tot = 0, 0, 0
    fpos_recs, faph_all, faph_0f = [], [], []
    per_unit = {}
    for sid, ud in unit_data.items():
        tot_f = ud["tot_F"]
        hrs = max(ud["hours"], 1e-6)
        if ud["n_s2"] > 0:
            sc = ud[score_key]
            pred = sc >= thr
            if gate_key is not None:
                pred = pred & (ud[gate_key] >= 0.0)
            y = ud["ys2"]
            tp = int(np.sum(pred & (y == 2)))
            fp_ns = int(np.sum(pred & ((y == 0) | (y == 3))))
            fp_v = int(np.sum(pred & (y == 1)))
        else:
            tp, fp_ns, fp_v = 0, 0, 0
        fp = fp_ns + fp_v
        faph = fp / hrs
        tp_tot += tp; fp_ns_tot += fp_ns; fp_v_tot += fp_v
        faph_all.append(faph)
        if tot_f > 0:
            fpos_recs.append(tp / float(tot_f))
        else:
            faph_0f.append(faph)
        per_unit[sid] = {"tot_F": tot_f, "tp": tp, "fp_ns": fp_ns, "fp_v": fp_v, "fp": fp, "faph": faph, "rec": (tp / float(tot_f)) if tot_f > 0 else None, "db": ud["db"]}

    faph_arr = np.array(faph_all)
    faph_0f_arr = np.array(faph_0f)
    prec = tp_tot / max(1, tp_tot + fp_ns_tot + fp_v_tot)
    rec_pool = tp_tot / max(1, sum(ud["tot_F"] for ud in unit_data.values()))
    f1 = 2 * prec * rec_pool / max(1e-12, prec + rec_pool)
    return {
        "TP": tp_tot, "FP_NS": fp_ns_tot, "FP_V": fp_v_tot,
        "F1": round(f1, 4),
        "MacroRec": round(100.0 * float(np.mean(fpos_recs)), 2),
        "MeanFAPH_All": round(float(np.mean(faph_arr)), 3),
        "MeanFAPH_0F": round(float(np.mean(faph_0f_arr)), 3),
        "Pct_le_2FAPH": round(100.0 * float(np.mean(faph_arr <= 2.0)), 1),
        "p95_FAPH_All": round(float(np.percentile(faph_arr, 95)), 2),
        "p95_FAPH_0F": round(float(np.percentile(faph_0f_arr, 95)), 2),
        "per_unit": per_unit,
    }


def matched_budget_sweep(unit_data, score_key, gate_key, target_faphs, mode="all"):
    """
    Finds the lowest threshold thr on `score_key` (with optional fixed gate `gate_key >= 0`)
    such that MeanFAPH_All <= B (mode='all') or MeanFAPH_0F <= B (mode='0f').
    """
    # Collect all Non-F candidate scores to form exact event thresholds
    ev_scores, ev_weights = [], []
    n_u_all = len(unit_data)
    n_u_0f = sum(1 for ud in unit_data.values() if ud["tot_F"] == 0)
    denom = float(n_u_all if mode == "all" else n_u_0f)

    for sid, ud in unit_data.items():
        if mode == "0f" and ud["tot_F"] > 0:
            continue
        if ud["n_s2"] == 0:
            continue
        mk = (ud["ys2"] != 2)
        if gate_key is not None:
            mk = mk & (ud[gate_key] >= 0.0)
        if np.any(mk):
            ev_scores.append(ud[score_key][mk])
            ev_weights.append(np.full(int(np.sum(mk)), 1.0 / (max(ud["hours"], 1e-6) * denom)))

    s_all = np.concatenate(ev_scores)
    w_all = np.concatenate(ev_weights)
    ord_desc = np.argsort(-s_all)
    s_desc = s_all[ord_desc]
    cum_faph = np.cumsum(w_all[ord_desc])

    res = {}
    for B in target_faphs:
        idx = int(np.searchsorted(cum_faph, B, side="right"))
        if idx == 0:
            thr = float(s_desc[0]) + 1e-6
        elif idx >= len(s_desc):
            thr = float(s_desc[-1]) - 1e-6
        else:
            thr = float(s_desc[idx]) + 1e-9
        ev = eval_at_score_threshold(unit_data, score_key, thr, gate_key=gate_key)
        res[str(B)] = {k: v for k, v in ev.items() if k != "per_unit"}
    return res


def per_patient_patf2_macro(unit_data, score_key, gate_key=None):
    """Per-subject recall at each patient's own <= 2.0 FAPH budget (66 F+ units)."""
    recs = {}
    for sid, ud in unit_data.items():
        if ud["tot_F"] == 0:
            continue
        if ud["n_s2"] == 0:
            recs[sid] = 0.0
            continue
        sc = ud[score_key]
        y = ud["ys2"]
        if gate_key is not None:
            gk = ud[gate_key] >= 0.0
            sc = sc[gk]; y = y[gk]
        if len(y) == 0:
            recs[sid] = 0.0
            continue
        max_fp = int(np.floor(2.0 * ud["hours"]))
        nonf_sc = np.sort(sc[y != 2])[::-1]
        if len(nonf_sc) <= max_fp:
            tp = int(np.sum(y == 2))
        else:
            thr = nonf_sc[max_fp] + 1e-12
            tp = int(np.sum((y == 2) & (sc >= thr)))
        recs[sid] = tp / float(ud["tot_F"])
    return recs


def compare_wtl(pu_new, pu_base):
    w_f, t_f, l_f = 0, 0, 0
    sdom_new, sdom_base = 0, 0
    dec_0f, tie_0f, inc_0f = 0, 0, 0
    for sid in pu_base:
        a, b = pu_new[sid], pu_base[sid]
        if b["tot_F"] > 0:
            if a["rec"] > b["rec"] + 1e-12:
                w_f += 1
            elif a["rec"] < b["rec"] - 1e-12:
                l_f += 1
            else:
                t_f += 1
            # Strict dominance: >= rec AND <= faph with at least one strict
            if (a["rec"] >= b["rec"] - 1e-12) and (a["faph"] <= b["faph"] + 1e-12) and ((a["rec"] > b["rec"] + 1e-12) or (a["faph"] < b["faph"] - 1e-12)):
                sdom_new += 1
            elif (b["rec"] >= a["rec"] - 1e-12) and (b["faph"] <= a["faph"] + 1e-12) and ((b["rec"] > a["rec"] + 1e-12) or (b["faph"] < a["faph"] - 1e-12)):
                sdom_base += 1
        else:
            if a["fp"] < b["fp"]:
                dec_0f += 1
            elif a["fp"] > b["fp"]:
                inc_0f += 1
            else:
                tie_0f += 1
    return {
        "Fpos_WTL": f"{w_f}W/{t_f}T/{l_f}L",
        "StrictDom_Fpos": f"{sdom_new} vs {sdom_base}",
        "ZeroF_DecTieInc": f"{dec_0f}Dec/{tie_0f}Tie/{inc_0f}Inc",
    }


def main():
    t0 = time.time()
    reg = json.load(open(REG_PATH, "r", encoding="utf-8"))
    rec_to_sid = {(r["db"], r["rec_id"]): r["verified_subject_id"] for r in reg["records"]}

    dbs_raw = {db: v47.build_or_load_v47_db(db) for db in DB_ORDER}
    dbs_base = v49.clone_dbs_with_mapping(dbs_raw, rec_to_sid)
    g16_by_db = build_or_load_v53_cache(dbs_base)
    dbs = augment_dbs_with_v53_features(dbs_base, g16_by_db)

    audit = {
        "Audit_Version": "v53b_lodo_disentanglement_ablation",
        "Date": "2026-11-23",
        "Scope": "6 source DBs (306 recs / 251 SubjectAware_251 units), strict 6-hospital LODO, same linear families (SVM + LogR), locked Stage-1 & F:V branch. Zero LTSTDB.",
        "Configs": {k: v for k, v in CONFIGS.items()},
        "Models": {},
    }

    for mtype in ("SVM", "LogR"):
        print(f"\n==================== v53b 6-HOSPITAL LODO ABLATION ({mtype}) ====================", flush=True)
        unit_data = {}
        for db in DB_ORDER:
            for it in dbs[db]:
                sid = it["patient_id"]
                if sid not in unit_data:
                    unit_data[sid] = {
                        "db": db, "tot_F": 0, "hours": 0.0, "ys2": [], "n_s2": 0,
                        **{f"mN_{cfg}": [] for cfg in CFG_ORDER},
                        **{f"mV_{cfg}": [] for cfg in CFG_ORDER},
                        **{f"Smin_{cfg}": [] for cfg in CFG_ORDER},
                    }
                ud = unit_data[sid]
                ud["tot_F"] += int(np.sum(it["y"] == 2))
                ud["hours"] += float(it["post_hours"])

        fold_summary = {}
        for tgt in DB_ORDER:
            tr_items = [it for db in DB_ORDER if db != tgt for it in dbs[db]]
            te_items = dbs[tgt]
            fitted = fit_lodo_fold_configs(tr_items, mtype)
            scV, clfV, sigV = fitted["scV"], fitted["clfV"], fitted["sigV"]

            fold_summary[tgt] = {
                cfg: {
                    "tau_N": round(fitted["configs"][cfg]["tau_N"], 4),
                    "tau_V": round(fitted["configs"][cfg]["tau_V"], 4),
                    "beta_N": [round(float(x), 4) for x in fitted["configs"][cfg]["clfN"].coef_[0]],
                }
                for cfg in CFG_ORDER
            }

            for it in te_items:
                if len(it["X_s2"]) == 0:
                    continue
                sid = it["patient_id"]
                ud = unit_data[sid]
                ys2 = it["y"][it["s2_mask"]].astype(np.int8)
                ud["ys2"].append(ys2)
                sV = clfV.decision_function(scV.transform(it["X_s2"][:, IDX_C3_V]))

                for cfg in CFG_ORDER:
                    cc = fitted["configs"][cfg]
                    sN = cc["clfN"].decision_function(cc["scN"].transform(it["X_s2"][:, cc["feat_idx_N"]]))
                    mN = (sN - cc["tau_N"]) / cc["sigN"]
                    mV = (sV - cc["tau_V"]) / sigV
                    ud[f"mN_{cfg}"].append(mN)
                    ud[f"mV_{cfg}"].append(mV)
                    ud[f"Smin_{cfg}"].append(np.minimum(mN, mV))

        for sid, ud in unit_data.items():
            if ud["ys2"]:
                ud["ys2"] = np.concatenate(ud["ys2"])
                ud["n_s2"] = len(ud["ys2"])
                for cfg in CFG_ORDER:
                    ud[f"mN_{cfg}"] = np.concatenate(ud[f"mN_{cfg}"])
                    ud[f"mV_{cfg}"] = np.concatenate(ud[f"mV_{cfg}"])
                    ud[f"Smin_{cfg}"] = np.concatenate(ud[f"Smin_{cfg}"])

        # Verify exact B0 parity with v49!
        ev_b0 = eval_at_score_threshold(unit_data, "Smin_B0_Baseline_C3", 0.0)
        if mtype == "SVM":
            assert ev_b0["TP"] == 1099 and ev_b0["FP_NS"] == 2058 and ev_b0["FP_V"] == 1406, ev_b0
        else:
            assert ev_b0["TP"] == 1169 and ev_b0["FP_NS"] == 2330 and ev_b0["FP_V"] == 1437, ev_b0
        print(f"  [B0 Parity OK] TP={ev_b0['TP']}, FP_NS={ev_b0['FP_NS']}, FP_V={ev_b0['FP_V']}, MacroRec={ev_b0['MacroRec']}%, MeanFAPH={ev_b0['MeanFAPH_All']}/{ev_b0['MeanFAPH_0F']}", flush=True)

        cfg_results = {}
        patf2_b0 = per_patient_patf2_macro(unit_data, "Smin_B0_Baseline_C3")
        patf2_mN_b0 = per_patient_patf2_macro(unit_data, "mN_B0_Baseline_C3", gate_key="mV_B0_Baseline_C3")

        print("\n  --- 1. DEPLOYED T1 (LOCKED SOURCE-OOF R1) ---")
        for cfg in CFG_ORDER:
            ev = eval_at_score_threshold(unit_data, f"Smin_{cfg}", 0.0)
            wtl = compare_wtl(ev["per_unit"], ev_b0["per_unit"])

            # Pooled AP(S_min) across all 251 units (including missed Stage-1 F via zero-padded sentinel? Use exact Stage-2 AP + per-DB AP)
            all_y_s2 = np.concatenate([ud["ys2"] for ud in unit_data.values() if ud["n_s2"] > 0])
            all_sc = np.concatenate([ud[f"Smin_{cfg}"] for ud in unit_data.values() if ud["n_s2"] > 0])
            ap_s2 = float(average_precision_score((all_y_s2 == 2).astype(int), all_sc))

            # Per-DB breakdown at T1
            per_db = {}
            for db in DB_ORDER:
                sub_u = {s: u for s, u in unit_data.items() if u["db"] == db}
                ev_db = eval_at_score_threshold(sub_u, f"Smin_{cfg}", 0.0)
                ys_db = np.concatenate([u["ys2"] for u in sub_u.values() if u["n_s2"] > 0])
                sc_db = np.concatenate([u[f"Smin_{cfg}"] for u in sub_u.values() if u["n_s2"] > 0])
                ap_db = float(average_precision_score((ys_db == 2).astype(int), sc_db)) if np.any(ys_db == 2) else 0.0
                per_db[db] = {
                    "TP": ev_db["TP"], "FP_NS": ev_db["FP_NS"], "FP_V": ev_db["FP_V"],
                    "MacroRec": ev_db["MacroRec"], "MeanFAPH": ev_db["MeanFAPH_All"],
                    "MeanFAPH_0F": ev_db["MeanFAPH_0F"], "AP_s2": round(ap_db, 4),
                }

            # Per-patient @PatF2
            p2 = per_patient_patf2_macro(unit_data, f"Smin_{cfg}")
            w2 = sum(1 for s in p2 if p2[s] > patf2_b0[s] + 1e-12)
            l2 = sum(1 for s in p2 if p2[s] < patf2_b0[s] - 1e-12)
            t2 = len(p2) - w2 - l2

            p2_mN = per_patient_patf2_macro(unit_data, f"mN_{cfg}", gate_key="mV_B0_Baseline_C3")
            w2n = sum(1 for s in p2_mN if p2_mN[s] > patf2_mN_b0[s] + 1e-12)
            l2n = sum(1 for s in p2_mN if p2_mN[s] < patf2_mN_b0[s] - 1e-12)
            t2n = len(p2_mN) - w2n - l2n

            # Matched-FA sweeps: (a) Joint S_min, (b) Pure m_N under FIXED B0 F:V gate!
            sw_smin_all = matched_budget_sweep(unit_data, f"Smin_{cfg}", None, [1.0, 1.5, 2.0, 2.5, 3.0], mode="all")
            sw_smin_0f = matched_budget_sweep(unit_data, f"Smin_{cfg}", None, [0.5, 1.0, 1.5], mode="0f")
            sw_mN_all = matched_budget_sweep(unit_data, f"mN_{cfg}", "mV_B0_Baseline_C3", [1.0, 1.5, 2.0, 2.5, 3.0], mode="all")
            sw_mN_0f = matched_budget_sweep(unit_data, f"mN_{cfg}", "mV_B0_Baseline_C3", [0.5, 1.0, 1.5], mode="0f")

            cfg_results[cfg] = {
                "T1": {k: v for k, v in ev.items() if k != "per_unit"},
                "vs_B0_T1": wtl,
                "AP_s2_pooled": round(ap_s2, 4),
                "PatF2_Smin_MacroRec": round(100.0 * float(np.mean(list(p2.values()))), 2),
                "PatF2_Smin_WTL": f"{w2}W/{t2}T/{l2}L",
                "PatF2_FixedFV_MacroRec": round(100.0 * float(np.mean(list(p2_mN.values()))), 2),
                "PatF2_FixedFV_WTL": f"{w2n}W/{t2n}T/{l2n}L",
                "per_db_T1": per_db,
                "matched_Smin_AllFAPH": sw_smin_all,
                "matched_Smin_0FFAPH": sw_smin_0f,
                "matched_FixedFV_AllFAPH": sw_mN_all,
                "matched_FixedFV_0FFAPH": sw_mN_0f,
            }
            print(
                f"    {cfg:22s} | TP={ev['TP']:4d} FP_NS={ev['FP_NS']:4d} FP_V={ev['FP_V']:4d} | "
                f"MacroRec={ev['MacroRec']:5.2f}% ({wtl['Fpos_WTL']}, SDom {wtl['StrictDom_Fpos']}) | "
                f"FAPH={ev['MeanFAPH_All']:.3f}/{ev['MeanFAPH_0F']:.3f} (≤2:{ev['Pct_le_2FAPH']}%, p95:{ev['p95_FAPH_All']}) | "
                f"0F:{wtl['ZeroF_DecTieInc']} | AP={ap_s2:.4f} | Pat@F2={cfg_results[cfg]['PatF2_Smin_MacroRec']}%({cfg_results[cfg]['PatF2_Smin_WTL']})"
            )

        print("\n  --- 2. CENTRAL CONTROL: PURE m_N SWEEP UNDER FIXED B0 F:V GATE (Matched MeanFAPH_All) ---")
        for B in ["1.0", "1.5", "2.0", "2.5", "3.0"]:
            line = f"    @AllFAPH<={B}: "
            for cfg in CFG_ORDER:
                r = cfg_results[cfg]["matched_FixedFV_AllFAPH"][B]
                line += f"{cfg[:2]}={r['MacroRec']:5.2f}%(TP={r['TP']:4d},0F={r['MeanFAPH_0F']:.2f},V={r['FP_V']:4d}) | "
            print(line)

        print("\n  --- 3. CENTRAL CONTROL: PURE m_N SWEEP UNDER FIXED B0 F:V GATE (Matched True-0F MeanFAPH) ---")
        for B in ["0.5", "1.0", "1.5"]:
            line = f"    @0FFAPH<={B}: "
            for cfg in CFG_ORDER:
                r = cfg_results[cfg]["matched_FixedFV_0FFAPH"][B]
                line += f"{cfg[:2]}={r['MacroRec']:5.2f}%(TP={r['TP']:4d},All={r['MeanFAPH_All']:.2f},V={r['FP_V']:4d}) | "
            print(line)

        print("\n  --- 4. PER-DB T1 BREAKDOWN ---")
        for db in DB_ORDER:
            line = f"    {db:8s}: "
            for cfg in CFG_ORDER:
                r = cfg_results[cfg]["per_db_T1"][db]
                line += f"{cfg[:2]}[TP={r['TP']:3d},NS={r['FP_NS']:4d},V={r['FP_V']:3d},R={r['MacroRec']:5.1f}%,F={r['MeanFAPH']:5.2f},AP={r['AP_s2']:.3f}] | "
            print(line)

        audit["Models"][mtype] = {
            "fold_weights_and_thresholds": fold_summary,
            "results": cfg_results,
        }

    jpath = os.path.join(AUD_DIR, "V53B_LODO_DISENTANGLEMENT_ABLATION_20261123.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[v53b] Saved full JSON -> {jpath} (total {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
