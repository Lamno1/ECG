r"""
lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle.py
=======================================================
PRE-QRS PHYSIOLOGICAL P-WINDOW [35:63] (-161 to -83 ms) INSIDE THE F:V BRANCH ONLY:
STRICTLY CAUSAL UNLABELED TEMPLATE vs. ORACLE DIAGNOSTIC UPPER-BOUND (LOCKED K=4 BASELINE)

Directly tests the user's pre-registered 3-way stopping tree on top of v46 C3:
  - F:N/S branch FIXED identical across C3, C4_Oracle, C5_Causal (10D pure geometry F2_IDX; ZERO RR, ZERO pre-QRS).
  - ONLY the F:V branch changes:
      1. C3_TwoBranch_FV_SignedRR        : F:V uses 10D geom + log(pre_rr_loc) [v46 baseline]
      2. C4_Oracle_PreQRS_Pwin [ORACLE]  : F:V adds quality-gated pre-QRS [35:63] using CLEAN LABELED tape-wide N template
                                           (DIAGNOSTIC UPPER BOUND ONLY — uses y==0 labels across tape; NOT deployable!)
      3. C5_Causal_Unlabeled_PreQRS_Pwin : F:V adds quality-gated pre-QRS [35:63] using STRICTLY CAUSAL, 100% UNLABELED
                                           25-beat rolling normal buffer initialized from minute-0 unlabeled bn_sub[0]
                                           (ZERO labels, ZERO future beats, ZERO end-of-tape anchors!)
  - Also includes C1_OneHead_Geom (reference) and Fixed-tau_N control (freezing tau_N* from C3 while optimizing only tau_V*
    on source OOF, isolating pure F:V improvement from threshold shifts).

Mandatory Reliability & Mechanism Audits:
  1. Causal Unlabeled Buffer Purity & F-Absorption Audit (post-hoc only!): % True N, % Absorbed F, % V, % S
     inside the past-only buffer at Stage-2 F & V beats, plus template fidelity corr(pN_causal(t), pN_oracle)
     and AF/low-P-quality gate accuracy.
  2. Within-patient caliper-matched cAUC(F vs V | |Delta s_geom| <= 0.25) comparing Oracle vs Causal Unlabeled pre-QRS.
  3. 6-Hospital LODO (both SVM and LogR, C=0.3): AP, DB Rank@F2, 67-patient MacroPat @PatF2 / @C3_FP / @C1_FP with
     exact Win/Tie/Loss counts, deployed T1 (TP, FP_NS, FP_V, F1, FAPH), and F_fail_V_only recovery on LTDB/MIT/INCART!
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
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import lab_diagnostic_v43_hypB2_representation_vs_veto as v43
import lab_diagnostic_v45_conditional_physio_and_error_budget as v45
import lab_diagnostic_v46_two_branch_stage2_rr as v46

AUD_DIR = str(_REPO_ROOT / "audits")
CACHE_V36 = str(_DATA_ROOT / "cache_gen3_6db")
CACHE_V37 = str(_DATA_ROOT / "cache_v37_waveforms_6db")
CACHE_V47 = str(_DATA_ROOT / "cache_v47_preqrs_causal_6db")
os.makedirs(AUD_DIR, exist_ok=True)
os.makedirs(CACHE_V47, exist_ok=True)

DB_ORDER = ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]
F2_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]
# Extended feature matrix X24 for Stage-2 beats:
#   0..17 : exact v36 X18
#   18    : log(pre_rr_loc)  [causal]
#   19    : pq_oracle_pwin_gated  [ORACLE DIAGNOSTIC ONLY]
#   20    : pq_causal_pwin_gated  [STRICTLY CAUSAL, 100% UNLABELED]
#   21    : pq_oracle_pwin_raw    [for diagnostic check]
#   22    : pq_causal_pwin_raw    [for diagnostic check]
#   23    : cons_causal_pwin      [causal unlabeled normal P-consistency]
IDX_C3 = F2_IDX + [18]
IDX_C4_ORACLE = F2_IDX + [18, 19]
IDX_C5_CAUSAL = F2_IDX + [18, 20]
IDX_C4_ORACLE_RAW = F2_IDX + [18, 21]

SL_PWIN = slice(35, 63)   # [-161 ms, -83 ms] clean physiological P-wave window
SL_QRS = v33.QRS_CORE_SLICE
P_BUF_LEN = 25

CLF_FAMILIES = ["SVM", "LogR"]
CFG_ORDER = [
    "C1_OneHead_Geom",
    "C3_TwoBranch_FV_SignedRR",
    "C4_Oracle_PreQRS_Pwin",
    "C5_Causal_Unlabeled_PreQRS_Pwin",
]


def _corr1d_fast(a, b):
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    return float(np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-12))


def _min_qrs_corr(w_a, w_b):
    c0 = _corr1d_fast(w_a[0, SL_QRS], w_b[0, SL_QRS])
    c1 = _corr1d_fast(w_a[1, SL_QRS], w_b[1, SL_QRS])
    return min(c0, c1)


def _mean_2lead_corr(w_a, w_b):
    return 0.5 * (_corr1d_fast(w_a[0], w_b[0]) + _corr1d_fast(w_a[1], w_b[1]))


def _mean_pwin_corr(p_a, p_b):
    return 0.5 * (_corr1d_fast(p_a[0], p_b[0]) + _corr1d_fast(p_a[1], p_b[1]))


def quality_gate(cons_val):
    """Smooth unlabeled quality ramp [0.30, 0.50] -> [0, 1] (mutes AF/noisy-P tapes)."""
    return float(np.clip((cons_val - 0.30) / 0.20, 0.0, 1.0))


def _process_rec_v47(args):
    """
    Computes BOTH Oracle (labeled tape-wide N) AND Strictly Causal Unlabeled (past-only
    25-beat rolling normal buffer initialized from minute-0 bn_sub[0]) pre-QRS [35:63] features,
    plus exact buffer purity & F-absorption telemetry!
    """
    db_name, m36, m37 = args
    rid = m36["rec_id"]
    d36 = np.load(os.path.join(CACHE_V36, f"{db_name}_gen3_4cls.npz"))
    rr3_all = np.load(os.path.join(CACHE_V37, f"{db_name}_rr3.npy"), mmap_mode="r")
    bts_all = np.load(os.path.join(CACHE_V37, f"{db_name}_beats.npy"), mmap_mode="r")
    bn_sub_all = np.load(os.path.join(CACHE_V37, f"{db_name}_bn_sub.npy"), mmap_mode="r")

    b0, b1 = m36["b0"], m36["b1"]
    N = b1 - b0
    y = d36["y_all"][b0:b1]
    s2_mask = d36["s2_all"][b0:b1]
    X18_s2 = d36["X18_s2_all"][m36["s0"]:m36["s1"]].astype(np.float64)
    rr3 = np.array(rr3_all[m37["b0"]:m37["b1"]], dtype=np.float64)
    pre_rr = rr3[:, 1]
    w0 = m37["w0"]
    # padded_beats: w0 is beat 49 (prev1 for t=0), w0+1..w0+N are beats 50..50+N-1!
    W_pad = np.array(bts_all[w0 : w0 + 1 + N], dtype=np.float64)  # (N+1, 2, 187)
    W_center = W_pad[1:]                                           # (N, 2, 187)
    bN_init = np.array(bn_sub_all[m37["bn0"]], dtype=np.float64)   # exact minute-0 unlabeled b_N_prefix (2, 187)

    # 1. ORACLE TEMPLATE (DIAGNOSTIC UPPER BOUND ONLY — uses y==0 across tape!)
    n_idx = np.where(y == 0)[0]
    if len(n_idx) >= 10:
        sub_n = n_idx[np.linspace(0, len(n_idx) - 1, min(500, len(n_idx)), dtype=int)]
        seg_n = W_center[sub_n, :, SL_PWIN]
        seg_n = seg_n - np.mean(seg_n, axis=2, keepdims=True)
        pN_ora = np.median(seg_n, axis=0)  # (2, 28)
        ora_c_n = v45.corr_rows_2lead(W_center[sub_n, :, SL_PWIN], pN_ora[0], pN_ora[1])
        cons_ora = float(np.median(ora_c_n))
        has_ora = True
    else:
        pN_ora = np.zeros((2, 28), dtype=np.float64)
        cons_ora = 0.0
        has_ora = False
    qgate_ora = quality_gate(cons_ora)

    # 2. STRICTLY CAUSAL, 100% UNLABELED ROLLING NORMAL PRE-QRS BUFFER (ZERO labels, ZERO future beats!)
    # Precompute strictly causal adjacent beat correlations (x_t vs x_{t-1} and x_{t-1} vs x_{t-2}) vectorized!
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

    # Pre-center P-windows for all beats once
    P_all = W_center[:, :, SL_PWIN]
    P_all = P_all - np.mean(P_all, axis=2, keepdims=True)

    # Initialize causal buffer with minute-0 unlabeled template bN_init (strictly from beats 0..49!)
    w_ring = np.zeros((P_BUF_LEN, 2, 187), dtype=np.float64)
    p_ring = np.zeros((P_BUF_LEN, 2, 28), dtype=np.float64)
    y_ring = np.full(P_BUF_LEN, -1, dtype=np.int8)  # POST-HOC AUDIT ONLY — NEVER READ BY ALGORITHM!
    w_ring[0] = bN_init
    p_init = bN_init[:, SL_PWIN] - np.mean(bN_init[:, SL_PWIN], axis=1, keepdims=True)
    p_ring[0] = p_init
    y_ring[0] = 0  # prefix seed placeholder (excluded from post-hoc admitted-beat purity count)
    buf_cnt = 1
    buf_ptr = 1 % P_BUF_LEN
    n_admitted = 0

    bN_cur = bN_init.copy()
    pN_cur = p_init.copy()
    cons_cur = 0.0  # stays 0 until >= 5 unlabeled past beats admitted (prevents false confidence at t=0!)

    s2_indices = np.where(s2_mask)[0]
    n_s2 = len(s2_indices)
    s2_pos = 0
    next_s2_t = int(s2_indices[0]) if n_s2 > 0 else -1

    pq_ora_raw = np.zeros(n_s2, dtype=np.float64)
    pq_cau_raw = np.zeros(n_s2, dtype=np.float64)
    cons_cau_arr = np.zeros(n_s2, dtype=np.float64)
    qgate_cau_arr = np.zeros(n_s2, dtype=np.float64)

    # Post-hoc audit accumulators (strictly at Stage-2 F & V beats + all admitted beats!)
    adm_y_counts = Counter()
    s2f_buf_n = []; s2f_buf_f = []; s2f_buf_v = []; s2f_fid = []
    s2v_buf_n = []; s2v_buf_f = []; s2v_buf_v = []; s2v_fid = []
    s2f_tot = 0; s2f_unwarmed = 0

    reg_rr_all = (v33.LOC_RR_MIN <= pre_rr) & (pre_rr <= v33.LOC_RR_MAX)

    for t in range(N):
        # STEP A (STRICTLY CAUSAL): Score beat t using ONLY past buffer pN_cur BEFORE inspecting beat t for admission!
        if t == next_s2_t:
            pt = P_all[t]
            pq_ora_raw[s2_pos] = _mean_pwin_corr(pt, pN_ora) if has_ora else 0.0
            pq_cau_raw[s2_pos] = _mean_pwin_corr(pt, pN_cur) if n_admitted >= 5 else 0.0
            cons_cau_arr[s2_pos] = cons_cur if n_admitted >= 5 else 0.0
            qgate_cau_arr[s2_pos] = quality_gate(cons_cur) if n_admitted >= 5 else 0.0

            if y[t] == 2:
                s2f_tot += 1
                if n_admitted < 5:
                    s2f_unwarmed += 1

            # Post-hoc purity & fidelity audit of current buffer at Stage-2 F & V beats!
            if n_admitted >= 5 and y[t] in (1, 2):
                active_y = y_ring[:buf_cnt]
                valid_y = active_y[active_y >= 0]
                if len(valid_y) > 0:
                    fn = float(np.mean(valid_y == 0))
                    ff = float(np.mean(valid_y == 2))
                    fv = float(np.mean(valid_y == 1))
                    fid = _mean_pwin_corr(pN_cur, pN_ora) if has_ora else 0.0
                    if y[t] == 2:
                        s2f_buf_n.append(fn); s2f_buf_f.append(ff); s2f_buf_v.append(fv); s2f_fid.append(fid)
                    else:
                        s2v_buf_n.append(fn); s2v_buf_f.append(ff); s2v_buf_v.append(fv); s2v_fid.append(fid)

            s2_pos += 1
            next_s2_t = int(s2_indices[s2_pos]) if s2_pos < n_s2 else -1

        # STEP B (STRICTLY CAUSAL UNLABELED ADMISSION): Decide whether beat t enters rolling normal buffer!
        if reg_rr_all[t]:
            c_p1 = c_prev1_all[t]
            c_p12 = c_prev12_all[t]
            # Fast pre-check before computing correlation with bN_cur
            if c_p1 >= 0.80:
                xt = W_center[t]
                qrs_c = _min_qrs_corr(xt, bN_cur)
                if qrs_c >= 0.85:
                    c_bn = _mean_2lead_corr(xt, bN_cur)
                    if (c_bn >= 0.80 and c_p1 >= 0.80) or (c_p1 >= 0.90 and c_p12 >= 0.90):
                        # Admit beat t!
                        if n_admitted == 0:
                            # Overwrite initial seed slot 0 once real beats start arriving
                            buf_ptr = 0; buf_cnt = 0
                        w_ring[buf_ptr] = xt
                        p_ring[buf_ptr] = P_all[t]
                        y_ring[buf_ptr] = y[t]  # audit only
                        buf_ptr = (buf_ptr + 1) % P_BUF_LEN
                        if buf_cnt < P_BUF_LEN:
                            buf_cnt += 1
                        n_admitted += 1
                        adm_y_counts[int(y[t])] += 1

                        # Refresh median template & consistency every 5 admissions (or while warming up < 25)
                        if buf_cnt < P_BUF_LEN or (n_admitted % 5 == 0):
                            bN_cur = np.median(w_ring[:buf_cnt], axis=0)
                            pN_cur = np.median(p_ring[:buf_cnt], axis=0)
                            if buf_cnt >= 5:
                                cs = v45.corr_rows_2lead(p_ring[:buf_cnt], pN_cur[0], pN_cur[1])
                                cons_cur = float(np.median(cs))

    # Assemble X24_s2
    log_rr_s2 = np.log(np.clip(pre_rr[s2_mask], 0.2, 3.0))
    pq_ora_gated = qgate_ora * pq_ora_raw
    pq_cau_gated = qgate_cau_arr * pq_cau_raw
    X24_s2 = np.column_stack([
        X18_s2, log_rr_s2, pq_ora_gated, pq_cau_gated, pq_ora_raw, pq_cau_raw, cons_cau_arr
    ]).astype(np.float32) if n_s2 > 0 else np.zeros((0, 24), dtype=np.float32)

    tot_adm = sum(adm_y_counts.values())
    buf_audit = {
        "rec_id": rid, "patient_id": m36["patient_id"], "db": db_name,
        "tot_F": int(np.sum(y == 2)), "s2_F": int(np.sum((y == 2) & s2_mask)),
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
    return {
        "db": db_name, "rec_id": rid, "patient_id": m36["patient_id"],
        "post_hours": m36["post_hours"], "K": m36["K"],
        "y": y, "s2_mask": s2_mask, "X24_s2": X24_s2, "buf_audit": buf_audit,
    }


def build_or_load_v47_db(db_name):
    meta_p = os.path.join(CACHE_V47, f"{db_name}_v47_meta.json")
    npz_p = os.path.join(CACHE_V47, f"{db_name}_v47.npz")
    if os.path.exists(meta_p) and os.path.exists(npz_p):
        meta = json.load(open(meta_p, "r", encoding="utf-8"))
        dat = np.load(npz_p)
        items = []
        for m in meta:
            items.append({
                "db": m["db"], "rec_id": m["rec_id"], "patient_id": m["patient_id"],
                "post_hours": m["post_hours"], "K": m["K"], "buf_audit": m["buf_audit"],
                "y": dat["y_all"][m["b0"]:m["b1"]],
                "s2_mask": dat["s2_all"][m["b0"]:m["b1"]],
                "X_s2": dat["X24_all"][m["s0"]:m["s1"]].astype(np.float64),
            })
        return items

    m36 = json.load(open(os.path.join(CACHE_V36, f"{db_name}_gen3_4cls_meta.json"), "r", encoding="utf-8"))
    m37 = json.load(open(os.path.join(CACHE_V37, f"{db_name}_meta.json"), "r", encoding="utf-8"))
    m37_map = {m["rec_id"]: m for m in m37}
    tasks = [(db_name, m, m37_map[m["rec_id"]]) for m in m36]
    nw = 4 if db_name in ("ltdb", "sddb") else 6
    t0 = time.time()
    print(f"Building v47 causal+oracle pre-QRS cache on {db_name:8s} ({len(tasks)} recs, {nw} workers)...", flush=True)
    raw = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for res in ex.map(_process_rec_v47, tasks):
            raw.append(res)

    y_l, s2_l, x_l, meta = [], [], [], []
    bc = sc = 0
    for it in raw:
        nb, ns = len(it["y"]), len(it["X24_s2"])
        y_l.append(it["y"]); s2_l.append(it["s2_mask"]); x_l.append(it["X24_s2"])
        meta.append({
            "db": it["db"], "rec_id": it["rec_id"], "patient_id": it["patient_id"],
            "post_hours": it["post_hours"], "K": it["K"], "buf_audit": it["buf_audit"],
            "b0": bc, "b1": bc + nb, "s0": sc, "s1": sc + ns,
        })
        bc += nb; sc += ns
    np.savez_compressed(npz_p, y_all=np.concatenate(y_l), s2_all=np.concatenate(s2_l), X24_all=np.vstack(x_l))
    json.dump(meta, open(meta_p, "w", encoding="utf-8"))
    print(f"  DONE {db_name:8s}: {bc:,} beats in {time.time()-t0:.1f}s", flush=True)
    return build_or_load_v47_db(db_name)


def fit_twobranch_fixed_tauN_check(tr_items, feat_idx_N, feat_idx_V, mtype, fixed_tau_N, cap_pat_pos=100):
    """Control check: freezes tau_N* from C3 and optimizes ONLY tau_V* on source OOF!"""
    pat_map = {}
    for it in tr_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: -sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    oof_sN, oof_sV, oof_y = [], [], []
    tot_b = sum(len(it["y"]) for it in tr_items)
    tot_f = sum(int(np.sum(it["y"] == 2)) for it in tr_items)
    for fid in range(n_folds):
        tr_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] == fid]
        XN, yN, wN = v46.build_branch_xy(tr_sub, feat_idx_N, {0, 3}, cap_pat_pos=cap_pat_pos)
        XV, yV, wV = v46.build_branch_xy(tr_sub, feat_idx_V, {1}, cap_pat_pos=cap_pat_pos)
        if np.sum(yN == 1) == 0 or np.sum(yV == 1) == 0: continue
        scN = StandardScaler(); clfN = v46.make_clf(mtype); clfN.fit(scN.fit_transform(XN), yN, sample_weight=wN)
        scV = StandardScaler(); clfV = v46.make_clf(mtype); clfV.fit(scV.fit_transform(XV), yV, sample_weight=wV)
        for te in te_sub:
            if len(te["X_s2"]) > 0:
                oof_sN.append(clfN.decision_function(scN.transform(te["X_s2"][:, feat_idx_N])))
                oof_sV.append(clfV.decision_function(scV.transform(te["X_s2"][:, feat_idx_V])))
                oof_y.append((te["y"][te["s2_mask"]] == 2).astype(int))
    sN = np.concatenate(oof_sN); sV = np.concatenate(oof_sV); yb = np.concatenate(oof_y)
    passN = (sN >= fixed_tau_N)
    sV_p = sV[passN]; yb_p = yb[passN]
    ord_v = np.argsort(sV_p)[::-1]
    tp_c = np.cumsum(yb_p[ord_v] == 1).astype(float); fp_c = np.cumsum(yb_p[ord_v] == 0).astype(float)
    pi_src = float(tot_f) / max(1.0, float(tot_b))
    neg_mult = (pi_src / max(1e-9, 1.0 - pi_src)) / (0.001 / 0.999)
    tau_V = float(sV_p[ord_v[int(np.argmax(2.0 * tp_c / (float(tot_f) + tp_c + fp_c * neg_mult + 1e-12)))]])
    sig_N = float(np.std(sN)) + 1e-9; sig_V = float(np.std(sV)) + 1e-9

    XN_all, yN_all, wN_all = v46.build_branch_xy(tr_items, feat_idx_N, {0, 3}, cap_pat_pos=cap_pat_pos)
    scN_f = StandardScaler(); clfN_f = v46.make_clf(mtype); clfN_f.fit(scN_f.fit_transform(XN_all), yN_all, sample_weight=wN_all)
    XV_all, yV_all, wV_all = v46.build_branch_xy(tr_items, feat_idx_V, {1}, cap_pat_pos=cap_pat_pos)
    scV_f = StandardScaler(); clfV_f = v46.make_clf(mtype); clfV_f.fit(scV_f.fit_transform(XV_all), yV_all, sample_weight=wV_all)
    return {
        "mode": "twobranch", "feat_idx_N": feat_idx_N, "feat_idx_V": feat_idx_V,
        "scN": scN_f, "clfN": clfN_f, "tau_N": fixed_tau_N, "sig_N": sig_N,
        "scV": scV_f, "clfV": clfV_f, "tau_V": tau_V, "sig_V": sig_V,
    }


def main():
    t0 = time.time()
    dbs = {db: build_or_load_v47_db(db) for db in DB_ORDER}

    # =========================================================================
    # AUDIT A: CAUSAL UNLABELED BUFFER PURITY, F-ABSORPTION & SIGNAL FIDELITY
    # =========================================================================
    print("\n=== AUDIT A: CAUSAL UNLABELED PRE-QRS BUFFER RELIABILITY & F-ABSORPTION ===", flush=True)
    buf_summary_by_db = {}
    spotlight_buf = {}
    spot_ids = {"14134", "15814", "14046", "14157", "208", "213", "233", "223", "210", "e0605", "e0614", "31", "49", "46", "I75"}
    for db in DB_ORDER:
        auds = [it["buf_audit"] for it in dbs[db]]
        f_auds = [a for a in auds if a["s2F_buf_N_pct"] is not None]
        buf_summary_by_db[db] = {
            "n_recs": len(auds),
            "all_adm_N_mean_%": round(float(np.mean([a["adm_N_pct"] for a in auds])), 2),
            "all_adm_F_mean_%": round(float(np.mean([a["adm_F_pct"] for a in auds])), 2),
            "all_adm_V_mean_%": round(float(np.mean([a["adm_V_pct"] for a in auds])), 2),
            "s2F_recs": len(f_auds),
            "s2F_buf_N_mean_%": round(float(np.mean([a["s2F_buf_N_pct"] for a in f_auds])), 1) if f_auds else None,
            "s2F_buf_F_mean_%": round(float(np.mean([a["s2F_buf_F_pct"] for a in f_auds])), 1) if f_auds else None,
            "s2F_buf_F_max_%": round(float(np.max([a["s2F_buf_F_pct"] for a in f_auds])), 1) if f_auds else None,
            "s2F_tmpl_fid_med": round(float(np.median([a["s2F_tmpl_fid"] for a in f_auds])), 3) if f_auds else None,
        }
        print(f"  {db:8s}: {buf_summary_by_db[db]}", flush=True)
        for a in auds:
            if a["rec_id"] in spot_ids:
                spotlight_buf[f"{db}_{a['rec_id']}"] = a

    print("\n  Spotlight Buffer Purity & F-Absorption (at Stage-2 F beats):", flush=True)
    for k in sorted(spotlight_buf.keys()):
        a = spotlight_buf[k]
        print(
            f"    {k:14s} (S2_F={a['s2_F']:3d}) | Buf@F: N={a['s2F_buf_N_pct']}% F_abs={a['s2F_buf_F_pct']}% V={a['s2F_buf_V_pct']}% | "
            f"TmplFid={a['s2F_tmpl_fid']} | Cons ora={a['cons_ora']:+.2f} vs cau={a['cons_cau_med']:+.2f}",
            flush=True,
        )

    # =========================================================================
    # AUDIT B: WITHIN-PATIENT MATCHED-GEOMETRY cAUC GAP (ORACLE vs CAUSAL UNLABELED)
    # =========================================================================
    print("\n=== AUDIT B: CONDITIONAL cAUC(F vs V | |Delta s_geom| <= 0.25) — ORACLE vs CAUSAL UNLABELED ===", flush=True)
    # Compute pure-geometry F2 LODO score s_geom
    pat_s2 = {}
    for tgt in DB_ORDER:
        tr = [it for db in DB_ORDER if db != tgt for it in dbs[db]]
        m_f2 = v46.fit_onehead_lodo(tr, F2_IDX, "SVM")
        for it in dbs[tgt]:
            if len(it["X_s2"]) == 0: continue
            s_g = m_f2["clf"].decision_function(m_f2["scaler"].transform(it["X_s2"][:, F2_IDX]))
            pd = pat_s2.setdefault(it["patient_id"], {
                "db": tgt, "y": [], "sg": [], "ora_raw": [], "cau_raw": [], "ora_g": [], "cau_g": [],
                "cons_c": [], "cons_ora_list": [],
            })
            pd["y"].append(it["y"][it["s2_mask"]]); pd["sg"].append(s_g)
            pd["ora_raw"].append(it["X_s2"][:, 21]); pd["cau_raw"].append(it["X_s2"][:, 22])
            pd["ora_g"].append(it["X_s2"][:, 19]); pd["cau_g"].append(it["X_s2"][:, 20])
            pd["cons_c"].append(it["X_s2"][:, 23])
            pd["cons_ora_list"].append(it["buf_audit"]["cons_ora"])

    cauc_gap = {}
    cauc_per_pat = {}
    for p, pd in pat_s2.items():
        y = np.concatenate(pd["y"]); sg = np.concatenate(pd["sg"])
        if int(np.sum(y == 2)) == 0 or int(np.sum(y == 1)) == 0:
            continue
        row = {"db": pd["db"], "cons_ora": round(float(np.median(pd["cons_ora_list"])), 3),
               "cons_cau": round(float(np.median(np.concatenate(pd["cons_c"]))), 3)}
        ok = False
        for fk in ("ora_raw", "cau_raw", "ora_g", "cau_g"):
            x = np.concatenate(pd[fk])
            c, _, _, _ = v45.caliper_matched_concordance(sg[y == 2], x[y == 2], sg[y == 1], x[y == 1], caliper=0.25)
            if c is not None:
                row[fk] = round(float(c), 3)
                ok = True
        if ok:
            cauc_per_pat[p] = row

    def _agg_cauc(pats_dict, fk):
        vals = [r[fk] for r in pats_dict.values() if fk in r]
        if not vals:
            return {"n_pats": 0, "pat_med": None, "iqr": [None, None], "ge60": 0, "le40": 0}
        return {
            "n_pats": len(vals), "pat_med": round(float(np.median(vals)), 3),
            "iqr": [round(float(np.percentile(vals, 25)), 3), round(float(np.percentile(vals, 75)), 3)],
            "ge60": sum(1 for v in vals if v >= 0.60),
            "le40": sum(1 for v in vals if v <= 0.40),
        }

    hi_p = {p: r for p, r in cauc_per_pat.items() if r["cons_ora"] >= 0.40}
    lo_p = {p: r for p, r in cauc_per_pat.items() if r["cons_ora"] < 0.40}
    for fk in ("ora_raw", "cau_raw", "ora_g", "cau_g"):
        cauc_gap[fk] = _agg_cauc(cauc_per_pat, fk)
        cauc_gap[f"{fk}_hiP"] = _agg_cauc(hi_p, fk)
        cauc_gap[f"{fk}_loP"] = _agg_cauc(lo_p, fk)
        g, gh, gl = cauc_gap[fk], cauc_gap[f"{fk}_hiP"], cauc_gap[f"{fk}_loP"]
        print(
            f"  {fk:10s}: ALL({g['n_pats']}p)={g['pat_med']:.3f} [{g['ge60']}/{g['le40']}] | "
            f"HighP({gh['n_pats']}p)={gh['pat_med']:.3f} [{gh['ge60']}/{gh['le40']}] | "
            f"LowP({gl['n_pats']}p)={gl['pat_med']:.3f} [{gl['ge60']}/{gl['le40']}]",
            flush=True,
        )

    # =========================================================================
    # AUDIT C: MAIN 6-HOSPITAL LODO COMPARISON (C1, C3, C4_Oracle, C5_Causal)
    # =========================================================================
    summary = {
        "AuditA_Buffer_ByDB": buf_summary_by_db,
        "AuditA_Buffer_Spotlights": spotlight_buf,
        "AuditB_cAUC_Gap_Oracle_vs_Causal": cauc_gap,
        "AuditB_cAUC_PerPatient": cauc_per_pat,
        "Models": {},
    }

    for mtype in CLF_FAMILIES:
        summary["Models"][mtype] = {"ByDB": {}, "Global_WTL_67Pats": {}}
        print(f"\n==================== CLASSIFIER FAMILY: {mtype} (C=0.3, K=4 BASELINE, FIXED F:NS) ====================", flush=True)
        all_rf2 = {c: {} for c in CFG_ORDER + ["C4b_Oracle_Raw"]}
        all_rc3 = {c: {} for c in CFG_ORDER + ["C4b_Oracle_Raw"]}
        all_rc1 = {c: {} for c in CFG_ORDER + ["C4b_Oracle_Raw"]}

        for tgt in DB_ORDER:
            tr = [it for db in DB_ORDER if db != tgt for it in dbs[db]]
            te = dbs[tgt]

            m_c1 = v46.fit_onehead_lodo(tr, F2_IDX, mtype)
            m_c3 = v46.fit_twobranch_lodo(tr, F2_IDX, IDX_C3, mtype)
            m_c4 = v46.fit_twobranch_lodo(tr, F2_IDX, IDX_C4_ORACLE, mtype)
            m_c5 = v46.fit_twobranch_lodo(tr, F2_IDX, IDX_C5_CAUSAL, mtype)
            m_c4_raw = v46.fit_twobranch_lodo(tr, F2_IDX, IDX_C4_ORACLE_RAW, mtype)

            # Fixed-tau_N controls (keeping C3's exact tau_N* frozen!)
            m_c4_fixN = fit_twobranch_fixed_tauN_check(tr, F2_IDX, IDX_C4_ORACLE, mtype, m_c3["tau_N"])
            m_c5_fixN = fit_twobranch_fixed_tauN_check(tr, F2_IDX, IDX_C5_CAUSAL, mtype, m_c3["tau_N"])

            _, c1_pfp, c1_rf2, _ = v46.score_and_eval_v46(te, m_c1)
            _, c3_pfp, c3_rf2, _ = v46.score_and_eval_v46(te, m_c3)
            ev3, _, c3_rf2, c3_rc3 = v46.score_and_eval_v46(te, m_c3, ref_pat_fp=c3_pfp, ref_pat_rf2=c3_rf2)

            ev_map = {}
            for cname, mobj in [
                ("C1_OneHead_Geom", m_c1),
                ("C3_TwoBranch_FV_SignedRR", m_c3),
                ("C4_Oracle_PreQRS_Pwin", m_c4),
                ("C5_Causal_Unlabeled_PreQRS_Pwin", m_c5),
                ("C4b_Oracle_Raw", m_c4_raw),
            ]:
                ev, _, rf2, rc3 = v46.score_and_eval_v46(te, mobj, ref_pat_fp=c3_pfp, ref_pat_rf2=c3_rf2, ref_pat_rc1fp=c3_rc3)
                _, _, _, rc1 = v46.score_and_eval_v46(te, mobj, ref_pat_fp=c1_pfp)
                ev["MacroPat_AtC3FP_%"] = ev["MacroPat_AtC1FP_%"]
                ev["WTL_Pat@F2_vs_C3"] = ev["WTL_Pat@F2_vs_C1"]
                ev["WTL_AtC3FP_vs_C3"] = ev["WTL_AtC1FP_vs_C1"]
                ev["MacroPat_AtC1FP_%"] = round(float(np.mean(list(rc1.values()))), 2) if rc1 else 0.0
                ev_map[cname] = ev
                all_rf2[cname].update(rf2); all_rc3[cname].update(rc3); all_rc1[cname].update(rc1)

            ev4_fn, _, _, _ = v46.score_and_eval_v46(te, m_c4_fixN, ref_pat_fp=c3_pfp, ref_pat_rf2=c3_rf2, ref_pat_rc1fp=c3_rc3)
            ev5_fn, _, _, _ = v46.score_and_eval_v46(te, m_c5_fixN, ref_pat_fp=c3_pfp, ref_pat_rf2=c3_rf2, ref_pat_rc1fp=c3_rc3)

            w_c4 = m_c4["clfV"].coef_.ravel(); w_c5 = m_c5["clfV"].coef_.ravel(); w_c4r = m_c4_raw["clfV"].coef_.ravel()
            summary["Models"][mtype]["ByDB"][tgt] = {
                "Eval": {k: ev_map[k] for k in CFG_ORDER},
                "Ungated_Oracle_Control": ev_map["C4b_Oracle_Raw"],
                "FixedTauN_Control": {"C4_Oracle_FixN": ev4_fn, "C5_Causal_FixN": ev5_fn},
                "Weights_FV": {
                    "C4_w_rr": round(float(w_c4[-2]), 3), "C4_w_pq": round(float(w_c4[-1]), 3),
                    "C5_w_rr": round(float(w_c5[-2]), 3), "C5_w_pq": round(float(w_c5[-1]), 3),
                    "C4raw_w_rr": round(float(w_c4r[-2]), 3), "C4raw_w_pq": round(float(w_c4r[-1]), 3),
                },
                "Thresholds": {
                    "C3": [round(m_c3["tau_N"], 4), round(m_c3["tau_V"], 4)],
                    "C4": [round(m_c4["tau_N"], 4), round(m_c4["tau_V"], 4)],
                    "C5": [round(m_c5["tau_N"], 4), round(m_c5["tau_V"], 4)],
                },
            }

            print(f"\n  [{tgt}] ({mtype}) | C4 w=(rr:{w_c4[-2]:+.2f}, pq:{w_c4[-1]:+.2f}) | C5 w=(rr:{w_c5[-2]:+.2f}, pq:{w_c5[-1]:+.2f}) | C4raw pq:{w_c4r[-1]:+.2f}", flush=True)
            for cname in CFG_ORDER:
                ev = ev_map[cname]
                fv_only = ev["Branch_Diag"].get("F_fail_V_only", "—")
                print(
                    f"    {cname:31s} | AP={ev['AP']:.4f} DB@F2={ev['DB_Rank@F2_%']:4.1f}% "
                    f"Pat@F2={ev['MacroPat_Pat@F2_%']:4.1f}%({ev['WTL_Pat@F2_vs_C3']}) "
                    f"@C3FP={ev['MacroPat_AtC3FP_%']:4.1f}%({ev['WTL_AtC3FP_vs_C3']}) | "
                    f"T1: F1={ev['T1_F1']:.4f} [{ev['T1_Counts']:16s}] FP_NS={ev['T1_FP_NS']:4d} FP_V={ev['T1_FPV']:4d} FAPH={ev['T1_FAPH']:5.2f} | F_failV={fv_only}",
                    flush=True,
                )
            ev4r = ev_map["C4b_Oracle_Raw"]
            print(
                f"      [Controls] C4_FixN: F1={ev4_fn['T1_F1']:.4f}[{ev4_fn['T1_Counts']}] AP={ev4_fn['AP']:.4f} | "
                f"C5_FixN: F1={ev5_fn['T1_F1']:.4f}[{ev5_fn['T1_Counts']}] AP={ev5_fn['AP']:.4f} | "
                f"C4_Raw: F1={ev4r['T1_F1']:.4f}[{ev4r['T1_Counts']}] AP={ev4r['AP']:.4f} Pat@F2={ev4r['MacroPat_Pat@F2_%']:.1f}%",
                flush=True,
            )

        def global_wtl(ma, mb):
            w = sum(1 for p in ma if ma[p] > mb[p] + 1e-6)
            l = sum(1 for p in ma if ma[p] < mb[p] - 1e-6)
            t = len(ma) - w - l
            return {"W": w, "T": t, "L": l, "mean_a": round(float(np.mean(list(ma.values()))), 2),
                    "mean_b": round(float(np.mean(list(mb.values()))), 2), "str": f"{w}W/{t}T/{l}L"}

        print(f"\n  === GLOBAL 67-PATIENT SUMMARY ({mtype}) ===", flush=True)
        for tag, ca, cb in [
            ("C4_Oracle_vs_C3     ", "C4_Oracle_PreQRS_Pwin", "C3_TwoBranch_FV_SignedRR"),
            ("C4b_OracleRaw_vs_C3 ", "C4b_Oracle_Raw", "C3_TwoBranch_FV_SignedRR"),
            ("C5_Causal_vs_C3     ", "C5_Causal_Unlabeled_PreQRS_Pwin", "C3_TwoBranch_FV_SignedRR"),
            ("C5_Causal_vs_C4_Ora ", "C5_Causal_Unlabeled_PreQRS_Pwin", "C4_Oracle_PreQRS_Pwin"),
            ("C4_Oracle_vs_C1     ", "C4_Oracle_PreQRS_Pwin", "C1_OneHead_Geom"),
            ("C5_Causal_vs_C1     ", "C5_Causal_Unlabeled_PreQRS_Pwin", "C1_OneHead_Geom"),
        ]:
            g2 = global_wtl(all_rf2[ca], all_rf2[cb])
            g3 = global_wtl(all_rc3[ca], all_rc3[cb])
            g1 = global_wtl(all_rc1[ca], all_rc1[cb])
            summary["Models"][mtype]["Global_WTL_67Pats"][tag.strip()] = {"Pat@F2": g2, "Pat@C3FP": g3, "Pat@C1FP": g1}
            print(
                f"    {tag} | @PatF2: {g2['mean_b']:4.1f}%->{g2['mean_a']:4.1f}% ({g2['str']}) | "
                f"@C3FP: {g3['mean_b']:4.1f}%->{g3['mean_a']:4.1f}% ({g3['str']}) | "
                f"@C1FP: {g1['mean_b']:4.1f}%->{g1['mean_a']:4.1f}% ({g1['str']})",
                flush=True,
            )

        summary["Models"][mtype]["PerPatient_Rec_PatF2"] = {c: {p: round(v, 2) for p, v in all_rf2[c].items()} for c in CFG_ORDER}
        summary["Models"][mtype]["PerPatient_Rec_AtC3FP"] = {c: {p: round(v, 2) for p, v in all_rc3[c].items()} for c in CFG_ORDER}

    out_json = os.path.join(AUD_DIR, "V47_TWOBRANCH_PREQRS_CAUSAL_VS_ORACLE_AUDIT_20261122.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_md = os.path.join(AUD_DIR, "V47_TWOBRANCH_PREQRS_CAUSAL_VS_ORACLE_AUDIT_20261122.md")
    lines = [
        "# v47 Pre-QRS P-Window `[35:63]` Inside the `F:V` Branch Only: Strictly Causal Unlabeled vs. Oracle Diagnostic Upper-Bound",
        "",
        "- **Strict Control**: Locked `K=4` Static `v36` baseline; **`F:N/S` branch FIXED identical (`10D` pure geometry)** across `C3`, `C4_Oracle`, and `C5_Causal`. All models & thresholds trained strictly on 5-hospital source OOF.",
        "- **Transparency Note**: `C4_Oracle_PreQRS_Pwin` uses labeled `y==0` tape-wide normal templates and is **strictly a diagnostic upper bound, NOT a deployable result**. `C5_Causal_Unlabeled_PreQRS_Pwin` uses a strictly causal, 100% unlabeled 25-beat rolling normal buffer initialized from minute-0 unlabeled `bn_sub[0]` (zero labels, zero future beats).",
        "",
        "## Audit A: Causal Unlabeled Buffer Purity, F-Absorption & Template Fidelity",
        "",
        "| DB | All Admitted (`% N / % F / % V`) | At Stage-2 F Beats: Buffer `% N / % F_abs (max)` | Median Template Fidelity `corr(pN_cau, pN_ora)` |",
        "|---|---:|---:|---:|",
    ]
    for db in DB_ORDER:
        b = buf_summary_by_db[db]
        lines.append(
            f"| `{db}` | {b['all_adm_N_mean_%']:.1f}% / {b['all_adm_F_mean_%']:.2f}% / {b['all_adm_V_mean_%']:.2f}% | "
            f"{b['s2F_buf_N_mean_%']}% / **{b['s2F_buf_F_mean_%']}%** (max {b['s2F_buf_F_max_%']}%) | **{b['s2F_tmpl_fid_med']}** |"
        )

    lines.extend([
        "", "## Audit B: Conditional `cAUC(F vs V | |Delta s_geom| <= 0.25)` — Oracle vs. Causal Unlabeled", "",
        "| Feature | Evaluable Pats | Pat-Median `cAUC` | IQR `[p25, p75]` | Consistent `≥0.60` / Reversed `≤0.40` |",
        "|---|---:|---:|---|---|",
    ])
    for fk, lbl in [
        ("ora_raw", "`pq_oracle_pwin_raw` [ORACLE]"),
        ("cau_raw", "`pq_causal_pwin_raw` [CAUSAL UNLABELED]"),
        ("ora_g", "`pq_oracle_pwin_gated` [ORACLE]"),
        ("cau_g", "`pq_causal_pwin_gated` [CAUSAL UNLABELED]"),
    ]:
        g = cauc_gap[fk]
        lines.append(f"| {lbl} | {g['n_pats']} | **{g['pat_med']:.3f}** | [{g['iqr'][0]:.2f}, {g['iqr'][1]:.2f}] | **{g['ge60']}** / **{g['le40']}** |")
    lines.append("")

    for mtype in CLF_FAMILIES:
        lines.extend([
            f"## 1. 6-Hospital LODO Comparison (`{mtype}`, `C=0.3`, Fixed `F:NS` Branch)", "",
            "| DB | Config | AP | DB Rank@F2 (%) | MacroPat @PatF2 (WTL vs C3) | Deployed `T1` F1 | `T1` TP | **`T1` FP_NS** | `T1` FP_V | FAPH | `F_fail_V_only` |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for tgt in DB_ORDER:
            em = summary["Models"][mtype]["ByDB"][tgt]["Eval"]
            for cname in CFG_ORDER:
                ev = em[cname]
                fv_o = ev["Branch_Diag"].get("F_fail_V_only", "—")
                lines.append(
                    f"| `{tgt}` | `{cname}` | **{ev['AP']:.4f}** | **{ev['DB_Rank@F2_%']:.1f}** | "
                    f"**{ev['MacroPat_Pat@F2_%']:.1f}%** ({ev['WTL_Pat@F2_vs_C3']}) | **{ev['T1_F1']:.4f}** | "
                    f"{ev['T1_TP']} | **{ev['T1_FP_NS']}** | {ev['T1_FPV']} | {ev['T1_FAPH']:.2f} | {fv_o} |"
                )
        lines.extend([
            "", f"## 2. Global 67-Patient Matched-FA Comparison (`{mtype}`)", "",
            "| Comparison | 67-Pat `@PatF2` (`B -> A`, WTL) | 67-Pat `@C3_FP` (`B -> A`, WTL) | 67-Pat `@C1_FP` (`B -> A`, WTL) |",
            "|---|---|---|---|",
        ])
        for k, v in summary["Models"][mtype]["Global_WTL_67Pats"].items():
            g2, g3, g1 = v["Pat@F2"], v["Pat@C3FP"], v["Pat@C1FP"]
            lines.append(
                f"| `{k}` | {g2['mean_b']:.2f}% -> **{g2['mean_a']:.2f}%** (**{g2['str']}**) | "
                f"{g3['mean_b']:.2f}% -> **{g3['mean_a']:.2f}%** (**{g3['str']}**) | "
                f"{g1['mean_b']:.2f}% -> **{g1['mean_a']:.2f}%** (**{g1['str']}**) |"
            )
        lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nALL DONE in {time.time()-t0:.1f}s!\nSaved: {out_json}\nSaved: {out_md}", flush=True)


if __name__ == "__main__":
    main()
