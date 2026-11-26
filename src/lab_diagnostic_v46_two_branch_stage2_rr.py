r"""
lab_diagnostic_v46_two_branch_stage2_rr.py
==========================================
TWO-BRANCH STAGE-2 DECOMPOSITION (F vs N/S  AND  F vs V) WITH SIGNED CAUSAL PRE-RR
STRICTLY ON LOCKED K=4 STATIC v36 BASELINE (ZERO CHANGES TO STAGE 1, ANCHORS, OR N-CHART;
ZERO PRE-QRS ORACLE FEATURES — PURELY CAUSAL SIGNED log(pre_rr_loc) ONLY!)

Exact Controlled Ladder (evaluated across all 6 hospitals, 306 recs, 2.68M beats, 67 F+ patients,
under BOTH LinearSVC(C=0.3) [matching F2] and LogisticRegression(C=0.3) [matching G3]):
  1. C1_OneHead_Geom           : Baseline 1-head F vs (N+S+V), 10D pure geometry (F2_IDX)
  2. C2_TwoBranch_GeomOnly     : 2-branch [F vs (N+S), 10D geom] AND [F vs V, 10D geom]
                                 -> Isolates whether splitting comparator classes helps by itself
  3. C3_TwoBranch_FV_SignedRR  : 2-branch [F vs (N+S), 10D geom] AND [F vs V, 10D geom + log(pre_rr_loc)]
                                 -> Isolates causal signed RR contribution specifically inside F:V
  4. C1b_OneHead_Geom_SignedRR : Control 1-head F vs (N+S+V), 10D geom + log(pre_rr_loc)
                                 -> Direct control vs C3: what happens when signed RR enters one global head?

Joint Thresholding & Ranking Protocol (NO fake probability multiplication!):
  - Deployed T1: joint pair (tau_N*, tau_V*) chosen on 5-fold patient-grouped SOURCE OOF
    maximizing the exact same prior-adjusted F1 objective J(tau_N, tau_V) as C1:
      predict F  <==>  (s_{F:NS} >= tau_N*)  AND  (s_{F:V} >= tau_V*).
  - Continuous ranking score for AP / Rank@F2 / Per-Patient Matched-FA Recall:
      S_min(x) = min( (s_{F:NS}(x) - tau_N*) / sigma_N , (s_{F:V}(x) - tau_V*) / sigma_V )
    where sigma_N, sigma_V are source-OOF standard deviations of s_{F:NS}, s_{F:V}.
    Exact identity: S_min(x) >= 0  <==>  deployed conjunction (s_{F:NS} >= tau_N* AND s_{F:V} >= tau_V*).

Success Criteria Audited:
  1. Higher per-patient recall at matched false-alarm budgets (@PatF2 <= 2.0 FAPH and @C1_FP)
     with exact 67-patient Win/Tie/Loss counts (C2 vs C1, C3 vs C1, C3 vs C2, C3 vs C1b).
  2. NO increase in N/S -> F false positives (T1_FP_NS = FP_N + FP_S tracked explicitly!).
  3. Exact branch-attribution of missed F & remaining FPs in C3.
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
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v43_hypB2_representation_vs_veto as v43

AUD_DIR = str(_REPO_ROOT / "audits")
CACHE_V36 = str(_DATA_ROOT / "cache_gen3_6db")
CACHE_V37 = str(_DATA_ROOT / "cache_v37_waveforms_6db")
os.makedirs(AUD_DIR, exist_ok=True)

DB_ORDER = ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]
F2_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]  # 10D pure geometry (zero RR, zero pre-QRS)
# Feature index 18 will hold strictly causal signed log(pre_rr_loc)!
F2_PLUS_RR_IDX = F2_IDX + [18]

CLF_FAMILIES = ["SVM", "LogR"]
CFG_ORDER = [
    "C1_OneHead_Geom",
    "C2_TwoBranch_GeomOnly",
    "C3_TwoBranch_FV_SignedRR",
    "C1b_OneHead_Geom_SignedRR",
]


def load_v46_dbs():
    dbs = {}
    for db in DB_ORDER:
        m36 = json.load(open(os.path.join(CACHE_V36, f"{db}_gen3_4cls_meta.json"), "r", encoding="utf-8"))
        d36 = np.load(os.path.join(CACHE_V36, f"{db}_gen3_4cls.npz"))
        m37 = json.load(open(os.path.join(CACHE_V37, f"{db}_meta.json"), "r", encoding="utf-8"))
        rr3 = np.load(os.path.join(CACHE_V37, f"{db}_rr3.npy"), mmap_mode="r")
        m37_map = {m["rec_id"]: m for m in m37}
        items = []
        for m in m36:
            rid = m["rec_id"]
            m7 = m37_map[rid]
            b0, b1 = m["b0"], m["b1"]
            assert (m7["b1"] - m7["b0"]) == (b1 - b0)
            y = d36["y_all"][b0:b1]
            s2 = d36["s2_all"][b0:b1]
            X18 = d36["X18_s2_all"][m["s0"]:m["s1"]].astype(np.float64)
            pre_rr_s2 = np.array(rr3[m7["b0"]:m7["b1"]][s2, 1], dtype=np.float64)
            log_rr_s2 = np.log(np.clip(pre_rr_s2, 0.2, 3.0))
            X19 = np.column_stack([X18, log_rr_s2]) if len(X18) > 0 else np.zeros((0, 19), dtype=np.float64)
            items.append({
                "db": db, "rec_id": rid, "patient_id": m["patient_id"],
                "post_hours": m["post_hours"], "K": m["K"],
                "y": y, "s2_mask": s2, "X_s2": X19,
            })
        dbs[db] = items
    return dbs


def make_clf(mtype):
    if mtype == "LogR":
        return LogisticRegression(C=0.3, max_iter=500, random_state=42)
    return LinearSVC(C=0.3, dual="auto", max_iter=3000, random_state=42)


def build_branch_xy(sub_items, feat_idx, neg_classes, cap_pat_pos=100):
    """
    Builds balanced (X, y_bin, sample_weight) for a branch:
      pos = Class F (y == 2), patient-capped at cap_pat_pos
      neg = y in neg_classes (e.g. {0, 3} for F:NS, {1} for F:V, {0, 1, 3} for OneHead)
    """
    neg_arr = np.array(list(neg_classes), dtype=np.int8)
    pf_c = Counter()
    for it in sub_items:
        if len(it["X_s2"]) > 0:
            pf_c[it["patient_id"]] += int(np.sum(it["y"][it["s2_mask"]] == 2))

    Xs, ys, ws = [], [], []
    for it in sub_items:
        if len(it["X_s2"]) == 0:
            continue
        ys2 = it["y"][it["s2_mask"]]
        keep = (ys2 == 2) | np.isin(ys2, neg_arr)
        if not np.any(keep):
            continue
        Xm = it["X_s2"][keep][:, feat_idx]
        ym = (ys2[keep] == 2).astype(int)
        wm = np.ones(len(ym), dtype=np.float64)
        pf = pf_c[it["patient_id"]]
        if cap_pat_pos is not None and pf > cap_pat_pos:
            wm[ym == 1] = float(cap_pat_pos) / float(pf)
        Xs.append(Xm); ys.append(ym); ws.append(wm)

    X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    sp, sn = float(np.sum(w[y == 1])), float(np.sum(w[y == 0]))
    w[y == 1] *= (0.5 * (sp + sn) / max(1e-12, sp))
    w[y == 0] *= (0.5 * (sp + sn) / max(1e-12, sn))
    return X, y, w


def fit_onehead_lodo(tr_items, feat_idx, mtype, cap_pat_pos=100):
    pat_map = {}
    for it in tr_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: -sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    oof_sc, oof_y = [], []
    tot_b = sum(len(it["y"]) for it in tr_items)
    tot_f = sum(int(np.sum(it["y"] == 2)) for it in tr_items)
    for fid in range(n_folds):
        tr_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] == fid]
        Xtr, ytr, wtr = build_branch_xy(tr_sub, feat_idx, {0, 1, 3}, cap_pat_pos=cap_pat_pos)
        if np.sum(ytr == 1) == 0: continue
        sc = StandardScaler(); clf = make_clf(mtype)
        clf.fit(sc.fit_transform(Xtr), ytr, sample_weight=wtr)
        for te in te_sub:
            if len(te["X_s2"]) > 0:
                oof_sc.append(clf.decision_function(sc.transform(te["X_s2"][:, feat_idx])))
                oof_y.append((te["y"][te["s2_mask"]] == 2).astype(int))

    s2_sc = np.concatenate(oof_sc); s2_y = np.concatenate(oof_y)
    order = np.argsort(s2_sc)[::-1]
    ys_o, ss_o = s2_y[order], s2_sc[order]
    tp_c = np.cumsum(ys_o == 1).astype(float); fp_c = np.cumsum(ys_o == 0).astype(float)
    pi_src = tot_f / max(1.0, float(tot_b))
    neg_mult = (pi_src / max(1e-9, 1.0 - pi_src)) / (0.001 / 0.999)
    tau1 = float(ss_o[int(np.argmax(2.0 * tp_c / (float(tot_f) + tp_c + fp_c * neg_mult + 1e-12)))])
    sig = float(np.std(s2_sc)) + 1e-9

    X_all, y_all, w_all = build_branch_xy(tr_items, feat_idx, {0, 1, 3}, cap_pat_pos=cap_pat_pos)
    scaler = StandardScaler(); clf = make_clf(mtype)
    clf.fit(scaler.fit_transform(X_all), y_all, sample_weight=w_all)
    return {"mode": "onehead", "feat_idx": feat_idx, "scaler": scaler, "clf": clf, "tau1": tau1, "sig": sig}


def joint_oof_2d_threshold(sN, sV, y_bin, tot_f, tot_b, n_grid=120):
    """
    Exact vectorized 2D suffix-sum search for (tau_N*, tau_V*) maximizing the
    EXACT same prior-adjusted F1 objective J(tau_N, tau_V) on source OOF!
    """
    pos = (y_bin == 1)
    # Build dense candidate grids focusing on both positive support & upper negative tails
    def make_grid(arr):
        p_vals = arr[pos]
        g1 = np.quantile(p_vals, np.linspace(0.0, 0.995, n_grid // 2))
        g2 = np.quantile(arr, np.linspace(0.50, 0.9999, n_grid // 2))
        g = np.unique(np.concatenate([g1, g2, [float(np.min(arr)) - 1.0]]))
        return g

    tN = make_grid(sN)
    tV = make_grid(sV)
    GN, GV = len(tN), len(tV)

    # Bin index i = largest index with tN[i] <= sN (guaranteed >= 0 because min-1 is included!)
    iN = np.clip(np.searchsorted(tN, sN, side="right") - 1, 0, GN - 1)
    iV = np.clip(np.searchsorted(tV, sV, side="right") - 1, 0, GV - 1)

    H_pos = np.zeros((GN, GV), dtype=np.float64)
    H_neg = np.zeros((GN, GV), dtype=np.float64)
    np.add.at(H_pos, (iN[pos], iV[pos]), 1.0)
    np.add.at(H_neg, (iN[~pos], iV[~pos]), 1.0)

    # Exact 2D reverse suffix sum: TP[i, j] = count(sN >= tN[i] AND sV >= tV[j])
    TP = H_pos[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
    FP = H_neg[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]

    pi_src = float(tot_f) / max(1.0, float(tot_b))
    neg_mult = (pi_src / max(1e-9, 1.0 - pi_src)) / (0.001 / 0.999)
    J = 2.0 * TP / (float(tot_f) + TP + FP * neg_mult + 1e-12)
    best_flat = int(np.argmax(J))
    bi, bj = divmod(best_flat, GV)
    return float(tN[bi]), float(tV[bj]), float(J[bi, bj])


def fit_twobranch_lodo(tr_items, feat_idx_N, feat_idx_V, mtype, cap_pat_pos=100):
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

        XN, yN, wN = build_branch_xy(tr_sub, feat_idx_N, {0, 3}, cap_pat_pos=cap_pat_pos)
        XV, yV, wV = build_branch_xy(tr_sub, feat_idx_V, {1}, cap_pat_pos=cap_pat_pos)
        if np.sum(yN == 1) == 0 or np.sum(yV == 1) == 0:
            continue

        scN = StandardScaler(); clfN = make_clf(mtype)
        clfN.fit(scN.fit_transform(XN), yN, sample_weight=wN)

        scV = StandardScaler(); clfV = make_clf(mtype)
        clfV.fit(scV.fit_transform(XV), yV, sample_weight=wV)

        for te in te_sub:
            if len(te["X_s2"]) > 0:
                oof_sN.append(clfN.decision_function(scN.transform(te["X_s2"][:, feat_idx_N])))
                oof_sV.append(clfV.decision_function(scV.transform(te["X_s2"][:, feat_idx_V])))
                oof_y.append((te["y"][te["s2_mask"]] == 2).astype(int))

    sN_cat = np.concatenate(oof_sN); sV_cat = np.concatenate(oof_sV); y_cat = np.concatenate(oof_y)
    tau_N, tau_V, best_J = joint_oof_2d_threshold(sN_cat, sV_cat, y_cat, tot_f, tot_b, n_grid=140)
    sig_N = float(np.std(sN_cat)) + 1e-9
    sig_V = float(np.std(sV_cat)) + 1e-9

    # Fit final full-source branches
    XN_all, yN_all, wN_all = build_branch_xy(tr_items, feat_idx_N, {0, 3}, cap_pat_pos=cap_pat_pos)
    scN_f = StandardScaler(); clfN_f = make_clf(mtype)
    clfN_f.fit(scN_f.fit_transform(XN_all), yN_all, sample_weight=wN_all)

    XV_all, yV_all, wV_all = build_branch_xy(tr_items, feat_idx_V, {1}, cap_pat_pos=cap_pat_pos)
    scV_f = StandardScaler(); clfV_f = make_clf(mtype)
    clfV_f.fit(scV_f.fit_transform(XV_all), yV_all, sample_weight=wV_all)

    return {
        "mode": "twobranch",
        "feat_idx_N": feat_idx_N, "feat_idx_V": feat_idx_V,
        "scN": scN_f, "clfN": clfN_f, "tau_N": tau_N, "sig_N": sig_N,
        "scV": scV_f, "clfV": clfV_f, "tau_V": tau_V, "sig_V": sig_V,
        "oof_J": round(best_J, 4),
        "w_V_rr": round(float(clfV_f.coef_.ravel()[-1]), 3) if len(feat_idx_V) == 11 else None,
    }


def score_and_eval_v46(te_items, model_obj, ref_pat_fp=None, ref_pat_rf2=None, ref_pat_rc1fp=None):
    """
    Scores target DB cleanly:
      - OneHead: score = (s - tau1) / sig  (deployed T1 <=> score >= 0.0)
      - TwoBranch: score = min((sN - tau_N)/sig_N, (sV - tau_V)/sig_V)
        (EXACT identity: deployed conjunction sN>=tau_N AND sV>=tau_V  <==>  score >= 0.0!)
    """
    tot_hours = sum(it["post_hours"] for it in te_items)
    sc_l, y_l, rec_sc = [], [], []
    # Branch diagnostics for twobranch
    b_diag = Counter()

    for it in te_items:
        n = len(it["y"])
        sc = np.full(n, -1e6, dtype=np.float64)
        if len(it["X_s2"]) > 0:
            if model_obj["mode"] == "onehead":
                raw = model_obj["clf"].decision_function(model_obj["scaler"].transform(it["X_s2"][:, model_obj["feat_idx"]]))
                sc[it["s2_mask"]] = (raw - model_obj["tau1"]) / model_obj["sig"]
            else:
                sN = model_obj["clfN"].decision_function(model_obj["scN"].transform(it["X_s2"][:, model_obj["feat_idx_N"]]))
                sV = model_obj["clfV"].decision_function(model_obj["scV"].transform(it["X_s2"][:, model_obj["feat_idx_V"]]))
                mN = (sN - model_obj["tau_N"]) / model_obj["sig_N"]
                mV = (sV - model_obj["tau_V"]) / model_obj["sig_V"]
                sc[it["s2_mask"]] = np.minimum(mN, mV)

                # Exact branch attribution on S2 beats!
                ys2 = it["y"][it["s2_mask"]]
                pN = (sN >= model_obj["tau_N"]); pV = (sV >= model_obj["tau_V"])
                isF = (ys2 == 2); isNS = (ys2 == 0) | (ys2 == 3); isV = (ys2 == 1)
                b_diag["F_pass_both"] += int(np.sum(isF & pN & pV))
                b_diag["F_fail_N_only"] += int(np.sum(isF & (~pN) & pV))
                b_diag["F_fail_V_only"] += int(np.sum(isF & pN & (~pV)))
                b_diag["F_fail_both"] += int(np.sum(isF & (~pN) & (~pV)))
                b_diag["NS_pass_N"] += int(np.sum(isNS & pN))
                b_diag["NS_pass_both"] += int(np.sum(isNS & pN & pV))
                b_diag["V_pass_V"] += int(np.sum(isV & pV))
                b_diag["V_pass_both"] += int(np.sum(isV & pN & pV))

        sc_l.append(sc); y_l.append(it["y"]); rec_sc.append((it, sc))

    sc_cat = np.concatenate(sc_l); y_cat = np.concatenate(y_l)
    y_bin = (y_cat == 2).astype(int)
    tot_ft = int(np.sum(y_bin == 1))
    ap = float(average_precision_score(y_bin, sc_cat)) if tot_ft > 0 else 0.0

    # Pooled oracle ranking @ DB FAPH <= 2.0
    valid = (sc_cat > -1e5)
    sv = sc_cat[valid]; yv = (y_cat[valid] == 2)
    ord_v = np.argsort(sv)[::-1]
    ok2 = np.where(np.cumsum(~yv[ord_v]) <= int(np.floor(2.0 * tot_hours)))[0]
    tau_f2 = float(sv[ord_v[ok2[-1]]]) if len(ok2) > 0 else float(sv[ord_v[0]]) + 1.0
    rec_db_f2 = round(100.0 * int(np.sum((sc_cat >= tau_f2) & (y_cat == 2))) / max(1, tot_ft), 2)

    # Per-patient matched FA budgets
    pat_data = {}
    for it, sc in rec_sc:
        p = it["patient_id"]
        d = pat_data.setdefault(p, {"sc": [], "y": [], "h": 0.0})
        d["sc"].append(sc); d["y"].append(it["y"]); d["h"] += it["post_hours"]

    pat_fp_t1 = {}
    pat_rec_f2_map = {}
    pat_rec_c1fp_map = {}
    for p, pd in pat_data.items():
        psc = np.concatenate(pd["sc"]); py = np.concatenate(pd["y"]); pyb = (py == 2).astype(int)
        pat_fp_t1[p] = int(np.sum((psc >= 0.0) & (py != 2)))
        if int(np.sum(pyb)) > 0:
            pat_rec_f2_map[p] = v43.recall_at_fp_budget(psc, pyb, int(np.floor(2.0 * pd["h"])))
            if ref_pat_fp is not None and p in ref_pat_fp:
                pat_rec_c1fp_map[p] = v43.recall_at_fp_budget(psc, pyb, ref_pat_fp[p])

    mp_pat_f2 = round(float(np.mean(list(pat_rec_f2_map.values()))), 2) if pat_rec_f2_map else 0.0
    mp_pat_c1fp = round(float(np.mean(list(pat_rec_c1fp_map.values()))), 2) if pat_rec_c1fp_map else 0.0

    def wtl(cur, ref):
        if ref is None or not cur:
            return "—"
        w = sum(1 for p in cur if cur[p] > ref.get(p, 0.0) + 1e-6)
        l = sum(1 for p in cur if cur[p] < ref.get(p, 0.0) - 1e-6)
        t = len(cur) - w - l
        return f"{w}W/{t}T/{l}L"

    # Deployed source-locked T1 (exact level set sc_cat >= 0.0!)
    pred = (sc_cat >= 0.0)
    tp1 = int(np.sum(pred & (y_cat == 2)))
    fpn1 = int(np.sum(pred & (y_cat == 0)))
    fpv1 = int(np.sum(pred & (y_cat == 1)))
    fps1 = int(np.sum(pred & (y_cat == 3)))
    fp_ns = fpn1 + fps1
    fp1 = fp_ns + fpv1
    f1_t1 = 2.0 * tp1 / max(1, 2 * tp1 + fp1 + (tot_ft - tp1))
    prec1 = 100.0 * tp1 / max(1, tp1 + fp1)
    rec1 = 100.0 * tp1 / max(1, tot_ft)
    faph1 = fp1 / max(1e-6, tot_hours)
    zf_tot = sum(len(it["y"]) for it, _ in rec_sc if int(np.sum(it["y"] == 2)) == 0)
    zf_fp = sum(int(np.sum((sc >= 0.0) & (it["y"] != 2))) for it, sc in rec_sc if int(np.sum(it["y"] == 2)) == 0)
    spec = 100.0 * (1.0 - zf_fp / max(1, zf_tot)) if zf_tot > 0 else 100.0 * (1.0 - fp1 / max(1, len(y_cat) - tot_ft))

    ev = {
        "AP": round(ap, 4),
        "DB_Rank@F2_%": rec_db_f2,
        "MacroPat_Pat@F2_%": mp_pat_f2, "WTL_Pat@F2_vs_C1": wtl(pat_rec_f2_map, ref_pat_rf2),
        "MacroPat_AtC1FP_%": mp_pat_c1fp, "WTL_AtC1FP_vs_C1": wtl(pat_rec_c1fp_map, ref_pat_rc1fp),
        "T1_F1": round(f1_t1, 4), "T1_Prec_%": round(prec1, 2), "T1_Rec_%": round(rec1, 2),
        "T1_Spec_%": round(spec, 3), "T1_FAPH": round(faph1, 2),
        "T1_TP": tp1, "T1_FP_NS": fp_ns, "T1_FPN": fpn1, "T1_FPS": fps1, "T1_FPV": fpv1,
        "T1_Counts": f"{tp1}/{fpn1}/{fpv1}/{fps1}S",
        "Branch_Diag": dict(b_diag),
    }
    return ev, pat_fp_t1, pat_rec_f2_map, pat_rec_c1fp_map


def main():
    t0 = time.time()
    dbs = load_v46_dbs()
    summary = {}

    for mtype in CLF_FAMILIES:
        summary[mtype] = {"ByDB": {}, "Global_WTL_67Pats": {}}
        print(f"\n==================== CLASSIFIER FAMILY: {mtype} (C=0.3, K=4 BASELINE, 6-HOSPITAL LODO) ====================", flush=True)

        # Store per-patient recall maps across ALL 67 F+ patients for global WTL (C2 vs C1, C3 vs C1, C3 vs C2, C3 vs C1b)
        all_rf2 = {c: {} for c in CFG_ORDER}
        all_rc1 = {c: {} for c in CFG_ORDER}

        for tgt in DB_ORDER:
            tr = [it for db in DB_ORDER if db != tgt for it in dbs[db]]
            te = dbs[tgt]

            m_c1 = fit_onehead_lodo(tr, F2_IDX, mtype)
            m_c1b = fit_onehead_lodo(tr, F2_PLUS_RR_IDX, mtype)
            m_c2 = fit_twobranch_lodo(tr, F2_IDX, F2_IDX, mtype)
            m_c3 = fit_twobranch_lodo(tr, F2_IDX, F2_PLUS_RR_IDX, mtype)

            # Evaluate C1 first to get per-patient C1 T1 FP counts & baseline recall maps
            _, c1_pfp, c1_rf2, _ = score_and_eval_v46(te, m_c1)
            ev1, _, c1_rf2, c1_rc1 = score_and_eval_v46(te, m_c1, ref_pat_fp=c1_pfp, ref_pat_rf2=c1_rf2)

            ev_map = {"C1_OneHead_Geom": ev1}
            all_rf2["C1_OneHead_Geom"].update(c1_rf2); all_rc1["C1_OneHead_Geom"].update(c1_rc1)

            for cname, mobj in [
                ("C2_TwoBranch_GeomOnly", m_c2),
                ("C3_TwoBranch_FV_SignedRR", m_c3),
                ("C1b_OneHead_Geom_SignedRR", m_c1b),
            ]:
                ev, _, rf2, rc1 = score_and_eval_v46(te, mobj, ref_pat_fp=c1_pfp, ref_pat_rf2=c1_rf2, ref_pat_rc1fp=c1_rc1)
                ev_map[cname] = ev
                all_rf2[cname].update(rf2); all_rc1[cname].update(rc1)

            summary[mtype]["ByDB"][tgt] = {
                "Eval": ev_map,
                "C3_w_V_rr": m_c3["w_V_rr"],
                "C1b_w_rr": round(float(m_c1b["clf"].coef_.ravel()[-1]), 3),
                "C2_taus": [round(m_c2["tau_N"], 3), round(m_c2["tau_V"], 3)],
                "C3_taus": [round(m_c3["tau_N"], 3), round(m_c3["tau_V"], 3)],
            }

            print(f"\n  [{tgt}] ({mtype}) | C3 w_V(log_rr)={m_c3['w_V_rr']} | C1b w(log_rr)={round(float(m_c1b['clf'].coef_.ravel()[-1]), 3)}", flush=True)
            for cname in CFG_ORDER:
                ev = ev_map[cname]
                print(
                    f"    {cname:26s} | AP={ev['AP']:.4f} DB@F2={ev['DB_Rank@F2_%']:4.1f}% "
                    f"Pat@F2={ev['MacroPat_Pat@F2_%']:4.1f}%({ev['WTL_Pat@F2_vs_C1']}) "
                    f"@C1FP={ev['MacroPat_AtC1FP_%']:4.1f}%({ev['WTL_AtC1FP_vs_C1']}) | "
                    f"T1: F1={ev['T1_F1']:.4f} [{ev['T1_Counts']:16s}] FP_NS={ev['T1_FP_NS']:4d} FP_V={ev['T1_FPV']:4d} FAPH={ev['T1_FAPH']:5.2f}",
                    flush=True,
                )

        # Compute exact global 67-patient Win/Tie/Loss pairs!
        def global_wtl(map_a, map_b):
            w = sum(1 for p in map_a if map_a[p] > map_b[p] + 1e-6)
            l = sum(1 for p in map_a if map_a[p] < map_b[p] - 1e-6)
            t = len(map_a) - w - l
            return {"W": w, "T": t, "L": l, "mean_a": round(float(np.mean(list(map_a.values()))), 2),
                    "mean_b": round(float(np.mean(list(map_b.values()))), 2), "str": f"{w}W/{t}T/{l}L"}

        print(f"\n  === GLOBAL 67-PATIENT SUMMARY ({mtype}) ===", flush=True)
        for tag, ca, cb in [
            ("C2_vs_C1 (Split Only)", "C2_TwoBranch_GeomOnly", "C1_OneHead_Geom"),
            ("C3_vs_C1 (Split+RR)  ", "C3_TwoBranch_FV_SignedRR", "C1_OneHead_Geom"),
            ("C3_vs_C2 (RR on 2Br) ", "C3_TwoBranch_FV_SignedRR", "C2_TwoBranch_GeomOnly"),
            ("C3_vs_C1b (2Br vs 1H)", "C3_TwoBranch_FV_SignedRR", "C1b_OneHead_Geom_SignedRR"),
            ("C1b_vs_C1 (RR on 1H) ", "C1b_OneHead_Geom_SignedRR", "C1_OneHead_Geom"),
        ]:
            g_f2 = global_wtl(all_rf2[ca], all_rf2[cb])
            g_c1 = global_wtl(all_rc1[ca], all_rc1[cb])
            summary[mtype]["Global_WTL_67Pats"][tag.strip()] = {"Pat@F2": g_f2, "Pat@C1FP": g_c1}
            print(f"    {tag} | @PatF2: {g_f2['mean_b']:4.1f}% -> {g_f2['mean_a']:4.1f}% ({g_f2['str']}) | @C1FP: {g_c1['mean_b']:4.1f}% -> {g_c1['mean_a']:4.1f}% ({g_c1['str']})", flush=True)

        # Also store per-patient recall maps for detailed inspection!
        summary[mtype]["PerPatient_Rec_PatF2"] = {c: {p: round(v, 2) for p, v in all_rf2[c].items()} for c in CFG_ORDER}
        summary[mtype]["PerPatient_Rec_AtC1FP"] = {c: {p: round(v, 2) for p, v in all_rc1[c].items()} for c in CFG_ORDER}

    out_json = os.path.join(AUD_DIR, "V46_TWO_BRANCH_STAGE2_RR_AUDIT_20261122.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_md = os.path.join(AUD_DIR, "V46_TWO_BRANCH_STAGE2_RR_AUDIT_20261122.md")
    lines = [
        "# v46 Stage-2 Two-Branch Decomposition (`F:NS` AND `F:V`) + Signed Causal `log(pre_rr_loc)`",
        "",
        "- **Strict Control**: Locked `K=4` Static `v36` cache (`cache_gen3_6db` + causal `rr3[:, 1]`), zero changes to Stage 1 / anchor library / N-chart, zero oracle pre-QRS features.",
        "- **Joint Conjunction Decision**: `(s_{F:NS} >= tau_N*) AND (s_{F:V} >= tau_V*)` with `(tau_N*, tau_V*)` jointly selected on 5-fold patient-grouped source OOF maximizing the exact same prior-adjusted F1 objective as `C1`. Continuous ranking uses exact signed bottleneck margin `S_min = min((sN - tau_N*)/sig_N, (sV - tau_V*)/sig_V)` (`S_min >= 0` iff deployed conjunction triggers — no fake probability multiplication).",
        "",
    ]
    for mtype in CLF_FAMILIES:
        lines.extend([
            f"## 1. 6-Hospital LODO Comparison (`{mtype}`, `C=0.3`, `K=4` Baseline)", "",
            "| DB | Config | AP | DB Rank@F2 (%) | MacroPat @PatF2 (WTL vs C1) | MacroPat @C1_FP (WTL vs C1) | Deployed `T1` F1 | `T1` TP | **`T1` FP_NS (N+S)** | `T1` FP_V | Spec (%) | FAPH |",
            "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for tgt in DB_ORDER:
            em = summary[mtype]["ByDB"][tgt]["Eval"]
            for cname in CFG_ORDER:
                ev = em[cname]
                lines.append(
                    f"| `{tgt}` | `{cname}` | **{ev['AP']:.4f}** | **{ev['DB_Rank@F2_%']:.1f}** | "
                    f"**{ev['MacroPat_Pat@F2_%']:.1f}%** ({ev['WTL_Pat@F2_vs_C1']}) | **{ev['MacroPat_AtC1FP_%']:.1f}%** ({ev['WTL_AtC1FP_vs_C1']}) | "
                    f"**{ev['T1_F1']:.4f}** | {ev['T1_TP']} | **{ev['T1_FP_NS']}** | {ev['T1_FPV']} | {ev['T1_Spec_%']:.3f} | {ev['T1_FAPH']:.2f} |"
                )
        lines.extend([
            "", f"## 2. Global 67-Patient Matched-FA Comparison (`{mtype}`)", "",
            "| Comparison | 67-Pat Mean `@PatF2` (`B -> A`) | 67-Pat WTL `@PatF2` | 67-Pat Mean `@C1_FP` (`B -> A`) | 67-Pat WTL `@C1_FP` |",
            "|---|---|---|---|---|",
        ])
        for k, v in summary[mtype]["Global_WTL_67Pats"].items():
            g2 = v["Pat@F2"]; gc = v["Pat@C1FP"]
            lines.append(
                f"| `{k}` | {g2['mean_b']:.2f}% -> **{g2['mean_a']:.2f}%** | **{g2['str']}** | {gc['mean_b']:.2f}% -> **{gc['mean_a']:.2f}%** | **{gc['str']}** |"
            )
        lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nALL DONE in {time.time()-t0:.1f}s!\nSaved: {out_json}\nSaved: {out_md}", flush=True)


if __name__ == "__main__":
    main()
