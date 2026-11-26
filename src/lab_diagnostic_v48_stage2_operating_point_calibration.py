r"""
lab_diagnostic_v48_stage2_operating_point_calibration.py
========================================================
FINAL BOUNDED STAGE-2 OPERATING-POINT CALIBRATION AUDIT (NO NEW FEATURES; LOCKED K=4 BASELINE)

Directly tests the user's exact hypothesis on reusing source-OOF predictions to compare
EXACTLY TWO calibration rules fairly across ALL THREE strictly causal models
(C1_OneHead_Geom, C3_TwoBranch_FV_SignedRR, C5_Causal_Unlabeled_PreQRS_Pwin)
under BOTH SVM (LinearSVC C=0.3) and LogR (LogisticRegression C=0.3):

  Rule R1 (Current Baseline — Pooled Prior-Adjusted F1):
    - Exact v46/v47 rule: maximizes pooled prior-adjusted F1 (pi_target=0.001) on 5-fold
      patient-grouped source OOF (scalar tau1* for C1; joint 2D (tau_N*, tau_V*) for C3 & C5).

  Rule R2 (Control — Patient-Level Macro-Recall under Pre-Specified Source FA Budget B = 2.0 FAPH):
    - Constraint includes ALL source patients P_all (explicitly including every zero-F patient
      and every patient with zero Stage-2 beats!):
          MeanPat_FAPH_all_src(tau) = (1 / |P_all|) * sum_{p in P_all} (FP_p(tau) / H_p) <= 2.0 FAPH
    - C1 (1-head): exact sorted-event threshold tau1* maximizing source MacroPatRec(F+)
      subject to MeanPat_FAPH_all_src(tau1) <= 2.0.
    - C3 & C5 (2-branch, IDENTICAL pure-geometry F:N/S branch sN):
        Step 1: Compute ONE shared frozen control threshold tau_N_ctrl on sN alone
                maximizing source MacroPatRec(F+) s.t. MeanPat_FAPH_NS_all_src(tau_N) <= 1.0 FAPH
                (symmetric 50% share of the 2.0 FAPH total budget allocated to N/S; identical
                for C3 and C5, immune to R1's 15-FAPH N/S collapse on holdout=ltdb).
        Step 2: Freeze tau_N = tau_N_ctrl identically for BOTH C3 and C5, and select ONLY tau_V*
                maximizing source MacroPatRec(F+) s.t. total MeanPat_FAPH_all_src(tau_N_ctrl, tau_V) <= 2.0.

ZERO target-DB peeking; ZERO refitting difference between R1 and R2 (bit-identical fitted
classifiers & source OOF scores). Simultaneously evaluates:
  1. Per-patient recall AT the source-locked operating point (T1_MacroPat_Rec_% + 67-pat WTL).
  2. Realized target FA transfer (MeanPat_FAPH_All, MeanPat_FAPH_ZeroF, Pooled_FAPH, % pats <= 2 FAPH).
  3. Separated FP-N/S and FP-V, TP, F1, and Branch_Diag (F_fail_N_only, F_fail_V_only, NS_pass_N).
  4. Exact per-patient helped vs harmed breakdown (tracking BOTH recall and FAPH per patient).
  5. AP (and DB@F2 / Pat@F2) to verify whether changing (tau_N*, tau_V*) masks any ranking degradation.
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

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v43_hypB2_representation_vs_veto as v43
import lab_diagnostic_v46_two_branch_stage2_rr as v46
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47

AUD_DIR = str(_REPO_ROOT / "audits")
os.makedirs(AUD_DIR, exist_ok=True)

DB_ORDER = v47.DB_ORDER
F2_IDX = v47.F2_IDX
IDX_C3 = v47.IDX_C3
IDX_C5 = v47.IDX_C5_CAUSAL

CLF_FAMILIES = ["SVM", "LogR"]
CFG_ORDER = [
    "C1_OneHead_Geom",
    "C3_TwoBranch_FV_SignedRR",
    "C5_Causal_Unlabeled_PreQRS_Pwin",
]
RULE_ORDER = ["R1_PooledPriorF1", "R2_PatMacroRec_FA2"]

B_TOTAL_FAPH = 2.0
B_NS_FAPH = 1.0  # Symmetric 50% share of B_TOTAL_FAPH for shared pure-geometry F:N/S control branch


def fit_fold_models_and_oof(tr_items, mtype, cap_pat_pos=100):
    """
    Fits C1, shared F:N/S, C3 F:V, and C5 F:V ONCE using the exact deterministic 5-fold
    patient-grouped source split from v46/v47, returning bit-identical source OOF scores
    and final full-source classifiers so R1 and R2 compare PURELY calibration rules.
    """
    pat_map = {}
    for it in tr_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: -sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    # Source patient registry across ALL source patients (including 0-F and 0-S2 patients!)
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

    oof_s1, oof_sN, oof_sV3, oof_sV5, oof_y, oof_pi = [], [], [], [], [], []
    for fid in range(n_folds):
        tr_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] == fid]

        X1, y1, w1 = v46.build_branch_xy(tr_sub, F2_IDX, {0, 1, 3}, cap_pat_pos=cap_pat_pos)
        XN, yN, wN = v46.build_branch_xy(tr_sub, F2_IDX, {0, 3}, cap_pat_pos=cap_pat_pos)
        XV3, yV3, wV3 = v46.build_branch_xy(tr_sub, IDX_C3, {1}, cap_pat_pos=cap_pat_pos)
        XV5, yV5, wV5 = v46.build_branch_xy(tr_sub, IDX_C5, {1}, cap_pat_pos=cap_pat_pos)
        if np.sum(y1 == 1) == 0 or np.sum(yN == 1) == 0 or np.sum(yV3 == 1) == 0:
            continue

        sc1 = StandardScaler(); clf1 = v46.make_clf(mtype); clf1.fit(sc1.fit_transform(X1), y1, sample_weight=w1)
        scN = StandardScaler(); clfN = v46.make_clf(mtype); clfN.fit(scN.fit_transform(XN), yN, sample_weight=wN)
        scV3 = StandardScaler(); clfV3 = v46.make_clf(mtype); clfV3.fit(scV3.fit_transform(XV3), yV3, sample_weight=wV3)
        scV5 = StandardScaler(); clfV5 = v46.make_clf(mtype); clfV5.fit(scV5.fit_transform(XV5), yV5, sample_weight=wV5)

        for te in te_sub:
            if len(te["X_s2"]) > 0:
                oof_s1.append(clf1.decision_function(sc1.transform(te["X_s2"][:, F2_IDX])))
                oof_sN.append(clfN.decision_function(scN.transform(te["X_s2"][:, F2_IDX])))
                oof_sV3.append(clfV3.decision_function(scV3.transform(te["X_s2"][:, IDX_C3])))
                oof_sV5.append(clfV5.decision_function(scV5.transform(te["X_s2"][:, IDX_C5])))
                oof_y.append(te["y"][te["s2_mask"]].astype(np.int8))
                oof_pi.append(np.full(len(te["X_s2"]), pat_to_idx[te["patient_id"]], dtype=np.int32))

    # Fit final full-source classifiers once
    X1a, y1a, w1a = v46.build_branch_xy(tr_items, F2_IDX, {0, 1, 3}, cap_pat_pos=cap_pat_pos)
    sc1_f = StandardScaler(); clf1_f = v46.make_clf(mtype); clf1_f.fit(sc1_f.fit_transform(X1a), y1a, sample_weight=w1a)

    XNa, yNa, wNa = v46.build_branch_xy(tr_items, F2_IDX, {0, 3}, cap_pat_pos=cap_pat_pos)
    scN_f = StandardScaler(); clfN_f = v46.make_clf(mtype); clfN_f.fit(scN_f.fit_transform(XNa), yNa, sample_weight=wNa)

    XV3a, yV3a, wV3a = v46.build_branch_xy(tr_items, IDX_C3, {1}, cap_pat_pos=cap_pat_pos)
    scV3_f = StandardScaler(); clfV3_f = v46.make_clf(mtype); clfV3_f.fit(scV3_f.fit_transform(XV3a), yV3a, sample_weight=wV3a)

    XV5a, yV5a, wV5a = v46.build_branch_xy(tr_items, IDX_C5, {1}, cap_pat_pos=cap_pat_pos)
    scV5_f = StandardScaler(); clfV5_f = v46.make_clf(mtype); clfV5_f.fit(scV5_f.fit_transform(XV5a), yV5a, sample_weight=wV5a)

    s1_cat = np.concatenate(oof_s1)
    sN_cat = np.concatenate(oof_sN)
    sV3_cat = np.concatenate(oof_sV3)
    sV5_cat = np.concatenate(oof_sV5)
    y_cat = np.concatenate(oof_y)
    pi_cat = np.concatenate(oof_pi)

    return {
        "oof_s1": s1_cat, "oof_sN": sN_cat, "oof_sV3": sV3_cat, "oof_sV5": sV5_cat,
        "oof_y": y_cat, "oof_pi": pi_cat,
        "src_pat_hours": src_pat_hours, "src_pat_totF": src_pat_totF,
        "tot_b": tot_b, "tot_f": tot_f,
        "sig1": float(np.std(s1_cat)) + 1e-9,
        "sigN": float(np.std(sN_cat)) + 1e-9,
        "sigV3": float(np.std(sV3_cat)) + 1e-9,
        "sigV5": float(np.std(sV5_cat)) + 1e-9,
        "sc1": sc1_f, "clf1": clf1_f,
        "scN": scN_f, "clfN": clfN_f,
        "scV3": scV3_f, "clfV3": clfV3_f,
        "scV5": scV5_f, "clfV5": clfV5_f,
    }


def solve_r2_1d_threshold(scores, y_cls, pat_idx, src_pat_hours, src_pat_totF, budget_faph, fp_classes):
    """
    Exact 1D threshold solver for Rule R2:
      Maximizes source MacroPatRec(F+ patients) subject to:
        MeanPat_FAPH_all_src(tau) = (1 / |P_all|) * sum_{p in P_all} (FP_p(tau) / H_p) <= budget_faph
      where P_all includes ALL source patients (both F+ and zero-F patients, including 0-S2 patients!).
    Handles tied scores strictly so pred = (scores >= tau) satisfies the constraint exactly.
    """
    n_all = len(src_pat_hours)
    if len(scores) == 0:
        return 1e6
    ord_i = np.argsort(scores)[::-1]
    s_o = scores[ord_i]
    y_o = y_cls[ord_i]
    p_o = pat_idx[ord_i]

    is_fp = np.isin(y_o, list(fp_classes))
    faph_inc = np.where(is_fp, 1.0 / (np.maximum(1e-6, src_pat_hours[p_o]) * float(n_all)), 0.0)
    cum_faph = np.cumsum(faph_inc)

    # Check only group-end indices of distinct score values so (scores >= tau) matches cum_faph exactly
    grp_end = np.empty(len(s_o), dtype=bool)
    grp_end[:-1] = (s_o[:-1] != s_o[1:])
    grp_end[-1] = True

    ok = np.where(grp_end & (cum_faph <= budget_faph + 1e-12))[0]
    if len(ok) == 0:
        return float(s_o[0]) + 1.0
    # Because MacroPatRec(tau) is monotonically non-increasing in tau, the smallest feasible tau
    # (largest prefix index ok[-1]) achieves the global maximum of MacroPatRec(F+) under the budget!
    return float(s_o[ok[-1]])


def summarize_source_oof_op(pred_mask, y_cls, pat_idx, src_pat_hours, src_pat_totF):
    """Exact source-OOF metrics at a calibrated operating point across ALL source patients."""
    n_all = len(src_pat_hours)
    fp_p = np.zeros(n_all, dtype=np.int64)
    fp_ns_p = np.zeros(n_all, dtype=np.int64)
    fp_v_p = np.zeros(n_all, dtype=np.int64)
    tp_p = np.zeros(n_all, dtype=np.int64)

    if np.any(pred_mask):
        pm_y = y_cls[pred_mask]
        pm_p = pat_idx[pred_mask]
        np.add.at(tp_p, pm_p[pm_y == 2], 1)
        np.add.at(fp_p, pm_p[pm_y != 2], 1)
        np.add.at(fp_ns_p, pm_p[(pm_y == 0) | (pm_y == 3)], 1)
        np.add.at(fp_v_p, pm_p[pm_y == 1], 1)

    faph_p = fp_p / np.maximum(1e-6, src_pat_hours)
    fpos = (src_pat_totF > 0)
    zpos = (src_pat_totF == 0)
    rec_fpos = 100.0 * tp_p[fpos] / np.maximum(1, src_pat_totF[fpos])
    return {
        "src_MacroPat_Rec_%": round(float(np.mean(rec_fpos)), 2) if np.any(fpos) else 0.0,
        "src_MeanPat_FAPH_All": round(float(np.mean(faph_p)), 3),
        "src_MeanPat_FAPH_ZeroF": round(float(np.mean(faph_p[zpos])), 3) if np.any(zpos) else 0.0,
        "src_Pooled_FAPH": round(float(np.sum(fp_p) / max(1e-6, float(np.sum(src_pat_hours)))), 3),
        "src_FracPats_LE2_%": round(100.0 * float(np.mean(faph_p <= 2.0 + 1e-9)), 1),
        "src_TP": int(np.sum(tp_p)),
        "src_FP_NS": int(np.sum(fp_ns_p)),
        "src_FP_V": int(np.sum(fp_v_p)),
    }


def calibrate_all_rules(bundle):
    """Applies BOTH R1 and R2 to the exact same fitted models & source OOF scores."""
    s1 = bundle["oof_s1"]; sN = bundle["oof_sN"]
    sV3 = bundle["oof_sV3"]; sV5 = bundle["oof_sV5"]
    y = bundle["oof_y"]; y_bin = (y == 2).astype(int)
    pi = bundle["oof_pi"]
    H = bundle["src_pat_hours"]; totF = bundle["src_pat_totF"]
    tot_b = bundle["tot_b"]; tot_f = bundle["tot_f"]

    # --- RULE R1: Current Baseline (Pooled Prior-Adjusted F1, exact v46/v47 parity!) ---
    ord1 = np.argsort(s1)[::-1]
    tp1 = np.cumsum(y_bin[ord1] == 1).astype(float); fp1 = np.cumsum(y_bin[ord1] == 0).astype(float)
    pi_src = float(tot_f) / max(1.0, float(tot_b))
    neg_mult = (pi_src / max(1e-9, 1.0 - pi_src)) / (0.001 / 0.999)
    tau1_r1 = float(s1[ord1[int(np.argmax(2.0 * tp1 / (float(tot_f) + tp1 + fp1 * neg_mult + 1e-12)))]])
    tN3_r1, tV3_r1, _ = v46.joint_oof_2d_threshold(sN, sV3, y_bin, tot_f, tot_b, n_grid=140)
    tN5_r1, tV5_r1, _ = v46.joint_oof_2d_threshold(sN, sV5, y_bin, tot_f, tot_b, n_grid=140)

    # --- RULE R2: Patient-Level Macro-Recall under Pre-Specified Source FA Budget B = 2.0 FAPH ---
    tau1_r2 = solve_r2_1d_threshold(s1, y, pi, H, totF, B_TOTAL_FAPH, fp_classes=(0, 1, 3))
    # Shared pure-geometry F:N/S control threshold (IDENTICAL & FROZEN for both C3 and C5!)
    tauN_ctrl_r2 = solve_r2_1d_threshold(sN, y, pi, H, totF, B_NS_FAPH, fp_classes=(0, 3))
    passN_r2 = (sN >= tauN_ctrl_r2)
    tV3_r2 = solve_r2_1d_threshold(sV3[passN_r2], y[passN_r2], pi[passN_r2], H, totF, B_TOTAL_FAPH, fp_classes=(0, 1, 3))
    tV5_r2 = solve_r2_1d_threshold(sV5[passN_r2], y[passN_r2], pi[passN_r2], H, totF, B_TOTAL_FAPH, fp_classes=(0, 1, 3))

    # Source-OOF verification & strict assertions
    src_r1_c1 = summarize_source_oof_op(s1 >= tau1_r1, y, pi, H, totF)
    src_r1_c3 = summarize_source_oof_op((sN >= tN3_r1) & (sV3 >= tV3_r1), y, pi, H, totF)
    src_r1_c5 = summarize_source_oof_op((sN >= tN5_r1) & (sV5 >= tV5_r1), y, pi, H, totF)

    src_r2_c1 = summarize_source_oof_op(s1 >= tau1_r2, y, pi, H, totF)
    src_r2_c3 = summarize_source_oof_op(passN_r2 & (sV3 >= tV3_r2), y, pi, H, totF)
    src_r2_c5 = summarize_source_oof_op(passN_r2 & (sV5 >= tV5_r2), y, pi, H, totF)
    src_r2_ns_only = summarize_source_oof_op(passN_r2, y, pi, H, totF)

    # Assert exact Rule R2 budget compliance on source OOF!
    assert src_r2_c1["src_MeanPat_FAPH_All"] <= B_TOTAL_FAPH + 1e-3, f"R2 C1 FA violated: {src_r2_c1}"
    assert src_r2_c3["src_MeanPat_FAPH_All"] <= B_TOTAL_FAPH + 1e-3, f"R2 C3 FA violated: {src_r2_c3}"
    assert src_r2_c5["src_MeanPat_FAPH_All"] <= B_TOTAL_FAPH + 1e-3, f"R2 C5 FA violated: {src_r2_c5}"

    def mk_c1(tau, src_stat):
        return {
            "mode": "onehead", "feat_idx": F2_IDX,
            "scaler": bundle["sc1"], "clf": bundle["clf1"],
            "tau1": tau, "sig": bundle["sig1"], "src_stat": src_stat,
        }

    def mk_2b(idx_v, scV, clfV, sigV, tN, tV, src_stat):
        return {
            "mode": "twobranch", "feat_idx_N": F2_IDX, "feat_idx_V": idx_v,
            "scN": bundle["scN"], "clfN": bundle["clfN"], "tau_N": tN, "sig_N": bundle["sigN"],
            "scV": scV, "clfV": clfV, "tau_V": tV, "sig_V": sigV, "src_stat": src_stat,
        }

    return {
        "R1_PooledPriorF1": {
            "C1_OneHead_Geom": mk_c1(tau1_r1, src_r1_c1),
            "C3_TwoBranch_FV_SignedRR": mk_2b(IDX_C3, bundle["scV3"], bundle["clfV3"], bundle["sigV3"], tN3_r1, tV3_r1, src_r1_c3),
            "C5_Causal_Unlabeled_PreQRS_Pwin": mk_2b(IDX_C5, bundle["scV5"], bundle["clfV5"], bundle["sigV5"], tN5_r1, tV5_r1, src_r1_c5),
        },
        "R2_PatMacroRec_FA2": {
            "C1_OneHead_Geom": mk_c1(tau1_r2, src_r2_c1),
            "C3_TwoBranch_FV_SignedRR": mk_2b(IDX_C3, bundle["scV3"], bundle["clfV3"], bundle["sigV3"], tauN_ctrl_r2, tV3_r2, src_r2_c3),
            "C5_Causal_Unlabeled_PreQRS_Pwin": mk_2b(IDX_C5, bundle["scV5"], bundle["clfV5"], bundle["sigV5"], tauN_ctrl_r2, tV5_r2, src_r2_c5),
            "_shared_tauN_ctrl": round(tauN_ctrl_r2, 4),
            "_shared_NS_only_src": src_r2_ns_only,
        },
    }


def eval_target_v48(te_items, model_obj):
    """
    Evaluates a source-locked model on target DB, reporting:
      - Deployed T1 per-patient recall (T1_MacroPat_Rec_%) across target F+ patients
      - Realized target FA transfer across ALL patients & Zero-F patients
      - Separated FP_NS and FP_V, TP, F1, Prec, Rec, Spec, Branch_Diag
      - AP, DB_Rank@F2_%, MacroPat_Pat@F2_%
      - Full per-patient T1 dict (Rec_%, FP, FP_NS, FP_V, FAPH, tot_F, hours) for helped/harmed audit!
    """
    tot_hours = sum(it["post_hours"] for it in te_items)
    sc_l, y_l, rec_sc = [], [], []
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

    # Per-patient deployed T1 & oracle Pat@F2
    pat_data = {}
    for it, sc in rec_sc:
        p = it["patient_id"]
        d = pat_data.setdefault(p, {"sc": [], "y": [], "h": 0.0, "db": it["db"]})
        d["sc"].append(sc); d["y"].append(it["y"]); d["h"] += it["post_hours"]

    pat_t1 = {}
    pat_rf2 = {}
    fpos_recs_t1 = []
    all_faphs = []
    zf_faphs = []
    for p, pd in pat_data.items():
        psc = np.concatenate(pd["sc"]); py = np.concatenate(pd["y"]); pyb = (py == 2).astype(int)
        pred_p = (psc >= 0.0)
        tp_p = int(np.sum(pred_p & (py == 2)))
        fp_ns_p = int(np.sum(pred_p & ((py == 0) | (py == 3))))
        fp_v_p = int(np.sum(pred_p & (py == 1)))
        fp_p = fp_ns_p + fp_v_p
        faph_p = fp_p / max(1e-6, pd["h"])
        nf_p = int(np.sum(pyb))
        all_faphs.append(faph_p)
        if nf_p == 0:
            zf_faphs.append(faph_p)
        else:
            rec_t1_p = 100.0 * tp_p / nf_p
            fpos_recs_t1.append(rec_t1_p)
            pat_rf2[p] = v43.recall_at_fp_budget(psc, pyb, int(np.floor(2.0 * pd["h"])))
        pat_t1[p] = {
            "db": pd["db"], "hours": round(pd["h"], 2), "tot_F": nf_p,
            "TP": tp_p, "Rec_%": round(100.0 * tp_p / nf_p, 2) if nf_p > 0 else None,
            "FP": fp_p, "FP_NS": fp_ns_p, "FP_V": fp_v_p, "FAPH": round(faph_p, 2),
        }

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
        "MacroPat_Pat@F2_%": round(float(np.mean(list(pat_rf2.values()))), 2) if pat_rf2 else 0.0,
        "T1_MacroPat_Rec_%": round(float(np.mean(fpos_recs_t1)), 2) if fpos_recs_t1 else 0.0,
        "T1_MeanPat_FAPH_All": round(float(np.mean(all_faphs)), 2),
        "T1_MeanPat_FAPH_ZeroF": round(float(np.mean(zf_faphs)), 2) if zf_faphs else None,
        "T1_FracPats_LE2_%": round(100.0 * float(np.mean(np.array(all_faphs) <= 2.0 + 1e-9)), 1),
        "T1_FracZeroF_LE2_%": round(100.0 * float(np.mean(np.array(zf_faphs) <= 2.0 + 1e-9)), 1) if zf_faphs else None,
        "T1_Pooled_FAPH": round(faph1, 2),
        "T1_F1": round(f1_t1, 4),
        "T1_Prec_%": round(prec1, 2),
        "T1_Rec_%": round(rec1, 2),
        "T1_Spec_%": round(spec, 3),
        "T1_TP": tp1, "T1_FP_NS": fp_ns, "T1_FPN": fpn1, "T1_FPS": fps1, "T1_FPV": fpv1,
        "T1_Counts": f"{tp1}/{fpn1}/{fpv1}/{fps1}S",
        "Branch_Diag": dict(b_diag),
        "Source_OOF": model_obj["src_stat"],
        "Threshold": round(model_obj["tau1"], 4) if model_obj["mode"] == "onehead" else [round(model_obj["tau_N"], 4), round(model_obj["tau_V"], 4)],
    }
    return ev, pat_t1, pat_rf2


def main():
    t0 = time.time()
    dbs = {db: v47.build_or_load_v47_db(db) for db in DB_ORDER}

    summary = {
        "Config": {
            "B_TOTAL_FAPH": B_TOTAL_FAPH,
            "B_NS_FAPH": B_NS_FAPH,
            "Models": CFG_ORDER,
            "Rules": RULE_ORDER,
        },
        "Models": {},
    }

    for mtype in CLF_FAMILIES:
        print(f"\n==================== CLASSIFIER FAMILY: {mtype} (R1 vs R2 CALIBRATION, LOCKED K=4) ====================", flush=True)
        summary["Models"][mtype] = {"ByDB": {}, "Global_Summary": {}, "PerPatient_T1": {}, "PerPatient_PatF2": {}}

        # Global accumulators across all 6 DBs (263 total patients, 67 F+ patients)
        glob_t1 = {r: {c: {} for c in CFG_ORDER} for r in RULE_ORDER}
        glob_rf2 = {r: {c: {} for c in CFG_ORDER} for r in RULE_ORDER}

        for tgt in DB_ORDER:
            tr = [it for db in DB_ORDER if db != tgt for it in dbs[db]]
            te = dbs[tgt]
            bundle = fit_fold_models_and_oof(tr, mtype)
            cal = calibrate_all_rules(bundle)

            # Assert bit-identical tau_N_ctrl between C3 and C5 under R2!
            assert cal["R2_PatMacroRec_FA2"]["C3_TwoBranch_FV_SignedRR"]["tau_N"] == cal["R2_PatMacroRec_FA2"]["C5_Causal_Unlabeled_PreQRS_Pwin"]["tau_N"]

            db_res = {"R1_PooledPriorF1": {}, "R2_PatMacroRec_FA2": {}, "R2_Shared_NS_Ctrl": {
                "tau_N_ctrl": cal["R2_PatMacroRec_FA2"]["_shared_tauN_ctrl"],
                "src_NS_only": cal["R2_PatMacroRec_FA2"]["_shared_NS_only_src"],
            }}
            for rule in RULE_ORDER:
                for cname in CFG_ORDER:
                    ev, pt1, prf2 = eval_target_v48(te, cal[rule][cname])
                    db_res[rule][cname] = ev
                    glob_t1[rule][cname].update(pt1)
                    glob_rf2[rule][cname].update(prf2)

            # Sanity assertion: C1 AP is invariant between R1 and R2!
            ap_r1 = db_res["R1_PooledPriorF1"]["C1_OneHead_Geom"]["AP"]
            ap_r2 = db_res["R2_PatMacroRec_FA2"]["C1_OneHead_Geom"]["AP"]
            assert abs(ap_r1 - ap_r2) < 1e-12, f"C1 AP changed between R1 and R2 on {tgt}: {ap_r1} vs {ap_r2}"

            summary["Models"][mtype]["ByDB"][tgt] = db_res

            print(f"\n  [{tgt}] ({mtype}) | R2 shared tau_N_ctrl={db_res['R2_Shared_NS_Ctrl']['tau_N_ctrl']:+.4f} (src NS_FAPH={db_res['R2_Shared_NS_Ctrl']['src_NS_only']['src_MeanPat_FAPH_All']:.2f})", flush=True)
            for cname in CFG_ORDER:
                for rule in RULE_ORDER:
                    ev = db_res[rule][cname]
                    rtag = "R1" if rule.startswith("R1") else "R2"
                    fv_o = ev["Branch_Diag"].get("F_fail_V_only", "—")
                    th_s = f"{ev['Threshold']:+.2f}" if isinstance(ev["Threshold"], float) else f"({ev['Threshold'][0]:+.2f},{ev['Threshold'][1]:+.2f})"
                    print(
                        f"    {cname:31s} [{rtag}] th={th_s:13s} | AP={ev['AP']:.4f} Pat@F2={ev['MacroPat_Pat@F2_%']:4.1f}% | "
                        f"T1: PatRec={ev['T1_MacroPat_Rec_%']:4.1f}% F1={ev['T1_F1']:.4f} TP={ev['T1_TP']:3d} "
                        f"FP_NS={ev['T1_FP_NS']:4d} FP_V={ev['T1_FPV']:4d} | "
                        f"FAPH(All/zF/Pool)={ev['T1_MeanPat_FAPH_All']:4.2f}/{str(ev['T1_MeanPat_FAPH_ZeroF']):>4s}/{ev['T1_Pooled_FAPH']:5.2f} "
                        f"<=2:{ev['T1_FracPats_LE2_%']:4.1f}% | F_failV={fv_o}",
                        flush=True,
                    )

        # Global 67-F+ and All-Patient summaries
        def summarize_global_rule_cfg(pt1_map, prf2_map):
            fpos_recs = [v["Rec_%"] for v in pt1_map.values() if v["tot_F"] > 0]
            all_faphs = [v["FAPH"] for v in pt1_map.values()]
            zf_faphs = [v["FAPH"] for v in pt1_map.values() if v["tot_F"] == 0]
            tot_tp = sum(v["TP"] for v in pt1_map.values())
            tot_f = sum(v["tot_F"] for v in pt1_map.values())
            tot_fp_ns = sum(v["FP_NS"] for v in pt1_map.values())
            tot_fp_v = sum(v["FP_V"] for v in pt1_map.values())
            tot_h = sum(v["hours"] for v in pt1_map.values())
            return {
                "n_all_pats": len(pt1_map),
                "n_fpos_pats": len(fpos_recs),
                "n_zf_pats": len(zf_faphs),
                "T1_MacroPat_Rec_%": round(float(np.mean(fpos_recs)), 2),
                "MacroPat_Pat@F2_%": round(float(np.mean(list(prf2_map.values()))), 2),
                "T1_MeanPat_FAPH_All": round(float(np.mean(all_faphs)), 2),
                "T1_MeanPat_FAPH_ZeroF": round(float(np.mean(zf_faphs)), 2),
                "T1_FracPats_LE2_%": round(100.0 * float(np.mean(np.array(all_faphs) <= 2.0 + 1e-9)), 1),
                "T1_FracZeroF_LE2_%": round(100.0 * float(np.mean(np.array(zf_faphs) <= 2.0 + 1e-9)), 1),
                "T1_Pooled_FAPH": round((tot_fp_ns + tot_fp_v) / max(1e-6, tot_h), 2),
                "T1_TP": tot_tp, "T1_FP_NS": tot_fp_ns, "T1_FP_V": tot_fp_v,
                "T1_Pooled_F1": round(2.0 * tot_tp / max(1, 2 * tot_tp + tot_fp_ns + tot_fp_v + (tot_f - tot_tp)), 4),
            }

        def compare_t1_wtl(ma, mb):
            """Compares F+ patient T1 recall (and notes FA trade-off!) between A and B."""
            fpos = [p for p in ma if ma[p]["tot_F"] > 0]
            w_rec = sum(1 for p in fpos if ma[p]["Rec_%"] > mb[p]["Rec_%"] + 1e-6)
            l_rec = sum(1 for p in fpos if ma[p]["Rec_%"] < mb[p]["Rec_%"] - 1e-6)
            t_rec = len(fpos) - w_rec - l_rec
            # Strict dominance (better or equal recall AND lower or equal FAPH, strictly better on at least one)
            dom_a = sum(1 for p in fpos if (ma[p]["Rec_%"] >= mb[p]["Rec_%"] - 1e-6 and ma[p]["FAPH"] <= mb[p]["FAPH"] + 1e-6) and (ma[p]["Rec_%"] > mb[p]["Rec_%"] + 1e-6 or ma[p]["FAPH"] < mb[p]["FAPH"] - 1e-6))
            dom_b = sum(1 for p in fpos if (mb[p]["Rec_%"] >= ma[p]["Rec_%"] - 1e-6 and mb[p]["FAPH"] <= ma[p]["FAPH"] + 1e-6) and (mb[p]["Rec_%"] > ma[p]["Rec_%"] + 1e-6 or mb[p]["FAPH"] < ma[p]["FAPH"] - 1e-6))
            # Zero-F patients helped (lower FAPH) vs harmed (higher FAPH)
            zf = [p for p in ma if ma[p]["tot_F"] == 0]
            zf_better = sum(1 for p in zf if ma[p]["FAPH"] < mb[p]["FAPH"] - 1e-6)
            zf_worse = sum(1 for p in zf if ma[p]["FAPH"] > mb[p]["FAPH"] + 1e-6)
            return {
                "Fpos_Rec_WTL": f"{w_rec}W/{t_rec}T/{l_rec}L",
                "Fpos_StrictDom_A_vs_B": f"{dom_a}_vs_{dom_b}",
                "ZeroF_FA_Better_vs_Worse": f"{zf_better}_vs_{zf_worse} (of {len(zf)})",
            }

        def compare_rf2_wtl(ra, rb):
            w = sum(1 for p in ra if ra[p] > rb[p] + 1e-6)
            l = sum(1 for p in ra if ra[p] < rb[p] - 1e-6)
            t = len(ra) - w - l
            return f"{w}W/{t}T/{l}L"

        print(f"\n  === GLOBAL ALL-6-DB SUMMARY ({mtype}) ===", flush=True)
        gsum = {"ByRuleCfg": {}, "Comparisons": {}}
        for rule in RULE_ORDER:
            for cname in CFG_ORDER:
                s = summarize_global_rule_cfg(glob_t1[rule][cname], glob_rf2[rule][cname])
                gsum["ByRuleCfg"][f"{rule}__{cname}"] = s
                rtag = "R1" if rule.startswith("R1") else "R2"
                print(
                    f"    [{rtag}] {cname:31s} | T1_PatRec={s['T1_MacroPat_Rec_%']:5.2f}% Pat@F2={s['MacroPat_Pat@F2_%']:4.2f}% | "
                    f"FAPH(All/zF/Pool)={s['T1_MeanPat_FAPH_All']:4.2f}/{s['T1_MeanPat_FAPH_ZeroF']:4.2f}/{s['T1_Pooled_FAPH']:4.2f} "
                    f"<=2(All/zF)={s['T1_FracPats_LE2_%']:4.1f}%/{s['T1_FracZeroF_LE2_%']:4.1f}% | "
                    f"TP={s['T1_TP']:4d} FP_NS={s['T1_FP_NS']:4d} FP_V={s['T1_FP_V']:4d}",
                    flush=True,
                )

        print(f"\n  === PAIRWISE GLOBAL COMPARISONS ({mtype}) ===", flush=True)
        pairs = [
            ("R2_vs_R1__C1", ("R2_PatMacroRec_FA2", "C1_OneHead_Geom"), ("R1_PooledPriorF1", "C1_OneHead_Geom")),
            ("R2_vs_R1__C3", ("R2_PatMacroRec_FA2", "C3_TwoBranch_FV_SignedRR"), ("R1_PooledPriorF1", "C3_TwoBranch_FV_SignedRR")),
            ("R2_vs_R1__C5", ("R2_PatMacroRec_FA2", "C5_Causal_Unlabeled_PreQRS_Pwin"), ("R1_PooledPriorF1", "C5_Causal_Unlabeled_PreQRS_Pwin")),
            ("Under_R2__C3_vs_C1", ("R2_PatMacroRec_FA2", "C3_TwoBranch_FV_SignedRR"), ("R2_PatMacroRec_FA2", "C1_OneHead_Geom")),
            ("Under_R2__C5_vs_C1", ("R2_PatMacroRec_FA2", "C5_Causal_Unlabeled_PreQRS_Pwin"), ("R2_PatMacroRec_FA2", "C1_OneHead_Geom")),
            ("Under_R2__C5_vs_C3", ("R2_PatMacroRec_FA2", "C5_Causal_Unlabeled_PreQRS_Pwin"), ("R2_PatMacroRec_FA2", "C3_TwoBranch_FV_SignedRR")),
            ("Under_R1__C5_vs_C3", ("R1_PooledPriorF1", "C5_Causal_Unlabeled_PreQRS_Pwin"), ("R1_PooledPriorF1", "C3_TwoBranch_FV_SignedRR")),
        ]
        for label, (ra, ca), (rb, cb) in pairs:
            c_t1 = compare_t1_wtl(glob_t1[ra][ca], glob_t1[rb][cb])
            c_f2 = compare_rf2_wtl(glob_rf2[ra][ca], glob_rf2[rb][cb])
            gsum["Comparisons"][label] = {**c_t1, "Pat@F2_WTL": c_f2}
            print(
                f"    {label:20s} | T1_Rec_WTL={c_t1['Fpos_Rec_WTL']:10s} StrictDom={c_t1['Fpos_StrictDom_A_vs_B']:10s} "
                f"ZeroF_FA={c_t1['ZeroF_FA_Better_vs_Worse']:18s} | Pat@F2_WTL={c_f2}",
                flush=True,
            )

        summary["Models"][mtype]["Global_Summary"] = gsum
        summary["Models"][mtype]["PerPatient_T1"] = glob_t1
        summary["Models"][mtype]["PerPatient_PatF2"] = {
            r: {c: {p: round(v, 2) for p, v in glob_rf2[r][c].items()} for c in CFG_ORDER} for r in RULE_ORDER
        }

    out_json = os.path.join(AUD_DIR, "V48_STAGE2_OPERATING_POINT_CALIBRATION_AUDIT_20261122.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_md = os.path.join(AUD_DIR, "V48_STAGE2_OPERATING_POINT_CALIBRATION_AUDIT_20261122.md")
    lines = [
        "# v48 Final Bounded Stage-2 Operating-Point Calibration Audit (`R1` vs `R2` on `C1`, `C3`, `C5`)",
        "",
        "- **Strict Symmetry & Zero Target Peeking**: Exact same fitted models & 5-fold patient-grouped source OOF predictions evaluated under exactly two calibration rules on locked `K=4` Static baseline (`0` new features, `0` oracle features):",
        "  - **`R1_PooledPriorF1`**: Existing pooled prior-adjusted F1 (`pi_target=0.001`) on source OOF (`tau1*` for `C1`, joint 2D `(tau_N*, tau_V*)` for `C3` & `C5`).",
        f"  - **`R2_PatMacroRec_FA2`**: Patient-level macro-recall on source `F+` patients subject to `MeanPat_FAPH_all_src <= {B_TOTAL_FAPH:.1f} FAPH` across **ALL source patients (including zero-F patients)**. For `C3` & `C5`, shared pure-geometry `F:N/S` threshold `tau_N_ctrl` is fixed ONCE (`MeanPat_FAPH_NS_all_src <= {B_NS_FAPH:.1f} FAPH`, bit-identical for `C3` & `C5`), and ONLY `tau_V*` is chosen under the `{B_TOTAL_FAPH:.1f} FAPH` total budget.",
        "",
    ]

    for mtype in CLF_FAMILIES:
        lines.extend([
            f"## 1. Per-Hospital LODO Results (`{mtype}`, `C=0.3`, Locked `K=4`)", "",
            "| DB | Model | Rule | AP | Pat@F2 (%) | **`T1` PatRec (%)** | `T1` F1 | `T1` TP | **`T1` FP_NS** | **`T1` FP_V** | `T1` MeanPat FAPH (All / zF / Pool) | `% Pats ≤ 2 FAPH` | `F_fail_V_only` |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ])
        for tgt in DB_ORDER:
            dbr = summary["Models"][mtype]["ByDB"][tgt]
            for cname in CFG_ORDER:
                for rule in RULE_ORDER:
                    ev = dbr[rule][cname]
                    rtag = "R1" if rule.startswith("R1") else "R2"
                    fv_o = ev["Branch_Diag"].get("F_fail_V_only", "—")
                    zf_s = f"{ev['T1_MeanPat_FAPH_ZeroF']:.2f}" if ev["T1_MeanPat_FAPH_ZeroF"] is not None else "—"
                    lines.append(
                        f"| `{tgt}` | `{cname}` | `{rtag}` | **{ev['AP']:.4f}** | {ev['MacroPat_Pat@F2_%']:.1f} | "
                        f"**{ev['T1_MacroPat_Rec_%']:.1f}** | {ev['T1_F1']:.4f} | {ev['T1_TP']} | "
                        f"**{ev['T1_FP_NS']}** | **{ev['T1_FPV']}** | "
                        f"{ev['T1_MeanPat_FAPH_All']:.2f} / {zf_s} / {ev['T1_Pooled_FAPH']:.2f} | "
                        f"{ev['T1_FracPats_LE2_%']:.1f}% | {fv_o} |"
                    )
        lines.extend([
            "", f"## 2. Global 6-Hospital Summary (`{mtype}` — 67 `F+` / All Patients)", "",
            "| Rule & Model | **`T1` 67-Pat Rec (%)** | `Pat@F2` (%) | `T1` MeanPat FAPH (All / ZeroF / Pool) | `% Pats ≤ 2 FAPH` (All / ZeroF) | Total `T1` TP / FP_NS / FP_V |",
            "|---|---:|---:|---|---|---|",
        ])
        gs = summary["Models"][mtype]["Global_Summary"]["ByRuleCfg"]
        for rule in RULE_ORDER:
            for cname in CFG_ORDER:
                s = gs[f"{rule}__{cname}"]
                rtag = "R1" if rule.startswith("R1") else "R2"
                lines.append(
                    f"| `{rtag}` `{cname}` | **{s['T1_MacroPat_Rec_%']:.2f}%** | {s['MacroPat_Pat@F2_%']:.2f}% | "
                    f"**{s['T1_MeanPat_FAPH_All']:.2f}** / {s['T1_MeanPat_FAPH_ZeroF']:.2f} / {s['T1_Pooled_FAPH']:.2f} | "
                    f"{s['T1_FracPats_LE2_%']:.1f}% / {s['T1_FracZeroF_LE2_%']:.1f}% | "
                    f"{s['T1_TP']} / **{s['T1_FP_NS']}** / **{s['T1_FP_V']}** |"
                )
        lines.extend([
            "", f"### Pairwise WTL & Dominance (`{mtype}`)", "",
            "| Comparison (`A vs B`) | 67-Pat `T1` Rec WTL | `F+` Strict Dominance (`Rec≥ & FA≤`) | Zero-F FA (`Better vs Worse`) | `Pat@F2` WTL |",
            "|---|---|---|---|---|",
        ])
        for k, v in summary["Models"][mtype]["Global_Summary"]["Comparisons"].items():
            lines.append(
                f"| `{k}` | **{v['Fpos_Rec_WTL']}** | **{v['Fpos_StrictDom_A_vs_B']}** | {v['ZeroF_FA_Better_vs_Worse']} | {v['Pat@F2_WTL']} |"
            )
        lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nALL DONE in {time.time()-t0:.1f}s!\nSaved: {out_json}\nSaved: {out_md}", flush=True)


if __name__ == "__main__":
    main()
