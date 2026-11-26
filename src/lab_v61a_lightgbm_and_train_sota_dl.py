r"""
lab_v61a_lightgbm_and_train_sota_dl.py
======================================
PART 1: Runs `LightGBM_A3_TwoBranch` (nonlinear GBDT on exact frozen A3 15D+11D Stage-2
        features inside frozen Stage 1) under identical SubjectAware_251 + R3_B20 protocol
        across:
          - 6-DB LODO (comparable directly to Table V v58)
          - Source 6-DB 5-fold OOF (251 units)
          - Locked External LTSTDB (70 subjects, from cache_v59_ltstdb_a3)
          - Locked External MGHDB (213 Tier-1 / 198 Tier-2 / 86 K>0, from cache_v60_mghdb_a3)
        Also asserts bit-exact evaluator parity on frozen `A3_SVM` vs v59 and v60!

PART 2: Trains and SHA-256 seals all 4 published 2024-2025 MIT-licensed SOTA Deep Learning
        baselines strictly on the 6 source DBs (251 SubjectAware_251 units, 2,676,362 beats,
        4,600 Class-F beats) on RTX 3050 Ti GPU (FP16 AMP):
          1. Net1D_ECGFounder         (NEJM AI 2025, 2-lead + RR, 100% beats)
          2. ModernTCN_ICLR24         (ICLR 2024 Spotlight, 2-lead + RR, 100% beats)
          3. Medformer_NeurIPS24      (NeurIPS 2024, 2-lead + RR, 100% beats)
          4. PatCond_Net1D_ECGFounder (Patient-adapted [x_t, x_t - b_N_loc] + RR, 100% beats)
        Each model undergoes:
          - Exact 5-fold deterministic subject-grouped OOF (8 epochs/fold, 80k balanced +
            Stage-2 hard-negative beats/epoch -> full-stream OOF inference on all 2.68M beats)
          - Exact 1D R3_B20 threshold calibration (MeanPatFAPH_All <= 2.0 across 251 units)
          - Full 6-DB locked training (12 epochs x 80k beats) -> saved & SHA-256 sealed in
            `paper_1_mechanisms/checkpoints/v61_dl_<model_id>_6db.pt`.
        Resumable per model if already saved! ZERO target (LTSTDB / MGHDB) waveforms read in Part 2.
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
import gc
import json
import time
import pickle
import hashlib
import argparse
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v46_two_branch_stage2_rr as v46
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_diagnostic_v56_qrs_stt_local_chord_lodo as v56
import lab_v59_package_a3_checkpoint as v59pkg
import lab_v59_ltstdb_a3_prospective_eval as v59eval
import lab_v60_mghdb_a3_supplementary_eval as v60eval
import v61_sota_models as m61

REG_6DB = str(_REPO_ROOT / "registry/subject_mapping_6db_v1.json")
REG_LTSTDB = str(_REPO_ROOT / "registry/target_cohort_ltstdb_v1.json")
REG_MGHDB = str(_REPO_ROOT / "registry/target_cohort_mghdb_v1.json")
CONTRACT_V61 = str(_REPO_ROOT / "registry/v61_dl_sota_benchmark_contract_v1.json")
FROZEN_A3_PKL = str(_REPO_ROOT / "checkpoints/stage2_source6db_subjectaware251_a3_r3b20_v2.pkl")

CACHE_V37 = str(_DATA_ROOT / "cache_v37_waveforms_6db")
CKPT_DIR = str(_REPO_ROOT / "checkpoints")
AUD_DIR = str(_REPO_ROOT / "audits")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(AUD_DIR, exist_ok=True)

DB_ORDER = ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]
IDX_A3_N = list(v59pkg.IDX_A3_N)
IDX_C3_V = list(v59pkg.IDX_C3_V)
BN_SUB_STRIDE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# PART 1: LIGHTGBM-ON-A3 + FROZEN A3 EVALUATOR PARITY ASSERTION
# =============================================================================
def make_lgbm(seed=42):
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )


def score_twobranch_fullstream(items, score_fn_N, score_fn_V, tau_N, sig_N, tau_V, sig_V, sid_key="patient_id"):
    """Converts a Stage-2 two-branch model into full-stream scalar S_min items (non-S2 = -1e9)."""
    out = []
    for it in items:
        n_all = len(it["y"])
        sc = np.full(n_all, -1e9, dtype=np.float64)
        X = it["X_s2"]
        if len(X) > 0:
            sN = score_fn_N(X[:, IDX_A3_N])
            sV = score_fn_V(X[:, IDX_C3_V])
            mN = (sN - tau_N) / sig_N
            mV = (sV - tau_V) / sig_V
            sc[it["s2_mask"]] = np.minimum(mN, mV)
        out.append({
            "patient_id": it[sid_key],
            "post_hours": float(it["post_hours"]),
            "y": it["y"].astype(np.int8),
            "score": sc,
            "K": int(it.get("K", 0)),
        })
    return out


def fit_lgbm_twobranch_and_calibrate(src_items, cap_pat_pos=100, seed=42):
    pat_map = {}
    for it in src_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: (-sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]), p))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    pat_to_idx = {p: i for i, p in enumerate(pat_list)}
    n_pats = len(pat_list)
    pat_hours = np.zeros(n_pats, dtype=np.float64)
    pat_totF = np.zeros(n_pats, dtype=np.int64)
    for p, items in pat_map.items():
        pi = pat_to_idx[p]
        pat_hours[pi] = sum(float(it["post_hours"]) for it in items)
        pat_totF[pi] = sum(int(np.sum(it["y"] == 2)) for it in items)

    oof_sN_list, oof_sV_list, oof_y_list, oof_pi_list = [], [], [], []
    oof_rec_sN = {}
    oof_rec_sV = {}

    for fid in range(n_folds):
        tr_sub = [it for it in src_items if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in src_items if fold_of_pat[it["patient_id"]] == fid]

        XN, yN, wN = v46.build_branch_xy(tr_sub, IDX_A3_N, {0, 3}, cap_pat_pos=cap_pat_pos)
        clfN = make_lgbm(seed=seed)
        clfN.fit(XN, yN, sample_weight=wN)

        XV, yV, wV = v46.build_branch_xy(tr_sub, IDX_C3_V, {1}, cap_pat_pos=cap_pat_pos)
        clfV = make_lgbm(seed=seed)
        clfV.fit(XV, yV, sample_weight=wV)

        for te in te_sub:
            rkey = (te["db"], te["rec_id"])
            if len(te["X_s2"]) > 0:
                sN = clfN.predict(te["X_s2"][:, IDX_A3_N], raw_score=True)
                sV = clfV.predict(te["X_s2"][:, IDX_C3_V], raw_score=True)
                oof_sN_list.append(sN)
                oof_sV_list.append(sV)
                oof_y_list.append(te["y"][te["s2_mask"]].astype(np.int8))
                oof_pi_list.append(np.full(len(sN), pat_to_idx[te["patient_id"]], dtype=np.int32))
                oof_rec_sN[rkey] = sN
                oof_rec_sV[rkey] = sV
            else:
                oof_rec_sN[rkey] = np.zeros(0, dtype=np.float64)
                oof_rec_sV[rkey] = np.zeros(0, dtype=np.float64)

    sN_all = np.concatenate(oof_sN_list)
    sV_all = np.concatenate(oof_sV_list)
    y_all = np.concatenate(oof_y_list)
    pi_all = np.concatenate(oof_pi_list)
    sig_N = float(np.std(sN_all)) + 1e-9
    sig_V = float(np.std(sV_all)) + 1e-9

    tau_N, tau_V = v56.solve_r3_joint2d_patfa(
        sN_all, sV_all, y_all, pi_all, pat_hours, pat_totF,
        b_tot=2.0, b_0f=1e9, n_grid=85,
    )

    # Full fit on src_items
    XN, yN, wN = v46.build_branch_xy(src_items, IDX_A3_N, {0, 3}, cap_pat_pos=cap_pat_pos)
    full_clfN = make_lgbm(seed=seed)
    full_clfN.fit(XN, yN, sample_weight=wN)

    XV, yV, wV = v46.build_branch_xy(src_items, IDX_C3_V, {1}, cap_pat_pos=cap_pat_pos)
    full_clfV = make_lgbm(seed=seed)
    full_clfV.fit(XV, yV, sample_weight=wV)

    # Build full-stream OOF items for unified metric evaluation
    oof_stream = []
    for it in src_items:
        rkey = (it["db"], it["rec_id"])
        sc = np.full(len(it["y"]), -1e9, dtype=np.float64)
        if len(it["X_s2"]) > 0:
            mN = (oof_rec_sN[rkey] - tau_N) / sig_N
            mV = (oof_rec_sV[rkey] - tau_V) / sig_V
            sc[it["s2_mask"]] = np.minimum(mN, mV)
        oof_stream.append({
            "patient_id": it["patient_id"],
            "post_hours": float(it["post_hours"]),
            "y": it["y"].astype(np.int8),
            "score": sc,
        })

    return {
        "clfN": full_clfN, "clfV": full_clfV,
        "tau_N": float(tau_N), "sig_N": float(sig_N),
        "tau_V": float(tau_V), "sig_V": float(sig_V),
        "oof_stream": oof_stream,
    }


def run_part1_lightgbm_and_a3_parity(rec_to_sid):
    t0 = time.time()
    print("\n================ PART 1: FROZEN A3 PARITY + LIGHTGBM-ON-A3 ================", flush=True)
    all_src = v59pkg.load_source_6db_x29(rec_to_sid)
    assert len(all_src) == 306

    # Load frozen A3 v2 checkpoint and verify SHA256
    a3_sha = sha256_file(FROZEN_A3_PKL)
    assert a3_sha == "596867f150e9e79c917c16f2b2f328b64ce02c9f56c6068302fb89ff0763bb7a"
    with open(FROZEN_A3_PKL, "rb") as f:
        a3_ckpt = pickle.load(f)

    # Load cached LTSTDB (v59) and MGHDB (v60)
    lt_reg = json.load(open(REG_LTSTDB, "r", encoding="utf-8"))
    lt_items, _ = v59eval.load_cached_ltstdb_v2(lt_reg["included_records"])
    mgh_reg = json.load(open(REG_MGHDB, "r", encoding="utf-8"))
    mgh_items, _ = v60eval.load_cached_mghdb(mgh_reg["included_records"])

    # 1. Verify exact evaluator parity on frozen A3_SVM vs v59 (LTSTDB) and v60 (MGHDB)
    a3_svm = a3_ckpt["models"]["SVM"]["A3_HPhiW_NS_R3_B20"]
    a3_logr = a3_ckpt["models"]["LogR"]["A3_HPhiW_NS_R3_B20"]

    def _eval_linear_a3(m_obj, items, sid_key="patient_id"):
        stream = score_twobranch_fullstream(
            items,
            lambda x: m_obj["clfN"].decision_function(m_obj["scN"].transform(x)),
            lambda x: m_obj["clfV"].decision_function(m_obj["scV"].transform(x)),
            m_obj["tau_N"], m_obj["sig_N"], m_obj["tau_V"], m_obj["sig_V"],
            sid_key=sid_key,
        )
        return m61.evaluate_scalar_stream_cohort(stream, tau=0.0)

    a3_lt_svm_sum, a3_lt_svm_pt1, _, a3_lt_svm_mfa, a3_lt_svm_zf = _eval_linear_a3(a3_svm, lt_items, "patient_id")
    assert (a3_lt_svm_sum["T1_TP"], a3_lt_svm_sum["T1_FP_NS"], a3_lt_svm_sum["T1_FPV"]) == (42, 682, 53), a3_lt_svm_sum
    assert abs(a3_lt_svm_sum["T1_MacroPat_Rec_%"] - 8.24) < 0.02, a3_lt_svm_sum
    assert abs(a3_lt_svm_sum["T1_MeanPat_FAPH_All"] - 0.48) < 0.01, a3_lt_svm_sum
    assert abs(a3_lt_svm_sum["Match_All_2.0"] - 16.77) < 0.05, a3_lt_svm_sum

    a3_mgh1_svm_sum, a3_mgh1_svm_pt1, _, a3_mgh1_svm_mfa, a3_mgh1_svm_zf = _eval_linear_a3(a3_svm, mgh_items, "subject_id_tier1")
    assert (a3_mgh1_svm_sum["T1_TP"], a3_mgh1_svm_sum["T1_FP_NS"], a3_mgh1_svm_sum["T1_FPV"]) == (10, 157, 41), a3_mgh1_svm_sum
    assert abs(a3_mgh1_svm_sum["T1_MacroPat_Rec_%"] - 3.61) < 0.02, a3_mgh1_svm_sum
    assert abs(a3_mgh1_svm_sum["T1_MeanPat_FAPH_All"] - 0.82) < 0.01, a3_mgh1_svm_sum

    a3_mgh2_svm_sum, a3_mgh2_svm_pt1, _, _, _ = _eval_linear_a3(a3_svm, mgh_items, "subject_id_tier2")
    a3_lt_logr_sum, a3_lt_logr_pt1, _, a3_lt_logr_mfa, a3_lt_logr_zf = _eval_linear_a3(a3_logr, lt_items, "patient_id")
    a3_mgh1_logr_sum, a3_mgh1_logr_pt1, _, a3_mgh1_logr_mfa, a3_mgh1_logr_zf = _eval_linear_a3(a3_logr, mgh_items, "subject_id_tier1")
    print(f"  [PARITY PASS] Frozen A3_SVM evaluator matches v59 LTSTDB (42/682/53, M@2={a3_lt_svm_sum['Match_All_2.0']}%) and v60 MGHDB (10/157/41, Rec={a3_mgh1_svm_sum['T1_MacroPat_Rec_%']}%) BIT-FOR-BIT!", flush=True)

    # 2. 6-DB LODO for LightGBM_A3_TwoBranch (exact v58 comparison)
    print("  [LightGBM_A3] Running 6-DB LODO (251 SubjectAware_251 units)...", flush=True)
    lodo_stream_all = []
    lodo_per_db = {}
    for test_db in DB_ORDER:
        tr_db = [it for it in all_src if it["db"] != test_db]
        te_db = [it for it in all_src if it["db"] == test_db]
        fit_obj = fit_lgbm_twobranch_and_calibrate(tr_db, cap_pat_pos=100, seed=42)
        te_stream = score_twobranch_fullstream(
            te_db,
            lambda x, f=fit_obj: f["clfN"].predict(x, raw_score=True),
            lambda x, f=fit_obj: f["clfV"].predict(x, raw_score=True),
            fit_obj["tau_N"], fit_obj["sig_N"], fit_obj["tau_V"], fit_obj["sig_V"],
        )
        db_sum, _, _, _, _ = m61.evaluate_scalar_stream_cohort(te_stream, tau=0.0)
        lodo_per_db[test_db] = db_sum
        lodo_stream_all.extend(te_stream)
        print(f"    LODO {test_db:9s} | TP={db_sum['T1_TP']:4d} FP_NS={db_sum['T1_FP_NS']:4d} FP_V={db_sum['T1_FPV']:4d} | Rec={db_sum['T1_MacroPat_Rec_%']:5.2f}% FAPH={db_sum['T1_MeanPat_FAPH_All']:5.3f} AP={db_sum['AP']:.4f} M@2={db_sum['Match_All_2.0']:5.2f}%", flush=True)

    lodo_tot_sum, _, _, lodo_mfa, lodo_zf = m61.evaluate_scalar_stream_cohort(lodo_stream_all, tau=0.0)
    print(f"    LODO 6-DB ALL | TP={lodo_tot_sum['T1_TP']:4d} FP_NS={lodo_tot_sum['T1_FP_NS']:4d} FP_V={lodo_tot_sum['T1_FPV']:4d} | Rec={lodo_tot_sum['T1_MacroPat_Rec_%']:5.2f}% FAPH={lodo_tot_sum['T1_MeanPat_FAPH_All']:5.3f} (0F={lodo_tot_sum['T1_MeanPat_FAPH_ZeroF']:5.3f}) AP={lodo_tot_sum['AP']:.4f} Pat@F2={lodo_tot_sum['MacroPat_Pat@F2_%']:5.2f}% M@2={lodo_tot_sum['Match_All_2.0']:5.2f}%", flush=True)

    # 3. Full 6-DB fit + 5-fold OOF R3_B20 calibration -> Locked LTSTDB & MGHDB
    print("  [LightGBM_A3] Fitting Full 6-DB + 5-fold OOF R3_B20 -> Locked LTSTDB & MGHDB...", flush=True)
    lgbm_full = fit_lgbm_twobranch_and_calibrate(all_src, cap_pat_pos=100, seed=42)
    src_oof_sum, _, _, src_oof_mfa, src_oof_zf = m61.evaluate_scalar_stream_cohort(lgbm_full["oof_stream"], tau=0.0)

    # Save LightGBM checkpoint
    lgbm_ckpt_p = os.path.join(CKPT_DIR, "v61_lightgbm_a3_6db.pkl")
    with open(lgbm_ckpt_p, "wb") as f:
        pickle.dump({
            "clfN": lgbm_full["clfN"], "clfV": lgbm_full["clfV"],
            "tau_N": lgbm_full["tau_N"], "sig_N": lgbm_full["sig_N"],
            "tau_V": lgbm_full["tau_V"], "sig_V": lgbm_full["sig_V"],
        }, f)
    lgbm_sha = sha256_file(lgbm_ckpt_p)

    def _eval_lgbm(items, sid_key="patient_id"):
        stream = score_twobranch_fullstream(
            items,
            lambda x: lgbm_full["clfN"].predict(x, raw_score=True),
            lambda x: lgbm_full["clfV"].predict(x, raw_score=True),
            lgbm_full["tau_N"], lgbm_full["sig_N"], lgbm_full["tau_V"], lgbm_full["sig_V"],
            sid_key=sid_key,
        )
        return m61.evaluate_scalar_stream_cohort(stream, tau=0.0)

    lg_lt_sum, lg_lt_pt1, _, lg_lt_mfa, lg_lt_zf = _eval_lgbm(lt_items, "patient_id")
    lg_mgh1_sum, lg_mgh1_pt1, _, lg_mgh1_mfa, lg_mgh1_zf = _eval_lgbm(mgh_items, "subject_id_tier1")
    lg_mgh2_sum, lg_mgh2_pt1, _, lg_mgh2_mfa, lg_mgh2_zf = _eval_lgbm(mgh_items, "subject_id_tier2")

    # Active-anchor K>0 MGHDB subset (86 Tier-1 subjects with max_K > 0 across their records, 91 records)
    kpos_subjs = {it["subject_id_tier1"] for it in mgh_items if int(it.get("K", 0)) > 0}
    mgh_kpos = [it for it in mgh_items if it["subject_id_tier1"] in kpos_subjs]
    lg_mghk_sum, _, _, _, _ = _eval_lgbm(mgh_kpos, "subject_id_tier1")
    a3_mghk_sum, _, _, _, _ = _eval_linear_a3(a3_svm, mgh_kpos, "subject_id_tier1")

    boot_lt_a3_vs_lg = m61.paired_bootstrap_compare(a3_lt_svm_pt1, lg_lt_pt1)
    boot_mgh1_a3_vs_lg = m61.paired_bootstrap_compare(a3_mgh1_svm_pt1, lg_mgh1_pt1)

    print(f"    Source 5F-OOF | TP={src_oof_sum['T1_TP']:4d} FP_NS={src_oof_sum['T1_FP_NS']:4d} FP_V={src_oof_sum['T1_FPV']:4d} | Rec={src_oof_sum['T1_MacroPat_Rec_%']:5.2f}% FAPH={src_oof_sum['T1_MeanPat_FAPH_All']:5.3f} AP={src_oof_sum['AP']:.4f} M@2={src_oof_sum['Match_All_2.0']:5.2f}%", flush=True)
    print(f"    LTSTDB (70)   | TP={lg_lt_sum['T1_TP']:4d} FP_NS={lg_lt_sum['T1_FP_NS']:4d} FP_V={lg_lt_sum['T1_FPV']:4d} | F1={lg_lt_sum['T1_F1']:.4f} Rec={lg_lt_sum['T1_MacroPat_Rec_%']:5.2f}% FAPH={lg_lt_sum['T1_MeanPat_FAPH_All']:5.3f} (0F={lg_lt_sum['T1_MeanPat_FAPH_ZeroF']:5.3f} max={lg_lt_sum['T1_ZeroF_Max_FAPH']:.2f}) AP={lg_lt_sum['AP']:.4f} Pat@F2={lg_lt_sum['MacroPat_Pat@F2_%']:5.2f}% M@2={lg_lt_sum['Match_All_2.0']:5.2f}%", flush=True)
    print(f"    MGHDB T1(213) | TP={lg_mgh1_sum['T1_TP']:4d} FP_NS={lg_mgh1_sum['T1_FP_NS']:4d} FP_V={lg_mgh1_sum['T1_FPV']:4d} | F1={lg_mgh1_sum['T1_F1']:.4f} Rec={lg_mgh1_sum['T1_MacroPat_Rec_%']:5.2f}% FAPH={lg_mgh1_sum['T1_MeanPat_FAPH_All']:5.3f} (0F={lg_mgh1_sum['T1_MeanPat_FAPH_ZeroF']:5.3f} max={lg_mgh1_sum['T1_ZeroF_Max_FAPH']:.2f}) AP={lg_mgh1_sum['AP']:.4f} Pat@F2={lg_mgh1_sum['MacroPat_Pat@F2_%']:5.2f}% M@2={lg_mgh1_sum['Match_All_2.0']:5.2f}%", flush=True)

    # Also compute Source 5-fold OOF summary for frozen A3_SVM & A3_LogR for clean apples-to-apples table
    a3_src_oof = {
        "SVM": a3_ckpt["models"]["SVM"]["A3_HPhiW_NS_R3_B20"]["src_stat"],
        "LogR": a3_ckpt["models"]["LogR"]["A3_HPhiW_NS_R3_B20"]["src_stat"],
    }

    part1_out = {
        "Audit_Version": "v61a_lightgbm_a3_and_parity",
        "Elapsed_Sec": round(time.time() - t0, 1),
        "LightGBM_Ckpt_SHA256": lgbm_sha,
        "Frozen_A3_Reference": {
            "Source_OOF": a3_src_oof,
            "LTSTDB_SVM": {"Summary": a3_lt_svm_sum, "Matched_FA": a3_lt_svm_mfa, "ZeroF_Tail": a3_lt_svm_zf, "PerSubject": a3_lt_svm_pt1},
            "LTSTDB_LogR": {"Summary": a3_lt_logr_sum, "Matched_FA": a3_lt_logr_mfa, "ZeroF_Tail": a3_lt_logr_zf},
            "MGHDB_Tier1_SVM": {"Summary": a3_mgh1_svm_sum, "Matched_FA": a3_mgh1_svm_mfa, "ZeroF_Tail": a3_mgh1_svm_zf, "PerSubject": a3_mgh1_svm_pt1},
            "MGHDB_Tier2_SVM": {"Summary": a3_mgh2_svm_sum, "PerSubject": a3_mgh2_svm_pt1},
            "MGHDB_Kpos_SVM": a3_mghk_sum,
            "MGHDB_Tier1_LogR": {"Summary": a3_mgh1_logr_sum, "Matched_FA": a3_mgh1_logr_mfa, "ZeroF_Tail": a3_mgh1_logr_zf},
        },
        "LightGBM_A3_TwoBranch": {
            "LODO_6DB": {"Summary": lodo_tot_sum, "PerDB": lodo_per_db, "Matched_FA": lodo_mfa, "ZeroF_Tail": lodo_zf},
            "Source_5Fold_OOF": {"Summary": src_oof_sum, "Matched_FA": src_oof_mfa, "ZeroF_Tail": src_oof_zf},
            "LTSTDB": {"Summary": lg_lt_sum, "Matched_FA": lg_lt_mfa, "ZeroF_Tail": lg_lt_zf, "PerSubject": lg_lt_pt1},
            "MGHDB_Tier1": {"Summary": lg_mgh1_sum, "Matched_FA": lg_mgh1_mfa, "ZeroF_Tail": lg_mgh1_zf, "PerSubject": lg_mgh1_pt1},
            "MGHDB_Tier2": {"Summary": lg_mgh2_sum, "Matched_FA": lg_mgh2_mfa, "PerSubject": lg_mgh2_pt1},
            "MGHDB_Kpos": lg_mghk_sum,
            "Bootstrap_A3SVM_minus_LightGBM": {"LTSTDB": boot_lt_a3_vs_lg, "MGHDB_Tier1": boot_mgh1_a3_vs_lg},
        },
    }
    out_p = os.path.join(AUD_DIR, "V61A_LIGHTGBM_A3_RESULTS_20261123.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(part1_out, f, indent=2)
    print(f"  [PART 1 SAVED] {out_p} ({time.time() - t0:.1f}s)", flush=True)
    return part1_out


# =============================================================================
# PART 2: TRAIN 4 SOTA 2024-2025 DL MODELS ON 6-DB (5-FOLD OOF + FULL 6-DB)
# =============================================================================
def open_v37_source_mmaps(rec_to_sid):
    """Opens memory-mapped 6-DB waveform caches and attaches verified SubjectAware_251 IDs."""
    dbs = {}
    records = []
    tot_beats = 0
    tot_f = 0
    for db_i, db in enumerate(DB_ORDER):
        meta = json.load(open(os.path.join(CACHE_V37, f"{db}_meta.json"), "r", encoding="utf-8"))
        beats = np.load(os.path.join(CACHE_V37, f"{db}_beats.npy"), mmap_mode="r")
        rr3 = np.load(os.path.join(CACHE_V37, f"{db}_rr3.npy"), mmap_mode="r")
        bn_sub = np.load(os.path.join(CACHE_V37, f"{db}_bn_sub.npy"), mmap_mode="r")
        y = np.load(os.path.join(CACHE_V37, f"{db}_y.npy"), mmap_mode="r")
        s2 = np.load(os.path.join(CACHE_V37, f"{db}_s2.npy"), mmap_mode="r")
        dbs[db] = {"db_i": db_i, "beats": beats, "rr3": rr3, "bn_sub": bn_sub, "y": y, "s2": s2}
        for m in meta:
            sid = rec_to_sid[(db, m["rec_id"])]
            b0, b1 = int(m["b0"]), int(m["b1"])
            nb = b1 - b0
            assert int(m["w1"]) - int(m["w0"]) == nb + 2, f"w-len mismatch {db}/{m['rec_id']}"
            y_slice = np.asarray(y[b0:b1], dtype=np.int8)
            s2_slice = np.asarray(s2[b0:b1], dtype=bool)
            nf = int(np.sum(y_slice == 2))
            tot_beats += nb
            tot_f += nf
            records.append({
                "db": db,
                "db_i": db_i,
                "rec_id": m["rec_id"],
                "subject_id": sid,
                "post_hours": float(m["post_hours"]),
                "b0": b0, "b1": b1, "nb": nb,
                "w0": int(m["w0"]), "w1": int(m["w1"]),
                "bn0": int(m["bn0"]), "bn1": int(m["bn1"]),
                "n_F": nf,
                "idx_F": np.where(y_slice == 2)[0].astype(np.int32),
                "idx_N": np.where(y_slice == 0)[0].astype(np.int32),
                "idx_V": np.where(y_slice == 1)[0].astype(np.int32),
                "idx_S": np.where(y_slice == 3)[0].astype(np.int32),
                "idx_S2H": np.where(s2_slice & (y_slice != 2))[0].astype(np.int32),
            })
    assert tot_beats == 2676362 and tot_f == 4600, f"6-DB mismatch: {tot_beats}, {tot_f}"
    return dbs, records


def build_epoch_arrays(dbs, tr_recs, rng, use_patcond=False, rr_mean=None, rr_std=None,
                       n_f_draws=8000, n_norm=30000, n_v=18000, n_s=8000, n_s2h=16000, cap_pat_pos=100):
    """
    Dynamically samples 80,000 beats per epoch directly from memory-mapped 6-DB arrays:
      - Patient-capped Class-F sampling (cap_pat_pos=100, identical weighting to v46/v59)
      - Database-stratified N, V, S, and Stage-2 hard-negative non-F beats
    Returns compact contiguous arrays (x_arr float16, rr_arr float32, y_arr int64).
    """
    # 1. Patient-capped Class-F weighted sampling
    pat_f_cnt = Counter()
    for r in tr_recs:
        pat_f_cnt[r["subject_id"]] += r["n_F"]

    f_rec_idx = []
    f_loc_idx = []
    f_weights = []
    for ri, r in enumerate(tr_recs):
        if r["n_F"] > 0:
            pf = pat_f_cnt[r["subject_id"]]
            w_each = min(1.0, float(cap_pat_pos) / float(pf))
            f_rec_idx.append(np.full(r["n_F"], ri, dtype=np.int32))
            f_loc_idx.append(r["idx_F"])
            f_weights.append(np.full(r["n_F"], w_each, dtype=np.float64))
    f_rec_idx = np.concatenate(f_rec_idx)
    f_loc_idx = np.concatenate(f_loc_idx)
    f_weights = np.concatenate(f_weights)
    f_probs = f_weights / f_weights.sum()
    f_pick = rng.choice(len(f_rec_idx), size=n_f_draws, replace=True, p=f_probs)

    chosen_ri = [f_rec_idx[f_pick]]
    chosen_loc = [f_loc_idx[f_pick]]

    # 2. Database-stratified negative & S2-hard-negative sampling
    def sample_key(key, k_total):
        by_db = {}
        for ri, r in enumerate(tr_recs):
            arr = r[key]
            if len(arr) > 0:
                by_db.setdefault(r["db"], []).append((ri, arr))
        if not by_db:
            return
        per_db = max(1, k_total // len(by_db))
        for db_k, pairs in by_db.items():
            lens = np.array([len(a) for _, a in pairs], dtype=np.int64)
            tot = int(lens.sum())
            take = min(tot, per_db)
            flat = rng.choice(tot, size=take, replace=False)
            flat.sort()
            cum = np.cumsum(lens)
            # Map flat positions to (ri, loc)
            p_idx = np.searchsorted(cum, flat, side="right")
            starts = np.concatenate([[0], cum[:-1]])
            offsets = flat - starts[p_idx]
            ris = np.empty(take, dtype=np.int32)
            locs = np.empty(take, dtype=np.int32)
            for k_u in np.unique(p_idx):
                mask = (p_idx == k_u)
                ri_val, arr_val = pairs[int(k_u)]
                ris[mask] = ri_val
                locs[mask] = arr_val[offsets[mask]]
            chosen_ri.append(ris)
            chosen_loc.append(locs)

    sample_key("idx_N", n_norm)
    sample_key("idx_V", n_v)
    sample_key("idx_S", n_s)
    sample_key("idx_S2H", n_s2h)

    all_ri = np.concatenate(chosen_ri)
    all_loc = np.concatenate(chosen_loc)
    N = len(all_ri)

    in_ch = 4 if use_patcond else 2
    x_arr = np.empty((N, in_ch, 187), dtype=np.float16)
    rr_arr = np.empty((N, 3), dtype=np.float32)
    y_arr = np.empty(N, dtype=np.int64)

    # Vectorized gather per record for fast mmap reading
    order = np.argsort(all_ri)
    s_ri = all_ri[order]
    s_loc = all_loc[order]
    uniq_ri, split_idx = np.unique(s_ri, return_index=True)
    split_ends = np.append(split_idx[1:], N)

    cur = 0
    for u_ri, i0, i1 in zip(uniq_ri, split_idx, split_ends):
        r = tr_recs[int(u_ri)]
        dobj = dbs[r["db"]]
        locs = s_loc[i0:i1]
        k = len(locs)
        bi = r["b0"] + locs
        wi = r["w0"] + 1 + locs
        bx = np.asarray(dobj["beats"][wi], dtype=np.float16)
        if use_patcond:
            bni = np.minimum(r["bn0"] + (locs // BN_SUB_STRIDE), r["bn1"] - 1)
            bn = np.asarray(dobj["bn_sub"][bni], dtype=np.float16)
            x_arr[cur : cur + k, :2] = bx
            x_arr[cur : cur + k, 2:] = bx - bn
        else:
            x_arr[cur : cur + k] = bx
        rr_arr[cur : cur + k] = np.asarray(dobj["rr3"][bi], dtype=np.float32)
        y_arr[cur : cur + k] = np.asarray(dobj["y"][bi], dtype=np.int64)
        cur += k

    if rr_mean is not None and rr_std is not None:
        rr_arr = (rr_arr - rr_mean[None, :]) / rr_std[None, :]

    # Shuffle samples
    perm = rng.permutation(N)
    return x_arr[perm], rr_arr[perm], y_arr[perm]


def compute_rr_stats(dbs, tr_recs):
    # Fast subsample RR mean/std over training records
    vals = []
    for r in tr_recs:
        step = max(1, r["nb"] // 2000)
        vals.append(np.asarray(dbs[r["db"]]["rr3"][r["b0"] : r["b1"] : step], dtype=np.float64))
    arr = np.vstack(vals)
    return arr.mean(axis=0).astype(np.float32), (arr.std(axis=0) + 1e-6).astype(np.float32)


def train_one_dl_model(model_id, dbs, tr_recs, n_epochs, seed=42, batch_size=512, log_prefix=""):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    use_patcond = (model_id == "PatCond_Net1D_ECGFounder")
    rr_mean, rr_std = compute_rr_stats(dbs, tr_recs)

    model = m61.build_dl_model(model_id).to(DEVICE)
    class_weights = torch.tensor([1.0, 1.5, 2.5, 1.5], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Approximate total steps for CosineAnnealingLR
    steps_per_ep = 80000 // batch_size + 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs * steps_per_ep, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        x_np, rr_np, y_np = build_epoch_arrays(
            dbs, tr_recs, rng, use_patcond=use_patcond, rr_mean=rr_mean, rr_std=rr_std
        )
        model.train()
        tot_loss = 0.0
        n_batches = 0
        N = len(y_np)
        for i0 in range(0, N, batch_size):
            i1 = min(i0 + batch_size, N)
            xb = torch.from_numpy(x_np[i0:i1]).to(DEVICE, dtype=torch.float32, non_blocking=True)
            rrb = torch.from_numpy(rr_np[i0:i1]).to(DEVICE, dtype=torch.float32, non_blocking=True)
            yb = torch.from_numpy(y_np[i0:i1]).to(DEVICE, dtype=torch.long, non_blocking=True)

            # Mild cross-hospital augmentation during training only
            scale = 1.0 + 0.06 * (2.0 * torch.rand((xb.shape[0], 1, 1), device=DEVICE) - 1.0)
            xb = xb * scale + 0.02 * torch.randn_like(xb)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(xb, rrb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tot_loss += float(loss.item())
            n_batches += 1

        if ep == 1 or ep == n_epochs or ep % 4 == 0:
            print(f"    {log_prefix} ep {ep:2d}/{n_epochs} | loss={tot_loss / max(1, n_batches):.4f} | elapsed={time.time() - t0:.1f}s", flush=True)

    return model, rr_mean, rr_std


@torch.no_grad()
def infer_dl_on_source_records(model, model_id, dbs, te_recs, rr_mean, rr_std, batch_size=2048):
    model.eval()
    use_patcond = (model_id == "PatCond_Net1D_ECGFounder")
    rr_mean_t = torch.from_numpy(rr_mean).to(DEVICE, dtype=torch.float32)
    rr_std_t = torch.from_numpy(rr_std).to(DEVICE, dtype=torch.float32)

    out_scores = {}
    for r in te_recs:
        dobj = dbs[r["db"]]
        nb = r["nb"]
        scores = np.empty(nb, dtype=np.float32)
        b0, w0, bn0, bn1 = r["b0"], r["w0"], r["bn0"], r["bn1"]
        for i0 in range(0, nb, batch_size):
            i1 = min(i0 + batch_size, nb)
            bx = np.asarray(dobj["beats"][w0 + 1 + i0 : w0 + 1 + i1], dtype=np.float16)
            if use_patcond:
                locs = np.arange(i0, i1, dtype=np.int32)
                bni = np.minimum(bn0 + (locs // BN_SUB_STRIDE), bn1 - 1)
                bn = np.asarray(dobj["bn_sub"][bni], dtype=np.float16)
                x_in = np.concatenate([bx, bx - bn], axis=1)
            else:
                x_in = bx
            xb = torch.from_numpy(x_in).to(DEVICE, dtype=torch.float32, non_blocking=True)
            rrb = torch.from_numpy(np.asarray(dobj["rr3"][b0 + i0 : b0 + i1], dtype=np.float32)).to(DEVICE, non_blocking=True)
            rrb = (rrb - rr_mean_t) / rr_std_t
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(xb, rrb)
                f_lo = m61.logits_to_fusion_logodds(logits.float())
            scores[i0:i1] = f_lo.cpu().numpy()
        out_scores[(r["db"], r["rec_id"])] = scores
    return out_scores


def run_part2_train_all_sota_dl(rec_to_sid, only_models=None, force=False):
    t_all = time.time()
    print(f"\n================ PART 2: TRAIN 4 SOTA 2024-2025 DL MODELS ON 6-DB ({DEVICE}) ================", flush=True)
    dbs, records = open_v37_source_mmaps(rec_to_sid)

    # Exact deterministic 5-fold subject-grouped split (identical to lab_v59_package_a3_checkpoint.py!)
    pat_map = {}
    for r in records:
        pat_map.setdefault(r["subject_id"], []).append(r)
    pat_list = sorted(pat_map.keys(), key=lambda p: (-sum(r["n_F"] for r in pat_map[p]), p))
    assert len(pat_list) == 251, f"Expected 251 subjects, got {len(pat_list)}"
    n_folds = 5
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}
    pat_to_idx = {p: i for i, p in enumerate(pat_list)}

    pat_hours = np.array([sum(r["post_hours"] for r in pat_map[p]) for p in pat_list], dtype=np.float64)
    pat_totF = np.array([sum(r["n_F"] for r in pat_map[p]) for p in pat_list], dtype=np.int64)

    summary_out = {
        "Audit_Version": "v61a_sota_dl_6db_oof_and_checkpoints",
        "Device": str(DEVICE),
        "GPU_Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "Contract_SHA256": sha256_file(CONTRACT_V61),
        "Models": {},
    }
    sum_json_p = os.path.join(AUD_DIR, "V61A_DL_SOURCE_OOF_SUMMARY_20261123.json")
    if os.path.exists(sum_json_p) and not force:
        try:
            prev = json.load(open(sum_json_p, "r", encoding="utf-8"))
            summary_out["Models"].update(prev.get("Models", {}))
        except Exception:
            pass

    models_to_run = only_models if only_models else list(m61.DL_MODEL_ORDER)
    for model_id in models_to_run:
        ckpt_p = os.path.join(CKPT_DIR, f"v61_dl_{model_id}_6db.pt")
        if os.path.exists(ckpt_p) and not force and model_id in summary_out["Models"]:
            print(f"\n  [{model_id}] Already trained & sealed ({ckpt_p}); skipping (use --force to retrain).", flush=True)
            continue

        t_m = time.time()
        n_params = sum(p.numel() for p in m61.build_dl_model(model_id).parameters())
        print(f"\n  >>> Training {model_id} (params={n_params:,}) — 5-fold OOF (8 ep/fold) + Full 6-DB (12 ep) <<<", flush=True)

        # 5-fold OOF
        oof_scores_by_rec = {}
        for fid in range(n_folds):
            tr_recs = [r for r in records if fold_of_pat[r["subject_id"]] != fid]
            te_recs = [r for r in records if fold_of_pat[r["subject_id"]] == fid]
            mdl_f, rr_m, rr_s = train_one_dl_model(
                model_id, dbs, tr_recs, n_epochs=8, seed=42 + fid, batch_size=512,
                log_prefix=f"[{model_id} F{fid}]",
            )
            sc_map = infer_dl_on_source_records(mdl_f, model_id, dbs, te_recs, rr_m, rr_s)
            oof_scores_by_rec.update(sc_map)
            del mdl_f
            torch.cuda.empty_cache()
            gc.collect()

        # Assemble exact 2,676,362 OOF scores & solve exact 1D R3_B20 threshold
        oof_s_list, oof_y_list, oof_pi_list = [], [], []
        oof_stream = []
        for r in records:
            sc = oof_scores_by_rec[(r["db"], r["rec_id"])]
            ys = np.asarray(dbs[r["db"]]["y"][r["b0"] : r["b1"]], dtype=np.int8)
            oof_s_list.append(sc)
            oof_y_list.append(ys)
            oof_pi_list.append(np.full(len(ys), pat_to_idx[r["subject_id"]], dtype=np.int32))
            oof_stream.append({
                "patient_id": r["subject_id"],
                "post_hours": r["post_hours"],
                "y": ys,
                "score": sc,
            })

        s_cat = np.concatenate(oof_s_list)
        y_cat = np.concatenate(oof_y_list)
        pi_cat = np.concatenate(oof_pi_list)
        tau_r3 = m61.solve_r3_1d_exact(s_cat, y_cat, pi_cat, pat_hours, pat_totF, b_tot=2.0)

        oof_sum, _, _, oof_mfa, oof_zf = m61.evaluate_scalar_stream_cohort(oof_stream, tau=tau_r3)
        assert oof_sum["T1_MeanPat_FAPH_All"] <= 2.0 + 1e-3, f"R3_B20 violated: {oof_sum}"
        print(
            f"  [{model_id} 5F-OOF] tau={tau_r3:+.4f} | TP={oof_sum['T1_TP']:4d} FP_NS={oof_sum['T1_FP_NS']:5d} FP_V={oof_sum['T1_FPV']:4d} | "
            f"F1={oof_sum['T1_F1']:.4f} Rec={oof_sum['T1_MacroPat_Rec_%']:5.2f}% FAPH={oof_sum['T1_MeanPat_FAPH_All']:5.3f} "
            f"(0F={oof_sum['T1_MeanPat_FAPH_ZeroF']:5.3f} max={oof_sum['T1_ZeroF_Max_FAPH']:.2f}) | "
            f"AP={oof_sum['AP']:.4f} Pat@F2={oof_sum['MacroPat_Pat@F2_%']:5.2f}% M@2={oof_sum['Match_All_2.0']:5.2f}%",
            flush=True,
        )

        # Full 6-DB locked model (12 epochs)
        mdl_full, rr_m_full, rr_s_full = train_one_dl_model(
            model_id, dbs, records, n_epochs=12, seed=42, batch_size=512,
            log_prefix=f"[{model_id} FULL]",
        )
        train_sec = round(time.time() - t_m, 1)

        payload = {
            "model_id": model_id,
            "n_params": int(n_params),
            "state_dict": {k: v.cpu() for k, v in mdl_full.state_dict().items()},
            "rr_mean": rr_m_full,
            "rr_std": rr_s_full,
            "tau_r3_b20": float(tau_r3),
            "oof_summary": oof_sum,
            "oof_matched_fa": oof_mfa,
            "oof_zero_f_tail": oof_zf,
            "train_sec": train_sec,
            "contract_sha256": summary_out["Contract_SHA256"],
        }
        torch.save(payload, ckpt_p)
        ckpt_sha = sha256_file(ckpt_p)
        print(f"  [{model_id} SEALED] {ckpt_p} | SHA256={ckpt_sha[:16]}... | elapsed={train_sec:.1f}s", flush=True)

        summary_out["Models"][model_id] = {
            "n_params": int(n_params),
            "ckpt_path": ckpt_p,
            "ckpt_sha256": ckpt_sha,
            "tau_r3_b20": float(tau_r3),
            "train_sec": train_sec,
            "Source_5Fold_OOF": {"Summary": oof_sum, "Matched_FA": oof_mfa, "ZeroF_Tail": oof_zf},
        }
        with open(sum_json_p, "w", encoding="utf-8") as f:
            json.dump(summary_out, f, indent=2)

        del mdl_full, oof_scores_by_rec, oof_stream, s_cat, y_cat, pi_cat
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n[v61a] ALL DONE! Total Part-2 time: {time.time() - t_all:.1f}s | Summary: {sum_json_p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-part1", action="store_true")
    ap.add_argument("--skip-part2", action="store_true")
    ap.add_argument("--models", nargs="+", default=None, help="Subset of DL model IDs to train")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    reg = json.load(open(REG_6DB, "r", encoding="utf-8"))
    rec_to_sid = {(r["db"], r["rec_id"]): r["verified_subject_id"] for r in reg["records"]}

    if not args.skip_part1:
        run_part1_lightgbm_and_a3_parity(rec_to_sid)
    if not args.skip_part2:
        run_part2_train_all_sota_dl(rec_to_sid, only_models=args.models, force=args.force)


if __name__ == "__main__":
    main()
