r"""
lab_v62a_source_lodo_and_lock.py
================================
Wave-4 Revision Audit — Part A (6 Source Databases, 306 records, 251 SubjectAware_251 units, 2.68M beats).

Executes under strict two-level nested 6-DB LODO AND locks full 6-DB checkpoints
under identical R3 source calibration (MeanFAPH_All_src <= 2.0) for:
  1. v62A Component Ablations (SVM + LightGBM):
       - Full_A3               (15D+11D; asserts exact v58 SVM LODO parity!)
       - Abl_No_ThalesPyth     (13D+9D; drops M_Pyth^loc & tau_min^loc)
       - Abl_No_DualPoleCoords (7D+3D; keeps Stage-1 gate + local normal + 5 A3 feats + f_RR only)
       - Abl_Static_bN         (14D+11D; b_N_loc(t) == b_N static prefix normal pole)
       - Abl_No_Tangent_U      (14D+11D; P = I, r = 0, no 4D SVD subspace projection)
       - Abl_SinglePole_N      (12D+14D full-stream; patient b_N, b_N_loc, U, 5 A3 feats, NO V-pole / NO Stage-1 gate)
  2. v62C 4-Tier Causality Evaluation (SVM + LightGBM):
       - Tier0_Offline         (filtfilt + finalized full-record V-anchors)
       - Tier1_OnlineAnchors   (filtfilt + strictly online-expanding V-anchors & radii < t; evaluated both with Tier0 checkpoint & native Tier1 refit)
       - Tier2_PrefixOnly      (filtfilt + prefix-only V-anchors from first 50 beats)
       - Tier3_StrictCausal    (causal double-median + causal FIR + online-expanding V-anchors; evaluated both with Tier0 checkpoint & native Tier3 refit)
  3. v62D Representation Isolation Baselines (100% full-stream, identical 2-branch R3 protocol, SVM + LightGBM + RF control):
       - Raw_ECG_RR            (84D+86D uniform 81.3 Hz 2-lead waveform + past-only RR)
       - Trad_Morph_RR         (32D+34D literature multi-window amplitudes/RMS/P2P/slope/HOS + past-only RR)
       - SinglePole_N_RR       (12D+14D patient normal pole + tangent subspace + 5 local feats + past-only RR)
       - Full_A3               (Dual-Pole N-V + 5 local feats + past-only RR)

Memory-safe design (peak RAM < 1 GB): caches compact per-DB .npz on disk and
evaluates one full-stream representation at a time with explicit gc.collect().
"""


import os as _os, sys as _sys
from pathlib import Path as _Path
_SRC_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_DATA_ROOT = _Path(_os.environ.get("ECG_DATA_ROOT", str(_REPO_ROOT / "data_public_ecg")))
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import gc
import json
import os
import pickle
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import wfdb

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v36_6hospital_lodo_benchmark as v36
import lab_diagnostic_v46_two_branch_stage2_rr as v46
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_audit_v49_subject_unit_and_registry as v49
import lab_diagnostic_v53b_lodo_disentanglement_ablation as v53b
import lab_diagnostic_v56_qrs_stt_local_chord_lodo as v56
import lab_audit_v58_exact_beatwise_m3_and_ablation as v58
import lab_v61a_lightgbm_and_train_sota_dl as v61a
import v62_feature_engine as eng

ROOT = _REPO_ROOT
CACHE_V62_SRC = (_DATA_ROOT / "cache_v62_source")
CKPT_OUT = ROOT / "checkpoints" / "v62_locked_models_6db.pkl"
AUD_OUT = ROOT / "audits" / "V62A_SOURCE_LODO_AND_ABLATIONS_20261124.json"
CACHE_V62_SRC.mkdir(parents=True, exist_ok=True)

DB_ORDER = list(v53b.DB_ORDER)


def get_source_channels(db_name, rec_path):
    if db_name == "incartdb":
        hdr = wfdb.rdheader(rec_path)
        sn = list(hdr.sig_name)
        return (sn.index("II"), sn.index("V1"))
    return (0, 1)


def _extract_one_source_record(args):
    db_name, rec_id, subject_id = args
    rp = os.path.join(v36.DB_DIRS[db_name], rec_id)
    chs = get_source_channels(db_name, rp)

    # 1. Offline filtfilt load ONCE -> extracts Tier0 + Tier1 + Static_bN + SinglePole + Forensics in ONE single loop!
    dat_off = eng.load_beats_and_rr(rp, "atr", chs, "AAMI_Map_v1.0", filter_mode="offline_filtfilt")
    t0 = eng.run_dualpole_variant(dat_off, use_tangent_U=True, use_local_bN=True, anchor_mode="offline_and_online", collect_singlepole=True, collect_stage1_forensics=True, also_static_bn=True)
    t1 = t0["online_causal"]
    st_bn = t0["static_bn"]
    no_u = eng.run_dualpole_variant(dat_off, use_tangent_U=False, use_local_bN=True, anchor_mode="offline")
    t2 = eng.run_dualpole_variant(dat_off, use_tangent_U=True, use_local_bN=True, anchor_mode="prefix_only")
    x_raw, x_morph = eng.extract_traditional_morphology_and_raw(dat_off["joint_all"][50:], dat_off["pre_rr"][50:], dat_off["pre_rr_loc"][50:])
    x_sp = t0["X_singlepole"]
    del dat_off

    # 2. Strict Causal filter load (Tier 3)
    dat_cau = eng.load_beats_and_rr(rp, "atr", chs, "AAMI_Map_v1.0", filter_mode="causal_double_median_fir")
    t3 = eng.run_dualpole_variant(dat_cau, use_tangent_U=True, use_local_bN=True, anchor_mode="online_causal")
    del dat_cau

    def pack_s2(d):
        return {
            "s2_mask": d["s2_mask"].astype(bool),
            "X29_s2": d["X29_s2"].astype(np.float32),
            "K": int(d["K"]),
            "tau1_beat": d["tau1_beat"],
            "tau1_min": d["tau1_min"],
            "pre_anchor_F": int(d["pre_anchor_F"]),
        }

    return {
        "db": db_name, "rec_id": rec_id, "patient_id": subject_id,
        "post_hours": float(t0["post_hours"]),
        "y": t0["y"].astype(np.int8),
        "y_t3": t3["y"].astype(np.int8),
        "post_hours_t3": float(t3["post_hours"]),
        "Tier0": pack_s2(t0),
        "No_Tangent_U": pack_s2(no_u),
        "Static_bN": pack_s2(st_bn),
        "Tier1_Online": pack_s2(t1),
        "Tier2_PrefixOnly": pack_s2(t2),
        "Tier3_StrictCausal": pack_s2(t3),
        "X_raw": x_raw.astype(np.float16),
        "X_morph": x_morph.astype(np.float32),
        "X_sp": x_sp.astype(np.float32),
        "forensics": t0["forensics"],
    }


def build_or_load_source_caches(rec_to_sid):
    """Build compact per-DB caches once (resumable per DB)."""
    dbs_ref = {db: v47.build_or_load_v47_db(db) for db in DB_ORDER}
    for db in DB_ORDER:
        s2_p = CACHE_V62_SRC / f"{db}_s2_variants.pkl"
        raw_p = CACHE_V62_SRC / f"{db}_raw_f16.npy"
        morph_p = CACHE_V62_SRC / f"{db}_morph_f32.npy"
        sp_p = CACHE_V62_SRC / f"{db}_sp_f32.npy"
        if s2_p.exists() and raw_p.exists() and morph_p.exists() and sp_p.exists():
            print(f"[cache] {db} already cached.", flush=True)
            continue

        t0 = time.time()
        tasks = [(db, it["rec_id"], rec_to_sid[(db, it["rec_id"])]) for it in dbs_ref[db]]
        print(f"[extract] {db}: {len(tasks)} records with 4 workers...", flush=True)
        with ProcessPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(_extract_one_source_record, tasks))

        # Concatenate full-stream arrays per DB & store offsets cleanly
        raw_list, morph_list, sp_list = [], [], []
        s2_meta = []
        off = 0
        for res in results:
            n = len(res["y"])
            raw_list.append(res.pop("X_raw"))
            morph_list.append(res.pop("X_morph"))
            sp_list.append(res.pop("X_sp"))
            res["b0"] = off
            res["b1"] = off + n
            off += n
            s2_meta.append(res)

        np.save(raw_p, np.vstack(raw_list))
        np.save(morph_p, np.vstack(morph_list))
        np.save(sp_p, np.vstack(sp_list))
        with open(s2_p, "wb") as f:
            pickle.dump(s2_meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        del raw_list, morph_list, sp_list, results
        gc.collect()
        print(f"[extract] {db} cached ({off:,} beats) in {time.time() - t0:.1f}s.", flush=True)


def make_head(family):
    if family == "SVM":
        return v46.make_clf("SVM")
    if family == "LightGBM":
        return v61a.make_lgbm(seed=42)
    if family == "RF":
        return RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=50,
            max_features="sqrt", random_state=42, n_jobs=4,
        )
    raise ValueError(family)


def score_head(clf, sc, X, family):
    Xt = sc.transform(X) if sc is not None else X
    if family == "SVM":
        return clf.decision_function(Xt)
    if family == "LightGBM":
        return clf.predict(Xt, raw_score=True)
    if family == "RF":
        p = np.clip(clf.predict_proba(Xt)[:, 1], 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))
    raise ValueError(family)


def fit_head(X, y, w, family):
    sc = StandardScaler() if family == "SVM" else None
    Xt = sc.fit_transform(X) if sc is not None else X
    clf = make_head(family)
    clf.fit(Xt, y, sample_weight=w)
    return sc, clf


def build_stage2_items(s2_all_by_db, var_key, use_t3_y=False):
    out = {}
    for db in DB_ORDER:
        items = []
        for r in s2_all_by_db[db]:
            v = r[var_key]
            items.append({
                "db": db, "rec_id": r["rec_id"], "patient_id": r["patient_id"],
                "post_hours": r["post_hours_t3"] if use_t3_y else r["post_hours"],
                "y": r["y_t3"] if use_t3_y else r["y"],
                "s2_mask": v["s2_mask"],
                "X_s2": v["X29_s2"].astype(np.float64),
                "K": v["K"],
                "tau1_min": v["tau1_min"],
                "pre_anchor_F": v["pre_anchor_F"],
            })
        out[db] = items
    return out


def fit_twobranch_r3(tr_items, fN, fV, family, cap_pat_pos=100):
    """Exact 5-fold subject-grouped OOF R3 calibration + full training refit for Stage-2 items."""
    pat_map = {}
    for it in tr_items:
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

    oof_sN, oof_sV, oof_y, oof_pi = [], [], [], []
    for fid in range(n_folds):
        tr_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in tr_items if fold_of_pat[it["patient_id"]] == fid]
        XN, yN, wN = v46.build_branch_xy(tr_sub, fN, {0, 3}, cap_pat_pos=cap_pat_pos)
        scN, clfN = fit_head(XN, yN, wN, family)
        XV, yV, wV = v46.build_branch_xy(tr_sub, fV, {1}, cap_pat_pos=cap_pat_pos)
        scV, clfV = fit_head(XV, yV, wV, family)
        for te in te_sub:
            if len(te["X_s2"]) > 0:
                oof_sN.append(score_head(clfN, scN, te["X_s2"][:, fN], family))
                oof_sV.append(score_head(clfV, scV, te["X_s2"][:, fV], family))
                oof_y.append(te["y"][te["s2_mask"]].astype(np.int8))
                oof_pi.append(np.full(len(te["X_s2"]), pat_to_idx[te["patient_id"]], dtype=np.int32))

    oof_sN = np.concatenate(oof_sN)
    oof_sV = np.concatenate(oof_sV)
    oof_y = np.concatenate(oof_y)
    oof_pi = np.concatenate(oof_pi)
    sigN = float(np.std(oof_sN)) + 1e-9
    sigV = float(np.std(oof_sV)) + 1e-9
    tN, tV = v56.solve_r3_joint2d_patfa(oof_sN, oof_sV, oof_y, oof_pi, pat_hours, pat_totF, b_tot=2.0, b_0f=1e9, n_grid=85)

    XN, yN, wN = v46.build_branch_xy(tr_items, fN, {0, 3}, cap_pat_pos=cap_pat_pos)
    scN, clfN = fit_head(XN, yN, wN, family)
    XV, yV, wV = v46.build_branch_xy(tr_items, fV, {1}, cap_pat_pos=cap_pat_pos)
    scV, clfV = fit_head(XV, yV, wV, family)

    return {
        "family": family, "fN": list(fN), "fV": list(fV),
        "scN": scN, "clfN": clfN, "tau_N": float(tN), "sig_N": sigN,
        "scV": scV, "clfV": clfV, "tau_V": float(tV), "sig_V": sigV,
    }


def summarize_unit_scores(unit_scores):
    """Exact subject-weighted summary + full-stream AP + bootstrap/Wilcoxon-ready per-unit dict."""
    subjs = sorted(unit_scores.keys())
    per_unit = {}
    all_y, all_sc, all_si = [], [], []
    for si, sid in enumerate(subjs):
        u = unit_scores[sid]
        y = np.concatenate(u["y_list"]) if u["y_list"] else np.zeros(0, dtype=np.int8)
        sc = np.concatenate(u["sc_list"]) if u["sc_list"] else np.zeros(0, dtype=np.float64)
        pred = (sc >= 0.0)
        tp = int(np.sum(pred & (y == 2)))
        fp_ns = int(np.sum(pred & ((y == 0) | (y == 3))))
        fp_v = int(np.sum(pred & (y == 1)))
        tot_f = int(u["tot_F"])
        hrs = float(u["hours"])
        faph = (fp_ns + fp_v) / max(hrs, 1e-9)
        per_unit[sid] = {
            "tot_F": tot_f, "hours": hrs, "TP": tp, "FP_NS": fp_ns, "FP_V": fp_v,
            "FAPH": faph, "Rec_%": (100.0 * tp / tot_f) if tot_f > 0 else None,
        }
        all_y.append(y)
        all_sc.append(sc)
        all_si.append(np.full(len(y), si, dtype=np.int32))

    fpos_u = [per_unit[s] for s in subjs if per_unit[s]["tot_F"] > 0]
    zf_u = [per_unit[s] for s in subjs if per_unit[s]["tot_F"] == 0]
    all_u = [per_unit[s] for s in subjs]
    faph_all = np.array([v["FAPH"] for v in all_u], dtype=np.float64)
    faph_0f = np.array([v["FAPH"] for v in zf_u], dtype=np.float64)
    tot_fp = sum(v["FP_NS"] + v["FP_V"] for v in all_u)
    tot_h = sum(v["hours"] for v in all_u)

    y_cat = np.concatenate(all_y)
    sc_cat = np.concatenate(all_sc)
    si_cat = np.concatenate(all_si)
    ap_full = float(average_precision_score((y_cat == 2).astype(int), sc_cat)) if np.any(y_cat == 2) else 0.0
    valid = sc_cat > -1e8
    ap_s2 = float(average_precision_score((y_cat[valid] == 2).astype(int), sc_cat[valid])) if np.any(y_cat[valid] == 2) else 0.0

    # Matched-FA sweep at <= 2.0 FAPH across all units (exact aligned subject indices!)
    n_all = len(subjs)
    n_fpos = max(1, len(fpos_u))
    totF_arr = np.array([per_unit[s]["tot_F"] for s in subjs], dtype=np.float64)
    hrs_arr = np.array([per_unit[s]["hours"] for s in subjs], dtype=np.float64)
    yv, scv, siv = y_cat[valid], sc_cat[valid], si_cat[valid]
    if len(scv) > 0:
        ord_i = np.argsort(scv)[::-1]
        so, yo, sio = scv[ord_i], yv[ord_i], siv[ord_i]
        is_f = (yo == 2)
        rec_inc = np.where(is_f, 100.0 / (np.maximum(1.0, totF_arr[sio]) * n_fpos), 0.0)
        fa_inc = np.where(~is_f, 1.0 / (np.maximum(1e-6, hrs_arr[sio]) * n_all), 0.0)
        cum_rec = np.cumsum(rec_inc)
        cum_fa = np.cumsum(fa_inc)
        grp_end = np.empty(len(so), dtype=bool)
        grp_end[:-1] = (so[:-1] != so[1:])
        grp_end[-1] = True
        ge = np.where(grp_end)[0]
        pos = np.searchsorted(cum_fa[ge], 2.0 + 1e-12, side="right") - 1
        m20 = float(cum_rec[ge][pos]) if pos >= 0 else 0.0
    else:
        m20 = 0.0

    return {
        "TP": int(sum(v["TP"] for v in all_u)),
        "FP_NS": int(sum(v["FP_NS"] for v in all_u)),
        "FP_V": int(sum(v["FP_V"] for v in all_u)),
        "MacroRec_%": round(float(np.mean([v["Rec_%"] for v in fpos_u])), 2),
        "Fpos_hit": int(sum(1 for v in fpos_u if v["TP"] > 0)),
        "n_Fpos": len(fpos_u),
        "MeanFAPH_All": round(float(np.mean(faph_all)), 3),
        "MeanFAPH_0F": round(float(np.mean(faph_0f)), 3),
        "PooledFAPH": round(float(tot_fp / max(tot_h, 1e-9)), 3),
        "MedFAPH": round(float(np.median(faph_all)), 3),
        "P95_FAPH": round(float(np.percentile(faph_all, 95)), 3),
        "Pct_LE2": round(100.0 * float(np.mean(faph_all <= 2.0)), 1),
        "ZeroF_Max": round(float(np.max(faph_0f)), 2),
        "AP_full": round(ap_full, 4),
        "AP_s2": round(ap_s2, 4),
        "Match20_%": round(m20, 2),
    }, per_unit


def paired_compare(pu_a, pu_b, B=10000, seed=42):
    subjs = sorted(pu_a.keys())
    fp_s = [s for s in subjs if pu_a[s]["tot_F"] > 0]
    zf_s = [s for s in subjs if pu_a[s]["tot_F"] == 0]
    rng = np.random.default_rng(seed)

    def stats_diff(da, db):
        d = da - db
        nz = d[np.abs(d) > 1e-12]
        w = int(np.sum(nz > 0)); l = int(np.sum(nz < 0)); t = len(d) - w - l
        if len(nz) == 0:
            pw = 1.0
        else:
            try:
                pw = float(wilcoxon(nz, method="exact").pvalue) if len(nz) <= 25 else float(wilcoxon(nz).pvalue)
            except Exception:
                pw = float(wilcoxon(nz).pvalue)
        idx = rng.integers(0, len(d), size=(B, len(d)))
        boot = np.mean(d[idx], axis=1)
        ci = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
        return {"diff": round(float(np.mean(d)), 3), "ci95": ci, "WTL": f"{w}/{t}/{l}", "p_wilcoxon": round(pw, 5)}

    rec_a = np.array([pu_a[s]["Rec_%"] for s in fp_s])
    rec_b = np.array([pu_b[s]["Rec_%"] for s in fp_s])
    fa_a = np.array([pu_a[s]["FAPH"] for s in subjs])
    fa_b = np.array([pu_b[s]["FAPH"] for s in subjs])
    zf_a = np.array([pu_a[s]["FAPH"] for s in zf_s])
    zf_b = np.array([pu_b[s]["FAPH"] for s in zf_s])
    return {"MacroRec": stats_diff(rec_a, rec_b), "FAPH_All": stats_diff(fa_a, fa_b), "FAPH_0F": stats_diff(zf_a, zf_b)}


def eval_stage2_nested_lodo(dbs_items, fN, fV, family, eval_dbs_override=None):
    """Strict 6-DB nested LODO; optional eval_dbs_override evaluates Tier-0 models on Tier-1/3 features."""
    eval_dbs = eval_dbs_override if eval_dbs_override is not None else dbs_items
    unit_scores = {}
    for db in DB_ORDER:
        for it in eval_dbs[db]:
            u = unit_scores.setdefault(it["patient_id"], {"tot_F": 0, "hours": 0.0, "y_list": [], "sc_list": []})
            u["tot_F"] += int(np.sum(it["y"] == 2))
            u["hours"] += float(it["post_hours"])

    for tgt in DB_ORDER:
        tr_items = [it for db in DB_ORDER if db != tgt for it in dbs_items[db]]
        mod = fit_twobranch_r3(tr_items, fN, fV, family)
        for te in eval_dbs[tgt]:
            sc = np.full(len(te["y"]), -1e9, dtype=np.float64)
            if len(te["X_s2"]) > 0:
                sN = score_head(mod["clfN"], mod["scN"], te["X_s2"][:, fN], family)
                sV = score_head(mod["clfV"], mod["scV"], te["X_s2"][:, fV], family)
                sc[te["s2_mask"]] = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
            u = unit_scores[te["patient_id"]]
            u["y_list"].append(te["y"])
            u["sc_list"].append(sc)

    summ, pu = summarize_unit_scores(unit_scores)
    # Also fit full 6-DB locked checkpoint
    all_tr = [it for db in DB_ORDER for it in dbs_items[db]]
    locked = fit_twobranch_r3(all_tr, fN, fV, family) if eval_dbs_override is None else None
    return summ, pu, locked


def build_fullstream_branch_xy(db_arrays, meta_list, cols, neg_classes, max_neg=120000, cap_pat_pos=100, seed=42):
    """
    RAM-safe balanced branch builder for full-stream representations (2.68M beats):
    Keeps 100% of Class-F beats (patient-capped weight at 100) + deterministic stratified
    subsample of negatives (up to max_neg; 100% of V since total V ~ 128k).
    """
    rng = np.random.default_rng(seed)
    neg_arr = np.array(list(neg_classes), dtype=np.int8)

    pf_c = Counter()
    tot_neg_avail = 0
    for r in meta_list:
        y = r["y"]
        pf_c[r["patient_id"]] += int(np.sum(y == 2))
        tot_neg_avail += int(np.sum(np.isin(y, neg_arr)))

    neg_keep_prob = min(1.0, float(max_neg) / max(1, tot_neg_avail))
    Xs, ys, ws = [], [], []
    for r in meta_list:
        y = r["y"]
        X_db = db_arrays[r["db"]]
        b0, b1 = r["b0"], r["b1"]
        pos_idx = np.where(y == 2)[0]
        neg_idx = np.where(np.isin(y, neg_arr))[0]
        if len(neg_idx) > 0 and neg_keep_prob < 1.0:
            k_n = max(1, int(round(len(neg_idx) * neg_keep_prob)))
            neg_idx = rng.choice(neg_idx, size=min(k_n, len(neg_idx)), replace=False)

        if len(pos_idx) > 0:
            pf = max(1, pf_c[r["patient_id"]])
            pw = min(1.0, float(cap_pat_pos) / float(pf))
            Xs.append(X_db[b0 + pos_idx][:, cols].astype(np.float32))
            ys.append(np.ones(len(pos_idx), dtype=np.int8))
            ws.append(np.full(len(pos_idx), pw, dtype=np.float64))
        if len(neg_idx) > 0:
            Xs.append(X_db[b0 + neg_idx][:, cols].astype(np.float32))
            ys.append(np.zeros(len(neg_idx), dtype=np.int8))
            ws.append(np.ones(len(neg_idx), dtype=np.float64))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    w_pos = float(np.sum(w[y == 1]))
    w_neg = float(np.sum(w[y == 0]))
    if w_pos > 0 and w_neg > 0:
        w[y == 1] *= (w_neg / w_pos)
    return X, y, w


def score_fullstream_chunks(clf, sc, X_slice, cols, family, chunk=50000):
    n = len(X_slice)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        xb = X_slice[i : i + chunk][:, cols].astype(np.float32)
        out[i : i + chunk] = score_head(clf, sc, xb, family)
    return out


def fit_fullstream_twobranch_r3(db_arrays, tr_meta, cols_N, cols_V, family):
    pat_map = {}
    for r in tr_meta:
        pat_map.setdefault(r["patient_id"], []).append(r)
    pat_list = sorted(pat_map.keys(), key=lambda p: (-sum(int(np.sum(r["y"] == 2)) for r in pat_map[p]), p))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}
    pat_to_idx = {p: i for i, p in enumerate(pat_list)}

    n_pats = len(pat_list)
    pat_hours = np.zeros(n_pats, dtype=np.float64)
    pat_totF = np.zeros(n_pats, dtype=np.int64)
    for p, rs in pat_map.items():
        pi = pat_to_idx[p]
        pat_hours[pi] = sum(float(r["post_hours"]) for r in rs)
        pat_totF[pi] = sum(int(np.sum(r["y"] == 2)) for r in rs)

    oof_sN, oof_sV, oof_y, oof_pi = [], [], [], []
    for fid in range(n_folds):
        tr_sub = [r for r in tr_meta if fold_of_pat[r["patient_id"]] != fid]
        te_sub = [r for r in tr_meta if fold_of_pat[r["patient_id"]] == fid]
        XN, yN, wN = build_fullstream_branch_xy(db_arrays, tr_sub, cols_N, {0, 3}, seed=42 + fid)
        scN, clfN = fit_head(XN, yN, wN, family)
        del XN, yN, wN
        XV, yV, wV = build_fullstream_branch_xy(db_arrays, tr_sub, cols_V, {1}, seed=142 + fid)
        scV, clfV = fit_head(XV, yV, wV, family)
        del XV, yV, wV
        for te in te_sub:
            X_sl = db_arrays[te["db"]][te["b0"] : te["b1"]]
            oof_sN.append(score_fullstream_chunks(clfN, scN, X_sl, cols_N, family))
            oof_sV.append(score_fullstream_chunks(clfV, scV, X_sl, cols_V, family))
            oof_y.append(te["y"])
            oof_pi.append(np.full(len(te["y"]), pat_to_idx[te["patient_id"]], dtype=np.int32))

    oof_sN = np.concatenate(oof_sN)
    oof_sV = np.concatenate(oof_sV)
    oof_y = np.concatenate(oof_y)
    oof_pi = np.concatenate(oof_pi)
    sigN = float(np.std(oof_sN)) + 1e-9
    sigV = float(np.std(oof_sV)) + 1e-9
    tN, tV = v56.solve_r3_joint2d_patfa(
        oof_sN, oof_sV, oof_y, oof_pi,
        pat_hours, pat_totF, b_tot=2.0, b_0f=1e9, n_grid=85,
    )
    del oof_sN, oof_sV, oof_y, oof_pi
    gc.collect()

    XN, yN, wN = build_fullstream_branch_xy(db_arrays, tr_meta, cols_N, {0, 3}, seed=42)
    scN, clfN = fit_head(XN, yN, wN, family)
    del XN, yN, wN
    XV, yV, wV = build_fullstream_branch_xy(db_arrays, tr_meta, cols_V, {1}, seed=142)
    scV, clfV = fit_head(XV, yV, wV, family)
    del XV, yV, wV
    gc.collect()

    return {
        "family": family, "fN": list(cols_N), "fV": list(cols_V),
        "scN": scN, "clfN": clfN, "tau_N": float(tN), "sig_N": sigN,
        "scV": scV, "clfV": clfV, "tau_V": float(tV), "sig_V": sigV,
    }


def eval_fullstream_nested_lodo(s2_all_by_db, array_suffix, cols_N, cols_V, family):
    db_arrays = {db: np.load(CACHE_V62_SRC / f"{db}_{array_suffix}.npy", mmap_mode="r") for db in DB_ORDER}
    unit_scores = {}
    for db in DB_ORDER:
        for r in s2_all_by_db[db]:
            u = unit_scores.setdefault(r["patient_id"], {"tot_F": 0, "hours": 0.0, "y_list": [], "sc_list": []})
            u["tot_F"] += int(np.sum(r["y"] == 2))
            u["hours"] += float(r["post_hours"])

    for tgt in DB_ORDER:
        tr_meta = [r for db in DB_ORDER if db != tgt for r in s2_all_by_db[db]]
        mod = fit_fullstream_twobranch_r3(db_arrays, tr_meta, cols_N, cols_V, family)
        for te in s2_all_by_db[tgt]:
            X_sl = db_arrays[tgt][te["b0"] : te["b1"]]
            sN = score_fullstream_chunks(mod["clfN"], mod["scN"], X_sl, cols_N, family)
            sV = score_fullstream_chunks(mod["clfV"], mod["scV"], X_sl, cols_V, family)
            sc = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
            u = unit_scores[te["patient_id"]]
            u["y_list"].append(te["y"])
            u["sc_list"].append(sc)
        del mod
        gc.collect()

    summ, pu = summarize_unit_scores(unit_scores)
    all_meta = [r for db in DB_ORDER for r in s2_all_by_db[db]]
    locked = fit_fullstream_twobranch_r3(db_arrays, all_meta, cols_N, cols_V, family)
    del db_arrays, unit_scores
    gc.collect()
    return summ, pu, locked


def main():
    t_start = time.time()
    reg = json.load(open(v53b.REG_PATH, "r", encoding="utf-8"))
    rec_to_sid = {(r["db"], r["rec_id"]): r["verified_subject_id"] for r in reg["records"]}

    build_or_load_source_caches(rec_to_sid)

    s2_all_by_db = {}
    for db in DB_ORDER:
        with open(CACHE_V62_SRC / f"{db}_s2_variants.pkl", "rb") as f:
            s2_all_by_db[db] = pickle.load(f)

    audit = {"Version": "v62a_source_lodo_and_lock", "Date": "2026-11-24", "Stage2_Ablations": {}, "Causality_Tiers": {}, "FullStream_Baselines": {}, "Paired_vs_FullA3": {}}
    locked_models = {}

    # 1. Stage-2 Dual-Pole Ablations & Parity Check
    dbs_t0 = build_stage2_items(s2_all_by_db, "Tier0")
    dbs_nou = build_stage2_items(s2_all_by_db, "No_Tangent_U")
    dbs_sbn = build_stage2_items(s2_all_by_db, "Static_bN")
    dbs_t1 = build_stage2_items(s2_all_by_db, "Tier1_Online")
    dbs_t2 = build_stage2_items(s2_all_by_db, "Tier2_PrefixOnly")
    dbs_t3 = build_stage2_items(s2_all_by_db, "Tier3_StrictCausal", use_t3_y=True)

    # Pre-anchor F coverage stats on 6 source DBs
    t1_tau1 = [it["tau1_min"] for db in DB_ORDER for it in dbs_t1[db] if it["tau1_min"] is not None]
    t1_preF = sum(it["pre_anchor_F"] for db in DB_ORDER for it in dbs_t1[db])
    t1_totF = sum(int(np.sum(it["y"] == 2)) for db in DB_ORDER for it in dbs_t1[db])
    t1_s2F = sum(int(np.sum(it["y"][it["s2_mask"]] == 2)) for db in DB_ORDER for it in dbs_t1[db])
    t0_s2F = sum(int(np.sum(it["y"][it["s2_mask"]] == 2)) for db in DB_ORDER for it in dbs_t0[db])
    t2_s2F = sum(int(np.sum(it["y"][it["s2_mask"]] == 2)) for db in DB_ORDER for it in dbs_t2[db])
    t3_s2F = sum(int(np.sum(it["y"][it["s2_mask"]] == 2)) for db in DB_ORDER for it in dbs_t3[db])
    t3_preF = sum(it["pre_anchor_F"] for db in DB_ORDER for it in dbs_t3[db])
    audit["Source_Causality_Coverage"] = {
        "tot_F": t1_totF, "Tier0_s2_F": t0_s2F, "Tier1_s2_F": t1_s2F, "Tier1_pre_anchor_F": t1_preF,
        "Tier1_tau1_med_min": round(float(np.median(t1_tau1)), 2) if t1_tau1 else None,
        "Tier1_tau1_iqr_min": [round(float(np.percentile(t1_tau1, 25)), 2), round(float(np.percentile(t1_tau1, 75)), 2)] if t1_tau1 else None,
        "Tier2_PrefixOnly_s2_F": t2_s2F, "Tier3_StrictCausal_s2_F": t3_s2F, "Tier3_pre_anchor_F": t3_preF,
    }
    print("Source Causality Coverage:", audit["Source_Causality_Coverage"], flush=True)

    pu_ref = {}
    for fam in ("SVM", "LightGBM", "RF"):
        print(f"\n=== Stage-2 Ablations ({fam}) ===", flush=True)
        cfgs = [
            ("Full_A3", dbs_t0, eng.IDX_A3_N, eng.IDX_C3_V),
            ("Abl_No_ThalesPyth", dbs_t0, eng.IDX_NO_TP_N, eng.IDX_NO_TP_V),
            ("Abl_No_DualPoleCoords", dbs_t0, eng.IDX_NO_DP_N, eng.IDX_NO_DP_V),
            ("Abl_Static_bN", dbs_sbn, eng.IDX_STATIC_BN_N, eng.IDX_C3_V),
            ("Abl_No_Tangent_U", dbs_nou, eng.IDX_NO_U_N, eng.IDX_C3_V),
        ]
        if fam == "RF":
            cfgs = cfgs[:1]  # RF control on Full_A3 + Full-stream baselines
        for cname, dbs_use, fN, fV in cfgs:
            summ, pu, locked = eval_stage2_nested_lodo(dbs_use, fN, fV, fam)
            key = f"{fam}__{cname}"
            audit["Stage2_Ablations"][key] = summ
            locked_models[key] = locked
            print(f"  {key:32s}: Rec={summ['MacroRec_%']:5.2f}% | TP={summ['TP']:4d} | FP_NS={summ['FP_NS']:4d} FP_V={summ['FP_V']:4d} | FAPH={summ['MeanFAPH_All']:.3f} (0F={summ['MeanFAPH_0F']:.3f}) | AP_f={summ['AP_full']:.4f} | M2={summ['Match20_%']:.2f}%", flush=True)
            if cname == "Full_A3":
                pu_ref[fam] = pu
                if fam == "SVM":
                    assert abs(summ["MacroRec_%"] - 7.06) <= 0.02 and summ["TP"] == 687 and summ["FP_NS"] == 256, f"v58 parity mismatch: {summ}"
                    print("  [VERIFIED] Exact v58 6-DB LODO parity (TP=687, FP_NS=256, MacroRec=7.06%)!", flush=True)
            else:
                audit["Paired_vs_FullA3"][key] = paired_compare(pu_ref[fam], pu)

    # 2. 4-Tier Causality Evaluation (SVM + LightGBM)
    for fam in ("SVM", "LightGBM"):
        print(f"\n=== 4-Tier Causality Evaluation ({fam}) ===", flush=True)
        for cname, dbs_tr, dbs_ev in [
            ("Tier1_Online_LockedT0", dbs_t0, dbs_t1),
            ("Tier1_Online_Native", dbs_t1, dbs_t1),
            ("Tier2_PrefixOnly_LockedT0", dbs_t0, dbs_t2),
            ("Tier2_PrefixOnly_Native", dbs_t2, dbs_t2),
            ("Tier3_StrictCausal_LockedT0", dbs_t0, dbs_t3),
            ("Tier3_StrictCausal_Native", dbs_t3, dbs_t3),
        ]:
            ev_ov = dbs_ev if (dbs_tr is not dbs_ev) else None
            summ, pu, locked = eval_stage2_nested_lodo(dbs_tr, eng.IDX_A3_N, eng.IDX_C3_V, fam, eval_dbs_override=ev_ov)
            key = f"{fam}__{cname}"
            audit["Causality_Tiers"][key] = summ
            if locked is not None:
                locked_models[key] = locked
            audit["Paired_vs_FullA3"][key] = paired_compare(pu_ref[fam], pu)
            print(f"  {key:36s}: Rec={summ['MacroRec_%']:5.2f}% | TP={summ['TP']:4d} | FAPH={summ['MeanFAPH_All']:.3f} (0F={summ['MeanFAPH_0F']:.3f}) | AP_f={summ['AP_full']:.4f}", flush=True)

    # 3. Full-Stream Representation Isolation Baselines (100% of 2.68M source beats!)
    fs_specs = [
        ("SinglePole_N_RR", "sp_f32", list(range(12)), list(range(14))),
        ("Trad_Morph_RR",   "morph_f32", list(range(32)), list(range(34))),
        ("Raw_ECG_RR",      "raw_f16", list(range(84)), list(range(86))),
    ]
    for cname, suf, cN, cV in fs_specs:
        for fam in ("SVM", "LightGBM"):
            print(f"\n=== Full-Stream Baseline: {fam}__{cname} ===", flush=True)
            summ, pu, locked = eval_fullstream_nested_lodo(s2_all_by_db, suf, cN, cV, fam)
            key = f"{fam}__{cname}"
            audit["FullStream_Baselines"][key] = summ
            locked_models[key] = locked
            audit["Paired_vs_FullA3"][key] = paired_compare(pu_ref[fam], pu)
            print(f"  {key:32s}: Rec={summ['MacroRec_%']:5.2f}% | TP={summ['TP']:4d} | FP_NS={summ['FP_NS']:5d} FP_V={summ['FP_V']:4d} | FAPH={summ['MeanFAPH_All']:.3f} (0F={summ['MeanFAPH_0F']:.3f}) | AP_f={summ['AP_full']:.4f} | M2={summ['Match20_%']:.2f}%", flush=True)

    audit["runtime_sec"] = round(time.time() - t_start, 1)
    AUD_OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with open(CKPT_OUT, "wb") as f:
        pickle.dump(locked_models, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nDONE v62a in {audit['runtime_sec']}s -> {AUD_OUT} & {CKPT_OUT}", flush=True)


if __name__ == "__main__":
    main()
