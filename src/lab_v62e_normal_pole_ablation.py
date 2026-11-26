r"""
lab_v62e_normal_pole_ablation.py
================================
Symmetric Normal-Pole Ablation Suite (completing Advisor Point 2 with 1-to-1
symmetry against Ventricular-Pole, Tangent-Subspace, and Thales-Pythagoras ablations):

  1. `Abl_No_LocalNormal` (Stage-2 Chord-Only, 6D+7D):
     Drops ALL local-normal-pole features (rho_locN [4], d_locN [5], M_Pyth^loc [13],
     tau_min^loc [14], and the 5 A3 local-normal features [24..28]), retaining ONLY
     global chord/bridge coordinates [0, 1, 2, 3, 8, 10] (+ f_RR [18] in F:V).
     Exact Stage-2 mirror of `Abl_No_DualPoleCoords` (7D+3D)!

  2. `SinglePole_V_RR` (`w/o Normal Pole` — Full-Stream Ventricular-Pole Only, 10D+12D):
     Removes the Normal pole (b_N, b_N^loc) and normal tangent subspace (U) COMPLETELY
     from both Stage 1 and Stage 2. Discovers up to K <= 4 recurring premature ectopic
     anchors {b_{V,k}} directly in standardized waveform space R^374 using ONLY
     prematurity (pre_rr/loc_rr <= 0.88), prefix-median HF noise gate (SQI_HF <= 1.20),
     and waveform cluster cohesion (>=6 beats within radius 12.0, dispersion <= 8.0),
     with ZERO reference to b_N or prefix normal clustering. Scores 100% of post-prefix
     beats (for records with K >= 1; -1e9 when K == 0) using 10 V-pole features in
     Branch F:N/S and +2 identical past-only RR features in Branch F:V under identical
     two-branch SVM & LightGBM heads and identical R3 calibration (MeanFAPH_src <= 2.0).
     Exact full-pipeline 1-to-1 mirror of `SinglePole_N_RR` (`w/o Ventricular Pole`)!

Preserves all frozen checkpoints/caches untouched and writes verified output to:
  `audits/V62E_NORMAL_POLE_ABLATION_20261124.json`
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
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import lab_diagnostic_v36_6hospital_lodo_benchmark as v36
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_audit_v49_subject_unit_and_registry as v49
import lab_diagnostic_v53b_lodo_disentanglement_ablation as v53b
import stage2_frozen_pipeline_v2 as pipe2
import v62_feature_engine as eng
import lab_v62a_source_lodo_and_lock as v62a
import lab_v62b_external_stream_eval as v62b

ROOT = _REPO_ROOT
CACHE_V62_SRC = (_DATA_ROOT / "cache_v62_source")
CACHE_V59 = (_DATA_ROOT / "cache_v59_ltstdb_a3")
CACHE_V60 = (_DATA_ROOT / "cache_v60_mghdb_a3")
CACHE_V62E_EXT = (_DATA_ROOT / "cache_v62e_vpole_ext")
CACHE_V62E_EXT.mkdir(parents=True, exist_ok=True)

AUD_V62A = ROOT / "audits" / "V62A_SOURCE_LODO_AND_ABLATIONS_20261124.json"
AUD_V62B = ROOT / "audits" / "V62B_EXTERNAL_LTSTDB_MGHDB_REVISION_20261124.json"
AUD_OUT = ROOT / "audits" / "V62E_NORMAL_POLE_ABLATION_20261124.json"

IDX_CHORD_ONLY_N = [0, 1, 2, 3, 8, 10]       # 6D (zero b_N_loc, zero Thales-Pyth, zero A3)
IDX_CHORD_ONLY_V = [0, 1, 2, 3, 8, 10, 18]   # 7D (+ f_RR)


def extract_vpole_only_features(dat):
    """
    Pure Ventricular-Pole-Only (`w/o Normal Pole`) extractor:
      - ZERO b_N, ZERO b_N_loc, ZERO tangent subspace U!
      - Discovers up to K <= 4 recurring premature clusters directly in R^374.
      - Computes 12D feature matrix (cols 0..9 V-pole morphology, 10=f_RR, 11=pre_rr_s)
        for all post-prefix beats when K >= 1 (returns K=0 if no cluster confirmed).
    """
    joint_all = dat["joint_all"]
    labels = dat["labels"]
    pre_rr = dat["pre_rr"]
    pre_rr_loc_all = dat["pre_rr_loc"]

    joint_50 = joint_all[:50]
    # Prefix HF noise median over ALL 50 prefix beats (NO normal-cluster identification!)
    hf0_50 = np.sum(np.diff(joint_50[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1_50 = np.sum(np.diff(joint_50[:, 187:], n=2, axis=1) ** 2, axis=1)
    med_hf0 = float(np.median(hf0_50)) + 1e-6
    med_hf1 = float(np.median(hf1_50)) + 1e-6

    post_joint = joint_all[50:]
    post_y = labels[50:]
    pre_rr_post = pre_rr[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    N = len(post_joint)

    hf0 = np.sum(np.diff(post_joint[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1 = np.sum(np.diff(post_joint[:, 187:], n=2, axis=1) ** 2, axis=1)
    sqi_hf_ratio = np.log1p(np.maximum(hf0 / med_hf0, hf1 / med_hf1))

    # Premature + clean beats ONLY (no ||y|| >= 2*eta_0 normal-pole gate!)
    cand_idx = np.where((pre_rr_loc <= v33.CAND_PRE_RR_LOC_MAX) & (sqi_hf_ratio <= v33.ONL_HF_MAX))[0]

    anchors = []
    cand_clusters = []
    for t in cand_idx:
        x_t = post_joint[t]
        matched_anc, best_adist = None, v33.CLUSTER_RADIUS
        for anc in anchors:
            d_a = float(np.linalg.norm(x_t - anc["b_V"]))
            if d_a <= best_adist:
                best_adist, matched_anc = d_a, anc
        if matched_anc is not None:
            if len(matched_anc["d_buf"]) >= 50:
                matched_anc["d_buf"].pop(0)
            matched_anc["d_buf"].append(best_adist)
            matched_anc["rad_V"] = max(float(np.percentile(matched_anc["d_buf"], 90)), 4.0)
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
                    arr_b = np.asarray(best_c["beats"], dtype=np.float64)
                    med_b = np.median(arr_b, axis=0)
                    dists_b = np.linalg.norm(arr_b - med_b, axis=1)
                    cohesion = float(np.median(dists_b))
                    if cohesion <= v33.CONFIRM_MAX_COHESION:
                        anchors.append({
                            "b_V": med_b,
                            "d_buf": list(dists_b[-50:]),
                            "rad_V": max(float(np.percentile(dists_b, 90)), 4.0),
                        })
                        cand_clusters.pop(best_c_idx)
            else:
                if len(cand_clusters) >= v33.MAX_CAND_CLUSTERS:
                    cand_clusters.sort(key=lambda c: len(c["beats"]))
                    cand_clusters.pop(0)
                cand_clusters.append({"beats": [x_t], "center": x_t.copy()})

    K = len(anchors)
    if K == 0:
        return {
            "y": post_y, "post_hours": dat["post_hours"], "K": 0,
            "X_vp": np.zeros((0, 12), dtype=np.float32),
        }

    # Vectorized 12D V-pole feature extraction across all N post-prefix beats
    bV_stack = np.vstack([a["b_V"] for a in anchors])       # (K, 374)
    rad_vec = np.array([a["rad_V"] for a in anchors])       # (K,)
    dists_all = np.empty((N, K), dtype=np.float64)
    corrs_mean = np.empty((N, K), dtype=np.float64)
    corrs_min = np.empty((N, K), dtype=np.float64)

    x0_all, x1_all = post_joint[:, :187], post_joint[:, 187:]
    for k in range(K):
        bv = bV_stack[k]
        dists_all[:, k] = np.linalg.norm(post_joint - bv[None, :], axis=1)
        c0 = v33.pearson_corr_rows(x0_all, bv[:187])
        c1 = v33.pearson_corr_rows(x1_all, bv[187:])
        corrs_mean[:, k] = 0.5 * (c0 + c1)
        corrs_min[:, k] = np.minimum(c0, c1)

    bk = np.argmin(dists_all, axis=1)
    idx_n = np.arange(N)
    d_min = dists_all[idx_n, bk]
    r_bk = rad_vec[bk]
    e_ball = np.maximum(0.0, d_min - r_bk)
    c_mean_bk = corrs_mean[idx_n, bk]
    c_min_bk = corrs_min[idx_n, bk]
    c_max_any = np.max(corrs_mean, axis=1)

    # Residual relative to nearest V-anchor b_{V,k*} (mirroring A3 sub-window norms without b_N!)
    dv = post_joint - bV_stack[bk]
    d_pre_v = np.sqrt(np.sum(dv[:, pipe2.SL_PRE] ** 2, axis=1) + np.sum(dv[:, 187 + np.arange(0, 71)] ** 2, axis=1))
    d_qrs_v = np.sqrt(np.sum(dv[:, pipe2.SL_QRS] ** 2, axis=1) + np.sum(dv[:, 187 + np.arange(71, 123)] ** 2, axis=1))
    d_stt_v = np.sqrt(np.sum(dv[:, pipe2.SL_STT] ** 2, axis=1) + np.sum(dv[:, 187 + np.arange(123, 187)] ** 2, axis=1))
    d_nonq_v = np.hypot(d_pre_v, d_stt_v)

    X_vp = np.column_stack([
        np.log1p(d_min / 10.0),
        np.log1p(e_ball / (r_bk + 1e-6)),
        d_min / (r_bk + 1e-6),
        c_mean_bk,
        c_min_bk,
        c_max_any,
        np.tanh((d_qrs_v / pipe2.R_QRS) / 0.75),
        np.tanh((d_stt_v / pipe2.R_STT) / 0.75),
        np.log((d_qrs_v + 0.5) / (d_nonq_v + 0.5)),
        np.log1p(d_pre_v / 5.0),
        np.log(np.clip(pre_rr_loc, 0.2, 3.0)),
        np.clip(pre_rr_post, 0.2, 2.5),
    ]).astype(np.float32)
    X_vp = np.nan_to_num(X_vp, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "y": post_y, "post_hours": dat["post_hours"], "K": K, "X_vp": X_vp,
    }


def _extract_src_vpole_one(args):
    db_name, rec_id, subject_id = args
    rp = os.path.join(v36.DB_DIRS[db_name], rec_id)
    chs = v62a.get_source_channels(db_name, rp)
    dat = eng.load_beats_and_rr(rp, "atr", chs, "AAMI_Map_v1.0", filter_mode="offline_filtfilt")
    vp = extract_vpole_only_features(dat)
    return {
        "db": db_name, "rec_id": rec_id, "patient_id": subject_id,
        "post_hours": float(vp["post_hours"]),
        "y": vp["y"].astype(np.int8),
        "s2_mask": np.ones(len(vp["y"]), dtype=bool) if vp["K"] > 0 else np.zeros(len(vp["y"]), dtype=bool),
        "X_s2": vp["X_vp"].astype(np.float64),
        "K": int(vp["K"]),
    }


def build_or_load_src_vpole(s2_by_db):
    done_all = all((CACHE_V62_SRC / f"{db}_vp_f32.npy").exists() and (CACHE_V62_SRC / f"{db}_vp_k.npy").exists() for db in v62a.DB_ORDER)
    if done_all:
        print("[cache] All 6 source DBs `SinglePole_V_RR` arrays already cached.", flush=True)
        return
    t0 = time.time()
    for db in v62a.DB_ORDER:
        vp_p = CACHE_V62_SRC / f"{db}_vp_f32.npy"
        vk_p = CACHE_V62_SRC / f"{db}_vp_k.npy"
        if vp_p.exists() and vk_p.exists():
            continue
        tasks = [(db, r["rec_id"], r["patient_id"]) for r in s2_by_db[db]]
        with ProcessPoolExecutor(max_workers=4) as ex:
            res_list = list(ex.map(_extract_src_vpole_one, tasks))
        rows = []
        ks = []
        for r_meta, vp in zip(s2_by_db[db], res_list):
            n = len(r_meta["y"])
            ks.append(int(vp["K"]))
            if vp["K"] > 0 and len(vp["X_s2"]) == n:
                rows.append(vp["X_s2"].astype(np.float32))
            else:
                rows.append(np.zeros((n, 12), dtype=np.float32))
        np.save(vp_p, np.vstack(rows))
        np.save(vk_p, np.array(ks, dtype=np.int16))
        print(f"  [vpole-src] {db}: {len(tasks)} recs cached ({time.time()-t0:.1f}s)", flush=True)


def fit_vpole_twobranch_r3(db_arrays, tr_meta, cols_N, cols_V, family):
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
        tr_sub = [r for r in tr_meta if fold_of_pat[r["patient_id"]] != fid and r["K_vp"] > 0]
        te_sub = [r for r in tr_meta if fold_of_pat[r["patient_id"]] == fid]
        XN, yN, wN = v62a.build_fullstream_branch_xy(db_arrays, tr_sub, cols_N, {0, 3}, seed=42 + fid)
        scN, clfN = v62a.fit_head(XN, yN, wN, family)
        del XN, yN, wN
        XV, yV, wV = v62a.build_fullstream_branch_xy(db_arrays, tr_sub, cols_V, {1}, seed=142 + fid)
        scV, clfV = v62a.fit_head(XV, yV, wV, family)
        del XV, yV, wV
        for te in te_sub:
            if te["K_vp"] > 0:
                X_sl = db_arrays[te["db"]][te["b0"] : te["b1"]]
                oof_sN.append(v62a.score_fullstream_chunks(clfN, scN, X_sl, cols_N, family))
                oof_sV.append(v62a.score_fullstream_chunks(clfV, scV, X_sl, cols_V, family))
                oof_y.append(te["y"])
                oof_pi.append(np.full(len(te["y"]), pat_to_idx[te["patient_id"]], dtype=np.int32))

    oof_sN = np.concatenate(oof_sN)
    oof_sV = np.concatenate(oof_sV)
    oof_y = np.concatenate(oof_y)
    oof_pi = np.concatenate(oof_pi)
    sigN = float(np.std(oof_sN)) + 1e-9
    sigV = float(np.std(oof_sV)) + 1e-9
    tN, tV = v62a.v56.solve_r3_joint2d_patfa(
        oof_sN, oof_sV, oof_y, oof_pi, pat_hours, pat_totF, b_tot=2.0, b_0f=1e9, n_grid=85,
    )
    del oof_sN, oof_sV, oof_y, oof_pi
    gc.collect()

    tr_kpos = [r for r in tr_meta if r["K_vp"] > 0]
    XN, yN, wN = v62a.build_fullstream_branch_xy(db_arrays, tr_kpos, cols_N, {0, 3}, seed=42)
    scN, clfN = v62a.fit_head(XN, yN, wN, family)
    del XN, yN, wN
    XV, yV, wV = v62a.build_fullstream_branch_xy(db_arrays, tr_kpos, cols_V, {1}, seed=142)
    scV, clfV = v62a.fit_head(XV, yV, wV, family)
    del XV, yV, wV
    gc.collect()
    return {
        "family": family, "fN": list(cols_N), "fV": list(cols_V),
        "scN": scN, "clfN": clfN, "tau_N": float(tN), "sig_N": sigN,
        "scV": scV, "clfV": clfV, "tau_V": float(tV), "sig_V": sigV,
    }


def eval_vpole_nested_lodo(s2_by_db, cols_N, cols_V, family):
    db_arrays = {db: np.load(CACHE_V62_SRC / f"{db}_vp_f32.npy", mmap_mode="r") for db in v62a.DB_ORDER}
    db_ks = {db: np.load(CACHE_V62_SRC / f"{db}_vp_k.npy") for db in v62a.DB_ORDER}
    meta_by_db = {}
    unit_scores = {}
    for db in v62a.DB_ORDER:
        mlist = []
        for idx, r in enumerate(s2_by_db[db]):
            rc = dict(r)
            rc["K_vp"] = int(db_ks[db][idx])
            mlist.append(rc)
            u = unit_scores.setdefault(r["patient_id"], {"tot_F": 0, "hours": 0.0, "y_list": [], "sc_list": []})
            u["tot_F"] += int(np.sum(r["y"] == 2))
            u["hours"] += float(r["post_hours"])
        meta_by_db[db] = mlist

    for tgt in v62a.DB_ORDER:
        tr_meta = [r for db in v62a.DB_ORDER if db != tgt for r in meta_by_db[db]]
        mod = fit_vpole_twobranch_r3(db_arrays, tr_meta, cols_N, cols_V, family)
        for te in meta_by_db[tgt]:
            if te["K_vp"] > 0:
                X_sl = db_arrays[tgt][te["b0"] : te["b1"]]
                sN = v62a.score_fullstream_chunks(mod["clfN"], mod["scN"], X_sl, cols_N, family)
                sV = v62a.score_fullstream_chunks(mod["clfV"], mod["scV"], X_sl, cols_V, family)
                sc = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
            else:
                sc = np.full(len(te["y"]), -1e9, dtype=np.float64)
            u = unit_scores[te["patient_id"]]
            u["y_list"].append(te["y"])
            u["sc_list"].append(sc)
        del mod
        gc.collect()

    summ, pu = v62a.summarize_unit_scores(unit_scores)
    all_meta = [r for db in v62a.DB_ORDER for r in meta_by_db[db]]
    locked = fit_vpole_twobranch_r3(db_arrays, all_meta, cols_N, cols_V, family)
    return summ, pu, locked


def eval_stage2_cache_cohort(cache_dir, reg_recs, sid_key, mod):
    unit_scores = {}
    for r in reg_recs:
        rid = r["rec_id"]
        sid = r[sid_key]
        z = np.load(cache_dir / f"{rid}.npz")
        m = json.loads((cache_dir / f"{rid}_meta.json").read_text(encoding="utf-8"))
        y = z["y"].astype(np.int8)
        s2_mask = z["s2_mask"].astype(bool)
        X = z["X29_s2"].astype(np.float64)
        sc = np.full(len(y), -1e9, dtype=np.float64)
        if len(X) > 0:
            sN = v62a.score_head(mod["clfN"], mod["scN"], X[:, mod["fN"]], mod["family"])
            sV = v62a.score_head(mod["clfV"], mod["scV"], X[:, mod["fV"]], mod["family"])
            sc[s2_mask] = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
        u = unit_scores.setdefault(sid, {"tot_F": 0, "hours": 0.0, "y_list": [], "sc_list": []})
        u["tot_F"] += int(np.sum(y == 2))
        u["hours"] += float(m["post_hours"])
        u["y_list"].append(y)
        u["sc_list"].append(sc)
    return v62a.summarize_unit_scores(unit_scores)


_EXT_MODELS = None


def _init_vp_worker(models_dict):
    global _EXT_MODELS
    _EXT_MODELS = models_dict


def _score_ext_vp_record(args):
    rec_info, raw_dir_s, ann_ext, label_map, cache_dir_s = args
    rid = rec_info["rec_id"]
    cp = Path(cache_dir_s) / f"{rid}_vp.pkl"
    if cp.exists():
        return str(cp)
    rp = str(Path(raw_dir_s) / rid)
    chs = tuple(rec_info.get("channel_indices", [0, 1]))
    dat = eng.load_beats_and_rr(rp, ann_ext, chs, label_map, filter_mode="offline_filtfilt")
    vp = extract_vpole_only_features(dat)
    y = vp["y"]
    res = {
        "rec_id": rid,
        "post_hours": float(vp["post_hours"]),
        "tot_F": int(np.sum(y == 2)),
        "K": int(vp["K"]),
        "models": {},
    }
    for mname, mod in _EXT_MODELS.items():
        if vp["K"] > 0 and len(vp["X_vp"]) > 0:
            X = vp["X_vp"].astype(np.float64)
            sN = v62a.score_head(mod["clfN"], mod["scN"], X[:, mod["fN"]], mod["family"])
            sV = v62a.score_head(mod["clfV"], mod["scV"], X[:, mod["fV"]], mod["family"])
            sc = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
            pred = (sc >= 0.0)
            tp = int(np.sum(pred & (y == 2)))
            fp_ns = int(np.sum(pred & ((y == 0) | (y == 3))))
            fp_v = int(np.sum(pred & (y == 1)))
        else:
            tp, fp_ns, fp_v = 0, 0, 0
        res["models"][mname] = {"TP": tp, "FP_NS": fp_ns, "FP_V": fp_v}
    with open(cp, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    dat_p = Path(raw_dir_s) / f"{rid}.dat"
    try:
        if dat_p.exists():
            dat_p.unlink()
    except OSError:
        pass
    return str(cp)


def stream_external_vpole(cohort_name, reg_path, sid_key, base_url, ann_ext, label_map, vp_models):
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    recs = reg["included_records"]
    cdir = CACHE_V62E_EXT / cohort_name
    rdir = CACHE_V62E_EXT / f"raw_{cohort_name}"
    cdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    todo = [r for r in recs if not (cdir / f"{r['rec_id']}_vp.pkl").exists()]
    print(f"[{cohort_name}] total={len(recs)}, cached={len(recs)-len(todo)}, todo={len(todo)}", flush=True)
    if todo:
        t0 = time.time()
        sums = v62b.fetch_sha256sums(rdir, base_url)
        sem = threading.Semaphore(12)
        q_ready = queue.Queue()

        def _dl_one(r):
            rid = r["rec_id"]
            sem.acquire()
            try:
                for ext in ("hea", ann_ext, "dat"):
                    fn = f"{rid}.{ext}"
                    v62b.download_verified(rdir, base_url, fn, sums.get(fn))
                q_ready.put(r)
            except Exception as e:
                q_ready.put(e)

        def _producer():
            with ThreadPoolExecutor(max_workers=8) as tex:
                futs = [tex.submit(_dl_one, r) for r in todo]
                for f in futs:
                    f.result()
            q_ready.put(None)

        prod_t = threading.Thread(target=_producer, daemon=True)
        prod_t.start()

        done_n = 0
        with ProcessPoolExecutor(max_workers=4, initializer=_init_vp_worker, initargs=(vp_models,)) as pex:
            pending = []
            while True:
                item = q_ready.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                fut = pex.submit(_score_ext_vp_record, (item, str(rdir), ann_ext, label_map, str(cdir)))
                fut.add_done_callback(lambda _: sem.release())
                pending.append(fut)
            for fut in pending:
                fut.result()
                done_n += 1
                if done_n % 15 == 0 or done_n == len(todo):
                    print(f"  [{cohort_name}] {done_n}/{len(todo)} streamed ({time.time()-t0:.1f}s)", flush=True)

    # Aggregate per-unit metrics
    per_model = {m: {} for m in vp_models}
    kpos_units = set()
    all_units = set()
    for r in recs:
        rid = r["rec_id"]
        sid = r[sid_key]
        all_units.add(sid)
        d = pickle.load(open(cdir / f"{rid}_vp.pkl", "rb"))
        if d["K"] > 0:
            kpos_units.add(sid)
        for m in vp_models:
            u = per_model[m].setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
            u["tot_F"] += d["tot_F"]
            u["hours"] += d["post_hours"]
            u["TP"] += d["models"][m]["TP"]
            u["FP_NS"] += d["models"][m]["FP_NS"]
            u["FP_V"] += d["models"][m]["FP_V"]

    summaries, per_units = {}, {}
    for m, umap in per_model.items():
        subjs = sorted(umap.keys())
        pu = {}
        for s in subjs:
            u = umap[s]
            faph = (u["FP_NS"] + u["FP_V"]) / max(u["hours"], 1e-9)
            pu[s] = {
                "tot_F": u["tot_F"], "hours": u["hours"],
                "TP": u["TP"], "FP_NS": u["FP_NS"], "FP_V": u["FP_V"],
                "FAPH": faph,
                "Rec_%": (100.0 * u["TP"] / u["tot_F"]) if u["tot_F"] > 0 else None,
            }
        fpos = [pu[s] for s in subjs if pu[s]["tot_F"] > 0]
        zf = [pu[s] for s in subjs if pu[s]["tot_F"] == 0]
        all_u = [pu[s] for s in subjs]
        fa_all = np.array([v["FAPH"] for v in all_u])
        fa_0f = np.array([v["FAPH"] for v in zf])
        summaries[m] = {
            "TP": int(sum(v["TP"] for v in all_u)),
            "FP_NS": int(sum(v["FP_NS"] for v in all_u)),
            "FP_V": int(sum(v["FP_V"] for v in all_u)),
            "MacroRec_%": round(float(np.mean([v["Rec_%"] for v in fpos])), 2),
            "Fpos_hit": int(sum(1 for v in fpos if v["TP"] > 0)),
            "n_Fpos": len(fpos),
            "MeanFAPH_All": round(float(np.mean(fa_all)), 3),
            "MeanFAPH_0F": round(float(np.mean(fa_0f)), 3),
            "PooledFAPH": round(float(sum(v["FP_NS"] + v["FP_V"] for v in all_u) / sum(v["hours"] for v in all_u)), 3),
            "P95_FAPH": round(float(np.percentile(fa_all, 95)), 2),
            "Pct_LE2": round(float(100.0 * np.mean(fa_all <= 2.0 + 1e-12)), 1),
            "ZeroF_Max": round(float(np.max(fa_0f)), 2),
            "Kpos_units": f"{len(kpos_units)}/{len(all_units)}",
        }
        per_units[m] = pu
    return summaries, per_units


def get_full_a3_per_unit_ext(cache_ext_sub, reg_recs, sid_key, mkey):
    umap = {}
    for r in reg_recs:
        rid = r["rec_id"]
        sid = r[sid_key]
        d = pickle.load(open(cache_ext_sub / f"{rid}_v62.pkl", "rb"))
        sc = d["scores"][mkey]
        u = umap.setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
        u["tot_F"] += int(d["tot_F"])
        u["hours"] += float(d["post_hours"])
        u["TP"] += int(sc["TP"])
        u["FP_NS"] += int(sc["FP_NS"])
        u["FP_V"] += int(sc["FP_V"])
    pu = {}
    for s, u in umap.items():
        pu[s] = {
            "tot_F": u["tot_F"], "hours": u["hours"],
            "TP": u["TP"], "FP_NS": u["FP_NS"], "FP_V": u["FP_V"],
            "FAPH": (u["FP_NS"] + u["FP_V"]) / max(u["hours"], 1e-9),
            "Rec_%": (100.0 * u["TP"] / u["tot_F"]) if u["tot_F"] > 0 else None,
        }
    return pu


def main():
    t_start = time.time()
    s2_by_db = {db: pickle.load(open(CACHE_V62_SRC / f"{db}_s2_variants.pkl", "rb")) for db in v62a.DB_ORDER}
    t0_items_by_db = v62a.build_stage2_items(s2_by_db, "Tier0")

    print("=== 1. Evaluating Stage-2 `Abl_No_LocalNormal` (Chord-Only 6D+7D) on 6-DB LODO ===", flush=True)
    lodo_ref_pu = {}
    lodo_nln_sum, lodo_nln_pu, locked_nln = {}, {}, {}
    for fam in ("SVM", "LightGBM"):
        _, lodo_ref_pu[fam], _ = v62a.eval_stage2_nested_lodo(t0_items_by_db, eng.IDX_A3_N, eng.IDX_C3_V, fam)
        s_nln, pu_nln, lk_nln = v62a.eval_stage2_nested_lodo(t0_items_by_db, IDX_CHORD_ONLY_N, IDX_CHORD_ONLY_V, fam)
        lodo_nln_sum[f"{fam}__Abl_No_LocalNormal"] = s_nln
        lodo_nln_pu[f"{fam}__Abl_No_LocalNormal"] = pu_nln
        locked_nln[f"{fam}__Abl_No_LocalNormal"] = lk_nln
        print(f"  [LODO] {fam}__Abl_No_LocalNormal: Rec={s_nln['MacroRec_%']}% TP={s_nln['TP']} FAPH={s_nln['MeanFAPH_All']}/{s_nln['MeanFAPH_0F']}", flush=True)

    print("=== 2. Evaluating Full-Pipeline `SinglePole_V_RR` (`w/o Normal Pole`, 10D+12D) on 6-DB LODO ===", flush=True)
    build_or_load_src_vpole(s2_by_db)
    fN_vp = list(range(10))
    fV_vp = list(range(12))
    lodo_vp_sum, lodo_vp_pu, locked_vp = {}, {}, {}
    for fam in ("SVM", "LightGBM"):
        s_vp, pu_vp, lk_vp = eval_vpole_nested_lodo(s2_by_db, fN_vp, fV_vp, fam)
        lodo_vp_sum[f"{fam}__SinglePole_V_RR"] = s_vp
        lodo_vp_pu[f"{fam}__SinglePole_V_RR"] = pu_vp
        locked_vp[f"{fam}__SinglePole_V_RR"] = lk_vp
        print(f"  [LODO] {fam}__SinglePole_V_RR: Rec={s_vp['MacroRec_%']}% TP={s_vp['TP']} FAPH={s_vp['MeanFAPH_All']}/{s_vp['MeanFAPH_0F']}", flush=True)

    # Source paired stats vs Full_A3
    src_paired = {}
    for fam in ("SVM", "LightGBM"):
        for mk, pu_m in [
            (f"{fam}__Abl_No_LocalNormal", lodo_nln_pu[f"{fam}__Abl_No_LocalNormal"]),
            (f"{fam}__SinglePole_V_RR", lodo_vp_pu[f"{fam}__SinglePole_V_RR"]),
        ]:
            src_paired[mk] = v62a.paired_compare(lodo_ref_pu[fam], pu_m)

    # External LTSTDB & MGHDB evaluation
    lt_reg = json.loads((ROOT / "registry/target_cohort_ltstdb_v1.json").read_text(encoding="utf-8"))["included_records"]
    mg_reg = json.loads((ROOT / "registry/target_cohort_mghdb_v1.json").read_text(encoding="utf-8"))["included_records"]

    print("=== 3. Evaluating `Abl_No_LocalNormal` on Cached LTSTDB & MGHDB ===", flush=True)
    ext_lt_sum, ext_lt_pu = {}, {}
    ext_mg_sum, ext_mg_pu = {}, {}
    for mk, mod in locked_nln.items():
        s_lt, pu_lt = eval_stage2_cache_cohort(CACHE_V59, lt_reg, "subject_id", mod)
        s_mg, pu_mg = eval_stage2_cache_cohort(CACHE_V60, mg_reg, "subject_id_tier1", mod)
        ext_lt_sum[mk], ext_lt_pu[mk] = s_lt, pu_lt
        ext_mg_sum[mk], ext_mg_pu[mk] = s_mg, pu_mg
        print(f"  [EXT] {mk}: LT Rec={s_lt['MacroRec_%']}% (TP={s_lt['TP']}, FAPH={s_lt['MeanFAPH_All']}) | MG Rec={s_mg['MacroRec_%']}% (TP={s_mg['TP']}, FAPH={s_mg['MeanFAPH_All']})", flush=True)

    print("=== 4. Streaming `SinglePole_V_RR` (`w/o Normal Pole`) on LTSTDB & MGHDB ===", flush=True)
    s_vp_mg, pu_vp_mg = stream_external_vpole(
        "mghdb", ROOT / "registry/target_cohort_mghdb_v1.json", "subject_id_tier1",
        "https://physionet.org/files/mghdb/1.0.0/", "ari", "AAMI_Map_v1.0", locked_vp,
    )
    s_vp_lt, pu_vp_lt = stream_external_vpole(
        "ltstdb", ROOT / "registry/target_cohort_ltstdb_v1.json", "subject_id",
        "https://physionet.org/files/ltstdb/1.0.0/", "atr", "AAMI_Map_v1.1", locked_vp,
    )
    ext_lt_sum.update(s_vp_lt); ext_lt_pu.update(pu_vp_lt)
    ext_mg_sum.update(s_vp_mg); ext_mg_pu.update(pu_vp_mg)

    # External paired stats vs Full_A3
    lt_paired, mg_paired = {}, {}
    for fam in ("SVM", "LightGBM"):
        ref_lt = get_full_a3_per_unit_ext((_DATA_ROOT / "cache_v62_external/ltstdb"), lt_reg, "subject_id", f"{fam}__Full_A3")
        ref_mg = get_full_a3_per_unit_ext((_DATA_ROOT / "cache_v62_external/mghdb"), mg_reg, "subject_id_tier1", f"{fam}__Full_A3")
        for sfx in ("Abl_No_LocalNormal", "SinglePole_V_RR"):
            mk = f"{fam}__{sfx}"
            lt_paired[mk] = v62a.paired_compare(ref_lt, ext_lt_pu[mk])
            mg_paired[mk] = v62a.paired_compare(ref_mg, ext_mg_pu[mk])

    out = {
        "Version": "v62e_symmetric_normal_pole_ablation",
        "Date": "2026-11-24",
        "Source_6DB_LODO": {
            "Models": {**lodo_nln_sum, **lodo_vp_sum},
            "Paired_vs_FullA3": src_paired,
        },
        "LTSTDB": {
            "Models": {k: {kk: vv for kk, vv in v.items() if kk != "Match20_%"} for k, v in ext_lt_sum.items()},
            "Paired_vs_FullA3": lt_paired,
        },
        "MGHDB_Tier1": {
            "Models": {k: {kk: vv for kk, vv in v.items() if kk != "Match20_%"} for k, v in ext_mg_sum.items()},
            "Paired_vs_FullA3": mg_paired,
        },
        "runtime_sec": round(time.time() - t_start, 1),
    }
    AUD_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"=== DONE v62e in {out['runtime_sec']}s -> {AUD_OUT} ===", flush=True)


if __name__ == "__main__":
    main()
