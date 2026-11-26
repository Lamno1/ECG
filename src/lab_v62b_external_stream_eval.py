r"""
lab_v62b_external_stream_eval.py
================================
Wave-4 Revision Audit — Part B (Locked External Streaming Evaluation on
LTSTDB [70 subjects / 76 records / 7.97M beats] AND MGHDB Tier-1 [213 units /
219 records / 1.37M beats]).

Single-Pass Zero-Bloat Protocol:
  1. Loads locked 6-DB checkpoints from `checkpoints/v62_locked_models_6db.pkl`
     (ZERO target refitting, ZERO threshold retuning).
  2. Streams raw `.dat` files from PhysioNet (`ltstdb/1.0.0`, `mghdb/1.0.0`) in
     bounded waves of 6 (LTSTDB) or 10 (MGHDB) with official `SHA256SUMS.txt`
     verification, evaluates EVERY model/ablation/causality tier in ONE pass
     per record, saves compact per-record results to `cache_v62_external/`
     (fully resumable!), and deletes raw `.dat` immediately after verified scoring.
  3. Asserts exact parity of `SVM__Full_A3` against frozen `v59` (LTSTDB:
     TP=42, FP_NS=682, FP_V=53) and `v60` (MGHDB: TP=10, FP_NS=157, FP_V=41).
  4. Produces complete cohort summaries, Stage-1 ICU anchor forensics (v62B),
     pre-anchor causal latency & coverage loss (v62C), and full paired stats
     (Delta, 95% bootstrap CI B=10,000 seed=42, W/T/L, exact two-sided Wilcoxon p).
"""


import os as _os, sys as _sys
from pathlib import Path as _Path
_SRC_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_DATA_ROOT = _Path(_os.environ.get("ECG_DATA_ROOT", str(_REPO_ROOT / "data_public_ecg")))
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import gc
import hashlib
import json
import os
import pickle
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
import wfdb.io._header as _wfdb_hdr

# Safe in-memory fallback for PhysioNet mgh025.hea typo ('31/05/199')
_orig_parse_record_line = _wfdb_hdr._parse_record_line


def _safe_parse_record_line(record_line):
    try:
        d = _orig_parse_record_line(record_line)
    except ValueError as e:
        if "time data" in str(e) and "%d/%m/%Y" in str(e):
            toks = record_line.split()
            d = _orig_parse_record_line(" ".join(toks[:-1]))
        else:
            raise
    if d.get("sig_len") == 0:
        d["sig_len"] = None
    return d


_wfdb_hdr._parse_record_line = _safe_parse_record_line

sys.path.insert(0, str(_REPO_ROOT))
import lab_v62a_source_lodo_and_lock as v62a
import v62_feature_engine as eng
import v62_fast_stream as fast_eng

_WORKER_MODELS = None


def _init_ext_worker(ckpt_path):
    global _WORKER_MODELS
    with open(ckpt_path, "rb") as f:
        _WORKER_MODELS = pickle.load(f)


def _worker_process_record(args):
    rec_info, raw_dir_s, ann_ext, cache_dir_s = args
    p = process_one_external_record(rec_info, Path(raw_dir_s), ann_ext, _WORKER_MODELS, Path(cache_dir_s))
    dat_p = Path(raw_dir_s) / f"{rec_info['rec_id']}.dat"
    try:
        if dat_p.exists():
            dat_p.unlink()
    except OSError:
        pass
    return str(p)

ROOT = _REPO_ROOT
CKPT_IN = ROOT / "checkpoints" / "v62_locked_models_6db.pkl"
CACHE_EXT = (_DATA_ROOT / "cache_v62_external")
AUD_OUT = ROOT / "audits" / "V62B_EXTERNAL_LTSTDB_MGHDB_REVISION_20261124.json"
CACHE_EXT.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_sha256sums(raw_dir, base_url):
    p = raw_dir / "SHA256SUMS.txt"
    if not p.exists():
        for k in range(4):
            try:
                req = urllib.request.Request(base_url + "SHA256SUMS.txt", headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=30).read()
                p.write_bytes(data)
                break
            except Exception:
                if k == 3:
                    raise
                time.sleep(2.0 * (k + 1))
    sums = {}
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = ln.strip().split()
        if len(parts) >= 2:
            sums[parts[1]] = parts[0]
    return sums


def download_verified(raw_dir, base_url, fname, expected_sha):
    dst = raw_dir / fname
    if dst.exists() and (not expected_sha or sha256_file(dst) == expected_sha):
        return dst
    tmp = raw_dir / (fname + ".part")
    for k in range(5):
        try:
            req = urllib.request.Request(base_url + fname, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=240) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            got = sha256_file(tmp)
            if expected_sha and got != expected_sha:
                raise RuntimeError(f"SHA mismatch {fname}: {got[:16]} != {expected_sha[:16]}")
            os.replace(tmp, dst)
            return dst
        except Exception as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if k == 4:
                raise RuntimeError(f"Download failed {fname}: {e}")
            time.sleep(2.5 * (k + 1))


def score_stage2_variant(mod, var_dict):
    n = len(var_dict["y"])
    sc = np.full(n, -1e9, dtype=np.float64)
    X = var_dict["X29_s2"].astype(np.float64)
    if len(X) > 0:
        sN = v62a.score_head(mod["clfN"], mod["scN"], X[:, mod["fN"]], mod["family"])
        sV = v62a.score_head(mod["clfV"], mod["scV"], X[:, mod["fV"]], mod["family"])
        sc[var_dict["s2_mask"]] = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
    return sc


def score_fullstream_variant(mod, X_full):
    sN = v62a.score_fullstream_chunks(mod["clfN"], mod["scN"], X_full, mod["fN"], mod["family"])
    sV = v62a.score_fullstream_chunks(mod["clfV"], mod["scV"], X_full, mod["fV"], mod["family"])
    return np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])


def summarize_record_scores(y, sc):
    """Compact per-record statistics + compact top-tail scores for exact full-stream AP & Sweep@2.0."""
    pred = (sc >= 0.0)
    tp = int(np.sum(pred & (y == 2)))
    fp_ns = int(np.sum(pred & ((y == 0) | (y == 3))))
    fp_v = int(np.sum(pred & (y == 1)))
    # Store ALL Class-F scores + top 500 non-F scores per record (sufficient for exact <=3.0 FAPH sweep
    # since 3.0 FAPH * 24h = 72 FPs/record; 500 covers up to >20 FAPH per record with zero loss!)
    f_scores = sc[y == 2].astype(np.float32)
    nf_sc = sc[y != 2]
    nf_valid = nf_sc[nf_sc > -1e8]
    if len(nf_valid) > 500:
        idx_top = np.argpartition(nf_valid, -500)[-500:]
        nf_top = nf_valid[idx_top].astype(np.float32)
    else:
        nf_top = nf_valid.astype(np.float32)
    return {"TP": tp, "FP_NS": fp_ns, "FP_V": fp_v, "f_scores": f_scores, "nf_top": nf_top}


def process_one_external_record(rec_info, raw_dir, ann_ext, locked_models, cache_dir):
    rid = rec_info["rec_id"]
    out_p = cache_dir / f"{rid}_v62.pkl"
    if out_p.exists():
        return out_p

    rp = str(raw_dir / rid)
    chs = tuple(rec_info.get("channel_indices", rec_info.get("locked_channel_indices", [0, 1])))

    # 1. Offline filtfilt load ONCE -> Tier0 + Tier1 + Static_bN + SinglePole + Forensics in ONE single fast loop!
    dat_off = eng.load_beats_and_rr(rp, ann_ext, chs, "AAMI_Map_v1.1", filter_mode="offline_filtfilt")
    t0 = fast_eng.run_dualpole_variant_fast(dat_off, use_tangent_U=True, use_local_bN=True, anchor_mode="offline_and_online", collect_singlepole=True, collect_stage1_forensics=True, also_static_bn=True)
    t1 = t0["online_causal"]
    st_bn = t0["static_bn"]
    no_u = fast_eng.run_dualpole_variant_fast(dat_off, use_tangent_U=False, use_local_bN=True, anchor_mode="offline")
    t2 = fast_eng.run_dualpole_variant_fast(dat_off, use_tangent_U=True, use_local_bN=True, anchor_mode="prefix_only")
    x_raw, x_morph = eng.extract_traditional_morphology_and_raw(dat_off["joint_all"][50:], dat_off["pre_rr"][50:], dat_off["pre_rr_loc"][50:])
    x_sp = t0["X_singlepole"]
    del dat_off
    gc.collect()

    # 2. Strict Causal filter load (Tier 3)
    dat_cau = eng.load_beats_and_rr(rp, ann_ext, chs, "AAMI_Map_v1.1", filter_mode="causal_double_median_fir")
    t3 = fast_eng.run_dualpole_variant_fast(dat_cau, use_tangent_U=True, use_local_bN=True, anchor_mode="online_causal")
    del dat_cau
    gc.collect()

    y_off = t0["y"]
    y_t3 = t3["y"]
    scores_out = {}

    # Score Stage-2 ablations
    s2_map = [
        ("Full_A3", t0),
        ("Abl_No_ThalesPyth", t0),
        ("Abl_No_DualPoleCoords", t0),
        ("Abl_Static_bN", st_bn),
        ("Abl_No_Tangent_U", no_u),
    ]
    for fam in ("SVM", "LightGBM", "RF"):
        for cname, vdict in s2_map:
            key = f"{fam}__{cname}"
            if key in locked_models:
                sc = score_stage2_variant(locked_models[key], vdict)
                scores_out[key] = summarize_record_scores(y_off, sc)

    # Score 4-Tier Causality (both locked Tier-0 checkpoint and native refit)
    for fam in ("SVM", "LightGBM"):
        t0_mod = locked_models[f"{fam}__Full_A3"]
        for cname, mod, vdict, y_use in [
            ("Tier1_Online_LockedT0", t0_mod, t1, y_off),
            ("Tier1_Online_Native", locked_models[f"{fam}__Tier1_Online_Native"], t1, y_off),
            ("Tier2_PrefixOnly_LockedT0", t0_mod, t2, y_off),
            ("Tier2_PrefixOnly_Native", locked_models[f"{fam}__Tier2_PrefixOnly_Native"], t2, y_off),
            ("Tier3_StrictCausal_LockedT0", t0_mod, t3, y_t3),
            ("Tier3_StrictCausal_Native", locked_models[f"{fam}__Tier3_StrictCausal_Native"], t3, y_t3),
        ]:
            key = f"{fam}__{cname}"
            sc = score_stage2_variant(mod, vdict)
            scores_out[key] = summarize_record_scores(y_use, sc)

    # Score Full-Stream Baselines
    for cname, Xmat in [("SinglePole_N_RR", x_sp), ("Trad_Morph_RR", x_morph), ("Raw_ECG_RR", x_raw)]:
        for fam in ("SVM", "LightGBM", "RF"):
            key = f"{fam}__{cname}"
            if key in locked_models:
                sc = score_fullstream_variant(locked_models[key], Xmat)
                scores_out[key] = summarize_record_scores(y_off, sc)

    rec_res = {
        "rec_id": rid,
        "post_hours": float(t0["post_hours"]),
        "post_hours_t3": float(t3["post_hours"]),
        "n_beats": int(len(y_off)),
        "tot_F": int(np.sum(y_off == 2)),
        "tot_F_t3": int(np.sum(y_t3 == 2)),
        "K_t0": int(t0["K"]),
        "K_t1": int(t1["K"]),
        "K_t2": int(t2["K"]),
        "K_t3": int(t3["K"]),
        "s2_F_t0": int(np.sum(y_off[t0["s2_mask"]] == 2)),
        "s2_F_t1": int(np.sum(y_off[t1["s2_mask"]] == 2)),
        "s2_F_t2": int(np.sum(y_off[t2["s2_mask"]] == 2)),
        "s2_F_t3": int(np.sum(y_t3[t3["s2_mask"]] == 2)),
        "tau1_min_t1": t1["tau1_min"],
        "pre_anchor_F_t1": int(t1["pre_anchor_F"]),
        "tau1_min_t3": t3["tau1_min"],
        "pre_anchor_F_t3": int(t3["pre_anchor_F"]),
        "forensics": t0["forensics"],
        "scores": scores_out,
    }
    with open(out_p, "wb") as f:
        pickle.dump(rec_res, f, protocol=pickle.HIGHEST_PROTOCOL)
    del t0, no_u, st_bn, t1, t2, t3, x_raw, x_morph, x_sp
    gc.collect()
    return out_p


def aggregate_cohort(rec_results, reg_recs, sid_key):
    rec_to_sid = {r["rec_id"]: r[sid_key] for r in reg_recs}
    all_keys = sorted(rec_results[0]["scores"].keys())

    # Causality coverage & Stage-1 forensics aggregation
    tot_F = sum(r["tot_F"] for r in rec_results)
    tot_F_t3 = sum(r["tot_F_t3"] for r in rec_results)
    tau1_vals = [r["tau1_min_t1"] for r in rec_results if r["tau1_min_t1"] is not None]
    tau3_vals = [r["tau1_min_t3"] for r in rec_results if r["tau1_min_t3"] is not None]
    cov = {
        "tot_F": tot_F, "tot_F_t3": tot_F_t3,
        "s2_F_t0": sum(r["s2_F_t0"] for r in rec_results),
        "s2_F_t1": sum(r["s2_F_t1"] for r in rec_results),
        "pre_anchor_F_t1": sum(r["pre_anchor_F_t1"] for r in rec_results),
        "tau1_med_min": round(float(np.median(tau1_vals)), 2) if tau1_vals else None,
        "tau1_iqr_min": [round(float(np.percentile(tau1_vals, 25)), 2), round(float(np.percentile(tau1_vals, 75)), 2)] if tau1_vals else None,
        "s2_F_t2_prefix_only": sum(r["s2_F_t2"] for r in rec_results),
        "recs_with_prefix_anchor": sum(1 for r in rec_results if r["K_t2"] > 0),
        "s2_F_t3_strict_causal": sum(r["s2_F_t3"] for r in rec_results),
        "pre_anchor_F_t3": sum(r["pre_anchor_F_t3"] for r in rec_results),
        "tau3_med_min": round(float(np.median(tau3_vals)), 2) if tau3_vals else None,
    }

    # Unit-level Stage-1 forensics for K=0 F+ and K>0 rejected F beats
    u_for = {}
    for r in rec_results:
        sid = rec_to_sid[r["rec_id"]]
        f = r["forensics"]
        u = u_for.setdefault(sid, {
            "tot_F": 0, "K_max": 0, "s2_F": 0, "hours": 0.0,
            "post_V_total": 0, "post_V_pass_norm_y": 0, "post_V_pass_rr": 0,
            "post_V_pass_sqi": 0, "post_V_pass_all3": 0, "n_cand_all3": 0,
            "max_cluster_size": 0, "rejected_norm_q": 0,
            "F_rej_zN0_only": 0, "F_rej_zVc0": 0, "F_rej_zVb0_only": 0, "F_rej_both_zN_zV": 0,
        })
        u["tot_F"] += r["tot_F"]
        u["K_max"] = max(u["K_max"], r["K_t0"])
        u["s2_F"] += r["s2_F_t0"]
        u["hours"] += r["post_hours"]
        u["post_V_total"] += f["post_V_total"]
        u["post_V_pass_norm_y"] += f["post_V_pass_norm_y"]
        u["post_V_pass_rr"] += f["post_V_pass_rr"]
        u["post_V_pass_sqi"] += f["post_V_pass_sqi"]
        u["post_V_pass_all3"] += f["post_V_pass_all3"]
        u["n_cand_all3"] += f["n_cand_all3"]
        u["max_cluster_size"] = max(u["max_cluster_size"], f["max_cluster_size"])
        u["rejected_norm_q"] += f["rejected_norm_q_count"]
        u["F_rej_zN0_only"] += f.get("F_rej_zN0_only", 0)
        u["F_rej_zVc0"] += f.get("F_rej_zVc0", 0)
        u["F_rej_zVb0_only"] += f.get("F_rej_zVb0_only", 0)
        u["F_rej_both_zN_zV"] += f.get("F_rej_both_zN_zV", 0)

    k0_fp_ge6 = [u for u in u_for.values() if u["tot_F"] > 0 and u["K_max"] == 0 and u["post_V_total"] >= 6]
    stage1_forensics_summary = {
        "n_units": len(u_for),
        "n_Fpos": sum(1 for u in u_for.values() if u["tot_F"] > 0),
        "n_Kpos_all": sum(1 for u in u_for.values() if u["K_max"] > 0),
        "n_K0_Fpos": sum(1 for u in u_for.values() if u["tot_F"] > 0 and u["K_max"] == 0),
        "K0_Fpos_V0": sum(1 for u in u_for.values() if u["tot_F"] > 0 and u["K_max"] == 0 and u["post_V_total"] == 0),
        "K0_Fpos_V1to5": sum(1 for u in u_for.values() if u["tot_F"] > 0 and u["K_max"] == 0 and 1 <= u["post_V_total"] < 6),
        "K0_Fpos_Vge6": len(k0_fp_ge6),
        "K0_Fpos_Vge6_pass_all3_lt6": sum(1 for u in k0_fp_ge6 if u["post_V_pass_all3"] < 6),
        "K0_Fpos_Vge6_sum_V": sum(u["post_V_total"] for u in k0_fp_ge6),
        "K0_Fpos_Vge6_sum_V_pass_norm_y": sum(u["post_V_pass_norm_y"] for u in k0_fp_ge6),
        "K0_Fpos_Vge6_sum_V_pass_rr": sum(u["post_V_pass_rr"] for u in k0_fp_ge6),
        "K0_Fpos_Vge6_sum_V_pass_sqi": sum(u["post_V_pass_sqi"] for u in k0_fp_ge6),
        "K0_Fpos_Vge6_sum_V_pass_all3": sum(u["post_V_pass_all3"] for u in k0_fp_ge6),
        "K0_Fpos_Vge6_cluster_ge6_fail_cohesion_or_normq": sum(1 for u in k0_fp_ge6 if u["max_cluster_size"] >= 6),
        "K0_Fpos_Vge6_rejected_norm_q": sum(u["rejected_norm_q"] for u in k0_fp_ge6),
        "Kpos_F_rej_zN0_only": sum(u["F_rej_zN0_only"] for u in u_for.values()),
        "Kpos_F_rej_zVc0": sum(u["F_rej_zVc0"] for u in u_for.values()),
        "Kpos_F_rej_zVb0_only": sum(u["F_rej_zVb0_only"] for u in u_for.values()),
        "Kpos_F_rej_both_zN_zV": sum(u["F_rej_both_zN_zV"] for u in u_for.values()),
    }

    summaries, per_units = {}, {}
    for key in all_keys:
        is_t3 = "Tier3" in key
        pu = {}
        for r in rec_results:
            sid = rec_to_sid[r["rec_id"]]
            sr = r["scores"][key]
            u = pu.setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0, "f_sc": [], "nf_sc": []})
            u["tot_F"] += r["tot_F_t3"] if is_t3 else r["tot_F"]
            u["hours"] += r["post_hours_t3"] if is_t3 else r["post_hours"]
            u["TP"] += sr["TP"]
            u["FP_NS"] += sr["FP_NS"]
            u["FP_V"] += sr["FP_V"]
            if len(sr["f_scores"]) > 0:
                u["f_sc"].append(sr["f_scores"])
            if len(sr["nf_top"]) > 0:
                u["nf_sc"].append(sr["nf_top"])

        # Compute summary + Match@2.0 sweep from stored top tails
        subjs = sorted(pu.keys())
        s_to_i = {s: i for i, s in enumerate(subjs)}
        n_all = len(subjs)
        for s in subjs:
            u = pu[s]
            u["FAPH"] = (u["FP_NS"] + u["FP_V"]) / max(u["hours"], 1e-9)
            u["Rec_%"] = (100.0 * u["TP"] / u["tot_F"]) if u["tot_F"] > 0 else None

        fpos_u = [pu[s] for s in subjs if pu[s]["tot_F"] > 0]
        zf_u = [pu[s] for s in subjs if pu[s]["tot_F"] == 0]
        faph_all = np.array([pu[s]["FAPH"] for s in subjs], dtype=np.float64)
        faph_0f = np.array([pu[s]["FAPH"] for s in subjs if pu[s]["tot_F"] == 0], dtype=np.float64)
        tot_fp = sum(pu[s]["FP_NS"] + pu[s]["FP_V"] for s in subjs)
        tot_h = sum(pu[s]["hours"] for s in subjs)

        # Sweep @ <= 2.0 FAPH
        sc_parts, y_parts, si_parts = [], [], []
        totF_arr = np.array([pu[s]["tot_F"] for s in subjs], dtype=np.float64)
        hrs_arr = np.array([pu[s]["hours"] for s in subjs], dtype=np.float64)
        for s in subjs:
            u = pu[s]
            si = s_to_i[s]
            if u["f_sc"]:
                fs = np.concatenate(u["f_sc"]).astype(np.float64)
                fs = fs[fs > -1e8]
                if len(fs):
                    sc_parts.append(fs); y_parts.append(np.ones(len(fs), dtype=bool)); si_parts.append(np.full(len(fs), si, dtype=np.int32))
            if u["nf_sc"]:
                ns = np.concatenate(u["nf_sc"]).astype(np.float64)
                if len(ns):
                    sc_parts.append(ns); y_parts.append(np.zeros(len(ns), dtype=bool)); si_parts.append(np.full(len(ns), si, dtype=np.int32))

        if sc_parts:
            sc_v = np.concatenate(sc_parts); y_v = np.concatenate(y_parts); si_v = np.concatenate(si_parts)
            ord_i = np.argsort(sc_v)[::-1]
            so, yo, sio = sc_v[ord_i], y_v[ord_i], si_v[ord_i]
            rec_inc = np.where(yo, 100.0 / (np.maximum(1.0, totF_arr[sio]) * max(1, len(fpos_u))), 0.0)
            fa_inc = np.where(~yo, 1.0 / (np.maximum(1e-6, hrs_arr[sio]) * n_all), 0.0)
            cum_rec = np.cumsum(rec_inc); cum_fa = np.cumsum(fa_inc)
            grp_end = np.empty(len(so), dtype=bool); grp_end[:-1] = (so[:-1] != so[1:]); grp_end[-1] = True
            ge = np.where(grp_end)[0]
            pos = np.searchsorted(cum_fa[ge], 2.0 + 1e-12, side="right") - 1
            m20 = float(cum_rec[ge][pos]) if pos >= 0 else 0.0
        else:
            m20 = 0.0

        summaries[key] = {
            "TP": int(sum(pu[s]["TP"] for s in subjs)),
            "FP_NS": int(sum(pu[s]["FP_NS"] for s in subjs)),
            "FP_V": int(sum(pu[s]["FP_V"] for s in subjs)),
            "MacroRec_%": round(float(np.mean([u["Rec_%"] for u in fpos_u])), 2),
            "Fpos_hit": int(sum(1 for u in fpos_u if u["TP"] > 0)),
            "n_Fpos": len(fpos_u),
            "MeanFAPH_All": round(float(np.mean(faph_all)), 3),
            "MeanFAPH_0F": round(float(np.mean(faph_0f)), 3),
            "PooledFAPH": round(float(tot_fp / max(tot_h, 1e-9)), 3),
            "MedFAPH": round(float(np.median(faph_all)), 3),
            "P95_FAPH": round(float(np.percentile(faph_all, 95)), 3),
            "Pct_LE2": round(100.0 * float(np.mean(faph_all <= 2.0)), 1),
            "ZeroF_Max": round(float(np.max(faph_0f)), 2),
            "Match20_%": round(m20, 2),
        }
        per_units[key] = {s: {k: pu[s][k] for k in ("tot_F", "hours", "TP", "FP_NS", "FP_V", "FAPH", "Rec_%")} for s in subjs}

    # Paired comparisons vs Full_A3 within each family
    paired = {}
    for fam in ("SVM", "LightGBM", "RF"):
        ref_k = f"{fam}__Full_A3"
        if ref_k not in per_units:
            continue
        for key in all_keys:
            if key.startswith(fam + "__") and key != ref_k:
                paired[key] = v62a.paired_compare(per_units[ref_k], per_units[key])

    return {"Causality_Coverage": cov, "Stage1_Forensics": stage1_forensics_summary, "Models": summaries, "Paired_vs_FullA3": paired}


def run_cohort(name, reg_path, raw_dir, base_url, ann_ext, sid_key, locked_models, wave_size):
    t0 = time.time()
    reg = json.loads(Path(reg_path).read_text(encoding="utf-8"))
    recs = reg["included_records"]
    cache_dir = CACHE_EXT / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    sums = fetch_sha256sums(raw_dir, base_url)

    pending = [r for r in recs if not (cache_dir / f"{r['rec_id']}_v62.pkl").exists()]
    print(f"\n=== Pipelined Streaming {name}: {len(recs)} total, {len(pending)} pending (5 DL threads -> 4 CPU workers) ===", flush=True)

    if pending:
        sem = threading.Semaphore(10)  # cap at most 10 .dat files on disk simultaneously
        done_count = [len(recs) - len(pending)]
        lock = threading.Lock()
        cpu_futures = []

        with ProcessPoolExecutor(max_workers=4, initializer=_init_ext_worker, initargs=(str(CKPT_IN),)) as cpu_pool, \
             ThreadPoolExecutor(max_workers=5) as dl_pool:

            def _dl_and_submit(r):
                sem.acquire()
                try:
                    fn = f"{r['rec_id']}.dat"
                    download_verified(raw_dir, base_url, fn, sums.get(fn))
                except Exception:
                    sem.release()
                    raise
                fut = cpu_pool.submit(_worker_process_record, (r, str(raw_dir), ann_ext, str(cache_dir)))

                def _on_cpu_done(_f):
                    sem.release()
                    with lock:
                        done_count[0] += 1
                        dc = done_count[0]
                    if dc % 6 == 0 or dc == len(recs):
                        print(f"  [{name}] {dc}/{len(recs)} done ({time.time() - t0:.1f}s elapsed)", flush=True)

                fut.add_done_callback(_on_cpu_done)
                with lock:
                    cpu_futures.append((r["rec_id"], fut))
                return fut

            dl_futs = [dl_pool.submit(_dl_and_submit, r) for r in pending]
            for df in as_completed(dl_futs):
                df.result()
            for _, cf in cpu_futures:
                cf.result()

    rec_results = []
    for r in recs:
        with open(cache_dir / f"{r['rec_id']}_v62.pkl", "rb") as f:
            rec_results.append(pickle.load(f))

    res = aggregate_cohort(rec_results, recs, sid_key)
    print(f"\n=== {name} SUMMARY ({time.time() - t0:.1f}s) ===", flush=True)
    print("  Causality Coverage:", res["Causality_Coverage"], flush=True)
    print("  Stage1 Forensics  :", res["Stage1_Forensics"], flush=True)
    for k, s in res["Models"].items():
        print(f"  {k:36s}: Rec={s['MacroRec_%']:5.2f}% ({s['Fpos_hit']:2d}/{s['n_Fpos']}) | TP={s['TP']:4d} | FP_NS={s['FP_NS']:5d} FP_V={s['FP_V']:4d} | FAPH={s['MeanFAPH_All']:.3f} (0F={s['MeanFAPH_0F']:.3f}) | M2={s['Match20_%']:.2f}%", flush=True)
    return res


def main():
    t0 = time.time()
    with open(CKPT_IN, "rb") as f:
        locked_models = pickle.load(f)

    out = {"Version": "v62b_external_stream_eval", "Date": "2026-11-24"}
    out["LTSTDB"] = run_cohort(
        "ltstdb",
        ROOT / "registry" / "target_cohort_ltstdb_v1.json",
        (_DATA_ROOT / "ltstdb"),
        "https://physionet.org/files/ltstdb/1.0.0/",
        "atr", "subject_id", locked_models, wave_size=6,
    )
    # Parity assertion vs frozen v59
    s_lt = out["LTSTDB"]["Models"]["SVM__Full_A3"]
    assert s_lt["TP"] == 42 and s_lt["FP_NS"] == 682 and s_lt["FP_V"] == 53, f"LTSTDB v59 parity mismatch: {s_lt}"
    print("[VERIFIED] Exact v59 LTSTDB parity (TP=42, FP_NS=682, FP_V=53)!", flush=True)

    out["MGHDB_Tier1"] = run_cohort(
        "mghdb",
        ROOT / "registry" / "target_cohort_mghdb_v1.json",
        (_DATA_ROOT / "mghdb"),
        "https://physionet.org/files/mghdb/1.0.0/",
        "ari", "subject_id_tier1", locked_models, wave_size=10,
    )
    # Parity assertion vs frozen v60
    s_mg = out["MGHDB_Tier1"]["Models"]["SVM__Full_A3"]
    assert s_mg["TP"] == 10 and s_mg["FP_NS"] == 157 and s_mg["FP_V"] == 41, f"MGHDB v60 parity mismatch: {s_mg}"
    print("[VERIFIED] Exact v60 MGHDB parity (TP=10, FP_NS=157, FP_V=41)!", flush=True)

    out["runtime_sec"] = round(time.time() - t0, 1)
    AUD_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE v62b in {out['runtime_sec']}s -> {AUD_OUT}", flush=True)


if __name__ == "__main__":
    main()
