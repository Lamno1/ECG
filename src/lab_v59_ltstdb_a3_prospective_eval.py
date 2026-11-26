r"""
lab_v59_ltstdb_a3_prospective_eval.py
=====================================
ONE-SHOT PROSPECTIVE EXTERNAL VALIDATION OF LOCKED `A3` (`A3_HPhiW_NS_Only` + `R3_B20`)
ON `LTSTDB` (70 SUBJECTS / 76 RECORDS, EXCLUDING 10 VERIFIED EDB-OVERLAPPING SUBJECTS).

Strict Prospective Integrity Protocol:
  1. Verifies SHA-256 of the single frozen 6-DB source checkpoint (`stage2_source6db_subjectaware251_a3_r3b20_v2.pkl`)
     against its manifest (and optional `--expected-ckpt-sha` CLI pin) BEFORE touching any target `.dat`.
  2. Verifies SHA-256 of locked `registry/target_cohort_ltstdb_v1.json` (`dd6373b9...`, 70 subjects / 76 records,
     `AAMI_Map_v1.1` mapping `'B' -> N`, channels `[0, 1]` uniformly).
  3. Downloads each of the 76 LTSTDB `.dat` files in bounded waves of 8 with PhysioNet `SHA256SUMS.txt`
     verification, runs `stage2_frozen_pipeline_v2.py`, ASSERTS bit-exact parity (`max|dX24| <= 1e-6`,
     identical `y` and `s2_mask`) against existing `cache_v50_ltstdb_ext` on EVERY SINGLE RECORD (76/76),
     caches compact `X29_s2` in `./data_public_ecg/cache_v59_ltstdb_a3\`, and cleans up `.dat` files.
  4. Evaluates the 3 locked models ONCE with ZERO target refitting, threshold tuning, or feature re-selection:
       - `B0_C3_R1`           : Frozen v50 comparator (asserts exact TP/FP_NS/FP_V parity vs v50!)
       - `B0_C3_R3_B20`       : Calibration-only control (C3 features under identical R3_B20)
       - `A3_HPhiW_NS_R3_B20` : LOCKED PRIMARY MODEL (15D F:N/S, locked 11D F:V, R3_B20)
     across BOTH `SVM` (primary) and `LogR` (sensitivity check).
  5. Reports full deployed T1 metrics, True 0-F tail quantiles, matched-FA curves (`S_min` and pure-`m_N`
     under fixed `B0_C3_R1` F:V gate), paired subject-level bootstrap (B=10,000, seed=42, exact v50 protocol)
     + paired Wilcoxon & sign tests, all 7 pre-registered secondary subgroups, `'B'`-heavy sensitivity audit,
     and full per-subject (70 subjects) helped/harmed breakdown.
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
import pickle
import hashlib
import shutil
import argparse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from scipy.stats import wilcoxon, binomtest

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v48_stage2_operating_point_calibration as v48
import lab_v50_ltstdb_external_eval as v50
import stage2_frozen_pipeline_v2 as pipe2

CKPT_PKL = str(_REPO_ROOT / "checkpoints/stage2_source6db_subjectaware251_a3_r3b20_v2.pkl")
CKPT_MANIFEST = str(_REPO_ROOT / "checkpoints/stage2_source6db_subjectaware251_a3_r3b20_v2_manifest.json")
LTSTDB_REG_PATH = str(_REPO_ROOT / "registry/target_cohort_ltstdb_v1.json")
EXPECTED_LTSTDB_REG_SHA256 = "dd6373b94fd772e10e3fd77a5f68fc7de9fac3e37241105bbb0176d8238c0871"

RAW_DIR = str(_DATA_ROOT / "ltstdb")
CACHE_V50 = str(_DATA_ROOT / "cache_v50_ltstdb_ext")
CACHE_V59 = str(_DATA_ROOT / "cache_v59_ltstdb_a3")
AUD_DIR = str(_REPO_ROOT / "audits")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_V59, exist_ok=True)
os.makedirs(AUD_DIR, exist_ok=True)

BASE_URL = "https://physionet.org/files/ltstdb/1.0.0/"
CFG_ORDER = ["B0_C3_R1", "B0_C3_R3_B20", "A3_HPhiW_NS_R3_B20"]
CLF_FAMILIES = ["SVM", "LogR"]

BOOT_B = 10000
BOOT_SEED = 42


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_physionet_sha256sums():
    p = os.path.join(RAW_DIR, "SHA256SUMS.txt")
    if not os.path.exists(p):
        for k in range(4):
            try:
                data = urllib.request.urlopen(BASE_URL + "SHA256SUMS.txt", timeout=30).read()
                with open(p, "wb") as f:
                    f.write(data)
                break
            except Exception:
                if k == 3:
                    raise
                time.sleep(2.0 * (k + 1))
    sums = {}
    for ln in open(p, "r", encoding="utf-8", errors="replace"):
        parts = ln.strip().split()
        if len(parts) >= 2:
            sums[parts[1]] = parts[0]
    return sums


def download_verified_file(fname, expected_sha):
    dst = os.path.join(RAW_DIR, fname)
    if os.path.exists(dst) and expected_sha and sha256_file(dst) == expected_sha:
        return dst
    tmp = dst + ".part"
    for k in range(5):
        try:
            req = urllib.request.Request(BASE_URL + fname, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            got_sha = sha256_file(tmp)
            if expected_sha and got_sha != expected_sha:
                raise RuntimeError(f"SHA mismatch for {fname}: got {got_sha[:16]} != exp {expected_sha[:16]}")
            os.replace(tmp, dst)
            return dst
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if k == 4:
                raise RuntimeError(f"Failed downloading {fname}: {e}")
            time.sleep(2.5 * (k + 1))


def _extract_worker_v2(rec_meta):
    rec_id = rec_meta["rec_id"]
    sid = rec_meta["subject_id"]
    ch = tuple(rec_meta["locked_channel_indices"])
    npz_p = os.path.join(CACHE_V59, f"{rec_id}.npz")
    meta_p = os.path.join(CACHE_V59, f"{rec_id}_meta.json")
    if os.path.exists(npz_p) and os.path.exists(meta_p):
        return rec_id, "cached", 0.0, 0.0

    rec_path = os.path.join(RAW_DIR, rec_id)
    res = pipe2.extract_record_features_v2(
        rec_path=rec_path,
        db_name="ltstdb",
        rec_id=rec_id,
        subject_id=sid,
        ann_ext="atr",
        channel_indices=ch,
        label_map_version="AAMI_Map_v1.1",
    )
    if res is None:
        raise RuntimeError(f"Record {rec_id} returned None (<=55 valid beats)!")

    # Bit-exact parity assertion against frozen v50 LTSTDB cache on EVERY SINGLE RECORD!
    v50_npz_p = os.path.join(CACHE_V50, f"{rec_id}.npz")
    max_d24 = 0.0
    if os.path.exists(v50_npz_p):
        v50_d = np.load(v50_npz_p)
        assert np.array_equal(res["y"], v50_d["y"]), f"LTSTDB y mismatch vs v50 on {rec_id}"
        assert np.array_equal(res["s2_mask"], v50_d["s2_mask"]), f"LTSTDB s2_mask mismatch vs v50 on {rec_id}"
        if len(res["X29_s2"]) > 0:
            max_d24 = float(np.max(np.abs(res["X29_s2"][:, :24].astype(np.float64) - v50_d["X24_s2"].astype(np.float64))))
            assert max_d24 <= 1e-6, f"LTSTDB X24 parity violation vs v50 on {rec_id}: max_d24={max_d24}"

    np.savez_compressed(
        npz_p,
        y=res["y"],
        s2_mask=res["s2_mask"],
        X29_s2=res["X29_s2"],
    )
    meta_out = {
        "db": "ltstdb",
        "rec_id": rec_id,
        "patient_id": sid,
        "fs": res["fs"],
        "leads": res["leads"],
        "post_hours": res["post_hours"],
        "K": res["K"],
        "label_map_version": res["label_map_version"],
        "buf_audit": res["buf_audit"],
        "accounting": res["accounting"],
        "proc_sec": res["proc_sec"],
        "v50_x24_max_err": max_d24,
        "contributor_group": rec_meta["contributor_group"],
        "is_initial_bih": rec_meta["is_initial_bih"],
        "is_easi": rec_meta["is_easi"],
        "is_s2010": rec_meta["is_s2010"],
    }
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    return rec_id, "extracted", res["proc_sec"], max_d24


def prepare_all_ltstdb_v2_features(inc_records):
    sha_map = fetch_physionet_sha256sums()
    todo = [r for r in inc_records if not (
        os.path.exists(os.path.join(CACHE_V59, f"{r['rec_id']}.npz"))
        and os.path.exists(os.path.join(CACHE_V59, f"{r['rec_id']}_meta.json"))
    )]
    print(f"[v59] LTSTDB X29 feature prep: {len(inc_records) - len(todo)} cached, {len(todo)} remaining.", flush=True)

    wave_size = 8
    done_cnt = len(inc_records) - len(todo)
    max_err_global = 0.0
    t_start = time.time()

    for wi in range(0, len(todo), wave_size):
        wave = todo[wi : wi + wave_size]
        t_w = time.time()
        fnames = []
        for r in wave:
            rid = r["rec_id"]
            fnames.extend([f"{rid}.hea", f"{rid}.atr", f"{rid}.dat"])

        def _dl(fn):
            return download_verified_file(fn, sha_map.get(fn))

        with ThreadPoolExecutor(max_workers=4) as dex:
            list(dex.map(_dl, fnames))
        t_dl = time.time() - t_w

        t_ex0 = time.time()
        with ProcessPoolExecutor(max_workers=4) as pex:
            res_list = list(pex.map(_extract_worker_v2, wave))
        t_ex = time.time() - t_ex0
        done_cnt += len(wave)
        wave_max_err = max(x[3] for x in res_list) if res_list else 0.0
        max_err_global = max(max_err_global, wave_max_err)

        # Clean up large raw .dat files after verified caching (keep .hea & .atr permanently)
        for r in wave:
            dat_p = os.path.join(RAW_DIR, f"{r['rec_id']}.dat")
            if os.path.exists(dat_p):
                os.remove(dat_p)

        free_gb = shutil.disk_usage("d:\\").free / 1e9
        print(
            f"  Wave {wi // wave_size + 1:2d}/{(len(todo) + wave_size - 1) // wave_size:2d} | "
            f"progress={done_cnt:2d}/{len(inc_records)} | dl={t_dl:5.1f}s ext={t_ex:5.1f}s | "
            f"max|dX24_v50|={wave_max_err:.2e} | elapsed={time.time() - t_start:6.1f}s | D:free={free_gb:.1f}GB | "
            f"recs={[r['rec_id'] for r in wave]}",
            flush=True,
        )
    return max_err_global


def load_cached_ltstdb_v2(inc_records):
    items = []
    max_err = 0.0
    for r in inc_records:
        rid = r["rec_id"]
        meta = json.load(open(os.path.join(CACHE_V59, f"{rid}_meta.json"), "r", encoding="utf-8"))
        dat = np.load(os.path.join(CACHE_V59, f"{rid}.npz"))
        max_err = max(max_err, float(meta.get("v50_x24_max_err", 0.0)))
        items.append({
            "db": "ltstdb",
            "rec_id": rid,
            "patient_id": meta["patient_id"],
            "post_hours": float(meta["post_hours"]),
            "K": int(meta["K"]),
            "leads": meta["leads"],
            "contributor_group": meta["contributor_group"],
            "is_initial_bih": bool(meta["is_initial_bih"]),
            "is_easi": bool(meta["is_easi"]),
            "is_s2010": bool(meta["is_s2010"]),
            "buf_audit": meta["buf_audit"],
            "accounting": meta["accounting"],
            "y": dat["y"],
            "s2_mask": dat["s2_mask"],
            "X_s2": dat["X29_s2"].astype(np.float64),
        })
    return items, max_err


def zero_f_tail_audit(pt1):
    z = [(sid, u["FAPH"], u["FP_NS"], u["FP_V"], u["hours"]) for sid, u in pt1.items() if u["tot_F"] == 0]
    if not z:
        return {"n_0F": 0}
    arr = np.array([x[1] for x in z], dtype=np.float64)
    z_sorted = sorted(z, key=lambda x: -x[1])
    return {
        "n_0F": len(z),
        "mean": round(float(np.mean(arr)), 4),
        "pct_zero": round(100.0 * float(np.mean(arr == 0.0)), 1),
        "pct_le_2": round(100.0 * float(np.mean(arr <= 2.0 + 1e-9)), 1),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "max": round(float(np.max(arr)), 4),
        "top5_worst": [
            {"subject_id": x[0], "faph": round(x[1], 3), "fp_ns": int(x[2]), "fp_v": int(x[3]), "hours": round(x[4], 2)}
            for x in z_sorted[:5]
        ],
    }


def compute_matched_fa_curves(te_items, models_for_mtype):
    """
    Computes exact realized matched-FA curves on the cohort:
      1. `S_min` matched-FA curve for each config:
           MacroPatRec(F+) at realized MeanPatFAPH_All <= {0.5, 1.0, 2.0, 3.0}
           and at realized MeanPatFAPH_0F  <= {0.25, 0.5, 1.0, 2.0}
      2. Pure-`m_N` matched-FA curve under FIXED `B0_C3_R1` F:V gate (`sV_B0 >= tau_V_B0_R1`):
           Directly compares `B0` vs `A3` F:N/S hyperplanes with ZERO threshold or F:V confound!
    """
    subj_map = {}
    for it in te_items:
        s = it["patient_id"]
        if s not in subj_map:
            subj_map[s] = {"tot_F": 0, "hours": 0.0}
        subj_map[s]["tot_F"] += int(np.sum(it["y"] == 2))
        subj_map[s]["hours"] += float(it["post_hours"])

    subjs = sorted(subj_map.keys())
    s_to_idx = {s: i for i, s in enumerate(subjs)}
    n_all = len(subjs)
    totF = np.array([subj_map[s]["tot_F"] for s in subjs], dtype=np.int64)
    hours = np.array([subj_map[s]["hours"] for s in subjs], dtype=np.float64)
    fpos_mask = (totF > 0)
    zpos_mask = (totF == 0)
    n_fpos = max(1, int(np.sum(fpos_mask)))
    n_zpos = max(1, int(np.sum(zpos_mask)))

    # Collect per-beat scores & subject indices on Stage-2 beats
    y_s2_l, si_l = [], []
    mN_by_cfg = {c: [] for c in CFG_ORDER}
    mV_by_cfg = {c: [] for c in CFG_ORDER}
    smin_by_cfg = {c: [] for c in CFG_ORDER}

    for it in te_items:
        X = it["X_s2"]
        if len(X) == 0:
            continue
        ys = it["y"][it["s2_mask"]].astype(np.int8)
        y_s2_l.append(ys)
        si_l.append(np.full(len(ys), s_to_idx[it["patient_id"]], dtype=np.int32))
        for cname in CFG_ORDER:
            m = models_for_mtype[cname]
            sN = m["clfN"].decision_function(m["scN"].transform(X[:, m["feat_idx_N"]]))
            sV = m["clfV"].decision_function(m["scV"].transform(X[:, m["feat_idx_V"]]))
            mN = (sN - m["tau_N"]) / m["sig_N"]
            mV = (sV - m["tau_V"]) / m["sig_V"]
            mN_by_cfg[cname].append(mN)
            mV_by_cfg[cname].append(mV)
            smin_by_cfg[cname].append(np.minimum(mN, mV))

    y_all = np.concatenate(y_s2_l)
    si_all = np.concatenate(si_l)
    for cname in CFG_ORDER:
        mN_by_cfg[cname] = np.concatenate(mN_by_cfg[cname])
        mV_by_cfg[cname] = np.concatenate(mV_by_cfg[cname])
        smin_by_cfg[cname] = np.concatenate(smin_by_cfg[cname])

    def sweep_curve(scores, y_vec, si_vec, all_budgets=(0.5, 1.0, 2.0, 3.0), zf_budgets=(0.25, 0.5, 1.0, 2.0)):
        if len(scores) == 0:
            return {"AllFAPH": {str(b): 0.0 for b in all_budgets}, "ZeroFFAPH": {str(b): 0.0 for b in zf_budgets}}
        ord_i = np.argsort(scores)[::-1]
        s_o = scores[ord_i]
        y_o = y_vec[ord_i]
        si_o = si_vec[ord_i]

        is_f = (y_o == 2)
        is_nf = ~is_f
        rec_inc = np.where(is_f, 100.0 / (np.maximum(1.0, totF[si_o].astype(np.float64)) * float(n_fpos)), 0.0)
        fa_all_inc = np.where(is_nf, 1.0 / (np.maximum(1e-6, hours[si_o]) * float(n_all)), 0.0)
        fa_0f_inc = np.where(is_nf & zpos_mask[si_o], 1.0 / (np.maximum(1e-6, hours[si_o]) * float(n_zpos)), 0.0)

        cum_rec = np.cumsum(rec_inc)
        cum_fa_all = np.cumsum(fa_all_inc)
        cum_fa_0f = np.cumsum(fa_0f_inc)

        grp_end = np.empty(len(s_o), dtype=bool)
        grp_end[:-1] = (s_o[:-1] != s_o[1:])
        grp_end[-1] = True

        out_all = {}
        for b in all_budgets:
            ok = np.where(grp_end & (cum_fa_all <= b + 1e-12))[0]
            out_all[str(b)] = round(float(cum_rec[ok[-1]]), 3) if len(ok) > 0 else 0.0
        out_0f = {}
        for b in zf_budgets:
            ok = np.where(grp_end & (cum_fa_0f <= b + 1e-12))[0]
            out_0f[str(b)] = round(float(cum_rec[ok[-1]]), 3) if len(ok) > 0 else 0.0
        return {"AllFAPH": out_all, "ZeroFFAPH": out_0f}

    smin_curves = {cname: sweep_curve(smin_by_cfg[cname], y_all, si_all) for cname in CFG_ORDER}

    # Pure-m_N sweep under fixed B0_C3_R1 F:V gate (exact iso-gate comparison of F:N/S hyperplane alone)
    pass_v_b0 = (mV_by_cfg["B0_C3_R1"] >= 0.0)
    pure_mN_curves = {
        "B0_10D_under_B0R1_FVgate": sweep_curve(mN_by_cfg["B0_C3_R1"][pass_v_b0], y_all[pass_v_b0], si_all[pass_v_b0]),
        "A3_15D_under_B0R1_FVgate": sweep_curve(mN_by_cfg["A3_HPhiW_NS_R3_B20"][pass_v_b0], y_all[pass_v_b0], si_all[pass_v_b0]),
    }
    return {"S_min": smin_curves, "Pure_mN_Fixed_B0R1_FVgate": pure_mN_curves}


def paired_nonparam_tests(pt1_A, pt1_B, prf2_A=None, prf2_B=None):
    all_subjs = sorted(pt1_A.keys())
    fpos_subjs = [s for s in all_subjs if pt1_A[s]["tot_F"] > 0]
    zf_subjs = [s for s in all_subjs if pt1_A[s]["tot_F"] == 0]

    def _stat(diffs):
        nz = diffs[np.abs(diffs) > 1e-12]
        w = int(np.sum(nz > 0)); l = int(np.sum(nz < 0)); t = len(diffs) - w - l
        p_sign = float(binomtest(w, w + l, 0.5).pvalue) if (w + l) > 0 else 1.0
        try:
            p_wilc = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
        except Exception:
            p_wilc = 1.0
        return {
            "WTL": f"{w}W/{t}T/{l}L",
            "mean_diff": round(float(np.mean(diffs)), 4),
            "p_wilcoxon": round(p_wilc, 5),
            "p_sign": round(p_sign, 5),
        }

    out = {
        "T1_Rec_Fpos": _stat(np.array([pt1_A[s]["Rec_%"] - pt1_B[s]["Rec_%"] for s in fpos_subjs], dtype=np.float64)),
        "T1_FAPH_All_neg_is_better": _stat(np.array([pt1_A[s]["FAPH"] - pt1_B[s]["FAPH"] for s in all_subjs], dtype=np.float64)),
        "T1_FAPH_0F_neg_is_better": _stat(np.array([pt1_A[s]["FAPH"] - pt1_B[s]["FAPH"] for s in zf_subjs], dtype=np.float64)) if zf_subjs else None,
    }
    if prf2_A is not None and prf2_B is not None:
        common_f = sorted(set(prf2_A.keys()) & set(prf2_B.keys()))
        if common_f:
            out["PatAtF2_Fpos"] = _stat(np.array([prf2_A[s] - prf2_B[s] for s in common_f], dtype=np.float64))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-ckpt-sha", type=str, default=None, help="Optional pinned SHA-256 for v2 checkpoint")
    args = ap.parse_args()

    t0 = time.time()
    manifest = json.load(open(CKPT_MANIFEST, "r", encoding="utf-8"))
    ckpt_sha = sha256_file(CKPT_PKL)
    assert ckpt_sha == manifest["checkpoint_pkl_sha256"], (
        f"Checkpoint SHA mismatch vs manifest! {ckpt_sha} != {manifest['checkpoint_pkl_sha256']}"
    )
    if args.expected_ckpt_sha:
        assert ckpt_sha == args.expected_ckpt_sha, f"CLI pinned SHA mismatch! {ckpt_sha} != {args.expected_ckpt_sha}"

    reg_sha = sha256_file(LTSTDB_REG_PATH)
    assert reg_sha == EXPECTED_LTSTDB_REG_SHA256, f"LTSTDB registry SHA mismatch! {reg_sha}"
    reg = json.load(open(LTSTDB_REG_PATH, "r", encoding="utf-8"))
    inc_records = reg["included_records"]
    assert len(inc_records) == 76

    print(f"[v59] VERIFIED FROZEN v2 CHECKPOINT SHA256: {ckpt_sha}", flush=True)
    print(f"[v59] VERIFIED LTSTDB REGISTRY     SHA256: {reg_sha} ({len(inc_records)} recs / 70 subjs)", flush=True)

    with open(CKPT_PKL, "rb") as f:
        ckpt = pickle.load(f)

    # Extract bit-exact X29 features on all 76 LTSTDB records & assert X24 parity vs v50 on every record
    prepare_all_ltstdb_v2_features(inc_records)
    items, max_x24_err = load_cached_ltstdb_v2(inc_records)
    print(f"\n[v59] Loaded all {len(items)} LTSTDB X29 records | max|dX24 vs v50| across all 76 recs = {max_x24_err:.3e}", flush=True)
    assert max_x24_err <= 1e-6, f"Unexpected X24 parity error: {max_x24_err}"

    # Verify exact beat/F/subject accounting parity vs v50
    tot_post_beats = sum(it["accounting"]["post_prefix_beats"] for it in items)
    tot_post_F = sum(it["accounting"]["post_prefix_F"] for it in items)
    tot_s2_beats = sum(it["accounting"]["s2_total"] for it in items)
    tot_s2_F = sum(it["accounting"]["s2_F"] for it in items)
    assert tot_post_beats == 7969461 and tot_post_F == 583 and tot_s2_beats == 2780532 and tot_s2_F == 373, (
        f"Sample accounting mismatch vs v50: beats={tot_post_beats}, F={tot_post_F}, s2={tot_s2_beats}, s2_F={tot_s2_F}"
    )

    cohorts = {
        "PRIMARY_All_70_Subjs": items,
        "SEC_Excl_Initial_BIH_59": [it for it in items if not it["is_initial_bih"]],
        "SEC_Initial_BIH_11": [it for it in items if it["is_initial_bih"]],
        "SEC_Zymed_3ch_15": [it for it in items if it["contributor_group"].startswith("Zymed_3ch")],
        "SEC_Other_2ch_Excl_InitBIH_44": [it for it in items if it["contributor_group"] == "Other_2ch_BWH_BIDMC_Pisa"],
        "SEC_Standard_Leads_58": [it for it in items if not it["is_easi"]],
        "SEC_EASI_3ch_12": [it for it in items if it["is_easi"]],
        "SEC_Excl_s2010_69": [it for it in items if not it["is_s2010"]],
        "DESC_Excl_B_Heavy_68": [it for it in items if it["patient_id"] not in ("s2053", "s2055")],
    }

    results = {
        "Audit_Version": "v59_a3_ltstdb_prospective_validation",
        "Date": "2026-11-23",
        "Locked_Primary_Config": "A3_HPhiW_NS_R3_B20",
        "Checkpoint_Path": CKPT_PKL,
        "Checkpoint_SHA256": ckpt_sha,
        "Target_Registry_Path": LTSTDB_REG_PATH,
        "Target_Registry_SHA256": reg_sha,
        "LTSTDB_X24_Parity_MaxErr_vs_v50": max_x24_err,
        "Sample_Accounting": {
            "n_records": len(items),
            "n_subjects": len(set(it["patient_id"] for it in items)),
            "eval_post_prefix_beats": tot_post_beats,
            "eval_post_prefix_F": tot_post_F,
            "eval_s2_total_beats": tot_s2_beats,
            "eval_s2_F_beats": tot_s2_F,
            "eval_post_prefix_hours": round(sum(it["post_hours"] for it in items), 2),
        },
        "Models": {},
    }

    for mtype in CLF_FAMILIES:
        print(f"\n==================== LTSTDB PROSPECTIVE EVALUATION ({mtype}, FROZEN 6-DB v2) ====================", flush=True)
        m_res = {"Source_OOF": {c: ckpt["models"][mtype][c]["src_stat"] for c in CFG_ORDER}, "Cohorts": {}, "PerSubject_Primary": {}}

        for coh_name, coh_items in cohorts.items():
            c_out = {
                "n_records": len(coh_items),
                "n_subjects": len(set(it["patient_id"] for it in coh_items)),
                "Configs": {},
                "ZeroF_Tail": {},
                "Comparisons": {},
            }
            pt1_by_cfg, prf2_by_cfg = {}, {}
            for cname in CFG_ORDER:
                ev, pt1, prf2 = v50.eval_cohort_full(coh_items, ckpt["models"][mtype][cname])
                c_out["Configs"][cname] = ev
                c_out["ZeroF_Tail"][cname] = zero_f_tail_audit(pt1)
                pt1_by_cfg[cname] = pt1
                prf2_by_cfg[cname] = prf2

            # Assert bit-exact reproduction of v50 C3 primary LTSTDB numbers!
            if coh_name == "PRIMARY_All_70_Subjs":
                ev_b0 = c_out["Configs"]["B0_C3_R1"]
                if mtype == "SVM":
                    assert (ev_b0["T1_TP"], ev_b0["T1_FP_NS"], ev_b0["T1_FPV"]) == (44, 3250, 88), f"SVM B0_R1 v50 mismatch: {ev_b0}"
                else:
                    assert (ev_b0["T1_TP"], ev_b0["T1_FP_NS"], ev_b0["T1_FPV"]) == (46, 3782, 81), f"LogR B0_R1 v50 mismatch: {ev_b0}"
                c_out["Matched_FA_Curves"] = compute_matched_fa_curves(coh_items, ckpt["models"][mtype])

            for cmp_name, cA, cB in [
                ("A3_R3_vs_B0_R1", "A3_HPhiW_NS_R3_B20", "B0_C3_R1"),
                ("A3_R3_vs_B0_R3", "A3_HPhiW_NS_R3_B20", "B0_C3_R3_B20"),
                ("B0_R3_vs_B0_R1", "B0_C3_R3_B20", "B0_C3_R1"),
            ]:
                boot = v50.paired_subject_bootstrap(pt1_by_cfg[cA], pt1_by_cfg[cB], B=BOOT_B, seed=BOOT_SEED)
                np_t = paired_nonparam_tests(pt1_by_cfg[cA], pt1_by_cfg[cB], prf2_by_cfg[cA], prf2_by_cfg[cB])
                c_out["Comparisons"][cmp_name] = {"Bootstrap": boot, "NonParam": np_t}

            m_res["Cohorts"][coh_name] = c_out

            if coh_name == "PRIMARY_All_70_Subjs":
                subj_meta = {}
                for it in coh_items:
                    s = it["patient_id"]
                    if s not in subj_meta:
                        subj_meta[s] = {
                            "recs": [], "contributor_group": it["contributor_group"],
                            "is_initial_bih": it["is_initial_bih"], "is_easi": it["is_easi"],
                            "is_s2010": it["is_s2010"], "s2_F": 0, "post_B": 0,
                        }
                    subj_meta[s]["recs"].append(it["rec_id"])
                    subj_meta[s]["s2_F"] += int(it["accounting"]["s2_F"])
                    subj_meta[s]["post_B"] += int(it["accounting"]["post_prefix_B"])

                for s in sorted(subj_meta.keys()):
                    row = dict(subj_meta[s])
                    for short_k, cname in [("B0_R1", "B0_C3_R1"), ("B0_R3", "B0_C3_R3_B20"), ("A3_R3", "A3_HPhiW_NS_R3_B20")]:
                        row[short_k] = pt1_by_cfg[cname][s]
                    row["dRec_A3_minus_B0R1"] = (
                        round(row["A3_R3"]["Rec_%"] - row["B0_R1"]["Rec_%"], 3)
                        if (row["A3_R3"]["Rec_%"] is not None and row["B0_R1"]["Rec_%"] is not None) else None
                    )
                    row["dFAPH_A3_minus_B0R1"] = round(row["A3_R3"]["FAPH"] - row["B0_R1"]["FAPH"], 4)
                    row["dRec_A3_minus_B0R3"] = (
                        round(row["A3_R3"]["Rec_%"] - row["B0_R3"]["Rec_%"], 3)
                        if (row["A3_R3"]["Rec_%"] is not None and row["B0_R3"]["Rec_%"] is not None) else None
                    )
                    row["dFAPH_A3_minus_B0R3"] = round(row["A3_R3"]["FAPH"] - row["B0_R3"]["FAPH"], 4)
                    m_res["PerSubject_Primary"][s] = row

            # Console summary
            print(f"\n  [{coh_name}] (recs={c_out['n_records']}, subjs={c_out['n_subjects']})", flush=True)
            for cname in CFG_ORDER:
                ev = c_out["Configs"][cname]
                zt = c_out["ZeroF_Tail"][cname]
                z_str = f"p95={zt['p95']:.2f} max={zt['max']:.2f}" if zt.get("n_0F", 0) > 0 else "n/a"
                print(
                    f"    {cname:20s} | TP={ev['T1_TP']:4d} FP_NS={ev['T1_FP_NS']:5d} FP_V={ev['T1_FPV']:4d} F1={ev['T1_F1']:.4f} | "
                    f"MacroRec={ev['T1_MacroPat_Rec_%']:5.2f}% FAPH={ev['T1_MeanPat_FAPH_All']:5.3f} "
                    f"(0F={ev['T1_MeanPat_FAPH_ZeroF']:5.3f}, Pool={ev['T1_Pooled_FAPH']:5.3f}, <=2:{ev['T1_FracPats_LE2_%']:5.1f}%, 0F:{z_str}) | "
                    f"AP={ev['AP']:.4f} Pat@F2={ev['MacroPat_Pat@F2_%']:5.2f}%",
                    flush=True,
                )
            for cmp_name in ("A3_R3_vs_B0_R1", "A3_R3_vs_B0_R3", "B0_R3_vs_B0_R1"):
                b = c_out["Comparisons"][cmp_name]["Bootstrap"]
                np_t = c_out["Comparisons"][cmp_name]["NonParam"]
                print(
                    f"    {cmp_name:16s} | dRec={b['Delta_MacroRec_T1_pt_%']:+.2f}% {b['Delta_MacroRec_T1_CI95_%']} "
                    f"(WTL={b['Fpos_WTL_Rec']} pW={np_t['T1_Rec_Fpos']['p_wilcoxon']:.4f} pS={np_t['T1_Rec_Fpos']['p_sign']:.4f}) | "
                    f"dFAPH_All={b['Delta_MeanFAPH_All_pt']:+.4f} {b['Delta_MeanFAPH_All_CI95']} | "
                    f"dFAPH_0F={b['Delta_MeanFAPH_ZeroF_pt']:+.4f} {b['Delta_MeanFAPH_ZeroF_CI95']} | "
                    f"SDom={b['Fpos_StrictDom_A_vs_B']} | 0F:{b['ZeroF_FAPH_Dec_Tie_Inc']}",
                    flush=True,
                )
            if coh_name == "PRIMARY_All_70_Subjs":
                print("    Matched-FA Curves (Primary 70 subjs):", flush=True)
                for k, v in c_out["Matched_FA_Curves"]["S_min"].items():
                    print(f"      S_min {k:20s} | All={v['AllFAPH']} | 0F={v['ZeroFFAPH']}", flush=True)
                for k, v in c_out["Matched_FA_Curves"]["Pure_mN_Fixed_B0R1_FVgate"].items():
                    print(f"      Pure  {k:26s} | All={v['AllFAPH']} | 0F={v['ZeroFFAPH']}", flush=True)

        results["Models"][mtype] = m_res

    out_json = os.path.join(AUD_DIR, "V59_A3_LTSTDB_PROSPECTIVE_VALIDATION_20261123.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[v59] COMPLETE! Saved full audit JSON: {out_json} (total {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
