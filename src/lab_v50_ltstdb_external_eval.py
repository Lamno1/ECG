r"""
lab_v50_ltstdb_external_eval.py
===============================
ONE-SHOT PRE-REGISTERED EXTERNAL VALIDATION ON LTSTDB (PRIMARY EXTERNAL COHORT)

Strictly enforces Stage2-ClaimContract-v1.0 + Addendum-LTSTDB (locked 2026-11-22):
  1. Verifies SHA-256 of the single frozen 6-DB source checkpoint BEFORE loading/downloading any target .dat.
  2. Verifies SHA-256 of locked registry/target_cohort_ltstdb_v1.json (70 subjects / 76 records after excluding
     all 10 verified EDB-overlapping subjects; AAMI_Map_v1.1 mapping 'B'->N; channels [0,1] uniformly).
  3. Downloads each of the 76 LTSTDB records (.hea, .atr, .dat) with PhysioNet SHA256SUMS verification,
     runs the bit-exact frozen feature extractor (`stage2_frozen_pipeline_v1.py`), and caches compact
     per-record feature arrays in `./data_public_ecg/cache_v50_ltstdb_ext\`.
  4. Re-confirms exact evaluated post-prefix beat/F/subject counts vs screening counts without altering rules.
  5. Evaluates frozen C1, C3, C5 (SVM primary + LogR sensitivity) ONCE with ZERO target refitting/tuning.
  6. Runs pre-registered subject-level paired bootstrap (B=10,000, seed=42) for primary C5 vs C3 on SVM:
       CONFIRMED iff CI95_low(Delta_MacroRec_T1) > 0 AND CI95_high(Delta_MeanFAPH_T1) <= 0.
  7. Reports all mandatory metrics, separated Spec_AllNonF vs Spec_ZeroFRecs, C5 causal buffer audit
     (including 'B'-heavy subjects s2053 & s2055), per-subject helped/harmed breakdown, and the three
     pre-registered secondary subgroups (contributor provenance / Excluding initial-BIH, Standard vs EASI,
     and Excluding s2010) strictly labeled secondary.
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
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v43_hypB2_representation_vs_veto as v43
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_diagnostic_v48_stage2_operating_point_calibration as v48
import stage2_frozen_pipeline_v1 as pipe

CKPT_PKL = str(_REPO_ROOT / "checkpoints/stage2_source6db_subjectaware251_r1_v1.pkl")
EXPECTED_CKPT_SHA256 = "a34ac3bc6ae84f53458a8124a27ca0445ad1df477ef85a7d5f7ac785790e9c5f"
LTSTDB_REG_PATH = str(_REPO_ROOT / "registry/target_cohort_ltstdb_v1.json")

RAW_DIR = str(_DATA_ROOT / "ltstdb")
CACHE_DIR = str(_DATA_ROOT / "cache_v50_ltstdb_ext")
AUD_DIR = str(_REPO_ROOT / "audits")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUD_DIR, exist_ok=True)

BASE_URL = "https://physionet.org/files/ltstdb/1.0.0/"
CFG_ORDER = v48.CFG_ORDER
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


def _extract_worker(rec_meta):
    rec_id = rec_meta["rec_id"]
    sid = rec_meta["subject_id"]
    ch = tuple(rec_meta["locked_channel_indices"])
    npz_p = os.path.join(CACHE_DIR, f"{rec_id}.npz")
    meta_p = os.path.join(CACHE_DIR, f"{rec_id}_meta.json")
    if os.path.exists(npz_p) and os.path.exists(meta_p):
        return rec_id, "cached", 0.0

    rec_path = os.path.join(RAW_DIR, rec_id)
    res = pipe.extract_record_features_v1(
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

    np.savez_compressed(
        npz_p,
        y=res["y"],
        s2_mask=res["s2_mask"],
        X24_s2=res["X24_s2"],
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
        "contributor_group": rec_meta["contributor_group"],
        "is_initial_bih": rec_meta["is_initial_bih"],
        "is_easi": rec_meta["is_easi"],
        "is_s2010": rec_meta["is_s2010"],
    }
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    return rec_id, "extracted", res["proc_sec"]


def prepare_all_ltstdb_features(inc_records):
    """Downloads verified files in bounded batches (keeping D: disk safe) and extracts cached features."""
    sha_map = fetch_physionet_sha256sums()
    todo = [r for r in inc_records if not (
        os.path.exists(os.path.join(CACHE_DIR, f"{r['rec_id']}.npz"))
        and os.path.exists(os.path.join(CACHE_DIR, f"{r['rec_id']}_meta.json"))
    )]
    print(f"[v50] LTSTDB feature prep: {len(inc_records) - len(todo)} cached, {len(todo)} remaining.", flush=True)

    # Process in waves of 8 records: download 8 (4 threads) -> extract 8 (4 processes) -> cleanup .dat if free < 12 GB
    wave_size = 8
    done_cnt = len(inc_records) - len(todo)
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
            res_list = list(pex.map(_extract_worker, wave))
        t_ex = time.time() - t_ex0
        done_cnt += len(wave)

        # Clean up large raw .dat files for this wave after verified caching (keep .hea & .atr permanently)
        # to keep D: drive comfortable while retaining exact cached features & full header/annotation provenance.
        for r in wave:
            dat_p = os.path.join(RAW_DIR, f"{r['rec_id']}.dat")
            if os.path.exists(dat_p):
                os.remove(dat_p)

        free_gb = shutil.disk_usage("d:\\").free / 1e9
        print(
            f"  Wave {wi // wave_size + 1:2d}/{(len(todo) + wave_size - 1) // wave_size:2d} | "
            f"progress={done_cnt:2d}/{len(inc_records)} | dl={t_dl:5.1f}s ext={t_ex:5.1f}s | "
            f"elapsed={time.time() - t_start:6.1f}s | D:free={free_gb:.1f}GB | "
            f"recs={[r['rec_id'] for r in wave]}",
            flush=True,
        )


def load_cached_ltstdb(inc_records):
    items = []
    for r in inc_records:
        rid = r["rec_id"]
        meta = json.load(open(os.path.join(CACHE_DIR, f"{rid}_meta.json"), "r", encoding="utf-8"))
        dat = np.load(os.path.join(CACHE_DIR, f"{rid}.npz"))
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
            "X_s2": dat["X24_s2"].astype(np.float64),
        })
    return items


def eval_cohort_full(te_items, model_obj):
    """Exact subject-level evaluation with separated Spec_AllNonF_% vs Spec_ZeroFRecs_%."""
    ev, pt1, prf2 = v48.eval_target_v48(te_items, model_obj)
    tot_nonf = sum(int(np.sum(it["y"] != 2)) for it in te_items)
    fp_all = ev["T1_FP_NS"] + ev["T1_FPV"]
    ev["T1_Spec_AllNonF_%"] = round(100.0 * (1.0 - fp_all / max(1, tot_nonf)), 4)
    ev["T1_Spec_ZeroFRecs_%"] = ev["T1_Spec_%"]
    return ev, pt1, prf2


def paired_subject_bootstrap(pt1_A, pt1_B, B=BOOT_B, seed=BOOT_SEED):
    """
    Pre-registered paired subject-level bootstrap (A minus B, e.g. C5 minus C3):
      - Delta_MacroRec_T1 : resampled across evaluated F+ subjects (equal subject weight)
      - Delta_MeanFAPH_T1 : resampled across ALL evaluated subjects (incl. 0-F subjects)
    Uses np.random.default_rng(seed) with 2.5th / 97.5th percentile 95% CIs.
    """
    all_subjs = sorted(pt1_A.keys())
    assert all_subjs == sorted(pt1_B.keys())
    fpos_subjs = [s for s in all_subjs if pt1_A[s]["tot_F"] > 0]

    rec_A = np.array([pt1_A[s]["Rec_%"] for s in fpos_subjs], dtype=np.float64)
    rec_B = np.array([pt1_B[s]["Rec_%"] for s in fpos_subjs], dtype=np.float64)
    d_rec = rec_A - rec_B

    fa_A = np.array([pt1_A[s]["FAPH"] for s in all_subjs], dtype=np.float64)
    fa_B = np.array([pt1_B[s]["FAPH"] for s in all_subjs], dtype=np.float64)
    d_fa = fa_A - fa_B

    zf_subjs = [s for s in all_subjs if pt1_A[s]["tot_F"] == 0]
    d_fa_zf = np.array([pt1_A[s]["FAPH"] - pt1_B[s]["FAPH"] for s in zf_subjs], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_f = len(fpos_subjs)
    n_a = len(all_subjs)
    n_z = len(zf_subjs)

    idx_f = rng.integers(0, n_f, size=(B, n_f))
    boot_rec = np.mean(d_rec[idx_f], axis=1)

    idx_a = rng.integers(0, n_a, size=(B, n_a))
    boot_fa = np.mean(d_fa[idx_a], axis=1)

    if n_z > 0:
        idx_z = rng.integers(0, n_z, size=(B, n_z))
        boot_fa_zf = np.mean(d_fa_zf[idx_z], axis=1)
        zf_ci = [round(float(np.percentile(boot_fa_zf, 2.5)), 4), round(float(np.percentile(boot_fa_zf, 97.5)), 4)]
        zf_pt = round(float(np.mean(d_fa_zf)), 4)
    else:
        zf_ci = [0.0, 0.0]
        zf_pt = 0.0

    rec_lo, rec_hi = float(np.percentile(boot_rec, 2.5)), float(np.percentile(boot_rec, 97.5))
    fa_lo, fa_hi = float(np.percentile(boot_fa, 2.5)), float(np.percentile(boot_fa, 97.5))

    # Exact subject-level WTL & strict dominance counts
    w_rec = int(np.sum(d_rec > 1e-9))
    l_rec = int(np.sum(d_rec < -1e-9))
    t_rec = n_f - w_rec - l_rec

    sdom_A = 0
    sdom_B = 0
    for s in fpos_subjs:
        rA, fA = pt1_A[s]["Rec_%"], pt1_A[s]["FAPH"]
        rB, fB = pt1_B[s]["Rec_%"], pt1_B[s]["FAPH"]
        if (rA >= rB - 1e-9) and (fA <= fB + 1e-9) and ((rA > rB + 1e-9) or (fA < fB - 1e-9)):
            sdom_A += 1
        elif (rB >= rA - 1e-9) and (fB <= fA + 1e-9) and ((rB > rA + 1e-9) or (fB < fA - 1e-9)):
            sdom_B += 1

    zf_inc = int(np.sum(d_fa_zf > 1e-9)) if n_z > 0 else 0
    zf_dec = int(np.sum(d_fa_zf < -1e-9)) if n_z > 0 else 0
    zf_tie = n_z - zf_inc - zf_dec

    pass_rec = bool(rec_lo > 0.0)
    pass_fa = bool(fa_hi <= 0.0)
    confirmed = bool(pass_rec and pass_fa)

    return {
        "n_Fpos_subjects": n_f,
        "n_All_subjects": n_a,
        "n_ZeroF_subjects": n_z,
        "B": B,
        "seed": seed,
        "rng": "numpy.random.default_rng",
        "Delta_MacroRec_T1_pt_%": round(float(np.mean(d_rec)), 4),
        "Delta_MacroRec_T1_CI95_%": [round(rec_lo, 4), round(rec_hi, 4)],
        "Delta_MeanFAPH_All_pt": round(float(np.mean(d_fa)), 4),
        "Delta_MeanFAPH_All_CI95": [round(fa_lo, 4), round(fa_hi, 4)],
        "Delta_MeanFAPH_ZeroF_pt": zf_pt,
        "Delta_MeanFAPH_ZeroF_CI95": zf_ci,
        "Fpos_WTL_Rec": f"{w_rec}W/{t_rec}T/{l_rec}L",
        "Fpos_StrictDom_A_vs_B": f"{sdom_A} vs {sdom_B}",
        "ZeroF_FAPH_Dec_Tie_Inc": f"{zf_dec}Dec/{zf_tie}Tie/{zf_inc}Inc",
        "Pass_CI95_low_Rec_gt_0": pass_rec,
        "Pass_CI95_high_FAPH_le_0": pass_fa,
        "Contract_Confirmed": confirmed,
    }


def summarize_buffer_audit(items):
    tot_adm = sum(it["buf_audit"]["n_admitted"] for it in items)
    adm_n = sum(it["buf_audit"]["n_admitted"] * it["buf_audit"]["adm_N_pct"] / 100.0 for it in items)
    adm_f = sum(it["buf_audit"]["n_admitted"] * it["buf_audit"]["adm_F_pct"] / 100.0 for it in items)
    adm_v = sum(it["buf_audit"]["n_admitted"] * it["buf_audit"]["adm_V_pct"] / 100.0 for it in items)
    adm_s = sum(it["buf_audit"]["n_admitted"] * it["buf_audit"]["adm_S_pct"] / 100.0 for it in items)

    s2f_n = [it["buf_audit"]["s2F_buf_N_pct"] for it in items if it["buf_audit"]["s2F_buf_N_pct"] is not None]
    s2f_f = [it["buf_audit"]["s2F_buf_F_pct"] for it in items if it["buf_audit"]["s2F_buf_F_pct"] is not None]
    s2f_v = [it["buf_audit"]["s2F_buf_V_pct"] for it in items if it["buf_audit"]["s2F_buf_V_pct"] is not None]
    s2f_unw = [it["buf_audit"]["s2F_unwarmed_pct"] for it in items if it["buf_audit"]["s2F_unwarmed_pct"] is not None]
    s2f_fid = [it["buf_audit"]["s2F_tmpl_fid"] for it in items if it["buf_audit"]["s2F_tmpl_fid"] is not None]
    cons_med = [it["buf_audit"]["cons_cau_med"] for it in items if it["accounting"]["s2_total"] > 0]

    b_recs = [it["buf_audit"] for it in items if it["rec_id"] in ("s20531", "s20551")]
    return {
        "total_admitted_beats": int(tot_adm),
        "global_adm_N_pct": round(100.0 * adm_n / max(1, tot_adm), 3),
        "global_adm_F_pct": round(100.0 * adm_f / max(1, tot_adm), 4),
        "global_adm_V_pct": round(100.0 * adm_v / max(1, tot_adm), 3),
        "global_adm_S_pct": round(100.0 * adm_s / max(1, tot_adm), 3),
        "n_recs_with_s2F_buf_audit": len(s2f_n),
        "s2F_buf_N_pct_med_min": [round(float(np.median(s2f_n)), 2), round(float(np.min(s2f_n)), 2)] if s2f_n else None,
        "s2F_buf_F_pct_med_max": [round(float(np.median(s2f_f)), 2), round(float(np.max(s2f_f)), 2)] if s2f_f else None,
        "s2F_buf_V_pct_med_max": [round(float(np.median(s2f_v)), 2), round(float(np.max(s2f_v)), 2)] if s2f_v else None,
        "s2F_unwarmed_pct_mean_max": [round(float(np.mean(s2f_unw)), 2), round(float(np.max(s2f_unw)), 2)] if s2f_unw else None,
        "s2F_tmpl_fid_med_min": [round(float(np.median(s2f_fid)), 3), round(float(np.min(s2f_fid)), 3)] if s2f_fid else None,
        "cons_cau_med_across_recs": round(float(np.median(cons_med)), 3) if cons_med else 0.0,
        "B_heavy_subjects_audit": b_recs,
    }


def main():
    t0 = time.time()
    # 1. Strict SHA-256 verification of frozen checkpoint & locked target cohort registry
    ckpt_sha = sha256_file(CKPT_PKL)
    assert ckpt_sha == EXPECTED_CKPT_SHA256, f"Frozen checkpoint SHA mismatch! {ckpt_sha} != {EXPECTED_CKPT_SHA256}"
    reg_sha = sha256_file(LTSTDB_REG_PATH)
    reg = json.load(open(LTSTDB_REG_PATH, "r", encoding="utf-8"))
    inc_records = reg["included_records"]
    assert len(inc_records) == 76

    print(f"[v50] VERIFIED FROZEN CHECKPOINT SHA256: {ckpt_sha}", flush=True)
    print(f"[v50] VERIFIED LTSTDB REGISTRY   SHA256: {reg_sha} ({len(inc_records)} recs / 70 subjs)", flush=True)

    with open(CKPT_PKL, "rb") as f:
        ckpt = pickle.load(f)

    # 2. Download verified WFDB files & extract bit-exact cached features
    prepare_all_ltstdb_features(inc_records)
    items = load_cached_ltstdb(inc_records)

    # 3. Exact Screening vs Evaluated Post-Prefix Sample Accounting (Requirement #4)
    subj_scr_F = Counter()
    subj_eval_F = Counter()
    subj_s2_F = Counter()
    subj_hours = Counter()
    subj_recs = {}
    for it in items:
        s = it["patient_id"]
        subj_recs.setdefault(s, []).append(it["rec_id"])
        subj_scr_F[s] += it["accounting"]["raw_atr_F"]
        subj_eval_F[s] += it["accounting"]["post_prefix_F"]
        subj_s2_F[s] += it["accounting"]["s2_F"]
        subj_hours[s] += it["post_hours"]

    changed_subjs = []
    for s in sorted(subj_recs.keys()):
        if subj_scr_F[s] != subj_eval_F[s]:
            changed_subjs.append({
                "subject_id": s, "recs": subj_recs[s],
                "screening_F": int(subj_scr_F[s]), "eval_post_F": int(subj_eval_F[s]),
                "dropped_F": int(subj_scr_F[s] - subj_eval_F[s]),
            })

    sample_accounting = {
        "n_records": len(items),
        "n_subjects": len(subj_recs),
        "screening_total_F": int(sum(subj_scr_F.values())),
        "screening_Fpos_subjects": int(sum(1 for v in subj_scr_F.values() if v > 0)),
        "eval_valid_win_F": int(sum(it["accounting"]["valid_win_F"] for it in items)),
        "eval_prefix_F": int(sum(it["accounting"]["prefix_F"] for it in items)),
        "eval_post_prefix_beats": int(sum(it["accounting"]["post_prefix_beats"] for it in items)),
        "eval_post_prefix_N_incl_B": int(sum(it["accounting"]["post_prefix_N_incl_B"] for it in items)),
        "eval_post_prefix_B_only": int(sum(it["accounting"]["post_prefix_B"] for it in items)),
        "eval_post_prefix_V": int(sum(it["accounting"]["post_prefix_V"] for it in items)),
        "eval_post_prefix_S": int(sum(it["accounting"]["post_prefix_S"] for it in items)),
        "eval_post_prefix_F": int(sum(subj_eval_F.values())),
        "eval_post_prefix_hours": round(float(sum(subj_hours.values())), 2),
        "eval_Fpos_subjects": int(sum(1 for v in subj_eval_F.values() if v > 0)),
        "eval_ZeroF_subjects": int(sum(1 for v in subj_eval_F.values() if v == 0)),
        "eval_s2_total_beats": int(sum(it["accounting"]["s2_total"] for it in items)),
        "eval_s2_F_beats": int(sum(subj_s2_F.values())),
        "eval_s2_F_pct": round(100.0 * sum(subj_s2_F.values()) / max(1, sum(subj_eval_F.values())), 2),
        "n_recs_K_eq_0": int(sum(1 for it in items if it["K"] == 0)),
        "subjects_F_changed_screening_to_post": changed_subjs,
    }
    print("\n=== LTSTDB SCREENING VS EVALUATED POST-PREFIX SAMPLE ACCOUNTING ===", flush=True)
    print(json.dumps(sample_accounting, indent=2), flush=True)

    buf_summary = summarize_buffer_audit(items)

    # Pre-registered cohorts: Primary + 3 secondary subgroup families
    cohorts = {
        "PRIMARY_All_70_Subjs": items,
        "SEC_Excl_Initial_BIH_59": [it for it in items if not it["is_initial_bih"]],
        "SEC_Initial_BIH_11": [it for it in items if it["is_initial_bih"]],
        "SEC_Zymed_3ch_15": [it for it in items if it["contributor_group"].startswith("Zymed_3ch")],
        "SEC_Other_2ch_Excl_InitBIH_44": [it for it in items if it["contributor_group"] == "Other_2ch_BWH_BIDMC_Pisa"],
        "SEC_Standard_Leads_58": [it for it in items if not it["is_easi"]],
        "SEC_EASI_3ch_12": [it for it in items if it["is_easi"]],
        "SEC_Excl_s2010_69": [it for it in items if not it["is_s2010"]],
    }

    results = {
        "Contract_Version": "Stage2-ClaimContract-v1.0 + Addendum-LTSTDB",
        "Checkpoint_Path": CKPT_PKL,
        "Checkpoint_SHA256": ckpt_sha,
        "Target_Registry_Path": LTSTDB_REG_PATH,
        "Target_Registry_SHA256": reg_sha,
        "Sample_Accounting": sample_accounting,
        "Causal_Buffer_Audit": buf_summary,
        "Primary_Verdict": {},
        "Models": {},
    }

    for mtype in CLF_FAMILIES:
        print(f"\n==================== LTSTDB EXTERNAL EVALUATION ({mtype}, FROZEN 6-DB R1) ====================", flush=True)
        m_res = {"Cohorts": {}, "PerSubject_Primary": {}}
        for coh_name, coh_items in cohorts.items():
            c_out = {"n_records": len(coh_items), "n_subjects": len(set(it["patient_id"] for it in coh_items)), "Configs": {}, "Comparisons": {}}
            pt1_by_cfg = {}
            for cname in CFG_ORDER:
                ev, pt1, prf2 = eval_cohort_full(coh_items, ckpt["models"][mtype][cname])
                c_out["Configs"][cname] = ev
                pt1_by_cfg[cname] = pt1

            # Paired subject-level bootstraps (primary C5 vs C3, plus secondary C5 vs C1 and C3 vs C1)
            c_out["Comparisons"]["C5_vs_C3"] = paired_subject_bootstrap(
                pt1_by_cfg["C5_Causal_Unlabeled_PreQRS_Pwin"], pt1_by_cfg["C3_TwoBranch_FV_SignedRR"]
            )
            c_out["Comparisons"]["C5_vs_C1"] = paired_subject_bootstrap(
                pt1_by_cfg["C5_Causal_Unlabeled_PreQRS_Pwin"], pt1_by_cfg["C1_OneHead_Geom"]
            )
            c_out["Comparisons"]["C3_vs_C1"] = paired_subject_bootstrap(
                pt1_by_cfg["C3_TwoBranch_FV_SignedRR"], pt1_by_cfg["C1_OneHead_Geom"]
            )
            m_res["Cohorts"][coh_name] = c_out

            if coh_name == "PRIMARY_All_70_Subjs":
                # Save full per-subject table for primary cohort
                subj_meta = {}
                for it in coh_items:
                    s = it["patient_id"]
                    if s not in subj_meta:
                        subj_meta[s] = {
                            "recs": [], "contributor_group": it["contributor_group"],
                            "is_initial_bih": it["is_initial_bih"], "is_easi": it["is_easi"],
                            "is_s2010": it["is_s2010"], "s2_F": 0,
                        }
                    subj_meta[s]["recs"].append(it["rec_id"])
                    subj_meta[s]["s2_F"] += it["accounting"]["s2_F"]

                for s in sorted(subj_meta.keys()):
                    row = dict(subj_meta[s])
                    for short_k, cname in [("C1", CFG_ORDER[0]), ("C3", CFG_ORDER[1]), ("C5", CFG_ORDER[2])]:
                        row[short_k] = pt1_by_cfg[cname][s]
                    row["dRec_C5_minus_C3"] = (
                        round(row["C5"]["Rec_%"] - row["C3"]["Rec_%"], 3)
                        if (row["C5"]["Rec_%"] is not None and row["C3"]["Rec_%"] is not None)
                        else None
                    )
                    row["dFAPH_C5_minus_C3"] = round(row["C5"]["FAPH"] - row["C3"]["FAPH"], 4)
                    m_res["PerSubject_Primary"][s] = row

            # Print concise summary for this cohort
            ev1 = c_out["Configs"][CFG_ORDER[0]]
            ev3 = c_out["Configs"][CFG_ORDER[1]]
            ev5 = c_out["Configs"][CFG_ORDER[2]]
            cmp53 = c_out["Comparisons"]["C5_vs_C3"]
            print(f"\n  [{coh_name}] (recs={c_out['n_records']}, subjs={c_out['n_subjects']}, F+={cmp53['n_Fpos_subjects']}, 0F={cmp53['n_ZeroF_subjects']})", flush=True)
            for tag, ev in [("C1", ev1), ("C3", ev3), ("C5", ev5)]:
                print(
                    f"    {tag} | TP={ev['T1_TP']:4d} FP_NS={ev['T1_FP_NS']:5d} FP_V={ev['T1_FPV']:5d} F1={ev['T1_F1']:.4f} | "
                    f"MacroRec={ev['T1_MacroPat_Rec_%']:5.2f}% FAPH={ev['T1_MeanPat_FAPH_All']:5.2f} (0F={ev['T1_MeanPat_FAPH_ZeroF']:5.2f}, Pool={ev['T1_Pooled_FAPH']:5.2f}, <=2:{ev['T1_FracPats_LE2_%']:5.1f}%) | "
                    f"SpecAll={ev['T1_Spec_AllNonF_%']:.4f}% Spec0F={ev['T1_Spec_ZeroFRecs_%']:.4f}% | AP={ev['AP']:.4f} Pat@F2={ev['MacroPat_Pat@F2_%']:5.2f}%",
                    flush=True,
                )
            print(
                f"    C5-C3 Bootstrap (B=10k): dMacroRec={cmp53['Delta_MacroRec_T1_pt_%']:+.2f}% CI95={cmp53['Delta_MacroRec_T1_CI95_%']} | "
                f"dMeanFAPH={cmp53['Delta_MeanFAPH_All_pt']:+.4f} CI95={cmp53['Delta_MeanFAPH_All_CI95']} | "
                f"WTL={cmp53['Fpos_WTL_Rec']} StrictDom={cmp53['Fpos_StrictDom_A_vs_B']} | 0F:{cmp53['ZeroF_FAPH_Dec_Tie_Inc']} | "
                f"CONFIRMED={cmp53['Contract_Confirmed']}",
                flush=True,
            )

        results["Models"][mtype] = m_res

    prim_cmp = results["Models"]["SVM"]["Cohorts"]["PRIMARY_All_70_Subjs"]["Comparisons"]["C5_vs_C3"]
    results["Primary_Verdict"] = {
        "Primary_Family": "SVM",
        "Primary_Cohort": "PRIMARY_All_70_Subjs (70 subjects / 76 records after excluding 10 verified EDB-overlapping subjects)",
        "Primary_Comparison": "C5_Causal_Unlabeled_PreQRS_Pwin vs C3_TwoBranch_FV_SignedRR",
        "Delta_MacroRec_T1_pt_%": prim_cmp["Delta_MacroRec_T1_pt_%"],
        "Delta_MacroRec_T1_CI95_%": prim_cmp["Delta_MacroRec_T1_CI95_%"],
        "Delta_MeanFAPH_All_pt": prim_cmp["Delta_MeanFAPH_All_pt"],
        "Delta_MeanFAPH_All_CI95": prim_cmp["Delta_MeanFAPH_All_CI95"],
        "Pass_CI95_low_Rec_gt_0": prim_cmp["Pass_CI95_low_Rec_gt_0"],
        "Pass_CI95_high_FAPH_le_0": prim_cmp["Pass_CI95_high_FAPH_le_0"],
        "Verdict_Code": "CONFIRMED" if prim_cmp["Contract_Confirmed"] else "NOT_CONFIRMED",
        "Verdict_Text_VI": (
            "XÁC NHẬN CẢI THIỆN theo hợp đồng bảo thủ Stage2-ClaimContract-v1.0"
            if prim_cmp["Contract_Confirmed"]
            else "CHƯA XÁC NHẬN ĐƯỢC CẢI THIỆN theo hợp đồng bảo thủ Stage2-ClaimContract-v1.0"
        ),
    }

    out_json = os.path.join(AUD_DIR, "V50_LTSTDB_PRIMARY_EXTERNAL_VALIDATION_20261122.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n==================== PRIMARY CONTRACT VERDICT ====================", flush=True)
    print(json.dumps(results["Primary_Verdict"], indent=2, ensure_ascii=False), flush=True)
    print(f"\nSaved full audit JSON: {out_json} (total {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
