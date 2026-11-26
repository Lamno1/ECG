r"""
lab_v63d_mghdb_pediatric_adult_window.py
========================================
Package A diagnostics for MGHDB critical-care transfer failure (strictly read-only
on all frozen checkpoints, registries, and canonical caches):

  A1. Exact prefix RR / HR / neighbor intrusion audit across ALL 219 MGHDB records
      (from local .ari/.hea files, 100% coverage) + direct spatial flank residual
      energy audit on streamed ECG waveforms (100% of 19 pediatric records, 100% of
      26 K=0 F+ records including all 11 high-eta_0 records, plus 30 adult controls).

  A2. Adult-only (age >= 18) vs Pediatric (<18) vs Unknown stratification of MGHDB
      Tier-1 locked predictions (Frozen SVM A3/R3, Frozen SVM C3/R3, Frozen Primary
      LightGBM A3/R3) across ALL 213 Tier-1 units, with paired 10,000-resample
      bootstrap 95% CIs and exact Wilcoxon p-values.

  A3. Scale-free & flank-masked 374D window ablation (avoiding invalid dimension-changed
      comparisons against fixed 374D thresholds):
        (a) Scale-free dimensionless ratio rho_{eta/V} = eta_0 / median_{v in V} ||P(v - b_N)||_2
            on full 517-ms window ([0:187]) vs core 300-ms window ([39:147], [-150,+150] ms),
            where rho_{eta/V} <= 0.50 iff median PVC exceeds 2*eta_0.
        (b) Flank-masked 374D Stage-1 anchor discovery (zeroing outer flanks [0:39] & [147:187]
            in x_t and b_N, keeping exact 374D space with RMS dimension-equalizing factor
            sqrt(187/108) so thresholds max(10, 2*eta_0) & radius 12.0 remain scale-matched),
            testing K>0 recovery separately on pediatric high-eta_0 records vs artifact/adult
            records.

Outputs: audits/V63D_MGHDB_PEDIATRIC_ADULT_WINDOW_20261125.json
"""


import os as _os, sys as _sys
from pathlib import Path as _Path
_SRC_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_DATA_ROOT = _Path(_os.environ.get("ECG_DATA_ROOT", str(_REPO_ROOT / "data_public_ecg")))
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import json
import pickle
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import wfdb
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(_REPO_ROOT))
import stage2_frozen_pipeline_v2 as pipe2
import v62_feature_engine as eng

ROOT = _REPO_ROOT
MGH_DIR = (_DATA_ROOT / "mghdb")
CACHE_V60 = (_DATA_ROOT / "cache_v60_mghdb_a3")
CACHE_V63B = (_DATA_ROOT / "cache_v63b_ext/mghdb")
TMP_RAW = (_DATA_ROOT / "tmp_v63d_mghdb")
REG_MGH = ROOT / "registry" / "target_cohort_mghdb_v1.json"
CKPT_V58 = ROOT / "checkpoints" / "stage2_source6db_subjectaware251_a3_r3b20_v2.pkl"
CKPT_V61_LGB = ROOT / "checkpoints" / "v61_lightgbm_a3_6db.pkl"
AUD_OUT = ROOT / "audits" / "V63D_MGHDB_PEDIATRIC_ADULT_WINDOW_20261125.json"

MGH_BASE_URL = "https://physionet.org/files/mghdb/1.0.0"
CORE_SLICE = slice(39, 147)  # 108 samples around fiducial 93 -> [-150.0, +150.0] ms (300 ms)
N_BOOT = 10000
SEED = 20261125


def parse_mgh_age_years(hea_path: Path):
    txt = hea_path.read_text(errors="replace")
    for line in txt.splitlines():
        if not line.startswith("#"):
            continue
        low = line.lower()
        if "<age>:" in low:
            m = re.search(r"<age>:\s*([0-9]+(?:\.[0-9]+)?)\s*([a-z]*)", low)
            if m:
                val, unit = float(m.group(1)), m.group(2)
                if unit.startswith("m"):
                    return val / 12.0
                if unit.startswith("w"):
                    return val / 52.0
                if unit.startswith("d"):
                    return val / 365.0
                return val
            return None
        if "age:" in low or "age " in low:
            m = re.search(r"age[:\s]+([0-9]+(?:\.[0-9]+)?)\s*(y|yr|yrs|year|years|m|mo|mos|month|months|w|wk|d)?", low)
            if m:
                val, unit = float(m.group(1)), (m.group(2) or "y")
                if unit.startswith("m"):
                    return val / 12.0
                if unit.startswith("w"):
                    return val / 52.0
                if unit.startswith("d"):
                    return val / 365.0
                return val
    return None


def age_group(age_yr):
    if age_yr is None:
        return "unknown"
    return "pediatric" if age_yr < 18.0 else "adult"


def build_core_mask_374():
    m1 = np.zeros(187, dtype=np.float64)
    m1[CORE_SLICE] = 1.0
    return np.concatenate([m1, m1])


def compute_prefix_tangent(joint_prefix):
    b_N = np.median(joint_prefix, axis=0)
    diffs = joint_prefix - b_N
    _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
    U = Vt[:4].T
    proj = diffs - (diffs @ U) @ U.T
    norms = np.linalg.norm(proj, axis=1)
    eta_0 = float(np.percentile(norms, 95))
    return b_N, U, norms, eta_0


def compute_stage1_K_exact(data, mask374=None, rms_scale=1.0):
    """
    Run BIT-EXACT Stage-1 bimodal prefix + offline ventricular anchor discovery via
    v62_feature_engine.run_dualpole_variant (optionally flank-masked with RMS
    dimension-equalizing scale rms_scale so 374D thresholds remain scale-matched).
    Guarantees 100% bit-exact parity with frozen v60 when mask374 is None.
    """
    if mask374 is not None:
        J = (data["joint_all"] * mask374[None, :]) * rms_scale
        d_in = dict(data)
        d_in["joint_all"] = J
    else:
        J = data["joint_all"]
        d_in = data

    # Exact Stage-1 bimodal prefix + eta_0 (matching v62_feature_engine lines 215-261)
    joint_50 = J[:50]
    rr_50 = d_in["pre_rr_loc"][:50]
    km2 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(joint_50)
    clusters = km2.labels_
    counts = np.bincount(clusters, minlength=2)
    sil = float(silhouette_score(joint_50, clusters)) if min(counts) >= 2 else 0.0
    if sil >= 0.35 and min(counts) >= 4:
        c0_idx, c1_idx = np.where(clusters == 0)[0], np.where(clusters == 1)[0]
        m0, m1 = np.median(joint_50[c0_idx], axis=0), np.median(joint_50[c1_idx], axis=0)
        s0 = float(np.max(np.abs(np.diff(m0[:187]))) + np.max(np.abs(np.diff(m0[187:]))))
        s1 = float(np.max(np.abs(np.diff(m1[:187]))) + np.max(np.abs(np.diff(m1[187:]))))
        sel = 0 if (counts[0] > counts[1] or (counts[0] == counts[1] and s0 >= s1)) else 1
        sel_idx = c0_idx if sel == 0 else c1_idx
        med_sel = np.median(joint_50[sel_idx], axis=0)
        dists = np.linalg.norm(joint_50[sel_idx] - med_sel, axis=1)
        q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
        final_norm_idx = sel_idx[dists <= np.median(dists) + 1.5 * (q75 - q25)]
        if len(final_norm_idx) < 8:
            final_norm_idx = sel_idx
    else:
        med_all = np.median(joint_50, axis=0)
        dists = np.linalg.norm(joint_50 - med_all, axis=1)
        q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
        final_norm_idx = np.where(dists <= np.median(dists) + 1.5 * (q75 - q25))[0]
        if len(final_norm_idx) < 10:
            final_norm_idx = np.arange(50)

    norm_beats = joint_50[final_norm_idx]
    b_N = np.median(norm_beats, axis=0)
    centered = norm_beats - b_N
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    U = Vt[:min(4, Vt.shape[0]), :].T
    norm_y_pref = centered - (centered @ U) @ U.T
    eta_0 = float(np.percentile(np.linalg.norm(norm_y_pref, axis=1), 95)) + 1e-6

    J_post = J[50:]
    lbl_post = d_in["labels"][50:]
    diffs_post = J_post - b_N[None, :]
    Y_post = diffs_post - (diffs_post @ U) @ U.T
    y_norms = np.linalg.norm(Y_post, axis=1)

    v_mask = (lbl_post == 1)
    n_v = int(np.sum(v_mask))
    if n_v > 0:
        v_yn = y_norms[v_mask]
        med_v_yn = float(np.median(v_yn))
        rho_eta_v = float(eta_0 / max(med_v_yn, 1e-9))
        pvc_pass_2eta = float(np.mean(v_yn >= 2.0 * eta_0))
        pvc_pass_floor = float(np.mean(v_yn >= max(10.0, 2.0 * eta_0)))
    else:
        med_v_yn = rho_eta_v = pvc_pass_2eta = pvc_pass_floor = None

    # Exact Stage-1 K & s2_F via v62_feature_engine
    res_dp = eng.run_dualpole_variant(d_in, anchor_mode="offline", use_tangent_U=True, use_local_bN=True)
    s2_f = int(np.sum(res_dp["y"][res_dp["s2_mask"]] == 2))

    return {
        "eta_0": round(eta_0, 3),
        "n_v": n_v,
        "med_v_yn": round(med_v_yn, 3) if med_v_yn is not None else None,
        "rho_eta_v": round(rho_eta_v, 4) if rho_eta_v is not None else None,
        "pvc_pass_2eta_frac": round(pvc_pass_2eta, 4) if pvc_pass_2eta is not None else None,
        "pvc_pass_floor_frac": round(pvc_pass_floor, 4) if pvc_pass_floor is not None else None,
        "K": int(res_dp["K"]),
        "s2_F": s2_f,
    }


def download_and_analyze_waveform(rid, m60, core_mask_374, flank_mask_374, rms_scale):
    """Stream .dat from PhysioNet into temp dir, run A1-flank + A3, and immediately unlink .dat."""
    TMP_RAW.mkdir(parents=True, exist_ok=True)
    cache_rec = TMP_RAW / f"{rid}_v63d.pkl"
    if cache_rec.exists():
        return rid, pickle.load(open(cache_rec, "rb"))

    for ext in (".hea", ".ari"):
        dst = TMP_RAW / f"{rid}{ext}"
        if not dst.exists():
            shutil.copy2(MGH_DIR / f"{rid}{ext}", dst)
    dat_p = TMP_RAW / f"{rid}.dat"
    url = f"{MGH_BASE_URL}/{rid}.dat"

    last_err = None
    for attempt in range(5):
        try:
            if dat_p.exists():
                dat_p.unlink()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as resp, open(dat_p, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)
            if dat_p.stat().st_size < 10000:
                raise IOError(f"Truncated download for {rid}: {dat_p.stat().st_size} bytes")
            # Patch sig_len == 0 in temp .hea (e.g. mgh061) using exact .dat size & header format
            hea_tmp = TMP_RAW / f"{rid}.hea"
            h_lines = hea_tmp.read_text(errors="replace").splitlines()
            h0 = h_lines[0].split()
            if len(h0) >= 4 and h0[3] == "0":
                hdr = wfdb.rdheader(str(TMP_RAW / rid))
                fmt0 = str(hdr.fmt[0])
                n_sig = int(hdr.n_sig)
                b_per_frame = (n_sig * 3) // 2 if fmt0 == "212" else (n_sig * 2 if fmt0 == "16" else n_sig * 2)
                n_frames = dat_p.stat().st_size // max(b_per_frame, 1)
                h0[3] = str(n_frames)
                h_lines[0] = " ".join(h0)
                hea_tmp.write_text("\n".join(h_lines) + "\n", encoding="latin-1")
            data = eng.load_beats_and_rr(
                str(TMP_RAW / rid), "ari", m60["locked_channel_indices"], m60["label_map_version"],
                filter_mode="offline_filtfilt",
            )
            break
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    else:
        raise last_err

    try:
        joint_all = data["joint_all"]
        labels = data["labels"]
        pre_rr_loc = data["pre_rr_loc"]

        # Direct spatial residual energy in prefix (raw & tangent-projected)
        pref_J = joint_all[:50]
        b_N = np.median(pref_J, axis=0)
        diffs = pref_J - b_N
        _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
        U = Vt[:4].T
        Y_pref = diffs - (diffs @ U) @ U.T

        raw_tot_e = np.sum(diffs ** 2, axis=1) + 1e-12
        raw_flank_e = np.sum((diffs * flank_mask_374[None, :]) ** 2, axis=1)
        raw_flank_frac = float(np.median(raw_flank_e / raw_tot_e))

        proj_tot_e = np.sum(Y_pref ** 2, axis=1) + 1e-12
        proj_flank_e = np.sum((Y_pref * flank_mask_374[None, :]) ** 2, axis=1)
        proj_flank_frac = float(np.median(proj_flank_e / proj_tot_e))

        flank_rms = float(np.median(np.sqrt(raw_flank_e / (2.0 * 79.0))))
        core_rms = float(np.median(np.sqrt(np.sum((diffs * core_mask_374[None, :]) ** 2, axis=1) / (2.0 * 108.0))))
        flank_to_core_rms = flank_rms / max(core_rms, 1e-9)

        st_full = compute_stage1_K_exact(data, mask374=None, rms_scale=1.0)
        st_mask = compute_stage1_K_exact(data, mask374=core_mask_374, rms_scale=rms_scale)

        res = {
            "raw_flank_e_frac": round(raw_flank_frac, 4),
            "proj_flank_e_frac": round(proj_flank_frac, 4),
            "flank_to_core_rms": round(flank_to_core_rms, 3),
            "full_517ms": st_full,
            "masked_300ms_374D": st_mask,
        }
        with open(cache_rec, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        return rid, res
    finally:
        for ext in (".dat", ".hea", ".ari"):
            p = TMP_RAW / f"{rid}{ext}"
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


def summarize_unit_subset(umap, unit_ids, rng):
    fpos = [u for u in unit_ids if umap[u]["tot_F"] > 0]
    zf = [u for u in unit_ids if umap[u]["tot_F"] == 0]
    ge2 = [u for u in fpos if umap[u]["tot_F"] >= 2]

    tp_f = np.array([umap[u]["TP"] for u in fpos], dtype=np.float64)
    f_f = np.array([umap[u]["tot_F"] for u in fpos], dtype=np.float64)
    rec_f = 100.0 * tp_f / np.maximum(f_f, 1.0) if len(fpos) > 0 else np.array([])

    tp_ge2 = np.array([umap[u]["TP"] for u in ge2], dtype=np.float64)
    f_ge2 = np.array([umap[u]["tot_F"] for u in ge2], dtype=np.float64)

    fa_all = np.array([(umap[u]["FP_NS"] + umap[u]["FP_V"]) / max(umap[u]["hours"], 1e-9) for u in unit_ids])
    fa_0f = np.array([(umap[u]["FP_NS"] + umap[u]["FP_V"]) / max(umap[u]["hours"], 1e-9) for u in zf]) if len(zf) > 0 else np.array([0.0])

    if len(fpos) > 0:
        idx = rng.integers(0, len(fpos), size=(N_BOOT, len(fpos)))
        tp_b = tp_f[idx].sum(axis=1)
        f_b = f_f[idx].sum(axis=1)
        pr_b = 100.0 * tp_b / np.maximum(f_b, 1.0)
        pr_ci = [round(float(np.percentile(pr_b, 2.5)), 2), round(float(np.percentile(pr_b, 97.5)), 2)]
    else:
        pr_ci = [0.0, 0.0]

    return {
        "n_units": len(unit_ids),
        "n_Fpos": len(fpos),
        "n_0F": len(zf),
        "tot_F": int(np.sum(f_f)) if len(fpos) > 0 else 0,
        "K_pos_units": int(sum(1 for u in unit_ids if umap[u]["K_max"] > 0)),
        "K_pos_Fpos": int(sum(1 for u in fpos if umap[u]["K_max"] > 0)),
        "s2_F": int(sum(umap[u]["s2_F"] for u in fpos)),
        "TP": int(np.sum(tp_f)) if len(fpos) > 0 else 0,
        "Fpos_hit": int(np.sum(tp_f > 0)) if len(fpos) > 0 else 0,
        "MacroRec_%": round(float(np.mean(rec_f)), 2) if len(fpos) > 0 else 0.0,
        "MacroRec_Fge2_%": round(float(np.mean(100.0 * tp_ge2 / f_ge2)), 2) if len(ge2) > 0 else 0.0,
        "PooledRec_%": round(100.0 * float(np.sum(tp_f)) / max(float(np.sum(f_f)), 1.0), 2) if len(fpos) > 0 else 0.0,
        "PooledRec_CI95": pr_ci,
        "MeanFAPH_All": round(float(np.mean(fa_all)), 3) if len(unit_ids) > 0 else 0.0,
        "MeanFAPH_0F": round(float(np.mean(fa_0f)), 3) if len(zf) > 0 else 0.0,
        "ZeroF_Max": round(float(np.max(fa_0f)), 2) if len(zf) > 0 else 0.0,
    }


def paired_test(vals_a, vals_b, rng):
    d = np.asarray(vals_a, dtype=np.float64) - np.asarray(vals_b, dtype=np.float64)
    w = int(np.sum(d > 1e-12))
    l = int(np.sum(d < -1e-12))
    t = int(len(d) - w - l)
    if w + l == 0:
        p = 1.0
    else:
        nz = d[np.abs(d) > 1e-12]
        p = float(wilcoxon(nz, alternative="two-sided", method="exact" if len(nz) <= 25 else "auto").pvalue)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return {
        "diff": round(float(np.mean(d)), 3),
        "ci95": [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)],
        "WTL": f"{w}/{t}/{l}",
        "p_W": round(p, 5),
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    reg = json.loads(REG_MGH.read_text(encoding="utf-8"))["included_records"]
    print(f"Loaded MGHDB registry: {len(reg)} included records", flush=True)

    core_mask_374 = build_core_mask_374()
    flank_mask_374 = 1.0 - core_mask_374
    rms_scale = float(np.sqrt(187.0 / 108.0))  # 1.31585

    high_11 = {"mgh026", "mgh042", "mgh217", "mgh190", "mgh219", "mgh213", "mgh216", "mgh212", "mgh045", "mgh049", "mgh047"}

    # =========================================================================
    # PART A1 (Local .ari/.hea): Exact RR / HR / neighbor intrusion across ALL 219 records
    # =========================================================================
    rec_rows = {}
    m60_map = {}
    for r in reg:
        rid = r["rec_id"]
        sid = r["subject_id_tier1"]
        age_yr = parse_mgh_age_years(MGH_DIR / f"{rid}.hea")
        grp = age_group(age_yr)
        m60 = json.loads((CACHE_V60 / f"{rid}_meta.json").read_text(encoding="utf-8"))
        m60_map[rid] = m60
        d63 = pickle.load(open(CACHE_V63B / f"{rid}_v63b.pkl", "rb"))
        fs = float(m60["fs"])
        ann = wfdb.rdann(str(MGH_DIR / rid), "ari")

        # Keep exact mapped beats matching load_beats_and_rr
        r_peaks, labels = [], []
        for s, sym in zip(ann.sample, ann.symbol):
            c = pipe2.map_symbol(sym, m60["label_map_version"])
            if c >= 0:
                r_peaks.append(float(s))
                labels.append(c)
        r_peaks = np.asarray(r_peaks, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int8)

        rr_ms = np.diff(r_peaks) / fs * 1000.0
        pref_pre_rr = np.empty(50, dtype=np.float64)
        pref_pre_rr[0] = rr_ms[0]
        pref_pre_rr[1:] = rr_ms[:49]
        pref_post_rr = rr_ms[:50]

        med_rr_ms = float(np.median(pref_pre_rr))
        med_hr_bpm = 60000.0 / max(med_rr_ms, 1.0)
        fid_intrusion_frac = float(np.mean((pref_pre_rr < 258.33) | (pref_post_rr < 261.11)))
        qrs_edge_320_frac = float(np.mean((pref_pre_rr < 320.0) | (pref_post_rr < 320.0)))
        win_overlap_517_frac = float(np.mean((pref_pre_rr < 516.67) | (pref_post_rr < 516.67)))
        pref_non_n_frac = float(np.mean(labels[:50] != 0))
        post_v_count = int(np.sum(labels[50:] == 1))

        rec_rows[rid] = {
            "rec_id": rid,
            "subject_id_tier1": sid,
            "age_yr": age_yr,
            "age_group": grp,
            "tot_F": int(d63["tot_F"]),
            "post_V": post_v_count,
            "K_orig": int(m60["K"]),
            "eta_0_orig": round(float(d63["eta_0"]), 3),
            "in_high11": rid in high_11,
            "med_rr_ms": round(med_rr_ms, 1),
            "med_hr_bpm": round(med_hr_bpm, 1),
            "fid_intrusion_frac": round(fid_intrusion_frac, 4),
            "qrs_edge_320_frac": round(qrs_edge_320_frac, 4),
            "win_overlap_517_frac": round(win_overlap_517_frac, 4),
            "pref_non_n_frac": round(pref_non_n_frac, 4),
        }

    # =========================================================================
    # PART A2 (Local cache_v60): Adult-only vs Pediatric Frozen Stratification (ALL 213 units)
    # =========================================================================
    ck58 = pickle.load(open(CKPT_V58, "rb"))
    ck61 = pickle.load(open(CKPT_V61_LGB, "rb"))
    svm_pack = ck58["models"]["SVM"]
    frozen_cfgs = {
        "Frozen_SVM__A3_R3": ("SVM", svm_pack["A3_HPhiW_NS_R3_B20"]),
        "Frozen_SVM__C3_R3": ("SVM", svm_pack["B0_C3_R3_B20"]),
        "Frozen_SVM__C3_R1": ("SVM", svm_pack["B0_C3_R1"]),
        "Frozen_LGB__A3_R3": ("LGB", ck61),
    }

    unit_age_sets = {}
    for x in rec_rows.values():
        unit_age_sets.setdefault(x["subject_id_tier1"], set()).add(x["age_group"])
    mixed_units = {u: list(s) for u, s in unit_age_sets.items() if len(s) > 1}
    print(f"Tier-1 mixed-age units: {mixed_units}", flush=True)
    unit_age_map = {}
    for u, s in unit_age_sets.items():
        if "pediatric" in s:
            unit_age_map[u] = "pediatric"
        elif "adult" in s:
            unit_age_map[u] = "adult"
        else:
            unit_age_map[u] = "unknown"

    frozen_umaps = {}
    for mname, (fam, mod) in frozen_cfgs.items():
        umap = {}
        fN = mod.get("feat_idx_N", mod.get("fN", eng.IDX_A3_N))
        fV = mod.get("feat_idx_V", mod.get("fV", eng.IDX_C3_V))
        for r in reg:
            rid, sid = r["rec_id"], r["subject_id_tier1"]
            z = np.load(CACHE_V60 / f"{rid}.npz")
            m = m60_map[rid]
            y = z["y"].astype(np.int8)
            s2_mask = z["s2_mask"].astype(bool)
            X = z["X29_s2"].astype(np.float64)
            tp = fp_ns = fp_v = 0
            s2_f = int(np.sum(y[s2_mask] == 2))
            if len(X) > 0:
                if fam == "SVM":
                    sN = mod["clfN"].decision_function(mod["scN"].transform(X[:, fN]))
                    sV = mod["clfV"].decision_function(mod["scV"].transform(X[:, fV]))
                else:
                    sN = mod["clfN"].predict(X[:, fN], raw_score=True)
                    sV = mod["clfV"].predict(X[:, fV], raw_score=True)
                sc = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
                ys2 = y[s2_mask]
                pred = sc >= 0.0
                tp = int(np.sum(pred & (ys2 == 2)))
                fp_ns = int(np.sum(pred & ((ys2 == 0) | (ys2 == 3))))
                fp_v = int(np.sum(pred & (ys2 == 1)))
            u = umap.setdefault(sid, {"tot_F": 0, "s2_F": 0, "K_max": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
            u["tot_F"] += int(np.sum(y == 2))
            u["s2_F"] += s2_f
            u["K_max"] = max(u["K_max"], int(m["K"]))
            u["hours"] += float(m["post_hours"])
            u["TP"] += tp
            u["FP_NS"] += fp_ns
            u["FP_V"] += fp_v
        frozen_umaps[mname] = umap

    all_units = sorted(unit_age_map.keys())
    subsets = {
        "All_213": all_units,
        "Adult_ge18": [u for u in all_units if unit_age_map[u] == "adult"],
        "Pediatric_lt18": [u for u in all_units if unit_age_map[u] == "pediatric"],
        "Unknown_age": [u for u in all_units if unit_age_map[u] == "unknown"],
    }

    strat_results = {}
    print("\n=== A2: ADULT-ONLY vs PEDIATRIC MGHDB TIER-1 FROZEN STRATIFICATION ===", flush=True)
    for sname, uids in subsets.items():
        strat_results[sname] = {}
        print(f"\n  [{sname}] (n_units={len(uids)})", flush=True)
        for mname in frozen_cfgs.keys():
            res = summarize_unit_subset(frozen_umaps[mname], uids, rng)
            strat_results[sname][mname] = res
            print(f"    {mname:20s}: F+={res['n_Fpos']:2d} (K>0:{res['K_pos_Fpos']:2d}, s2F={res['s2_F']:3d}/{res['tot_F']:3d}) | "
                  f"TP={res['TP']:2d} ({res['Fpos_hit']}/{res['n_Fpos']} hit) | Macro={res['MacroRec_%']:5.2f}% (F>=2:{res['MacroRec_Fge2_%']:5.2f}%) | "
                  f"Pool={res['PooledRec_%']:5.2f}% {res['PooledRec_CI95']} | FAPH={res['MeanFAPH_All']:.3f}/{res['MeanFAPH_0F']:.3f}", flush=True)
        if sname in ("All_213", "Adult_ge18"):
            fpos_u = [u for u in uids if frozen_umaps["Frozen_SVM__A3_R3"][u]["tot_F"] > 0]
            rec_a3 = [100.0 * frozen_umaps["Frozen_SVM__A3_R3"][u]["TP"] / frozen_umaps["Frozen_SVM__A3_R3"][u]["tot_F"] for u in fpos_u]
            rec_c3 = [100.0 * frozen_umaps["Frozen_SVM__C3_R3"][u]["TP"] / frozen_umaps["Frozen_SVM__C3_R3"][u]["tot_F"] for u in fpos_u]
            rec_lgb = [100.0 * frozen_umaps["Frozen_LGB__A3_R3"][u]["TP"] / frozen_umaps["Frozen_LGB__A3_R3"][u]["tot_F"] for u in fpos_u]
            fa_a3 = [(frozen_umaps["Frozen_SVM__A3_R3"][u]["FP_NS"] + frozen_umaps["Frozen_SVM__A3_R3"][u]["FP_V"]) / max(frozen_umaps["Frozen_SVM__A3_R3"][u]["hours"], 1e-9) for u in uids]
            fa_c3 = [(frozen_umaps["Frozen_SVM__C3_R3"][u]["FP_NS"] + frozen_umaps["Frozen_SVM__C3_R3"][u]["FP_V"]) / max(frozen_umaps["Frozen_SVM__C3_R3"][u]["hours"], 1e-9) for u in uids]
            fa_lgb = [(frozen_umaps["Frozen_LGB__A3_R3"][u]["FP_NS"] + frozen_umaps["Frozen_LGB__A3_R3"][u]["FP_V"]) / max(frozen_umaps["Frozen_LGB__A3_R3"][u]["hours"], 1e-9) for u in uids]
            strat_results[sname]["Paired"] = {
                "SVM_A3_vs_C3R3_Rec": paired_test(rec_a3, rec_c3, rng),
                "SVM_A3_vs_C3R3_FAPH": paired_test(fa_a3, fa_c3, rng),
                "LGB_vs_SVM_A3_Rec": paired_test(rec_lgb, rec_a3, rng),
                "LGB_vs_SVM_A3_FAPH": paired_test(fa_lgb, fa_a3, rng),
            }
            print("    Paired:", strat_results[sname]["Paired"], flush=True)

    # =========================================================================
    # PART A3 (Streamed waveforms): 100% pediatric (19) + 100% K=0 F+ (26) + 30 adult K>0 controls
    # =========================================================================
    ped_ids = sorted([rid for rid, x in rec_rows.items() if x["age_group"] == "pediatric"])
    k0_fpos_ids = sorted([rid for rid, x in rec_rows.items() if x["K_orig"] == 0 and x["tot_F"] > 0])
    adult_kpos_pool = sorted([rid for rid, x in rec_rows.items() if x["age_group"] == "adult" and x["K_orig"] > 0 and x["post_V"] >= 6])
    adult_ctrl_ids = sorted(rng.choice(adult_kpos_pool, size=min(30, len(adult_kpos_pool)), replace=False).tolist())
    wave_target_ids = sorted(set(ped_ids) | set(k0_fpos_ids) | set(adult_ctrl_ids))
    print(f"\n=== A3: Streaming {len(wave_target_ids)} MGHDB records (19 ped, {len(k0_fpos_ids)} K=0 F+, {len(adult_ctrl_ids)} adult K>0 controls) ===", flush=True)

    wave_out = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(download_and_analyze_waveform, rid, m60_map[rid], core_mask_374, flank_mask_374, rms_scale): rid
            for rid in wave_target_ids
        }
        done_n = 0
        for fut in as_completed(futs):
            rid, res = fut.result()
            wave_out[rid] = res
            rec_rows[rid].update(res)
            done_n += 1
            if done_n % 15 == 0 or done_n == len(wave_target_ids):
                print(f"  Streamed {done_n}/{len(wave_target_ids)} records ({time.time()-t0:.1f}s)", flush=True)

    # Clean up temp dir
    try:
        if TMP_RAW.exists():
            shutil.rmtree(TMP_RAW)
    except OSError:
        pass

    # Aggregate A1 (all 219) & A3 (streamed subsets)
    all_list = list(rec_rows.values())

    def agg_rr(rows):
        rrs = [x["med_rr_ms"] for x in rows]
        hrs = [x["med_hr_bpm"] for x in rows]
        e0s = [x["eta_0_orig"] for x in rows]
        return {
            "n_records": len(rows),
            "med_rr_ms_med_iqr": [round(float(np.median(rrs)), 1), round(float(np.percentile(rrs, 25)), 1), round(float(np.percentile(rrs, 75)), 1)],
            "med_hr_bpm_med_iqr": [round(float(np.median(hrs)), 1), round(float(np.percentile(hrs, 25)), 1), round(float(np.percentile(hrs, 75)), 1)],
            "n_med_rr_lt_517": int(sum(1 for x in rows if x["med_rr_ms"] < 516.67)),
            "n_med_rr_lt_320": int(sum(1 for x in rows if x["med_rr_ms"] < 320.0)),
            "n_any_fid_intrusion_lt260": int(sum(1 for x in rows if x["fid_intrusion_frac"] > 0)),
            "eta_0_med_iqr": [round(float(np.median(e0s)), 2), round(float(np.percentile(e0s, 25)), 2), round(float(np.percentile(e0s, 75)), 2)],
            "n_eta0_gt_5": int(sum(1 for x in rows if x["eta_0_orig"] > 5.0)),
        }

    def agg_wave(rows):
        w = [x for x in rows if "full_517ms" in x]
        flk = [x["raw_flank_e_frac"] for x in w]
        pflk = [x["proj_flank_e_frac"] for x in w]
        f2c = [x["flank_to_core_rms"] for x in w]
        rho_f = [x["full_517ms"]["rho_eta_v"] for x in w if x["full_517ms"]["rho_eta_v"] is not None]
        rho_m = [x["masked_300ms_374D"]["rho_eta_v"] for x in w if x["masked_300ms_374D"]["rho_eta_v"] is not None]
        p2e_f = [x["full_517ms"]["pvc_pass_2eta_frac"] for x in w if x["full_517ms"]["pvc_pass_2eta_frac"] is not None]
        p2e_m = [x["masked_300ms_374D"]["pvc_pass_2eta_frac"] for x in w if x["masked_300ms_374D"]["pvc_pass_2eta_frac"] is not None]
        e0_f = [x["full_517ms"]["eta_0"] for x in w]
        e0_m = [x["masked_300ms_374D"]["eta_0"] for x in w]
        return {
            "n_wave": len(w),
            "raw_flank_e_frac_med_iqr": [round(float(np.median(flk)), 3), round(float(np.percentile(flk, 25)), 3), round(float(np.percentile(flk, 75)), 3)],
            "proj_flank_e_frac_med_iqr": [round(float(np.median(pflk)), 3), round(float(np.percentile(pflk, 25)), 3), round(float(np.percentile(pflk, 75)), 3)],
            "flank_to_core_rms_med_iqr": [round(float(np.median(f2c)), 3), round(float(np.percentile(f2c, 25)), 3), round(float(np.percentile(f2c, 75)), 3)],
            "eta_0_517ms_med": round(float(np.median(e0_f)), 2),
            "eta_0_300ms_masked_med": round(float(np.median(e0_m)), 2),
            "n_with_V": len(rho_f),
            "rho_eta_v_517ms_med_iqr": [round(float(np.median(rho_f)), 3), round(float(np.percentile(rho_f, 25)), 3), round(float(np.percentile(rho_f, 75)), 3)] if rho_f else None,
            "rho_eta_v_300ms_med_iqr": [round(float(np.median(rho_m)), 3), round(float(np.percentile(rho_m, 25)), 3), round(float(np.percentile(rho_m, 75)), 3)] if rho_m else None,
            "pvc_pass_2eta_517ms_mean": round(float(np.mean(p2e_f)), 3) if p2e_f else None,
            "pvc_pass_2eta_300ms_mean": round(float(np.mean(p2e_m)), 3) if p2e_m else None,
            "K_pos_517ms": int(sum(1 for x in w if x["full_517ms"]["K"] > 0)),
            "K_pos_300ms_masked": int(sum(1 for x in w if x["masked_300ms_374D"]["K"] > 0)),
        }

    rr_summary = {
        "all_219": agg_rr(all_list),
        "pediatric_19": agg_rr([x for x in all_list if x["age_group"] == "pediatric"]),
        "adult_187": agg_rr([x for x in all_list if x["age_group"] == "adult"]),
        "unknown_13": agg_rr([x for x in all_list if x["age_group"] == "unknown"]),
        "high11_ped_8": agg_rr([x for x in all_list if x["in_high11"] and x["age_group"] == "pediatric"]),
        "high11_nonped_3": agg_rr([x for x in all_list if x["in_high11"] and x["age_group"] != "pediatric"]),
    }

    wave_summary = {
        "pediatric_all_19": agg_wave([rec_rows[r] for r in ped_ids]),
        "high11_ped_8": agg_wave([x for x in all_list if x["in_high11"] and x["age_group"] == "pediatric"]),
        "high11_nonped_3": agg_wave([x for x in all_list if x["in_high11"] and x["age_group"] != "pediatric"]),
        "k0_fpos_all_26": agg_wave([rec_rows[r] for r in k0_fpos_ids]),
        "adult_kpos_ctrl_30": agg_wave([rec_rows[r] for r in adult_ctrl_ids]),
    }

    print("\n=== A1 SUMMARY (ALL 219 MGHDB RECORDS) ===")
    for k, v in rr_summary.items():
        print(f"  {k:18s}: n={v['n_records']:3d}, RR={v['med_rr_ms_med_iqr'][0]:5.1f}ms {v['med_rr_ms_med_iqr'][1:]} (HR={v['med_hr_bpm_med_iqr'][0]:5.1f}), "
              f"RR<517:{v['n_med_rr_lt_517']:2d}, RR<320:{v['n_med_rr_lt_320']:2d}, fidIntr<260:{v['n_any_fid_intrusion_lt260']:2d}, eta0>5:{v['n_eta0_gt_5']:2d}")

    print("\n=== A3 WAVEFORM FLANK CONTAMINATION & SCALE-FREE WINDOW ABLATION ===")
    for k, v in wave_summary.items():
        print(f"  {k:20s} (n={v['n_wave']:2d}): rawFlankE={v['raw_flank_e_frac_med_iqr'][0]:.3f} (f/c={v['flank_to_core_rms_med_iqr'][0]:.2f}), "
              f"eta0={v['eta_0_517ms_med']:.2f}->{v['eta_0_300ms_masked_med']:.2f}, "
              f"rho_etaV={v['rho_eta_v_517ms_med_iqr'][0] if v['rho_eta_v_517ms_med_iqr'] else 'NA'}->"
              f"{v['rho_eta_v_300ms_med_iqr'][0] if v['rho_eta_v_300ms_med_iqr'] else 'NA'}, "
              f"Vpass2eta={v['pvc_pass_2eta_517ms_mean']}->{v['pvc_pass_2eta_300ms_mean']}, "
              f"K>0: {v['K_pos_517ms']}->{v['K_pos_300ms_masked']}")

    print("\n=== DETAILED 11 HIGH-eta_0 K=0 F+ RECORDS ===")
    high11_details = []
    for x in sorted([r for r in all_list if r["in_high11"]], key=lambda r: (r["age_group"], r["rec_id"])):
        high11_details.append(x)
        print(f"  {x['rec_id']} (age={str(x['age_yr']):>4s} {x['age_group']:9s}, F={x['tot_F']:2d}, V={x['post_V']:3d}): "
              f"RR={x['med_rr_ms']:5.1f}ms ({x['med_hr_bpm']:5.1f}bpm), fid<260={x['fid_intrusion_frac']:.2f}, ov517={x['win_overlap_517_frac']:.2f}, "
              f"flankE={x['raw_flank_e_frac']:.2f} (f/c={x['flank_to_core_rms']:.2f}), "
              f"eta0={x['full_517ms']['eta_0']:5.2f}->{x['masked_300ms_374D']['eta_0']:5.2f}, "
              f"rho={x['full_517ms']['rho_eta_v']}->{x['masked_300ms_374D']['rho_eta_v']}, "
              f"K={x['full_517ms']['K']}->{x['masked_300ms_374D']['K']}")

    out = {
        "Version": "V63D_MGHDB_PEDIATRIC_ADULT_WINDOW",
        "Date": "2026-11-25",
        "Core_Window_300ms_Slice": [39, 147],
        "RMS_Equalizing_Scale_374D": round(rms_scale, 5),
        "Mixed_Age_Tier1_Units": mixed_units,
        "A1_RR_Summary_All219": rr_summary,
        "A2_Adult_Pediatric_Frozen_Stratification": strat_results,
        "A3_Waveform_Flank_And_ScaleFree_Window": wave_summary,
        "High11_Details": high11_details,
        "runtime_sec": round(time.time() - t0, 2),
    }
    AUD_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== SAVED {AUD_OUT} in {out['runtime_sec']}s ===", flush=True)


if __name__ == "__main__":
    main()
