r"""
lab_v63a_metrics_influence_holm.py
==================================
Computes subject-cluster bootstrap pooled-beat recall, 1-beat & s2010
influence analyses, exposure-weighted pooled FAPH, and Holm-Bonferroni
multiplicity-adjusted p-values across ALL frozen and revision models on
LTSTDB (70 subjects, 583 F) and MGHDB Tier-1 (213 units, 771 F).

Strictly read-only on all frozen checkpoints and caches.
Outputs: audits/V63A_METRICS_INFLUENCE_HOLM_20261125.json
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
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(_REPO_ROOT))
import v62_feature_engine as eng

ROOT = _REPO_ROOT
CACHE_EXT = (_DATA_ROOT / "cache_v62_external")
CACHE_V62E = (_DATA_ROOT / "cache_v62e_vpole_ext")
CACHE_V59 = (_DATA_ROOT / "cache_v59_ltstdb_a3")
CACHE_V60 = (_DATA_ROOT / "cache_v60_mghdb_a3")
CKPT_V58 = ROOT / "checkpoints" / "stage2_source6db_subjectaware251_a3_r3b20_v2.pkl"
CKPT_V61_LGB = ROOT / "checkpoints" / "v61_lightgbm_a3_6db.pkl"
AUD_OUT = ROOT / "audits" / "V63A_METRICS_INFLUENCE_HOLM_20261125.json"

N_BOOT = 10000
SEED = 20261125


def holm_bonferroni(pvals_dict):
    """Step-down Holm-Bonferroni adjusted p-values for a dict {name: p}."""
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * float(p))
        running_max = max(running_max, val)
        adj[k] = round(running_max, 5)
    return adj


def cluster_boot_pooled_rec(tp_arr, f_arr, rng, n_boot=N_BOOT):
    """Subject-cluster bootstrap 95% CI for pooled-beat recall (%)."""
    n = len(tp_arr)
    tot_f = float(np.sum(f_arr))
    est = 100.0 * float(np.sum(tp_arr)) / max(tot_f, 1.0)
    idx = rng.integers(0, n, size=(n_boot, n))
    tp_b = tp_arr[idx].sum(axis=1)
    f_b = f_arr[idx].sum(axis=1)
    valid = f_b > 0
    recs = np.where(valid, 100.0 * tp_b / np.maximum(f_b, 1.0), np.nan)
    lo, hi = np.nanpercentile(recs, [2.5, 97.5])
    return round(est, 2), [round(float(lo), 2), round(float(hi), 2)]


def cluster_boot_diff_pooled_rec(tp_a, tp_b, f_arr, rng, n_boot=N_BOOT):
    """Subject-cluster bootstrap 95% CI for paired difference in pooled-beat recall (%)."""
    n = len(f_arr)
    tot_f = float(np.sum(f_arr))
    diff = 100.0 * float(np.sum(tp_a - tp_b)) / max(tot_f, 1.0)
    idx = rng.integers(0, n, size=(n_boot, n))
    da = tp_a[idx].sum(axis=1) - tp_b[idx].sum(axis=1)
    fb = f_arr[idx].sum(axis=1)
    valid = fb > 0
    diffs = np.where(valid, 100.0 * da / np.maximum(fb, 1.0), np.nan)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return round(diff, 2), [round(float(lo), 2), round(float(hi), 2)]


def paired_wilcoxon_and_ci(vals_a, vals_b, rng, n_boot=N_BOOT):
    d = np.asarray(vals_a, dtype=np.float64) - np.asarray(vals_b, dtype=np.float64)
    wins = int(np.sum(d > 1e-12))
    losses = int(np.sum(d < -1e-12))
    ties = int(len(d) - wins - losses)
    if wins + losses == 0:
        p = 1.0
    else:
        nz = d[np.abs(d) > 1e-12]
        p = float(wilcoxon(nz, alternative="two-sided", method="exact" if len(nz) <= 25 else "auto").pvalue)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "diff": round(float(np.mean(d)), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "WTL": f"{wins}/{ties}/{losses}",
        "p_wilcoxon": round(p, 5),
    }


def summarize_units(umap, cohort_name, rng):
    subjs = sorted(umap.keys())
    fpos_ids = [s for s in subjs if umap[s]["tot_F"] > 0]
    zf_ids = [s for s in subjs if umap[s]["tot_F"] == 0]
    ge2_ids = [s for s in fpos_ids if umap[s]["tot_F"] >= 2]
    one_beat_ids = [s for s in fpos_ids if umap[s]["tot_F"] == 1]

    tp_f = np.array([umap[s]["TP"] for s in fpos_ids], dtype=np.float64)
    f_f = np.array([umap[s]["tot_F"] for s in fpos_ids], dtype=np.float64)
    rec_f = 100.0 * tp_f / f_f

    pooled_rec, pooled_ci = cluster_boot_pooled_rec(tp_f, f_f, rng)

    tp_ge2 = np.array([umap[s]["TP"] for s in ge2_ids], dtype=np.float64)
    f_ge2 = np.array([umap[s]["tot_F"] for s in ge2_ids], dtype=np.float64)
    pooled_ge2, pooled_ge2_ci = cluster_boot_pooled_rec(tp_ge2, f_ge2, rng)
    macro_ge2 = float(np.mean(100.0 * tp_ge2 / f_ge2))

    # FAPH metrics
    fa_all = np.array([(umap[s]["FP_NS"] + umap[s]["FP_V"]) / max(umap[s]["hours"], 1e-9) for s in subjs])
    fa_0f = np.array([(umap[s]["FP_NS"] + umap[s]["FP_V"]) / max(umap[s]["hours"], 1e-9) for s in zf_ids])
    tot_fp = float(sum(umap[s]["FP_NS"] + umap[s]["FP_V"] for s in subjs))
    tot_h = float(sum(umap[s]["hours"] for s in subjs))
    tot_fp_0f = float(sum(umap[s]["FP_NS"] + umap[s]["FP_V"] for s in zf_ids))
    tot_h_0f = float(sum(umap[s]["hours"] for s in zf_ids))

    res = {
        "TP": int(np.sum(tp_f)),
        "tot_F": int(np.sum(f_f)),
        "Fpos_hit": int(np.sum(tp_f > 0)),
        "n_Fpos": len(fpos_ids),
        "n_1beat_Fpos": len(one_beat_ids),
        "n_1beat_hit": int(sum(1 for s in one_beat_ids if umap[s]["TP"] > 0)),
        "MacroRec_%": round(float(np.mean(rec_f)), 2),
        "MacroRec_Fge2_%": round(macro_ge2, 2),
        "PooledRec_%": pooled_rec,
        "PooledRec_CI95": pooled_ci,
        "PooledRec_Fge2_%": pooled_ge2,
        "PooledRec_Fge2_CI95": pooled_ge2_ci,
        "MeanFAPH_All": round(float(np.mean(fa_all)), 3),
        "MedFAPH_All": round(float(np.median(fa_all)), 3),
        "IQR_FAPH_All": [round(float(np.percentile(fa_all, 25)), 3), round(float(np.percentile(fa_all, 75)), 3)],
        "PooledFAPH_All": round(tot_fp / max(tot_h, 1e-9), 3),
        "MeanFAPH_0F": round(float(np.mean(fa_0f)), 3),
        "PooledFAPH_0F": round(tot_fp_0f / max(tot_h_0f, 1e-9), 3),
        "ZeroF_Max": round(float(np.max(fa_0f)), 2),
    }

    if cohort_name == "ltstdb":
        no_s2010 = [s for s in fpos_ids if s != "ltstdb_s2010"]
        tp_ns = np.array([umap[s]["TP"] for s in no_s2010], dtype=np.float64)
        f_ns = np.array([umap[s]["tot_F"] for s in no_s2010], dtype=np.float64)
        pr_ns, pr_ns_ci = cluster_boot_pooled_rec(tp_ns, f_ns, rng)
        no_s2010_ge2 = [s for s in ge2_ids if s != "ltstdb_s2010"]
        tp_ns2 = np.array([umap[s]["TP"] for s in no_s2010_ge2], dtype=np.float64)
        f_ns2 = np.array([umap[s]["tot_F"] for s in no_s2010_ge2], dtype=np.float64)
        pr_ns2, pr_ns2_ci = cluster_boot_pooled_rec(tp_ns2, f_ns2, rng)
        res["No_s2010_TP"] = int(np.sum(tp_ns))
        res["No_s2010_totF"] = int(np.sum(f_ns))
        res["No_s2010_PooledRec_%"] = pr_ns
        res["No_s2010_PooledRec_CI95"] = pr_ns_ci
        res["No_s2010_MacroRec_%"] = round(float(np.mean(100.0 * tp_ns / f_ns)), 2)
        res["No_s2010_Fge2_MacroRec_%"] = round(float(np.mean(100.0 * tp_ns2 / f_ns2)), 2)
        res["No_s2010_Fge2_PooledRec_%"] = pr_ns2
        res["No_s2010_Fge2_PooledRec_CI95"] = pr_ns2_ci
    return res


def load_frozen_stage2_models(cache_dir, reg_recs, sid_key):
    """Score untouched frozen v58 (A3_R3, C3_R3, C3_R1) & v61 (LightGBM_A3) on v59/v60 caches."""
    ck58 = pickle.load(open(CKPT_V58, "rb"))
    ck61 = pickle.load(open(CKPT_V61_LGB, "rb"))
    svm_pack = ck58["models"]["SVM"]

    # Inspect keys of svm_pack to grab C3_R3, C3_R1, A3_R3 cleanly
    out_models = {}
    configs = {
        "Frozen_SVM__A3_R3": ("SVM", svm_pack["A3_HPhiW_NS_R3_B20"]),
        "Frozen_SVM__C3_R3": ("SVM", svm_pack["B0_C3_R3_B20"]),
        "Frozen_SVM__C3_R1": ("SVM", svm_pack["B0_C3_R1"]),
        "Frozen_LGB__A3_R3": ("LGB", ck61),
    }

    for mname, (fam, mod) in configs.items():
        umap = {}
        fN = mod.get("feat_idx_N", mod.get("fN", eng.IDX_A3_N))
        fV = mod.get("feat_idx_V", mod.get("fV", eng.IDX_C3_V))
        for r in reg_recs:
            rid, sid = r["rec_id"], r[sid_key]
            z = np.load(cache_dir / f"{rid}.npz")
            m = json.loads((cache_dir / f"{rid}_meta.json").read_text(encoding="utf-8"))
            y = z["y"].astype(np.int8)
            s2_mask = z["s2_mask"].astype(bool)
            X = z["X29_s2"].astype(np.float64)
            tp = fp_ns = fp_v = 0
            if len(X) > 0:
                if fam == "SVM":
                    scN, clfN = mod["scN"], mod["clfN"]
                    scV, clfV = mod["scV"], mod["clfV"]
                    sN = clfN.decision_function(scN.transform(X[:, fN]))
                    sV = clfV.decision_function(scV.transform(X[:, fV]))
                else:
                    sN = mod["clfN"].predict(X[:, fN], raw_score=True)
                    sV = mod["clfV"].predict(X[:, fV], raw_score=True)
                sc = np.minimum((sN - mod["tau_N"]) / mod["sig_N"], (sV - mod["tau_V"]) / mod["sig_V"])
                ys2 = y[s2_mask]
                pred = sc >= 0.0
                tp = int(np.sum(pred & (ys2 == 2)))
                fp_ns = int(np.sum(pred & ((ys2 == 0) | (ys2 == 3))))
                fp_v = int(np.sum(pred & (ys2 == 1)))
            u = umap.setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
            u["tot_F"] += int(np.sum(y == 2))
            u["hours"] += float(m["post_hours"])
            u["TP"] += tp
            u["FP_NS"] += fp_ns
            u["FP_V"] += fp_v
        out_models[mname] = umap
    return out_models


def load_all_cohort_units(cohort_name, reg_path, sid_key, cache_s2_dir):
    reg = json.loads(reg_path.read_text(encoding="utf-8"))["included_records"]
    units_by_model = {}

    # 1. All 29 models from cache_v62_external
    c_ext = CACHE_EXT / cohort_name
    for r in reg:
        rid, sid = r["rec_id"], r[sid_key]
        d = pickle.load(open(c_ext / f"{rid}_v62.pkl", "rb"))
        for mname, sc in d["scores"].items():
            is_t3 = ("Tier3" in mname)
            tot_f = int(d["tot_F_t3"]) if is_t3 else int(d["tot_F"])
            hrs = float(d["post_hours_t3"]) if is_t3 else float(d["post_hours"])
            u = units_by_model.setdefault(mname, {}).setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
            u["tot_F"] += tot_f
            u["hours"] += hrs
            u["TP"] += int(sc["TP"])
            u["FP_NS"] += int(sc["FP_NS"])
            u["FP_V"] += int(sc["FP_V"])

    # 2. SinglePole_V_RR from cache_v62e_vpole_ext
    c_vp = CACHE_V62E / cohort_name
    for r in reg:
        rid, sid = r["rec_id"], r[sid_key]
        d = pickle.load(open(c_vp / f"{rid}_vp.pkl", "rb"))
        for mname, sc in d["models"].items():
            u = units_by_model.setdefault(mname, {}).setdefault(sid, {"tot_F": 0, "hours": 0.0, "TP": 0, "FP_NS": 0, "FP_V": 0})
            u["tot_F"] += int(d["tot_F"])
            u["hours"] += float(d["post_hours"])
            u["TP"] += int(sc["TP"])
            u["FP_NS"] += int(sc["FP_NS"])
            u["FP_V"] += int(sc["FP_V"])

    # 3. Frozen v58 & v61 checkpoints
    frozen = load_frozen_stage2_models(cache_s2_dir, reg, sid_key)
    units_by_model.update(frozen)
    return units_by_model


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = {"Version": "v63a_metrics_influence_holm", "Date": "2026-11-25", "N_boot": N_BOOT}

    for cname, reg_f, sid_key, s2_dir in [
        ("ltstdb", ROOT / "registry/target_cohort_ltstdb_v1.json", "subject_id", CACHE_V59),
        ("mghdb", ROOT / "registry/target_cohort_mghdb_v1.json", "subject_id_tier1", CACHE_V60),
    ]:
        units_by_model = load_all_cohort_units(cname, reg_f, sid_key, s2_dir)
        summaries = {}
        for mname in sorted(units_by_model.keys()):
            summaries[mname] = summarize_units(units_by_model[mname], cname, rng)

        # Paired pooled-recall & macro-recall differences vs Full_A3 within family
        paired = {}
        for fam in ("SVM", "LightGBM"):
            ref_k = f"{fam}__Full_A3"
            ref_u = units_by_model[ref_k]
            subjs = sorted(ref_u.keys())
            fpos = [s for s in subjs if ref_u[s]["tot_F"] > 0]
            ge2 = [s for s in fpos if ref_u[s]["tot_F"] >= 2]
            f_arr = np.array([ref_u[s]["tot_F"] for s in fpos], dtype=np.float64)
            tp_ref = np.array([ref_u[s]["TP"] for s in fpos], dtype=np.float64)
            rec_ref = 100.0 * tp_ref / f_arr
            rec_ref_ge2 = np.array([100.0 * ref_u[s]["TP"] / ref_u[s]["tot_F"] for s in ge2], dtype=np.float64)

            p_rec_map = {}
            for mname, umap in sorted(units_by_model.items()):
                if not mname.startswith(f"{fam}__") or mname == ref_k:
                    continue
                tp_m = np.array([umap[s]["TP"] for s in fpos], dtype=np.float64)
                rec_m = 100.0 * tp_m / f_arr
                rec_m_ge2 = np.array([100.0 * umap[s]["TP"] / umap[s]["tot_F"] for s in ge2], dtype=np.float64)
                dp, dp_ci = cluster_boot_diff_pooled_rec(tp_ref, tp_m, f_arr, rng)
                pw_all = paired_wilcoxon_and_ci(rec_ref, rec_m, rng)
                pw_ge2 = paired_wilcoxon_and_ci(rec_ref_ge2, rec_m_ge2, rng)
                paired[mname] = {
                    "Delta_PooledRec_%": dp,
                    "Delta_PooledRec_CI95": dp_ci,
                    "MacroRec_All": pw_all,
                    "MacroRec_Fge2": pw_ge2,
                }
                p_rec_map[mname] = pw_all["p_wilcoxon"]

            # Exploratory Holm adjustment within family
            holm_fam = holm_bonferroni(p_rec_map)
            for mname, padj in holm_fam.items():
                paired[mname]["p_wilcoxon_holm_family"] = padj

        out[cname] = {"Models": summaries, "Paired_vs_FullA3": paired}

    # Explicit 3-test Primary Family Holm adjustment on LTSTDB (frozen checkpoints)
    lt_u = load_all_cohort_units("ltstdb", ROOT / "registry/target_cohort_ltstdb_v1.json", "subject_id", CACHE_V59)
    print("Frozen keys found on LTSTDB:", [k for k in lt_u if k.startswith("Frozen_")])

    # Compute exact primary family Holm if C3_R3 & Frozen_LGB exist
    if "Frozen_SVM__A3_R3" in lt_u and "Frozen_LGB__A3_R3" in lt_u:
        subjs = sorted(lt_u["Frozen_SVM__A3_R3"].keys())
        fpos = [s for s in subjs if lt_u["Frozen_SVM__A3_R3"][s]["tot_F"] > 0]
        zf = [s for s in subjs if lt_u["Frozen_SVM__A3_R3"][s]["tot_F"] == 0]
        rec_svm = [100.0 * lt_u["Frozen_SVM__A3_R3"][s]["TP"] / lt_u["Frozen_SVM__A3_R3"][s]["tot_F"] for s in fpos]
        rec_lgb = [100.0 * lt_u["Frozen_LGB__A3_R3"][s]["TP"] / lt_u["Frozen_LGB__A3_R3"][s]["tot_F"] for s in fpos]
        fa0_svm = [(lt_u["Frozen_SVM__A3_R3"][s]["FP_NS"] + lt_u["Frozen_SVM__A3_R3"][s]["FP_V"]) / lt_u["Frozen_SVM__A3_R3"][s]["hours"] for s in zf]
        fa0_lgb = [(lt_u["Frozen_LGB__A3_R3"][s]["FP_NS"] + lt_u["Frozen_LGB__A3_R3"][s]["FP_V"]) / lt_u["Frozen_LGB__A3_R3"][s]["hours"] for s in zf]
        rec_sp_lgb = [100.0 * lt_u["LightGBM__SinglePole_N_RR"][s]["TP"] / lt_u["LightGBM__SinglePole_N_RR"][s]["tot_F"] for s in fpos]

        prim = {
            "LGB_vs_SVM_Recall": paired_wilcoxon_and_ci(rec_lgb, rec_svm, rng),
            "LGB_vs_SVM_0F_FAPH": paired_wilcoxon_and_ci(fa0_lgb, fa0_svm, rng),
            "DualPoleLGB_vs_SinglePoleN_Recall": paired_wilcoxon_and_ci(rec_lgb, rec_sp_lgb, rng),
        }
        if "Frozen_SVM__C3_R3" in lt_u:
            rec_c3 = [100.0 * lt_u["Frozen_SVM__C3_R3"][s]["TP"] / lt_u["Frozen_SVM__C3_R3"][s]["tot_F"] for s in fpos]
            prim["A3_vs_C3R3_SVM_Recall"] = paired_wilcoxon_and_ci(rec_svm, rec_c3, rng)
        p_map = {k: v["p_wilcoxon"] for k, v in prim.items()}
        h_map = holm_bonferroni(p_map)
        for k in prim:
            prim[k]["p_holm"] = h_map[k]
        out["LTSTDB_Primary_Holm"] = prim

    out["runtime_sec"] = round(time.time() - t0, 2)
    AUD_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Print concise executive summary
    print(f"\n=== DONE v63a in {out['runtime_sec']}s -> {AUD_OUT} ===")
    for cname in ("ltstdb", "mghdb"):
        print(f"\n--- {cname.upper()} ---")
        for mk in [
            "SVM__Full_A3", "LightGBM__Full_A3", "Frozen_LGB__A3_R3", "RF__Full_A3",
            "SVM__Abl_No_DualPoleCoords", "LightGBM__Abl_No_DualPoleCoords",
            "SVM__Abl_Static_bN", "LightGBM__Abl_Static_bN",
            "SVM__Abl_No_Tangent_U", "LightGBM__Abl_No_Tangent_U",
            "SVM__Abl_No_ThalesPyth", "LightGBM__Abl_No_ThalesPyth",
            "SVM__SinglePole_N_RR", "LightGBM__SinglePole_N_RR",
            "SVM__SinglePole_V_RR", "LightGBM__SinglePole_V_RR",
            "SVM__Tier3_StrictCausal_LockedT0", "LightGBM__Tier3_StrictCausal_LockedT0",
        ]:
            if mk in out[cname]["Models"]:
                m = out[cname]["Models"][mk]
                extra = f" | no_s2010 Pool={m['No_s2010_PooledRec_%']}% ({m['No_s2010_TP']}/{m['No_s2010_totF']}) MacF>=2={m['No_s2010_Fge2_MacroRec_%']}%" if cname == "ltstdb" else ""
                print(f"  {mk:36s}: TP={m['TP']:3d} | Mac={m['MacroRec_%']:5.2f}% | Mac(F>=2)={m['MacroRec_Fge2_%']:5.2f}% | Pool={m['PooledRec_%']:5.2f}% {m['PooledRec_CI95']} | PoolFAPH={m['PooledFAPH_All']:.3f}{extra}")


if __name__ == "__main__":
    main()
