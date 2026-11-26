r"""
lab_v63c_round2_diagnostics.py
==============================
Supplementary diagnostic checks (checkpoint parity, Holm family, and active-anchor summaries):
  1. Replicates counter-class RR timing asymmetry (f_RR ROC AUC for F vs V and F vs N/S,
     with median, IQR, 10,000-resample subject bootstrap 95% CI, and beat-pooled AUC)
     across ALL THREE COHORTS: 6-DB Source, External Holter LTSTDB, External ICU MGHDB.
  2. Computes beat-pooled recall, leave-s2010-out pooled recall, and paired subject-cluster
     bootstrap 95% CI vs Full A3 for ALL Stage-1-gated baselines & Minimal Core 4D on LTSTDB,
     plus exact post-hoc 4-test Holm-Bonferroni family using GATED Single-Pole N!
  3. Inspects headers, clinical rhythm notes, and prefix beat-symbol composition for all 11
     high-eta_0 MGHDB K=0 F+ recordings to identify exact physiological/signal causes rather
     than guessing "sustained corruption".
  4. Checks PDF footer page numbering on paper1_bspc_main.pdf.

Outputs: audits/V63C_ROUND2_TIMING_GATED_ICU_20261125.json
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
from collections import Counter
from pathlib import Path
import numpy as np
import pypdf
import wfdb
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

ROOT = _REPO_ROOT
AUD_OUT = ROOT / "audits" / "V63C_ROUND2_TIMING_GATED_ICU_20261125.json"
RNG = np.random.default_rng(20261125)


def boot_med_ci(vals, B=10000):
    a = np.asarray(vals, dtype=np.float64)
    idx = RNG.integers(0, len(a), size=(B, len(a)))
    meds = np.median(a[idx], axis=1)
    return {
        "n_units": len(a),
        "median": round(float(np.median(a)), 3),
        "ci95_median": [round(float(np.percentile(meds, 2.5)), 3), round(float(np.percentile(meds, 97.5)), 3)],
        "iqr": [round(float(np.percentile(a, 25)), 3), round(float(np.percentile(a, 75)), 3)],
        "n_ge_060": int(np.sum(a >= 0.60)),
        "n_le_040": int(np.sum(a <= 0.40)),
    }


def eval_timing_auc(unit_dict, name):
    auc_fv, auc_fn = [], []
    all_y, all_f = [], []
    for pid, u in sorted(unit_dict.items()):
        y = np.concatenate(u["y"])
        f = np.concatenate(u["frr"])
        all_y.append(y)
        all_f.append(f)
        m_f = (y == 2)
        m_v = (y == 1)
        m_n = (y == 0) | (y == 3)
        if np.sum(m_f) >= 2 and np.sum(m_v) >= 2:
            yv = np.concatenate([np.ones(np.sum(m_f)), np.zeros(np.sum(m_v))])
            fv = np.concatenate([f[m_f], f[m_v]])
            auc_fv.append(roc_auc_score(yv, fv))
        if np.sum(m_f) >= 2 and np.sum(m_n) >= 2:
            yn = np.concatenate([np.ones(np.sum(m_f)), np.zeros(np.sum(m_n))])
            fn = np.concatenate([f[m_f], f[m_n]])
            auc_fn.append(roc_auc_score(yn, fn))
    ay = np.concatenate(all_y)
    af = np.concatenate(all_f)
    m_fv = (ay == 2) | (ay == 1)
    pool_fv = round(float(roc_auc_score((ay[m_fv] == 2).astype(int), af[m_fv])), 3)
    m_fn = (ay == 2) | (ay == 0) | (ay == 3)
    pool_fn = round(float(roc_auc_score((ay[m_fn] == 2).astype(int), af[m_fn])), 3)
    sv = boot_med_ci(auc_fv)
    sv["pooled_auc"] = pool_fv
    sn = boot_med_ci(auc_fn)
    sn["pooled_auc"] = pool_fn
    print(f"  {name:18s} | F:V (n={sv['n_units']:2d}): med={sv['median']:.3f} CI={sv['ci95_median']} IQR={sv['iqr']} pool={pool_fv:.3f} | F:N/S (n={sn['n_units']:2d}): med={sn['median']:.3f} CI={sn['ci95_median']} IQR={sn['iqr']} pool={pool_fn:.3f}")
    return {"F_vs_V": sv, "F_vs_NS": sn}


def holm_bonferroni(p_dict):
    items = sorted(p_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * float(p)))
        adj[k] = round(run, 5)
    return adj


def main():
    out = {}

    print("=== 1. TIMING ASYMMETRY (f_RR AUC) REPLICATION ACROSS ALL 3 COHORTS ===")
    timing = {}
    src_units = {}
    for db in ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]:
        recs = pickle.load(open((_DATA_ROOT / "cache_v62_source") / f"{db}_s2_variants.pkl", "rb"))
        for r in recs:
            pid = r["patient_id"]
            y = r["y"][r["Tier0"]["s2_mask"]]
            frr = r["Tier0"]["X29_s2"][:, 18]
            if len(y) > 0:
                u = src_units.setdefault(pid, {"y": [], "frr": []})
                u["y"].append(y)
                u["frr"].append(frr)
    timing["Source_6DB_S2"] = eval_timing_auc(src_units, "6-DB Source S2")

    for cname, reg_f, sid_k, cdir in [
        ("LTSTDB_Holter_S2", ROOT / "registry/target_cohort_ltstdb_v1.json", "subject_id", (_DATA_ROOT / "cache_v59_ltstdb_a3")),
        ("MGHDB_ICU_S2", ROOT / "registry/target_cohort_mghdb_v1.json", "subject_id_tier1", (_DATA_ROOT / "cache_v60_mghdb_a3")),
    ]:
        reg = json.loads(reg_f.read_text(encoding="utf-8"))["included_records"]
        ud = {}
        for r in reg:
            rid = r["rec_id"]
            z = np.load(cdir / f"{rid}.npz")
            y = z["y"][z["s2_mask"]]
            if len(y) > 0:
                frr = z["X29_s2"][:, 18]
                u = ud.setdefault(r[sid_k], {"y": [], "frr": []})
                u["y"].append(y)
                u["frr"].append(frr)
        timing[cname] = eval_timing_auc(ud, cname)
    out["Timing_Asymmetry_Replication"] = timing

    print("\n=== 2. GATED BASELINES POOLED + NO-S2010 + PAIRED CLUSTER CI + HONEST GATED HOLM ===")
    lt_reg = json.loads((ROOT / "registry/target_cohort_ltstdb_v1.json").read_text(encoding="utf-8"))["included_records"]
    ref_u = {}
    for fam in ("SVM", "LightGBM"):
        um = {}
        for r in lt_reg:
            rid, sid = r["rec_id"], r["subject_id"]
            d = pickle.load(open((_DATA_ROOT / "cache_v62_external/ltstdb") / f"{rid}_v62.pkl", "rb"))
            sc = d["scores"][f"{fam}__Full_A3"]
            u = um.setdefault(sid, {"tot_F": 0, "TP": 0})
            u["tot_F"] += int(d["tot_F"])
            u["TP"] += int(sc["TP"])
        ref_u[fam] = um

    gated_u = {}
    for r in lt_reg:
        rid, sid = r["rec_id"], r["subject_id"]
        d = pickle.load(open((_DATA_ROOT / "cache_v63b_ext/ltstdb") / f"{rid}_v63b.pkl", "rb"))
        for mk, sc in d["models"].items():
            u = gated_u.setdefault(mk, {}).setdefault(sid, {"tot_F": 0, "TP": 0})
            u["tot_F"] += int(d["tot_F"])
            u["TP"] += int(sc["TP"])

    subjs = sorted(ref_u["SVM"].keys())
    fpos = [s for s in subjs if ref_u["SVM"][s]["tot_F"] > 0]
    f_arr = np.array([ref_u["SVM"][s]["tot_F"] for s in fpos], dtype=np.float64)
    no_s = [s for s in fpos if s != "ltstdb_s2010"]
    f_ns = np.array([ref_u["SVM"][s]["tot_F"] for s in no_s], dtype=np.float64)
    idx = RNG.integers(0, len(fpos), size=(10000, len(fpos)))
    idx_ns = RNG.integers(0, len(no_s), size=(10000, len(no_s)))

    gated_stats = {}
    for mk in sorted(gated_u.keys()):
        fam = mk.split("__", 1)[0]
        tp_ref = np.array([ref_u[fam][s]["TP"] for s in fpos], dtype=np.float64)
        tp_m = np.array([gated_u[mk][s]["TP"] for s in fpos], dtype=np.float64)
        tp_ns = np.array([gated_u[mk][s]["TP"] for s in no_s], dtype=np.float64)
        tp_ref_ns = np.array([ref_u[fam][s]["TP"] for s in no_s], dtype=np.float64)
        s2010_tp = int(gated_u[mk]["ltstdb_s2010"]["TP"])

        d_pool = 100.0 * (tp_ref.sum() - tp_m.sum()) / f_arr.sum()
        db = 100.0 * (tp_ref[idx].sum(1) - tp_m[idx].sum(1)) / np.maximum(f_arr[idx].sum(1), 1.0)
        ci = [round(float(np.percentile(db, 2.5)), 2), round(float(np.percentile(db, 97.5)), 2)]

        d_pool_ns = 100.0 * (tp_ref_ns.sum() - tp_ns.sum()) / f_ns.sum()
        db_ns = 100.0 * (tp_ref_ns[idx_ns].sum(1) - tp_ns[idx_ns].sum(1)) / np.maximum(f_ns[idx_ns].sum(1), 1.0)
        ci_ns = [round(float(np.percentile(db_ns, 2.5)), 2), round(float(np.percentile(db_ns, 97.5)), 2)]

        gated_stats[mk] = {
            "TP": int(tp_m.sum()),
            "s2010_TP": s2010_tp,
            "No_s2010_TP": int(tp_ns.sum()),
            "No_s2010_PooledRec_%": round(100.0 * float(tp_ns.sum()) / float(f_ns.sum()), 2),
            "Delta_PooledRec_vs_A3_%": round(float(d_pool), 2),
            "Delta_PooledRec_CI95": ci,
            "Delta_No_s2010_PooledRec_vs_A3_%": round(float(d_pool_ns), 2),
            "Delta_No_s2010_PooledRec_CI95": ci_ns,
        }
        print(f"  {mk:32s}: TP={int(tp_m.sum()):3d} (s2010={s2010_tp:2d}) | No-s2010={100*tp_ns.sum()/f_ns.sum():5.2f}% ({int(tp_ns.sum())}/199) | dPool={d_pool:+6.2f}% CI={ci} | dPool_no2010={d_pool_ns:+6.2f}% CI={ci_ns}")
    out["LTSTDB_Gated_Pooled_Influence"] = gated_stats

    # Post-hoc 4-test Holm family using GATED Single-Pole N (unconfounded!)
    p_posthoc = {
        "A3_vs_C3R3_SVM_Recall": 0.125,
        "LGB_vs_SVM_Recall": 0.10938,
        "LGB_vs_SVM_0F_FAPH": 0.00049,
        "DualPoleLGB_vs_GatedSinglePoleN_Recall": 0.09229,
    }
    holm_posthoc = holm_bonferroni(p_posthoc)
    print("\n  Post-hoc 4-test Holm family (using GATED Single-Pole N!):", holm_posthoc)
    out["PostHoc_4Test_Holm_Gated"] = {"raw_p": p_posthoc, "holm_p": holm_posthoc}

    print("\n=== 3. FORENSIC INSPECTION OF ALL 11 HIGH-eta_0 MGHDB K=0 F+ RECORDS ===")
    high_ids = ["mgh026", "mgh042", "mgh217", "mgh190", "mgh219", "mgh213", "mgh216", "mgh212", "mgh045", "mgh049", "mgh047"]
    icu_inspect = {}
    for rid in high_ids:
        hea_p = (_DATA_ROOT / "mghdb") / f"{rid}.hea"
        d63 = pickle.load(open((_DATA_ROOT / "cache_v63b_ext/mghdb") / f"{rid}_v63b.pkl", "rb"))
        m60 = json.loads(((_DATA_ROOT / "cache_v60_mghdb_a3") / f"{rid}_meta.json").read_text(encoding="utf-8"))
        ann = wfdb.rdann(str((_DATA_ROOT / "mghdb") / rid), "ari")
        syms60 = dict(Counter(ann.symbol[:60]))
        aux60 = [a.strip() for a in ann.aux_note[:60] if a and a.strip()]
        hdr_lines = [l.strip() for l in hea_p.read_text(errors="replace").splitlines() if l.startswith("#")]
        ecg_hdr = [l for l in hdr_lines if any(w in l.lower() for w in ["ecg", "rhythm", "history", "comment", "notes", "diagnos", "fibril", "flutter", "block", "tachy", "brady", "paced", "vent"])]
        icu_inspect[rid] = {
            "tot_F": d63["tot_F"],
            "eta_0": round(d63["eta_0"], 2),
            "eta_mad": round(d63["eta_mad"], 2),
            "med_ny": round(d63["med_ny"], 2),
            "leads": m60["leads"],
            "first60_syms": syms60,
            "first60_aux": aux60[:5],
            "clinical_hdr": ecg_hdr[:5],
        }
        print(f"  {rid} (F={d63['tot_F']:2d}, eta0={d63['eta_0']:5.2f}, etaMAD={d63['eta_mad']:5.2f}, med={d63['med_ny']:5.2f}, leads={m60['leads']}):")
        print(f"      first60_syms={syms60} | aux={aux60[:5]}")
        print(f"      hdr={ecg_hdr[:3]}")
    out["MGHDB_11_HighEta0_Inspection"] = icu_inspect

    print("\n=== 4. CHECK PDF FOOTER PAGE NUMBERS ===")
    pdf = pypdf.PdfReader(str(_REPO_ROOT / "submission_bspc/paper1_bspc_main.pdf"))
    for pi in [0, 1, 2, len(pdf.pages) - 2, len(pdf.pages) - 1]:
        txt = pdf.pages[pi].extract_text()
        last_line = [l.strip() for l in txt.splitlines() if l.strip()][-1]
        print(f"  PDF page {pi+1}/{len(pdf.pages)} footer: {last_line}")

    AUD_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== SAVED {AUD_OUT} ===")


if __name__ == "__main__":
    main()
