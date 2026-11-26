r"""
lab_audit_v49_subject_unit_and_registry.py
==========================================
SUBJECT-UNIT VERIFICATION REGISTRY & SENSITIVITY AUDIT (ZERO NEW MODELS, ZERO NEW TUNING)

1. Builds & saves canonical hashed record->subject mapping registry for all 306 records across
   the 6 public PhysioNet databases (`./registry\subject_mapping_6db_v1.json`):
     - mitdb    : 44 non-paced records -> 43 verified subjects (201 + 202 merged; PhysioNet mitdb/1.0.0 & 202.hea)
     - incartdb : 75 records -> 32 verified subjects (patient 1..32 from .hea comments)
     - ltdb     : 7 records  -> 7 verified subjects (PhysioNet ltdb/1.0.0 & distinct .hea age/sex)
     - edb      : 90 records -> 79 verified subjects (7 same-subject groups verified from PhysioNet edb/1.0.0 & .hea:
                  {e0118,e0119,e0121,e0122}, {e0123,e0124,e0125,e0126}, {e0129,e0133}, {e0136,e0139},
                  {e0147,e0148}, {e0154,e0155}, {e0162,e0163}; 90 - 11 = 79)
     - sddb     : 12 .atr records -> 12 verified subjects (PhysioNet sddb/1.0.0 clinical table)
     - svdb     : 78 records -> 78 UNVERIFIED record-level IDs (0/78 .hea demographic comments; no published subject table)
   Total: 251 subject-grouped analysis units (173 verified subjects + 78 unverified SVDB record IDs),
   consisting of 66 F+ units and 185 zero-F units (vs old proxy mapping: 263 IDs = 67 F+ + 196 zero-F).

2. Quantifies exact sensitivity of frozen C1 / C3 / C5 under R1 (both SVM and LogR) to replacing
   OldProxy_263 with VerifiedSubj_251:
     - Checks exact full-source classifier weight invariance (max |Delta w|, expected 0.000e+00 since
       merged F+ subjects MIT-201+202 [3 F] and EDB-e0118..e0122 [3 F] are both << cap_pat_pos=100).
     - Measures exact shifts in 5-fold source-OOF thresholds (tau1*, tau_N*, tau_V*) and sig_N/sig_V
       once same-subject records are forced into the same OOF fold (with deterministic tie-break (-tot_F, subj_id)).
     - Measures exact shifts in deployed T1 metrics (TP, FP_NS, FP_V, T1_MacroSubj_Rec_%, T1_MeanSubj_FAPH_All/ZeroF,
       Spec_AllNonF_% vs Spec_ZeroFRecs_%), composite-score AP of S_min, and post-hoc descriptive Pat@F2.
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
import hashlib
import platform
from collections import Counter
import numpy as np
import scipy
import sklearn
import wfdb
import pandas as pd

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v43_hypB2_representation_vs_veto as v43
import lab_diagnostic_v46_two_branch_stage2_rr as v46
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_diagnostic_v48_stage2_operating_point_calibration as v48

AUD_DIR = str(_REPO_ROOT / "audits")
REG_DIR = str(_REPO_ROOT / "registry")
os.makedirs(AUD_DIR, exist_ok=True)
os.makedirs(REG_DIR, exist_ok=True)

DB_ORDER = v47.DB_ORDER
CFG_ORDER = v48.CFG_ORDER
CLF_FAMILIES = ["SVM", "LogR"]

# Official PhysioNet verified same-subject record groups
MIT_SAME_SUBJECT_GROUPS = [
    ("mitdb_subj_201_202", ["201", "202"], "PhysioNet mitdb/1.0.0 + 202.hea ('same analog tape as record 201')"),
]

EDB_SAME_SUBJECT_GROUPS = [
    ("edb_subj_e0118_e0122", ["e0118", "e0119", "e0121", "e0122"], "PhysioNet edb/1.0.0 + identical .hea (51M, Resting angina, Normal CA, ICR 7200)"),
    ("edb_subj_e0123_e0126", ["e0123", "e0124", "e0125", "e0126"], "PhysioNet edb/1.0.0 + identical .hea (58M, Resting angina, Anterior MI, LAD, AVR, ICR 7200)"),
    ("edb_subj_e0129_e0133", ["e0129", "e0133"], "PhysioNet edb/1.0.0 + identical .hea (48M, Mixed angina, RCA, nitrates+nifedipine)"),
    ("edb_subj_e0136_e0139", ["e0136", "e0139"], "PhysioNet edb/1.0.0 + identical .hea (47M, Mixed angina, RCA, nitrates+diltiazem, ICR 7200)"),
    ("edb_subj_e0147_e0148", ["e0147", "e0148"], "PhysioNet edb/1.0.0 + identical .hea (35M, Mixed angina, 3-vessel, Oxford Medilog 4-24)"),
    ("edb_subj_e0154_e0155", ["e0154", "e0155"], "PhysioNet edb/1.0.0 + identical .hea (51M, Resting angina, LAD, diltiazem+nitrates, ICR 7200)"),
    ("edb_subj_e0162_e0163", ["e0162", "e0163"], "PhysioNet edb/1.0.0 + identical .hea (66M, Effort angina, MI, HTN, 3-vessel, ICR 7200)"),
]

DB_PROVENANCE = {
    "mitdb": {"institution": "BIH_Boston_MIT", "status": "verified_physionet_header_and_doc", "cross_db_overlap_note": "Same institution pool (BIH Boston) as ltdb, sddb, svdb; cross-database subject non-overlap not independently verified."},
    "incartdb": {"institution": "StPetersburg_INCART", "status": "verified_physionet_header", "cross_db_overlap_note": "St. Petersburg Institute of Cardiological Technics; explicit patient 1..32 labels in .hea."},
    "ltdb": {"institution": "BIH_Boston_MIT", "status": "verified_physionet_doc_and_header", "cross_db_overlap_note": "Same institution pool (BIH Boston) as mitdb, sddb, svdb; 7 distinct age/sex records."},
    "edb": {"institution": "European_Multicenter_CNR_ESC", "status": "verified_physionet_doc_and_header", "cross_db_overlap_note": "European ST-T Database (90 records from 79 verified subjects; 7 multi-record groups)."},
    "sddb": {"institution": "BIH_Boston_MIT", "status": "verified_physionet_doc", "cross_db_overlap_note": "Same institution pool (BIH Boston); shares underlying subjects with vfdb (DO NOT use vfdb as external test!)."},
    "svdb": {"institution": "BIH_Boston_MIT", "status": "unverified_record_proxy", "cross_db_overlap_note": "0/78 .hea demographic comments and no published PhysioNet subject-ID table; 78 record-level IDs (subject uniqueness unverified)."},
}


def build_subject_registry(dbs):
    """Builds verified record->subject mapping for all 306 records and saves immutable JSON registry."""
    mit_lookup = {r: (sid, note) for sid, recs, note in MIT_SAME_SUBJECT_GROUPS for r in recs}
    edb_lookup = {r: (sid, note) for sid, recs, note in EDB_SAME_SUBJECT_GROUPS for r in recs}

    records_list = []
    db_counts = {}
    for db in DB_ORDER:
        prov = DB_PROVENANCE[db]
        old_ids = set()
        new_ids = set()
        old_fpos = set()
        new_fpos = set()
        for it in dbs[db]:
            rid = it["rec_id"]
            old_pid = it["patient_id"]
            tot_f = int(np.sum(it["y"] == 2))
            if db == "mitdb" and rid in mit_lookup:
                new_sid, group_note = mit_lookup[rid]
            elif db == "edb" and rid in edb_lookup:
                new_sid, group_note = edb_lookup[rid]
            else:
                new_sid = old_pid
                group_note = "1-to-1 record" if db != "incartdb" else "grouped via .hea patient tag"
            old_ids.add(old_pid); new_ids.add(new_sid)
            if tot_f > 0:
                old_fpos.add(old_pid); new_fpos.add(new_sid)
            records_list.append({
                "db": db, "rec_id": rid,
                "old_proxy_id": old_pid, "verified_subject_id": new_sid,
                "changed": (old_pid != new_sid),
                "verification_status": prov["status"],
                "institution_pool": prov["institution"],
                "post_beats": int(len(it["y"])), "post_F": tot_f,
                "post_hours": round(float(it["post_hours"]), 4),
                "mapping_note": group_note,
            })
        db_counts[db] = {
            "n_records": len(dbs[db]),
            "old_proxy_units": len(old_ids), "old_Fpos_units": len(old_fpos),
            "verified_units": len(new_ids), "verified_Fpos_units": len(new_fpos),
            "verification_status": prov["status"],
            "institution_pool": prov["institution"],
        }

    # Exact assertions on verified counts
    assert db_counts["mitdb"]["verified_units"] == 43
    assert db_counts["incartdb"]["verified_units"] == 32
    assert db_counts["ltdb"]["verified_units"] == 7
    assert db_counts["edb"]["verified_units"] == 79
    assert db_counts["sddb"]["verified_units"] == 12
    assert db_counts["svdb"]["verified_units"] == 78

    tot_old = sum(v["old_proxy_units"] for v in db_counts.values())
    tot_new = sum(v["verified_units"] for v in db_counts.values())
    tot_old_f = sum(v["old_Fpos_units"] for v in db_counts.values())
    tot_new_f = sum(v["verified_Fpos_units"] for v in db_counts.values())

    registry = {
        "schema_version": "subject_mapping_6db_v1",
        "created_date": "2026-11-22",
        "summary": {
            "total_records": len(records_list),
            "old_proxy_total_units": tot_old,
            "old_proxy_Fpos_units": tot_old_f,
            "old_proxy_ZeroF_units": tot_old - tot_old_f,
            "verified_total_units": tot_new,
            "verified_Fpos_units": tot_new_f,
            "verified_ZeroF_units": tot_new - tot_new_f,
            "verified_distinct_subjects_excl_svdb": tot_new - 78,
            "unverified_record_proxy_units_svdb": 78,
        },
        "db_provenance": DB_PROVENANCE,
        "db_counts": db_counts,
        "merged_groups": {
            "mitdb": [{"subject_id": s, "records": r, "source": n} for s, r, n in MIT_SAME_SUBJECT_GROUPS],
            "edb": [{"subject_id": s, "records": r, "source": n} for s, r, n in EDB_SAME_SUBJECT_GROUPS],
        },
        "records": records_list,
    }
    reg_path = os.path.join(REG_DIR, "subject_mapping_6db_v1.json")
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    reg_sha = hashlib.sha256(open(reg_path, "rb").read()).hexdigest()
    return registry, reg_path, reg_sha


def clone_dbs_with_mapping(dbs, rec_to_sid):
    out = {}
    for db in DB_ORDER:
        items = []
        for it in dbs[db]:
            c = dict(it)
            c["patient_id"] = rec_to_sid[(db, it["rec_id"])]
            items.append(c)
        out[db] = items
    return out


def fit_fold_deterministic(tr_items, mtype, cap_pat_pos=100, deterministic_tiebreak=True):
    """Same as v48.fit_fold_models_and_oof, with optional deterministic tie-break (-tot_F, patient_id)."""
    pat_map = {}
    for it in tr_items:
        pat_map.setdefault(it["patient_id"], []).append(it)
    if deterministic_tiebreak:
        pat_list = sorted(pat_map.keys(), key=lambda p: (-sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]), p))
    else:
        pat_list = sorted(pat_map.keys(), key=lambda p: -sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

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

        X1, y1, w1 = v46.build_branch_xy(tr_sub, v47.F2_IDX, {0, 1, 3}, cap_pat_pos=cap_pat_pos)
        XN, yN, wN = v46.build_branch_xy(tr_sub, v47.F2_IDX, {0, 3}, cap_pat_pos=cap_pat_pos)
        XV3, yV3, wV3 = v46.build_branch_xy(tr_sub, v47.IDX_C3, {1}, cap_pat_pos=cap_pat_pos)
        XV5, yV5, wV5 = v46.build_branch_xy(tr_sub, v47.IDX_C5_CAUSAL, {1}, cap_pat_pos=cap_pat_pos)
        if np.sum(y1 == 1) == 0 or np.sum(yN == 1) == 0 or np.sum(yV3 == 1) == 0:
            continue

        sc1 = v48.StandardScaler(); clf1 = v46.make_clf(mtype); clf1.fit(sc1.fit_transform(X1), y1, sample_weight=w1)
        scN = v48.StandardScaler(); clfN = v46.make_clf(mtype); clfN.fit(scN.fit_transform(XN), yN, sample_weight=wN)
        scV3 = v48.StandardScaler(); clfV3 = v46.make_clf(mtype); clfV3.fit(scV3.fit_transform(XV3), yV3, sample_weight=wV3)
        scV5 = v48.StandardScaler(); clfV5 = v46.make_clf(mtype); clfV5.fit(scV5.fit_transform(XV5), yV5, sample_weight=wV5)

        for te in te_sub:
            if len(te["X_s2"]) > 0:
                oof_s1.append(clf1.decision_function(sc1.transform(te["X_s2"][:, v47.F2_IDX])))
                oof_sN.append(clfN.decision_function(scN.transform(te["X_s2"][:, v47.F2_IDX])))
                oof_sV3.append(clfV3.decision_function(scV3.transform(te["X_s2"][:, v47.IDX_C3])))
                oof_sV5.append(clfV5.decision_function(scV5.transform(te["X_s2"][:, v47.IDX_C5_CAUSAL])))
                oof_y.append(te["y"][te["s2_mask"]].astype(np.int8))
                oof_pi.append(np.full(len(te["X_s2"]), pat_to_idx[te["patient_id"]], dtype=np.int32))

    X1a, y1a, w1a = v46.build_branch_xy(tr_items, v47.F2_IDX, {0, 1, 3}, cap_pat_pos=cap_pat_pos)
    sc1_f = v48.StandardScaler(); clf1_f = v46.make_clf(mtype); clf1_f.fit(sc1_f.fit_transform(X1a), y1a, sample_weight=w1a)
    XNa, yNa, wNa = v46.build_branch_xy(tr_items, v47.F2_IDX, {0, 3}, cap_pat_pos=cap_pat_pos)
    scN_f = v48.StandardScaler(); clfN_f = v46.make_clf(mtype); clfN_f.fit(scN_f.fit_transform(XNa), yNa, sample_weight=wNa)
    XV3a, yV3a, wV3a = v46.build_branch_xy(tr_items, v47.IDX_C3, {1}, cap_pat_pos=cap_pat_pos)
    scV3_f = v48.StandardScaler(); clfV3_f = v46.make_clf(mtype); clfV3_f.fit(scV3_f.fit_transform(XV3a), yV3a, sample_weight=wV3a)
    XV5a, yV5a, wV5a = v46.build_branch_xy(tr_items, v47.IDX_C5_CAUSAL, {1}, cap_pat_pos=cap_pat_pos)
    scV5_f = v48.StandardScaler(); clfV5_f = v46.make_clf(mtype); clfV5_f.fit(scV5_f.fit_transform(XV5a), yV5a, sample_weight=wV5a)

    s1_cat = np.concatenate(oof_s1); sN_cat = np.concatenate(oof_sN)
    sV3_cat = np.concatenate(oof_sV3); sV5_cat = np.concatenate(oof_sV5)
    return {
        "oof_s1": s1_cat, "oof_sN": sN_cat, "oof_sV3": sV3_cat, "oof_sV5": sV5_cat,
        "oof_y": np.concatenate(oof_y), "oof_pi": np.concatenate(oof_pi),
        "src_pat_hours": src_pat_hours, "src_pat_totF": src_pat_totF,
        "tot_b": tot_b, "tot_f": tot_f,
        "sig1": float(np.std(s1_cat)) + 1e-9, "sigN": float(np.std(sN_cat)) + 1e-9,
        "sigV3": float(np.std(sV3_cat)) + 1e-9, "sigV5": float(np.std(sV5_cat)) + 1e-9,
        "sc1": sc1_f, "clf1": clf1_f, "scN": scN_f, "clfN": clfN_f,
        "scV3": scV3_f, "clfV3": clfV3_f, "scV5": scV5_f, "clfV5": clfV5_f,
    }


def eval_with_split_spec(te_items, model_obj):
    """Extends v48.eval_target_v48 with SEPARATED Spec_AllNonF_% vs Spec_ZeroFRecs_%."""
    ev, pt1, prf2 = v48.eval_target_v48(te_items, model_obj)
    tot_nonf = sum(int(np.sum(it["y"] != 2)) for it in te_items)
    fp_all = ev["T1_FP_NS"] + ev["T1_FPV"]
    ev["T1_Spec_AllNonF_%"] = round(100.0 * (1.0 - fp_all / max(1, tot_nonf)), 4)
    ev["T1_Spec_ZeroFRecs_%"] = ev["T1_Spec_%"]
    return ev, pt1, prf2


def main():
    t0 = time.time()
    dbs_old = {db: v47.build_or_load_v47_db(db) for db in DB_ORDER}
    registry, reg_path, reg_sha = build_subject_registry(dbs_old)
    print(f"Saved registry: {reg_path} (SHA256[:16]={reg_sha[:16]})", flush=True)
    print("Registry summary:", registry["summary"], flush=True)

    rec_to_new = {(r["db"], r["rec_id"]): r["verified_subject_id"] for r in registry["records"]}
    dbs_new = clone_dbs_with_mapping(dbs_old, rec_to_new)

    audit = {
        "Registry_Path": reg_path,
        "Registry_SHA256": reg_sha,
        "Registry_Summary": registry["summary"],
        "Environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "wfdb": wfdb.__version__, "pandas": pd.__version__,
        },
        "Models": {},
    }

    for mtype in CLF_FAMILIES:
        print(f"\n==================== SUBJECT-UNIT SENSITIVITY AUDIT ({mtype}, R1, LOCKED K=4) ====================", flush=True)
        audit["Models"][mtype] = {"Max_Weight_Diff": {}, "ByDB": {}, "Global": {}}
        max_wdiff = {"clf1": 0.0, "clfN": 0.0, "clfV3": 0.0, "clfV5": 0.0}

        glob = {"OldProxy_263": {c: {"t1": {}, "rf2": {}, "nonf": 0, "fp": 0} for c in CFG_ORDER},
                "VerifiedSubj_251": {c: {"t1": {}, "rf2": {}, "nonf": 0, "fp": 0} for c in CFG_ORDER}}

        for tgt in DB_ORDER:
            tr_old = [it for db in DB_ORDER if db != tgt for it in dbs_old[db]]
            te_old = dbs_old[tgt]
            tr_new = [it for db in DB_ORDER if db != tgt for it in dbs_new[db]]
            te_new = dbs_new[tgt]

            b_old = fit_fold_deterministic(tr_old, mtype, deterministic_tiebreak=False)
            b_new = fit_fold_deterministic(tr_new, mtype, deterministic_tiebreak=True)

            # Verify full-source classifier weight invariance!
            for ck in ("clf1", "clfN", "clfV3", "clfV5"):
                d_w = float(np.max(np.abs(b_old[ck].coef_ - b_new[ck].coef_)))
                d_b = float(np.max(np.abs(b_old[ck].intercept_ - b_new[ck].intercept_)))
                max_wdiff[ck] = max(max_wdiff[ck], d_w, d_b)

            c_old = v48.calibrate_all_rules(b_old)["R1_PooledPriorF1"]
            c_new = v48.calibrate_all_rules(b_new)["R1_PooledPriorF1"]

            db_rec = {"OldProxy_263": {}, "VerifiedSubj_251": {}, "Threshold_Delta": {}}
            tot_nonf_db = sum(int(np.sum(it["y"] != 2)) for it in te_new)

            print(f"\n  [{tgt}] ({mtype})", flush=True)
            for cname in CFG_ORDER:
                ev_o, pt1_o, prf2_o = eval_with_split_spec(te_old, c_old[cname])
                ev_n, pt1_n, prf2_n = eval_with_split_spec(te_new, c_new[cname])
                db_rec["OldProxy_263"][cname] = ev_o
                db_rec["VerifiedSubj_251"][cname] = ev_n
                db_rec["Threshold_Delta"][cname] = {"old": ev_o["Threshold"], "new": ev_n["Threshold"]}

                glob["OldProxy_263"][cname]["t1"].update(pt1_o); glob["OldProxy_263"][cname]["rf2"].update(prf2_o)
                glob["OldProxy_263"][cname]["nonf"] += tot_nonf_db; glob["OldProxy_263"][cname]["fp"] += (ev_o["T1_FP_NS"] + ev_o["T1_FPV"])
                glob["VerifiedSubj_251"][cname]["t1"].update(pt1_n); glob["VerifiedSubj_251"][cname]["rf2"].update(prf2_n)
                glob["VerifiedSubj_251"][cname]["nonf"] += tot_nonf_db; glob["VerifiedSubj_251"][cname]["fp"] += (ev_n["T1_FP_NS"] + ev_n["T1_FPV"])

                print(
                    f"    {cname:31s} | th: {str(ev_o['Threshold']):16s}->{str(ev_n['Threshold']):16s} | "
                    f"AP: {ev_o['AP']:.4f}->{ev_n['AP']:.4f} | "
                    f"TP: {ev_o['T1_TP']:3d}->{ev_n['T1_TP']:3d} FP_NS:{ev_o['T1_FP_NS']:4d}->{ev_n['T1_FP_NS']:4d} FP_V:{ev_o['T1_FPV']:4d}->{ev_n['T1_FPV']:4d} | "
                    f"PatRec:{ev_o['T1_MacroPat_Rec_%']:4.1f}->{ev_n['T1_MacroPat_Rec_%']:4.1f}% FAPH:{ev_o['T1_MeanPat_FAPH_All']:5.2f}->{ev_n['T1_MeanPat_FAPH_All']:5.2f}",
                    flush=True,
                )
            audit["Models"][mtype]["ByDB"][tgt] = db_rec

        audit["Models"][mtype]["Max_Weight_Diff"] = {k: float(f"{v:.3e}") for k, v in max_wdiff.items()}
        print(f"\n  Max Full-Source Classifier Weight Diff ({mtype}): {audit['Models'][mtype]['Max_Weight_Diff']}", flush=True)

        print(f"\n  === GLOBAL COMPARISON: OldProxy_263 (67 F+) vs VerifiedSubj_251 (66 F+) [{mtype}, R1] ===", flush=True)
        for mode_k in ("OldProxy_263", "VerifiedSubj_251"):
            audit["Models"][mtype]["Global"][mode_k] = {}
            for cname in CFG_ORDER:
                g = glob[mode_k][cname]
                pt1 = g["t1"]; prf2 = g["rf2"]
                fpos = [v["Rec_%"] for v in pt1.values() if v["tot_F"] > 0]
                all_fa = [v["FAPH"] for v in pt1.values()]
                zf_fa = [v["FAPH"] for v in pt1.values() if v["tot_F"] == 0]
                tp = sum(v["TP"] for v in pt1.values())
                fp_ns = sum(v["FP_NS"] for v in pt1.values())
                fp_v = sum(v["FP_V"] for v in pt1.values())
                tot_h = sum(v["hours"] for v in pt1.values())
                spec_all = round(100.0 * (1.0 - g["fp"] / max(1, g["nonf"])), 4)
                s = {
                    "n_units": len(pt1), "n_Fpos": len(fpos), "n_ZeroF": len(zf_fa),
                    "T1_TP": tp, "T1_FP_NS": fp_ns, "T1_FP_V": fp_v,
                    "T1_MacroSubj_Rec_%": round(float(np.mean(fpos)), 2),
                    "PostHoc_Pat@F2_%": round(float(np.mean(list(prf2.values()))), 2),
                    "T1_MeanSubj_FAPH_All": round(float(np.mean(all_fa)), 2),
                    "T1_MeanSubj_FAPH_ZeroF": round(float(np.mean(zf_fa)), 2),
                    "T1_Pooled_FAPH": round((fp_ns + fp_v) / max(1e-6, tot_h), 2),
                    "T1_FracUnits_LE2_%": round(100.0 * float(np.mean(np.array(all_fa) <= 2.0 + 1e-9)), 1),
                    "T1_FracZeroF_LE2_%": round(100.0 * float(np.mean(np.array(zf_fa) <= 2.0 + 1e-9)), 1),
                    "T1_Spec_AllNonF_%": spec_all,
                }
                audit["Models"][mtype]["Global"][mode_k][cname] = s
                print(
                    f"    [{mode_k:16s}] {cname:31s} | n={s['n_units']}({s['n_Fpos']}F+) | "
                    f"TP={tp:4d} NS={fp_ns:4d} V={fp_v:4d} | Rec={s['T1_MacroSubj_Rec_%']:5.2f}% Pat@F2={s['PostHoc_Pat@F2_%']:4.2f}% | "
                    f"FAPH(All/zF/Pool)={s['T1_MeanSubj_FAPH_All']:4.2f}/{s['T1_MeanSubj_FAPH_ZeroF']:4.2f}/{s['T1_Pooled_FAPH']:4.2f} "
                    f"SpecAllNonF={spec_all:.4f}%",
                    flush=True,
                )

        # Also check pairwise C5 vs C3 and C5 vs C1 WTL under VerifiedSubj_251!
        audit["Models"][mtype]["VerifiedSubj_251_Comparisons"] = {}
        for lbl, ca, cb in [
            ("C5_vs_C3", "C5_Causal_Unlabeled_PreQRS_Pwin", "C3_TwoBranch_FV_SignedRR"),
            ("C5_vs_C1", "C5_Causal_Unlabeled_PreQRS_Pwin", "C1_OneHead_Geom"),
            ("C3_vs_C1", "C3_TwoBranch_FV_SignedRR", "C1_OneHead_Geom"),
        ]:
            ma = glob["VerifiedSubj_251"][ca]["t1"]; mb = glob["VerifiedSubj_251"][cb]["t1"]
            ra = glob["VerifiedSubj_251"][ca]["rf2"]; rb = glob["VerifiedSubj_251"][cb]["rf2"]
            fpos_k = [p for p in ma if ma[p]["tot_F"] > 0]
            zf_k = [p for p in ma if ma[p]["tot_F"] == 0]
            w_r = sum(1 for p in fpos_k if ma[p]["Rec_%"] > mb[p]["Rec_%"] + 1e-6)
            l_r = sum(1 for p in fpos_k if ma[p]["Rec_%"] < mb[p]["Rec_%"] - 1e-6)
            t_r = len(fpos_k) - w_r - l_r
            dom_a = sum(1 for p in fpos_k if (ma[p]["Rec_%"] >= mb[p]["Rec_%"] - 1e-6 and ma[p]["FAPH"] <= mb[p]["FAPH"] + 1e-6) and (ma[p]["Rec_%"] > mb[p]["Rec_%"] + 1e-6 or ma[p]["FAPH"] < mb[p]["FAPH"] - 1e-6))
            dom_b = sum(1 for p in fpos_k if (mb[p]["Rec_%"] >= ma[p]["Rec_%"] - 1e-6 and mb[p]["FAPH"] <= ma[p]["FAPH"] + 1e-6) and (mb[p]["Rec_%"] > ma[p]["Rec_%"] + 1e-6 or mb[p]["FAPH"] < ma[p]["FAPH"] - 1e-6))
            zf_b = sum(1 for p in zf_k if ma[p]["FAPH"] < mb[p]["FAPH"] - 1e-6)
            zf_w = sum(1 for p in zf_k if ma[p]["FAPH"] > mb[p]["FAPH"] + 1e-6)
            w_f2 = sum(1 for p in ra if ra[p] > rb[p] + 1e-6)
            l_f2 = sum(1 for p in ra if ra[p] < rb[p] - 1e-6)
            t_f2 = len(ra) - w_f2 - l_f2
            res = {
                "T1_Rec_WTL_66Fpos": f"{w_r}W/{t_r}T/{l_r}L",
                "StrictDom_A_vs_B": f"{dom_a}_vs_{dom_b}",
                "ZeroF_FA_Better_vs_Worse_185": f"{zf_b}_vs_{zf_w}",
                "PostHoc_Pat@F2_WTL_66Fpos": f"{w_f2}W/{t_f2}T/{l_f2}L",
            }
            audit["Models"][mtype]["VerifiedSubj_251_Comparisons"][lbl] = res
            print(f"    [VerifiedSubj_251] {lbl:8s}: {res}", flush=True)

    out_json = os.path.join(AUD_DIR, "V49_SUBJECT_UNIT_SENSITIVITY_AUDIT_20261122.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    out_md = os.path.join(AUD_DIR, "V49_SUBJECT_UNIT_SENSITIVITY_AUDIT_20261122.md")
    lines = [
        "# v49 Subject-Unit Verification Registry & Mapping Sensitivity Audit (Zero New Models)",
        "",
        f"- **Canonical Registry**: `registry/subject_mapping_6db_v1.json` (`SHA256[:16] = {reg_sha[:16]}`)",
        "- **Analysis Units After Subject Verification**: **251** total units (**173** verified distinct subjects across `mitdb`, `incartdb`, `ltdb`, `edb`, `sddb` + **78** unverified record-level IDs in `svdb`), comprising **66 `F+`** and **185 zero-F** units (vs old proxy mapping: 263 IDs = 67 `F+` + 196 zero-F).",
        "- **Merged Same-Subject Groups**: `mitdb` (`201+202`, both `F+`, `3 F` total); `edb` (`7` groups = `11` extra records: `e0118..e0122` [contains `e0121` with `3 F`], `e0123..e0126`, `e0129+e0133`, `e0136+e0139`, `e0147+e0148`, `e0154+e0155`, `e0162+e0163`).",
        "",
    ]
    for mtype in CLF_FAMILIES:
        lines.extend([
            f"## 1. Global Sensitivity: `OldProxy_263` vs `VerifiedSubj_251` (`{mtype}`, `R1`)",
            f"- Full-source classifier weight invariance (`max |Delta w|`): `{audit['Models'][mtype]['Max_Weight_Diff']}`", "",
            "| Mapping | Model | Units (`F+`/`0-F`) | `T1` TP | `T1` FP-N/S | `T1` FP-V | `T1` Macro Rec (%) | `T1` Mean FAPH (All / 0-F / Pool) | `Spec_AllNonF` (%) | Post-hoc `Pat@F2` (%) |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
        ])
        for mk in ("OldProxy_263", "VerifiedSubj_251"):
            for cname in CFG_ORDER:
                s = audit["Models"][mtype]["Global"][mk][cname]
                lines.append(
                    f"| `{mk}` | `{cname}` | {s['n_units']} ({s['n_Fpos']}/{s['n_ZeroF']}) | {s['T1_TP']} | {s['T1_FP_NS']} | {s['T1_FP_V']} | "
                    f"**{s['T1_MacroSubj_Rec_%']:.2f}** | {s['T1_MeanSubj_FAPH_All']:.2f} / {s['T1_MeanSubj_FAPH_ZeroF']:.2f} / {s['T1_Pooled_FAPH']:.2f} | "
                    f"{s['T1_Spec_AllNonF_%']:.4f} | {s['PostHoc_Pat@F2_%']:.2f} |"
                )
        lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nALL DONE in {time.time()-t0:.1f}s!\nSaved: {out_json}\nSaved: {out_md}", flush=True)


if __name__ == "__main__":
    main()
