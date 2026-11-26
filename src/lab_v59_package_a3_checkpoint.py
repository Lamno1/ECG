r"""
lab_v59_package_a3_checkpoint.py
================================
PACKAGES AND SHA-256 SEALS THE SINGLE FROZEN 6-DATABASE SOURCE CHECKPOINT FOR LOCKED `A3`
(`A3_HPhiW_NS_Only` + `R3_B20`) ALONGSIDE EXACT `B0_C3_R1` AND `B0_C3_R3_B20` CONTROLS
BEFORE ANY EXTERNAL TARGET (`LTSTDB`) INFERENCE.

Strict Integrity & Parity Guarantees:
  - Source training corpus : All 6 public PhysioNet DBs (`mitdb`, `incartdb`, `ltdb`, `edb`,
                             `sddb`, `svdb`; 306 records / 251 `SubjectAware_251` units from
                             `registry/subject_mapping_6db_v1.json`, `SHA256[:16]=789c38b112c7dfd7`).
  - Features               : Exact 24-col `cache_v47_preqrs_causal_6db` + exact beat-by-beat
                             5 `A3` cols (`C_drift, log_tan_orth, tanh_d_qrs, tanh_d_stt,
                             log_qrs_vs_nonqrs`) from verified `cache_v58_exact_m3_6db`.
  - Primary model family   : `SVM`  (`LinearSVC`, C=0.3, dual='auto', max_iter=3000, random_state=42)
  - Supporting family      : `LogR` (`LogisticRegression`, C=0.3, max_iter=500, random_state=42)
  - Locked configurations  :
      1. `B0_C3_R1`           : Frozen v50 comparator (`10D F:N/S`, `11D F:V`, `R1`)
                                — asserts bit-exact `< 1e-12` parity vs v50 checkpoint `.pkl`!
      2. `B0_C3_R3_B20`       : Calibration-only control (`10D F:N/S`, `11D F:V`, `R3_B20`)
      3. `A3_HPhiW_NS_R3_B20` : LOCKED PRIMARY UPGRADE (`15D F:N/S`, locked `11D F:V`, `R3_B20`)
  - Calibration rule       : `R3_B20` (`solve_r3_joint2d_patfa`, `b_tot=2.0`, `b_0f=1e9`, `n_grid=85`)
                             on deterministic 5-fold subject-grouped source OOF sorted by
                             `(-tot_F, verified_subject_id)`, `cap_pat_pos=100`.
  - ZERO target (`LTSTDB`) data read, downloaded, or inspected by this script.
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
import platform
import numpy as np
import scipy
import sklearn
import wfdb
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v46_two_branch_stage2_rr as v46
import lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle as v47
import lab_diagnostic_v48_stage2_operating_point_calibration as v48
import lab_audit_v49_subject_unit_and_registry as v49
import lab_diagnostic_v56_qrs_stt_local_chord_lodo as v56

REG_PATH = str(_REPO_ROOT / "registry/subject_mapping_6db_v1.json")
CACHE_V58 = str(_DATA_ROOT / "cache_v58_exact_m3_6db")
V50_CKPT_PKL = str(_REPO_ROOT / "checkpoints/stage2_source6db_subjectaware251_r1_v1.pkl")
CKPT_DIR = str(_REPO_ROOT / "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

CKPT_PKL = os.path.join(CKPT_DIR, "stage2_source6db_subjectaware251_a3_r3b20_v2.pkl")
CKPT_MANIFEST = os.path.join(CKPT_DIR, "stage2_source6db_subjectaware251_a3_r3b20_v2_manifest.json")

DB_ORDER = list(v47.DB_ORDER)
CLF_FAMILIES = ["SVM", "LogR"]
EXPECTED_REG_SHA16 = "789c38b112c7dfd7"
EXPECTED_V50_CKPT_SHA256 = "a34ac3bc6ae84f53458a8124a27ca0445ad1df477ef85a7d5f7ac785790e9c5f"

F2_IDX = list(v47.F2_IDX)                       # [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]
IDX_C3_V = list(v47.IDX_C3)                     # F2_IDX + [18]
IDX_A3_N = F2_IDX + [24, 25, 26, 27, 28]        # 15D locked Branch F:N/S

CFG_SPEC = {
    "B0_C3_R1":           {"fN": F2_IDX,   "fV": IDX_C3_V, "rule": "R1"},
    "B0_C3_R3_B20":       {"fN": F2_IDX,   "fV": IDX_C3_V, "rule": "R3_B20"},
    "A3_HPhiW_NS_R3_B20": {"fN": IDX_A3_N, "fV": IDX_C3_V, "rule": "R3_B20"},
}
CFG_ORDER = list(CFG_SPEC.keys())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source_6db_x29(rec_to_sid):
    dbs_raw = {db: v47.build_or_load_v47_db(db) for db in DB_ORDER}
    dbs_base = v49.clone_dbs_with_mapping(dbs_raw, rec_to_sid)
    all_src = []
    for db in DB_ORDER:
        meta = json.load(open(os.path.join(CACHE_V58, f"{db}_meta.json"), "r", encoding="utf-8"))
        arr_m6 = np.load(os.path.join(CACHE_V58, f"{db}_m6.npz"))["m6"]
        m_by_rec = {m["rec_id"]: (arr_m6[m["i0"]:m["i1"], :5] if m["i1"] > m["i0"] else None) for m in meta}
        for it in dbs_base[db]:
            c = dict(it)
            if len(it["X_s2"]) > 0:
                a3_5 = m_by_rec[it["rec_id"]]
                assert a3_5 is not None and len(a3_5) == len(it["X_s2"]), f"v58 len mismatch on {db}/{it['rec_id']}"
                c["X_s2"] = np.column_stack([it["X_s2"].astype(np.float64), a3_5.astype(np.float64)])
            else:
                c["X_s2"] = np.zeros((0, 29), dtype=np.float64)
            all_src.append(c)
    return all_src


def fit_full_source_and_calibrate(all_src, mtype, cap_pat_pos=100):
    pat_map = {}
    for it in all_src:
        pat_map.setdefault(it["patient_id"], []).append(it)
    pat_list = sorted(pat_map.keys(), key=lambda p: (-sum(int(np.sum(it["y"] == 2)) for it in pat_map[p]), p))
    n_folds = min(5, len(pat_list))
    fold_of_pat = {p: i % n_folds for i, p in enumerate(pat_list)}

    pat_to_idx = {p: i for i, p in enumerate(pat_list)}
    n_src_pats = len(pat_list)
    src_pat_hours = np.zeros(n_src_pats, dtype=np.float64)
    src_pat_totF = np.zeros(n_src_pats, dtype=np.int64)
    for p, items in pat_map.items():
        pi = pat_to_idx[p]
        src_pat_hours[pi] = sum(float(it["post_hours"]) for it in items)
        src_pat_totF[pi] = sum(int(np.sum(it["y"] == 2)) for it in items)

    tot_b = sum(len(it["y"]) for it in all_src)
    tot_f = int(np.sum(src_pat_totF))

    uniq_N = sorted({tuple(v["fN"]) for v in CFG_SPEC.values()})
    uniq_V = sorted({tuple(v["fV"]) for v in CFG_SPEC.values()})

    oof_sN = {k: [] for k in uniq_N}
    oof_sV = {k: [] for k in uniq_V}
    oof_y, oof_pi = [], []

    for fid in range(n_folds):
        tr_sub = [it for it in all_src if fold_of_pat[it["patient_id"]] != fid]
        te_sub = [it for it in all_src if fold_of_pat[it["patient_id"]] == fid]

        clfs_N = {}
        for fset in uniq_N:
            XN, yN, wN = v46.build_branch_xy(tr_sub, list(fset), {0, 3}, cap_pat_pos=cap_pat_pos)
            sc = StandardScaler(); clf = v46.make_clf(mtype); clf.fit(sc.fit_transform(XN), yN, sample_weight=wN)
            clfs_N[fset] = (sc, clf)

        clfs_V = {}
        for fset in uniq_V:
            XV, yV, wV = v46.build_branch_xy(tr_sub, list(fset), {1}, cap_pat_pos=cap_pat_pos)
            sc = StandardScaler(); clf = v46.make_clf(mtype); clf.fit(sc.fit_transform(XV), yV, sample_weight=wV)
            clfs_V[fset] = (sc, clf)

        for te in te_sub:
            if len(te["X_s2"]) > 0:
                for fset, (sc, clf) in clfs_N.items():
                    oof_sN[fset].append(clf.decision_function(sc.transform(te["X_s2"][:, list(fset)])))
                for fset, (sc, clf) in clfs_V.items():
                    oof_sV[fset].append(clf.decision_function(sc.transform(te["X_s2"][:, list(fset)])))
                oof_y.append(te["y"][te["s2_mask"]].astype(np.int8))
                oof_pi.append(np.full(len(te["X_s2"]), pat_to_idx[te["patient_id"]], dtype=np.int32))

    full_N, full_V, sig_N, sig_V = {}, {}, {}, {}
    for fset in uniq_N:
        XN, yN, wN = v46.build_branch_xy(all_src, list(fset), {0, 3}, cap_pat_pos=cap_pat_pos)
        sc = StandardScaler(); clf = v46.make_clf(mtype); clf.fit(sc.fit_transform(XN), yN, sample_weight=wN)
        full_N[fset] = (sc, clf)
        oof_sN[fset] = np.concatenate(oof_sN[fset])
        sig_N[fset] = float(np.std(oof_sN[fset])) + 1e-9

    for fset in uniq_V:
        XV, yV, wV = v46.build_branch_xy(all_src, list(fset), {1}, cap_pat_pos=cap_pat_pos)
        sc = StandardScaler(); clf = v46.make_clf(mtype); clf.fit(sc.fit_transform(XV), yV, sample_weight=wV)
        full_V[fset] = (sc, clf)
        oof_sV[fset] = np.concatenate(oof_sV[fset])
        sig_V[fset] = float(np.std(oof_sV[fset])) + 1e-9

    y_cat = np.concatenate(oof_y)
    y_bin = (y_cat == 2).astype(int)
    pi_cat = np.concatenate(oof_pi)

    out = {}
    for cname in CFG_ORDER:
        spec = CFG_SPEC[cname]
        fN = tuple(spec["fN"])
        fV = tuple(spec["fV"])
        sN = oof_sN[fN]
        sV = oof_sV[fV]

        if spec["rule"] == "R1":
            tau_N, tau_V, _ = v46.joint_oof_2d_threshold(sN, sV, y_bin, tot_f, tot_b, n_grid=140)
        elif spec["rule"] == "R3_B20":
            tau_N, tau_V = v56.solve_r3_joint2d_patfa(
                sN, sV, y_cat, pi_cat, src_pat_hours, src_pat_totF,
                b_tot=2.0, b_0f=1e9, n_grid=85,
            )
        else:
            raise ValueError(f"Unknown rule: {spec['rule']}")

        pred_oof = (sN >= tau_N) & (sV >= tau_V)
        src_stat = v48.summarize_source_oof_op(pred_oof, y_cat, pi_cat, src_pat_hours, src_pat_totF)
        if spec["rule"] == "R3_B20":
            assert src_stat["src_MeanPat_FAPH_All"] <= 2.0 + 1e-3, f"R3_B20 budget violated on {mtype}/{cname}: {src_stat}"

        out[cname] = {
            "mode": "twobranch",
            "rule": spec["rule"],
            "feat_idx_N": list(fN),
            "feat_idx_V": list(fV),
            "scN": full_N[fN][0],
            "clfN": full_N[fN][1],
            "tau_N": float(tau_N),
            "sig_N": float(sig_N[fN]),
            "scV": full_V[fV][0],
            "clfV": full_V[fV][1],
            "tau_V": float(tau_V),
            "sig_V": float(sig_V[fV]),
            "src_stat": src_stat,
        }
    return out


def assert_v50_b0_r1_parity(models_payload):
    v50_sha = sha256_file(V50_CKPT_PKL)
    assert v50_sha == EXPECTED_V50_CKPT_SHA256, f"v50 checkpoint SHA mismatch: {v50_sha}"
    with open(V50_CKPT_PKL, "rb") as f:
        v50_ckpt = pickle.load(f)

    for mtype in CLF_FAMILIES:
        new_obj = models_payload[mtype]["B0_C3_R1"]
        ref_obj = v50_ckpt["models"][mtype]["C3_TwoBranch_FV_SignedRR"]
        checks = {
            "tau_N": abs(new_obj["tau_N"] - ref_obj["tau_N"]),
            "tau_V": abs(new_obj["tau_V"] - ref_obj["tau_V"]),
            "sig_N": abs(new_obj["sig_N"] - ref_obj["sig_N"]),
            "sig_V": abs(new_obj["sig_V"] - ref_obj["sig_V"]),
            "clfN_coef": float(np.max(np.abs(new_obj["clfN"].coef_ - ref_obj["clfN"].coef_))),
            "clfN_int": float(np.max(np.abs(new_obj["clfN"].intercept_ - ref_obj["clfN"].intercept_))),
            "clfV_coef": float(np.max(np.abs(new_obj["clfV"].coef_ - ref_obj["clfV"].coef_))),
            "clfV_int": float(np.max(np.abs(new_obj["clfV"].intercept_ - ref_obj["clfV"].intercept_))),
            "scN_mean": float(np.max(np.abs(new_obj["scN"].mean_ - ref_obj["scN"].mean_))),
            "scV_mean": float(np.max(np.abs(new_obj["scV"].mean_ - ref_obj["scV"].mean_))),
        }
        max_err = max(checks.values())
        print(f"  [PARITY vs v50 C3] {mtype:4s} max_diff={max_err:.3e} | checks={ {k: f'{v:.1e}' for k, v in checks.items()} }", flush=True)
        assert max_err < 1e-12, f"v50 C3 parity failed on {mtype}: {checks}"
        assert new_obj["src_stat"] == ref_obj["src_stat"], f"src_stat mismatch on {mtype}: {new_obj['src_stat']} != {ref_obj['src_stat']}"


def main():
    t0 = time.time()
    reg_sha = sha256_file(REG_PATH)
    assert reg_sha[:16] == EXPECTED_REG_SHA16, f"Registry SHA mismatch: {reg_sha[:16]}"
    registry = json.load(open(REG_PATH, "r", encoding="utf-8"))
    rec_to_sid = {(r["db"], r["rec_id"]): r["verified_subject_id"] for r in registry["records"]}

    all_src = load_source_6db_x29(rec_to_sid)
    n_recs = len(all_src)
    tot_beats = sum(len(it["y"]) for it in all_src)
    tot_f = sum(int(np.sum(it["y"] == 2)) for it in all_src)
    tot_hours = sum(float(it["post_hours"]) for it in all_src)
    pat_f = {}
    for it in all_src:
        pat_f[it["patient_id"]] = pat_f.get(it["patient_id"], 0) + int(np.sum(it["y"] == 2))
    n_units = len(pat_f)
    n_fpos = sum(1 for v in pat_f.values() if v > 0)
    n_zerof = n_units - n_fpos

    assert n_recs == 306 and n_units == 251 and n_fpos == 66 and n_zerof == 185
    assert tot_beats == 2676362 and tot_f == 4600

    print(
        f"[v59] Verified 6-DB source corpus (X29): {n_recs} recs, {n_units} SubjectAware_251 units "
        f"({n_fpos} F+ / {n_zerof} 0-F), {tot_beats:,} beats, {tot_f:,} F, {tot_hours:.2f} h",
        flush=True,
    )

    models_payload = {}
    manifest_models = {}

    for mtype in CLF_FAMILIES:
        print(f"\n[v59] Fitting full 6-DB source checkpoint ({mtype}, SubjectAware_251, B0_R1 + B0_R3_B20 + A3_R3_B20)...", flush=True)
        fitted = fit_full_source_and_calibrate(all_src, mtype, cap_pat_pos=100)
        models_payload[mtype] = fitted

        m_info = {}
        for cname in CFG_ORDER:
            obj = fitted[cname]
            m_info[cname] = {
                "mode": "twobranch",
                "rule": obj["rule"],
                "feat_idx_N": list(obj["feat_idx_N"]),
                "feat_idx_V": list(obj["feat_idx_V"]),
                "tau_N": float(obj["tau_N"]),
                "sig_N": float(obj["sig_N"]),
                "tau_V": float(obj["tau_V"]),
                "sig_V": float(obj["sig_V"]),
                "scN_mean": [float(x) for x in obj["scN"].mean_],
                "scN_scale": [float(x) for x in obj["scN"].scale_],
                "clfN_coef": [float(x) for x in obj["clfN"].coef_.ravel()],
                "clfN_intercept": float(obj["clfN"].intercept_.ravel()[0]),
                "scV_mean": [float(x) for x in obj["scV"].mean_],
                "scV_scale": [float(x) for x in obj["scV"].scale_],
                "clfV_coef": [float(x) for x in obj["clfV"].coef_.ravel()],
                "clfV_intercept": float(obj["clfV"].intercept_.ravel()[0]),
                "source_oof_stat": obj["src_stat"],
            }
            st = obj["src_stat"]
            print(
                f"  {cname:22s} | tau_N={obj['tau_N']:+.6f} (sig={obj['sig_N']:.6f}), "
                f"tau_V={obj['tau_V']:+.6f} (sig={obj['sig_V']:.6f}) | "
                f"src_OOF: MacroRec={st['src_MacroPat_Rec_%']:5.2f}%, "
                f"FAPH={st['src_MeanPat_FAPH_All']:.3f} (0F={st['src_MeanPat_FAPH_ZeroF']:.3f}, <=2:{st['src_FracPats_LE2_%']:5.1f}%), "
                f"TP={st['src_TP']:4d}, FP_NS={st['src_FP_NS']:4d}, FP_V={st['src_FP_V']:4d}",
                flush=True,
            )
        manifest_models[mtype] = m_info

    print("\n[v59] Asserting bit-exact parity of B0_C3_R1 against sealed v50 checkpoint...", flush=True)
    assert_v50_b0_r1_parity(models_payload)

    code_hashes = {}
    for fn in [
        "stage2_frozen_pipeline_v2.py",
        "lab_v59_package_a3_checkpoint.py",
        "lab_diagnostic_v33_greek_multibeat_mit44_and_incart75.py",
        "lab_diagnostic_v46_two_branch_stage2_rr.py",
        "lab_diagnostic_v47_twobranch_preqrs_causal_vs_oracle.py",
        "lab_diagnostic_v48_stage2_operating_point_calibration.py",
        "lab_audit_v49_subject_unit_and_registry.py",
        "lab_diagnostic_v56_qrs_stt_local_chord_lodo.py",
        "lab_audit_v58_exact_beatwise_m3_and_ablation.py",
    ]:
        p = os.path.join(str(_REPO_ROOT), fn)
        code_hashes[fn] = sha256_file(p)

    meta = {
        "checkpoint_version": "stage2_source6db_subjectaware251_a3_r3b20_v2",
        "contract_version": "Stage2-A3-Prospective-LTSTDB-v2.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "locked_primary_config": "A3_HPhiW_NS_R3_B20",
        "primary_family": "SVM",
        "sensitivity_family": "LogR",
        "operating_point_rules": {
            "R1": "Pooled Prior-Adjusted F1 (pi_0=0.001, n_grid=140, exact v50 parity)",
            "R3_B20": "Joint-2D Patient-Level MacroRec under Source MeanPatFAPH_All <= 2.0 (b_tot=2.0, b_0f=1e9, n_grid=85)",
        },
        "source_corpus": {
            "databases": DB_ORDER,
            "n_records": n_recs,
            "n_units_SubjectAware_251": n_units,
            "n_Fpos_units": n_fpos,
            "n_ZeroF_units": n_zerof,
            "post_prefix_beats": tot_beats,
            "post_prefix_F": tot_f,
            "post_hours": round(tot_hours, 4),
            "source_label_map_version": "AAMI_Map_v1.0",
            "registry_path": REG_PATH,
            "registry_sha256": reg_sha,
            "v50_reference_ckpt_sha256": EXPECTED_V50_CKPT_SHA256,
        },
        "feature_schema": {
            "F2_IDX_10D": F2_IDX,
            "IDX_C3_V_11D": IDX_C3_V,
            "IDX_A3_N_15D": IDX_A3_N,
            "A3_new_col_names_24_to_28": [
                "C_drift",
                "log_tan_orth",
                "tanh_d_qrs",
                "tanh_d_stt",
                "log_qrs_vs_nonqrs",
            ],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "wfdb": wfdb.__version__,
            "pandas": pd.__version__,
        },
        "code_sha256": code_hashes,
    }

    payload = {"meta": meta, "models": models_payload}
    with open(CKPT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    pkl_sha = sha256_file(CKPT_PKL)

    manifest = dict(meta)
    manifest["checkpoint_pkl_path"] = CKPT_PKL
    manifest["checkpoint_pkl_sha256"] = pkl_sha
    manifest["checkpoint_pkl_bytes"] = os.path.getsize(CKPT_PKL)
    manifest["models_manifest"] = manifest_models
    with open(CKPT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    man_sha = sha256_file(CKPT_MANIFEST)

    print(f"\n[v59] FROZEN A3 CHECKPOINT SEALED ({time.time()-t0:.1f}s):", flush=True)
    print(f"  PKL     : {CKPT_PKL} ({os.path.getsize(CKPT_PKL):,} bytes) SHA256={pkl_sha}", flush=True)
    print(f"  MANIFEST: {CKPT_MANIFEST} SHA256={man_sha}", flush=True)


if __name__ == "__main__":
    main()
