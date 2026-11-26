#!/usr/bin/env python3
"""
verify_release.py
=================
Quick reproducibility verification (< 5 seconds, no PhysioNet downloads required):
  1. Verifies SHA-256 digests of every file listed in CHECKSUMS.sha256.
  2. Verifies byte-compile integrity of all modules in src/.
  3. Loads the locked 6-DB checkpoints and reports key parameters & thresholds.
"""

import hashlib
import json
import pickle
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_checksums():
    chk_file = ROOT / "CHECKSUMS.sha256"
    lines = [ln.strip() for ln in chk_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        expected, rel = ln.split(None, 1)
        p = ROOT / rel.strip()
        assert p.exists(), f"Missing file: {rel}"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == expected, f"SHA-256 mismatch for {rel}: {got[:16]} != {expected[:16]}"
    print(f"[PASS] Verified {len(lines)}/{len(lines)} SHA-256 checksums in CHECKSUMS.sha256.")


def verify_compile():
    py_files = sorted((ROOT / "src").glob("*.py"))
    for p in py_files:
        py_compile.compile(str(p), doraise=True)
    print(f"[PASS] Byte-compiled {len(py_files)}/{len(py_files)} Python modules in src/.")


def verify_checkpoints():
    # 1. Locked SVM / LogR A3 checkpoint
    with open(ROOT / "checkpoints" / "stage2_source6db_subjectaware251_a3_r3b20_v2.pkl", "rb") as f:
        ckpt_svm = pickle.load(f)
    svm = ckpt_svm["models"]["SVM"]["A3_HPhiW_NS_R3_B20"]
    print(
        f"[PASS] Loaded A3 Linear SVM checkpoint: "
        f"dim(F:N/S)={len(svm['clfN'].coef_.ravel())}, dim(F:V)={len(svm['clfV'].coef_.ravel())}, "
        f"tau_N={svm['tau_N']:.4f}, sig_N={svm['sig_N']:.4f}, tau_V={svm['tau_V']:.4f}, sig_V={svm['sig_V']:.4f}"
    )

    # 2. Primary locked LightGBM A3 checkpoint
    with open(ROOT / "checkpoints" / "v61_lightgbm_a3_6db.pkl", "rb") as f:
        ckpt_lgb = pickle.load(f)
    print(
        f"[PASS] Loaded Primary LightGBM A3 checkpoint: "
        f"tau_N={ckpt_lgb['tau_N']:.4f}, sig_N={ckpt_lgb['sig_N']:.4f}, "
        f"tau_V={ckpt_lgb['tau_V']:.4f}, sig_V={ckpt_lgb['sig_V']:.4f}"
    )

    # 3. Ablation / causality / gated baseline checkpoints
    with open(ROOT / "checkpoints" / "v62_locked_models_6db.pkl", "rb") as f:
        m62 = pickle.load(f)
    with open(ROOT / "checkpoints" / "v63b_gated_and_minimal_models_6db.pkl", "rb") as f:
        m63b = pickle.load(f)
    print(f"[PASS] Loaded {len(m62)} ablation/causality models and {len(m63b)} Stage-1-gated/minimal models.")

    # 4. Registries summary
    lt = json.loads((ROOT / "registry" / "target_cohort_ltstdb_v1.json").read_text(encoding="utf-8"))
    mg = json.loads((ROOT / "registry" / "target_cohort_mghdb_v1.json").read_text(encoding="utf-8"))
    print(
        f"[PASS] Loaded external registries: LTSTDB ({len(lt['included_records'])} records), "
        f"MGHDB ({len(mg['included_records'])} records)."
    )


if __name__ == "__main__":
    verify_checksums()
    verify_compile()
    verify_checkpoints()
    print("\nAll reproducibility release checks PASSED.")
