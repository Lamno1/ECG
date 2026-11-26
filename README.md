# Patient-Specific Dual-Pole ECG Geometry for Ventricular Fusion Beat Detection

Reproducibility code, cohort registries, and locked model checkpoints for patient-specific dual-pole normal–ventricular ($N$–$V$) tangent ECG geometry and decoupled counter-class evaluation across eight PhysioNet databases (12.01 million beats, 534 analysis units).

> **Scope of this repository:** This repository contains exclusively the reproducible source code, locked model checkpoints, cryptographic checksums, and PhysioNet cohort registries required to verify and reproduce the computational system and evaluation metrics. Raw ECG recordings are not redistributed and must be obtained directly from PhysioNet.

---

## Repository Structure

```text
ECG/
├── README.md                 # Setup and end-to-end reproduction instructions
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── CHECKSUMS.sha256          # SHA-256 digests for all locked checkpoints & registries
├── verify_release.py         # Quick verification of checksums & locked checkpoints
├── checkpoints/              # Frozen 6-DB trained checkpoints & JSON manifests
├── registry/                 # Locked LTSTDB & MGHDB cohort/channel registries
└── src/                      # Feature engines, frozen pipelines & experiment scripts
```

---

## 1. Installation & Quick Verification (< 5 seconds)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Verify SHA-256 checksums, locked checkpoints, and module integrity
python verify_release.py
```

`verify_release.py` checks every entry in `CHECKSUMS.sha256`, loads the locked linear SVM, Logistic Regression, LightGBM, and neural checkpoints, prints the 28 locked linear $A_3$ parameters and thresholds, and validates compilation of all modules in `src/`.

---

## 2. Public PhysioNet Datasets

All eight ECG databases are publicly available from [PhysioNet](https://physionet.org/). Place downloaded databases under `./data_public_ecg/` (or point `ECG_DATA_ROOT` to your local directory):

- **Six source development databases (`SubjectAware_251`, 306 records, 251 units):**
  - `mitdb` (MIT-BIH Arrhythmia Database)
  - `incartdb` (St. Petersburg INCART 12-lead Arrhythmia Database)
  - `ltdb` (MIT-BIH Long-Term ECG Database)
  - `edb` (European ST-T Database)
  - `sddb` (MIT-BIH Sudden Cardiac Death Holter Database)
  - `svdb` (MIT-BIH Supraventricular Arrhythmia Database)
- **External Holter cohort (`ltstdb`, 70 subjects, 76 records):** `https://physionet.org/content/ltstdb/1.0.0/`
- **External critical-care cohort (`mghdb`, 213 Tier-1 units, 219 records):** `https://physionet.org/content/mghdb/1.0.0/`

Locked lead-selection rules, subject crosswalks, and exclusion records for external cohorts are stored in `registry/target_cohort_ltstdb_v1.json` and `registry/target_cohort_mghdb_v1.json`. Note that the external streaming scripts (`lab_v62b_external_stream_eval.py`, `lab_v63b_gated_baselines_and_minimal_geom.py`, `lab_v63d_mghdb_pediatric_adult_window.py`) can stream `.dat` files on-the-fly from PhysioNet with `SHA256SUMS.txt` verification and automatic cleanup.

### Optional: Upstream Neural Comparator Architectures (MIT-Licensed)

To re-run or re-train the neural comparators (`src/v61_sota_models.py`, `src/lab_v61a_lightgbm_and_train_sota_dl.py`, `src/lab_v61b_external_eval_ltstdb_mghdb.py`), clone the three pinned upstream repositories into `./external_repos/` (or set `ECG_EXTERNAL_REPOS_DIR`):

```bash
mkdir -p external_repos && cd external_repos
git clone https://github.com/PKUDigitalHealth/ECGFounder.git ECGFounder_04edac70 && (cd ECGFounder_04edac70 && git checkout 04edac70)
git clone https://github.com/luodhhh/ModernTCN.git ModernTCN_56a9a2c0 && (cd ModernTCN_56a9a2c0 && git checkout 56a9a2c0)
git clone https://github.com/DL4mHealth/Medformer.git Medformer_891f65b8 && (cd Medformer_891f65b8 && git checkout 891f65b8)
```

---

## 3. Reproducing Main Pipeline & Evaluation Experiments

Set `ECG_DATA_ROOT` if your PhysioNet directory is outside `./data_public_ecg`:

```bash
# Example (Linux/macOS): export ECG_DATA_ROOT=/path/to/data_public_ecg
# Example (PowerShell):  $env:ECG_DATA_ROOT="D:\path\to\data_public_ecg"
```

| Stage / Experiment | Command | Output / Purpose |
|---|---|---|
| **Feature-engine parity self-test** | `python src/v62_feature_engine.py` | Verifies bit-exact parity (`max diff < 1e-5`) across Tier 0–3 extractors and baselines on source records. |
| **6-DB nested LODO mechanistic ablation ($A_0$–$A_5$)** | `python src/lab_audit_v58_exact_beatwise_m3_and_ablation.py` | Reproduces nested 6-database source LODO (`SubjectAware_251`) and $R_1$/$R_3$ calibration comparison. |
| **Package & verify locked $A_3$ SVM/LogR checkpoint** | `python src/lab_v59_package_a3_checkpoint.py` | Fits and locks `checkpoints/stage2_source6db_subjectaware251_a3_r3b20_v2.pkl` strictly on the 6 source DBs. |
| **Locked Holter (LTSTDB) $A_3$ evaluation** | `python src/lab_v59_ltstdb_a3_prospective_eval.py` | Evaluates locked SVM/LogR checkpoints on 70 LTSTDB subjects (7.97M beats). |
| **Locked ICU (MGHDB) $A_3$ evaluation** | `python src/lab_v60_mghdb_a3_supplementary_eval.py` | Evaluates locked SVM/LogR checkpoints on 213 Tier-1 / 198 Tier-2 MGHDB units (1.37M beats). |
| **LightGBM-$A_3$ & 2024–2025 neural comparators** | `python src/lab_v61a_lightgbm_and_train_sota_dl.py`<br>`python src/lab_v61b_external_eval_ltstdb_mghdb.py` | Trains on 6 source DBs under $\overline{\mathrm{FAPH}}_{\mathrm{all}}^{\mathrm{src}} \le 2.0$ and evaluates on LTSTDB & MGHDB. |
| **Component ablations, 4-tier causality & baselines** | `python src/lab_v62a_source_lodo_and_lock.py`<br>`python src/lab_v62b_external_stream_eval.py`<br>`python src/lab_v62e_normal_pole_ablation.py` | Reproduces ungated baselines, leave-one-component-out ablations, single-pole ventricular ablation, and Tier 0–3 causality streaming. |
| **Stage-1-gated baselines, Minimal Core 4D & ICU MAD audit** | `python src/lab_v63b_gated_baselines_and_minimal_geom.py` | Evaluates Raw ECG, Traditional Morphology, and Single-Pole Normal strictly inside the frozen Stage-1 gate, plus Minimal Core 4D and ICU MAD-gating. |
| **MGHDB pediatric vs. adult & flank-masked 374D audit** | `python src/lab_v63d_mghdb_pediatric_adult_window.py` | Reproduces age-stratified MGHDB metrics, prefix RR/flank energy forensics, and RMS-equalized flank-masked 374D Stage-1 ablation. |
| **Multiplicity, 5-seed LightGBM, runtime & paired stats** | `python src/lab_v63a_metrics_influence_holm.py`<br>`python src/lab_v63c_round2_diagnostics.py`<br>`python src/lab_v63e_final_reviewer_closure.py` | Reproduces cluster-bootstrapped pooled recall, single-beat / `s2010` / `B`-beat sensitivity, Holm-Bonferroni tables, 5-seed LightGBM stability, and single-core latency. |

---

## 4. License

This code is released under the [MIT License](LICENSE). Third-party ECG recordings are governed by their respective PhysioNet licenses; upstream neural architectures are governed by their original MIT licenses.
