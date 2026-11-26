r"""
lab_v60_mghdb_a3_supplementary_eval.py
======================================
ONE-SHOT SUPPLEMENTARY CROSS-DOMAIN ICU EVALUATION OF LOCKED `A3`
(`stage2_source6db_subjectaware251_a3_r3b20_v2.pkl`) ON `MGHDB` (`mghdb/1.0.0`).

Strict Pre-Registered Protocol (`v60_mghdb_supplementary_contract_v1.json`):
  1. Verifies SHA-256 of frozen v2 checkpoint (`596867f1...`), locked MGHDB registry
     (`target_cohort_mghdb_v1.json`), and locked contract BEFORE downloading any `.dat`.
  2. Downloads `.dat` files in bounded waves of 10 with official PhysioNet `SHA256SUMS.txt`
     verification, runs UNCHANGED `stage2_frozen_pipeline_v2.py` (`ann_ext="ari"`,
     locked 2-ECG channels from `.hea`, `AAMI_Map_v1.1`), caches compact `X29_s2` `.npz`,
     and deletes raw `.dat` immediately after verified caching.
  3. Evaluates locked `B0_C3_R1`, `B0_C3_R3_B20`, and `A3_HPhiW_NS_R3_B20` ONCE across
     SVM (primary) and LogR (control) with ZERO parameter or threshold retuning.
  4. Comparison Hierarchy (pre-registered):
       - Primary: SVM `A3_R3 vs B0_R3` at locked `T1` (`Tier1_Explicit` 7-group merge;
         `Tier2_Conservative` +11 suspected groups sensitivity).
       - Key End-to-End Secondary: SVM `A3_R3 vs B0_R1` at locked `T1`.
       - Supporting: LogR, Matched-FA curves (`S_min` & pure `m_N` under fixed `B0_R1` F:V gate),
         AP, Pat@F2, Active-Anchor (`K>0`), Stage-2 conditional recall (`s2_F > 0`),
         and `Excl mgh247+mgh046`.
  5. Mandatory 4-number ICU diagnostic accounting (Stage-1->Stage-2 F / total post-prefix F,
     TP, FP_NS + FP_V, evaluated hours) + full-denominator F+ recall (s2_F==0 subjects kept
     as 0.0% in primary recall, conditional Stage-2 recall reported separately) + subject-summed
     counts & hours before computing FAPH.
"""


import os as _os, sys as _sys
from pathlib import Path as _Path
_SRC_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_DATA_ROOT = _Path(_os.environ.get("ECG_DATA_ROOT", str(_REPO_ROOT / "data_public_ecg")))
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import wfdb.io._header as _wfdb_hdr

# Safe in-memory fallback for PhysioNet mgh025.hea typo ('31/05/199' missing final year digit);
# canonical .hea files on disk remain 100% unmodified and SHA-256 verified against PhysioNet.
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
import lab_v50_ltstdb_external_eval as v50
import lab_v59_ltstdb_a3_prospective_eval as v59
import stage2_frozen_pipeline_v2 as pipe2

CKPT_PKL = str(_REPO_ROOT / "checkpoints/stage2_source6db_subjectaware251_a3_r3b20_v2.pkl")
CKPT_MANIFEST = str(_REPO_ROOT / "checkpoints/stage2_source6db_subjectaware251_a3_r3b20_v2_manifest.json")
REG_PATH = str(_REPO_ROOT / "registry/target_cohort_mghdb_v1.json")
CONTRACT_PATH = str(_REPO_ROOT / "registry/v60_mghdb_supplementary_contract_v1.json")

RAW_DIR = str(_DATA_ROOT / "mghdb")
CACHE_V60 = str(_DATA_ROOT / "cache_v60_mghdb_a3")
AUD_DIR = str(_REPO_ROOT / "audits")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_V60, exist_ok=True)
os.makedirs(AUD_DIR, exist_ok=True)

BASE_URL = "https://physionet.org/files/mghdb/1.0.0/"
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


def load_physionet_sha256sums():
    p = os.path.join(RAW_DIR, "SHA256SUMS.txt")
    assert os.path.exists(p), f"Missing {p}; run build_mghdb_registry_and_contract_v1.py first!"
    sums = {}
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) >= 2:
                sums[parts[1]] = parts[0]
    return sums


def download_verified_file(fname, expected_sha):
    dst = os.path.join(RAW_DIR, fname)
    if os.path.exists(dst) and expected_sha and sha256_file(dst) == expected_sha:
        return dst
    tmp = dst + ".part"
    for k in range(6):
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
            if k == 5:
                raise RuntimeError(f"Failed downloading {fname}: {e}")
            time.sleep(2.0 * (k + 1))


def _extract_worker_mghdb(rec_meta):
    rec_id = rec_meta["rec_id"]
    sid1 = rec_meta["subject_id_tier1"]
    sid2 = rec_meta["subject_id_tier2"]
    ch = tuple(rec_meta["locked_channel_indices"])
    npz_p = os.path.join(CACHE_V60, f"{rec_id}.npz")
    meta_p = os.path.join(CACHE_V60, f"{rec_id}_meta.json")
    if os.path.exists(meta_p):
        m = json.load(open(meta_p, "r", encoding="utf-8"))
        if m.get("skipped_short_record") or os.path.exists(npz_p):
            return rec_id, "cached", 0.0

    rec_path = os.path.join(RAW_DIR, rec_id)
    res = pipe2.extract_record_features_v2(
        rec_path=rec_path,
        db_name="mghdb",
        rec_id=rec_id,
        subject_id=sid1,
        ann_ext="ari",
        channel_indices=ch,
        label_map_version="AAMI_Map_v1.1",
    )
    if res is None:
        meta_out = {
            "db": "mghdb",
            "rec_id": rec_id,
            "subject_id_tier1": sid1,
            "subject_id_tier2": sid2,
            "skipped_short_record": True,
            "skip_reason": "<=55 valid mapped beats in frozen pipeline v2",
            "locked_channel_indices": list(ch),
            "locked_leads": rec_meta["locked_leads"],
        }
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)
        return rec_id, "skipped_short", 0.0

    np.savez_compressed(
        npz_p,
        y=res["y"],
        s2_mask=res["s2_mask"],
        X29_s2=res["X29_s2"],
    )
    meta_out = {
        "db": "mghdb",
        "rec_id": rec_id,
        "subject_id_tier1": sid1,
        "subject_id_tier2": sid2,
        "skipped_short_record": False,
        "fs": res["fs"],
        "leads": res["leads"],
        "locked_channel_indices": list(ch),
        "post_hours": res["post_hours"],
        "K": res["K"],
        "label_map_version": res["label_map_version"],
        "buf_audit": res["buf_audit"],
        "accounting": res["accounting"],
        "proc_sec": res["proc_sec"],
    }
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    return rec_id, "extracted", res["proc_sec"]


def prepare_all_mghdb_v2_features(inc_records):
    sha_map = load_physionet_sha256sums()
    todo = []
    for r in inc_records:
        rid = r["rec_id"]
        mp = os.path.join(CACHE_V60, f"{rid}_meta.json")
        npzp = os.path.join(CACHE_V60, f"{rid}.npz")
        if os.path.exists(mp):
            m = json.load(open(mp, "r", encoding="utf-8"))
            if m.get("skipped_short_record") or os.path.exists(npzp):
                continue
        todo.append(r)

    print(f"[v60] MGHDB X29 feature prep: {len(inc_records) - len(todo)} cached, {len(todo)} remaining.", flush=True)
    wave_size = 10
    done_cnt = len(inc_records) - len(todo)
    t_start = time.time()

    for wi in range(0, len(todo), wave_size):
        wave = todo[wi : wi + wave_size]
        t_w = time.time()
        fnames = []
        for r in wave:
            rid = r["rec_id"]
            fnames.extend([f"{rid}.hea", f"{rid}.ari", f"{rid}.dat"])

        def _dl(fn):
            return download_verified_file(fn, sha_map.get(fn))

        with ThreadPoolExecutor(max_workers=4) as dex:
            list(dex.map(_dl, fnames))
        t_dl = time.time() - t_w

        t_ex0 = time.time()
        with ProcessPoolExecutor(max_workers=4) as pex:
            res_list = list(pex.map(_extract_worker_mghdb, wave))
        t_ex = time.time() - t_ex0
        done_cnt += len(wave)

        # Remove raw .dat immediately after verified .npz cache (keep .hea & .ari)
        for r in wave:
            dat_p = os.path.join(RAW_DIR, f"{r['rec_id']}.dat")
            if os.path.exists(dat_p):
                os.remove(dat_p)

        free_gb = shutil.disk_usage("d:\\").free / 1e9
        sk = [x[0] for x in res_list if x[1] == "skipped_short"]
        print(
            f"  Wave {wi // wave_size + 1:2d}/{(len(todo) + wave_size - 1) // wave_size:2d} | "
            f"progress={done_cnt:3d}/{len(inc_records)} | dl={t_dl:5.1f}s ext={t_ex:5.1f}s | "
            f"elapsed={time.time() - t_start:6.1f}s | D:free={free_gb:.1f}GB"
            + (f" | skipped_short={sk}" if sk else ""),
            flush=True,
        )


def load_cached_mghdb(inc_records):
    items = []
    skipped = []
    for r in inc_records:
        rid = r["rec_id"]
        meta = json.load(open(os.path.join(CACHE_V60, f"{rid}_meta.json"), "r", encoding="utf-8"))
        if meta.get("skipped_short_record"):
            skipped.append(meta)
            continue
        dat = np.load(os.path.join(CACHE_V60, f"{rid}.npz"))
        items.append({
            "db": "mghdb",
            "rec_id": rid,
            "patient_id": meta["subject_id_tier1"],
            "subject_id_tier1": meta["subject_id_tier1"],
            "subject_id_tier2": meta["subject_id_tier2"],
            "post_hours": float(meta["post_hours"]),
            "K": int(meta["K"]),
            "leads": meta["leads"],
            "buf_audit": meta["buf_audit"],
            "accounting": meta["accounting"],
            "y": dat["y"],
            "s2_mask": dat["s2_mask"],
            "X_s2": dat["X29_s2"].astype(np.float64),
        })
    return items, skipped


def clone_with_tier(items, tier_key):
    out = []
    for it in items:
        c = dict(it)
        c["patient_id"] = it[tier_key]
        out.append(c)
    return out


def compute_stage_and_cond_metrics(coh_items, pt1_by_cfg):
    """
    Computes the mandatory 4-number ICU diagnostic breakdown & conditional Stage-2 recall
    (cleanly distinguishing Stage-1 admission ceiling from Stage-2 ranking).
    """
    subj_acc = {}
    for it in coh_items:
        s = it["patient_id"]
        d = subj_acc.setdefault(s, {"recs": 0, "recs_Kgt0": 0, "max_K": 0, "post_beats": 0, "post_F": 0, "s2_beats": 0, "s2_F": 0, "hours": 0.0})
        d["recs"] += 1
        if it["K"] > 0:
            d["recs_Kgt0"] += 1
        d["max_K"] = max(d["max_K"], int(it["K"]))
        acc = it["accounting"]
        d["post_beats"] += int(acc["post_prefix_beats"])
        d["post_F"] += int(acc["post_prefix_F"])
        d["s2_beats"] += int(acc["s2_total"])
        d["s2_F"] += int(acc["s2_F"])
        d["hours"] += float(it["post_hours"])

    n_subjs = len(subj_acc)
    n_subjs_Kgt0 = sum(1 for d in subj_acc.values() if d["max_K"] > 0)
    fpos_subjs = [s for s, d in subj_acc.items() if d["post_F"] > 0]
    s2_fpos_subjs = [s for s, d in subj_acc.items() if d["s2_F"] > 0]
    tot_post_F = sum(d["post_F"] for d in subj_acc.values())
    tot_s2_F = sum(d["s2_F"] for d in subj_acc.values())
    tot_post_beats = sum(d["post_beats"] for d in subj_acc.values())
    tot_s2_beats = sum(d["s2_beats"] for d in subj_acc.values())
    tot_hours = sum(d["hours"] for d in subj_acc.values())

    cond_by_cfg = {}
    for cname, pt1 in pt1_by_cfg.items():
        tot_tp = sum(pt1[s]["TP"] for s in subj_acc)
        cond_macro = (
            100.0 * float(np.mean([pt1[s]["TP"] / subj_acc[s]["s2_F"] for s in s2_fpos_subjs]))
            if s2_fpos_subjs else 0.0
        )
        cond_pooled = 100.0 * tot_tp / max(1, tot_s2_F)
        fpos_faph = float(np.mean([pt1[s]["FAPH"] for s in fpos_subjs])) if fpos_subjs else 0.0
        s2_fpos_faph = float(np.mean([pt1[s]["FAPH"] for s in s2_fpos_subjs])) if s2_fpos_subjs else 0.0
        cond_by_cfg[cname] = {
            "Cond_Pooled_S2_Rec_%": round(cond_pooled, 2),
            "Cond_Macro_S2Fpos_Rec_%": round(cond_macro, 2),
            "MeanFAPH_Fpos": round(fpos_faph, 3),
            "MeanFAPH_S2Fpos": round(s2_fpos_faph, 3),
        }

    return {
        "n_records": len(coh_items),
        "n_records_Kgt0": sum(1 for it in coh_items if it["K"] > 0),
        "n_subjects": n_subjs,
        "n_subjects_Kgt0": n_subjs_Kgt0,
        "n_Fpos_subjects_total": len(fpos_subjs),
        "n_Fpos_subjects_with_s2F": len(s2_fpos_subjs),
        "n_Fpos_subjects_zero_s2F": len(fpos_subjs) - len(s2_fpos_subjs),
        "n_ZeroF_subjects": n_subjs - len(fpos_subjs),
        "eval_post_prefix_beats": tot_post_beats,
        "eval_post_prefix_F": tot_post_F,
        "eval_s2_total_beats": tot_s2_beats,
        "eval_s2_F_beats": tot_s2_F,
        "stage1_F_admission_%": round(100.0 * tot_s2_F / max(1, tot_post_F), 2),
        "stage1_beat_admission_%": round(100.0 * tot_s2_beats / max(1, tot_post_beats), 2),
        "eval_post_prefix_hours": round(tot_hours, 2),
        "Conditional_By_Config": cond_by_cfg,
    }


def write_markdown_report(results, out_md):
    sa = results["Sample_Accounting"]
    svm_t1 = results["Models"]["SVM"]["Cohorts"]["PRIMARY_Tier1_Explicit"]
    svm_t2 = results["Models"]["SVM"]["Cohorts"]["SENS_Tier2_Conservative"]
    svm_k = results["Models"]["SVM"]["Cohorts"]["DIAG_ActiveAnchor_Kgt0_Tier1"]
    logr_t1 = results["Models"]["LogR"]["Cohorts"]["PRIMARY_Tier1_Explicit"]

    lines = [
        "# Wave-3 Supplementary Cross-Domain ICU Evaluation (`v60`): Locked `A3` on `MGHDB`",
        "",
        f"- **Date**: `{results['Date']}`",
        f"- **Role**: Pre-designated supplementary cross-domain ICU/OR evaluation (zero parameter or threshold retuning; `.ari` annotation caveat applies).",
        f"- **Frozen v2 Checkpoint SHA-256**: `{results['Checkpoint_SHA256']}`",
        f"- **Locked Registry SHA-256**: `{results['Target_Registry_SHA256']}`",
        f"- **Locked Contract SHA-256**: `{results['Contract_SHA256']}`",
        f"- **Evaluated Cohort (Tier-1 Explicit)**: `{sa['n_records']}` records (`{sa['n_records_Kgt0']}` with $K>0$), `{sa['n_subjects']}` Tier-1 subjects (`{sa['n_Fpos_subjects_total']}` `F+` total, of which `{sa['n_Fpos_subjects_with_s2F']}` have `s2_F > 0` and `{sa['n_Fpos_subjects_zero_s2F']}` have `s2_F = 0`; `{sa['n_ZeroF_subjects']}` `0-F`), `{sa['eval_post_prefix_hours']:.2f}` hours, `{sa['eval_post_prefix_beats']:,}` beats, `{sa['eval_post_prefix_F']}` post-prefix `F` beats (`{sa['eval_s2_F_beats']}` [`{sa['stage1_F_admission_%']:.2f}%`] passing Stage 1).",
        "",
        "## 1. Mandatory 4-Number ICU Diagnostic & Primary Tier-1 Results (`SVM` + `LogR`)",
        "",
        "| Model | Config | S1→S2 F / Total F | T1 TP | T1 FP-N/S | T1 FP-V | Hours | Full MacroRec (`F+`) | Cond S2 MacroRec (`s2_F>0`) | Cond Pooled (`TP/s2_F`) | Mean FAPH (All / 0-F) | 0-F max (`<=2`) | AP | Pat@F2 | Match@2.0 (`S_min` / Pure `m_N`) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for mtype, coh in [("SVM", svm_t1), ("LogR", logr_t1)]:
        st = coh["Stage_Diagnostic"]
        mc_smin = coh["Matched_FA_Curves"]["S_min"]
        mc_pure = coh["Matched_FA_Curves"]["Pure_mN_Fixed_B0R1_FVgate"]
        for cname in CFG_ORDER:
            ev = coh["Configs"][cname]
            cd = st["Conditional_By_Config"][cname]
            zt = coh["ZeroF_Tail"][cname]
            m2 = mc_smin[cname]["AllFAPH"]["2.0"]
            if cname == "B0_C3_R1":
                pm2 = f"{mc_pure['B0_10D_under_B0R1_FVgate']['AllFAPH']['2.0']:.2f}%"
            elif cname == "A3_HPhiW_NS_R3_B20":
                pm2 = f"{mc_pure['A3_15D_under_B0R1_FVgate']['AllFAPH']['2.0']:.2f}%"
            else:
                pm2 = "—"
            lines.append(
                f"| {mtype} | `{cname}` | `{st['eval_s2_F_beats']}/{st['eval_post_prefix_F']}` ({st['stage1_F_admission_%']:.1f}%) "
                f"| {ev['T1_TP']} | {ev['T1_FP_NS']:,} | {ev['T1_FPV']} | {st['eval_post_prefix_hours']:.1f} "
                f"| {ev['T1_MacroPat_Rec_%']:.2f}% | {cd['Cond_Macro_S2Fpos_Rec_%']:.2f}% | {cd['Cond_Pooled_S2_Rec_%']:.2f}% "
                f"| {ev['T1_MeanPat_FAPH_All']:.3f} / {ev['T1_MeanPat_FAPH_ZeroF']:.3f} "
                f"| {zt['max']:.2f} ({zt['pct_le_2']:.1f}%) | {ev['AP']:.4f} | {ev['MacroPat_Pat@F2_%']:.2f}% | {m2:.2f}% / {pm2} |"
            )

    lines += [
        "",
        "## 2. Pre-Registered Comparisons & Tier-1 vs Tier-2 Bootstrap CIs (`B=10,000, seed=42`)",
        "",
        "| Cohort / Tier | Model | Pair | dMacroRec (95% CI, WTL) | dMeanFAPH_All (95% CI) | dMeanFAPH_0F (95% CI) | Strict Dom (A vs B) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, mtype, coh in [
        ("PRIMARY Tier-1 (7 explicit merges)", "SVM", svm_t1),
        ("SENS Tier-2 (+11 suspected merges)", "SVM", svm_t2),
        ("DIAG Active-Anchor K>0 (Tier-1)", "SVM", svm_k),
        ("PRIMARY Tier-1 (LogR control)", "LogR", logr_t1),
        ("SENS Tier-2 (LogR control)", "LogR", results["Models"]["LogR"]["Cohorts"]["SENS_Tier2_Conservative"]),
    ]:
        for pair in ("A3_R3_vs_B0_R3", "A3_R3_vs_B0_R1"):
            b = coh["Comparisons"][pair]["Bootstrap"]
            lines.append(
                f"| {label} | {mtype} | `{pair}` "
                f"| `{b['Delta_MacroRec_T1_pt_%']:+.2f}%` `{b['Delta_MacroRec_T1_CI95_%']}` ({b['Fpos_WTL_Rec']}) "
                f"| `{b['Delta_MeanFAPH_All_pt']:+.3f}` `{b['Delta_MeanFAPH_All_CI95']}` "
                f"| `{b['Delta_MeanFAPH_ZeroF_pt']:+.3f}` `{b['Delta_MeanFAPH_ZeroF_CI95']}` "
                f"| `{b['Fpos_StrictDom_A_vs_B']}` |"
            )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-ckpt-sha", default="596867f150e9e79c917c16f2b2f328b64ce02c9f56c6068302fb89ff0763bb7a")
    args = ap.parse_args()

    t0 = time.time()
    manifest = json.load(open(CKPT_MANIFEST, "r", encoding="utf-8"))
    ckpt_sha = sha256_file(CKPT_PKL)
    assert ckpt_sha == manifest["checkpoint_pkl_sha256"] == args.expected_ckpt_sha, f"Checkpoint SHA mismatch: {ckpt_sha}"

    contract = json.load(open(CONTRACT_PATH, "r", encoding="utf-8"))
    contract_sha = sha256_file(CONTRACT_PATH)
    reg_sha = sha256_file(REG_PATH)
    assert reg_sha == contract["target_registry_sha256"], f"Registry SHA mismatch vs contract!"
    assert ckpt_sha == contract["frozen_checkpoint_sha256"], f"Ckpt SHA mismatch vs contract!"

    reg = json.load(open(REG_PATH, "r", encoding="utf-8"))
    inc_records = reg["included_records"]
    print(f"[v60] VERIFIED FROZEN v2 CHECKPOINT SHA256: {ckpt_sha}", flush=True)
    print(f"[v60] VERIFIED MGHDB REGISTRY       SHA256: {reg_sha} ({len(inc_records)} eligible recs)", flush=True)
    print(f"[v60] VERIFIED MGHDB CONTRACT       SHA256: {contract_sha}", flush=True)

    with open(CKPT_PKL, "rb") as f:
        ckpt = pickle.load(f)

    # Bounded-wave download & extraction (UNCHANGED stage2_frozen_pipeline_v2.py)
    prepare_all_mghdb_v2_features(inc_records)
    items_t1, skipped = load_cached_mghdb(inc_records)
    items_t2 = clone_with_tier(items_t1, "subject_id_tier2")

    # Active-anchor subjects (Tier-1 subjects with >=1 recording having K>0)
    kgt0_subjs_t1 = {it["subject_id_tier1"] for it in items_t1 if it["K"] > 0}
    items_kgt0_t1 = [it for it in items_t1 if it["subject_id_tier1"] in kgt0_subjs_t1]

    # Exclude top F-burden explicit subject mgh247 + mgh046
    top_f_gid = next(it["subject_id_tier1"] for it in items_t1 if it["rec_id"] in ("mgh247", "mgh046"))
    items_excl_top_t1 = [it for it in items_t1 if it["subject_id_tier1"] != top_f_gid]

    cohorts = {
        "PRIMARY_Tier1_Explicit": items_t1,
        "SENS_Tier2_Conservative": items_t2,
        "DIAG_ActiveAnchor_Kgt0_Tier1": items_kgt0_t1,
        "SENS_Excl_mgh247_mgh046_Tier1": items_excl_top_t1,
    }

    # Global accounting (Tier-1)
    dummy_stage = compute_stage_and_cond_metrics(items_t1, {})
    print(f"\n[v60] Loaded {len(items_t1)} MGHDB recs ({len(skipped)} skipped <=55 beats) | "
          f"Tier1={dummy_stage['n_subjects']} subjs ({dummy_stage['n_Fpos_subjects_total']} F+, "
          f"{dummy_stage['n_Fpos_subjects_with_s2F']} s2_F+, {dummy_stage['n_subjects_Kgt0']} K>0) | "
          f"beats={dummy_stage['eval_post_prefix_beats']:,}, F={dummy_stage['eval_post_prefix_F']}, "
          f"s2_F={dummy_stage['eval_s2_F_beats']} ({dummy_stage['stage1_F_admission_%']}%), "
          f"hours={dummy_stage['eval_post_prefix_hours']:.2f}", flush=True)

    results = {
        "Audit_Version": "v60_mghdb_a3_supplementary_icu_eval",
        "Date": "2026-11-23",
        "Locked_Primary_Config": "A3_HPhiW_NS_R3_B20",
        "Primary_Comparison": "SVM A3_HPhiW_NS_R3_B20 vs B0_C3_R3_B20 at locked T1 (Tier1_Explicit)",
        "Checkpoint_Path": CKPT_PKL,
        "Checkpoint_SHA256": ckpt_sha,
        "Target_Registry_Path": REG_PATH,
        "Target_Registry_SHA256": reg_sha,
        "Contract_Path": CONTRACT_PATH,
        "Contract_SHA256": contract_sha,
        "Skipped_Short_Records": skipped,
        "Sample_Accounting": {k: v for k, v in dummy_stage.items() if k != "Conditional_By_Config"},
        "Models": {},
    }

    for mtype in CLF_FAMILIES:
        print(f"\n==================== MGHDB SUPPLEMENTARY ICU EVALUATION ({mtype}, FROZEN 6-DB v2) ====================", flush=True)
        m_res = {"Cohorts": {}, "PerSubject_Tier1": {}}

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
                c_out["ZeroF_Tail"][cname] = v59.zero_f_tail_audit(pt1)
                pt1_by_cfg[cname] = pt1
                prf2_by_cfg[cname] = prf2

            c_out["Stage_Diagnostic"] = compute_stage_and_cond_metrics(coh_items, pt1_by_cfg)
            if coh_name in ("PRIMARY_Tier1_Explicit", "DIAG_ActiveAnchor_Kgt0_Tier1"):
                c_out["Matched_FA_Curves"] = v59.compute_matched_fa_curves(coh_items, ckpt["models"][mtype])

            for cmp_name, cA, cB in [
                ("A3_R3_vs_B0_R3", "A3_HPhiW_NS_R3_B20", "B0_C3_R3_B20"),
                ("A3_R3_vs_B0_R1", "A3_HPhiW_NS_R3_B20", "B0_C3_R1"),
                ("B0_R3_vs_B0_R1", "B0_C3_R3_B20", "B0_C3_R1"),
            ]:
                boot = v50.paired_subject_bootstrap(pt1_by_cfg[cA], pt1_by_cfg[cB], B=BOOT_B, seed=BOOT_SEED)
                np_t = v59.paired_nonparam_tests(pt1_by_cfg[cA], pt1_by_cfg[cB], prf2_by_cfg[cA], prf2_by_cfg[cB])
                c_out["Comparisons"][cmp_name] = {"Bootstrap": boot, "NonParam": np_t}

            m_res["Cohorts"][coh_name] = c_out

            if coh_name == "PRIMARY_Tier1_Explicit":
                subj_meta = {}
                for it in coh_items:
                    s = it["patient_id"]
                    d = subj_meta.setdefault(s, {"recs": [], "max_K": 0, "s2_F": 0, "s2_total": 0})
                    d["recs"].append(it["rec_id"])
                    d["max_K"] = max(d["max_K"], int(it["K"]))
                    d["s2_F"] += int(it["accounting"]["s2_F"])
                    d["s2_total"] += int(it["accounting"]["s2_total"])
                for s in sorted(subj_meta.keys()):
                    row = dict(subj_meta[s])
                    for sk, cn in [("B0_R1", "B0_C3_R1"), ("B0_R3", "B0_C3_R3_B20"), ("A3_R3", "A3_HPhiW_NS_R3_B20")]:
                        row[sk] = pt1_by_cfg[cn][s]
                    m_res["PerSubject_Tier1"][s] = row

            # Console summary
            st = c_out["Stage_Diagnostic"]
            print(f"\n  [{coh_name}] (recs={st['n_records']} [K>0:{st['n_records_Kgt0']}], "
                  f"subjs={st['n_subjects']} [F+:{st['n_Fpos_subjects_total']}, s2_F+:{st['n_Fpos_subjects_with_s2F']}, K>0:{st['n_subjects_Kgt0']}], "
                  f"s2_F={st['eval_s2_F_beats']}/{st['eval_post_prefix_F']} [{st['stage1_F_admission_%']}%], h={st['eval_post_prefix_hours']:.1f})", flush=True)
            for cname in CFG_ORDER:
                ev = c_out["Configs"][cname]
                cd = st["Conditional_By_Config"][cname]
                zt = c_out["ZeroF_Tail"][cname]
                z_str = f"p95={zt['p95']:.2f} max={zt['max']:.2f} <=2:{zt['pct_le_2']:.1f}%" if zt.get("n_0F", 0) > 0 else "n/a"
                print(
                    f"    {cname:20s} | TP={ev['T1_TP']:3d} FP_NS={ev['T1_FP_NS']:5d} FP_V={ev['T1_FPV']:4d} | "
                    f"FullRec={ev['T1_MacroPat_Rec_%']:5.2f}% CondS2Rec={cd['Cond_Macro_S2Fpos_Rec_%']:5.2f}% (PoolS2={cd['Cond_Pooled_S2_Rec_%']:5.2f}%) | "
                    f"FAPH={ev['T1_MeanPat_FAPH_All']:6.3f} (0F={ev['T1_MeanPat_FAPH_ZeroF']:5.3f}, {z_str}) | "
                    f"AP={ev['AP']:.4f} Pat@F2={ev['MacroPat_Pat@F2_%']:5.2f}%",
                    flush=True,
                )
            for pair in ("A3_R3_vs_B0_R3", "A3_R3_vs_B0_R1"):
                b = c_out["Comparisons"][pair]["Bootstrap"]
                np_t = c_out["Comparisons"][pair]["NonParam"]
                print(
                    f"    {pair:16s} | dRec={b['Delta_MacroRec_T1_pt_%']:+.2f}% {b['Delta_MacroRec_T1_CI95_%']} "
                    f"({b['Fpos_WTL_Rec']} pW={np_t['T1_Rec_Fpos']['p_wilcoxon']:.4f}) | "
                    f"dFAPH_All={b['Delta_MeanFAPH_All_pt']:+.3f} {b['Delta_MeanFAPH_All_CI95']} | "
                    f"dFAPH_0F={b['Delta_MeanFAPH_ZeroF_pt']:+.3f} {b['Delta_MeanFAPH_ZeroF_CI95']} | "
                    f"SDom={b['Fpos_StrictDom_A_vs_B']}",
                    flush=True,
                )
            if "Matched_FA_Curves" in c_out:
                print("    Matched-FA Curves:", flush=True)
                for k, v in c_out["Matched_FA_Curves"]["S_min"].items():
                    print(f"      S_min {k:20s} | All={v['AllFAPH']} | 0F={v['ZeroFFAPH']}", flush=True)
                for k, v in c_out["Matched_FA_Curves"]["Pure_mN_Fixed_B0R1_FVgate"].items():
                    print(f"      Pure  {k:26s} | All={v['AllFAPH']} | 0F={v['ZeroFFAPH']}", flush=True)

        results["Models"][mtype] = m_res

    out_json = os.path.join(AUD_DIR, "V60_MGHDB_A3_SUPPLEMENTARY_ICU_20261123.json")
    out_md = os.path.join(AUD_DIR, "V60_MGHDB_A3_SUPPLEMENTARY_ICU_20261123.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_markdown_report(results, out_md)
    print(f"\n[v60] COMPLETE! Saved:\n  JSON: {out_json}\n  MD  : {out_md}\n  Elapsed: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
