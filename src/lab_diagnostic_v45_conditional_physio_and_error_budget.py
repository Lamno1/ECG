r"""
lab_diagnostic_v45_conditional_physio_and_error_budget.py
=========================================================
READ-ONLY MECHANISTIC AUDIT ON EXISTING 6-HOSPITAL CACHES (2.68M BEATS, 4,600 F, 67 F+ PATIENTS):
  PART I  : Overall Error Budget by BOTH # of F beats AND # of F+ patients (K=4 Static v36 baseline),
            pinpointing where the largest salvageable F population actually sits (E1 NoAnchor,
            E2 Stage-1 Rej, E3 In-Stage-2 V-dominated vs N/S-dominated, E4 Detected@F2).
  PART II : Conditional Physiological Separability WITHIN Geometrically Equivalent Beats:
            Using pure-geometry F2_RefFrame10D_SVM (ZERO RR, ZERO pre-QRS) out-of-database score s_geom,
            measures exact WITHIN-PATIENT caliper-matched concordance cAUC_p(X | |Delta s_geom| <= 0.25)
            for F vs V AND F vs N across all evaluable patients, checking:
              1. Cross-patient directional consistency (Consistent >0.60 / Neutral / Reversed <0.40)
                 rather than pooled AUC driven by a few easy records.
              2. Pre-QRS signal quality & artifact decomposition directly from cached waveforms:
                 - Normal-beat pre-QRS template consistency (identifies AF / noisy baselines)
                 - Sub-window split of PRE_QRS_SLICE (7:73):
                     * Early T-tail window  [7:35]  (-239 to -161 ms)
                     * Physiological P-win  [35:63] (-161 to -83 ms)
                     * QRS-onset edge       [63:73] (-83 to -56 ms, wide-QRS bleed check!)
                 - Prematurity-controlled check: cAUC_p(preqrs_Pwin | s_geom AND |Delta pre_rr| <= 0.08)
                   to separate true P-wave presence from previous-beat T-wave tail shift!
              3. F-vs-V vs F-vs-N directional conflict in G3's global linear weights.
              4. Strictly causal features (pre_rr_loc, neg_rr_dev, preqrs) separated from
                 1-beat-lookahead Holter features (post_rr_loc, pre_rr + post_rr).

ZERO new models trained beyond existing locked F2 & G3 LODO baselines; ZERO new gates or sweeps.
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
from collections import Counter
import numpy as np

sys.path.insert(0, str(_REPO_ROOT))
import lab_diagnostic_v43_hypB2_representation_vs_veto as v43

AUD_DIR = str(_REPO_ROOT / "audits")
CACHE_V36 = str(_DATA_ROOT / "cache_gen3_6db")
CACHE_V37 = str(_DATA_ROOT / "cache_v37_waveforms_6db")
os.makedirs(AUD_DIR, exist_ok=True)

DB_ORDER = ["mitdb", "incartdb", "ltdb", "edb", "sddb", "svdb"]
F2_IDX = [0, 1, 2, 3, 4, 5, 8, 10, 13, 14]                 # Pure geometry: ZERO RR, ZERO pre-QRS!
G3_IDX = list(range(12)) + [13, 14, 15, 16, 17]            # Includes preqrs_c (6) and rr_dev (7)

# Pre-QRS sub-windows inside 187-sample beat (R-peak at index 93, 360 Hz = 2.778 ms/sample)
SL_FULL = slice(7, 73)    # [-239 ms, -56 ms] exact v33 PRE_QRS_SLICE
SL_EARLY = slice(7, 35)   # [-239 ms, -161 ms] previous-beat T-wave tail / baseline zone
SL_PWIN = slice(35, 63)   # [-161 ms, -83 ms]  clean physiological sinus P-wave window
SL_QEDGE = slice(63, 73)  # [-83 ms, -56 ms]   QRS-onset edge (wide-QRS bleed zone)


def corr_rows_2lead(W, tmpl0, tmpl1):
    """Mean 2-lead Pearson correlation of (M, 2, L) segments against (L,) templates."""
    def _c1(X, t):
        tc = t - np.mean(t)
        tn = np.linalg.norm(tc) + 1e-12
        Xc = X - np.mean(X, axis=1, keepdims=True)
        Xn = np.linalg.norm(Xc, axis=1) + 1e-12
        return (Xc @ tc) / (Xn * tn)
    return 0.5 * (_c1(W[:, 0, :], tmpl0) + _c1(W[:, 1, :], tmpl1))


def load_aligned_db(db_name):
    """Loads v36 K=4 cache aligned 1-to-1 with v37 waveforms & rr3."""
    m36 = json.load(open(os.path.join(CACHE_V36, f"{db_name}_gen3_4cls_meta.json"), "r", encoding="utf-8"))
    d36 = np.load(os.path.join(CACHE_V36, f"{db_name}_gen3_4cls.npz"))
    m37 = json.load(open(os.path.join(CACHE_V37, f"{db_name}_meta.json"), "r", encoding="utf-8"))
    rr3_all = np.load(os.path.join(CACHE_V37, f"{db_name}_rr3.npy"), mmap_mode="r")
    bts_all = np.load(os.path.join(CACHE_V37, f"{db_name}_beats.npy"), mmap_mode="r")

    m37_map = {m["rec_id"]: m for m in m37}
    items = []
    for m in m36:
        rid = m["rec_id"]
        m7 = m37_map[rid]
        b0, b1 = m["b0"], m["b1"]
        N = b1 - b0
        assert (m7["b1"] - m7["b0"]) == N, f"Alignment mismatch on {db_name}/{rid}"
        y = d36["y_all"][b0:b1]
        ready = d36["ready_all"][b0:b1]
        s2_mask = d36["s2_all"][b0:b1]
        X_s2 = d36["X18_s2_all"][m["s0"]:m["s1"]]
        rr3 = np.array(rr3_all[m7["b0"]:m7["b1"]], dtype=np.float64)  # [rr_prev, pre_rr_loc, post_rr_loc]

        # Extract normalized center beat waveform for each beat (padded_beats[w0+1 : w0+1+N]!)
        w0 = m7["w0"]
        W_center = np.array(bts_all[w0 + 1 : w0 + 1 + N], dtype=np.float64)  # (N, 2, 187)
        assert len(W_center) == N

        # Compute clean normal-beat pre-QRS templates & patient-level pre-QRS signal quality metrics!
        # Use first 200 normal beats (or all normal beats if fewer) — wait: for strict causal local vs record quality,
        # use the patient's own normal beats (y==0) to measure true P-wave quality & consistency in that tape!
        n_idx = np.where(y == 0)[0]
        if len(n_idx) >= 10:
            # Sample up to 500 evenly spaced normal beats across the tape to assess clean record-wide P-wave SNR
            sub_n = n_idx[np.linspace(0, len(n_idx) - 1, min(500, len(n_idx)), dtype=int)]
            Wn = W_center[sub_n]
            # Templates in each sub-window (mean-centered per beat first, then median across N beats)
            def build_tmpl(sl):
                seg = Wn[:, :, sl] - np.mean(Wn[:, :, sl], axis=2, keepdims=True)
                t0 = np.median(seg[:, 0, :], axis=0)
                t1 = np.median(seg[:, 1, :], axis=0)
                c_n = corr_rows_2lead(Wn[:, :, sl], t0, t1)
                sig_norm = 0.5 * (np.linalg.norm(t0) + np.linalg.norm(t1))
                resid = 0.5 * (np.median(np.linalg.norm(seg[:, 0, :] - t0[None, :], axis=1)) +
                               np.median(np.linalg.norm(seg[:, 1, :] - t1[None, :], axis=1)))
                snr = float(sig_norm / (resid + 1e-6))
                return t0, t1, float(np.median(c_n)), snr

            t_full0, t_full1, n_cons_full, n_snr_full = build_tmpl(SL_FULL)
            t_early0, t_early1, n_cons_early, n_snr_early = build_tmpl(SL_EARLY)
            t_pwin0, t_pwin1, n_cons_pwin, n_snr_pwin = build_tmpl(SL_PWIN)
            t_qedge0, t_qedge1, n_cons_qedge, n_snr_qedge = build_tmpl(SL_QEDGE)

            # Compute pre-QRS correlations ONLY for Stage-2 beats (fast!)
            s2_idx = np.where(s2_mask)[0]
            if len(s2_idx) > 0:
                Ws2 = W_center[s2_idx]
                pq_full_w = corr_rows_2lead(Ws2[:, :, SL_FULL], t_full0, t_full1)
                pq_early_w = corr_rows_2lead(Ws2[:, :, SL_EARLY], t_early0, t_early1)
                pq_pwin_w = corr_rows_2lead(Ws2[:, :, SL_PWIN], t_pwin0, t_pwin1)
                pq_qedge_w = corr_rows_2lead(Ws2[:, :, SL_QEDGE], t_qedge0, t_qedge1)
                # Also energy ratio of P-window vs QRS-edge (bleed check)
                e_pwin = np.sum((Ws2[:, :, SL_PWIN] - np.mean(Ws2[:, :, SL_PWIN], axis=2, keepdims=True)) ** 2, axis=(1, 2))
                e_qedge = np.sum((Ws2[:, :, SL_QEDGE] - np.mean(Ws2[:, :, SL_QEDGE], axis=2, keepdims=True)) ** 2, axis=(1, 2))
                qedge_energy_ratio = e_qedge / (e_pwin + e_qedge + 1e-6)
            else:
                pq_full_w = pq_early_w = pq_pwin_w = pq_qedge_w = qedge_energy_ratio = np.zeros(0, dtype=np.float64)
        else:
            n_cons_full = n_snr_full = n_cons_pwin = n_snr_pwin = n_cons_qedge = n_snr_qedge = 0.0
            n_s2 = int(np.sum(s2_mask))
            pq_full_w = pq_early_w = pq_pwin_w = pq_qedge_w = qedge_energy_ratio = np.zeros(n_s2, dtype=np.float64)

        items.append({
            "db": db_name, "rec_id": rid, "patient_id": m["patient_id"],
            "post_hours": m["post_hours"], "K": m["K"],
            "y": y, "ready": ready, "s2_mask": s2_mask, "X_s2": X_s2,
            "rr3_s2": rr3[s2_mask],
            "pq_sub_s2": np.column_stack([pq_full_w, pq_early_w, pq_pwin_w, pq_qedge_w, qedge_energy_ratio]) if len(pq_full_w) else np.zeros((0, 5)),
            "pq_qual": {
                "n_cons_full": round(n_cons_full, 3), "n_snr_full": round(n_snr_full, 3),
                "n_cons_pwin": round(n_cons_pwin, 3), "n_snr_pwin": round(n_snr_pwin, 3),
                "n_cons_qedge": round(n_cons_qedge, 3),
            },
        })
    return items


def caliper_matched_concordance(s_pos, x_pos, s_neg, x_neg, caliper=0.25, rr_pos=None, rr_neg=None, rr_caliper=None):
    """
    Exact pairwise concordance P(x_pos > x_neg | |s_pos - s_neg| <= caliper)
    (with ties counted as 0.5, and optional secondary matching on |rr_pos - rr_neg| <= rr_caliper!).
    Returns: (cAUC, n_matched_pairs, n_pos_matched, n_neg_matched).
    """
    if len(s_pos) == 0 or len(s_neg) == 0:
        return None, 0, 0, 0
    # Subsample negatives deterministically only if > 4000 to keep exact pairwise matrix instant
    if len(s_neg) > 4000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(s_neg), 4000, replace=False)
        s_neg = s_neg[idx]; x_neg = x_neg[idx]
        if rr_neg is not None: rr_neg = rr_neg[idx]
    if len(s_pos) > 2000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(s_pos), 2000, replace=False)
        s_pos = s_pos[idx]; x_pos = x_pos[idx]
        if rr_pos is not None: rr_pos = rr_pos[idx]

    match = (np.abs(s_pos[:, None] - s_neg[None, :]) <= caliper)
    if rr_caliper is not None and rr_pos is not None and rr_neg is not None:
        match = match & (np.abs(rr_pos[:, None] - rr_neg[None, :]) <= rr_caliper)

    n_pairs = int(np.sum(match))
    if n_pairs < 10:
        return None, n_pairs, 0, 0
    n_p_m = int(np.sum(np.any(match, axis=1)))
    n_n_m = int(np.sum(np.any(match, axis=0)))
    if n_p_m < 2 or n_n_m < 2:
        return None, n_pairs, n_p_m, n_n_m

    pi, nj = np.where(match)
    dx = x_pos[pi] - x_neg[nj]
    conc = float(np.mean((dx > 0).astype(np.float64) + 0.5 * (dx == 0).astype(np.float64)))
    return conc, n_pairs, n_p_m, n_n_m


def main():
    t0 = time.time()
    print("=== Step 1: Loading Aligned v36 (K=4) + v37 (Waveforms/RR3) Caches Across 6 Hospitals ===", flush=True)
    dbs = {}
    for db in DB_ORDER:
        dbs[db] = load_aligned_db(db)
        n_f = sum(int(np.sum(it["y"] == 2)) for it in dbs[db])
        n_s2f = sum(int(np.sum((it["y"] == 2) & it["s2_mask"])) for it in dbs[db])
        print(f"  Loaded {db:8s}: {len(dbs[db]):2d} recs | Tot_F={n_f:4d} | S2_F={n_s2f:4d}", flush=True)

    print("\n=== Step 2: Computing Out-of-Database Pure-Geometry Score s_geom (F2_RefFrame10D_SVM) & G3 Reference ===", flush=True)
    g3_physio_weights = {}
    pat_records = {}  # patient_id -> aggregated patient dict

    for tgt in DB_ORDER:
        tr = [it for db in DB_ORDER if db != tgt for it in dbs[db]]
        sc_f2, clf_f2, tau_f2 = v43.fit_lodo_on_pool(tr, F2_IDX, "SVM", cap_pat_pos=100)
        sc_g3, clf_g3, tau_g3 = v43.fit_lodo_on_pool(tr, G3_IDX, "LogR", cap_pat_pos=100)
        w_g3 = clf_g3.coef_.ravel()
        # In G3_IDX = 0..11, 13..17: index 6 is preqrs_c, index 7 is rr_dev!
        g3_physio_weights[tgt] = {
            "w_preqrs_c": round(float(w_g3[6]), 3),
            "w_rr_dev": round(float(w_g3[7]), 3),
            "w_m_bridge": round(float(w_g3[0]), 3),
            "w_z_V": round(float(w_g3[2]), 3),
            "w_pyth_loc": round(float(w_g3[12]), 3),
        }

        # Score target DB
        db_hours = sum(it["post_hours"] for it in dbs[tgt])
        all_f2_sc, all_y = [], []
        rec_scored = []
        for it in dbs[tgt]:
            sf2 = np.full(len(it["y"]), -1e6, dtype=np.float64)
            sg3 = np.full(len(it["y"]), -1e6, dtype=np.float64)
            if len(it["X_s2"]) > 0:
                sf2[it["s2_mask"]] = clf_f2.decision_function(sc_f2.transform(it["X_s2"][:, F2_IDX].astype(np.float64)))
                sg3[it["s2_mask"]] = clf_g3.decision_function(sc_g3.transform(it["X_s2"][:, G3_IDX].astype(np.float64)))
            all_f2_sc.append(sf2); all_y.append(it["y"])
            rec_scored.append((it, sf2, sg3))

        # DB-level oracle threshold at FAPH <= 2.0 (for pure geometry F2)
        sc_cat = np.concatenate(all_f2_sc); y_cat = np.concatenate(all_y)
        valid = (sc_cat > -1e5)
        sv = sc_cat[valid]; yv = (y_cat[valid] == 2)
        ord_v = np.argsort(sv)[::-1]
        ok2 = np.where(np.cumsum(~yv[ord_v]) <= int(np.floor(2.0 * db_hours)))[0]
        tau_db_f2 = float(sv[ord_v[ok2[-1]]]) if len(ok2) > 0 else float(sv[ord_v[0]]) + 1.0

        for it, sf2, sg3 in rec_scored:
            p = it["patient_id"]
            pd = pat_records.setdefault(p, {
                "db": tgt, "patient_id": p, "recs": [], "hours": 0.0, "K_max": 0,
                "tot_F": 0, "E1_NoAnc": 0, "E2_S1Rej": 0, "S2_F": 0, "S2_V": 0, "S2_N": 0, "S2_S": 0,
                "E4_Det_DB_F2": 0, "E4_Det_T1": 0,
                "s2_y": [], "s2_sf2": [], "s2_sg3": [], "s2_X18": [], "s2_rr3": [], "s2_pq": [],
                "pq_cons_pwin": [], "pq_snr_pwin": [], "pq_cons_full": [],
            })
            y = it["y"]; rd = it["ready"]; s2 = it["s2_mask"]
            pd["recs"].append(it["rec_id"])
            pd["hours"] += it["post_hours"]
            pd["K_max"] = max(pd["K_max"], it["K"])
            pd["tot_F"] += int(np.sum(y == 2))
            pd["E1_NoAnc"] += int(np.sum((y == 2) & (~rd)))
            pd["E2_S1Rej"] += int(np.sum((y == 2) & rd & (~s2)))
            pd["S2_F"] += int(np.sum((y == 2) & s2))
            pd["S2_V"] += int(np.sum((y == 1) & s2))
            pd["S2_N"] += int(np.sum((y == 0) & s2))
            pd["S2_S"] += int(np.sum((y == 3) & s2))
            pd["E4_Det_DB_F2"] += int(np.sum((y == 2) & s2 & (sf2 >= tau_db_f2)))
            pd["E4_Det_T1"] += int(np.sum((y == 2) & s2 & (sf2 >= tau_f2)))
            if np.any(s2):
                pd["s2_y"].append(y[s2])
                pd["s2_sf2"].append(sf2[s2])
                pd["s2_sg3"].append(sg3[s2])
                pd["s2_X18"].append(it["X_s2"].astype(np.float64))
                pd["s2_rr3"].append(it["rr3_s2"])
                pd["s2_pq"].append(it["pq_sub_s2"])
            pd["pq_cons_pwin"].append(it["pq_qual"]["n_cons_pwin"])
            pd["pq_snr_pwin"].append(it["pq_qual"]["n_snr_pwin"])
            pd["pq_cons_full"].append(it["pq_qual"]["n_cons_full"])

    # =========================================================================
    # PART I: OVERALL ERROR BUDGET BY BOTH # OF F BEATS AND # OF F+ PATIENTS!
    # =========================================================================
    print("\n=== PART I: OVERALL ERROR BUDGET (K=4 BASELINE, PURE-GEOMETRY F2 SCORE) ===", flush=True)
    budget_by_db = {}
    pat_budget_rows = []
    for p, pd in pat_records.items():
        if pd["tot_F"] == 0:
            continue
        db = pd["db"]
        y_s2 = np.concatenate(pd["s2_y"]) if pd["s2_y"] else np.zeros(0, dtype=np.int8)
        sf2_s2 = np.concatenate(pd["s2_sf2"]) if pd["s2_sf2"] else np.zeros(0)
        sg3_s2 = np.concatenate(pd["s2_sg3"]) if pd["s2_sg3"] else np.zeros(0)

        # Patient-level FAPH<=2.0 budget on pure-geometry s_geom!
        max_fp_pat = int(np.floor(2.0 * pd["hours"]))
        e4_pat_f2 = 0
        comp_v = comp_n = comp_s = 0
        if len(y_s2) > 0 and pd["S2_F"] > 0:
            ord_p = np.argsort(sf2_s2)[::-1]
            yv = (y_s2[ord_p] == 2)
            okp = np.where(np.cumsum(~yv) <= max_fp_pat)[0]
            tau_p = float(sf2_s2[ord_p[okp[-1]]]) if len(okp) > 0 else float(sf2_s2[ord_p[0]]) + 1.0
            e4_pat_f2 = int(np.sum((y_s2 == 2) & (sf2_s2 >= tau_p)))
            # Who outranks the missed F beats in Stage 2? Look at non-F beats scoring >= median(missed_F)!
            missed_f_scores = sf2_s2[(y_s2 == 2) & (sf2_s2 < tau_p)]
            if len(missed_f_scores) > 0:
                ref_sc = float(np.median(missed_f_scores))
                comp_v = int(np.sum((y_s2 == 1) & (sf2_s2 >= ref_sc)))
                comp_n = int(np.sum((y_s2 == 0) & (sf2_s2 >= ref_sc)))
                comp_s = int(np.sum((y_s2 == 3) & (sf2_s2 >= ref_sc)))

        e3_pat = pd["S2_F"] - e4_pat_f2
        v_dom = (comp_v >= (comp_n + comp_s)) if (comp_v + comp_n + comp_s) > 0 else (pd["S2_V"] >= pd["S2_N"])

        # Dominant patient-level failure bottleneck (largest bucket of missed F for this patient!)
        b_counts = {
            "E1_NoAnc": pd["E1_NoAnc"],
            "E2_S1Rej": pd["E2_S1Rej"],
            "E3_S2_VDom": e3_pat if v_dom else 0,
            "E3_S2_NDom": e3_pat if (not v_dom) else 0,
            "E4_DetPatF2": e4_pat_f2,
        }
        missed_only = {k: v for k, v in b_counts.items() if k != "E4_DetPatF2"}
        dom_fail = max(missed_only, key=missed_only.get) if sum(missed_only.values()) > 0 else "Solved"

        row = {
            "db": db, "patient_id": p, "recs": pd["recs"], "hours": round(pd["hours"], 2),
            "tot_F": pd["tot_F"], "S2_F": pd["S2_F"], "S2_V": pd["S2_V"], "S2_N": pd["S2_N"],
            **b_counts,
            "E4_Det_DB_F2": pd["E4_Det_DB_F2"], "E4_Det_T1": pd["E4_Det_T1"],
            "comp_V": comp_v, "comp_N": comp_n, "comp_S": comp_s, "dom_fail": dom_fail,
            "pq_cons_pwin": round(float(np.mean(pd["pq_cons_pwin"])), 3),
            "pq_snr_pwin": round(float(np.mean(pd["pq_snr_pwin"])), 3),
        }
        pat_budget_rows.append(row)

        bd = budget_by_db.setdefault(db, Counter())
        bd["n_pats"] += 1
        bd["tot_F"] += pd["tot_F"]
        for k, v in b_counts.items():
            bd[f"F_{k}"] += v
            if v > 0:
                bd[f"PatsAny_{k}"] += 1
        bd[f"PatsDom_{dom_fail}"] += 1

    for db in DB_ORDER:
        bd = budget_by_db[db]
        print(
            f"  {db:8s} ({bd['n_pats']:2d} F+ pats, {bd['tot_F']:4d} F) | "
            f"E1_NoAnc={bd['F_E1_NoAnc']:3d}F({bd['PatsDom_E1_NoAnc']}p) | "
            f"E2_S1Rej={bd['F_E2_S1Rej']:4d}F({bd['PatsDom_E2_S1Rej']}p) | "
            f"E3_S2_VDom={bd['F_E3_S2_VDom']:4d}F({bd['PatsDom_E3_S2_VDom']}p) | "
            f"E3_S2_NDom={bd['F_E3_S2_NDom']:3d}F({bd['PatsDom_E3_S2_NDom']}p) | "
            f"E4_Det@PatF2={bd['F_E4_DetPatF2']:3d}F",
            flush=True,
        )

    # =========================================================================
    # PART II: WITHIN-PATIENT CONDITIONALLY MATCHED PHYSIOLOGY (|Delta s_geom| <= 0.25)
    # =========================================================================
    print("\n=== PART II: WITHIN-PATIENT CONDITIONED ON EQUIVALENT GEOMETRY (|Delta s_geom| <= 0.25) ===", flush=True)
    # Variables to test (orientation: higher value expected to favor F over V!):
    #   CAUSAL:
    #     1. pre_rr_loc      : signed pre-RR (F expected higher / less premature than V)
    #     2. neg_rr_dev      : -|pre_rr_loc - 1.0| (F expected closer to 1.0 than V)
    #     3. preqrs_v36_7_73 : static prefix pre-QRS corr [7:73] (exact X18[:, 6] in G3)
    #     4. preqrs_full     : tape-N-template pre-QRS corr [7:73]
    #     5. preqrs_pwin     : clean physiological P-wave window [35:63] (-161 to -83 ms)
    #     6. preqrs_early    : early T-tail window [7:35] (-239 to -161 ms)
    #     7. preqrs_qedge    : QRS-onset edge [63:73] (-83 to -56 ms, wide-QRS bleed check!)
    #     8. neg_qedge_eratio: -(QRS-edge energy fraction)
    #   1-BEAT LOOKAHEAD (NON-CAUSAL / HOLTER ONLY):
    #     9. neg_post_rr     : -post_rr_loc (V has longer compensatory pause -> F has lower post_rr)
    #    10. neg_rr_sum_dev  : -|pre_rr + post_rr - 2.0| (full compensatory pause check)
    #   PREMATURITY-CONTROLLED P-WAVE CHECK:
    #    11. preqrs_pwin__rr_matched : preqrs_pwin matched on BOTH |Delta s_geom|<=0.25 AND |Delta pre_rr|<=0.08!

    PHYS_KEYS = [
        ("pre_rr_loc",              "causal",    "F > V (less premature)"),
        ("neg_rr_dev",              "causal",    "F closer to RR=1.0"),
        ("preqrs_v36_7_73",         "causal",    "v36 prefix [7:73]"),
        ("preqrs_pwin_35_63",       "causal",    "clean P-win [-161,-83ms]"),
        ("preqrs_early_7_35",       "causal",    "early T-tail [-239,-161ms]"),
        ("preqrs_qedge_63_73",      "causal",    "QRS-onset edge [-83,-56ms]"),
        ("preqrs_pwin_RRmatched",   "causal",    "P-win | s_geom & pre_rr"),
        ("neg_post_rr_1beat",       "lookahead", "-post_rr (no comp. pause)"),
    ]

    per_pat_cond = []
    for p, pd in sorted(pat_records.items(), key=lambda kv: -kv[1]["S2_F"]):
        if pd["S2_F"] < 2:
            continue
        y_s2 = np.concatenate(pd["s2_y"])
        sf2 = np.concatenate(pd["s2_sf2"])
        X18 = np.vstack(pd["s2_X18"])
        rr3 = np.vstack(pd["s2_rr3"])
        pq = np.vstack(pd["s2_pq"])

        # Normalize s_geom within-DB? Keep raw SVM decision score (SD ~ 0.8-1.2); also check 0.10 caliper!
        feats = {
            "pre_rr_loc": rr3[:, 1],
            "neg_rr_dev": -X18[:, 7],
            "preqrs_v36_7_73": X18[:, 6],
            "preqrs_pwin_35_63": pq[:, 2],
            "preqrs_early_7_35": pq[:, 1],
            "preqrs_qedge_63_73": pq[:, 3],
            "preqrs_pwin_RRmatched": pq[:, 2],
            "neg_post_rr_1beat": -rr3[:, 2],
        }
        mF = (y_s2 == 2); mV = (y_s2 == 1); mN = (y_s2 == 0)

        entry = {
            "db": pd["db"], "patient_id": p, "recs": pd["recs"],
            "tot_F": pd["tot_F"], "S2_F": int(np.sum(mF)), "S2_V": int(np.sum(mV)), "S2_N": int(np.sum(mN)),
            "pq_cons_pwin": round(float(np.mean(pd["pq_cons_pwin"])), 3),
            "pq_snr_pwin": round(float(np.mean(pd["pq_snr_pwin"])), 3),
            "F_vs_V_cal025": {}, "F_vs_V_cal010": {}, "F_vs_N_cal025": {},
            "medians_FV_overlap": {},
        }

        # Check F vs V matched on s_geom (caliper 0.25 and 0.10) AND F vs N matched on s_geom (caliper 0.25)
        for fk, _, _ in PHYS_KEYS:
            x = feats[fk]
            rr_c = 0.08 if fk == "preqrs_pwin_RRmatched" else None
            c25, np25, nf25, nv25 = caliper_matched_concordance(
                sf2[mF], x[mF], sf2[mV], x[mV], caliper=0.25,
                rr_pos=rr3[mF, 1], rr_neg=rr3[mV, 1], rr_caliper=rr_c,
            )
            c10, np10, _, _ = caliper_matched_concordance(
                sf2[mF], x[mF], sf2[mV], x[mV], caliper=0.10,
                rr_pos=rr3[mF, 1], rr_neg=rr3[mV, 1], rr_caliper=rr_c,
            )
            cn25, npn25, _, _ = caliper_matched_concordance(
                sf2[mF], x[mF], sf2[mN], x[mN], caliper=0.25,
                rr_pos=rr3[mF, 1], rr_neg=rr3[mN, 1], rr_caliper=rr_c,
            )
            entry["F_vs_V_cal025"][fk] = {"cAUC": round(c25, 4) if c25 is not None else None, "pairs": np25, "nF": nf25, "nV": nv25}
            entry["F_vs_V_cal010"][fk] = {"cAUC": round(c10, 4) if c10 is not None else None, "pairs": np10}
            entry["F_vs_N_cal025"][fk] = {"cAUC": round(cn25, 4) if cn25 is not None else None, "pairs": npn25}

        # Also record exact medians on the caliper-0.25 matched subset for pre_rr, post_rr, preqrs_pwin!
        if np.sum(mF) >= 2 and np.sum(mV) >= 2:
            mm = (np.abs(sf2[mF][:, None] - sf2[mV][None, :]) <= 0.25)
            mf_ov = np.any(mm, axis=1); mv_ov = np.any(mm, axis=0)
            if np.sum(mf_ov) >= 2 and np.sum(mv_ov) >= 2:
                entry["medians_FV_overlap"] = {
                    "nF_ov": int(np.sum(mf_ov)), "nV_ov": int(np.sum(mv_ov)),
                    "F_pre_rr": round(float(np.median(rr3[mF, 1][mf_ov])), 3),
                    "V_pre_rr": round(float(np.median(rr3[mV, 1][mv_ov])), 3),
                    "F_pwin": round(float(np.median(pq[mF, 2][mf_ov])), 3),
                    "V_pwin": round(float(np.median(pq[mV, 2][mv_ov])), 3),
                    "F_v36pq": round(float(np.median(X18[mF, 6][mf_ov])), 3),
                    "V_v36pq": round(float(np.median(X18[mV, 6][mv_ov])), 3),
                }
        per_pat_cond.append(entry)

    # Aggregate cross-patient consistency across ALL evaluable patients AND stratified by P-wave quality!
    def summarize_feature(fk, block_key, min_f=1, pq_filter=None):
        vals = []
        weights_f = []
        by_db = {}
        for e in per_pat_cond:
            if pq_filter is not None and not pq_filter(e):
                continue
            r = e[block_key][fk]
            if r["cAUC"] is None:
                continue
            nf = r.get("nF", e["S2_F"])
            if nf < min_f:
                continue
            vals.append(r["cAUC"]); weights_f.append(nf)
            by_db.setdefault(e["db"], []).append(r["cAUC"])
        if not vals:
            return {"n_pats": 0}
        arr = np.array(vals); wf = np.array(weights_f, dtype=float)
        n_pos60 = int(np.sum(arr >= 0.60))
        n_pos55 = int(np.sum(arr >= 0.55))
        n_neu = int(np.sum((arr > 0.40) & (arr < 0.60)))
        n_rev40 = int(np.sum(arr <= 0.40))
        n_rev45 = int(np.sum(arr <= 0.45))
        # Also per-DB medians!
        db_meds = {db: f"{np.median(v):.2f} (n={len(v)})" for db, v in by_db.items()}
        return {
            "n_pats": len(vals),
            "tot_matched_F": int(np.sum(wf)),
            "pat_median_cAUC": round(float(np.median(arr)), 3),
            "pat_mean_cAUC": round(float(np.mean(arr)), 3),
            "pat_iqr": [round(float(np.percentile(arr, 25)), 3), round(float(np.percentile(arr, 75)), 3)],
            "F_weighted_mean_cAUC": round(float(np.sum(arr * wf) / np.sum(wf)), 3),
            "consistent_ge60": n_pos60, "consistent_ge55": n_pos55,
            "neutral_40_60": n_neu, "reversed_le40": n_rev40, "reversed_le45": n_rev45,
            "by_db_med": db_meds,
        }

    summary_fv_025 = {fk: summarize_feature(fk, "F_vs_V_cal025") for fk, _, _ in PHYS_KEYS}
    summary_fv_010 = {fk: summarize_feature(fk, "F_vs_V_cal010") for fk, _, _ in PHYS_KEYS}
    summary_fn_025 = {fk: summarize_feature(fk, "F_vs_N_cal025") for fk, _, _ in PHYS_KEYS}
    summary_fv_hiP = {fk: summarize_feature(fk, "F_vs_V_cal025", pq_filter=lambda e: e["pq_cons_pwin"] >= 0.40) for fk, _, _ in PHYS_KEYS}
    summary_fv_loP = {fk: summarize_feature(fk, "F_vs_V_cal025", pq_filter=lambda e: e["pq_cons_pwin"] < 0.40) for fk, _, _ in PHYS_KEYS}

    print("\n--- F vs V MATCHED ON EQUIVALENT GEOMETRY (|Delta s_geom| <= 0.25, PER-PATIENT CONSISTENCY) ---")
    for fk, caus, desc in PHYS_KEYS:
        s = summary_fv_025[fk]
        s10 = summary_fv_010[fk]
        sn = summary_fn_025[fk]
        if s["n_pats"] == 0: continue
        print(
            f"  {fk:22s} [{caus:9s}] | n={s['n_pats']:2d}p ({s['tot_matched_F']:4d}F) | "
            f"PatMed={s['pat_median_cAUC']:.3f} IQR={s['pat_iqr']} (cal0.10={s10.get('pat_median_cAUC')}) | "
            f">=0.60:{s['consistent_ge60']:2d} / neu:{s['neutral_40_60']:2d} / <=0.40:{s['reversed_le40']:2d} | "
            f"F_vs_N Med={sn.get('pat_median_cAUC')} (n={sn.get('n_pats',0)})"
        )
        print(f"      by_db: {s['by_db_med']}")

    print("\n--- PRE-QRS QUALITY STRATIFICATION (High normal P-consistency >= 0.40 vs Low < 0.40) ---")
    for fk in ("preqrs_v36_7_73", "preqrs_pwin_35_63", "preqrs_early_7_35", "preqrs_qedge_63_73", "preqrs_pwin_RRmatched", "pre_rr_loc", "neg_rr_dev"):
        hi = summary_fv_hiP[fk]; lo = summary_fv_loP[fk]
        print(f"  {fk:22s} | Hi-PQual(n={hi.get('n_pats',0):2d}): Med={hi.get('pat_median_cAUC')} (>=.60:{hi.get('consistent_ge60',0)}, <=.40:{hi.get('reversed_le40',0)}) | "
              f"Lo-PQual(n={lo.get('n_pats',0):2d}): Med={lo.get('pat_median_cAUC')} (>=.60:{lo.get('consistent_ge60',0)}, <=.40:{lo.get('reversed_le40',0)})")

    print("\n--- G3 LEARNED LODO WEIGHTS (CONFIRMING GLOBAL LINEAR SIGN) ---")
    for db, w in g3_physio_weights.items():
        print(f"  holdout {db:8s}: {w}")

    # Save full audited JSON + MD
    out_obj = {
        "Part1_Budget_By_DB": {db: dict(budget_by_db[db]) for db in DB_ORDER},
        "Part1_Budget_All67Pats": sorted(pat_budget_rows, key=lambda r: -r["tot_F"]),
        "Part2_Summary_FvsV_Cal025": summary_fv_025,
        "Part2_Summary_FvsV_Cal010": summary_fv_010,
        "Part2_Summary_FvsN_Cal025": summary_fn_025,
        "Part2_Summary_HiPQual": summary_fv_hiP,
        "Part2_Summary_LoPQual": summary_fv_loP,
        "Part2_G3_LODO_Weights": g3_physio_weights,
        "Part2_PerPatient_Details": per_pat_cond,
    }
    out_json = os.path.join(AUD_DIR, "V45_CONDITIONAL_PHYSIO_AND_ERROR_BUDGET_AUDIT_20261122.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    # Write structured Markdown report
    out_md = os.path.join(AUD_DIR, "V45_CONDITIONAL_PHYSIO_AND_ERROR_BUDGET_AUDIT_20261122.md")
    lines = [
        "# v45 Read-Only Audit: Overall Error Budget (Beats & Patients) + Conditional Physiological Separability at Equivalent Geometry",
        "",
        "- **Read-Only Scope**: Uses existing `K=4 Static v36` cache (`cache_gen3_6db`) aligned 1-to-1 with `v37` waveforms & 3-beat RR (`cache_v37_waveforms_6db`). Pure-geometry score `s_geom` comes from out-of-database LODO `F2_RefFrame10D_SVM` (**ZERO RR, ZERO pre-QRS**).",
        "",
        "## Part I: Overall Error Budget Across All 67 F-Positive Patients (`4,600` Class-F Beats, Pure-Geometry `F2`)",
        "",
        "| DB | F+ Pats | Total F | `E1` NoAnchor (F / Dom Pats) | `E2` Stage-1 Rej (F / Dom Pats) | `E3a` In-S2 **V-Dominated** (F / Dom Pats) | `E3b` In-S2 **N/S-Dominated** (F / Dom Pats) | `E4` Detected @PatF2 (F) |",
        "|---|---:|---:|---|---|---|---|---:|",
    ]
    tot_b = Counter()
    for db in DB_ORDER:
        bd = budget_by_db[db]
        for k, v in bd.items(): tot_b[k] += v
        lines.append(
            f"| `{db}` | {bd['n_pats']} | {bd['tot_F']} | {bd['F_E1_NoAnc']} ({bd['PatsDom_E1_NoAnc']}p) | "
            f"{bd['F_E2_S1Rej']} ({bd['PatsDom_E2_S1Rej']}p) | **{bd['F_E3_S2_VDom']} ({bd['PatsDom_E3_S2_VDom']}p)** | "
            f"{bd['F_E3_S2_NDom']} ({bd['PatsDom_E3_S2_NDom']}p) | {bd['F_E4_DetPatF2']} |"
        )
    lines.append(
        f"| **TOTAL** | **{tot_b['n_pats']}** | **{tot_b['tot_F']}** | **{tot_b['F_E1_NoAnc']} ({tot_b['PatsDom_E1_NoAnc']}p)** | "
        f"**{tot_b['F_E2_S1Rej']} ({tot_b['PatsDom_E2_S1Rej']}p)** | **{tot_b['F_E3_S2_VDom']} ({tot_b['PatsDom_E3_S2_VDom']}p)** | "
        f"**{tot_b['F_E3_S2_NDom']} ({tot_b['PatsDom_E3_S2_NDom']}p)** | **{tot_b['F_E4_DetPatF2']}** |"
    )

    lines.extend([
        "", "## Part II: Conditional Physiological Separability at Equivalent Geometry (`|Delta s_geom| <= 0.25`, Within-Patient)", "",
        "| Feature | Causality | Evaluable Pats (Matched F) | **Pat-Median `cAUC` (`F vs V`)** | IQR `[p25, p75]` | Stricter `|Δs|≤0.10` | Consistent `≥0.60` / Neutral / Reversed `≤0.40` | **`F vs N` Pat-Med `cAUC`** |",
        "|---|---|---:|---:|---|---:|---|---:|",
    ])
    for fk, caus, desc in PHYS_KEYS:
        s = summary_fv_025[fk]; s10 = summary_fv_010[fk]; sn = summary_fn_025[fk]
        if s["n_pats"] == 0: continue
        lines.append(
            f"| `{fk}` ({desc}) | {caus} | {s['n_pats']} ({s['tot_matched_F']}F) | **{s['pat_median_cAUC']:.3f}** | "
            f"[{s['pat_iqr'][0]:.2f}, {s['pat_iqr'][1]:.2f}] | {s10.get('pat_median_cAUC', 0):.3f} | "
            f"**{s['consistent_ge60']}** / {s['neutral_40_60']} / **{s['reversed_le40']}** | "
            f"{sn.get('pat_median_cAUC', '—')} (n={sn.get('n_pats', 0)}) |"
        )

    lines.extend([
        "", "## Part III: Pre-QRS Signal-Quality Stratification (High Normal P-Consistency `≥ 0.40` vs Low `< 0.40`)", "",
        "| Feature | High P-Quality Pats (`n`, Pat-Med `cAUC`, `≥0.60 / ≤0.40`) | Low P-Quality Pats (`n`, Pat-Med `cAUC`, `≥0.60 / ≤0.40`) |",
        "|---|---|---|",
    ])
    for fk in ("pre_rr_loc", "neg_rr_dev", "preqrs_v36_7_73", "preqrs_pwin_35_63", "preqrs_early_7_35", "preqrs_qedge_63_73", "preqrs_pwin_RRmatched", "neg_post_rr_1beat"):
        hi = summary_fv_hiP[fk]; lo = summary_fv_loP[fk]
        lines.append(
            f"| `{fk}` | n={hi.get('n_pats',0)}: **{hi.get('pat_median_cAUC','—')}** ({hi.get('consistent_ge60',0)} / {hi.get('reversed_le40',0)}) | "
            f"n={lo.get('n_pats',0)}: **{lo.get('pat_median_cAUC','—')}** ({lo.get('consistent_ge60',0)} / {lo.get('reversed_le40',0)}) |"
        )

    lines.extend([
        "", "## Part IV: Top-25 F-Rich Patients — Individual Conditional `cAUC` (`F vs V` at `|Delta s_geom| <= 0.25`)", "",
        "| DB | Patient | Tot_F (S2_F / S2_V) | Dom Fail | Normal P-Cons (`35:63`) | `pre_rr` cAUC | `neg_rr_dev` cAUC | `v36_preqrs` cAUC | `P-win 35:63` cAUC | `QRS-edge 63:73` cAUC | `P-win|RR` cAUC | `-post_rr` (1-beat) | Overlap Meds (`pre_rr`: F/V; `P-win`: F/V) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    bmap = {r["patient_id"]: r for r in pat_budget_rows}
    for e in sorted(per_pat_cond, key=lambda x: -x["S2_F"])[:25]:
        p = e["patient_id"]; br = bmap[p]
        def g(fk):
            v = e["F_vs_V_cal025"][fk]["cAUC"]
            return f"{v:.2f}" if v is not None else "—"
        ov = e["medians_FV_overlap"]
        ov_s = f"rr:{ov['F_pre_rr']:.2f}/{ov['V_pre_rr']:.2f}; P:{ov['F_pwin']:+.2f}/{ov['V_pwin']:+.2f}" if ov else "—"
        lines.append(
            f"| `{e['db']}` | `{p}` | {e['tot_F']} ({e['S2_F']}/{e['S2_V']}) | {br['dom_fail']} | {e['pq_cons_pwin']:+.2f} | "
            f"**{g('pre_rr_loc')}** | {g('neg_rr_dev')} | {g('preqrs_v36_7_73')} | **{g('preqrs_pwin_35_63')}** | "
            f"{g('preqrs_qedge_63_73')} | {g('preqrs_pwin_RRmatched')} | {g('neg_post_rr_1beat')} | {ov_s} |"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nALL DONE in {time.time()-t0:.1f}s!\nSaved: {out_json}\nSaved: {out_md}", flush=True)


if __name__ == "__main__":
    main()
