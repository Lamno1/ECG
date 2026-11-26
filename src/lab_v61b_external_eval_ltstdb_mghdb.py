r"""
lab_v61b_external_eval_ltstdb_mghdb.py
======================================
ONE-SHOT LOCKED EXTERNAL EVALUATION OF ALL 4 PUBLISHED 2024-2025 SOTA DL MODELS
(`Net1D_ECGFounder`, `ModernTCN_ICLR24`, `Medformer_NeurIPS24`, `PatCond_Net1D_ECGFounder`)
ON:
  1. Primary External Holter Cohort `LTSTDB` (70 verified subjects, 76 records, 1,756.6 h,
     7,969,461 post-prefix beats, 583 Class-F beats)
  2. Supplementary External ICU Cohort `MGHDB` (219 eligible records, 213 Tier-1 / 198 Tier-2
     subjects, 246.66 h, 1,367,094 post-prefix beats, 771 Class-F beats)

Strict Integrity Guarantees:
  - Loads all 4 SHA-256-sealed checkpoints from `v61a` (ZERO target refitting or threshold tuning).
  - Downloads `.dat` files in bounded waves with official PhysioNet `SHA256SUMS.txt` check.
  - Uses IDENTICAL waveform filtering, 360 Hz 187-sample grid, `AAMI_Map_v1.1`, `v33.normalize_lead`,
    50-beat prefix, and causal `b_N_loc` ring buffer as `stage2_frozen_pipeline_v2.py` & `v37`.
  - ASSERTS BIT-EXACT LABEL PARITY (`np.array_equal(y, cached_v59_or_v60_y)`) on EVERY SINGLE RECORD!
  - Runs all 4 DL models simultaneously on RTX 3050 Ti GPU (FP16 AMP), caches compact per-record
    float32 score `.npz` files (resumable per wave!), and deletes raw `.dat` immediately after each wave.
  - Combines frozen `A3_SVM`, `A3_LogR`, `LightGBM_A3_TwoBranch`, and all 4 SOTA DL models into the
    unified canonical audit report:
      `audits/V61_SOTA_DL_AND_GBDT_EXTERNAL_BENCHMARK_20261123.json` & `.md`.
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
import gc
import json
import time
import shutil
import hashlib
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import wfdb
import wfdb.io._header as _wfdb_hdr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Safe in-memory fallback for PhysioNet mgh025.hea malformed date & sig_len==0 (mgh061, mgh127)
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
import lab_diagnostic_v33_greek_multibeat_mit44_and_incart75 as v33
import stage2_frozen_pipeline_v2 as pipe2
import v61_sota_models as m61

REG_LTSTDB = str(_REPO_ROOT / "registry/target_cohort_ltstdb_v1.json")
REG_MGHDB = str(_REPO_ROOT / "registry/target_cohort_mghdb_v1.json")
CONTRACT_V61 = str(_REPO_ROOT / "registry/v61_dl_sota_benchmark_contract_v1.json")

CKPT_DIR = str(_REPO_ROOT / "checkpoints")
AUD_DIR = str(_REPO_ROOT / "audits")

RAW_LTSTDB = str(_DATA_ROOT / "ltstdb")
RAW_MGHDB = str(_DATA_ROOT / "mghdb")
CACHE_V59 = str(_DATA_ROOT / "cache_v59_ltstdb_a3")
CACHE_V60 = str(_DATA_ROOT / "cache_v60_mghdb_a3")

SCORE_CACHE_LTSTDB = str(_DATA_ROOT / "cache_v61_dl_scores_ltstdb")
SCORE_CACHE_MGHDB = str(_DATA_ROOT / "cache_v61_dl_scores_mghdb")
os.makedirs(SCORE_CACHE_LTSTDB, exist_ok=True)
os.makedirs(SCORE_CACHE_MGHDB, exist_ok=True)

URL_LTSTDB = "https://physionet.org/files/ltstdb/1.0.0/"
URL_MGHDB = "https://physionet.org/files/mghdb/1.0.0/"
BN_SUB_STRIDE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sha256sums(raw_dir, base_url):
    p = os.path.join(raw_dir, "SHA256SUMS.txt")
    if not os.path.exists(p):
        data = urllib.request.urlopen(base_url + "SHA256SUMS.txt", timeout=30).read()
        with open(p, "wb") as f:
            f.write(data)
    sums = {}
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) >= 2:
                sums[parts[1]] = parts[0]
    return sums


def download_verified_file(raw_dir, base_url, fname, expected_sha):
    dst = os.path.join(raw_dir, fname)
    if os.path.exists(dst) and expected_sha and sha256_file(dst) == expected_sha:
        return dst
    tmp = dst + ".part"
    for k in range(6):
        try:
            req = urllib.request.Request(base_url + fname, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            got = sha256_file(tmp)
            if expected_sha and got != expected_sha:
                raise RuntimeError(f"SHA mismatch for {fname}: {got[:16]} != {expected_sha[:16]}")
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


def extract_dl_inputs_worker(args):
    """
    Bit-exact waveform + RR3 + causal b_N_loc_sub extraction matching stage2_frozen_pipeline_v2.py & v37.
    Asserts exact label parity against cached v59/v60 `.npz` on every record!
    """
    raw_dir, ref_cache_dir, rec_id, ann_ext, ch_indices = args
    rec_path = os.path.join(raw_dir, rec_id)
    rec = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, ann_ext)
    fs = float(rec.fs)

    i0, i1 = int(ch_indices[0]), int(ch_indices[1])
    s0_raw = np.nan_to_num(rec.p_signal[:, i0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    s1_raw = np.nan_to_num(rec.p_signal[:, i1].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sig0, delay = v33.filter_signal_generic(s0_raw, fs, mode="offline_filtfilt")
    sig1, _ = v33.filter_signal_generic(s1_raw, fs, mode="offline_filtfilt")

    samples, symbols = ann.sample, ann.symbol
    n_sig = len(sig0)
    if abs(fs - v33.TARGET_FS) < 1e-3:
        beats0, beats1, labels, r_peaks = [], [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < 93 or sc + 94 > n_sig:
                continue
            c = pipe2.map_symbol(sym, "AAMI_Map_v1.1")
            if c >= 0:
                beats0.append(sig0[sc - 93 : sc + 94])
                beats1.append(sig1[sc - 93 : sc + 94])
                labels.append(c)
                r_peaks.append(s)
        beats0 = np.asarray(beats0, dtype=np.float64)
        beats1 = np.asarray(beats1, dtype=np.float64)
    else:
        half_win = int(np.ceil(0.30 * fs))
        local_t = np.arange(-half_win, half_win + 1, dtype=np.float64) / fs
        grid_idx = (v33.PHYS_TIME_GRID - local_t[0]) * fs
        lo_i = np.floor(grid_idx).astype(np.int64)
        hi_i = np.minimum(lo_i + 1, len(local_t) - 1)
        frac = (grid_idx - lo_i).astype(np.float64)

        valid_sc, labels, r_peaks = [], [], []
        for s, sym in zip(samples, symbols):
            sc = int(s) + delay
            if sc < half_win or sc + half_win + 1 > n_sig:
                continue
            c = pipe2.map_symbol(sym, "AAMI_Map_v1.1")
            if c >= 0:
                valid_sc.append(sc)
                labels.append(c)
                r_peaks.append(s)
        valid_sc = np.array(valid_sc, dtype=np.int64)
        base_off = valid_sc[:, None] - half_win
        beats0 = (1.0 - frac[None, :]) * sig0[base_off + lo_i[None, :]] + frac[None, :] * sig0[base_off + hi_i[None, :]]
        beats1 = (1.0 - frac[None, :]) * sig1[base_off + lo_i[None, :]] + frac[None, :] * sig1[base_off + hi_i[None, :]]

    labels = np.asarray(labels, dtype=np.int8)
    r_peaks = np.asarray(r_peaks, dtype=np.float64)
    N_all = len(labels)
    assert N_all > 55, f"Unexpected short record {rec_id}: {N_all}"

    # Bit-exact label parity check against frozen v59 / v60 cache!
    post_y = labels[50:]
    ref_npz = np.load(os.path.join(ref_cache_dir, f"{rec_id}.npz"))
    assert np.array_equal(post_y, ref_npz["y"]), f"Label parity mismatch on {rec_id}!"

    # Exact causal 11-beat rolling median pre-RR + 3-beat rr3 [prev, curr, next]
    pre_rr = np.zeros(N_all, dtype=np.float64)
    pre_rr[1:] = (r_peaks[1:] - r_peaks[:-1]) / fs
    pre_rr[0] = pre_rr[1]
    local_rr = np.zeros(N_all, dtype=np.float64)
    for i in range(min(10, N_all)):
        local_rr[i] = float(np.median(pre_rr[: i + 1])) + 1e-6
    if N_all > 10:
        local_rr[10:] = np.median(sliding_window_view(pre_rr, 11), axis=1) + 1e-6
    pre_rr_loc_all = pre_rr / local_rr
    post_hours = float((r_peaks[-1] - r_peaks[50]) / fs / 3600.0)

    N = len(post_y)
    rr_prev = np.clip(pre_rr_loc_all[49 : 49 + N], 0.2, 3.0)
    rr_curr = np.clip(pre_rr_loc_all[50 : 50 + N], 0.2, 3.0)
    rr_next = np.empty(N, dtype=np.float64)
    rr_next[:-1] = rr_curr[1:]
    rr_next[-1] = rr_curr[-1]
    rr3 = np.column_stack([rr_prev, rr_curr, rr_next]).astype(np.float32)

    # Exact normalized 2-lead beats & Stage-1 causal b_N_loc (bit-identical to v37 & pipe2)
    n0_all = v33.normalize_lead(beats0)
    n1_all = v33.normalize_lead(beats1)
    joint_all = np.column_stack([n0_all, n1_all])
    joint_50 = joint_all[:50]
    rr_50 = pre_rr_loc_all[:50]
    M, D = joint_50.shape

    km = KMeans(n_clusters=2, n_init=20, random_state=42)
    clusters = km.fit_predict(joint_50)
    sil = float(silhouette_score(joint_50, clusters)) if len(np.unique(clusters)) > 1 else 0.0
    n0_c, n1_c = int(np.sum(clusters == 0)), int(np.sum(clusters == 1))
    c0_0, c0_1 = km.cluster_centers_[0][:187], km.cluster_centers_[0][187:]
    c1_0, c1_1 = km.cluster_centers_[1][:187], km.cluster_centers_[1][187:]
    slope0 = float(max(np.max(np.abs(np.diff(c0_0))), np.max(np.abs(np.diff(c0_1)))))
    slope1 = float(max(np.max(np.abs(np.diff(c1_0))), np.max(np.abs(np.diff(c1_1)))))
    sel_cluster = 0 if slope0 >= slope1 else 1
    slope_ratio = min(slope0, slope1) / (max(slope0, slope1) + 1e-12)
    selected_idx = np.where(clusters == sel_cluster)[0]
    is_multi = (sil >= 0.35) and (min(n0_c, n1_c) >= 4) and (len(selected_idx) >= 8)

    if is_multi:
        center = km.cluster_centers_[sel_cluster]
        dists = np.linalg.norm(joint_50[selected_idx] - center, axis=1)
        q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
        trim_mask = (dists <= np.median(dists) + 1.5 * (q75 - q25))
        final_norm_idx = selected_idx[trim_mask]
        if len(final_norm_idx) < 8:
            final_norm_idx = selected_idx
        ectopic_idx = np.where(clusters == (1 - sel_cluster))[0]
        med_rr_ect = float(np.median(rr_50[ectopic_idx]))
        if not ((med_rr_ect <= 0.92) or (slope_ratio <= 0.70)):
            ectopic_idx = np.array([], dtype=int)
    else:
        med_all = np.median(joint_50, axis=0)
        dists = np.linalg.norm(joint_50 - med_all, axis=1)
        q25, q75 = np.percentile(dists, 25), np.percentile(dists, 75)
        trim_mask = (dists <= np.median(dists) + 1.5 * (q75 - q25))
        final_norm_idx = np.where(trim_mask)[0]
        if len(final_norm_idx) < 10:
            final_norm_idx = np.arange(M)
        ectopic_idx = np.array([], dtype=int)

    norm_beats = joint_50[final_norm_idx]
    b_N = np.median(norm_beats, axis=0)
    centered_norm = norm_beats - b_N
    _, _, Vt = np.linalg.svd(centered_norm, full_matrices=False)
    U = Vt[:min(4, Vt.shape[0]), :].T
    P = np.eye(D) - U @ U.T
    norm_y_prefix = centered_norm @ P.T
    eta_0 = float(np.percentile(np.linalg.norm(norm_y_prefix, axis=1), 95)) + 1e-6

    hf0_50 = np.sum(np.diff(joint_50[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1_50 = np.sum(np.diff(joint_50[:, 187:], n=2, axis=1) ** 2, axis=1)
    med_hf0 = float(np.median(hf0_50[final_norm_idx])) + 1e-6
    med_hf1 = float(np.median(hf1_50[final_norm_idx])) + 1e-6

    anchors = []
    if len(ectopic_idx) >= 4:
        anc = v33.make_anchor_obj(joint_50[ectopic_idx], norm_y_prefix, b_N, P, 0, "prefix")
        if anc is not None:
            anchors.append(anc)

    post_joint = joint_all[50:]
    pre_rr_loc = pre_rr_loc_all[50:]
    hf0 = np.sum(np.diff(post_joint[:, :187], n=2, axis=1) ** 2, axis=1)
    hf1 = np.sum(np.diff(post_joint[:, 187:], n=2, axis=1) ** 2, axis=1)
    sqi_hf_ratio = np.log1p(np.maximum(hf0 / med_hf0, hf1 / med_hf1))

    diff_post = post_joint - b_N[None, :]
    y_mat = diff_post - (diff_post @ U) @ U.T
    norm_y = np.linalg.norm(y_mat, axis=1)
    norm_y_thresh = max(v33.CAND_NORM_Y_MIN, v33.CAND_ETA_MULT * eta_0)

    loc_ring = np.zeros((v33.LOC_BUF_LEN, D), dtype=np.float64)
    init_nb = norm_beats[-v33.LOC_BUF_LEN:]
    loc_cnt = len(init_nb)
    loc_ring[:loc_cnt] = init_nb
    loc_ptr = loc_cnt % v33.LOC_BUF_LEN
    b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)

    n_sub = (N + BN_SUB_STRIDE - 1) // BN_SUB_STRIDE
    bn_sub = np.empty((n_sub, 2, 187), dtype=np.float16)
    prev1, prev2 = joint_all[49], joint_all[48]
    cand_clusters = []

    for t in range(N):
        x_t = post_joint[t]
        if (t % BN_SUB_STRIDE) == 0:
            bn_sub[t // BN_SUB_STRIDE] = b_N_loc.reshape(2, 187).astype(np.float16)
        c_loc = v33.mean_lead_corr(x_t, b_N_loc)

        if (norm_y[t] >= norm_y_thresh) and (pre_rr_loc[t] <= v33.CAND_PRE_RR_LOC_MAX) and (sqi_hf_ratio[t] <= v33.ONL_HF_MAX):
            matched_anc, best_adist = None, v33.CLUSTER_RADIUS
            for anc in anchors:
                d_a = float(np.linalg.norm(x_t - anc["b_V"]))
                if d_a <= best_adist:
                    best_adist, matched_anc = d_a, anc
            if matched_anc is not None:
                q = matched_anc["q"]
                nq = matched_anc["norm_q"]
                matched_anc["v_lam_buf"].append(float(np.dot(y_mat[t], q) / (nq ** 2 + 1e-12)))
                matched_anc["v_dy_buf"].append(float(np.linalg.norm(y_mat[t] - q)))
                v33._refresh_v_radii(matched_anc)
            elif len(anchors) < v33.MAX_V_ANCHORS:
                best_c_idx, best_dist = -1, v33.CLUSTER_RADIUS
                for ci, cl in enumerate(cand_clusters):
                    dist = float(np.linalg.norm(x_t - cl["center"]))
                    if dist <= best_dist:
                        best_dist, best_c_idx = dist, ci
                if best_c_idx >= 0:
                    best_c = cand_clusters[best_c_idx]
                    best_c["beats"].append(x_t)
                    best_c["center"] = np.median(best_c["beats"], axis=0)
                    if len(best_c["beats"]) >= v33.CONFIRM_MIN_BEATS:
                        arr_b = np.array(best_c["beats"])
                        med_b = np.median(arr_b, axis=0)
                        cohesion = float(np.median(np.linalg.norm(arr_b - med_b, axis=1)))
                        if cohesion <= v33.CONFIRM_MAX_COHESION:
                            anc = v33.make_anchor_obj(arr_b, norm_y_prefix, b_N, P, int(t + 1), "online")
                            if anc is not None:
                                anchors.append(anc)
                                cand_clusters.pop(best_c_idx)
                else:
                    if len(cand_clusters) >= v33.MAX_CAND_CLUSTERS:
                        cand_clusters.sort(key=lambda c: len(c["beats"]))
                        cand_clusters.pop(0)
                    cand_clusters.append({"beats": [x_t], "center": x_t.copy()})

        reg_rr = (v33.LOC_RR_MIN <= pre_rr_loc[t] <= v33.LOC_RR_MAX)
        far_from_v = all(float(np.linalg.norm(x_t - anc["b_V"])) >= v33.LOC_V_EXCLUSION_DIST for anc in anchors)
        if reg_rr and far_from_v:
            c_prev1 = v33.mean_lead_corr(x_t, prev1)
            c_prev12 = v33.mean_lead_corr(prev1, prev2)
            if (c_loc >= v33.LOC_ADMIT_CORR_BN and c_prev1 >= 0.70) or (c_prev1 >= v33.LOC_ADMIT_RUN_CORR and c_prev12 >= v33.LOC_ADMIT_RUN_CORR):
                loc_ring[loc_ptr] = x_t
                loc_ptr = (loc_ptr + 1) % v33.LOC_BUF_LEN
                if loc_cnt < v33.LOC_BUF_LEN:
                    loc_cnt += 1
                b_N_loc = np.median(loc_ring[:loc_cnt], axis=0)
        prev2, prev1 = prev1, x_t

    beats_2ch = np.stack([n0_all[50:], n1_all[50:]], axis=1).astype(np.float16)
    return {
        "rec_id": rec_id,
        "post_hours": post_hours,
        "y": post_y,
        "beats_2ch": beats_2ch,
        "rr3": rr3,
        "bn_sub": bn_sub,
    }


@torch.no_grad()
def score_record_all_dl_models(loaded_models, rec_data, batch_size=2048):
    beats_2ch = rec_data["beats_2ch"]
    rr3 = rec_data["rr3"]
    bn_sub = rec_data["bn_sub"]
    N = len(rec_data["y"])
    n_sub = len(bn_sub)

    scores_out = {mid: np.empty(N, dtype=np.float32) for mid in m61.DL_MODEL_ORDER}
    for i0 in range(0, N, batch_size):
        i1 = min(i0 + batch_size, N)
        bx = beats_2ch[i0:i1]
        locs = np.arange(i0, i1, dtype=np.int32)
        bni = np.minimum(locs // BN_SUB_STRIDE, n_sub - 1)
        bn = bn_sub[bni]

        xb2 = torch.from_numpy(bx).to(DEVICE, dtype=torch.float32, non_blocking=True)
        xb4 = torch.from_numpy(np.concatenate([bx, bx - bn], axis=1)).to(DEVICE, dtype=torch.float32, non_blocking=True)
        rrb_raw = torch.from_numpy(rr3[i0:i1]).to(DEVICE, dtype=torch.float32, non_blocking=True)

        for mid in m61.DL_MODEL_ORDER:
            obj = loaded_models[mid]
            rrb = (rrb_raw - obj["rr_mean_t"]) / obj["rr_std_t"]
            x_in = xb4 if mid == "PatCond_Net1D_ECGFounder" else xb2
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = obj["model"](x_in, rrb)
                f_lo = m61.logits_to_fusion_logodds(logits.float())
            scores_out[mid][i0:i1] = f_lo.cpu().numpy()
    return scores_out


def stream_score_cohort(cohort_name, records_meta, raw_dir, base_url, ann_ext, ref_cache_dir, score_cache_dir, loaded_models, wave_size=8):
    sha_map = load_sha256sums(raw_dir, base_url)
    todo = [r for r in records_meta if not os.path.exists(os.path.join(score_cache_dir, f"{r['rec_id']}.npz"))]
    print(f"\n[v61b {cohort_name}] Score cache: {len(records_meta) - len(todo)}/{len(records_meta)} done, {len(todo)} remaining.", flush=True)

    t0 = time.time()
    done = len(records_meta) - len(todo)
    for wi in range(0, len(todo), wave_size):
        wave = todo[wi : wi + wave_size]
        t_w = time.time()
        fnames = []
        for r in wave:
            rid = r["rec_id"]
            fnames.extend([f"{rid}.hea", f"{rid}.{ann_ext}", f"{rid}.dat"])

        def _dl(fn):
            return download_verified_file(raw_dir, base_url, fn, sha_map.get(fn))

        with ThreadPoolExecutor(max_workers=4) as dex:
            list(dex.map(_dl, fnames))
        t_dl = time.time() - t_w

        t_ex0 = time.time()
        worker_args = [(raw_dir, ref_cache_dir, r["rec_id"], ann_ext, tuple(r["locked_channel_indices"])) for r in wave]
        # Process 2 records at a time to keep RAM low on 24h LTSTDB records
        n_proc = 2 if cohort_name == "LTSTDB" else 4
        t_inf_tot = 0.0
        with ProcessPoolExecutor(max_workers=n_proc) as pex:
            for rec_data in pex.map(extract_dl_inputs_worker, worker_args):
                t_i0 = time.time()
                sc_dict = score_record_all_dl_models(loaded_models, rec_data)
                t_inf_tot += time.time() - t_i0
                out_p = os.path.join(score_cache_dir, f"{rec_data['rec_id']}.npz")
                np.savez_compressed(
                    out_p,
                    y=rec_data["y"],
                    post_hours=np.float64(rec_data["post_hours"]),
                    **{f"s_{mid}": sc_dict[mid] for mid in m61.DL_MODEL_ORDER},
                )
                del rec_data, sc_dict
                gc.collect()
        t_ex_inf = time.time() - t_ex0

        # Delete large raw .dat files immediately after verified score caching
        for r in wave:
            dat_p = os.path.join(raw_dir, f"{r['rec_id']}.dat")
            if os.path.exists(dat_p):
                os.remove(dat_p)

        done += len(wave)
        free_gb = shutil.disk_usage("d:\\").free / 1e9
        print(
            f"  [{cohort_name}] Wave {wi // wave_size + 1:2d}/{(len(todo) + wave_size - 1) // wave_size:2d} | "
            f"progress={done:3d}/{len(records_meta)} | dl={t_dl:5.1f}s ext+gpu={t_ex_inf:5.1f}s (gpu={t_inf_tot:4.1f}s) | "
            f"elapsed={time.time() - t0:6.1f}s | D:free={free_gb:.1f}GB",
            flush=True,
        )


def evaluate_cached_dl_scores(cohort_name, records_meta, score_cache_dir, loaded_models, sid_field, ref_a3_pt1, k_map=None):
    out = {}
    for mid in m61.DL_MODEL_ORDER:
        tau = loaded_models[mid]["tau_r3_b20"]
        stream_items = []
        tot_beats = 0
        tot_f = 0
        for r in records_meta:
            rid = r["rec_id"]
            d = np.load(os.path.join(score_cache_dir, f"{rid}.npz"))
            y = d["y"].astype(np.int8)
            tot_beats += len(y)
            tot_f += int(np.sum(y == 2))
            stream_items.append({
                "patient_id": r[sid_field],
                "post_hours": float(d["post_hours"]),
                "y": y,
                "score": d[f"s_{mid}"],
                "K": int(k_map[rid]) if k_map is not None else 0,
            })

        summ, pt1, _, mfa, zf = m61.evaluate_scalar_stream_cohort(stream_items, tau=tau)
        boot = m61.paired_bootstrap_compare(ref_a3_pt1, pt1) if ref_a3_pt1 is not None else None
        kpos_sum = None
        if k_map is not None:
            kpos_sids = {it["patient_id"] for it in stream_items if it["K"] > 0}
            k_items = [it for it in stream_items if it["patient_id"] in kpos_sids]
            kpos_sum, _, _, _, _ = m61.evaluate_scalar_stream_cohort(k_items, tau=tau)

        out[mid] = {
            "Summary": summ,
            "Matched_FA": mfa,
            "ZeroF_Tail": zf,
            "Bootstrap_A3SVM_minus_Model": boot,
            "ActiveAnchor_Kpos_Summary": kpos_sum,
        }
        print(
            f"  [{cohort_name} | {mid:24s}] TP={summ['T1_TP']:4d} FP_NS={summ['T1_FP_NS']:6d} FP_V={summ['T1_FPV']:5d} | "
            f"Prec={summ['T1_Prec_%']:5.2f}% F1={summ['T1_F1']:.4f} Rec={summ['T1_MacroPat_Rec_%']:5.2f}% | "
            f"FAPH={summ['T1_MeanPat_FAPH_All']:6.3f} (0F={summ['T1_MeanPat_FAPH_ZeroF']:6.3f} max={summ['T1_ZeroF_Max_FAPH']:6.2f}) | "
            f"AP={summ['AP']:.4f} Pat@F2={summ['MacroPat_Pat@F2_%']:5.2f}% M@2={summ['Match_All_2.0']:5.2f}%",
            flush=True,
        )
        if boot is not None:
            print(
                f"    A3_SVM minus {mid}: dRec={boot['Delta_MacroRec_pt_%']:+.2f}% {boot['Delta_MacroRec_CI95_%']} ({boot['Fpos_WTL_Rec']}) | "
                f"dFAPH_All={boot['Delta_MeanFAPH_All_pt']:+.3f} {boot['Delta_MeanFAPH_All_CI95']} | dFAPH_0F={boot['Delta_MeanFAPH_0F_pt']:+.3f} {boot['Delta_MeanFAPH_0F_CI95']}",
                flush=True,
            )
    return out


def write_markdown_report(report_dict, md_path):
    lines = [
        "# v61 External Benchmark: Frozen Dual-Pole Geometric `A3` vs. Published 2024–2025 SOTA Deep Learning & Modern Nonlinear GBDT",
        "",
        f"- **Generated (UTC)**: `{report_dict['Generated_UTC']}`",
        f"- **Contract SHA-256**: `{report_dict['Contract_SHA256']}`",
        f"- **GPU**: `{report_dict['GPU_Name']}`",
        "- **Protocol**: All 5 competitors trained strictly on the identical 6 source DBs (`251` `SubjectAware_251` units, `2,676,362` beats, `4,600` Class-$F$), calibrated via the identical patient-level `R3_B20` rule ($\\overline{\\mathrm{FAPH}}_{\\mathrm{all}}^{\\mathrm{src}} \\le 2.0$) on deterministic 5-fold subject-grouped OOF, and evaluated locked once on `LTSTDB` (`70` subjects, `7.97M` beats, `583 F`) and `MGHDB` (`213` Tier-1 / `198` Tier-2 subjects, `1.37M` beats, `771 F`).",
        "- Frozen `A3` (`v59`/`v60`) checkpoints held strictly unchanged.",
        "",
    ]

    def _add_table(title, rows):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Model | Params | TP | FP-N/S | FP-V | Prec (%) | F1 | MacroRec (%) | FAPH All | FAPH 0-F | 0-F max (<=2%) | AP | Pat@F2 (%) | M@0.5 | M@1.0 | M@2.0 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        for label, npar, s in rows:
            lines.append(
                f"| **{label}** | {npar} | {s['T1_TP']} | {s['T1_FP_NS']} | {s['T1_FPV']} | {s['T1_Prec_%']:.2f} | {s['T1_F1']:.4f} | "
                f"{s['T1_MacroPat_Rec_%']:.2f} | {s['T1_MeanPat_FAPH_All']:.3f} | {s['T1_MeanPat_FAPH_ZeroF']:.3f} | "
                f"{s['T1_ZeroF_Max_FAPH']:.2f} ({s['T1_Frac0F_LE2_%']:.1f}%) | {s['AP']:.4f} | {s['MacroPat_Pat@F2_%']:.2f} | "
                f"{s['Match_All_0.5']:.2f} | {s['Match_All_1.0']:.2f} | {s['Match_All_2.0']:.2f} |"
            )
        lines.append("")

    # Table 1: Source 6-DB 5-Fold OOF
    src_rows = [
        ("LightGBM_A3 (TwoBranch, Stage1+A3)", "GBDT(200x2)", report_dict["Part1_LightGBM_and_A3"]["LightGBM_A3_TwoBranch"]["Source_5Fold_OOF"]["Summary"]),
    ]
    for mid in m61.DL_MODEL_ORDER:
        m = report_dict["DL_Checkpoints"][mid]
        src_rows.append((mid, f"{m['n_params']:,}", m["Source_5Fold_OOF"]["Summary"]))
    _add_table("1. Source 6-DB 5-Fold Subject-Grouped OOF (251 Units, 2.68M Beats, 4,600 F; R3_B20 <= 2.0 FAPH)", src_rows)

    # Table 2: External Primary LTSTDB (70 subjects)
    lt_rows = [
        ("A3_SVM (LOCKED v59)", "28", report_dict["Part1_LightGBM_and_A3"]["Frozen_A3_Reference"]["LTSTDB_SVM"]["Summary"]),
        ("A3_LogR (LOCKED v59)", "28", report_dict["Part1_LightGBM_and_A3"]["Frozen_A3_Reference"]["LTSTDB_LogR"]["Summary"]),
        ("LightGBM_A3 (TwoBranch)", "GBDT(200x2)", report_dict["Part1_LightGBM_and_A3"]["LightGBM_A3_TwoBranch"]["LTSTDB"]["Summary"]),
    ]
    for mid in m61.DL_MODEL_ORDER:
        lt_rows.append((mid, f"{report_dict['DL_Checkpoints'][mid]['n_params']:,}", report_dict["External_LTSTDB_70"][mid]["Summary"]))
    _add_table("2. Primary External Holter Cohort: LTSTDB (70 Subjects, 7.97M Beats, 583 F; Zero Target Tuning)", lt_rows)

    # Table 3: External Supplementary ICU MGHDB Tier-1 (213 subjects)
    mgh_rows = [
        ("A3_SVM (LOCKED v60)", "28", report_dict["Part1_LightGBM_and_A3"]["Frozen_A3_Reference"]["MGHDB_Tier1_SVM"]["Summary"]),
        ("A3_LogR (LOCKED v60)", "28", report_dict["Part1_LightGBM_and_A3"]["Frozen_A3_Reference"]["MGHDB_Tier1_LogR"]["Summary"]),
        ("LightGBM_A3 (TwoBranch)", "GBDT(200x2)", report_dict["Part1_LightGBM_and_A3"]["LightGBM_A3_TwoBranch"]["MGHDB_Tier1"]["Summary"]),
    ]
    for mid in m61.DL_MODEL_ORDER:
        mgh_rows.append((mid, f"{report_dict['DL_Checkpoints'][mid]['n_params']:,}", report_dict["External_MGHDB_Tier1_213"][mid]["Summary"]))
    _add_table("3. Supplementary External ICU Cohort: MGHDB Tier-1 (213 Subjects, 1.37M Beats, 771 F; Zero Target Tuning)", mgh_rows)

    # Table 4: Paired Bootstrap (A3_SVM minus Competitor)
    lines.append("## 4. Paired Subject-Level Bootstrap (`A3_SVM` minus Competitor; B=10,000, seed=42)")
    lines.append("")
    lines.append("| Cohort | Competitor | ΔMacroRec (%) [95% CI] | W/T/L (Rec) | ΔFAPH_All [95% CI] | ΔFAPH_0F [95% CI] |")
    lines.append("|---|---|---|---|---|---|")
    for coh_lbl, boot_map in [
        ("LTSTDB (70)", [("LightGBM_A3", report_dict["Part1_LightGBM_and_A3"]["LightGBM_A3_TwoBranch"]["Bootstrap_A3SVM_minus_LightGBM"]["LTSTDB"])] + [(mid, report_dict["External_LTSTDB_70"][mid]["Bootstrap_A3SVM_minus_Model"]) for mid in m61.DL_MODEL_ORDER]),
        ("MGHDB Tier-1 (213)", [("LightGBM_A3", report_dict["Part1_LightGBM_and_A3"]["LightGBM_A3_TwoBranch"]["Bootstrap_A3SVM_minus_LightGBM"]["MGHDB_Tier1"])] + [(mid, report_dict["External_MGHDB_Tier1_213"][mid]["Bootstrap_A3SVM_minus_Model"]) for mid in m61.DL_MODEL_ORDER]),
    ]:
        for cname, b in boot_map:
            lines.append(
                f"| {coh_lbl} | **{cname}** | `{b['Delta_MacroRec_pt_%']:+.2f}%` `{b['Delta_MacroRec_CI95_%']}` | `{b['Fpos_WTL_Rec']}` | "
                f"`{b['Delta_MeanFAPH_All_pt']:+.3f}` `{b['Delta_MeanFAPH_All_CI95']}` | `{b['Delta_MeanFAPH_0F_pt']:+.3f}` `{b['Delta_MeanFAPH_0F_CI95']}` |"
            )
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ltstdb", action="store_true")
    ap.add_argument("--skip-mghdb", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    part1 = json.load(open(os.path.join(AUD_DIR, "V61A_LIGHTGBM_A3_RESULTS_20261123.json"), "r", encoding="utf-8"))
    dl_meta = json.load(open(os.path.join(AUD_DIR, "V61A_DL_SOURCE_OOF_SUMMARY_20261123.json"), "r", encoding="utf-8"))

    # Load all 4 SHA-256 verified DL checkpoints onto GPU
    loaded_models = {}
    for mid in m61.DL_MODEL_ORDER:
        info = dl_meta["Models"][mid]
        ckpt_p = info["ckpt_path"]
        got_sha = sha256_file(ckpt_p)
        assert got_sha == info["ckpt_sha256"], f"Checkpoint SHA mismatch for {mid}!"
        ck = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
        mdl = m61.build_dl_model(mid).to(DEVICE)
        mdl.load_state_dict(ck["state_dict"])
        mdl.eval()
        loaded_models[mid] = {
            "model": mdl,
            "rr_mean_t": torch.from_numpy(ck["rr_mean"]).to(DEVICE, dtype=torch.float32),
            "rr_std_t": torch.from_numpy(ck["rr_std"]).to(DEVICE, dtype=torch.float32),
            "tau_r3_b20": float(ck["tau_r3_b20"]),
            "ckpt_sha256": got_sha,
        }
        print(f"[v61b] Loaded & verified {mid:24s} | tau={ck['tau_r3_b20']:+.4f} | SHA={got_sha[:12]}", flush=True)

    # 1. LTSTDB (76 recs / 70 subjects)
    lt_reg = json.load(open(REG_LTSTDB, "r", encoding="utf-8"))
    lt_recs = lt_reg["included_records"]
    assert len(lt_recs) == 76
    if not args.skip_ltstdb:
        stream_score_cohort("LTSTDB", lt_recs, RAW_LTSTDB, URL_LTSTDB, "atr", CACHE_V59, SCORE_CACHE_LTSTDB, loaded_models, wave_size=8)

    # 2. MGHDB (219 eligible recs / 213 Tier-1 / 198 Tier-2 subjects)
    mgh_reg = json.load(open(REG_MGHDB, "r", encoding="utf-8"))
    mgh_recs = mgh_reg["included_records"]
    assert len(mgh_recs) == 219
    if not args.skip_mghdb:
        stream_score_cohort("MGHDB", mgh_recs, RAW_MGHDB, URL_MGHDB, "ari", CACHE_V60, SCORE_CACHE_MGHDB, loaded_models, wave_size=10)

    # Build MGHDB K_end map from frozen v60 meta
    mgh_k_map = {}
    for r in mgh_recs:
        m = json.load(open(os.path.join(CACHE_V60, f"{r['rec_id']}_meta.json"), "r", encoding="utf-8"))
        mgh_k_map[r["rec_id"]] = int(m["K"])

    print("\n================ UNIFIED EXTERNAL EVALUATION: LTSTDB (70 SUBJECTS) ================", flush=True)
    lt_res = evaluate_cached_dl_scores(
        "LTSTDB", lt_recs, SCORE_CACHE_LTSTDB, loaded_models,
        sid_field="subject_id",
        ref_a3_pt1=part1["Frozen_A3_Reference"]["LTSTDB_SVM"]["PerSubject"],
    )

    print("\n================ UNIFIED EXTERNAL EVALUATION: MGHDB TIER-1 (213 SUBJECTS) ================", flush=True)
    mgh1_res = evaluate_cached_dl_scores(
        "MGHDB_T1", mgh_recs, SCORE_CACHE_MGHDB, loaded_models,
        sid_field="subject_id_tier1",
        ref_a3_pt1=part1["Frozen_A3_Reference"]["MGHDB_Tier1_SVM"]["PerSubject"],
        k_map=mgh_k_map,
    )

    print("\n================ UNIFIED EXTERNAL EVALUATION: MGHDB TIER-2 (198 SUBJECTS) ================", flush=True)
    mgh2_res = evaluate_cached_dl_scores(
        "MGHDB_T2", mgh_recs, SCORE_CACHE_MGHDB, loaded_models,
        sid_field="subject_id_tier2",
        ref_a3_pt1=part1["Frozen_A3_Reference"]["MGHDB_Tier2_SVM"]["PerSubject"],
    )

    # Strip heavy per-subject dicts before saving compact unified JSON
    part1_compact = json.loads(json.dumps(part1))
    for k in ("LTSTDB_SVM", "MGHDB_Tier1_SVM", "MGHDB_Tier2_SVM"):
        part1_compact["Frozen_A3_Reference"][k].pop("PerSubject", None)
    for k in ("LTSTDB", "MGHDB_Tier1", "MGHDB_Tier2"):
        part1_compact["LightGBM_A3_TwoBranch"][k].pop("PerSubject", None)

    final_report = {
        "Audit_Version": "v61_sota_dl_and_gbdt_external_benchmark",
        "Generated_UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "Contract_SHA256": sha256_file(CONTRACT_V61),
        "GPU_Name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "Total_Eval_Sec": round(time.time() - t0, 1),
        "DL_Checkpoints": dl_meta["Models"],
        "Part1_LightGBM_and_A3": part1_compact,
        "External_LTSTDB_70": lt_res,
        "External_MGHDB_Tier1_213": mgh1_res,
        "External_MGHDB_Tier2_198": mgh2_res,
    }

    out_json = os.path.join(AUD_DIR, "V61_SOTA_DL_AND_GBDT_EXTERNAL_BENCHMARK_20261123.json")
    out_md = os.path.join(AUD_DIR, "V61_SOTA_DL_AND_GBDT_EXTERNAL_BENCHMARK_20261123.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    write_markdown_report(final_report, out_md)
    print(f"\n[v61b] COMPLETE! Saved:\n  JSON: {out_json}\n  MD  : {out_md}", flush=True)


if __name__ == "__main__":
    main()
