r"""
v61_sota_models.py
==================
Shared model constructors (importing directly from pinned MIT-licensed official
MIT-licensed upstream repositories inside `external_repos/`), exact 1D R3_B20 source OOF
calibrator, full-stream cohort metrics evaluator, matched-FA sweep, and paired
subject-level bootstrap matching `v59` and `v60` bit-for-bit.
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
import types
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from scipy.stats import wilcoxon, binomtest

SANDBOX_ROOT = str(_Path(_os.environ.get("ECG_EXTERNAL_REPOS_DIR", str(_REPO_ROOT / "external_repos"))))
ECGFOUNDER_DIR = os.path.join(SANDBOX_ROOT, "ECGFounder_04edac70")
MODERNTCN_DIR = os.path.join(SANDBOX_ROOT, "ModernTCN_56a9a2c0", "ModernTCN-classification")
MEDFORMER_DIR = os.path.join(SANDBOX_ROOT, "Medformer_891f65b8")


def _load_net1d_cls():
    p = os.path.join(ECGFOUNDER_DIR, "net1d.py")
    spec = importlib.util.spec_from_file_location("ecgfounder_net1d", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Net1D


def _load_moderntcn_cls():
    saved_sys_path = list(sys.path)
    saved_mods = {k: sys.modules[k] for k in list(sys.modules.keys()) if k == "layers" or k.startswith("layers.") or k == "models" or k.startswith("models.")}
    try:
        for k in list(saved_mods.keys()):
            del sys.modules[k]
        sys.path.insert(0, MODERNTCN_DIR)
        from models.ModernTCN import ModernTCN
        return ModernTCN
    finally:
        for k in list(sys.modules.keys()):
            if k == "layers" or k.startswith("layers.") or k == "models" or k.startswith("models."):
                del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved_sys_path


def _load_medformer_cls():
    saved_sys_path = list(sys.path)
    saved_mods = {k: sys.modules[k] for k in list(sys.modules.keys()) if k == "layers" or k.startswith("layers.") or k == "models" or k.startswith("models.")}
    try:
        for k in list(saved_mods.keys()):
            del sys.modules[k]
        if "reformer_pytorch" not in sys.modules:
            m_ref = types.ModuleType("reformer_pytorch")
            m_ref.LSHSelfAttention = object
            sys.modules["reformer_pytorch"] = m_ref
        sys.path.insert(0, MEDFORMER_DIR)
        from models.Medformer import Model as MedformerModel
        return MedformerModel
    finally:
        for k in list(sys.modules.keys()):
            if k == "layers" or k.startswith("layers.") or k == "models" or k.startswith("models."):
                del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved_sys_path


Net1D = _load_net1d_cls()
OfficialModernTCN = _load_moderntcn_cls()
OfficialMedformer = _load_medformer_cls()


class Net1DClassifier(nn.Module):
    """
    ECGFounder Net1D (Li et al., NEJM AI 2025, MIT) + 3-beat RR fusion head.
    Supports in_channels=2 (raw 2-lead) or in_channels=4 (patient-anchored [x_t, x_t - b_N_loc]).
    """

    def __init__(self, in_channels=2, n_classes=4):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = Net1D(
            in_channels=in_channels,
            base_filters=32,
            ratio=1,
            filter_list=[64, 128, 192, 256],
            m_blocks_list=[2, 2, 2, 2],
            kernel_size=9,
            stride=2,
            groups_width=16,
            n_classes=n_classes,
            use_bn=True,
            use_do=False,
            return_features=True,
            verbose=False,
        )
        self.rr_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256 + 32),
            nn.Dropout(0.20),
            nn.Linear(256 + 32, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, rr):
        _, feat = self.backbone(x)
        rr_f = self.rr_mlp(rr)
        return self.head(torch.cat([feat, rr_f], dim=-1))


class ModernTCNClassifier(nn.Module):
    """
    ModernTCN (Luo & Wang, ICLR 2024 Spotlight, MIT) + 3-beat RR fusion head.
    """

    def __init__(self, in_channels=2, n_classes=4):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = OfficialModernTCN(
            task_name="classification",
            patch_size=8,
            patch_stride=4,
            stem_ratio=1,
            downsample_ratio=2,
            ffn_ratio=2,
            num_blocks=[2, 2],
            large_size=[15, 11],
            small_size=[5, 5],
            dims=[64, 128],
            dw_dims=[64, 128],
            nvars=in_channels,
            small_kernel_merged=False,
            backbone_dropout=0.10,
            head_dropout=0.10,
            use_multi_scale=False,
            revin=False,
            affine=False,
            subtract_last=False,
            seq_len=187,
            c_in=(in_channels,),
            individual=False,
            target_window=n_classes,
            class_drop=0.10,
            class_num=n_classes,
        )
        feat_dim = in_channels * 128 * 2  # avg + max pool across patches per variable
        self.rr_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim + 32),
            nn.Dropout(0.20),
            nn.Linear(feat_dim + 32, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, rr):
        # x: (B, C, 187) -> fmap: (B, M, D, N_patch)
        fmap = self.backbone.forward_feature(x, te=None)
        fmap = F.gelu(fmap)
        f_avg = fmap.mean(dim=-1)
        f_max = fmap.amax(dim=-1)
        feat = torch.cat([f_avg, f_max], dim=-1).reshape(x.shape[0], -1)
        rr_f = self.rr_mlp(rr)
        return self.head(torch.cat([feat, rr_f], dim=-1))


class MedformerClassifier(nn.Module):
    """
    Medformer (Wang et al., NeurIPS 2024, MIT) + 3-beat RR fusion head.
    Pads 187 -> 188 by 1 trailing zero sample to satisfy even PositionalEmbedding dim.
    """

    def __init__(self, in_channels=2, n_classes=4):
        super().__init__()
        self.in_channels = in_channels
        cfg = types.SimpleNamespace(
            task_name="classification",
            pred_len=0,
            output_attention=False,
            enc_in=in_channels,
            single_channel=False,
            patch_len_list="8,16,32",
            seq_len=188,
            d_model=128,
            n_heads=4,
            e_layers=3,
            d_ff=256,
            dropout=0.10,
            activation="gelu",
            augmentations="none",
            no_inter_attn=False,
            num_class=n_classes,
        )
        self.backbone = OfficialMedformer(cfg)
        # 3 patch granularities * d_model(128) = 384D
        feat_dim = 3 * 128
        self.rr_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim + 32),
            nn.Dropout(0.20),
            nn.Linear(feat_dim + 32, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, rr):
        # x: (B, C, 187) -> (B, 188, C)
        x_pad = F.pad(x, (0, 1), mode="constant", value=0.0).permute(0, 2, 1)
        enc_out = self.backbone.enc_embedding(x_pad)
        enc_out, _ = self.backbone.encoder(enc_out, attn_mask=None)
        enc_out = F.gelu(enc_out)
        feat = enc_out.reshape(x.shape[0], -1)
        rr_f = self.rr_mlp(rr)
        return self.head(torch.cat([feat, rr_f], dim=-1))


DL_MODEL_ORDER = [
    "Net1D_ECGFounder",
    "ModernTCN_ICLR24",
    "Medformer_NeurIPS24",
    "PatCond_Net1D_ECGFounder",
]


def build_dl_model(model_id):
    if model_id == "Net1D_ECGFounder":
        return Net1DClassifier(in_channels=2, n_classes=4)
    if model_id == "ModernTCN_ICLR24":
        return ModernTCNClassifier(in_channels=2, n_classes=4)
    if model_id == "Medformer_NeurIPS24":
        return MedformerClassifier(in_channels=2, n_classes=4)
    if model_id == "PatCond_Net1D_ECGFounder":
        return Net1DClassifier(in_channels=4, n_classes=4)
    raise ValueError(f"Unknown model_id: {model_id}")


def logits_to_fusion_logodds(logits):
    """
    Exact monotonic fusion log-odds: z_F - logsumexp(z_N, z_V, z_S).
    Avoids float32 softmax saturation near 0/1 in extreme-specificity tails.
    """
    z_f = logits[:, 2]
    z_other = torch.stack([logits[:, 0], logits[:, 1], logits[:, 3]], dim=-1)
    return z_f - torch.logsumexp(z_other, dim=-1)


# =============================================================================
# EXACT 1D R3_B20 SOURCE OOF CALIBRATION (IDENTICAL OBJECTIVE TO v56/v59)
# =============================================================================
def solve_r3_1d_exact(scores, y_cat, pi_cat, pat_hours, pat_totF, b_tot=2.0):
    """
    Exact prefix-sum sweep over ALL unique scalar thresholds maximizing the exact R3_B20 objective:
      J(tau) = MacroRec_{F+}(tau) + 0.05 * PoolRec(tau) - 0.002 * MeanPatFAPH_All(tau)
      subject to MeanPatFAPH_All(tau) <= b_tot (2.0).
    """
    n_pats = len(pat_hours)
    fpos_mask = pat_totF > 0
    n_fpos = max(1, int(np.sum(fpos_mask)))
    tot_f_all = max(1, int(np.sum(pat_totF)))

    ord_i = np.argsort(scores)[::-1]
    s_o = scores[ord_i].astype(np.float64)
    y_o = y_cat[ord_i]
    pi_o = pi_cat[ord_i]

    is_f = (y_o == 2)
    is_nf = ~is_f

    macro_inc = np.where(is_f, 1.0 / (np.maximum(1.0, pat_totF[pi_o].astype(np.float64)) * float(n_fpos)), 0.0)
    pool_inc = np.where(is_f, 1.0 / float(tot_f_all), 0.0)
    fa_inc = np.where(is_nf, 1.0 / (np.maximum(1e-6, pat_hours[pi_o]) * float(n_pats)), 0.0)

    cum_macro = np.cumsum(macro_inc)
    cum_pool = np.cumsum(pool_inc)
    cum_fa = np.cumsum(fa_inc)

    # Evaluate only at tie-group endings so pred = (scores >= tau) matches exactly
    grp_end = np.empty(len(s_o), dtype=bool)
    grp_end[:-1] = (s_o[:-1] != s_o[1:])
    grp_end[-1] = True

    ok = np.where(grp_end & (cum_fa <= b_tot + 1e-12))[0]
    if len(ok) == 0:
        return float(s_o[0] + 1.0)

    obj = cum_macro[ok] + 0.05 * cum_pool[ok] - 0.002 * cum_fa[ok]
    best_k = int(ok[int(np.argmax(obj))])
    # Threshold strictly separating best_k and best_k+1 if possible
    if best_k + 1 < len(s_o):
        tau = 0.5 * (float(s_o[best_k]) + float(s_o[best_k + 1]))
    else:
        tau = float(s_o[best_k]) - 1e-6
    return float(tau)


# =============================================================================
# UNIFIED FULL-STREAM COHORT EVALUATION (MATCHES v50 / v59 / v60 EXACTLY)
# =============================================================================
def evaluate_scalar_stream_cohort(rec_items, tau, all_budgets=(0.5, 1.0, 2.0, 3.0), zf_budgets=(0.25, 0.5, 1.0, 2.0)):
    """
    Evaluates a scalar fusion-score model over a list of record dicts:
      rec = {"patient_id", "post_hours", "y" (int8), "score" (float32/64)}
    Sums beat counts & hours per subject BEFORE computing FAPH & recall (exact v50/v59/v60 convention).
    Returns: (summary_dict, per_subject_t1_dict, per_subject_pat_at_f2_dict, matched_fa_dict, zero_f_tail_dict)
    """
    subj_recs = {}
    for it in rec_items:
        subj_recs.setdefault(it["patient_id"], []).append(it)

    subjs = sorted(subj_recs.keys())
    s_to_idx = {s: i for i, s in enumerate(subjs)}
    n_all = len(subjs)

    pt1 = {}
    prf2 = {}
    tot_tp = tot_fp_ns = tot_fp_v = tot_f = tot_beats = 0
    tot_hours = 0.0

    # For pooled AP and cohort matched-FA sweep
    all_y_l, all_s_l, all_si_l = [], [], []
    hours_arr = np.zeros(n_all, dtype=np.float64)
    totF_arr = np.zeros(n_all, dtype=np.int64)

    for si, s in enumerate(subjs):
        recs = subj_recs[s]
        y_s = np.concatenate([r["y"] for r in recs]).astype(np.int8)
        sc_s = np.concatenate([r["score"] for r in recs]).astype(np.float64)
        hrs = float(sum(float(r["post_hours"]) for r in recs))
        hours_arr[si] = hrs

        n_f = int(np.sum(y_s == 2))
        totF_arr[si] = n_f
        tot_f += n_f
        tot_beats += len(y_s)
        tot_hours += hrs

        pred = sc_s >= tau
        tp = int(np.sum(pred & (y_s == 2)))
        fp_ns = int(np.sum(pred & ((y_s == 0) | (y_s == 3))))
        fp_v = int(np.sum(pred & (y_s == 1)))
        fp_tot = fp_ns + fp_v
        faph = float(fp_tot / max(hrs, 1e-6))

        tot_tp += tp
        tot_fp_ns += fp_ns
        tot_fp_v += fp_v

        rec_pct = (100.0 * tp / n_f) if n_f > 0 else None
        pt1[s] = {
            "tot_F": n_f,
            "hours": round(hrs, 4),
            "TP": tp,
            "FP_NS": fp_ns,
            "FP_V": fp_v,
            "FAPH": round(faph, 5),
            "Rec_%": round(rec_pct, 4) if rec_pct is not None else None,
        }

        # Pat@F2: per-subject recall at <= 2.0 FAPH for this individual subject
        if n_f > 0:
            max_fp_s = int(np.floor(2.0 * hrs + 1e-9))
            ord_s = np.argsort(sc_s)[::-1]
            y_so = y_s[ord_s]
            s_so = sc_s[ord_s]
            cum_tp_s = np.cumsum(y_so == 2)
            cum_fp_s = np.cumsum(y_so != 2)
            ge_s = np.empty(len(s_so), dtype=bool)
            ge_s[:-1] = (s_so[:-1] != s_so[1:])
            ge_s[-1] = True
            ok_s = np.where(ge_s & (cum_fp_s <= max_fp_s))[0]
            tp_f2 = int(cum_tp_s[ok_s[-1]]) if len(ok_s) > 0 else 0
            prf2[s] = round(100.0 * tp_f2 / n_f, 4)

        all_y_l.append(y_s)
        all_s_l.append(sc_s)
        all_si_l.append(np.full(len(y_s), si, dtype=np.int32))

    y_all = np.concatenate(all_y_l)
    s_all = np.concatenate(all_s_l)
    si_all = np.concatenate(all_si_l)

    # Pooled AP across ALL beats (100% stream)
    ap = float(average_precision_score((y_all == 2).astype(np.int8), s_all)) if tot_f > 0 else 0.0

    # Summary metrics
    fp_all = tot_fp_ns + tot_fp_v
    prec = (100.0 * tot_tp / (tot_tp + fp_all)) if (tot_tp + fp_all) > 0 else 0.0
    pool_rec = (100.0 * tot_tp / tot_f) if tot_f > 0 else 0.0
    f1 = (2.0 * tot_tp / (2.0 * tot_tp + fp_all + (tot_f - tot_tp))) if (tot_tp + fp_all + tot_f) > 0 else 0.0

    fpos_recs = [u["Rec_%"] for u in pt1.values() if u["tot_F"] > 0]
    all_faphs = np.array([u["FAPH"] for u in pt1.values()], dtype=np.float64)
    zf_faphs = np.array([u["FAPH"] for u in pt1.values() if u["tot_F"] == 0], dtype=np.float64)

    # Zero-F tail audit
    z_list = [(sid, u["FAPH"], u["FP_NS"], u["FP_V"], u["hours"]) for sid, u in pt1.items() if u["tot_F"] == 0]
    z_sorted = sorted(z_list, key=lambda x: -x[1])
    zero_f_tail = {
        "n_0F": len(z_list),
        "mean": round(float(np.mean(zf_faphs)), 4) if len(zf_faphs) > 0 else 0.0,
        "pct_zero": round(100.0 * float(np.mean(zf_faphs == 0.0)), 1) if len(zf_faphs) > 0 else 0.0,
        "pct_le_2": round(100.0 * float(np.mean(zf_faphs <= 2.0 + 1e-9)), 1) if len(zf_faphs) > 0 else 0.0,
        "p50": round(float(np.percentile(zf_faphs, 50)), 4) if len(zf_faphs) > 0 else 0.0,
        "p90": round(float(np.percentile(zf_faphs, 90)), 4) if len(zf_faphs) > 0 else 0.0,
        "p95": round(float(np.percentile(zf_faphs, 95)), 4) if len(zf_faphs) > 0 else 0.0,
        "p99": round(float(np.percentile(zf_faphs, 99)), 4) if len(zf_faphs) > 0 else 0.0,
        "max": round(float(np.max(zf_faphs)), 4) if len(zf_faphs) > 0 else 0.0,
        "top5_worst": [
            {"subject_id": x[0], "faph": round(x[1], 3), "fp_ns": int(x[2]), "fp_v": int(x[3]), "hours": round(x[4], 2)}
            for x in z_sorted[:5]
        ],
    }

    # Cohort Matched-FA curves (exact prefix-sum sweep over ALL beats)
    fpos_mask = totF_arr > 0
    zpos_mask = totF_arr == 0
    n_fpos = max(1, int(np.sum(fpos_mask)))
    n_zpos = max(1, int(np.sum(zpos_mask)))

    ord_i = np.argsort(s_all)[::-1]
    s_o = s_all[ord_i]
    y_o = y_all[ord_i]
    si_o = si_all[ord_i]
    is_f = (y_o == 2)
    is_nf = ~is_f
    rec_inc = np.where(is_f, 100.0 / (np.maximum(1.0, totF_arr[si_o].astype(np.float64)) * float(n_fpos)), 0.0)
    fa_all_inc = np.where(is_nf, 1.0 / (np.maximum(1e-6, hours_arr[si_o]) * float(n_all)), 0.0)
    fa_0f_inc = np.where(is_nf & zpos_mask[si_o], 1.0 / (np.maximum(1e-6, hours_arr[si_o]) * float(n_zpos)), 0.0)

    cum_rec = np.cumsum(rec_inc)
    cum_fa_all = np.cumsum(fa_all_inc)
    cum_fa_0f = np.cumsum(fa_0f_inc)

    grp_end = np.empty(len(s_o), dtype=bool)
    grp_end[:-1] = (s_o[:-1] != s_o[1:])
    grp_end[-1] = True

    out_all = {}
    for b in all_budgets:
        ok = np.where(grp_end & (cum_fa_all <= b + 1e-12))[0]
        out_all[str(b)] = round(float(cum_rec[ok[-1]]), 3) if len(ok) > 0 else 0.0
    out_0f = {}
    for b in zf_budgets:
        ok = np.where(grp_end & (cum_fa_0f <= b + 1e-12))[0]
        out_0f[str(b)] = round(float(cum_rec[ok[-1]]), 3) if len(ok) > 0 else 0.0
    matched_fa = {"AllFAPH": out_all, "ZeroFFAPH": out_0f}

    summary = {
        "n_subjects": n_all,
        "n_Fpos": int(np.sum(fpos_mask)),
        "n_0F": int(np.sum(zpos_mask)),
        "tot_beats": tot_beats,
        "tot_F": tot_f,
        "tot_hours": round(tot_hours, 2),
        "tau": round(float(tau), 6),
        "T1_TP": tot_tp,
        "T1_FP_NS": tot_fp_ns,
        "T1_FPV": tot_fp_v,
        "T1_Prec_%": round(prec, 3),
        "T1_PoolRec_%": round(pool_rec, 3),
        "T1_F1": round(f1, 4),
        "T1_MacroPat_Rec_%": round(float(np.mean(fpos_recs)), 3) if fpos_recs else 0.0,
        "T1_MeanPat_FAPH_All": round(float(np.mean(all_faphs)), 4),
        "T1_MeanPat_FAPH_ZeroF": round(float(np.mean(zf_faphs)), 4) if len(zf_faphs) > 0 else 0.0,
        "T1_Pooled_FAPH": round(float(fp_all / max(tot_hours, 1e-6)), 4),
        "T1_FracPats_LE2_%": round(100.0 * float(np.mean(all_faphs <= 2.0 + 1e-9)), 2),
        "T1_Frac0F_LE2_%": zero_f_tail["pct_le_2"],
        "T1_ZeroF_Max_FAPH": zero_f_tail["max"],
        "AP": round(ap, 5),
        "MacroPat_Pat@F2_%": round(float(np.mean(list(prf2.values()))), 3) if prf2 else 0.0,
        "Match_All_0.5": out_all.get("0.5", 0.0),
        "Match_All_1.0": out_all.get("1.0", 0.0),
        "Match_All_2.0": out_all.get("2.0", 0.0),
    }
    return summary, pt1, prf2, matched_fa, zero_f_tail


def paired_bootstrap_compare(pt1_A, pt1_B, B=10000, seed=42):
    """Exact paired subject-level bootstrap (A minus B), matching v50/v59/v60."""
    rng = np.random.default_rng(seed)
    all_s = sorted(set(pt1_A.keys()) & set(pt1_B.keys()))
    fpos_s = [s for s in all_s if pt1_A[s]["tot_F"] > 0]
    zf_s = [s for s in all_s if pt1_A[s]["tot_F"] == 0]

    d_rec = np.array([pt1_A[s]["Rec_%"] - pt1_B[s]["Rec_%"] for s in fpos_s], dtype=np.float64)
    d_fa = np.array([pt1_A[s]["FAPH"] - pt1_B[s]["FAPH"] for s in all_s], dtype=np.float64)
    d_zf = np.array([pt1_A[s]["FAPH"] - pt1_B[s]["FAPH"] for s in zf_s], dtype=np.float64)

    def _boot(arr):
        if len(arr) == 0:
            return 0.0, [0.0, 0.0]
        idx = rng.integers(0, len(arr), size=(B, len(arr)))
        means = arr[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return round(float(np.mean(arr)), 4), [round(float(lo), 4), round(float(hi), 4)]

    pt_rec, ci_rec = _boot(d_rec)
    pt_fa, ci_fa = _boot(d_fa)
    pt_zf, ci_zf = _boot(d_zf)

    w_r = int(np.sum(d_rec > 1e-9))
    l_r = int(np.sum(d_rec < -1e-9))
    t_r = len(d_rec) - w_r - l_r

    w_fa = int(np.sum(d_fa < -1e-9))  # A has lower FAPH
    l_fa = int(np.sum(d_fa > 1e-9))
    t_fa = len(d_fa) - w_fa - l_fa

    try:
        nz_r = d_rec[np.abs(d_rec) > 1e-9]
        p_w_rec = float(wilcoxon(nz_r).pvalue) if len(nz_r) > 0 else 1.0
    except Exception:
        p_w_rec = 1.0
    try:
        nz_f = d_fa[np.abs(d_fa) > 1e-9]
        p_w_fa = float(wilcoxon(nz_f).pvalue) if len(nz_f) > 0 else 1.0
    except Exception:
        p_w_fa = 1.0

    return {
        "Delta_MacroRec_pt_%": pt_rec,
        "Delta_MacroRec_CI95_%": ci_rec,
        "Fpos_WTL_Rec": f"{w_r}W/{t_r}T/{l_r}L",
        "p_wilcoxon_Rec": round(p_w_rec, 5),
        "Delta_MeanFAPH_All_pt": pt_fa,
        "Delta_MeanFAPH_All_CI95": ci_fa,
        "All_FAPH_Better_Tie_Worse": f"{w_fa}B/{t_fa}T/{l_fa}W",
        "p_wilcoxon_FAPH": round(p_w_fa, 5),
        "Delta_MeanFAPH_0F_pt": pt_zf,
        "Delta_MeanFAPH_0F_CI95": ci_zf,
    }
