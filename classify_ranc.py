"""
classify_ranc.py
================
RANC (Retrieval-Augmented Node Classification) on frozen TGN embeddings.

This script takes a trained TGN checkpoint and preprocessed graph files,
builds licit prototype banks, trains a RANC classifier with focal loss, and evaluates on the test set.

Usage
-----
    python classify_ranc.py \
        --cache-dir /path/to/preprocessed \
        --pretrained-model /path/to/TGN_PRETRAIN_42_0.pth \
        --ranc-top-m 10 --ranc-num-scales 2 \
        --num-epoch 40 --bs 512 --seed 42

Outputs
-------
    saved_models/elliptic_full/RANC_elliptic_full_42_0.pth
    saved_results/elliptic_full/RANC_full_run0.json
"""

import argparse
import random
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.amp import GradScaler, autocast
from torch.nn import Dropout, LayerNorm, Linear, ReLU, Sequential

from modules.decoder import LinkPredictor
from modules.early_stopping import EarlyStopMonitor
from modules.emb_module import GraphAttentionEmbedding
from modules.memory_module import TGNMemory
from modules.msg_agg import LastAggregator
from modules.msg_func import IdentityMessage
from modules.neighbor_loader import LastNeighborLoader


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

VALID_SIMILARITIES = {"cosine", "pearson", "l2"}


def _amp_spec(device: torch.device):
    """Return (device_type, dtype, enabled) for autocast/GradScaler."""
    if device.type == "cuda" and torch.cuda.is_available():
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                 else torch.float16)
        return "cuda", dtype, True
    return device.type, torch.float32, False


def parse_args():
    p = argparse.ArgumentParser(
        description="RANC classification using pretrained TGN embeddings.")
    p.add_argument("--cache-dir", type=str, required=True,
                   help="path to preprocessed cache")
    p.add_argument("--pretrained-model", type=str, required=True,
                   help="path to TGN pretrain checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-epoch", type=int, default=40)
    p.add_argument("--bs", type=int, default=512,
                   help="batch size for RANC training")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--mem-dim", type=int, default=172)
    p.add_argument("--time-dim", type=int, default=172)
    p.add_argument("--emb-dim", type=int, default=172)
    p.add_argument("--num-neighbors", type=int, default=20)
    # Split
    p.add_argument("--split-type",
                   choices=["temporal", "random", "stratified",
                            "paper_temporal"],
                   default="stratified")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--paper-cut-ts", type=int, default=33)
    p.add_argument("--target-illicit-pct", type=float, default=None)
    p.add_argument("--bank-licit-max", type=int, default=None,
                   help="cap the licit prototype bank")
    # RANC
    p.add_argument("--ranc-top-m", type=int, default=10)
    p.add_argument("--ranc-num-scales", type=int, default=2)
    p.add_argument("--ranc-similarity", choices=list(VALID_SIMILARITIES),
                   default="cosine")
    p.add_argument("--ranc-temperature", type=float, default=0.07)
    p.add_argument("--ranc-hidden", type=int, default=256)
    p.add_argument("--ranc-dropout", type=float, default=0.3)
    p.add_argument("--dual-bank", action="store_true",
                   help="also retrieve from an illicit prototype bank")
    # Loss
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.75)
    p.add_argument("--no-balanced-sampler", action="store_true")
    # I/O
    p.add_argument("--dataset-tag", type=str, default="elliptic_full")
    p.add_argument("--num-run", type=int, default=1)
    p.add_argument("--fine-tune-tgn", action="store_true")
    p.add_argument("--log-test-per-epoch", action="store_true")
    return p.parse_args()


@torch.no_grad()
def replay_full_graph(memory, neighbor_loader, data, device, bs: int):
    """Populate memory from the whole edge stream."""
    memory.eval()
    memory.reset_state(); neighbor_loader.reset_state()
    edges, ts = data["edges"], data["ts"]
    edge_raw = data["edge_raw"]
    RBS = bs * 20
    for k in range(0, len(edges), RBS):
        e = min(len(edges), k + RBS)
        src = torch.from_numpy(edges[k:e, 0]).long().to(device)
        dst = torch.from_numpy(edges[k:e, 1]).long().to(device)
        t   = torch.from_numpy(ts[k:e]).float().to(device)
        msg = edge_raw[k:e]
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


def extract_embeddings(node_ids: torch.Tensor, memory, gnn, neighbor_loader,
                       node_raw, edge_raw, ts_all, assoc, device):
    n_id, ei, eid = neighbor_loader(node_ids)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z, lu = memory(n_id)
    z_s = node_raw[n_id]
    z = gnn(torch.cat([z, z_s], dim=1), lu, ei, ts_all[eid], edge_raw[eid])
    return torch.cat([z, z_s], dim=1)[assoc[node_ids]]


def load_pretrained(models, path, device):
    ck = torch.load(path, map_location=device)
    for key, mod in [("memory_state_dict", "memory"),
                     ("gnn_state_dict", "gnn"),
                     ("link_pred_state_dict", "link_pred")]:
        if key in ck:
            models[mod].load_state_dict(ck[key])
    print(f"[pretrain] loaded weights from {path}")


class PrototypeBank:
    """Stores class-specific prototypes; supports cosine / pearson / L2 retrieval."""

    def __init__(self, device, temperature=0.07):
        self.device = device
        self.temperature = temperature
        self.keys = None
        self.values = None

    def build(self, embs: torch.Tensor, tag: str = ""):
        emb = embs.to(self.device).detach().float()
        self.values = emb
        self.keys = F.normalize(emb, dim=-1)
        print(f"[Bank{tag}] indexed {self.keys.shape[0]:,} "
              f"prototypes (dim={self.keys.shape[1]})")

    def is_ready(self):
        return self.keys is not None

    def retrieve(self, q: torch.Tensor, top_m: int, similarity: str = "cosine"):
        q = q.to(self.device)
        if similarity == "cosine":
            scores = F.normalize(q, dim=-1) @ self.keys.T
        elif similarity == "pearson":
            q_c = q - q.mean(dim=-1, keepdim=True)
            v_c = self.values - self.values.mean(dim=-1, keepdim=True)
            scores = F.normalize(q_c, dim=-1) @ F.normalize(v_c, dim=-1).T
        else:  # l2 (negative squared dist)
            d = torch.cdist(q, self.values)
            scores = -d
        top_m = min(top_m, self.keys.shape[0])
        top_s, top_i = torch.topk(scores, k=top_m, dim=-1)
        attn = torch.softmax(top_s / self.temperature, dim=-1)
        topk_vals = self.values[top_i]
        z_ret = torch.einsum("bm,bmd->bd", attn, topk_vals)
        return z_ret


class MultiScaleRetriever(nn.Module):
    """S learnable projections; each retrieves from the bank and is fused."""

    def __init__(self, emb_dim: int, num_scales: int = 2):
        super().__init__()
        self.projs = nn.ModuleList(
            [Linear(emb_dim, emb_dim) for _ in range(num_scales)])
        self.unify = nn.ModuleList(
            [Linear(emb_dim, emb_dim) for _ in range(num_scales)])
        self.ln = LayerNorm(emb_dim)

    def forward(self, q, bank: PrototypeBank, top_m: int, similarity: str):
        ref = 0
        for p, u in zip(self.projs, self.unify):
            ref = ref + u(bank.retrieve(p(q), top_m=top_m, similarity=similarity))
        return self.ln(ref)


class RANCClassifier(nn.Module):
    """MLP head on [q || r_licit (|| r_illicit) || |q-r_licit|]."""

    def __init__(self, emb_dim: int, num_scales: int, top_m: int,
                 similarity: str, hidden: int, dropout: float,
                 temperature: float, dual_bank: bool = False):
        super().__init__()
        assert similarity in VALID_SIMILARITIES
        self.top_m = top_m
        self.similarity = similarity
        self.dual_bank = dual_bank
        self.retriever_licit = MultiScaleRetriever(emb_dim, num_scales)
        self.retriever_illicit = (MultiScaleRetriever(emb_dim, num_scales)
                                  if dual_bank else None)
        in_dim = emb_dim * (4 if dual_bank else 3)
        self.mlp = Sequential(
            Linear(in_dim, hidden), ReLU(), Dropout(dropout),
            Linear(hidden, hidden // 2), ReLU(), Dropout(dropout),
            Linear(hidden // 2, 1),
        )

    def forward(self, q: torch.Tensor, bank_licit: PrototypeBank,
                bank_illicit: PrototypeBank = None):
        r_licit = self.retriever_licit(q, bank_licit, self.top_m, self.similarity)
        dev = torch.abs(q - r_licit)
        if self.dual_bank and self.retriever_illicit is not None and bank_illicit is not None:
            r_illicit = self.retriever_illicit(
                q, bank_illicit, self.top_m, self.similarity)
            feat = torch.cat([q, r_licit, r_illicit, dev], dim=-1)
        else:
            feat = torch.cat([q, r_licit, dev], dim=-1)
        return self.mlp(feat).squeeze(-1)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1 - p)
        a_t = torch.where(targets > 0.5,
                          torch.full_like(p, self.alpha),
                          torch.full_like(p, 1 - self.alpha))
        return (a_t * (1 - pt).pow(self.gamma) * bce).mean()


def build_splits(labels: np.ndarray, last_ts: np.ndarray, split_type: str,
                 train_ratio: float, val_ratio: float, seed: int,
                 paper_cut_ts: int = 33,
                 target_illicit_pct: float = None):
    idx = np.where((labels == 0) | (labels == 1))[0]
    y = labels[idx]
    t = last_ts[idx]

    if target_illicit_pct is not None:
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        total = n_pos + n_neg
        cur_pct = 100.0 * n_pos / total if total else 0.0
        tgt = float(target_illicit_pct)
        if tgt <= 0 or tgt >= 100:
            raise ValueError(
                f"--target-illicit-pct must be in (0, 100); got {tgt}")
        if tgt <= cur_pct:
            print(f"  [rebalance] current illicit%={cur_pct:.2f} already "
                  f"≥ target={tgt:.2f} → no downsampling applied")
        else:
            needed_neg = int(round(n_pos * (100.0 - tgt) / tgt))
            if needed_neg >= n_neg:
                print(f"  [rebalance] need {needed_neg:,} licit but only "
                      f"{n_neg:,} available → no downsampling applied")
            else:
                pos_local = np.where(y == 1)[0]
                neg_local = np.where(y == 0)[0]
                rng_ds = np.random.RandomState(seed)
                rng_ds.shuffle(neg_local)
                keep_neg = neg_local[:needed_neg]
                keep = np.sort(np.concatenate([pos_local, keep_neg]))
                idx = idx[keep]
                y = y[keep]
                t = t[keep]
                new_pos = int((y == 1).sum())
                new_neg = int((y == 0).sum())
                new_pct = 100.0 * new_pos / (new_pos + new_neg)
                print(f"  [rebalance] licit {n_neg:,} → {new_neg:,}  "
                      f"illicit {n_pos:,} (unchanged)  "
                      f"illicit% {cur_pct:.2f} → {new_pct:.2f} "
                      f"(target {tgt:.2f})")

    rng = np.random.RandomState(seed)

    if split_type == "random":
        order = rng.permutation(len(idx))

    elif split_type == "temporal":
        order = np.argsort(t, kind="mergesort")
        n = len(order)
        ntr = int(train_ratio * n)
        nva = int(val_ratio * n)
        train = order[:ntr]
        val   = order[ntr:ntr + nva]
        test  = order[ntr + nva:]

        def _pct(sub_y):
            n = len(sub_y)
            return 0.0 if n == 0 else 100.0 * float((sub_y == 1).sum()) / n
        print(f"  [temporal] illicit% — "
            f"train={_pct(y[train]):.2f}  val={_pct(y[val]):.2f}  "
            f"test={_pct(y[test]):.2f}")
        return (idx[train], y[train],
                idx[val],   y[val],
                idx[test],  y[test])

    elif split_type == "paper_temporal":
        CUT = int(paper_cut_ts)
        trainval_mask = t <= CUT
        test_mask     = t >  CUT
        trainval = np.where(trainval_mask)[0]
        test     = np.where(test_mask)[0]

        tv_y = y[trainval]
        pos = trainval[tv_y == 1]
        neg = trainval[tv_y == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        ntr_p = int(0.8 * len(pos))
        ntr_n = int(0.8 * len(neg))
        train = np.concatenate([pos[:ntr_p], neg[:ntr_n]])
        val   = np.concatenate([pos[ntr_p:], neg[ntr_n:]])
        rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

        def _pct(sub_y):
            n = len(sub_y)
            return 0.0 if n == 0 else 100.0 * float((sub_y == 1).sum()) / n
        print(f"  [paper_temporal] ts≤{CUT}: {len(trainval):,} (train+val)  "
              f"ts>{CUT}: {len(test):,} (test)  "
              f"illicit% — train={_pct(y[train]):.2f}  val={_pct(y[val]):.2f}  "
              f"test={_pct(y[test]):.2f}")
        return (idx[train], y[train],
                idx[val],   y[val],
                idx[test],  y[test])

    else:  # stratified
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        ntr_p = int(train_ratio * len(pos))
        ntr_n = int(train_ratio * len(neg))
        nva_p = int(val_ratio * len(pos))
        nva_n = int(val_ratio * len(neg))
        train = np.concatenate([pos[:ntr_p], neg[:ntr_n]])
        val   = np.concatenate([pos[ntr_p:ntr_p + nva_p],
                                neg[ntr_n:ntr_n + nva_n]])
        test  = np.concatenate([pos[ntr_p + nva_p:], neg[ntr_n + nva_n:]])
        rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

        def _pct(sub_y):
            n = len(sub_y)
            return 0.0 if n == 0 else 100.0 * float((sub_y == 1).sum()) / n
        print(f"  [stratified] illicit% — train={_pct(y[train]):.2f}  "
              f"val={_pct(y[val]):.2f}  test={_pct(y[test]):.2f}")
        return (idx[train], y[train],
                idx[val],   y[val],
                idx[test],  y[test])

    n = len(order)
    ntr = int(train_ratio * n)
    nva = int(val_ratio * n)
    tr, va, te = order[:ntr], order[ntr:ntr + nva], order[ntr + nva:]
    return (idx[tr], y[tr], idx[va], y[va], idx[te], y[te])


def train_ranc_epoch(classifier, bank_l, bank_i, models, data, device,
                     train_nodes, train_labels, args, opt, scaler, loss_fn,
                     tgn_params=None, tgn_opt=None):
    classifier.train()
    frozen = tgn_params is None
    if frozen:
        for m in ("memory", "gnn"):
            models[m].eval()
    else:
        for m in ("memory", "gnn"):
            models[m].train()
    amp_dev, dtype, amp_enabled = _amp_spec(device)

    if not args.no_balanced_sampler:
        pos = np.where(train_labels == 1)[0]
        neg = np.where(train_labels == 0)[0]
        np.random.shuffle(pos); np.random.shuffle(neg)
        num_batches = max(1, len(train_nodes) // args.bs)
        half = args.bs // 2
        batches = []
        for b in range(num_batches):
            p = pos[(b * half) % len(pos): (b * half) % len(pos) + half]
            n = neg[(b * half) % len(neg): (b * half) % len(neg) + half]
            if len(p) < half:
                p = np.concatenate([p, np.random.choice(pos, half - len(p))])
            if len(n) < half:
                n = np.concatenate([n, np.random.choice(neg, half - len(n))])
            batches.append(np.concatenate([p, n]))
    else:
        order = np.random.permutation(len(train_nodes))
        batches = [order[i:i + args.bs] for i in range(0, len(order), args.bs)]

    total, nsamp = 0.0, 0
    for b_idx in batches:
        nodes = torch.from_numpy(train_nodes[b_idx]).long().to(device)
        y = torch.from_numpy(train_labels[b_idx]).float().to(device)

        opt.zero_grad(set_to_none=True)
        if tgn_opt is not None:
            tgn_opt.zero_grad(set_to_none=True)

        with autocast(device_type=amp_dev, dtype=dtype,
                      enabled=amp_enabled):
            if frozen:
                with torch.no_grad():
                    q = extract_embeddings(
                        nodes, models["memory"], models["gnn"],
                        data["neighbor_loader"], data["node_raw"],
                        data["edge_raw"], data["ts_all"], data["assoc"],
                        device)
            else:
                q = extract_embeddings(
                    nodes, models["memory"], models["gnn"],
                    data["neighbor_loader"], data["node_raw"],
                    data["edge_raw"], data["ts_all"], data["assoc"], device)
            logits = classifier(q, bank_l, bank_i)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        if tgn_opt is not None:
            scaler.unscale_(tgn_opt)
            torch.nn.utils.clip_grad_norm_(tgn_params, max_norm=1.0)
            scaler.step(tgn_opt)
        scaler.step(opt); scaler.update()

        total += float(loss.detach()) * len(b_idx)
        nsamp += len(b_idx)
    return total / max(1, nsamp)


@torch.no_grad()
def evaluate_ranc(classifier, bank_l, bank_i, models, data, device,
                  nodes, labels, bs, tune_threshold: bool,
                  current_threshold: float = 0.5):
    classifier.eval()
    models["memory"].eval(); models["gnn"].eval()
    amp_dev, dtype, amp_enabled = _amp_spec(device)
    logits_all = []
    for k in range(0, len(nodes), bs):
        b = nodes[k:k + bs]
        nt = torch.from_numpy(b).long().to(device)
        q = extract_embeddings(
            nt, models["memory"], models["gnn"],
            data["neighbor_loader"], data["node_raw"], data["edge_raw"],
            data["ts_all"], data["assoc"], device)
        with autocast(device_type=amp_dev, dtype=dtype,
                      enabled=amp_enabled):
            logits_all.append(classifier(q, bank_l, bank_i).float().cpu().numpy())
    logits = np.concatenate(logits_all)
    probs = 1.0 / (1.0 + np.exp(-logits))
    if tune_threshold:
        best_f1, best_th = 0.0, 0.5
        for th in np.arange(0.1, 0.9, 0.02):
            preds = (probs >= th).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        current_threshold = float(best_th)
    preds = (probs >= current_threshold).astype(int)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0
    try:
        ap = average_precision_score(labels, probs)
    except Exception:
        ap = 0.0
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auc":       auc,
        "ap":        ap,
        "threshold": current_threshold,
    }


def main():
    args = parse_args()
    t0 = timeit.default_timer()
    print("INFO: args =", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise ValueError(f"Cache dir does not exist: {cache_dir}")

    node_feats = np.load(cache_dir / "node_feats.npy")
    node_labels = np.load(cache_dir / "node_labels.npy")
    edges = np.load(cache_dir / "edges.npy")
    ts = np.load(cache_dir / "ts.npy")
    edge_feats = np.load(cache_dir / "edge_feats.npy")

    N = node_feats.shape[0]
    NODE_FEAT_DIM = node_feats.shape[1]
    EDGE_FEAT_DIM = edge_feats.shape[1]
    print(f"[data] N={N:,}  E={len(edges):,}  "
          f"node_feat_dim={NODE_FEAT_DIM}  edge_feat_dim={EDGE_FEAT_DIM}")

    node_raw = torch.from_numpy(node_feats).float().to(device)
    edge_raw = torch.from_numpy(edge_feats).float().to(device)
    ts_all   = torch.from_numpy(ts).float().to(device)

    last_ts = np.zeros(N, dtype=np.int64)
    np.maximum.at(last_ts, edges[:, 0], ts)
    np.maximum.at(last_ts, edges[:, 1], ts)

    results_dir = Path("saved_results") / args.dataset_tag
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("saved_models") / args.dataset_tag
    models_dir.mkdir(parents=True, exist_ok=True)

    agg_metrics = {"precision": [], "recall": [], "f1": [], "auc": [], "ap": []}

    for run in range(args.num_run):
        print("=" * 80)
        print(f">>> RUN {run} <<<")
        seed = args.seed + run
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        neighbor_loader = LastNeighborLoader(N, size=args.num_neighbors,
                                             device=device)
        memory = TGNMemory(
            N, EDGE_FEAT_DIM, args.mem_dim, args.time_dim,
            message_module=IdentityMessage(
                EDGE_FEAT_DIM, args.mem_dim, args.time_dim),
            aggregator_module=LastAggregator()).to(device)
        gnn = GraphAttentionEmbedding(
            in_channels=args.mem_dim + NODE_FEAT_DIM,
            out_channels=args.emb_dim,
            msg_dim=EDGE_FEAT_DIM,
            time_enc=memory.time_enc).to(device)
        link_pred = LinkPredictor(
            in_channels=args.emb_dim + NODE_FEAT_DIM).to(device)
        models = {"memory": memory, "gnn": gnn, "link_pred": link_pred}

        assoc = torch.full((N,), fill_value=2 ** 62, dtype=torch.long,
                           device=device)

        load_pretrained(models, args.pretrained_model, device)

        for m in models.values():
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
        print("[step] TGN frozen")

        ranc_data = {
            "node_raw": node_raw, "edge_raw": edge_raw, "ts_all": ts_all,
            "assoc": assoc, "neighbor_loader": neighbor_loader,
        }

        replay_full_graph(memory, neighbor_loader,
                          {"edges": edges, "ts": ts, "edge_raw": edge_raw},
                          device, args.bs)

        tr_idx, tr_y, va_idx, va_y, te_idx, te_y = build_splits(
            node_labels, last_ts, args.split_type,
            args.train_ratio, args.val_ratio, seed,
            paper_cut_ts=args.paper_cut_ts,
            target_illicit_pct=args.target_illicit_pct)
        print(f"[split] train={len(tr_idx):,}  val={len(va_idx):,}  test={len(te_idx):,}")
        print(f"        train pos/neg = {int((tr_y==1).sum()):,}/"
              f"{int((tr_y==0).sum()):,}")

        def _bank_from_ids(ids_np):
            bank_embs = []
            for k in range(0, len(ids_np), args.bs):
                b = torch.from_numpy(ids_np[k:k + args.bs]).long().to(device)
                with torch.no_grad():
                    emb = extract_embeddings(
                        b, memory, gnn, neighbor_loader,
                        node_raw, edge_raw, ts_all, assoc, device)
                bank_embs.append(emb.float().cpu())
            return torch.cat(bank_embs, dim=0)

        licit_ids   = tr_idx[tr_y == 0]
        illicit_ids = tr_idx[tr_y == 1]

        if args.bank_licit_max is not None and \
           len(licit_ids) > int(args.bank_licit_max):
            cap = int(args.bank_licit_max)
            rng_bank = np.random.RandomState(seed)
            pick = rng_bank.choice(len(licit_ids), size=cap, replace=False)
            print(f"[bank] subsampling licit prototypes "
                  f"{len(licit_ids):,} → {cap:,} (seed={seed})")
            licit_ids = np.sort(licit_ids[pick])

        print(f"[bank] extracting {len(licit_ids):,} licit prototypes")
        bank_licit = PrototypeBank(device, args.ranc_temperature)
        bank_licit.build(_bank_from_ids(licit_ids), tag="-licit")

        bank_illicit = None
        if args.dual_bank and len(illicit_ids) > 0:
            print(f"[bank] extracting {len(illicit_ids):,} illicit prototypes")
            bank_illicit = PrototypeBank(device, args.ranc_temperature)
            bank_illicit.build(_bank_from_ids(illicit_ids), tag="-illicit")

        emb_total = args.emb_dim + NODE_FEAT_DIM
        classifier = RANCClassifier(
            emb_dim=emb_total, num_scales=args.ranc_num_scales,
            top_m=args.ranc_top_m, similarity=args.ranc_similarity,
            hidden=args.ranc_hidden, dropout=args.ranc_dropout,
            temperature=args.ranc_temperature,
            dual_bank=bank_illicit is not None).to(device)

        opt = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
        tgn_params = None; tgn_opt = None
        if args.fine_tune_tgn:
            tgn_params = (list(memory.parameters())
                          + list(gnn.parameters()))
            for p in tgn_params:
                p.requires_grad = True
            tgn_opt = torch.optim.AdamW(tgn_params, lr=args.lr / 10)
            print("[step] TGN fine-tuning enabled at LR/10")

        _amp_dev, _, _amp_enabled = _amp_spec(device)
        scaler = GradScaler(device=_amp_dev, enabled=_amp_enabled)
        loss_fn = BinaryFocalLoss(alpha=args.focal_alpha,
                                  gamma=args.focal_gamma)

        stopper = EarlyStopMonitor(
            save_model_dir=str(models_dir),
            save_model_id=f"RANC_{args.dataset_tag}_{args.seed}_{run}",
            tolerance=args.tolerance, patience=args.patience,
            higher_better=True)

        best_thr = 0.5
        val_f1_hist = []
        for epoch in range(1, args.num_epoch + 1):
            te = timeit.default_timer()
            loss = train_ranc_epoch(
                classifier, bank_licit, bank_illicit, models, ranc_data, device,
                tr_idx, tr_y, args, opt, scaler, loss_fn,
                tgn_params=tgn_params, tgn_opt=tgn_opt)
            vm = evaluate_ranc(
                classifier, bank_licit, bank_illicit, models, ranc_data, device,
                va_idx, va_y, args.bs, tune_threshold=True)
            best_thr = vm["threshold"]
            val_f1_hist.append(vm["f1"])
            print(f"[ranc] ep {epoch:02d}  loss={loss:.4f}  "
                  f"val P={vm['precision']:.3f} R={vm['recall']:.3f} "
                  f"F1={vm['f1']:.3f} AUC={vm['auc']:.3f} AP={vm['ap']:.3f} "
                  f"th={vm['threshold']:.2f}  "
                  f"t={timeit.default_timer() - te:.1f}s")

            if args.log_test_per_epoch:
                tm_ep = evaluate_ranc(
                    classifier, bank_licit, bank_illicit, models, ranc_data,
                    device, te_idx, te_y, args.bs, tune_threshold=False,
                    current_threshold=vm["threshold"])
                print(f"[ranc] ep {epoch:02d}  "
                      f"test P={tm_ep['precision']:.3f} R={tm_ep['recall']:.3f} "
                      f"F1={tm_ep['f1']:.3f} AUC={tm_ep['auc']:.3f} "
                      f"AP={tm_ep['ap']:.3f} th={tm_ep['threshold']:.2f}")

            if stopper.step_check(vm["f1"], {"classifier": classifier}):
                print(f"[ranc] early stop at epoch {epoch}")
                break

        stopper.load_checkpoint({"classifier": classifier})
        tm = evaluate_ranc(
            classifier, bank_licit, bank_illicit, models, ranc_data, device,
            te_idx, te_y, args.bs, tune_threshold=False,
            current_threshold=best_thr)
        print("-" * 60)
        print(f"TEST run={run}  P={tm['precision']:.4f}  R={tm['recall']:.4f}  "
              f"F1={tm['f1']:.4f}  AUC={tm['auc']:.4f}  AP={tm['ap']:.4f}  "
              f"th={tm['threshold']:.2f}")
        print("-" * 60)
        for k in agg_metrics:
            agg_metrics[k].append(tm[k])

        out = {
            "model":          "TGN_RANC_full",
            "dataset":        args.dataset_tag,
            "run":            run, "seed": seed,
            "split_type":     args.split_type,
            "target_illicit_pct": args.target_illicit_pct,
            "dual_bank":      bank_illicit is not None,
            "fine_tune_tgn":  args.fine_tune_tgn,
            "val_f1":         val_f1_hist,
            "test_precision": tm["precision"], "test_recall": tm["recall"],
            "test_f1":        tm["f1"], "test_auc": tm["auc"],
            "test_ap":        tm["ap"], "threshold": tm["threshold"],
            "node_feat_dim":  NODE_FEAT_DIM, "edge_feat_dim": EDGE_FEAT_DIM,
        }
        out_path = results_dir / f"RANC_full_run{run}.json"
        pd.Series(out).to_json(out_path)
        print(f"[io] saved {out_path}")

    print("=" * 80)
    print("SUMMARY (TGN_RANC_full)")
    for k, vs in agg_metrics.items():
        arr = np.array(vs)
        print(f"  {k:<10s}: {arr.mean():.4f} ± {arr.std():.4f}  (runs={len(arr)})")
    print(f"total elapsed: {timeit.default_timer() - t0:.1f}s")


if __name__ == "__main__":
    main()
