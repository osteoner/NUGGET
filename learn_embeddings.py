"""
learn_embeddings.py
===================
Learn node embeddings using TGN with link-prediction pretraining.

This script takes preprocessed graph files and trains a TGN model via
hard-negative mining with AMP optimization. The trained checkpoint is saved
for use by the RANC classification stage.

Usage
-----
    python learn_embeddings.py \
        --cache-dir /path/to/preprocessed \
        --pretrain-epochs 5 \
        --mem-dim 128 --time-dim 128 --emb-dim 128 \
        --seed 42

Outputs
-------
    saved_models/elliptic_full/TGN_PRETRAIN_42_0.pth
"""

import argparse
import random
import timeit
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.memory_module import TGNMemory
from modules.msg_agg import LastAggregator
from modules.msg_func import IdentityMessage
from modules.neighbor_loader import LastNeighborLoader


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _amp_spec(device: torch.device):
    """Return (device_type, dtype, enabled) for autocast/GradScaler."""
    if device.type == "cuda" and torch.cuda.is_available():
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                 else torch.float16)
        return "cuda", dtype, True
    return device.type, torch.float32, False


def parse_args():
    p = argparse.ArgumentParser(
        description="Learn TGN node embeddings via link-prediction pretraining.")
    p.add_argument("--cache-dir", type=str, required=True,
                   help="path to preprocessed cache (must contain node_feats.npy, "
                        "node_labels.npy, edges.npy, ts.npy, edge_feats.npy)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=512,
                   help="batch size for pretraining")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mem-dim", type=int, default=172)
    p.add_argument("--time-dim", type=int, default=172)
    p.add_argument("--emb-dim", type=int, default=172)
    p.add_argument("--num-neighbors", type=int, default=20)
    p.add_argument("--dataset-tag", type=str, default="elliptic_full")
    p.add_argument("--num-run", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def replay_full_graph(memory, neighbor_loader, data, device, bs: int):
    """Populate memory from the whole edge stream."""
    memory.eval()
    memory.reset_state(); neighbor_loader.reset_state()
    edges, ts = data["edges"], data["ts"]
    edge_raw = data["edge_raw"]
    RBS = bs * 20
    for k in tqdm(range(0, len(edges), RBS), desc="[replay]"):
        e = min(len(edges), k + RBS)
        src = torch.from_numpy(edges[k:e, 0]).long().to(device)
        dst = torch.from_numpy(edges[k:e, 1]).long().to(device)
        t   = torch.from_numpy(ts[k:e]).float().to(device)
        msg = edge_raw[k:e]
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)


def pretrain_tgn(models, data, device, args, save_path):
    """Full link-prediction pretraining with hard-negative mining and AMP."""
    memory, gnn, link_pred = models["memory"], models["gnn"], models["link_pred"]
    node_raw, edge_raw, ts_all = data["node_raw"], data["edge_raw"], data["ts_all"]
    assoc, neighbor_loader = data["assoc"], data["neighbor_loader"]
    edges, ts = data["edges"], data["ts"]
    train_idx, val_idx = data["train_edge_idx"], data["val_edge_idx"]
    test_idx = data["test_edge_idx"]

    opt = torch.optim.Adam(
        list(memory.parameters()) + list(gnn.parameters())
        + list(link_pred.parameters()), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    amp_dev, dtype, amp_enabled = _amp_spec(device)
    scaler = GradScaler(device=amp_dev, enabled=amp_enabled)

    valid_dst = np.unique(edges[train_idx, 1])
    NUM_NEG_EVAL = 20

    def _train_epoch():
        memory.train(); gnn.train(); link_pred.train()
        memory.reset_state(); neighbor_loader.reset_state()
        total = 0.0
        n = 0
        BS = args.bs
        for k in range(0, len(train_idx), BS):
            b = train_idx[k:k + BS]
            src = torch.from_numpy(edges[b, 0]).long().to(device)
            dst = torch.from_numpy(edges[b, 1]).long().to(device)
            t   = torch.from_numpy(ts[b]).float().to(device)
            msg = edge_raw[b]

            neg = np.random.choice(valid_dst, size=len(b), replace=True)
            neg = torch.from_numpy(neg).long().to(device)

            opt.zero_grad(set_to_none=True)
            n_id = torch.cat([src, dst, neg]).unique()
            n_id, ei, eid = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            with autocast(device_type=amp_dev, dtype=dtype,
                          enabled=amp_enabled):
                z, lu = memory(n_id)
                z_s = node_raw[n_id]
                z = gnn(torch.cat([z, z_s], dim=1), lu, ei,
                        ts_all[eid], edge_raw[eid])
                z_f = torch.cat([z, z_s], dim=1)

                pos = link_pred(z_f[assoc[src]], z_f[assoc[dst]])
                ng  = link_pred(z_f[assoc[src]], z_f[assoc[neg]])
                loss = criterion(pos, torch.ones_like(pos)) + \
                       criterion(ng, torch.zeros_like(ng))

            memory.update_state(src, dst, t, msg)
            neighbor_loader.insert(src, dst)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(memory.parameters()) + list(gnn.parameters())
                + list(link_pred.parameters()), max_norm=1.0)
            scaler.step(opt); scaler.update()
            memory.detach()
            total += float(loss.detach()) * len(b)
            n += len(b)
        return total / max(1, n)

    @torch.no_grad()
    def _eval_mrr():
        memory.eval(); gnn.eval(); link_pred.eval()
        memory.reset_state(); neighbor_loader.reset_state()
        # Warm up memory on train history
        BS = args.bs * 20
        for k in range(0, len(train_idx), BS):
            b = train_idx[k:k + BS]
            memory.update_state(
                torch.from_numpy(edges[b, 0]).long().to(device),
                torch.from_numpy(edges[b, 1]).long().to(device),
                torch.from_numpy(ts[b]).float().to(device),
                edge_raw[b])
            neighbor_loader.insert(
                torch.from_numpy(edges[b, 0]).long().to(device),
                torch.from_numpy(edges[b, 1]).long().to(device))

        mrrs = []
        for k in range(0, len(val_idx), args.bs):
            b = val_idx[k:k + args.bs]
            src = torch.from_numpy(edges[b, 0]).long().to(device)
            pos = torch.from_numpy(edges[b, 1]).long().to(device)
            t   = torch.from_numpy(ts[b]).float().to(device)
            msg = edge_raw[b]

            neg = np.random.choice(valid_dst, size=(len(b), NUM_NEG_EVAL),
                                   replace=True)
            neg_t = torch.from_numpy(neg).long().to(device)

            n_id = torch.cat([src, pos, neg_t.view(-1)]).unique()
            n_id, ei, eid = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, lu = memory(n_id)
            z_s = node_raw[n_id]
            z = gnn(torch.cat([z, z_s], dim=1), lu, ei,
                    ts_all[eid], edge_raw[eid])
            z_f = torch.cat([z, z_s], dim=1)

            ps = link_pred(z_f[assoc[src]], z_f[assoc[pos]]).squeeze(-1)
            z_src_e = z_f[assoc[src]].unsqueeze(1).expand(-1, NUM_NEG_EVAL, -1)
            z_neg = torch.zeros(len(b) * NUM_NEG_EVAL, z_f.size(-1),
                                device=device)
            flat = neg_t.view(-1)
            mask = assoc[flat] < z_f.size(0)
            if mask.any():
                z_neg[mask] = z_f[assoc[flat[mask]]]
            z_neg = z_neg.view(len(b), NUM_NEG_EVAL, -1)
            h = link_pred.lin_src(z_src_e) + link_pred.lin_dst(z_neg)
            ns = link_pred.lin_final(h.relu()).sigmoid().squeeze(-1)
            ranks = 1 + (ns >= ps.unsqueeze(1)).sum(dim=1).float()
            mrrs.append((1.0 / ranks).cpu())

            memory.update_state(src, pos, t, msg)
            neighbor_loader.insert(src, pos)
        return float(torch.cat(mrrs).mean())

    @torch.no_grad()
    def _eval_test_mrr_hits():
        """Evaluate on last-15% edges (ts > q85) after warming up on ts <= q70."""
        memory.eval(); gnn.eval(); link_pred.eval()
        memory.reset_state(); neighbor_loader.reset_state()
        # Warm up on train window only (ts <= q70).
        BS = args.bs * 20
        for k in range(0, len(train_idx), BS):
            b = train_idx[k:k + BS]
            memory.update_state(
                torch.from_numpy(edges[b, 0]).long().to(device),
                torch.from_numpy(edges[b, 1]).long().to(device),
                torch.from_numpy(ts[b]).float().to(device),
                edge_raw[b])
            neighbor_loader.insert(
                torch.from_numpy(edges[b, 0]).long().to(device),
                torch.from_numpy(edges[b, 1]).long().to(device))

        K_VALS = [1, 3, 10]
        mrrs = []
        hits = {k: [] for k in K_VALS}
        for k in range(0, len(test_idx), args.bs):
            b = test_idx[k:k + args.bs]
            src = torch.from_numpy(edges[b, 0]).long().to(device)
            pos = torch.from_numpy(edges[b, 1]).long().to(device)
            t   = torch.from_numpy(ts[b]).float().to(device)
            msg = edge_raw[b]

            neg = np.random.choice(valid_dst, size=(len(b), NUM_NEG_EVAL),
                                   replace=True)
            neg_t = torch.from_numpy(neg).long().to(device)

            n_id = torch.cat([src, pos, neg_t.view(-1)]).unique()
            n_id, ei, eid = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, lu = memory(n_id)
            z_s = node_raw[n_id]
            z = gnn(torch.cat([z, z_s], dim=1), lu, ei,
                    ts_all[eid], edge_raw[eid])
            z_f = torch.cat([z, z_s], dim=1)

            ps = link_pred(z_f[assoc[src]], z_f[assoc[pos]]).squeeze(-1)
            z_src_e = z_f[assoc[src]].unsqueeze(1).expand(-1, NUM_NEG_EVAL, -1)
            z_neg = torch.zeros(len(b) * NUM_NEG_EVAL, z_f.size(-1), device=device)
            flat = neg_t.view(-1)
            mask = assoc[flat] < z_f.size(0)
            if mask.any():
                z_neg[mask] = z_f[assoc[flat[mask]]]
            z_neg = z_neg.view(len(b), NUM_NEG_EVAL, -1)
            h = link_pred.lin_src(z_src_e) + link_pred.lin_dst(z_neg)
            ns = link_pred.lin_final(h.relu()).sigmoid().squeeze(-1)
            ranks = 1 + (ns >= ps.unsqueeze(1)).sum(dim=1).float()
            mrrs.append((1.0 / ranks).cpu())
            for kv in K_VALS:
                hits[kv].append((ranks <= kv).float().cpu())

            memory.update_state(src, pos, t, msg)
            neighbor_loader.insert(src, pos)

        mrr_val = float(torch.cat(mrrs).mean())
        hits_val = {kv: float(torch.cat(hits[kv]).mean()) for kv in K_VALS}
        return mrr_val, hits_val

    best_mrr = -1.0
    for ep in range(1, args.pretrain_epochs + 1):
        t0 = timeit.default_timer()
        loss = _train_epoch()
        mrr  = _eval_mrr()
        test_mrr, test_hits = _eval_test_mrr_hits()
        hits_str = "  ".join(f"H@{k}={test_hits[k]:.4f}" for k in [1, 3, 10])
        print(f"[pretrain] epoch {ep:02d}  loss={loss:.4f}  val-MRR={mrr:.4f}  "
              f"test-MRR={test_mrr:.4f}  {hits_str}  "
              f"time={timeit.default_timer() - t0:.1f}s")
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save({
                "memory_state_dict":    memory.state_dict(),
                "gnn_state_dict":       gnn.state_dict(),
                "link_pred_state_dict": link_pred.state_dict(),
                "val_mrr": mrr,
                "test_mrr": test_mrr,
                "test_hits": test_hits,
            }, save_path)
            print(f"  [pretrain] saved best → {save_path}")
    print(f"[pretrain] best val-MRR = {best_mrr:.4f}")


def load_pretrained(models, path, device):
    ck = torch.load(path, map_location=device)
    for key, mod in [("memory_state_dict", "memory"),
                     ("gnn_state_dict", "gnn"),
                     ("link_pred_state_dict", "link_pred")]:
        if key in ck:
            models[mod].load_state_dict(ck[key])
    print(f"[pretrain] loaded weights from {path}")


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

    ts_arr = ts
    q70 = int(np.quantile(ts_arr, 0.70))
    q85 = int(np.quantile(ts_arr, 0.85))
    train_edge_idx = np.where(ts_arr <= q70)[0]
    val_edge_idx   = np.where((ts_arr > q70) & (ts_arr <= q85))[0]
    test_edge_idx  = np.where(ts_arr  > q85)[0]
    print(f"[split] pretrain edge split — train:{len(train_edge_idx):,}  "
          f"val:{len(val_edge_idx):,}  test:{len(test_edge_idx):,}  "
          f"ts quantiles q70={q70} q85={q85}")

    models_dir = Path("saved_models") / args.dataset_tag
    models_dir.mkdir(parents=True, exist_ok=True)

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

        pretrain_data = {
            "node_raw": node_raw, "edge_raw": edge_raw, "ts_all": ts_all,
            "assoc": assoc, "neighbor_loader": neighbor_loader,
            "edges": edges, "ts": ts,
            "train_edge_idx": train_edge_idx, "val_edge_idx": val_edge_idx,
            "test_edge_idx": test_edge_idx,
        }

        pretrain_ckpt = str(models_dir /
                            f"TGN_PRETRAIN_{args.seed}_{run}.pth")

        print("[step] Link-prediction pretraining")
        pretrain_tgn(models, pretrain_data, device, args, pretrain_ckpt)

    print(f"\ntotal elapsed: {timeit.default_timer() - t0:.1f}s")


if __name__ == "__main__":
    main()
