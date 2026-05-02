
import argparse
import hashlib
import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ══════════════════════════════════════════════════════════════════════
# "Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics", KDD '23 — arXiv:2306.06108v1).
#
#   Figure 11b  — RF permutation importance on the Actors dataset:
#                 Top-10 most informative features are {Addr_interactions_total,
#                 Block_first_sent, Fees_median, Fees_max, Fees_mean,
#                 Block_last, Block_first, Fees_min, Fees_total,
#                 Block_first_receive}.
#   Figure 11b  — Bottom-10 (lowest importance, candidates to drop):
#                 Blocks_input_{median,total,min}, Addr_interactions_{mean,max,
#                 min,median}, Blocks_output_{mean,min}.
#   Figure 8/20 — Box plots flag Fees_share_{min,mean,median} as non-
#                 discriminative between licit/illicit distributions.
#
# The refined list below keeps paper's Top-10 plus BTC_received family
# (flagged as discriminative in Figure 20), volume aggregates, and basic
# structural features that temporal-graph models benefit from.
# ══════════════════════════════════════════════════════════════════════

# Exact Top-10 from Figure 11b (RF permutation importance, actors split)
TOP10_FEATURES_ACTORS = [
    "transacted_w_address_total",   # Addr_interactions_total
    "first_sent_block",             # Block_first_sent
    "fees_median",                  # Fees_median
    "fees_max",                     # Fees_max
    "fees_mean",                    # Fees_mean
    "last_block_appeared_in",       # Block_last
    "first_block_appeared_in",      # Block_first
    "fees_min",                     # Fees_min
    "fees_total",                   # Fees_total
    "first_received_block",         # Block_first_receive
]

# Paper Top-10 + BTC_received family + structural context (~27 feats)
REFINED_FEATURES_ACTORS = [
    # — Paper Top-10 —
    "transacted_w_address_total",
    "first_sent_block",
    "fees_median",
    "fees_max",
    "fees_mean",
    "last_block_appeared_in",
    "first_block_appeared_in",
    "fees_min",
    "fees_total",
    "first_received_block",
    # — BTC_received family (Fig. 20 green highlights) —
    "btc_received_min",
    "btc_received_max",
    "btc_received_total",
    "btc_received_mean",
    "btc_received_median",
    # — Volume / flow aggregates —
    "btc_transacted_total",
    "btc_transacted_max",
    "btc_sent_total",
    "btc_sent_max",
    # — Structural / temporal context —
    "lifetime_in_blocks",
    "num_timesteps_appeared_in",
    "total_txs",
    "num_txs_as_sender",
    "num_txs_as receiver",          # NB: raw CSV has a space, not '_'
    "num_addr_transacted_multiple",
    "blocks_btwn_txs_total",
    "blocks_btwn_txs_mean",
]

# Features flagged by the paper as noisy / non-discriminative — always drop
# in 'refined' and 'top10' modes.
PAPER_DROP_FEATURES_ACTORS = [
    # Bottom-10 (Fig. 11b)
    "blocks_btwn_input_txs_median",
    "blocks_btwn_input_txs_total",
    "blocks_btwn_input_txs_min",
    "transacted_w_address_mean",
    "transacted_w_address_max",
    "transacted_w_address_min",
    "transacted_w_address_median",
    "blocks_btwn_output_txs_mean",
    "blocks_btwn_output_txs_min",
    # Non-discriminative box-plots (Fig. 8/20)
    "fees_as_share_min",
    "fees_as_share_mean",
    "fees_as_share_median",
]


# ══════════════════════════════════════════════════════════════════════
# Preprocessing (runs once, then cached)  — EXACT EXTRACTION
# ══════════════════════════════════════════════════════════════════════
def _cache_signature(data_root: str, feature_mode: str = "refined",
                     feature_agg: str = "last",
                     add_topology_feats: bool = False) -> str:
    """mtime+size fingerprint of the three raw files + feature-mode tag.

    feature_mode/feature_agg/add_topology_feats are mixed in so changing
    any of them invalidates the cache — otherwise a 'refined'-mode cache
    could leak into an 'all'-mode run with a silently wrong feature matrix.
    """
    files = ["Actors Dataset/wallets_features.csv",
             "Actors Dataset/wallets_classes.csv",
             "Actors Dataset/addr1_addr2_with_timestamp.csv"]
    h = hashlib.md5()
    for f in files:
        p = osp.join(data_root, f)
        if osp.exists(p):
            h.update(f"{os.path.getmtime(p)}|{os.path.getsize(p)}".encode())
    h.update((f"|mode={feature_mode}|agg={feature_agg}"
              f"|topo={int(add_topology_feats)}").encode())
    return h.hexdigest()[:10]


def _select_feature_columns(all_cols: list, mode: str) -> list:
    """Apply paper-curated feature selection to the CSV columns.

    'refined' : paper Top-10 + BTC_received family + structural (kept in
                the declared order; unknown names skipped with a warning).
    'top10'   : paper Top-10 only.
    'all'     : every numeric column, minus features the paper flags as
                non-discriminative (PAPER_DROP_FEATURES_ACTORS).
    """
    present = set(all_cols)
    if mode == "top10":
        sel = [c for c in TOP10_FEATURES_ACTORS if c in present]
        missing = [c for c in TOP10_FEATURES_ACTORS if c not in present]
    elif mode == "refined":
        sel = [c for c in REFINED_FEATURES_ACTORS if c in present]
        missing = [c for c in REFINED_FEATURES_ACTORS if c not in present]
    elif mode == "all":
        drop = set(PAPER_DROP_FEATURES_ACTORS)
        sel = [c for c in all_cols if c not in drop]
        missing = []
    else:
        raise ValueError(f"unknown feature mode: {mode}")
    if missing:
        print(f"  [feature-mode={mode}] WARN: missing from CSV → {missing}")
    print(f"  [feature-mode={mode}] using {len(sel)} features: {sel}")
    return sel


def preprocess(data_root: str, cache_dir: str, force: bool = False,
               feature_mode: str = "refined",
               feature_agg: str = "last",
               add_topology_feats: bool = False) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig_file = cache_dir / "signature.txt"
    expected = _cache_signature(data_root, feature_mode, feature_agg,
                                add_topology_feats)
    out_files = ["node_feats.npy", "node_labels.npy",
                 "edges.npy", "ts.npy", "edge_feats.npy"]
    cache_hit = (not force
                 and sig_file.exists()
                 and sig_file.read_text().strip() == expected
                 and all((cache_dir / f).exists() for f in out_files))

    if cache_hit:
        print(f"[preprocess] Cache hit → {cache_dir}")
        return {
            "node_feats":  np.load(cache_dir / "node_feats.npy"),
            "node_labels": np.load(cache_dir / "node_labels.npy"),
            "edges":       np.load(cache_dir / "edges.npy"),
            "ts":          np.load(cache_dir / "ts.npy"),
            "edge_feats":  np.load(cache_dir / "edge_feats.npy"),
        }

    print(f"[preprocess] Building fresh cache at {cache_dir}  (2-6 min)")

    wf_path = osp.join(data_root, "Actors Dataset/wallets_features.csv")
    wc_path = osp.join(data_root, "Actors Dataset/wallets_classes.csv")
    ed_path = osp.join(data_root, "Actors Dataset/addr1_addr2_with_timestamp.csv")

    print("  loading wallets_features.csv")
    wf = pd.read_csv(wf_path)
    print("  loading wallets_classes.csv")
    wc = pd.read_csv(wc_path)
    print("  loading addr1_addr2_with_timestamp.csv")
    ed = pd.read_csv(ed_path)

    # ── ID mapper over the union of addresses ─────────────────────────
    addr_col_wf = wf.columns[0]
    addr_col_wc = wc.columns[0]
    src_col, dst_col = ed.columns[0], ed.columns[1]
    ts_col = "Time step" if "Time step" in ed.columns else ed.columns[4]

    all_addr = pd.concat([
        wf[addr_col_wf].astype(str),
        wc[addr_col_wc].astype(str),
        ed[src_col].astype(str),
        ed[dst_col].astype(str),
    ]).unique()
    id_mapper = pd.Series(
        np.arange(len(all_addr), dtype=np.int64), index=all_addr)
    N = len(id_mapper)
    print(f"  unique addresses: {N:,}")

    # ── Labels from wallets_classes.csv [FIX-1] ───────────────────────
    wc = wc.rename(columns={addr_col_wc: "address"})
    wc["address"] = wc["address"].astype(str)
    lab_map = {1: 1, 2: 0, 3: -1}
    wc["label"] = wc["class"].map(lab_map).fillna(-1).astype(np.int8)
    # Worst-case reduction: illicit > licit > unknown (if duplicates exist)
    red = wc.groupby("address")["label"].max()
    node_labels = np.full(N, -1, dtype=np.int8)
    red_idx = id_mapper.reindex(red.index).dropna().astype(np.int64).values
    valid = ~np.isnan(id_mapper.reindex(red.index).values)
    node_labels[red_idx] = red.values[valid]
    vc = pd.Series(node_labels).value_counts()
    print(f"  labels: licit={int(vc.get(0,0)):,}  "
          f"illicit={int(vc.get(1,0)):,}  unknown={int(vc.get(-1,0)):,}")

    # ── Paper-curated wallet features [FIX-2, paper Fig. 11b/20] ──────
    wf = wf.rename(columns={addr_col_wf: "address"})
    wf["address"] = wf["address"].astype(str)
    exclude = {"address", "Time step"}
    all_numeric = [c for c in wf.columns if c not in exclude]

    # Paper-aware selection (Top-10 / refined / all-minus-noisy)
    feat_cols = _select_feature_columns(all_numeric, feature_mode)

    wf[feat_cols] = wf[feat_cols].apply(pd.to_numeric,
                                        errors="coerce").fillna(0.0)
    print(f"  aggregating {len(feat_cols)} wallet features / address "
          f"(agg={feature_agg})")

    # Elliptic++ features are cumulative within each time-step's snapshot,
    # so 'last' (most recent time-step per address) mirrors the paper's
    # evaluation. 'mean' is kept as an alternative (original behaviour).
    # 'temporal_rich' concatenates [first, last, mean, std, delta, slope]
    # per feature to expose the behavioural dynamics the snapshots encode.
    wf_sorted = wf.sort_values(["address", "Time step"], kind="mergesort")
    if feature_agg == "last":
        agg = (wf_sorted.groupby("address", sort=False)[feat_cols]
                        .last().reset_index())
    elif feature_agg == "mean":
        agg = (wf_sorted.groupby("address", sort=False)[feat_cols]
                        .mean().reset_index())
    elif feature_agg == "temporal_rich":
        gb = wf_sorted.groupby("address", sort=False)
        first = gb[feat_cols].first().reset_index()
        last  = gb[feat_cols].last().reset_index()
        mean  = gb[feat_cols].mean().reset_index()
        std   = gb[feat_cols].std(ddof=0).fillna(0.0).reset_index()
        # delta = last - first (behavioural change over observed window)
        delta = last[feat_cols].values - first[feat_cols].values

        # Vectorised per-address least-squares slope over (Time step, v).
        ts_vec = wf_sorted["Time step"].astype(np.float64).values
        X = wf_sorted[feat_cols].values.astype(np.float64)
        # group row-counts → starts/ends per address (already sorted by addr)
        grp_sizes = gb.size().values
        grp_ends  = np.cumsum(grp_sizes)
        grp_starts = np.concatenate([[0], grp_ends[:-1]])
        slopes = np.zeros((len(grp_sizes), X.shape[1]), dtype=np.float64)
        # Σts, Σts² per group (same for every feature column)
        # Use np.add.reduceat for segmented reductions.
        sum_ts   = np.add.reduceat(ts_vec,        grp_starts)
        sum_ts2  = np.add.reduceat(ts_vec * ts_vec, grp_starts)
        n_i      = grp_sizes.astype(np.float64)
        denom    = n_i * sum_ts2 - sum_ts * sum_ts    # (G,)
        # Σv, Σ(ts·v) per group, per feature
        ts_bcast = ts_vec[:, None]               # (E, 1)
        sum_v  = np.add.reduceat(X,            grp_starts, axis=0)
        sum_tv = np.add.reduceat(X * ts_bcast, grp_starts, axis=0)
        num = n_i[:, None] * sum_tv - sum_ts[:, None] * sum_v
        safe = denom > 1e-9
        slopes[safe] = num[safe] / denom[safe, None]

        # Stitch wide frame in a single pd.concat — avoids the fragmentation
        # warning that per-block assignment on a wide frame triggers.
        first_cols = [c + "__first" for c in feat_cols]
        last_cols  = [c + "__last"  for c in feat_cols]
        mean_cols  = [c + "__mean"  for c in feat_cols]
        std_cols   = [c + "__std"   for c in feat_cols]
        delta_cols = [c + "__delta" for c in feat_cols]
        slope_cols = [c + "__slope" for c in feat_cols]
        addr_ser = first["address"].reset_index(drop=True)
        blocks = [
            addr_ser.rename("address").to_frame(),
            pd.DataFrame(first[feat_cols].values, columns=first_cols),
            pd.DataFrame(last[feat_cols].values,  columns=last_cols),
            pd.DataFrame(mean[feat_cols].values,  columns=mean_cols),
            pd.DataFrame(std[feat_cols].values,   columns=std_cols),
            pd.DataFrame(delta,                   columns=delta_cols),
            pd.DataFrame(slopes.astype(np.float64), columns=slope_cols),
        ]
        agg = pd.concat(blocks, axis=1, copy=False)
        feat_cols = (first_cols + last_cols + mean_cols
                     + std_cols + delta_cols + slope_cols)
        print(f"  temporal_rich expanded → {len(feat_cols)} columns "
              f"(6× per base feature)")
    else:
        raise ValueError(f"unknown feature_agg: {feature_agg}")
    ts_count = wf_sorted.groupby("address", sort=False).size().rename(
        "n_time_steps_observed").reset_index()
    agg = agg.merge(ts_count, on="address", how="left")
    feat_cols = feat_cols + ["n_time_steps_observed"]

    # signed log1p (robust against negatives like balance)
    vals = agg[feat_cols].values.astype(np.float64)
    signed_log = np.sign(vals) * np.log1p(np.abs(vals))

    mat = np.zeros((N, signed_log.shape[1]), dtype=np.float32)
    addr_new = id_mapper.reindex(agg["address"]).values.astype(np.int64)
    ok = addr_new >= 0
    mat[addr_new[ok]] = signed_log[ok].astype(np.float32)

    # ── Edges + timestamps [FIX-5] ────────────────────────────────────
    ed = ed.rename(columns={src_col: "src", dst_col: "dst", ts_col: "ts"})
    ed["src"] = ed["src"].astype(str)
    ed["dst"] = ed["dst"].astype(str)
    ed["ts"]  = pd.to_numeric(ed["ts"], errors="coerce").fillna(0).astype(np.int64)
    ed["src_id"] = id_mapper.reindex(ed["src"]).values
    ed["dst_id"] = id_mapper.reindex(ed["dst"]).values
    ed = ed.dropna(subset=["src_id", "dst_id"])
    ed["src_id"] = ed["src_id"].astype(np.int64)
    ed["dst_id"] = ed["dst_id"].astype(np.int64)

    # Normalise ts → 0..Tmax
    ts_min = int(ed["ts"].min())
    ed["ts"] = (ed["ts"] - ts_min).astype(np.int32)
    ed = ed.sort_values("ts", kind="mergesort").reset_index(drop=True)

    edges = ed[["src_id", "dst_id"]].values.astype(np.int64)
    ts_arr = ed["ts"].values.astype(np.int32)
    print(f"  edges: {len(edges):,}  ts range: 0..{int(ts_arr.max())}  "
          f"unique ts: {len(np.unique(ts_arr))}")

    # ── Optional: edge-list-derived topology features ─────────────────
    if add_topology_feats:
        src = edges[:, 0]
        dst = edges[:, 1]
        out_deg = np.bincount(src, minlength=N).astype(np.float64)
        in_deg  = np.bincount(dst, minlength=N).astype(np.float64)
        total_deg = out_deg + in_deg
        log_total = np.log1p(total_deg)
        out_in_ratio = out_deg / (in_deg + 1.0)

        # active_span / n_timesteps_with_edges / n_unique_counterparties
        # via grouping. Treat each endpoint participation as one "touch".
        endpoints = np.concatenate([src, dst])
        ts_touch  = np.concatenate([ts_arr, ts_arr]).astype(np.int64)
        counterp  = np.concatenate([dst, src]).astype(np.int64)

        # per-node min/max ts via ufunc.at (O(E), no sort needed)
        ts_min_node = np.full(N, np.iinfo(np.int64).max, dtype=np.int64)
        ts_max_node = np.full(N, np.iinfo(np.int64).min, dtype=np.int64)
        np.minimum.at(ts_min_node, endpoints, ts_touch)
        np.maximum.at(ts_max_node, endpoints, ts_touch)
        active = total_deg > 0
        active_span = np.zeros(N, dtype=np.float64)
        active_span[active] = (ts_max_node[active]
                               - ts_min_node[active]).astype(np.float64)

        # n_unique_counterparties and n_timesteps_with_edges via pandas
        # (vectorised, single pass — acceptable for 2.87M edges ×2).
        df_touch = pd.DataFrame({"node": endpoints, "ts": ts_touch,
                                 "cp": counterp})
        n_ts_per_node = (df_touch.groupby("node")["ts"]
                                 .nunique().reindex(range(N), fill_value=0)
                                 .values.astype(np.float64))
        n_cp_per_node = (df_touch.groupby("node")["cp"]
                                 .nunique().reindex(range(N), fill_value=0)
                                 .values.astype(np.float64))

        topo_cols = ["out_deg", "in_deg", "total_deg", "log_total_deg",
                     "out_in_ratio", "active_span",
                     "n_unique_counterparties", "n_timesteps_with_edges"]
        topo_raw = np.stack([out_deg, in_deg, total_deg, log_total,
                             out_in_ratio, active_span,
                             n_cp_per_node, n_ts_per_node], axis=1)
        topo_signed = (np.sign(topo_raw)
                       * np.log1p(np.abs(topo_raw))).astype(np.float32)
        print(f"  topology features: {topo_signed.shape[1]} cols "
              f"({topo_cols})")
        mat = np.hstack([mat, topo_signed]).astype(np.float32)
        feat_cols = feat_cols + topo_cols

    # Drop dead columns [FIX-3]
    std = mat.std(axis=0)
    keep = std > 1e-8
    dropped = [c for c, k in zip(feat_cols, keep) if not k]
    if dropped:
        print(f"  dropped {len(dropped)} dead feature cols: {dropped}")
    mat = mat[:, keep]
    feat_cols = [c for c, k in zip(feat_cols, keep) if k]

    # Outlier-robust scaling then clip
    rs = RobustScaler(quantile_range=(5, 95))
    mat = rs.fit_transform(mat)
    mat = np.clip(mat, -5.0, 5.0).astype(np.float32)
    print(f"  node feature matrix: {mat.shape}  dtype={mat.dtype}")

    # ── Edge feature: log1p(btc_transacted_total) of the source ───────
    btc_col = None
    for c in agg.columns:
        cl = c.lower()
        if "btc" in cl and "transact" in cl and "total" in cl:
            btc_col = c
            break
    if btc_col is None:
        for c in agg.columns:
            if "btc" in c.lower() and "total" in c.lower():
                btc_col = c
                break

    edge_feats = np.zeros((len(edges), 1), dtype=np.float32)
    if btc_col is not None:
        btc_vals = np.log1p(np.abs(
            pd.to_numeric(agg[btc_col], errors="coerce").fillna(0).values))
        btc_lookup = np.zeros(N, dtype=np.float32)
        btc_lookup[addr_new[ok]] = btc_vals[ok].astype(np.float32)
        edge_feats[:, 0] = btc_lookup[edges[:, 0]]
        print(f"  edge feature: log1p(|{btc_col}|) of source")
    else:
        print("  [WARN] no BTC column found — edge feature set to 0")

    # ── Save cache ────────────────────────────────────────────────────
    np.save(cache_dir / "node_feats.npy",  mat)
    np.save(cache_dir / "node_labels.npy", node_labels)
    np.save(cache_dir / "edges.npy",       edges)
    np.save(cache_dir / "ts.npy",          ts_arr)
    np.save(cache_dir / "edge_feats.npy",  edge_feats)
    id_mapper.reset_index().rename(
        columns={"index": "address", 0: "new_id"}).to_csv(
        cache_dir / "id_mapper.csv", index=False)
    pd.Series(feat_cols, name="feature").to_csv(
        cache_dir / "feature_names.csv", index=False)
    sig_file.write_text(expected)
    print(f"[preprocess] wrote cache to {cache_dir}")

    return {"node_feats": mat, "node_labels": node_labels,
            "edges": edges, "ts": ts_arr, "edge_feats": edge_feats}


# ══════════════════════════════════════════════════════════════════════
# RogueChainDB loader (alternative dataset; same output schema) — EXACT
# ══════════════════════════════════════════════════════════════════════
def _rdb_build_node_labels(data_dir: str, N: int) -> np.ndarray:
    """Rebuild per-node labels from the raw RogueChainDB source files.

    Uses address_adress_interactions.parquet (to reconstruct the original
    id_mapper), address_identifier.pickle (original_id → address string),
    and address_labels.csv (address → 0/1) — exactly mirroring process.py.

    Result is cached to <data_dir>/rdb_node_labels.npy so the 387 MB parquet
    is only parsed once.
    """
    import pickle as _pkl

    cache_path = osp.join(data_dir, "rdb_node_labels.npy")
    if osp.exists(cache_path):
        print(f"[roguechaindb] label cache hit → {cache_path}")
        return np.load(cache_path)

    parquet_path  = osp.join(data_dir, "address_adress_interactions.parquet")
    ident_path    = osp.join(data_dir, "address_identifier.pickle")
    labels_path   = osp.join(data_dir, "address_labels.csv")

    print("[roguechaindb] rebuilding node labels from source files "
          "(one-time, ~1 min) ...")
    print("  loading parquet ...")
    df_p = pd.read_parquet(parquet_path,
                           columns=["addr_id1", "addr_id2"])
    unique_nodes = pd.concat(
        [df_p["addr_id1"], df_p["addr_id2"]]).unique()
    unique_nodes.sort()
    id_mapper = {int(old): int(new)
                 for new, old in enumerate(unique_nodes)}
    del df_p
    print(f"  id_mapper built: {len(id_mapper):,} nodes")

    print("  loading address identifier ...")
    with open(ident_path, "rb") as f:
        ident = _pkl.load(f)

    print("  loading address labels ...")
    lbl_df = pd.read_csv(labels_path)
    addr_to_label = dict(zip(lbl_df["address"], lbl_df["label_int"]))
    del lbl_df

    print("  assigning labels ...")
    node_labels = np.full(N, -1, dtype=np.int8)
    for orig_id, addr in ident.items():
        new_id = id_mapper.get(int(orig_id))
        if new_id is not None and new_id < N:
            lab = addr_to_label.get(addr)
            if lab is not None:
                node_labels[new_id] = np.int8(lab)

    np.save(cache_path, node_labels)
    print(f"[roguechaindb] label cache saved → {cache_path}")
    return node_labels


def load_roguechaindb(transactions_path: str,
                      node_features_path: str,
                      edge_features_path: str,
                      ts_bins: int = 0,
                      ts_mode: str = "raw") -> dict:
    """Load the RogueChainDB artefacts into the same dict schema as preprocess().

    Input files:
      * bitcoin_transactions.csv   header-less, columns: u, i, ts, label, idx
      * bitcoin_address_features.npy   shape (N, F_node)
      * bitcoin_edge_features.npy      shape (E, F_edge)

    Label derivation strategy (two-tier):
      1. PREFERRED — if the raw source files are in the same directory
         (address_adress_interactions.parquet + address_identifier.pickle +
          address_labels.csv), rebuild the exact per-node mapping used by
          process.py. Result cached to rdb_node_labels.npy for fast reuse.
      2. FALLBACK  — if the raw files are absent, derive labels from the
         'label' column in the transactions CSV. That column encodes the
         SOURCE node's class (0/1/-1), so only the u side is used; nodes
         that only appear as destinations remain -1.
    ts_mode controls timestamp discretisation (see --rdb-ts-mode help):
      "raw"              : shifted integer block heights (default).
      "quantile"         : equal-population bins (--rdb-ts-bins required).
      "calendar2w"       : fixed 2016-block bins (= 2 weeks by Bitcoin
                           difficulty-retarget schedule), same semantics as
                           Elliptic++ time steps. Produces ~428 bins over the
                           full dataset.
      "calendar2w_dense" : same 2016-block bins but restricted to the densest
                           consecutive window of --rdb-ts-bins bins (default 49,
                           same count as Elliptic++). Edges outside the window
                           are dropped; bins are renumbered 0..W-1.

    Returns
    -------
    dict with keys compatible with preprocess(): node_feats, node_labels,
    edges, ts (int32 shifted to start at 0), edge_feats.
    """
    print(f"[roguechaindb] loading transactions: {transactions_path}")
    g_df = pd.read_csv(transactions_path, header=None,
                       names=["u", "i", "ts", "label", "idx"],
                       dtype=np.float64)
    g_df.dropna(inplace=True)
    g_df = g_df.astype({"u": np.int64, "i": np.int64,
                        "ts": np.float64,
                        "label": np.int8, "idx": np.int64})

    print(f"[roguechaindb] loading node features: {node_features_path}")
    n_feat = np.load(node_features_path)
    if n_feat.ndim == 1:
        n_feat = n_feat[:, None]
    n_feat = n_feat.astype(np.float32, copy=False)

    print(f"[roguechaindb] loading edge features: {edge_features_path}")
    e_feat = np.load(edge_features_path)
    if e_feat.ndim == 1:
        e_feat = e_feat[:, None]
    e_feat = e_feat.astype(np.float32, copy=False)

    N = int(n_feat.shape[0])
    max_id = int(max(g_df["u"].max(), g_df["i"].max()))
    if max_id >= N:
        raise ValueError(
            f"node feature matrix has N={N} rows but transactions reference "
            f"node id={max_id}. Check that bitcoin_address_features.npy was "
            f"generated from the same pipeline as bitcoin_transactions.csv.")

    # ── Label derivation ──────────────────────────────────────────────
    data_dir = osp.dirname(osp.abspath(transactions_path))
    _required = ["address_adress_interactions.parquet",
                 "address_identifier.pickle",
                 "address_labels.csv"]
    if all(osp.exists(osp.join(data_dir, f)) for f in _required):
        node_labels = _rdb_build_node_labels(data_dir, N)
    else:
        print("[roguechaindb] raw source files not found — falling back to "
              "source-side CSV labels (recovers ~43% of labeled nodes).")
        node_labels = np.full(N, -1, dtype=np.int8)
        src_labels = (g_df[g_df["label"] != -1][["u", "label"]]
                      .drop_duplicates("u"))
        valid = src_labels["u"].values < N
        node_labels[src_labels["u"].values[valid]] = (
            src_labels["label"].values[valid].astype(np.int8))

    vc = pd.Series(node_labels).value_counts()
    print(f"[roguechaindb] labels: licit={int(vc.get(0,0)):,}  "
          f"illicit={int(vc.get(1,0)):,}  unknown={int(vc.get(-1,0)):,}")

    # Sort edges chronologically.
    g_df = g_df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    edges = g_df[["u", "i"]].values.astype(np.int64)
    ts_arr = g_df["ts"].values
    ts_min_raw = float(ts_arr.min())

    # ── Timestamp discretisation ──────────────────────────────────────
    # Bitcoin block heights are integers; shift so the first edge is t=0.
    ts_shift = ts_arr - ts_min_raw

    if ts_mode == "raw":
        # Raw shifted block heights. Integer-valued — keep int32 directly.
        if np.allclose(ts_shift, np.round(ts_shift)):
            ts_int = np.round(ts_shift).astype(np.int32)
        else:
            ts_int = (ts_shift / max(ts_shift.max(), 1.0) * 1000.0
                      ).astype(np.int32)

    elif ts_mode == "quantile":
        # Equal-population bins — same as old --rdb-ts-bins behaviour.
        # ts_bins must be > 0.
        if ts_bins <= 0:
            raise ValueError("--rdb-ts-mode quantile requires --rdb-ts-bins > 0")
        ts_shift_i = (np.round(ts_shift).astype(np.int64)
                      if np.allclose(ts_shift, np.round(ts_shift))
                      else (ts_shift / max(ts_shift.max(), 1.0) * 1000.0
                            ).astype(np.int64))
        quantiles = np.linspace(0.0, 1.0, ts_bins + 1)
        bin_edges = np.quantile(ts_shift_i, quantiles)
        ts_int = (np.searchsorted(bin_edges[1:], ts_shift_i, side="right")
                  .clip(0, ts_bins - 1).astype(np.int32))
        print(f"[roguechaindb] ts quantile-bucketed → {ts_bins} bins "
              f"(range 0..{ts_int.max()}, unique={len(np.unique(ts_int))})")

    elif ts_mode in ("calendar2w", "calendar2w_dense"):
        # ── Calendar-time bins matching Elliptic++ semantics ─────────
        # Bitcoin's difficulty retarget happens every 2016 blocks, designed
        # to keep block time at exactly 10 min → 2016 blocks ≈ 2 weeks.
        # Elliptic++ uses one "Time step" per 2-week period (steps 1..49).
        # Using the same bin width makes temporal granularity directly
        # comparable between the two datasets.
        BLOCKS_PER_BIN = 2016  # 2016 blocks = 1 Bitcoin retarget period = ~2 weeks
        # Assign each edge to its 2-week bin (0-indexed from ts_min_raw).
        raw_bins = (np.floor(ts_shift / BLOCKS_PER_BIN)
                    .astype(np.int32))  # 0-indexed, range 0..~427

        if ts_mode == "calendar2w":
            # Keep all bins; renumber so the minimum bin is 0.
            ts_int = (raw_bins - raw_bins.min()).astype(np.int32)
            n_unique = len(np.unique(ts_int))
            print(f"[roguechaindb] ts calendar2w → {n_unique} bins of 2016 blocks "
                  f"(range 0..{int(ts_int.max())})")

        else:  # calendar2w_dense
            # Restrict to the densest consecutive W-bin window (default W=49).
            W = ts_bins if ts_bins > 0 else 49
            # Count edges per bin via bincount.
            n_raw_bins = int(raw_bins.max()) + 1
            counts = np.bincount(raw_bins, minlength=n_raw_bins).astype(np.int64)
            # Sliding window sum to find densest W consecutive bins.
            cumsum = np.concatenate([[0], np.cumsum(counts)])
            window_sums = cumsum[W:] - cumsum[:len(cumsum) - W]
            best_start = int(np.argmax(window_sums))
            best_end   = best_start + W - 1  # inclusive
            n_kept = int(window_sums[best_start])
            block_lo = int(ts_min_raw + best_start * BLOCKS_PER_BIN)
            block_hi = int(ts_min_raw + (best_end + 1) * BLOCKS_PER_BIN)
            print(f"[roguechaindb] ts calendar2w_dense → window bins "
                  f"{best_start}..{best_end} (W={W}, blocks {block_lo}..{block_hi})")
            print(f"  edges kept: {n_kept:,} / {len(edges):,} "
                  f"({100*n_kept/len(edges):.1f}%)")

            # Keep only edges inside the window.
            mask = (raw_bins >= best_start) & (raw_bins <= best_end)
            edges  = edges[mask]
            e_feat = e_feat[mask]  # reindex edge features below if needed
            g_df   = g_df.iloc[mask].reset_index(drop=True)
            ts_int = (raw_bins[mask] - best_start).astype(np.int32)
            print(f"  ts range 0..{int(ts_int.max())}  unique bins: "
                  f"{len(np.unique(ts_int))}")

    else:
        raise ValueError(f"unknown ts_mode: {ts_mode!r}")

    print(f"[roguechaindb] edges: {len(edges):,}  "
          f"ts range: 0..{int(ts_int.max())}  unique ts: {len(np.unique(ts_int))}")

    # If edge-feat row count doesn't match, try to reindex by the 'idx' column.
    if e_feat.shape[0] != len(edges):
        idx_col = g_df["idx"].values
        if e_feat.shape[0] > int(idx_col.max()):
            e_feat = e_feat[idx_col]
            print(f"[roguechaindb] edge features reindexed via 'idx' column "
                  f"→ new shape {e_feat.shape}")
        else:
            raise ValueError(
                f"edge feature rows ({e_feat.shape[0]}) != #edges "
                f"({len(edges)}) and 'idx' column cannot resolve it.")

    return {"node_feats": n_feat, "node_labels": node_labels,
            "edges": edges, "ts": ts_int, "edge_feats": e_feat}


# ══════════════════════════════════════════════════════════════════════
# CLI — choose the dataset, everything else mirrors tgn_ranc_elliptic.py
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: preprocess Elliptic++ or RogueChainDB.")
    # Data
    p.add_argument("--dataset", choices=["elliptic", "roguechaindb"],
                   default="roguechaindb",
                   help="elliptic = raw Elliptic++ CSVs (default); "
                        "roguechaindb = bitcoin_transactions.csv + two .npy files")
    p.add_argument("--data-root", type=str,
                   default="./EllipticPlusPlus",
                   help="Path to Elliptic++ root directory (containing 'Actors Dataset/').")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="preprocess output dir (default: <data-root>/full_cache_<mode>_<agg>)")
    p.add_argument("--force-preprocess", action="store_true")
    # RogueChainDB paths
    p.add_argument("--rdb-transactions", type=str,
                   default="./RogueChainDB/bitcoin_transactions.csv",
                   help="header-less CSV: u, i, ts, label, idx")
    p.add_argument("--rdb-node-features", type=str,
                   default="./RogueChainDB/bitcoin_address_features.npy")
    p.add_argument("--rdb-edge-features", type=str,
                   default="./RogueChainDB/bitcoin_edge_features.npy")
    p.add_argument("--rdb-ts-bins", type=int, default=0,
                   help="If > 0, quantile-bucket RogueChainDB block-height "
                        "timestamps into this many equal-population bins. "
                        "0 = raw shifted block heights.")
    p.add_argument("--rdb-ts-mode",
                   choices=["raw", "quantile", "calendar2w", "calendar2w_dense"],
                   default="raw",
                   help="Timestamp discretisation for RogueChainDB.")
    # Feature selection (Elliptic only)
    p.add_argument("--feature-mode",
                   choices=["refined", "top10", "all"],
                   default="refined",
                   help="'refined' = paper Top-10 + BTC_received + structural; "
                        "'top10' = paper Top-10 only; "
                        "'all' = every numeric wallet feature minus dead cols")
    p.add_argument("--feature-agg",
                   choices=["last", "mean", "temporal_rich"],
                   default="last",
                   help="aggregation across an address's time-steps.")
    p.add_argument("--add-topology-feats", action="store_true",
                   help="Append edge-list-derived topology features.")
    return p.parse_args()


def main():
    args = parse_args()
    print("INFO: args =", vars(args))

    if args.dataset == "roguechaindb":
        if args.cache_dir is None:
            args.cache_dir = osp.join(
                osp.dirname(osp.abspath(args.rdb_transactions)),
                f"full_cache_rdb_{args.rdb_ts_mode}")
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        out_files = ["node_feats.npy", "node_labels.npy",
                     "edges.npy", "ts.npy", "edge_feats.npy"]
        cache_hit = (not args.force_preprocess
                     and all((cache_dir / f).exists() for f in out_files))
        if cache_hit:
            print(f"[preprocess] RDB cache hit → {cache_dir}")
        else:
            print(f"[data] dataset=roguechaindb  (ts_mode={args.rdb_ts_mode}, "
                  f"ts_bins={args.rdb_ts_bins})")
            cache = load_roguechaindb(args.rdb_transactions,
                                      args.rdb_node_features,
                                      args.rdb_edge_features,
                                      ts_bins=args.rdb_ts_bins,
                                      ts_mode=args.rdb_ts_mode)
            np.save(cache_dir / "node_feats.npy",  cache["node_feats"])
            np.save(cache_dir / "node_labels.npy", cache["node_labels"])
            np.save(cache_dir / "edges.npy",       cache["edges"])
            np.save(cache_dir / "ts.npy",          cache["ts"])
            np.save(cache_dir / "edge_feats.npy",  cache["edge_feats"])
            print(f"[preprocess] wrote RDB cache to {cache_dir}")
    else:
        print(f"[data] dataset=elliptic  (feature_mode={args.feature_mode}, "
              f"feature_agg={args.feature_agg}, "
              f"topo={args.add_topology_feats})")
        topo_tag = "_topo" if args.add_topology_feats else ""
        default_cache = osp.join(
            args.data_root,
            f"full_cache_{args.feature_mode}_{args.feature_agg}{topo_tag}")
        cache_dir = args.cache_dir or default_cache
        preprocess(args.data_root, cache_dir,
                   force=args.force_preprocess,
                   feature_mode=args.feature_mode,
                   feature_agg=args.feature_agg,
                   add_topology_feats=args.add_topology_feats)
        args.cache_dir = cache_dir

    print(f"✓ Preprocessing complete → {args.cache_dir}")


if __name__ == "__main__":
    main()
