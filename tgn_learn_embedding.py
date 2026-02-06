# Copyright (c) 2026, The RogueChainDB Authors
# All rights reserved.
#
# This source code is licensed under the BSD-style license.

import math
import timeit
import os
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import random

import torch
from torch.amp import autocast, GradScaler

# internal imports
from utils.utils import get_args, set_random_seed, save_results
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.decoder import LinkPredictor
from modules.early_stopping import EarlyStopMonitor
from tqdm import tqdm

# ==========
# ========== GPU Optimization Settings
# ==========

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

# ==========
# ========== Define helper function...
# ==========

def train():
    r"""
    Training procedure for TGN model with GPU optimizations
    """
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    # TGN Start of epoch: Reset state
    model['memory'].reset_state() 
    neighbor_loader.reset_state()

    total_loss = 0
    num_batches = 0
    
    # TGN Requirement: Training must be chronological to update memory correctly.
    
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(len(train_src_l), s_idx + BATCH_SIZE)
        
        batch_idx = np.arange(s_idx, e_idx)
        batch_size = len(batch_idx)
        
        # Move data to GPU
        src = torch.from_numpy(train_src_l[batch_idx]).long().to(device, non_blocking=True)
        pos_dst = torch.from_numpy(train_dst_l[batch_idx]).long().to(device, non_blocking=True)
        t = torch.from_numpy(train_ts_l[batch_idx]).float().to(device, non_blocking=True)
        msg = torch.from_numpy(train_e_feat_l[batch_idx]).float().to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Sample negative destination nodes
        neg_dst_np = np.random.choice(valid_dst_nodes, size=batch_size, replace=True)
        neg_dst = torch.from_numpy(neg_dst_np).long().to(device, non_blocking=True)

        # 1. Neighborhood Sampling
        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        
        # Map global node indices to local ones for the GNN
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        
        # Use mixed precision training
        with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            # 2. Memory Retrieval & Update
            z, last_update = model['memory'](n_id)

            z_static = node_raw_features[n_id]
            z_input = torch.cat([z, z_static], dim=1) # Concatenate: 128 + 11 = 139
            
            # 3. GNN Embedding
            z = model['gnn'](
                z_input,
                last_update,
                edge_index,
                ts_tensor[e_id],     # Time of sampled edges
                edge_raw_features[e_id], # Features of sampled edges
            )
            # === INSERT THIS (For Link Predictor) ===
            z_static_pred = node_raw_features[n_id]
            z_final = torch.cat([z, z_static_pred], dim=1) # Concatenate again for predictor
            # ========================================

            # 4. Prediction
            pos_out = model['link_pred'](z_final[assoc[src]], z_final[assoc[pos_dst]])
            neg_out = model['link_pred'](z_final[assoc[src]], z_final[assoc[neg_dst]])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))
        
        # 5. Update State (Critical for TGN)
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model['memory'].parameters()) + list(model['gnn'].parameters()) + list(model['link_pred'].parameters()), 
            max_norm=1.0
        )
        
        scaler.step(optimizer)
        scaler.update()
        
        # Detach memory for TBPTT
        model['memory'].detach()
        
        total_loss += loss.detach().float().item() * batch_size
        num_batches += batch_size
    
    return total_loss / num_batches


@torch.no_grad()
def test(src_l, dst_l, ts_l, e_feat_l, split_mode='val'):
    r"""
    Evaluated the dynamic link prediction with MRR metric.
    Includes 'Warm-up' (Replay) phase to update TGN Memory.
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()
    
    # Reset State
    model['memory'].reset_state()
    neighbor_loader.reset_state()
    
    # ==========================================
    # PHASE 1: WARM-UP (Replay History)
    # ==========================================
    
    replay_src, replay_dst, replay_ts, replay_msg = None, None, None, None
    
    if split_mode == 'val':
        # Replay TRAIN
        replay_src, replay_dst, replay_ts, replay_msg = train_src_l, train_dst_l, train_ts_l, train_e_feat_l
    elif split_mode == 'test':
        # Replay TRAIN + VAL
        replay_src = np.concatenate([train_src_l, val_src_l])
        replay_dst = np.concatenate([train_dst_l, val_dst_l])
        replay_ts = np.concatenate([train_ts_l, val_ts_l])
        replay_msg = np.concatenate([train_e_feat_l, val_e_feat_l])

    # Perform Replay in chunks
    if replay_src is not None:
        num_replay = len(replay_src)
        for k in range(0, num_replay, BATCH_SIZE):
            e_idx = min(num_replay, k + BATCH_SIZE)
            
            src_batch = torch.from_numpy(replay_src[k:e_idx]).long().to(device)
            dst_batch = torch.from_numpy(replay_dst[k:e_idx]).long().to(device)
            t_batch = torch.from_numpy(replay_ts[k:e_idx]).float().to(device)
            msg_batch = torch.from_numpy(replay_msg[k:e_idx]).float().to(device)
            
            # Update Memory and Graph
            model['memory'].update_state(src_batch, dst_batch, t_batch, msg_batch)
            neighbor_loader.insert(src_batch, dst_batch)
            
    # ==========================================
    # PHASE 2: EVALUATION (MRR Metric)
    # ==========================================
    
    TEST_BATCH_SIZE = 200
    num_test_instance = len(src_l)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    
    perf_list = []
    
    for k in tqdm(range(num_test_batch), desc=f"{split_mode} evaluation", disable=True):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
        batch_size = e_idx - s_idx
        
        # Get positive edges
        pos_src = torch.from_numpy(src_l[s_idx:e_idx]).long().to(device, non_blocking=True)
        pos_dst = torch.from_numpy(dst_l[s_idx:e_idx]).long().to(device, non_blocking=True)
        pos_t = torch.from_numpy(ts_l[s_idx:e_idx]).float().to(device, non_blocking=True)
        pos_msg = torch.from_numpy(e_feat_l[s_idx:e_idx]).float().to(device, non_blocking=True)
        
        # Evaluate each edge individually (one-vs-one)
        for idx in range(batch_size):
            src = pos_src[idx:idx+1]
            dst_pos = pos_dst[idx:idx+1]
            
            # Sample negative destination
            neg_dst_np = np.random.choice(valid_dst_nodes, size=1)
            dst_neg = torch.from_numpy(neg_dst_np).long().to(device)
            
            # Combine positive and negative destinations
            dst = torch.cat([dst_pos, dst_neg])
            src_expanded = src.expand(2)
            
            # 1. Neighborhood Sampling
            n_id = torch.cat([src_expanded, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
                # 2. Get Memory & Embeddings
                z, last_update = model['memory'](n_id)
                # === INSERT THIS ===
                z_static = node_raw_features[n_id]
                z_input = torch.cat([z, z_static], dim=1)

                z = model['gnn'](
                    z_input,
                    last_update,
                    edge_index,
                    ts_tensor[e_id],
                    edge_raw_features[e_id],
                )
                # === INSERT THIS ===
                z_static_pred = node_raw_features[n_id]
                z_final = torch.cat([z, z_static_pred], dim=1)
                # 3. Predict
                y_pred = model['link_pred'](z_final[assoc[src_expanded]], z_final[assoc[dst]])

            # Compute MRR: positive should rank higher than negative
            y_pred_pos = y_pred[0].item()
            y_pred_neg = y_pred[1].item()
            
            # MRR calculation: rank=1 if pos > neg, else rank=2
            if y_pred_pos > y_pred_neg:
                perf_list.append(1.0)  # MRR = 1/1
            else:
                perf_list.append(0.5)  # MRR = 1/2
        
        # 4. Post-batch Update (update memory with positive edges)
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)
    
    perf_metric = float(np.mean(perf_list)) if perf_list else 0.0
    return perf_metric

# ==========
# ==========
# ==========

# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()

print("INFO: Arguments:", args)
LR = args.lr
BATCH_SIZE = 200 
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10

MODEL_NAME = 'TGN'

# GPU optimization settings
USE_AMP = True
if torch.cuda.is_available():
    USE_BFLOAT16 = torch.cuda.is_bf16_supported()
else:
    USE_BFLOAT16 = False
AMP_DTYPE = torch.bfloat16 if USE_BFLOAT16 else torch.float16
print(f"INFO: Using mixed precision with dtype: {AMP_DTYPE}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========
# Load data
# ==========
print("Loading data...")
g_df = pd.read_csv('./processed/bitcoin_transactions.csv')
e_feat = np.load('./processed/bitcoin_transaction_features.npy')
n_feat = np.load('./processed/bitcoin_address_features.npy')

# Extract data
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values # These are the indices into e_feat
label_l = g_df.label.values
ts_l = g_df.ts.values

max_idx = max(src_l.max(), dst_l.max())
valid_dst_nodes = np.unique(dst_l)

# Split data
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

random.seed(SEED)

valid_train_flag = (ts_l <= val_time)
valid_val_flag = (ts_l <= test_time) & (ts_l > val_time)
valid_test_flag = ts_l > test_time

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
# Use edge indices to get features
train_e_feat_l = e_feat[e_idx_l[valid_train_flag]]

val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
# Use edge indices to get features
val_e_feat_l = e_feat[e_idx_l[valid_val_flag]]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
# Use edge indices to get features
test_e_feat_l = e_feat[e_idx_l[valid_test_flag]]
# ===================

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

print(f"=================*** {MODEL_NAME}: LinkPropPred ***=============")
print(f"Number of nodes: {max_idx + 1}")
print(f"Train edges: {len(train_src_l)}")
print(f"Val edges: {len(val_src_l)}")
print(f"Test edges: {len(test_src_l)}")

# For saving results
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_results.json'

MODEL_SAVE_PATH = f'./saved_models/{MODEL_NAME}.pth'
Path('./saved_models').mkdir(parents=True, exist_ok=True)

# Initialize features and move to GPU
node_raw_features = torch.from_numpy(n_feat).float().to(device)
edge_raw_features = torch.from_numpy(e_feat).float().to(device)
ts_tensor = torch.from_numpy(ts_l).float().to(device)

# === ADD THIS ===
NODE_FEAT_DIM = n_feat.shape[1]

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()
    
    # Set seed
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)
    np.random.seed(run_idx + SEED)
    
    # Neighborhood sampler
    neighbor_loader = LastNeighborLoader(max_idx + 1, size=NUM_NEIGHBORS, device=device)
    
    # Define TGN Model Components
    memory = TGNMemory(
        max_idx + 1,
        edge_raw_features.size(-1),
        MEM_DIM,
        TIME_DIM,
        message_module=IdentityMessage(edge_raw_features.size(-1), MEM_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM + NODE_FEAT_DIM,
        out_channels=EMB_DIM,
        msg_dim=edge_raw_features.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=EMB_DIM + NODE_FEAT_DIM).to(device)

    model = {'memory': memory,
            'gnn': gnn,
            'link_pred': link_pred}
    
    optimizer = torch.optim.Adam(
        set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
        lr=LR,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    
    assoc = torch.empty(max_idx + 1, dtype=torch.long, device=device)
    scaler = GradScaler(device='cuda', enabled=USE_AMP)
    
    # Early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(
        save_model_dir=save_model_dir, 
        save_model_id=save_model_id, 
        tolerance=TOLERANCE, 
        patience=PATIENCE
    )
    
    # ==================================================== Train & Validation
    val_perf_list = []
    start_train_val = timeit.default_timer()
    
    for epoch in range(1, NUM_EPOCH + 1):
        start_epoch_train = timeit.default_timer()
        loss = train()
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Time: {timeit.default_timer() - start_epoch_train: .4f}")
        
        start_val = timeit.default_timer()
        perf_metric_val = test(val_src_l, val_dst_l, val_ts_l, val_e_feat_l, split_mode="val")
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        print(f"\tValidation MRR: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_perf_list.append(perf_metric_val)
        
        checkpoint_path = f'{save_model_dir}/{save_model_id}_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'memory_state_dict': model['memory'].state_dict(),
            'gnn_state_dict': model['gnn'].state_dict(),
            'link_pred_state_dict': model['link_pred'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss,
            'val_mrr': perf_metric_val,
        }, checkpoint_path)
        print(f"\tModel saved to {checkpoint_path}")
                
        if early_stopper.step_check(perf_metric_val, model):
            break
    
    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")
    
    # ==================================================== Test
    early_stopper.load_checkpoint(model)
    
    start_test = timeit.default_timer()
    perf_metric_test = test(test_src_l, test_dst_l, test_ts_l, test_e_feat_l, split_mode="test")
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-ONE <<< ")
    print(f"\tTest MRR: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
    
    save_results({
        'model': MODEL_NAME,
        'run': run_idx,
        'seed': SEED,
        'val MRR': val_perf_list,
        'test MRR': perf_metric_test,
        'test_time': test_time,
        'tot_train_val_time': train_val_time
    }, results_filename)
    
    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
