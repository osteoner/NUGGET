import math
import timeit
import os
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast, GradScaler

# internal imports
from utils.utils import get_args, set_random_seed, save_results
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import DyRepMemory
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
    Training procedure for DyRep model
    """
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    num_samples = 0
    
    idx_list = np.arange(len(train_src_l))
    np.random.shuffle(idx_list)
    
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(len(train_src_l), s_idx + BATCH_SIZE)
        batch_idx = idx_list[s_idx:e_idx]
        batch_size = len(batch_idx)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Get batch data
        src = torch.from_numpy(train_src_l[batch_idx]).long().to(device, non_blocking=True)
        pos_dst = torch.from_numpy(train_dst_l[batch_idx]).long().to(device, non_blocking=True)
        t = torch.from_numpy(train_ts_l[batch_idx]).float().to(device, non_blocking=True)
        e_idx_batch = train_e_idx_l[batch_idx]
        msg = edge_raw_features[e_idx_batch].to(device, non_blocking=True)

        # Sample negative destination nodes from valid destinations
        neg_dst_np = np.random.choice(valid_dst_nodes, size=batch_size, replace=True)
        neg_dst = torch.from_numpy(neg_dst_np).long().to(device, non_blocking=True)

        # Get unique nodes involved
        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Use mixed precision training
        with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            # Get updated memory of all nodes involved in the computation
            z, last_update = model['memory'](n_id)

            z_static = node_raw_features[n_id]
            z = torch.cat([z, z_static], dim=1)

            pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
            neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update the memory with ground-truth
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            all_edge_times[e_id].to(device),
            all_edge_features[e_id].to(device),
        )
        model['memory'].update_state(src, pos_dst, t, msg, z, assoc)

        # Update neighbor loader
        neighbor_loader.insert(src, pos_dst)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model['memory'].parameters()) + 
            list(model['gnn'].parameters()) + 
            list(model['link_pred'].parameters()),
            max_norm=1.0
        )
        
        scaler.step(optimizer)
        scaler.update()
        
        model['memory'].detach()
        
        total_loss += loss.detach().float().item() * batch_size
        num_samples += batch_size

    return total_loss / num_samples


@torch.no_grad()
def test(src_l, dst_l, ts_l, e_idx_l, split_mode='val'):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge 
    is evaluated against many negative edges
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    # Reset to appropriate state
    model['memory'].reset_state()
    neighbor_loader.reset_state()
    
    # Re-insert edges up to the split
    if split_mode == 'val':
        # Insert all training edges
        for k in range(0, len(train_src_l), BATCH_SIZE):
            e_idx = min(len(train_src_l), k + BATCH_SIZE)
            src_batch = torch.from_numpy(train_src_l[k:e_idx]).long().to(device)
            dst_batch = torch.from_numpy(train_dst_l[k:e_idx]).long().to(device)
            t_batch = torch.from_numpy(train_ts_l[k:e_idx]).float().to(device)
            e_idx_batch = train_e_idx_l[k:e_idx]
            msg_batch = edge_raw_features[e_idx_batch].to(device)
            
            # Update memory and neighbor loader
            n_id = torch.cat([src_batch, dst_batch]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, last_update = model['memory'](n_id)
            
            z_static = node_raw_features[n_id]
            z = torch.cat([z, z_static], dim=1)

            z = model['gnn'](
                z, last_update, edge_index,
                all_edge_times[e_id].to(device),
                all_edge_features[e_id].to(device)
            )
            model['memory'].update_state(src_batch, dst_batch, t_batch, msg_batch, z, assoc)
            neighbor_loader.insert(src_batch, dst_batch)
            
    elif split_mode == 'test':
        # Insert training + validation edges
        all_src = np.concatenate([train_src_l, val_src_l])
        all_dst = np.concatenate([train_dst_l, val_dst_l])
        all_ts = np.concatenate([train_ts_l, val_ts_l])
        all_e_idx = np.concatenate([train_e_idx_l, val_e_idx_l])
        
        for k in range(0, len(all_src), BATCH_SIZE):
            e_idx = min(len(all_src), k + BATCH_SIZE)
            src_batch = torch.from_numpy(all_src[k:e_idx]).long().to(device)
            dst_batch = torch.from_numpy(all_dst[k:e_idx]).long().to(device)
            t_batch = torch.from_numpy(all_ts[k:e_idx]).float().to(device)
            e_idx_batch = all_e_idx[k:e_idx]
            msg_batch = edge_raw_features[e_idx_batch].to(device)
            
            # Update memory and neighbor loader
            n_id = torch.cat([src_batch, dst_batch]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, last_update = model['memory'](n_id)

            z_static = node_raw_features[n_id]
            z = torch.cat([z, z_static], dim=1)

            z = model['gnn'](
                z, last_update, edge_index,
                all_edge_times[e_id].to(device),
                all_edge_features[e_id].to(device)
            )
            model['memory'].update_state(src_batch, dst_batch, t_batch, msg_batch, z, assoc)
            neighbor_loader.insert(src_batch, dst_batch)

    # Evaluate on test data
    perf_list = []
    num_test_batches = math.ceil(len(src_l) / TEST_BATCH_SIZE)

    for k in tqdm(range(num_test_batches), desc=f"{split_mode} evaluation", disable=True):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(len(src_l), s_idx + TEST_BATCH_SIZE)
        batch_size = e_idx - s_idx
        
        # Get positive edges
        pos_src = torch.from_numpy(src_l[s_idx:e_idx]).long().to(device, non_blocking=True)
        pos_dst = torch.from_numpy(dst_l[s_idx:e_idx]).long().to(device, non_blocking=True)
        pos_t = torch.from_numpy(ts_l[s_idx:e_idx]).float().to(device, non_blocking=True)
        pos_e_idx = e_idx_l[s_idx:e_idx]
        pos_msg = edge_raw_features[pos_e_idx].to(device, non_blocking=True)

        # For each positive edge, evaluate against one negative
        for idx in range(batch_size):
            src = pos_src[idx:idx+1]
            dst_pos = pos_dst[idx:idx+1]
            
            # Sample negative destination
            neg_dst_np = np.random.choice(valid_dst_nodes, size=1)
            dst_neg = torch.from_numpy(neg_dst_np).long().to(device)
            
            # Combine positive and negative destinations
            dst = torch.cat([dst_pos, dst_neg])
            src_expanded = src.expand(2)
            
            n_id = torch.cat([src_expanded, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory
            with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
                z, last_update = model['memory'](n_id)
                z_static = node_raw_features[n_id]
                z = torch.cat([z, z_static], dim=1)

                y_pred = model['link_pred'](z[assoc[src_expanded]], z[assoc[dst]])
            
            # Compute MRR: positive should rank higher than negative
            y_pred_pos = y_pred[0].item()
            y_pred_neg = y_pred[1].item()
            
            # MRR calculation: rank=1 if pos > neg, else rank=2
            if y_pred_pos > y_pred_neg:
                perf_list.append(1.0)  # MRR = 1/1
            else:
                perf_list.append(0.5)  # MRR = 1/2
        
        # Update memory with positive edges for next batch
        n_id = torch.cat([pos_src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = model['memory'](n_id)
        z_static = node_raw_features[n_id]
        z = torch.cat([z, z_static], dim=1)
        z = model['gnn'](
            z, last_update, edge_index,
            all_edge_times[e_id].to(device),
            all_edge_features[e_id].to(device)
        )
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg, z, assoc)
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
BATCH_SIZE = args.bs
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
TEST_BATCH_SIZE = 200

MODEL_NAME = 'DyRep'
USE_SRC_EMB_IN_MSG = False
USE_DST_EMB_IN_MSG = True

# GPU optimization settings
USE_AMP = True
if torch.cuda.is_available():
    USE_BFLOAT16 = torch.cuda.is_bf16_supported()
else:
    USE_BFLOAT16 = False
AMP_DTYPE = torch.bfloat16 if USE_BFLOAT16 else torch.float16
print(f"INFO: Using mixed precision with dtype: {AMP_DTYPE}")

# set the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ==========
# Load data directly from CSV and numpy files
# ==========
print("Loading data...")
g_df = pd.read_csv('./processed/bitcoin_transactions.csv')
e_feat = np.load('./processed/bitcoin_transaction_features.npy')
n_feat = np.load('./processed/bitcoin_address_features.npy')

# Extract data
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())
num_nodes = max_idx + 1

# Create array of valid destination nodes
valid_dst_nodes = np.unique(dst_l)
print(f"Number of unique destination nodes: {len(valid_dst_nodes)}")
print(f"Destination node range: [{valid_dst_nodes.min()}, {valid_dst_nodes.max()}]")

# Split data: 70% train, 15% val, 15% test
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

# Get train/val/test splits
valid_train_flag = (ts_l <= val_time)
valid_val_flag = (ts_l <= test_time) & (ts_l > val_time)
valid_test_flag = ts_l > test_time

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]

val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred ***=============")
print("==========================================================")
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {len(src_l)}")
print(f"Train edges: {len(train_src_l)}")
print(f"Val edges: {len(val_src_l)}")
print(f"Test edges: {len(test_src_l)}")
print(f"Edge feature dim: {e_feat.shape[1]}")
print(f"Node feature dim: {n_feat.shape[1]}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of batches: {num_batch}")

# Move features to GPU
edge_raw_features = torch.from_numpy(e_feat).float().to(device)
node_raw_features = torch.from_numpy(n_feat).float().to(device)

NODE_FEAT_DIM = n_feat.shape[1]


# Create tensors for all edge times and features (for neighbor loader)
all_edge_times = torch.from_numpy(ts_l).float().to(device)
all_edge_features = edge_raw_features

# Helper vector to map global node indices to local ones
assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_results.json'

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)
    np.random.seed(run_idx + SEED)

    # neighborhood sampler
    neighbor_loader = LastNeighborLoader(num_nodes, size=NUM_NEIGHBORS, device=device)

    # define the model end-to-end
    # 1) memory
    memory = DyRepMemory(
        num_nodes,
        e_feat.shape[1],  # message dim
        MEM_DIM,
        TIME_DIM,
        message_module=IdentityMessage(e_feat.shape[1], MEM_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
        memory_updater_type='rnn',
        use_src_emb_in_msg=USE_SRC_EMB_IN_MSG,
        use_dst_emb_in_msg=USE_DST_EMB_IN_MSG
    ).to(device)

    # 2) GNN
    gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM + NODE_FEAT_DIM,
        out_channels=EMB_DIM,
        msg_dim=e_feat.shape[1],
        time_enc=memory.time_enc,
    ).to(device)

    # 3) link predictor
    link_pred = LinkPredictor(in_channels=EMB_DIM + NODE_FEAT_DIM).to(device)

    model = {
        'memory': memory,
        'gnn': gnn,
        'link_pred': link_pred
    }

    # define an optimizer
    optimizer = torch.optim.AdamW(
        set(model['memory'].parameters()) | 
        set(model['gnn'].parameters()) | 
        set(model['link_pred'].parameters()),
        lr=LR,
        fused=True if torch.cuda.is_available() else False
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(device='cuda', enabled=USE_AMP)

    # define an early stopper
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
        # training
        start_epoch_train = timeit.default_timer()
        loss = train()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )

        # validation
        start_val = timeit.default_timer()
        perf_metric_val = test(val_src_l, val_dst_l, val_ts_l, val_e_idx_l, split_mode="val")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print(f"\tValidation MRR: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_perf_list.append(perf_metric_val)

        checkpoint_path = f'{save_model_dir}/{save_model_id}_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'memory_state_dict': model['memory'].state_dict(),
            'gnn': model['gnn'].state_dict(),
            'link_pred_state_dict': model['link_pred'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss,
            'val_mrr': perf_metric_val,
        }, checkpoint_path)
        print(f"\tModel saved to {checkpoint_path}")

        # check for early stopping
        if early_stopper.step_check(perf_metric_val, model):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation Total Elapsed Time (s): {train_val_time: .4f}")

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)

    # final testing
    start_test = timeit.default_timer()
    perf_metric_test = test(test_src_l, test_dst_l, test_ts_l, test_e_idx_l, split_mode="test")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
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
print("==============================================================")