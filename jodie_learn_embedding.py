import math
import timeit
import os
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch.amp import autocast, GradScaler
import wandb
from utils.utils import get_args, set_random_seed, save_results
from modules.neighbor_loader import LastNeighborLoader
from modules.early_stopping import EarlyStopMonitor
from modules.jodie_ctan import JODIE
from tqdm import tqdm
import types
import pickle


# ==========
# ========== GPU Optimization Settings
# ==========

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')


# ==========
# ========== Helper Functions
# ==========

def fix_tgn_memory_dtype(memory_module):
    """Fix PyG TGN memory to use Float for timestamps instead of Long."""
    if memory_module is None:
        return
    
    if hasattr(memory_module, 'last_update'):
        memory_module.last_update = memory_module.last_update.float()
        if hasattr(memory_module, '_buffers') and 'last_update' in memory_module._buffers:
            memory_module._buffers['last_update'] = memory_module.last_update
    
    if hasattr(memory_module, 'reset_state') and not hasattr(memory_module.reset_state, '_float_patched'):
        original_reset = memory_module.reset_state
        
        def float_reset_state(self):
            result = original_reset()
            if hasattr(self, 'last_update'):
                self.last_update = self.last_update.float()
                if hasattr(self, '_buffers') and 'last_update' in self._buffers:
                    self._buffers['last_update'] = self.last_update
            return result
        
        float_reset_state._float_patched = True
        memory_module.reset_state = types.MethodType(float_reset_state, memory_module)
    
    if hasattr(memory_module, '_update_msg_store') and not hasattr(memory_module._update_msg_store, '_float_patched'):
        original_update_msg_store = memory_module._update_msg_store
        
        def float_update_msg_store(self, src, dst, t, raw_msg, msg_store):
            if t.dtype in [torch.int32, torch.int64]:
                t = t.float()
            return original_update_msg_store(src, dst, t, raw_msg, msg_store)
        
        float_update_msg_store._float_patched = True
        memory_module._update_msg_store = types.MethodType(float_update_msg_store, memory_module)
    
    if hasattr(memory_module, '_update_memory') and not hasattr(memory_module._update_memory, '_float_patched'):
        original_update_memory = memory_module._update_memory

        def float_update_memory(self, n_id):
            original_update_memory(n_id)
            self.last_update[n_id] = self.last_update[n_id].float()

        float_update_memory._float_patched = True
        memory_module._update_memory = types.MethodType(float_update_memory, memory_module)


def checkpoint_memory_state(model, neighbor_loader):
    """Save memory and neighbor loader state for fast restoration."""
    state = {
        'memory': model['jodie'].memory.memory.detach().clone() if hasattr(model['jodie'], 'memory') else None,
        'last_update': model['jodie'].memory.last_update.detach().clone() if hasattr(model['jodie'], 'memory') else None,
        'neighbor_last_neighbor': {k: v.clone() for k, v in neighbor_loader.__dict__.items() if isinstance(v, torch.Tensor)}
    }
    return state


def restore_memory_state(model, neighbor_loader, state):
    """Restore memory and neighbor loader state from checkpoint."""
    if state['memory'] is not None and hasattr(model['jodie'], 'memory'):
        model['jodie'].memory.memory.copy_(state['memory'])
        model['jodie'].memory.last_update.copy_(state['last_update'])
    
    for k, v in state['neighbor_last_neighbor'].items():
        if hasattr(neighbor_loader, k):
            getattr(neighbor_loader, k).copy_(v)


def train():
    """Training procedure with optimized batch processing."""
    model['jodie'].train()
    model['jodie'].reset_memory()
    
    if hasattr(model['jodie'], 'memory'):
        fix_tgn_memory_dtype(model['jodie'].memory)

    neighbor_loader.reset_state()
    
    total_loss = 0
    num_samples = 0
    
    # Pre-shuffle indices
    idx_list = np.random.permutation(len(train_src_l))
    
    # Pre-allocate negative sampling buffer
    neg_buffer = torch.empty(BATCH_SIZE, dtype=torch.long, device=device)
    
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(len(train_src_l), s_idx + BATCH_SIZE)
        batch_idx = idx_list[s_idx:e_idx]
        batch_size = len(batch_idx)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Get batch data (optimized transfers)
        src = torch.from_numpy(train_src_l[batch_idx]).long().to(device, non_blocking=True)
        pos_dst = torch.from_numpy(train_dst_l[batch_idx]).long().to(device, non_blocking=True)
        t = torch.from_numpy(train_ts_l[batch_idx]).float().to(device, non_blocking=True)
        e_idx_batch = train_e_idx_l[batch_idx]
        msg = edge_raw_features[e_idx_batch]

        # Efficient negative sampling
        neg_idx = np.random.randint(0, len(valid_dst_nodes), size=batch_size)
        neg_dst = neg_buffer[:batch_size]
        neg_dst.copy_(torch.from_numpy(valid_dst_nodes[neg_idx]), non_blocking=True)

        # Prepare batch
        original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        batch_src = torch.cat([src, src])
        batch_dst = torch.cat([pos_dst, neg_dst])
        
        # Get neighborhood
        n_id = original_n_id
        edge_index = torch.empty(size=(2, 0), device=device).long()
        e_id = neighbor_loader.e_id[n_id]
        
        for _ in range(model["jodie"].num_gnn_layers):
            n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Forward pass with AMP
        with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            t_expanded = t.repeat(2)

            # Use simple namespace instead of type() for efficiency
            from types import SimpleNamespace
            batch_obj = SimpleNamespace(
                src=batch_src,
                dst=batch_dst,
                x=node_raw_features,
                n_neg=len(neg_dst),
                t=t_expanded
            )
            
            pos_out, neg_out, _, _ = model["jodie"](
                batch=batch_obj, 
                n_id=n_id, 
                msg=all_edge_features[e_id],
                t=all_edge_times[e_id], 
                edge_index=edge_index, 
                id_mapper=assoc
            )

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory
        model['jodie'].update(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model['jodie'].parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        model['jodie'].detach_memory()
        
        # Accumulate loss
        total_loss += loss.detach().float().item() * batch_size
        num_samples += batch_size

    avg_loss = total_loss / num_samples
    return avg_loss


@torch.no_grad()
def test_vectorized(src_l, dst_l, ts_l, e_idx_l, split_mode='val', use_fast_eval=True):
    model['jodie'].eval()
    
    # Restore from checkpoint instead of replaying
    if split_mode == 'val' and 'val_checkpoint' in memory_checkpoints:
        restore_memory_state(model, neighbor_loader, memory_checkpoints['val_checkpoint'])
    elif split_mode == 'test' and 'test_checkpoint' in memory_checkpoints:
        restore_memory_state(model, neighbor_loader, memory_checkpoints['test_checkpoint'])
    else:
        # Only replay if no checkpoint available
        _replay_edges_for_split(split_mode)
    
    # Vectorized evaluation
    perf_list = []
    
    if use_fast_eval:
        # FAST MODE: Evaluate in larger batches
        eval_batch_size = TEST_BATCH_SIZE
    else:
        # EXACT MODE: One-by-one (matches original)
        eval_batch_size = 1
    
    num_test_batches = math.ceil(len(src_l) / eval_batch_size)
    
    # Pre-allocate buffers
    neg_buffer = torch.empty(eval_batch_size, dtype=torch.long, device=device)

    for k in tqdm(range(num_test_batches), desc=f"{split_mode} eval", disable=not SHOW_PROGRESS):
        s_idx = k * eval_batch_size
        e_idx = min(len(src_l), s_idx + eval_batch_size)
        batch_size = e_idx - s_idx
        
        # Get positive edges
        pos_src = torch.from_numpy(src_l[s_idx:e_idx]).long().to(device, non_blocking=True)
        pos_dst = torch.from_numpy(dst_l[s_idx:e_idx]).long().to(device, non_blocking=True)
        pos_t = torch.from_numpy(ts_l[s_idx:e_idx]).float().to(device, non_blocking=True)
        pos_e_idx = e_idx_l[s_idx:e_idx]
        pos_msg = edge_raw_features[pos_e_idx]

        # Sample negatives efficiently
        neg_idx = np.random.randint(0, len(valid_dst_nodes), size=batch_size)
        neg_dst = neg_buffer[:batch_size]
        neg_dst.copy_(torch.from_numpy(valid_dst_nodes[neg_idx]), non_blocking=True)
        
        # Get unique nodes for neighborhood expansion
        original_n_id = torch.cat([pos_src, pos_dst, neg_dst]).unique()
        
        # Expand neighborhood once for all edges in batch
        n_id = original_n_id
        edge_index = torch.empty(size=(2, 0), device=device).long()
        e_id = neighbor_loader.e_id[n_id]
        
        for _ in range(model["jodie"].num_gnn_layers):
            n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Evaluate each edge with shared neighborhood
        with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            for i in range(batch_size):
                from types import SimpleNamespace
                
                _src = pos_src[i:i+1].expand(2)
                _batch = SimpleNamespace(
                    src=_src,
                    dst=torch.cat([pos_dst[i:i+1], neg_dst[i:i+1]]),
                    x=node_raw_features,
                    n_neg=1,
                    t=pos_t[i:i+1].expand(2)
                )

                pos_out, neg_out, _, _ = model["jodie"](
                    batch=_batch, 
                    n_id=n_id, 
                    msg=all_edge_features[e_id],
                    t=all_edge_times[e_id], 
                    edge_index=edge_index, 
                    id_mapper=assoc
                )
                
                # MRR computation
                if pos_out.item() > neg_out.item():
                    perf_list.append(1.0)
                else:
                    perf_list.append(0.5)
        
        # Update memory with positive edges
        model['jodie'].update(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    return float(np.mean(perf_list)) if perf_list else 0.0


def _replay_edges_for_split(split_mode):
    """Replay edges to restore memory state for a split."""
    model['jodie'].reset_memory()
    neighbor_loader.reset_state()
    if hasattr(model['jodie'], 'memory'):
        fix_tgn_memory_dtype(model['jodie'].memory)

    if split_mode == 'val':
        replay_src = train_src_l
        replay_dst = train_dst_l
        replay_ts = train_ts_l
        replay_e_idx = train_e_idx_l
    elif split_mode == 'test':
        replay_src = np.concatenate([train_src_l, val_src_l])
        replay_dst = np.concatenate([train_dst_l, val_dst_l])
        replay_ts = np.concatenate([train_ts_l, val_ts_l])
        replay_e_idx = np.concatenate([train_e_idx_l, val_e_idx_l])
    
    # Replay in batches
    for k in range(0, len(replay_src), BATCH_SIZE):
        e_idx = min(len(replay_src), k + BATCH_SIZE)
        src_batch = torch.from_numpy(replay_src[k:e_idx]).long().to(device)
        dst_batch = torch.from_numpy(replay_dst[k:e_idx]).long().to(device)
        t_batch = torch.from_numpy(replay_ts[k:e_idx]).float().to(device)
        e_idx_batch = replay_e_idx[k:e_idx]
        msg_batch = edge_raw_features[e_idx_batch]
        
        model['jodie'].update(src_batch, dst_batch, t_batch, msg_batch)
        neighbor_loader.insert(src_batch, dst_batch)


# ==========================================================================
# Main Training Loop
# ==========================================================================

start_overall = timeit.default_timer()

# Parse arguments
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

MODEL_NAME = 'JODIE'

# Optimization flags
USE_FAST_EVAL = os.environ.get('FAST_EVAL', '1') == '1'  # Enable fast vectorized eval
USE_MEMORY_CHECKPOINTS = True  # Cache memory states

# WandB configuration
USE_WANDB = os.environ.get('WANDB_MODE') != 'disabled'
WANDB_PROJECT = getattr(args, 'wandb_project', 'jodie-link-prediction')
WANDB_ENTITY = getattr(args, 'wandb_entity', None)
SHOW_PROGRESS = not USE_WANDB
WANDB_LOG_FREQ = 10  # Log every N epochs instead of every epoch

# GPU optimization settings
USE_AMP = True
if torch.cuda.is_available():
    USE_BFLOAT16 = torch.cuda.is_bf16_supported()
else:
    USE_BFLOAT16 = False
AMP_DTYPE = torch.bfloat16 if USE_BFLOAT16 else torch.float16
print(f"INFO: Using mixed precision with dtype: {AMP_DTYPE}")
print(f"INFO: Fast evaluation mode: {USE_FAST_EVAL}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(1)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.2f} GB")

# Load data
print("Loading data...")
g_df = pd.read_csv('./processed/bitcoin_transactions.csv')
e_feat = np.load('./processed/bitcoin_transaction_features.npy')
n_feat = np.load('./processed/bitcoin_address_features.npy')

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
ts_l = g_df.ts.values

max_idx = max(src_l.max(), dst_l.max())
num_nodes = max_idx + 1

# Pre-compute valid destination nodes
valid_dst_nodes = np.unique(dst_l)
print(f"Number of unique destination nodes: {len(valid_dst_nodes)}")

# Split data
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

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

# =========================================================
# INSERT THIS BLOCK TO PRINT BLOCK HEIGHTS
# =========================================================
# Access the 3rd column (index 2) directly from the dataframe
block_vals = g_df.iloc[:, 2].values

print("\n" + "="*50)
print("=== BLOCK HEIGHT STATISTICS (3rd Column) ===")
if valid_train_flag.sum() > 0:
    b_train = block_vals[valid_train_flag]
    print(f"Train Blocks: {int(b_train.min())} --> {int(b_train.max())}")

if valid_val_flag.sum() > 0:
    b_val = block_vals[valid_val_flag]
    print(f"Valid Blocks: {int(b_val.min())} --> {int(b_val.max())}")

if valid_test_flag.sum() > 0:
    b_test = block_vals[valid_test_flag]
    print(f"Test Blocks:  {int(b_test.min())} --> {int(b_test.max())}")
print("="*50 + "\n")
# =========================================================

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

print("="*60)
print(f"=== {MODEL_NAME}: LinkPropPred ===")
print("="*60)
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {len(src_l)}")
print(f"Train edges: {len(train_src_l)}")
print(f"Val edges: {len(val_src_l)}")
print(f"Test edges: {len(test_src_l)}")

# Move features to GPU with pinned memory for faster transfers
edge_raw_features = torch.from_numpy(e_feat).float().to(device)
# node_raw_features = torch.zeros((num_nodes, EMB_DIM), device=device)
node_raw_features = torch.from_numpy(n_feat).float().to(device)

NODE_FEAT_DIM = n_feat.shape[1]

all_edge_times = torch.from_numpy(ts_l).float().to(device)
all_edge_features = edge_raw_features

# Helper vector
assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

# Results path
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_results.json'

for run_idx in range(NUM_RUNS):
    print('-'*60)
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # Set seeds
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)
    np.random.seed(run_idx + SEED)

    # Initialize WandB
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"{MODEL_NAME}_run{run_idx}",
            config={
                'model': MODEL_NAME,
                'lr': LR,
                'batch_size': BATCH_SIZE,
                'num_epoch': NUM_EPOCH,
                'mem_dim': MEM_DIM,
                'time_dim': TIME_DIM,
                'emb_dim': EMB_DIM,
                'use_fast_eval': USE_FAST_EVAL,
                'seed': SEED,
                'run': run_idx,
            },
            reinit=True
        )

    # Initialize components
    neighbor_loader = LastNeighborLoader(num_nodes, size=NUM_NEIGHBORS, device=device)
    
    jodie = JODIE(
        num_nodes=num_nodes,
        memory_dim=MEM_DIM,
        time_dim=TIME_DIM,
        node_dim=NODE_FEAT_DIM,
        edge_dim=e_feat.shape[1]
    ).to(device)

    if hasattr(jodie, 'memory'):
        fix_tgn_memory_dtype(jodie.memory)

    model = {'jodie': jodie}

    # Optimizer with fused operations
    optimizer = torch.optim.AdamW(
        model['jodie'].parameters(),
        lr=LR,
        fused=True if torch.cuda.is_available() else False
    )
    criterion = torch.nn.BCEWithLogitsLoss()
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

    # Memory checkpoints for fast evaluation
    memory_checkpoints = {}

    # Training loop
    val_perf_list = []
    best_val_mrr = 0.0
    start_train_val = timeit.default_timer()
    
    for epoch in range(1, NUM_EPOCH + 1):
        # Training
        start_epoch_train = timeit.default_timer()
        loss = train()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        epoch_train_time = timeit.default_timer() - start_epoch_train
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Time: {epoch_train_time:.4f}s")

        # Create checkpoint after training for validation
        if USE_MEMORY_CHECKPOINTS:
            memory_checkpoints['val_checkpoint'] = checkpoint_memory_state(model, neighbor_loader)

        # Validation
        start_val = timeit.default_timer()
        perf_metric_val = test_vectorized(
            val_src_l, val_dst_l, val_ts_l, val_e_idx_l, 
            split_mode="val", 
            use_fast_eval=USE_FAST_EVAL
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        val_time = timeit.default_timer() - start_val
        print(f"\tVal MRR: {perf_metric_val:.4f}, Time: {val_time:.4f}s")
        val_perf_list.append(perf_metric_val)
        
        if perf_metric_val > best_val_mrr:
            best_val_mrr = perf_metric_val

        # Efficient WandB logging
        if USE_WANDB and (epoch % WANDB_LOG_FREQ == 0 or epoch == NUM_EPOCH):
            wandb.log({
                'epoch': epoch,
                'train/loss': loss,
                'train/time': epoch_train_time,
                'val/mrr': perf_metric_val,
                'val/time': val_time,
                'val/best_mrr': best_val_mrr,
            })

        # Early stopping
        if early_stopper.step_check(perf_metric_val, model):
            print(f"Early stopping at epoch {epoch}")
            if USE_WANDB:
                wandb.log({'early_stopped_epoch': epoch})
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Val Time: {train_val_time:.4f}s")

    # Load best model
    early_stopper.load_checkpoint(model)

    # Create test checkpoint
    if USE_MEMORY_CHECKPOINTS:
        _replay_edges_for_split('test')
        memory_checkpoints['test_checkpoint'] = checkpoint_memory_state(model, neighbor_loader)

    # Testing
    start_test = timeit.default_timer()
    perf_metric_test = test_vectorized(
        test_src_l, test_dst_l, test_ts_l, test_e_idx_l, 
        split_mode="test",
        use_fast_eval=USE_FAST_EVAL
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    test_time = timeit.default_timer() - start_test
    print(f"Test MRR: {perf_metric_test:.4f}, Time: {test_time:.4f}s")

    # Final WandB logging
    if USE_WANDB:
        wandb.log({
            'test/mrr': perf_metric_test,
            'test/time': test_time,
            'best_val_mrr': max(val_perf_list),
            'total_train_val_time': train_val_time,
        })
        
        wandb.run.summary.update({
            'best_test_mrr': perf_metric_test,
            'best_val_mrr': max(val_perf_list),
            'total_time': timeit.default_timer() - start_run,
            'epochs_trained': epoch,
        })
        
        wandb.finish()

    # Save results
    save_results({
        'model': MODEL_NAME,
        'run': run_idx,
        'seed': SEED,
        'val_mrr': val_perf_list,
        'test_mrr': perf_metric_test,
        'test_time': test_time,
        'tot_train_val_time': train_val_time
    }, results_filename)

    print(f"Run {run_idx} elapsed: {timeit.default_timer() - start_run:.4f}s")
    print('-'*60)

print(f"Overall Elapsed Time: {timeit.default_timer() - start_overall:.4f}s")
print("="*60)