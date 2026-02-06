from calendar import c
import math
import timeit
import os
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import random

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.nn import Linear, ReLU, Dropout, Sequential
from torch.amp import autocast, GradScaler

# WandB for experiment tracking
import wandb

# internal imports
from utils.utils import get_args, set_random_seed, save_results
from modules.neighbor_loader import LastNeighborLoader
from modules.early_stopping import EarlyStopMonitor
from modules.jodie_ctan import JODIE
from tqdm import tqdm
import types

# GPU Optimization Settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')


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


class MLPClassifier(torch.nn.Module):
    """3-layer MLP for binary node classification"""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.mlp = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)

def extract_node_embeddings_from_memory(jodie_model, node_ids):
    """Extract current temporal embeddings from JODIE memory for given nodes"""
    if hasattr(jodie_model, 'memory') and jodie_model.memory is not None:
        memory_state = jodie_model.memory.memory[node_ids]
        static_feat = node_raw_features[node_ids]
        combined_emb = torch.cat([memory_state, static_feat], dim=1)
        return combined_emb
    else:
        raise ValueError("JODIE model has no memory module")


def save_jodie_memory_state(jodie_model, neighbor_loader, save_path):
    """Save JODIE memory and neighbor loader state to disk"""
    state = {
        'memory': jodie_model.memory.memory.detach().cpu() if hasattr(jodie_model, 'memory') else None,
        'last_update': jodie_model.memory.last_update.detach().cpu() if hasattr(jodie_model, 'memory') else None,
        'neighbor_last_neighbor_id': neighbor_loader.last_neighbor_id.cpu() if hasattr(neighbor_loader, 'last_neighbor_id') else None,
        'neighbor_e_id': neighbor_loader.e_id.cpu() if hasattr(neighbor_loader, 'e_id') else None,
    }
    
    # Save with metadata for validation
    state['metadata'] = {
        'num_nodes': jodie_model.memory.memory.shape[0] if hasattr(jodie_model, 'memory') else 0,
        'mem_dim': jodie_model.memory.memory.shape[1] if hasattr(jodie_model, 'memory') else 0,
    }
    
    torch.save(state, save_path)
    print(f"JODIE memory state saved to: {save_path}")


def load_jodie_memory_state(jodie_model, neighbor_loader, load_path):
    """Load JODIE memory and neighbor loader state from disk"""
    if not os.path.exists(load_path):
        return False
    
    try:
        print(f"Loading JODIE memory state from: {load_path}")
        state = torch.load(load_path, map_location=device)
        
        # Validate metadata
        if 'metadata' in state:
            if state['metadata']['num_nodes'] != jodie_model.memory.memory.shape[0]:
                print("WARNING: Cached state has different num_nodes. Ignoring cache.")
                return False
            if state['metadata']['mem_dim'] != jodie_model.memory.memory.shape[1]:
                print("WARNING: Cached state has different mem_dim. Ignoring cache.")
                return False
        
        # Restore memory state
        if state['memory'] is not None and hasattr(jodie_model, 'memory'):
            jodie_model.memory.memory.copy_(state['memory'].to(device))
            jodie_model.memory.last_update.copy_(state['last_update'].to(device))
        
        # Restore neighbor loader state
        if state['neighbor_last_neighbor_id'] is not None and hasattr(neighbor_loader, 'last_neighbor_id'):
            neighbor_loader.last_neighbor_id.copy_(state['neighbor_last_neighbor_id'].to(device))
        if state['neighbor_e_id'] is not None and hasattr(neighbor_loader, 'e_id'):
            neighbor_loader.e_id.copy_(state['neighbor_e_id'].to(device))
        
        print("JODIE memory state loaded successfully!")
        return True
    
    except Exception as e:
        print(f"ERROR loading cached state: {e}")
        return False


def train_classifier(epoch):
    """Train MLP classifier on frozen JODIE embeddings"""
    classifier.train()
    if FINETUNE_JODIE:
        jodie_model.train()
    else:
        jodie_model.eval()
    
    total_loss = 0
    num_samples = 0
    
    idx_list = np.arange(len(train_nodes))
    np.random.shuffle(idx_list)
    
    num_batches = math.ceil(len(train_nodes) / BATCH_SIZE)
    
    for k in range(num_batches):
        s_idx = k * BATCH_SIZE
        e_idx = min(len(train_nodes), s_idx + BATCH_SIZE)
        batch_idx = idx_list[s_idx:e_idx]
        batch_size = len(batch_idx)
        
        optimizer.zero_grad(set_to_none=True)
        
        batch_nodes = train_nodes[batch_idx]
        batch_labels = train_labels[batch_idx]
        
        batch_nodes_tensor = torch.from_numpy(batch_nodes).long().to(device)
        batch_labels_tensor = torch.from_numpy(batch_labels).float().to(device)
        
        # with torch.no_grad():
        #     node_embeddings = extract_node_embeddings_from_memory(jodie_model, batch_nodes_tensor)
        if FINETUNE_JODIE:
            # Allow gradients to flow through JODIE
            node_embeddings = extract_node_embeddings_from_memory(jodie_model, batch_nodes_tensor)
        else:
            # Keep original frozen behavior
            with torch.no_grad():
                node_embeddings = extract_node_embeddings_from_memory(jodie_model, batch_nodes_tensor)
        
        with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = classifier(node_embeddings)
            loss = criterion(logits, batch_labels_tensor)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if FINETUNE_JODIE:
            torch.nn.utils.clip_grad_norm_(jodie_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.detach().float().item() * batch_size
        num_samples += batch_size
    
    return total_loss / num_samples


@torch.no_grad()
def evaluate_classifier(nodes, labels, split_name='val'):
    """Evaluate classifier with threshold search"""
    classifier.eval()
    jodie_model.eval()
    
    all_logits = []
    all_labels = []
    
    num_batches = math.ceil(len(nodes) / TEST_BATCH_SIZE)
    
    for k in range(num_batches):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(len(nodes), s_idx + TEST_BATCH_SIZE)
        
        batch_nodes = nodes[s_idx:e_idx]
        batch_labels = labels[s_idx:e_idx]
        
        batch_nodes_tensor = torch.from_numpy(batch_nodes).long().to(device)
        
        node_embeddings = extract_node_embeddings_from_memory(jodie_model, batch_nodes_tensor)
        
        with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = classifier(node_embeddings)
        
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch_labels)
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    probs = 1 / (1 + np.exp(-all_logits))
    
    if split_name == 'val':
        best_threshold = 0.5
        best_f1 = 0.0
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        global BEST_THRESHOLD
        BEST_THRESHOLD = best_threshold
        threshold = best_threshold
    else:
        threshold = BEST_THRESHOLD
    
    preds = (probs >= threshold).astype(int)
    
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, probs)
    except:
        auc = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': threshold
    }


def load_pretrained_jodie(model_path, jodie_model):
    """Load pretrained JODIE model from link prediction task"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model not found at: {model_path}")
    
    print(f"Loading pretrained JODIE model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'jodie' in checkpoint:
            jodie_model.load_state_dict(checkpoint['jodie'])
        elif 'model_state_dict' in checkpoint:
            jodie_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            jodie_model.load_state_dict(checkpoint['state_dict'])
        else:
            jodie_model.load_state_dict(checkpoint)
    else:
        jodie_model.load_state_dict(checkpoint)
    
    print("Pretrained JODIE model loaded successfully!")
    
    if hasattr(jodie_model, 'memory'):
        fix_tgn_memory_dtype(jodie_model.memory)
    
    return jodie_model


def replay_temporal_graph_and_update_memory():
    """Replay the entire temporal graph to update JODIE memory"""
    print("Replaying temporal graph to build JODIE memory...")
    jodie_model.eval()
    jodie_model.reset_memory()
    neighbor_loader.reset_state()
    
    if hasattr(jodie_model, 'memory'):
        fix_tgn_memory_dtype(jodie_model.memory)
    
    all_src = src_l
    all_dst = dst_l
    all_ts = ts_l
    all_e_idx = e_idx_l
    
    num_batches = math.ceil(len(all_src) / BATCH_SIZE)
    
    with torch.no_grad():
        for k in tqdm(range(num_batches), desc="Replaying graph"):
            s_idx = k * BATCH_SIZE
            e_idx = min(len(all_src), s_idx + BATCH_SIZE)
            
            src_batch = torch.from_numpy(all_src[s_idx:e_idx]).long().to(device)
            dst_batch = torch.from_numpy(all_dst[s_idx:e_idx]).long().to(device)
            t_batch = torch.from_numpy(all_ts[s_idx:e_idx]).float().to(device)
            e_idx_batch = all_e_idx[s_idx:e_idx]
            msg_batch = edge_raw_features[e_idx_batch].to(device)
            
            jodie_model.update(src_batch, dst_batch, t_batch, msg_batch)
            neighbor_loader.insert(src_batch, dst_batch)
    
    print("Memory replay complete!")

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

MODEL_NAME = 'JODIE_NodeClassification'

# OPTIMIZATION: Check if we need to replay or can use cached state
memory_cache_path = f'{osp.dirname(osp.abspath(__file__))}/saved_models/jodie_memory_cache_mem{MEM_DIM}_time{TIME_DIM}_emb{EMB_DIM}.pth'
use_cached_memory = False

# WandB configuration
USE_WANDB = False#os.environ.get('WANDB_MODE') != 'disabled'
WANDB_PROJECT = getattr(args, 'wandb_project', 'jodie-node-classification')
WANDB_ENTITY = getattr(args, 'wandb_entity', None)

# Get MLP hyperparameters if provided
MLP_HIDDEN = getattr(args, 'mlp_hidden', 128)
DROPOUT = getattr(args, 'dropout', 0.3)

# Pretrained model path
PRETRAINED_MODEL_PATH = args.pretrained_model if hasattr(args, 'pretrained_model') else None
if PRETRAINED_MODEL_PATH is None:
    PRETRAINED_MODEL_PATH = f'{osp.dirname(osp.abspath(__file__))}/saved_models/JODIE_42_0.pth'

print(f"INFO: Pretrained model path: {PRETRAINED_MODEL_PATH}")

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
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load data
print("Loading data...")
g_df = pd.read_csv('./processed/bitcoin_transactions.csv')
e_feat = np.load('./processed/bitcoin_transaction_features.npy')
n_feat = np.load('./processed/bitcoin_address_features.npy')


src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
ts_l = g_df.ts.values


if 'label' in g_df.columns:
    edge_labels = g_df.label.values
else:
    raise ValueError("Dataset must have 'label' column for node classification")


node_labels = {}
for src, label in zip(src_l, edge_labels):
    if src not in node_labels:
        node_labels[src] = label

max_idx = max(src_l.max(), dst_l.max())
num_nodes = max_idx + 1

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {len(src_l)}")

# Filter nodes
valid_nodes = []
valid_labels = []

for node_id, label in node_labels.items():
    if label in [0, 1]:
        valid_nodes.append(node_id)
        valid_labels.append(label)

valid_nodes = np.array(valid_nodes)
valid_labels = np.array(valid_labels)

print(f"Nodes with valid labels {0, 1}: {len(valid_nodes)}")
print(f"Label distribution: 0={np.sum(valid_labels == 0)}, 1={np.sum(valid_labels == 1)}")

pos_count = np.sum(valid_labels == 1)
neg_count = np.sum(valid_labels == 0)
pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
print(f"Class imbalance: pos_weight={pos_weight:.2f}")

# Split nodes
# num_valid = len(valid_nodes)
# train_end = int(args.train_ratio * num_valid)
# val_end = train_end + int(args.val_ratio * num_valid)

# shuffle_idx = np.random.permutation(num_valid)
# valid_nodes = valid_nodes[shuffle_idx]
# valid_labels = valid_labels[shuffle_idx]

# train_nodes = valid_nodes[:train_end]
# train_labels = valid_labels[:train_end]
# val_nodes = valid_nodes[train_end:val_end]
# val_labels = valid_labels[train_end:val_end]
# test_nodes = valid_nodes[val_end:]
# test_labels = valid_labels[val_end:]

print("="*80)
print("Using TEMPORAL split (state-of-the-art for temporal graphs)")
print("="*80)

# Step 1: Create node-to-timestamp mapping (last interaction time for each node)
node_last_timestamp = {}
for src, dst, ts in zip(src_l, dst_l, ts_l):
    for n in [src, dst]:
        if n not in node_last_timestamp or ts < node_last_timestamp[n]:
            node_last_timestamp[n] = ts

# Step 2: Filter nodes with valid labels
valid_node_info = []  # (node_id, label, last_timestamp)
for node_id, label in node_labels.items():
    if label in [0, 1] and node_id in node_last_timestamp:
        valid_node_info.append((node_id, label, node_last_timestamp[node_id]))

print(f"Nodes with valid labels: {len(valid_node_info)}")

if len(valid_node_info) == 0:
    raise ValueError("No valid nodes found with labels in {0, 1}")

# Step 3: Sort by timestamp (temporal ordering)
valid_node_info.sort(key=lambda x: x[2])  # Sort by last_timestamp

# Step 4: Temporal split based on sorted timestamps
num_valid = len(valid_node_info)
train_end = int(args.train_ratio * num_valid)
val_end = train_end + int(args.val_ratio * num_valid)

# Extract nodes and labels in temporal order
train_data = valid_node_info[:train_end]
val_data = valid_node_info[train_end:val_end]
test_data = valid_node_info[val_end:]

train_nodes = np.array([x[0] for x in train_data])
train_labels = np.array([x[1] for x in train_data])
val_nodes = np.array([x[0] for x in val_data])
val_labels = np.array([x[1] for x in val_data])
test_nodes = np.array([x[0] for x in test_data])
test_labels = np.array([x[1] for x in test_data])

# Step 5: Log temporal split information
if len(train_data) > 0:
    train_time_range = (train_data[0][2], train_data[-1][2])
    print(f"Train temporal range: {train_time_range[0]:.2f} - {train_time_range[1]:.2f}")

if len(val_data) > 0:
    val_time_range = (val_data[0][2], val_data[-1][2])
    print(f"Val temporal range: {val_time_range[0]:.2f} - {val_time_range[1]:.2f}")

if len(test_data) > 0:
    test_time_range = (test_data[0][2], test_data[-1][2])
    print(f"Test temporal range: {test_time_range[0]:.2f} - {test_time_range[1]:.2f}")

print("="*80)
print(f"Train nodes: {len(train_nodes)} (earliest {args.train_ratio*100:.1f}% temporal data)")
print(f"Val nodes: {len(val_nodes)} (next {args.val_ratio*100:.1f}% temporal data)")
print(f"Test nodes: {len(test_nodes)} (latest {(1-args.train_ratio-args.val_ratio)*100:.1f}% temporal data)")
print(f"✓ No temporal leakage - respecting time order")
print("="*80)

# Verify no overlap and temporal ordering
assert len(set(train_nodes) & set(val_nodes)) == 0 or len(val_nodes) == 0, "Train-Val overlap detected!"
assert len(set(train_nodes) & set(test_nodes)) == 0 or len(test_nodes) == 0, "Train-Test overlap detected!"
assert len(set(val_nodes) & set(test_nodes)) == 0 or len(val_nodes) == 0 or len(test_nodes) == 0, "Val-Test overlap detected!"

if len(train_data) > 0 and len(val_data) > 0:
    assert train_data[-1][2] <= val_data[0][2], "Temporal order violated: Train after Val!"

if len(val_data) > 0 and len(test_data) > 0:
    assert val_data[-1][2] <= test_data[0][2], "Temporal order violated: Val after Test!"

# print("✓ Temporal split validation passed")
print("="*60)
print(f"=== {MODEL_NAME} ===")
print("="*60)
print(f"Train nodes: {len(train_nodes)}")
print(f"Val nodes: {len(val_nodes)}")
print(f"Test nodes: {len(test_nodes)}")



edge_raw_features = torch.from_numpy(e_feat).float().to(device)
# node_raw_features = torch.zeros((num_nodes, EMB_DIM), device=device)
node_raw_features = torch.from_numpy(n_feat).float().to(device)
NODE_FEAT_DIM = n_feat.shape[1]

assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

BEST_THRESHOLD = 0.5

results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_results.json'

for run_idx in range(NUM_RUNS):
    print('-'*60)
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

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
                'lr': LR,
                'batch_size': BATCH_SIZE,
                'mem_dim': MEM_DIM,
                'time_dim': TIME_DIM,
                'emb_dim': EMB_DIM,
                'mlp_hidden': MLP_HIDDEN,
                'dropout': DROPOUT,
                'num_epoch': NUM_EPOCH,
                'patience': PATIENCE,
                'seed': SEED,
                'run': run_idx,
                'pos_weight': pos_weight,
                'use_cached_memory': use_cached_memory,
                'finetune_jodie': FINETUNE_JODIE,  # ADD THIS
                'jodie_lr': JODIE_LR if FINETUNE_JODIE else None,  # ADD THIS
            },
            reinit=True
        )

    neighbor_loader = LastNeighborLoader(num_nodes, size=NUM_NEIGHBORS, device=device)
    
    jodie_model = JODIE(
        num_nodes=num_nodes,
        memory_dim=MEM_DIM,
        time_dim=TIME_DIM,
        node_dim=NODE_FEAT_DIM,
        edge_dim=e_feat.shape[1]
    ).to(device)

    if hasattr(jodie_model, 'memory'):
        fix_tgn_memory_dtype(jodie_model.memory)

    try:
        jodie_model = load_pretrained_jodie(PRETRAINED_MODEL_PATH, jodie_model)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("WARNING: Using random initialization")
    
    FINETUNE_JODIE = getattr(args, 'finetune_jodie', False)  # Add to args parsing section
    JODIE_LR = getattr(args, 'jodie_lr', 1e-5)

    if FINETUNE_JODIE:
        # Unfreeze JODIE for fine-tuning
        for param in jodie_model.parameters():
            param.requires_grad = True
        print("INFO: JODIE model is UNFROZEN for fine-tuning")
    else:
        # Keep original frozen behavior
        for param in jodie_model.parameters():
            param.requires_grad = False
        jodie_model.eval()
        print("INFO: JODIE model is FROZEN")

    # replay_temporal_graph_and_update_memory()
    # OPTIMIZATION: Try to load cached memory state, otherwise replay
    if run_idx == 0:
        # First run: try to load cache or replay
        if not load_jodie_memory_state(jodie_model, neighbor_loader, memory_cache_path):
            replay_temporal_graph_and_update_memory()
            # Save state for future runs
            save_jodie_memory_state(jodie_model, neighbor_loader, memory_cache_path)
            use_cached_memory = False
        else:
            print("INFO: Using cached JODIE memory state (skipping replay)")
            use_cached_memory = True
    else:
        # Subsequent runs: always load from cache (we know it exists now)
        if not load_jodie_memory_state(jodie_model, neighbor_loader, memory_cache_path):
            print("ERROR: Cache should exist but doesn't. Replaying...")
            replay_temporal_graph_and_update_memory()
        else:
            print("INFO: Using cached JODIE memory state (skipping replay)")
    classifier = MLPClassifier(
        input_dim=MEM_DIM + NODE_FEAT_DIM,
        hidden_dim=MLP_HIDDEN,
        dropout=DROPOUT
    ).to(device)

    if FINETUNE_JODIE:
        # Separate parameter groups with different learning rates
        optimizer = torch.optim.Adam([
            {'params': classifier.parameters(), 'lr': LR},
            {'params': jodie_model.parameters(), 'lr': JODIE_LR}
        ], fused=True if torch.cuda.is_available() else False)
        print(f"INFO: Optimizer includes JODIE (lr={JODIE_LR}) and Classifier (lr={LR})")
    else:
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=LR,
            fused=True if torch.cuda.is_available() else False
        )
        print(f"INFO: Optimizer only includes Classifier (lr={LR})")

    
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
   
    scaler = GradScaler(device='cuda', enabled=USE_AMP)

    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(
        save_model_dir=save_model_dir,
        save_model_id=save_model_id,
        tolerance=TOLERANCE,
        patience=PATIENCE
    )

    val_f1_list = []
    start_train_val = timeit.default_timer()
    
    for epoch in range(1, NUM_EPOCH + 1):
        start_epoch_train = timeit.default_timer()
        loss = train_classifier(epoch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        epoch_train_time = timeit.default_timer() - start_epoch_train
        
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training Time (s): {epoch_train_time:.4f}")

        start_val = timeit.default_timer()
        val_metrics = evaluate_classifier(val_nodes, val_labels, split_name="val")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        val_time = timeit.default_timer() - start_val
        
        print(f"\tVal Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}, "
              f"Threshold: {val_metrics['threshold']:.4f}, Time: {val_time:.4f}s")
        
        val_f1_list.append(val_metrics['f1'])

        # Log to WandB
        if USE_WANDB:
            wandb.log({
                'epoch': epoch,
                'train/loss': loss,
                'train/time': epoch_train_time,
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'val/auc': val_metrics['auc'],
                'val/threshold': val_metrics['threshold'],
                'val/time': val_time,
            })

        # if early_stopper.step_check(val_metrics['f1'], {'classifier': classifier}):
        #     print(f"Early stopping at epoch {epoch}")
        #     break
        if FINETUNE_JODIE:
            models_to_save = {'classifier': classifier, 'jodie': jodie_model}
        else:
            models_to_save = {'classifier': classifier}

        if early_stopper.step_check(val_metrics['f1'], models_to_save):
            print(f"Early stopping at epoch {epoch}")
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time:.4f}")

    # Testing
    # early_stopper.load_checkpoint({'classifier': classifier})
    if FINETUNE_JODIE:
        early_stopper.load_checkpoint({'classifier': classifier, 'jodie': jodie_model})
    else:
        early_stopper.load_checkpoint({'classifier': classifier})

    start_test = timeit.default_timer()
    test_metrics = evaluate_classifier(test_nodes, test_labels, split_name="test")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"INFO: Test Results:")
    print(f"\tTest Precision: {test_metrics['precision']:.4f}")
    print(f"\tTest Recall: {test_metrics['recall']:.4f}")
    print(f"\tTest F1: {test_metrics['f1']:.4f}")
    print(f"\tTest AUC: {test_metrics['auc']:.4f}")
    
    test_time = timeit.default_timer() - start_test
    print(f"\tTest Time: {test_time:.4f}s")

    # Log test results to WandB
    if USE_WANDB:
        wandb.log({
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
            'test/f1': test_metrics['f1'],
            'test/auc': test_metrics['auc'],
            'test/time': test_time,
            'best_val_f1': max(val_f1_list) if val_f1_list else 0,
        })
        
        # Log summary
        wandb.run.summary.update({
            'best_test_f1': test_metrics['f1'],
            'best_test_auc': test_metrics['auc'],
            'total_train_time': train_val_time,
        })

    save_results({
        'model': MODEL_NAME,
        'run': run_idx,
        'seed': SEED,
        'val_f1': val_f1_list,
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics['auc'],
        'test_time': test_time,
        'tot_train_val_time': train_val_time
    }, results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run:.4f}s <<<<<")
    print('-'*60)
    
    # Finish WandB run
    if USE_WANDB:
        wandb.finish()

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall:.4f}")
print("="*60)

