import math
import timeit
import os
import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, ReLU, Dropout, Sequential
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
# ========== Define helper functions...
# ==========

class MLPClassifier(torch.nn.Module):
    """
    3-layer MLP for binary node classification
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.mlp = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, 1)  # Binary classification (1 logit)
        )
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)

def load_pretrained_dyrep(model_path, memory, gnn, link_pred):
    """
    Load pretrained DyRep model from link prediction task
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model not found at: {model_path}")
    
    print(f"Loading pretrained DyRep model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'memory_state_dict' in checkpoint:
            memory.load_state_dict(checkpoint['memory_state_dict'])
        elif 'memory' in checkpoint:
            memory.load_state_dict(checkpoint['memory'])
        
        if 'gnn_state_dict' in checkpoint:
            gnn.load_state_dict(checkpoint['gnn_state_dict'])
        elif 'gnn' in checkpoint:
            gnn.load_state_dict(checkpoint['gnn'])
        
        if 'link_pred_state_dict' in checkpoint:
            link_pred.load_state_dict(checkpoint['link_pred_state_dict'])
        elif 'link_pred' in checkpoint:
            link_pred.load_state_dict(checkpoint['link_pred'])
    
    print("Pretrained DyRep model loaded successfully!")
    return memory, gnn, link_pred


def extract_node_embeddings(dyrep_memory, dyrep_gnn, node_ids, neighbor_loader):
    """
    Extract current temporal embeddings from DyRep for given nodes
    """
    with torch.no_grad():
        # Get neighborhood
        n_id, edge_index, e_id = neighbor_loader(node_ids)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        
        # Get memory state
        z, last_update = dyrep_memory(n_id)

        z_static = node_raw_features[n_id]
        z_combined = torch.cat([z, z_static], dim=1)
        
        # Return embeddings for requested nodes
        return z_combined[assoc[node_ids]]

def save_jodie_memory_state(memory_module, neighbor_loader, save_path):
    """
    Save Memory module state and neighbor loader state to disk
    Args:
        memory_module: The DyRepMemory module (containing .memory and .last_update tensors)
        neighbor_loader: The neighbor loader
        save_path: Location to save .pth file
    """
    # memory_module.memory is the Tensor, memory_module.last_update is the Tensor
    state = {
        'memory': memory_module.memory.detach().cpu() if hasattr(memory_module, 'memory') else None,
        'last_update': memory_module.last_update.detach().cpu() if hasattr(memory_module, 'last_update') else None,
        'neighbor_last_neighbor_id': neighbor_loader.last_neighbor_id.cpu() if hasattr(neighbor_loader, 'last_neighbor_id') else None,
        'neighbor_e_id': neighbor_loader.e_id.cpu() if hasattr(neighbor_loader, 'e_id') else None,
    }
    
    # Save with metadata for validation
    state['metadata'] = {
        'num_nodes': memory_module.memory.shape[0] if hasattr(memory_module, 'memory') else 0,
        'mem_dim': memory_module.memory.shape[1] if hasattr(memory_module, 'memory') else 0,
    }
    
    torch.save(state, save_path)
    print(f"JODIE memory state saved to: {save_path}")


def load_jodie_memory_state(memory_module, neighbor_loader, load_path):
    """
    Load Memory module state and neighbor loader state from disk
    """
    if not os.path.exists(load_path):
        return False
    
    try:
        print(f"Loading JODIE memory state from: {load_path}")
        state = torch.load(load_path, map_location=device)
        
        # Validate metadata
        if 'metadata' in state:
            if state['metadata']['num_nodes'] != memory_module.memory.shape[0]:
                print(f"WARNING: Cached state has different num_nodes ({state['metadata']['num_nodes']} vs {memory_module.memory.shape[0]}). Ignoring cache.")
                return False
            if state['metadata']['mem_dim'] != memory_module.memory.shape[1]:
                print("WARNING: Cached state has different mem_dim. Ignoring cache.")
                return False
        
        # Restore memory state
        if state['memory'] is not None and hasattr(memory_module, 'memory'):
            memory_module.memory.copy_(state['memory'].to(device))
        
        if state['last_update'] is not None and hasattr(memory_module, 'last_update'):
            memory_module.last_update.copy_(state['last_update'].to(device))
        
        # Restore neighbor loader state
        if state['neighbor_last_neighbor_id'] is not None and hasattr(neighbor_loader, 'last_neighbor_id'):
            neighbor_loader.last_neighbor_id.copy_(state['neighbor_last_neighbor_id'].to(device))
        if state['neighbor_e_id'] is not None and hasattr(neighbor_loader, 'e_id'):
            neighbor_loader.e_id.copy_(state['neighbor_e_id'].to(device))
        
        print("JODIE memory state loaded successfully!")
        return True
    
    except Exception as e:
        print(f"ERROR loading cached state: {e}")
        # Return False to force replay if loading fails
        return False
    
def train_classifier(epoch):
    """
    Train MLP classifier on frozen DyRep embeddings
    """
    classifier.train()
    dyrep_memory.eval()  # DyRep is frozen
    dyrep_gnn.eval()
    
    total_loss = 0
    num_samples = 0
    
    # Shuffle training nodes
    idx_list = np.arange(len(train_nodes))
    np.random.shuffle(idx_list)
    
    num_batches = math.ceil(len(train_nodes) / BATCH_SIZE)
    
    for k in range(num_batches):
        s_idx = k * BATCH_SIZE
        e_idx = min(len(train_nodes), s_idx + BATCH_SIZE)
        batch_idx = idx_list[s_idx:e_idx]
        batch_size = len(batch_idx)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Get batch nodes and labels
        batch_nodes = train_nodes[batch_idx]
        batch_labels = train_labels[batch_idx]
        
        batch_nodes_tensor = torch.from_numpy(batch_nodes).long().to(device)
        batch_labels_tensor = torch.from_numpy(batch_labels).float().to(device)
        
        # Extract frozen embeddings from DyRep
        with torch.no_grad():
            node_embeddings = extract_node_embeddings(
                dyrep_memory, dyrep_gnn, batch_nodes_tensor, neighbor_loader
            )
        
        # Forward pass through classifier
        with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = classifier(node_embeddings)
            loss = criterion(logits, batch_labels_tensor)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.detach().float().item() * batch_size
        num_samples += batch_size
    
    return total_loss / num_samples


@torch.no_grad()
def evaluate_classifier(nodes, labels, split_name='val'):
    """
    Evaluate classifier with threshold search
    """
    classifier.eval()
    dyrep_memory.eval()
    dyrep_gnn.eval()
    
    all_logits = []
    all_labels = []
    
    num_batches = math.ceil(len(nodes) / TEST_BATCH_SIZE)
    
    for k in range(num_batches):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(len(nodes), s_idx + TEST_BATCH_SIZE)
        
        batch_nodes = nodes[s_idx:e_idx]
        batch_labels = labels[s_idx:e_idx]
        
        batch_nodes_tensor = torch.from_numpy(batch_nodes).long().to(device)
        
        # Extract embeddings
        node_embeddings = extract_node_embeddings(
            dyrep_memory, dyrep_gnn, batch_nodes_tensor, neighbor_loader
        )
        
        with autocast(device_type='cuda', enabled=USE_AMP, dtype=AMP_DTYPE):
            logits = classifier(node_embeddings)
        
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch_labels)
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-all_logits))
    
    # Search for best threshold on validation set
    if split_name == 'val':
        best_threshold = 0.5
        best_f1 = 0.0
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        # Store best threshold globally
        global BEST_THRESHOLD
        BEST_THRESHOLD = best_threshold
        threshold = best_threshold
    else:
        # Use stored threshold for test
        threshold = BEST_THRESHOLD
    
    # Final predictions with chosen threshold
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


from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast

def replay_temporal_graph_and_update_memory(replay_batch_size=None):
    """
    Replay the entire temporal graph to update DyRep memory.
    Optimized with DataLoader, AMP, and efficient transfers.
    """
    print("Replaying temporal graph to build DyRep memory...")
    
    # 1. Setup Models
    dyrep_memory.eval()
    dyrep_gnn.eval()
    dyrep_memory.reset_state()
    neighbor_loader.reset_state()
    
    # If not provided, default to 2x the training batch size or 1000
    actual_bs = replay_batch_size if replay_batch_size else (BATCH_SIZE * 2)
    print(f"Replay Batch Size: {actual_bs}")


    # We use CPU tensors here to avoid OOM, but pin_memory will speed up transfer
    dataset = TensorDataset(
        torch.from_numpy(src_l).long(),
        torch.from_numpy(dst_l).long(),
        torch.from_numpy(ts_l).float(),
        torch.from_numpy(e_idx_l).long()
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=actual_bs, 
        shuffle=False,      # strict temporal order
        pin_memory=True,    # fast CPU->GPU transfer
        num_workers=2       # prefetch next batch
    )

    try:
        edge_raw_feats_gpu = edge_raw_features.to(device)
        gpu_feats = True
    except RuntimeError:
        gpu_feats = False
        print("Edge features too large for GPU, streaming from CPU...")

    # 5. Main Loop
    with torch.no_grad():
        for batch in tqdm(loader, desc="Replaying graph"):
            # Unpack and move to device (non-blocking if pinned)
            src_batch, dst_batch, t_batch, e_idx_batch = [x.to(device, non_blocking=True) for x in batch]
            
            # Efficient Feature Lookup
            if gpu_feats:
                msg_batch = edge_raw_feats_gpu[e_idx_batch]
            else:
                # If on CPU, we must index on CPU then move
                # Note: e_idx_batch must be moved back to cpu for indexing if it's on gpu
                msg_batch = edge_raw_features[e_idx_batch.cpu()].to(device, non_blocking=True)

            # --- Critical Computation Section ---
            # Using AMP (Automatic Mixed Precision) for speed
            with autocast(device_type='cuda', dtype=torch.float16):
                
                # 1. Get Involved Nodes
                # Note: 'unique' can be slow. If batch is large, consider strictly updating 
                # src/dst without aggregating neighbors if architecture allows.
                n_id = torch.cat([src_batch, dst_batch]).unique()
                
                # 2. Neighbor Sampling
                n_id, edge_index, e_id = neighbor_loader(n_id)
                
                # 3. Create Association Mapping
                # Faster than manual assignment if n_id is already sorted/unique
                assoc[n_id] = torch.arange(n_id.size(0), device=device)
                
                # 4. Get Memory State
                z, last_update = dyrep_memory(n_id)

                # === CHANGE HERE ===
                # Concatenate static features before GNN
                z_static = node_raw_features[n_id]
                z_combined = torch.cat([z, z_static], dim=1)
                
                # 5. GNN Forward Pass
                # Ensure all_edge_* tensors are handled efficiently
                # (Assuming all_edge_times/features are global tensors)
                z = dyrep_gnn(
                    z_combined, 
                    last_update, 
                    edge_index,
                    all_edge_times[e_id].to(device, non_blocking=True),
                    all_edge_features[e_id].to(device, non_blocking=True)
                )
                
                # 6. Update Memory State
                dyrep_memory.update_state(src_batch, dst_batch, t_batch, msg_batch, z, assoc)
            
            # 7. Update Neighbor Loader (Graph Connectivity)
            # This happens outside autocast as it involves int indexing
            neighbor_loader.insert(src_batch, dst_batch)
            
    print("Memory replay complete!")


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

MODEL_NAME = 'DyRep_NodeClassification'
USE_SRC_EMB_IN_MSG = False
USE_DST_EMB_IN_MSG = True

memory_cache_path = f'{osp.dirname(osp.abspath(__file__))}/saved_models/dyrep_memory_cache_mem{MEM_DIM}_time{TIME_DIM}_emb{EMB_DIM}.pth'
use_cached_memory = False

# ========== NEW: Path to pretrained link prediction model ==========
PRETRAINED_MODEL_PATH = args.pretrained_model if hasattr(args, 'pretrained_model') else None

# If not provided via args, construct default path
if PRETRAINED_MODEL_PATH is None:
    PRETRAINED_MODEL_PATH = f'{osp.dirname(osp.abspath(__file__))}/saved_models/DyRep_{SEED}_0.pth'

print(f"INFO: Pretrained model path: {PRETRAINED_MODEL_PATH}")

# GPU optimization settings
USE_AMP = True
if torch.cuda.is_available():
    USE_BFLOAT16 = torch.cuda.is_bf16_supported()
else:
    USE_BFLOAT16 = False
AMP_DTYPE = torch.bfloat16 if USE_BFLOAT16 else torch.float16
print(f"INFO: Using mixed precision with dtype: {AMP_DTYPE}")

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Extract data - including labels
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
ts_l = g_df.ts.values

# Node labels are derived from source nodes
# Assuming g_df has a 'label' column for edges
if 'label' in g_df.columns:
    edge_labels = g_df.label.values
else:
    raise ValueError("Dataset must have 'label' column for node classification")

# Create node label mapping from edges
# Each source node gets labeled by its edges
node_labels = {}
for src, label in zip(src_l, edge_labels):
    if src not in node_labels:
        node_labels[src] = label

max_idx = max(src_l.max(), dst_l.max())
num_nodes = max_idx + 1

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {len(src_l)}")

# Filter nodes: only keep those with labels {0, 1}
valid_nodes = []
valid_labels = []

for node_id, label in node_labels.items():
    if label in [0, 1]:  # Only binary labels
        valid_nodes.append(node_id)
        valid_labels.append(label)

valid_nodes = np.array(valid_nodes)
valid_labels = np.array(valid_labels)

print(f"Nodes with valid labels {0, 1}: {len(valid_nodes)}")
print(f"Nodes with label -1 (filtered out): {sum(1 for l in node_labels.values() if l == -1)}")
print(f"Label distribution: 0={np.sum(valid_labels == 0)}, 1={np.sum(valid_labels == 1)}")

# # Check class imbalance
# pos_count = np.sum(valid_labels == 1)
# neg_count = np.sum(valid_labels == 0)
# pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

# print(f"Class imbalance: pos_weight={pos_weight:.2f}")

num_valid = len(valid_nodes)
train_end = int(0.60 * num_valid)
train_labels = valid_labels[:train_end]

train_pos_count = np.sum(train_labels == 1)
train_neg_count = np.sum(train_labels == 0)
pos_weight = train_neg_count / train_pos_count if train_pos_count > 0 else 1.0

print(f"Train Label distribution: 0={train_neg_count}, 1={train_pos_count}")
print(f"Class imbalance (calculated on Train only): pos_weight={pos_weight:.2f}")

# Split nodes: 70% train, 15% val, 15% test
num_valid = len(valid_nodes)
train_end = int(args.train_ratio * num_valid)
val_end = train_end + int(args.val_ratio * num_valid)

# Shuffle nodes for random split
# shuffle_idx = np.random.permutation(num_valid)
# valid_nodes = valid_nodes[shuffle_idx]
# valid_labels = valid_labels[shuffle_idx]

# train_nodes = valid_nodes[:train_end]
# train_labels = valid_labels[:train_end]

# val_nodes = valid_nodes[train_end:val_end]
# val_labels = valid_labels[train_end:val_end]

# test_nodes = valid_nodes[val_end:]
# test_labels = valid_labels[val_end:]

# print("==========================================================")
# print(f"=================*** {MODEL_NAME} ***=============")
# print("==========================================================")
# print(f"Train nodes: {len(train_nodes)}")
# print(f"Val nodes: {len(val_nodes)}")
# print(f"Test nodes: {len(test_nodes)}")
# print(f"Edge feature dim: {e_feat.shape[1]}")
# print(f"Node feature dim: {n_feat.shape[1]}")

print("="*80)
print("Using TEMPORAL split (state-of-the-art for temporal graphs)")
print("="*80)

# Step 1: Create node-to-timestamp mapping (last interaction time for each node)
node_last_timestamp = {}
for src, dst, ts in zip(src_l, dst_l, ts_l):
    for n in [src, dst]:
        if n not in node_last_timestamp or ts > node_last_timestamp[n]:
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

print("✓ Temporal split validation passed")
print("="*60)
print(f"=== {MODEL_NAME} ===")
print("="*60)
print(f"Train nodes: {len(train_nodes)}")
print(f"Val nodes: {len(val_nodes)}")
print(f"Test nodes: {len(test_nodes)}")

# Move features to GPU
edge_raw_features = torch.from_numpy(e_feat).float().to(device)
node_raw_features = torch.from_numpy(n_feat).float().to(device)

NODE_FEAT_DIM = n_feat.shape[1]

# Create tensors for all edge times and features
all_edge_times = torch.from_numpy(ts_l).float().to(device)
all_edge_features = edge_raw_features

# Helper vector to map global node indices to local ones
assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

# Global threshold for test evaluation
BEST_THRESHOLD = 0.5

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

    # Define DyRep model (FROZEN - no training)
    # 1) memory
    dyrep_memory = DyRepMemory(
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
    dyrep_gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM + NODE_FEAT_DIM,
        out_channels=EMB_DIM,
        msg_dim=e_feat.shape[1],
        time_enc=dyrep_memory.time_enc,
    ).to(device)

    # 3) link predictor (not used for node classification, but part of pretrained model)
    link_pred = LinkPredictor(in_channels=EMB_DIM + NODE_FEAT_DIM).to(device)

    # ========== LOAD PRETRAINED DyRep MODEL ==========
    try:
        dyrep_memory, dyrep_gnn, link_pred = load_pretrained_dyrep(
            PRETRAINED_MODEL_PATH, dyrep_memory, dyrep_gnn, link_pred
        )
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("WARNING: Pretrained model not found. Will use random initialization and replay graph.")
        print("WARNING: For best results, train DyRep on link prediction first.")

    # FREEZE DyRep completely
    for param in dyrep_memory.parameters():
        param.requires_grad = False
    for param in dyrep_gnn.parameters():
        param.requires_grad = False
    for param in link_pred.parameters():
        param.requires_grad = False
    
    dyrep_memory.eval()
    dyrep_gnn.eval()
    link_pred.eval()

    print("INFO: DyRep model is FROZEN (no gradient updates)")

    # Replay temporal graph to build memory (required even with pretrained weights)
    # replay_temporal_graph_and_update_memory()
        # OPTIMIZATION: Try to load cached memory state, otherwise replay
    if run_idx == 0:
        # First run: try to load cache or replay
        if not load_jodie_memory_state(dyrep_memory, neighbor_loader, memory_cache_path):
            replay_temporal_graph_and_update_memory()
            # Save state for future runs
            save_jodie_memory_state(dyrep_memory, neighbor_loader, memory_cache_path)
            use_cached_memory = False
        else:
            print("INFO: Using cached JODIE memory state (skipping replay)")
            use_cached_memory = True
    else:
        # Subsequent runs: always load from cache (we know it exists now)
        if not load_jodie_memory_state(dyrep_memory, neighbor_loader, memory_cache_path):
            print("ERROR: Cache should exist but doesn't. Replaying...")
            replay_temporal_graph_and_update_memory()
        else:
            print("INFO: Using cached JODIE memory state (skipping replay)")

    classifier = MLPClassifier(
        input_dim=MEM_DIM + NODE_FEAT_DIM,  # Use memory dimension
        hidden_dim=128,
        dropout=0.3
    ).to(device)

    # Optimizer (only for classifier)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LR,
        fused=True if torch.cuda.is_available() else False
    )
    
    # Loss with pos_weight for class imbalance
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

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
    val_f1_list = []
    start_train_val = timeit.default_timer()
    
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss = train_classifier(epoch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )

        # validation
        start_val = timeit.default_timer()
        val_metrics = evaluate_classifier(val_nodes, val_labels, split_name="val")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print(f"\tValidation Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}, Threshold: {val_metrics['threshold']:.4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_f1_list.append(val_metrics['f1'])

        # check for early stopping (use F1 as metric)
        if early_stopper.step_check(val_metrics['f1'], {'classifier': classifier}):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint({'classifier': classifier})

    # final testing
    start_test = timeit.default_timer()
    test_metrics = evaluate_classifier(test_nodes, test_labels, split_name="test")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"INFO: Test Results:")
    print(f"\tTest Precision: {test_metrics['precision']:.4f}")
    print(f"\tTest Recall: {test_metrics['recall']:.4f}")
    print(f"\tTest F1: {test_metrics['f1']:.4f}")
    print(f"\tTest AUC: {test_metrics['auc']:.4f}")
    print(f"\tUsed Threshold: {test_metrics['threshold']:.4f}")
    
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

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

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")