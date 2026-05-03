# RogueChainDB & NUGGET Architecture

**RogueChainDB** is a labeled dataset of Bitcoin designed for the study of illicit activity detection on blockchain networks. **NUGGET** is a self-supervised framework that uses continuous-time dynamic graphs to model transaction graphs and detect illicit cryptocurrency addresses.

This repository provides the **modular three-stage pipeline** of NUGGET (preprocessing → temporal embedding learning → retrieval-augmented classification) on top of either Elliptic++ or RogueChainDB.

---

## 📚 1. Datasets

### 1.1 RogueChainDB

RogueChainDB comprises **1,085,800 labeled Bitcoin addresses**, constructed by harmonizing data from the underground forum *HackForums* and public academic repositories.

This dataset is organized into **four main components**:

#### Part 1: Address Features
*   **File:** `bitcoin_address_features.npy`
*   **Description:** Contains **eleven transaction and temporal features** for each cryptocurrency address. These characterize address behavior over time and serve as static node attributes for graph-based models.

#### Part 2: Address–Transaction Mapping
*   **File:** `bitcoin_transactions.csv`
*   **Description:** Maps each address to its corresponding **transaction identifiers**.

#### Part 3: Address Labels
*   **Description:** Ground-truth data where each address is associated with:
    1.  A **binary class label**: *Licit* or *Illicit*.
    2.  A **fine-grained illicit category** (e.g., Ransomware, Scamming, Black Market, Ponzi, Laundering).
*   *Note: Labels are curated from Hackforums and public sources.*

#### Part 4: Address Interactions (The Graph)
*   **Description:** Captures **pairwise interactions** between input and output addresses.
*   **Attributes:**
    *   **Transfer Volume:** Measured in satoshis.
    *   **Timestamp:** Represented by **Block Height**.

> **Note on Block Height:** Block height is the sequential index of a block, starting from zero at the Genesis Block. It provides an immutable temporal reference for the network's evolution and consensus history.

#### 📥 Data Availability (Anonymous)
To maintain the integrity of the **double-blind review process**, the dataset has been hosted on an anonymous Google Drive account created specifically for this submission.

**🔗 Download Link:** https://drive.google.com/file/d/1a9Ek0IvbxnS-RHvUo_2wtIqg5ExS0Om6/view?usp=sharing

### 1.2 Elliptic++

Elliptic++ is the public benchmark used as a secondary evaluation. It is released by the original authors and can be downloaded directly from their repository.

**🔗 Download Link:** https://github.com/git-disl/EllipticPlusPlus

After download, the expected layout is:
```
EllipticPlusPlus/
└── Actors Dataset/
    ├── wallets_features.csv
    ├── wallets_classes.csv
    └── addr1_addr2_with_timestamp.csv
```

---

## 🧠 2. Learning Architecture: NUGGET

### Architecture Overview
1.  **Self-Supervised Embedding Learning:**
    *   Uses a **Temporal Graph Neural Network (TGNN)** to learn dynamic node embeddings via a link-prediction objective.
    *   Captures structural and temporal patterns without using any ground-truth labels.
2.  **Retrieval-Augmented Node Classification (RANC):**
    *   Uses the frozen embeddings from Phase 1 to train a retrieval-augmented classifier with licit banks.
    *   Detects illicit addresses even in highly imbalanced settings.

The pipeline is split into **three independent scripts** so that any stage can be re-run without redoing the previous ones.

---

## 🛠️ 3. Setup and Dependencies

### Prerequisites
All dependencies required to run the NUGGET architecture are listed in `requirements.txt`.

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### ⚠️ Artifact Evaluation
Due to the large size of the **processed features**, we host them on an anonymous drive. You need to download the processed dataset from the provided link and place it in the `processed/` directory. The dataset includes the **processed address feature matrices, edge feature matrices, and transaction graphs**.

```
processed/
├── bitcoin_address_features.npy
├── bitcoin_transaction_features.npy
└── bitcoin_transactions.csv
```

**🔗 Download Link:** https://drive.google.com/file/d/15eFKi4vbmhCZDViX3fZRaDvZ-TK73PPL/view?usp=sharing

---

## 🚀 4. Usage — Modular Three-Stage Pipeline

### Stage 1: Preprocessing (`preprocess.py`)
Converts raw graph CSVs/NPYs into dense numpy arrays consumed by stages 2 and 3.

- **Input**: Raw Elliptic++ CSVs (`wallets_features.csv`, `wallets_classes.csv`, `addr1_addr2_with_timestamp.csv`)
  or RogueChainDB (`bitcoin_transactions.csv` + node/edge feature `.npy` files)
- **Output**:
  - `node_feats.npy` — normalized node features (signed log1p + RobustScaler + clipping)
  - `node_labels.npy` — class labels (-1 unknown, 0 licit, 1 illicit)
  - `edges.npy` — edge list (E, 2)
  - `ts.npy` — timestamps (E,)
  - `edge_feats.npy` — edge features
- **Caching**: Automatically cached under `--cache-dir`; rerun with `--force-preprocess` to rebuild.

### Stage 2: Learn Node Embeddings (`learn_embeddings.py`)
Trains a TGN backbone via link prediction with hard-negative mining.

- **Input**: Preprocessed files from Stage 1 (`--cache-dir`)
- **Output**: `saved_models/{dataset_tag}/TGN_PRETRAIN_{seed}_{run}.pth`
- **Key features**:
  - AMP autocast + gradient scaling for memory efficiency
  - MRR + Hits@k evaluation on held-out temporal windows

### Stage 3: RANC Classification (`classify_ranc.py`)
Builds class-specific prototype banks and trains a retrieval-augmented classifier.

- **Input**:
  - Preprocessed files from Stage 1 (`--cache-dir`)
  - Trained TGN checkpoint from Stage 2 (`--pretrained-model`)
- **Output**:
  - `saved_models/{dataset_tag}/RANC_{dataset_tag}_{seed}_{run}.pth` — best classifier
  - `saved_results/{dataset_tag}/RANC_full_run{run}.json` — metrics (P/R/F1/AUC/AP)
- **Key features**:
  - Multi-scale retrieval with learnable projections
  - Binary focal loss 
  - Optional TGN fine-tuning at LR/10
  - Threshold tuning on validation set

---

## 5. Quick Start

### RogueChainDB

```bash
# Stage 1: Preprocess (11 static features)
python preprocess.py \
    --dataset roguechaindb \
    --rdb-transactions ./processed/bitcoin_transactions.csv \
    --rdb-node-features ./processed/bitcoin_address_features.npy \
    --rdb-edge-features ./processed/bitcoin_transaction_features.npy \
    --rdb-ts-mode calendar2w \
    --cache-dir ./cache/rdb_calendar2w

# Stage 2: Learn embeddings
python learn_embeddings.py \
    --cache-dir  ./cache/rdb_calendar2w \
    --pretrain-epochs 5 \
    --mem-dim 128 --time-dim 128 --emb-dim 128 \
    --seed 42

# Stage 3: RANC classification 
python classify_ranc.py \
    --cache-dir  ./cache/rdb_calendar2w \
    --pretrained-model saved_models/roguechaindb/TGN_PRETRAIN_42_0.pth \
    --ranc-top-m 10 --ranc-num-scales 2 \
    --num-epoch 40 \
    --seed 42

```

### Elliptic++

```bash
# Stage 1: Preprocess
python preprocess.py \
    --dataset elliptic \
    --data-root ./EllipticPlusPlus \
    --cache-dir ./cache/elliptic_refined \
    --feature-mode refined

# Stage 2 & 3: same as Roguechaindb but with --cache-dir ./cache/elliptic_refined
```


Metrics are printed to stdout and also written to `saved_results/{dataset_tag}/RANC_full_run{run}.json`.

---

## 6. Arguments by Stage

### `preprocess.py`

**Data & I/O**
- `--dataset`: `elliptic` or `roguechaindb`
- `--data-root`: Path to raw Elliptic++ files (containing `Actors Dataset/`)
- `--cache-dir`: Output directory for preprocessed `.npy` files
- `--force-preprocess`: Rebuild cache even if it exists

**Feature Selection** (Elliptic++ only)
- `--feature-mode`: `refined` (Top-10 + BTC_received + structural), `top10`, or `all`
- `--feature-agg`: `last` (default), `mean`, or `temporal_rich` (6× dim)
- `--add-topology-feats`: Include graph topology features (degree, span, etc.)

**RogueChainDB**
- `--rdb-transactions`: Path to `bitcoin_transactions.csv`
- `--rdb-node-features`: Path to address features `.npy`
- `--rdb-edge-features`: Path to edge features `.npy`
- `--rdb-ts-mode`: `raw`, `quantile`, `calendar2w` (2016-block bins = 2 weeks), or `calendar2w_dense`
- `--rdb-ts-bins`: Quantile-bucket count (for `quantile` mode); window size for `calendar2w_dense`

### `learn_embeddings.py`

**Data**
- `--cache-dir`: Path to preprocessed files (required)

**Training**
- `--seed`: Random seed 
- `--pretrain-epochs`: Number of pretraining epochs (default 20)
- `--bs`: Batch size 
- `--lr`: Learning rate

**TGN Architecture**
- `--mem-dim`: Memory dimension 
- `--time-dim`: Time encoding dimension 
- `--emb-dim`: GNN embedding dimension 
- `--num-neighbors`: K-hop neighborhood size 

**I/O**
- `--dataset-tag`: Tag for saved models dir
- `--num-run`: Number of runs 

### `classify_ranc.py`

**Data**
- `--cache-dir`: Path to preprocessed files (required)
- `--pretrained-model`: Path to TGN checkpoint from Stage 2 (required)

**Training**
- `--seed`: Random seed 
- `--num-epoch`: RANC training epochs 
- `--bs`: Batch size 
- `--lr`: Classifier learning rate 
- `--fine-tune-tgn`: Unfreeze TGN during RANC training at LR/10 (flag)

**Splits**
- `--split-type`: `stratified`, `temporal`, `random`, or `paper_temporal` 
- `--train-ratio`: Fraction for train 
- `--val-ratio`: Fraction for validation 
- `--paper-cut-ts`: Cutoff for `paper_temporal` mode (default 33, matching Elliptic++ paper)
- `--bank-licit-max`: Cap licit prototype bank size 

**RANC Hyperparameters**
- `--ranc-top-m`: Number of neighbors to retrieve
- `--ranc-num-scales`: Number of retrieval scales 
- `--ranc-similarity`: Similarity metric: `cosine`, `pearson`, or `l2` (default `cosine`)
- `--ranc-temperature`: Softmax temperature for attention 
- `--ranc-hidden`: Hidden dimension of MLP head 
- `--ranc-dropout`: Dropout in classifier
- `--dual-bank`: Build both licit AND illicit prototype banks

**Loss & Optimization**
- `--focal-gamma`: Focal loss γ 
- `--focal-alpha`: Focal loss α

**Early Stopping**
- `--tolerance`: Tolerance for improvement 
- `--patience`: Patience for early stopping
  
**I/O**
- `--dataset-tag`: Tag for saved results 
- `--log-test-per-epoch`: Also evaluate test set every epoch (flag, diagnostic only)

---

## 7. Output Structure

```
saved_models/
  elliptic_full/
    TGN_PRETRAIN_42_0.pth         # Stage 2 output
    RANC_elliptic_full_42_0.pth   # Stage 3 checkpoint (best F1)

saved_results/
  elliptic_full/
    RANC_full_run0.json           # Stage 3 metrics
```

---

## 8. Dependencies

Core: `torch`, `numpy`, `pandas`, `scikit-learn`, `tqdm`

TGN modules (assumed in `modules/`):
- `memory_module.py`: `TGNMemory`
- `neighbor_loader.py`: `LastNeighborLoader`
- `msg_func.py`: `IdentityMessage`
- `msg_agg.py`: `LastAggregator`
- `emb_module.py`: `GraphAttentionEmbedding`
- `decoder.py`: `LinkPredictor`
- `early_stopping.py`: `EarlyStopMonitor`

---

## 📄 License

This project (RogueChainDB & NUGGET) is licensed under the **BSD 3-Clause License**.
