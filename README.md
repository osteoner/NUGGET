# RogueChainDB & Nugget Architecture

**RogueChainDB** is a structured cryptocurrency transaction dataset designed for the study of illicit activity detection on blockchain networks. **Nugget** is the accompanying modular learning architecture for continuous-time dynamic graphs.

---

## 📚 1. Dataset: RogueChainDB

RogueChainDB is organized into **four main components**, enabling the reconstruction of transaction histories and address-level activity patterns.

### 📥 Data Availability (Anonymous)
To maintain the integrity of the **double-blind review process**, the full dataset has been hosted on an anonymous Google Drive account created specifically for this submission.

**🔗 Download Link:** [INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE]

### Part 1: Address Features
*   **File:** `processed/bitcoin_address_features.npy`
*   **Description:** Contains **eight transaction and temporal features** for each cryptocurrency address. These characterize address behavior over time and serve as static node attributes for graph-based models.

### Part 2: Address–Transaction Mapping
*   **File:** `processed/bitcoin_transactions.csv`
*   **Description:** Maps each address to its corresponding **transaction identifiers**.

### Part 3: Address Labels
*   **Description:** Ground-truth data where each address is associated with:
    1.  A **binary class label**: *Licit* or *Illicit*.
    2.  A **fine-grained illicit category** (e.g., Ransomware, Scamming, Black Market, Ponzi, Laundering).
*   *Note: Labels are curated from public sources and expert validation.*

### Part 4: Address Interactions (The Graph)
*   **Description:** Captures **pairwise interactions** between input and output addresses.
*   **Attributes:**
    *   **Transfer Volume:** Measured in satoshis.
    *   **Timestamp:** Represented by **Block Height**.
    
> **Note on Block Height:** Block height is the sequential index of a block, starting from zero at the Genesis Block. It provides an immutable temporal reference for the network’s evolution and consensus history.

---

## 🧠 2. Learning Architecture: Nugget

**Nugget** is a modular framework for learning address embeddings and performing node-level classification on continuous-time dynamic graphs.

### Architecture Overview
The learning pipeline consists of two main stages:
1.  **Embedding Learning:** Utilizing temporal graph models to capture evolving structural patterns.
2.  **Node Classification:** Detecting illicit addresses based on the learned representations.

The architecture supports three state-of-the-art backbones:
*   **DyRep**
*   **TGN** (Temporal Graph Networks)
*   **JODIE**

---

## 🛠️ 3. Setup and Dependencies

### Prerequisites
All dependencies required to run the Nugget architecture are listed in `requirements.txt`.

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## Ensure the preprocessed data is placed in the processed/ directory:
processed/
├── bitcoin_address_features.npy
├── bitcoin_transaction_features.npy
└── bitcoin_transactions.csv

## 🚀 4. Usage

The pipeline allows you to train embeddings using different backbones and then perform classification.

### Step 1: Embedding Learning
Run one of the following scripts to learn temporal node embeddings from the transaction graph.

```bash
# Option A: DyRep
python dyrep_learn_embedding.py --data processed/bitcoin_transactions.csv

# Option B: Temporal Graph Networks (TGN)
python tgn_learn_embedding.py --data processed/bitcoin_transactions.csv

# Option C: JODIE
python jodie_learn_embedding.py --data processed/bitcoin_transactions.csv
```
### Step 2: Node Classification
Pass the learned embeddings to the classifier to predict Licit vs Illicit labels.


```bash
# Classify using DyRep embeddings
python dyrep_node_classification.py

# Classify using TGN embeddings
python tgn_node_classification.py

# Classify using JODIE embeddings
python jodie_node_classification.py
```
