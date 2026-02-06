## RogueChainDB & NUGGET Architecture

**RogueChainDB** is a labeled dataset of Bitcoin designed for the study of illicit activity detection on blockchain networks. **NUGGET** is a self-supervised framework that uses continuous-time dynamic graphs to model transaction graphs and detect illicit cryptocurrency addresses.
---

## 📚 1. Dataset: RogueChainDB
RogueChainDB comprises **1,085,800 labeled Bitcoin addresses**, constructed by harmonizing data from the underground forum *HackForums* and public academic repositories.
This dataset is organized into **four main components**:

### Part 1: Address Features
*   **File:** `bitcoin_address_features.npy`
*   **Description:** Contains **eight transaction and temporal features** for each cryptocurrency address. These characterize address behavior over time and serve as static node attributes for graph-based models.

### Part 2: Address–Transaction Mapping
*   **File:** `bitcoin_transactions.csv`
*   **Description:** Maps each address to its corresponding **transaction identifiers**.

### Part 3: Address Labels
*   **Description:** Ground-truth data where each address is associated with:
    1.  A **binary class label**: *Licit* or *Illicit*.
    2.  A **fine-grained illicit category** (e.g., Ransomware, Scamming, Black Market, Ponzi, Laundering).
*   *Note: Labels are curated from Hackforums and public sources.*

### Part 4: Address Interactions (The Graph)
*   **Description:** Captures **pairwise interactions** between input and output addresses.
*   **Attributes:**
    *   **Transfer Volume:** Measured in satoshis.
    *   **Timestamp:** Represented by **Block Height**.
    
> **Note on Block Height:** Block height is the sequential index of a block, starting from zero at the Genesis Block. It provides an immutable temporal reference for the network’s evolution and consensus history.

### 📥 Data Availability (Anonymous)
To maintain the integrity of the **double-blind review process**, the  dataset has been hosted on an anonymous Google Drive account created specifically for this submission.
**🔗 Download Link:** [https://drive.google.com/file/d/1a9Ek0IvbxnS-RHvUo_2wtIqg5ExS0Om6/view?usp=sharing]


---

## 🧠 2. Learning Architecture: NUGGET

### Architecture Overview
1.  **Self-Supervised Embedding Learning:** 
    *   Uses a **Temporal Graph Neural Network (TGNN)** to learn dynamic node embeddings via a link-prediction objective.
    *   Captures structural and temporal patterns without using any ground-truth labels.
2.  **Supervised Node Classification:** 
    *   Uses the frozen embeddings from Phase 1 to train a lightweight **Multi-Layer Perceptron (MLP)**.
    *   Detects illicit addresses even in highly imbalanced settings.

### Supported Backbones
The framework implements and compares two state-of-the-art continuous-time models:
*   **TGN** (Temporal Graph Networks) 
*   **DyRep**

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
Due to the large size of the **processed features**, we host them on an anonymous drive. You need to download the processed dataset from the provided link and place it in the processed/ directory. The dataset includes the **processed address feature matrices, edge feature matrices, and transaction graphs**.
```bash
processed/
├── bitcoin_address_features.npy
├── bitcoin_transaction_features.npy
└── bitcoin_transactions.csv
```

**🔗 Download Link:** [https://drive.google.com/file/d/15eFKi4vbmhCZDViX3fZRaDvZ-TK73PPL/view?usp=sharing]
## 🚀 4. Usage

The pipeline allows you to pretrain the models using different backbones and then perform classification.

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
