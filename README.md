# RogueChainDB

**RogueChainDB** is a structured cryptocurrency transaction dataset designed for the study of illicit activity detection on blockchain networks. It provides address-level features, transaction mappings, behavioral labels, and interaction graphs to support graph-based and temporal learning tasks.

---

## Dataset Structure

RogueChainDB is organized into **four main components**, summarized below:

### Part 1: Address Features
This component contains **eight transaction and temporal features** for each cryptocurrency address. These features characterize address behavior over time and are intended to serve as node attributes in graph-based models.

---

### Part 2: Address–Transaction Mapping
This component maps each address to its corresponding **transaction identifiers**, enabling the reconstruction of transaction histories and address-level activity patterns.

---

### Part 3: Address Labels
Each address is associated with:
- A **binary class label**: *Licit* or *Illicit*
- A **fine-grained illicit category**, when applicable (e.g., Ransomware, Scamming, Black Market, Ponzi, Laundering)

These labels are curated from public sources and expert validation.

---

### Part 4: Address Interactions
This component captures **pairwise interactions** between input and output addresses. Each interaction includes:
- **Block height**, serving as a timestamp and enabling chronological ordering
- **Transfer volume**, measured in satoshis

Block height is the sequential index of a block in the blockchain, starting from zero at the Genesis Block. It provides an immutable temporal reference for the network’s evolution and consensus history.

---

## Use Cases
RogueChainDB is suitable for:
- Illicit address detection
- Transaction graph analysis
- Temporal and continuous-time graph learning
- Self-supervised and semi-supervised node classification
- Blockchain forensics and security research

---

## License and Usage
This dataset is intended for **research purposes**. Please cite the associated paper if you use RogueChainDB in your work.
