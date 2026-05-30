# TeRDy

**TeRDy: Temporal Relation Dynamics through Frequency Decomposition for Temporal Knowledge Graph Completion**

This repository contains the code for **TeRDy**, an ACL 2025 paper on temporal knowledge graph completion.

TeRDy studies a simple but important question:  
**how can we model relations that stay stable, change slowly, and also change quickly over time?**

To answer this, TeRDy splits temporal relation dynamics into different frequency parts and combines them with timestamp signals for better prediction.

## 🧠 What TeRDy Does

In a temporal knowledge graph, not all relations change in the same way:

- **Time-invariant relations** stay almost the same over time.
- **Long-term dynamic relations** change slowly across a long period.
- **Short-term dynamic relations** can change quickly in a short period.

TeRDy models these patterns by:

- using relation embeddings with **frequency decomposition**
- separating **low-frequency** and **high-frequency** relation signals
- combining them with timestamp information through **temporal smoothing** and **temporal gradient**

This helps the model capture both stable trends and short-time changes.

## ✨ Highlights

- Clear temporal modeling with low-frequency and high-frequency relation components
- Simple training pipeline for **ICEWS14**, **ICEWS05-15**, and **GDELT**
- Ready-to-run scripts for reproducing the main results in the paper
- Lightweight codebase built on top of earlier TKGC embedding frameworks

## 📄 Paper

- **Title**: *TeRDy: Temporal Relation Dynamics through Frequency Decomposition for Temporal Knowledge Graph Completion*
- **Venue**: ACL 2025
- [🔗](https://aclanthology.org/2025.acl-long.473/)

## 🗂️ Repository Structure

```text
.
├── learner.py                 # training and evaluation entry
├── models.py                  # TeRDy model and related scoring code
├── optimizers.py              # training loop and loss combination
├── datasets.py                # dataset loading and filtered evaluation
├── regularizers.py            # embedding and temporal regularization
├── process_icews.py           # preprocess ICEWS14 and ICEWS05-15
├── process_gdelt.py           # preprocess GDELT
├── run_TeRDy_GDELT.sh         # reproduce GDELT results
├── run_TeRDy_ICEWS14.sh       # reproduce ICEWS14 results
├── run_TeRDy_ICEWS15.sh       # reproduce ICEWS05-15 results
├── requirements.txt           # package list
└── src_data/                  # raw dataset files
```

## ⚙️ Installation

We recommend creating a fresh conda environment first:

```bash
conda create --name terdy_env python=3.8
conda activate terdy_env
conda install --file requirements.txt -c pytorch
```

If your system uses `source activate`, that also works:

```bash
source activate terdy_env
```

## 📦 Dataset Preparation

This project uses three benchmark datasets:

- `ICEWS14`
- `ICEWS05-15`
- `GDELT`

Raw files should be placed under `src_data/`.  
If you need the complete dataset package, you can download it from:

[TCompoundE dataset folder](https://github.com/nk-ruiying/TCompoundE/tree/main/src_data)

Then run the preprocessing scripts:

```bash
python process_icews.py
python process_gdelt.py
```

These scripts will:

- map entities, relations, and timestamps to ids
- create `train.pickle`, `valid.pickle`, and `test.pickle`
- build `to_skip.pickle` for filtered metrics
- save processed data under the `data/` folder

## 🚀 Reproducing the Main Results

To reproduce the results reported in the paper, run:

```bash
bash run_TeRDy_GDELT.sh
bash run_TeRDy_ICEWS14.sh
bash run_TeRDy_ICEWS15.sh
```

The shell scripts call `learner.py` with dataset-specific settings such as:

- embedding rank
- batch size
- learning rate
- temporal regularization strength
- frequency regularization strength
- GPU id

## 🔧 Main Training Command

You can also train the model directly:

```bash
python learner.py \
  --dataset ICEWS14 \
  --rank 6000 \
  --batch_size 4000 \
  --learning_rate 0.02 \
  --emb_reg 0.005 \
  --time_reg 0.005 \
  --freq_reg 0.0005 \
  --alpha 10 \
  --valid_freq 20 \
  --max_epochs 101 \
  --gpu 0
```

## 🧪 Important Arguments

- `--dataset`: dataset name
- `--rank`: embedding size
- `--batch_size`: training batch size
- `--learning_rate`: optimizer learning rate
- `--emb_reg`: embedding regularization weight
- `--time_reg`: temporal regularization weight
- `--freq_reg`: frequency regularization weight
- `--alpha`: controls low-frequency and high-frequency separation
- `--valid_freq`: validation interval
- `--max_epochs`: maximum training epochs
- `--gpu`: GPU index

## 📈 Outputs

Training outputs are saved under the `results/` directory, grouped by:

- dataset
- model
- rank
- learning rate
- batch size
- regularization settings

Each run stores:

- `result.txt` for logged evaluation results
- `TeRDy.pkl` for the best saved model

## 🧩 Method Summary

At a high level, TeRDy works like this:

1. Learn embeddings for entities, relations, and timestamps.
2. Decompose relation embeddings into low-frequency and high-frequency parts with FFT.
3. Use timestamp smoothing to capture slow temporal trends.
4. Use timestamp gradients to capture quick temporal changes.
5. Fuse these signals for temporal link prediction.

## Acknowledgement

This training framework is mainly built on ideas and code structure from:

- **TeAST**  
  *TeAST: Temporal Knowledge Graph Embedding via Archimedean Spiral Timeline*  
  ACL 2023

- **TCompoundE**  
  *Simple but Effective Compound Geometric Operations for Temporal Knowledge Graph Completion*  
  ACL 2024

We thank the authors of these works for making related ideas and resources available.

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{liu2025terdy,
  title={TeRDy: Temporal Relation Dynamics through Frequency Decomposition for Temporal Knowledge Graph Completion},
  author={Liu, Ziyang and Wang, Chaokun},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2025},
  pages={9611--9622}
}
```
