# Surrogate encoder under output perturbations (model stealing)

PyTorch project that trains a **surrogate image encoder** to mimic a **black-box victim model** that returns **1024-dimensional embeddings**, in the presence of a **perturbation-based defense** (B4B-style noise on API outputs). The pipeline covers **batched API querying**, **defense-aware supervised training**, **Optuna hyperparameter search**, and **ONNX export** for evaluation or deployment.


## Results

Reported **mean L2 distance** between surrogate and victim embeddings on held-out evaluation: **4.706** (lower is better). Earlier iterations (larger architecture / naive training) were on the order of **L2 ≈ 120**; intermediate BESA-inspired experiments reached roughly **~25** before the final architecture and loss design.


## What this repo contains

| File | Purpose |
|------|--------|
| [`collect_victim_embeddings.py`](collect_victim_embeddings.py) | Load the public image dataset, query the victim API in batches, save `out*.pickle` (indices + embeddings). |
| [`train_surrogate_encoder.ipynb`](train_surrogate_encoder.ipynb) | Main pipeline: `StealingDataset`, `EnhancedResNetEncoder`, `HybridLoss`, Optuna, training, ONNX export, optional HTTP submit. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies. |
| [`.env.example`](.env.example) | Template for **non-committed** secrets and paths. |
| [`.gitignore`](.gitignore) | Ignores `.env`, datasets, checkpoints, and large artifacts. |

## Method

1. **Data collection:** Shuffle indices, query the victim in batches (e.g. 13×1000 images), persist embeddings next to indices for reproducible pairing with images.
2. **Model:** Lightweight residual CNN for **32×32** RGB inputs, **1024-D** output, **GELU**, **dropout**, configurable bottleneck width.
3. **Loss:** **HybridLoss** — weighted **MSE** + **cosine** term + regularizer using **shuffled** batch targets to reduce overfitting to noisy directions.
4. **Training:** AdamW, cosine warm restarts, gradient clipping, early stopping; **Optuna** (e.g. 20 trials) over LR, batch size, bottleneck, dropout, loss mix, weight decay.
5. **Export:** ONNX with dynamic batch axis; optional verification with ONNX Runtime.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS

pip install -r requirements.txt
cp .env.example .env       # then edit: set VICTIM_API_TOKEN, paths, etc.
```

Place the public tensor dataset (e.g. `ModelStealingPub.pt`) where `PUBLIC_DATASET_PATH` points. Put API query results under `EMBEDDINGS_DIR` (default `embeddings/`) as `out1.pickle`, `out2.pickle`, …

## Running

1. **Collect embeddings** (requires a running victim API and valid token):

   ```bash
   set VICTIM_API_TOKEN=your_token
   python collect_victim_embeddings.py
   ```

2. **Train** — open `train_surrogate_encoder.ipynb`, set environment variables (or use your IDE’s env loader), run all cells. Uncomment `submit_model()` in the notebook if you submit ONNX to a grading server.


## References (techniques cited in the original write-up)

- Contrastive / representation ideas: [CVPR 2023 paper](https://yangzhangalmo.github.io/papers/CVPR23.pdf) (MoCo-style inspiration in early experiments).
- BESA / perturbation recovery: [arXiv:2506.04556](https://arxiv.org/pdf/2506.04556).

