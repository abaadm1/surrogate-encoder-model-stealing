"""
Batch-query a black-box encoder API and save (image indices, embeddings) to pickle files.

Set environment variables (see `.env.example`). Do not commit real tokens.
"""
import base64
import io
import json
import os
import pickle
import time

import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from typing import Tuple

# === Configuration (use environment variables) ===
TOKEN = os.environ.get("VICTIM_API_TOKEN", "")
# Base URL without trailing path, e.g. http://hostname — query uses port from VICTIM_PORT
VICTIM_QUERY_HOST = os.environ.get("VICTIM_QUERY_HOST", "http://34.122.51.94")
PORT = os.environ.get("VICTIM_PORT", "9025")
PUBLIC_DATASET_PATH = os.environ.get("PUBLIC_DATASET_PATH", "ModelStealingPub.pt")

# Optional delay between batch requests (seconds)
DELAY_SECONDS = int(os.environ.get("QUERY_DELAY_SECONDS", "90"))
N_QUERIES = int(os.environ.get("N_QUERY_BATCHES", "13"))
BATCH_SIZE = int(os.environ.get("QUERY_BATCH_SIZE", "1000"))


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


torch.serialization.add_safe_globals({"TaskDataset": TaskDataset})


def query_encoder(images, port: str) -> list:
    endpoint = "/query"
    url = f"{VICTIM_QUERY_HOST.rstrip('/')}:{port}{endpoint}"
    image_data = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        image_data.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    payload = json.dumps(image_data)
    response = requests.get(url, files={"file": payload}, headers={"token": TOKEN})
    if response.status_code == 200:
        return response.json()["representations"]
    raise RuntimeError(
        f"API query failed. Code: {response.status_code}, content: {response.text}"
    )


def main():
    if not TOKEN:
        raise SystemExit(
            "Set VICTIM_API_TOKEN (see .env.example). Never commit secrets to git."
        )

    dataset = torch.load(PUBLIC_DATASET_PATH, weights_only=False)
    all_indices = np.random.permutation(len(dataset.imgs))

    print(f"Dataset loaded with {len(dataset.imgs)} images from {PUBLIC_DATASET_PATH}.")
    print(
        f"Starting {N_QUERIES} queries ({BATCH_SIZE} images each), "
        f"{DELAY_SECONDS}s delay between batches..."
    )

    for i in range(N_QUERIES):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        if end_idx > len(dataset.imgs):
            print(f"Not enough images to fill batch {i + 1}")
            break

        batch_indices = all_indices[start_idx:end_idx]
        batch_images = [dataset.imgs[idx] for idx in batch_indices]

        print(f"[{i + 1}/{N_QUERIES}] Querying API for indices {start_idx}–{end_idx - 1}...")
        embeddings = query_encoder(batch_images, port=PORT)

        save_data = {
            "indices": batch_indices.tolist(),
            "embeddings": embeddings,
        }
        filename = f"out{i + 1}.pickle"
        with open(filename, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved results to {filename}")

        if i < N_QUERIES - 1:
            print(f"Waiting {DELAY_SECONDS} seconds before next query...\n")
            time.sleep(DELAY_SECONDS)

    print("All queries completed.")


if __name__ == "__main__":
    main()
