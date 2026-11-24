"""Usage:
    git clone --depth 1 https://github.com/bolattleubayev/cmnist.git
    python load_cmnist_russian.py cmnist
Returns 28x28 grayscale images as numpy arrays plus labels 0..32 (А..Я).
"""
import csv
from pathlib import Path
import os
import sys
import constants
import file_handler

import numpy as np
from PIL import Image

def load_split(repo, split="train", variant="allgr"):
    """split: train | validation | test.  variant: allgr (grayscale)."""
    repo = Path(repo)
    xs, ys = [], []
    with open(repo / f"{split}.csv", newline="") as f:
        for row in csv.DictReader(f):
            label = int(float(row["Russian"])) # last row
            if label < 0:  # Kazakh-only letter, skip
                continue
            
            # reads the PNG file and gives you an image object holding the pixel data plus its "mode", which describes how each pixel is stored.
            # .convert("L") changes the mode to L, which stands for luminance: 8-bit grayscale, one value per pixel from 0 (black) to 255 (white).
            # allgr should already be already be gray tho
            img = Image.open(repo / variant / row["filename"]).convert("L")
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(label)
            
    # np.stack means stack these arrays on top of each other
    return np.stack(xs), np.array(ys, dtype=np.uint8)

def get_train_data(repo_folder = "cmnist"):
    if not os.path.exists(repo_folder):
        return False
    arrays = {}
    for split in ("train", "validation", "test"):
        x, y = load_split(repo_folder, split)
        arrays[f"x_{split}"], arrays[f"y_{split}"] = x, y
        print(f"[INFO] {split:<10} x={x.shape} y={y.shape} classes={len(set(y.tolist()))}")
    file_handler.save_cyrilic_dataset(constants.CYRILIC_DATASET_ARCHIVE, **arrays)
    print("[INFO] Saved " + constants.CYRILIC_DATASET_ARCHIVE + " (x_train, y_train, x_validation, ...)")
    return True

def invert_normalize(x):
    x = (255 - x.astype(np.float32)) / 255.0
    return x[..., np.newaxis]                  

def load_or_build_cyrilic_dataset():
    dataset = file_handler.load_cyrilic_dataset(constants.CYRILIC_DATASET_ARCHIVE)
    if dataset is None:
        print("[WARN] Dataset archive not found, building it from the cmnist repo...")
        if not get_train_data():
            sys.exit("[ERROR] cmnist repo not found. ...")
        dataset = file_handler.load_cyrilic_dataset(constants.CYRILIC_DATASET_ARCHIVE)

    return {k: invert_normalize(v) if k.startswith("x_") else v for k, v in dataset.items()}