import os
import json
import csv
import numpy as np
from typing import List, Tuple, Dict


# -----------------------------
# User settings
# -----------------------------
CHUNK_DIR = "labeled_data_2025_parallel"   # folder containing v251209_chunk_XXXX.npz
CHUNK_PREFIX = "v251209_chunk_"
CHUNK_SUFFIX = ".npz"

OUT_DIR = "ml_splits_v251209"              # output folder for splits + diagnostics

TEST_SIZE = 100_000
VAL_SIZE = 50_000

RNG_SEED = 12345                           # fixed seed for reproducibility; do not change after splits are generated

SHUFFLE = True                             # keep True: random split across global index space

# If True, also saves the full concatenated arrays X_all.npy and y_all.npy.
# Not required for the learning-curve analyses; set to False by default.
SAVE_FULL_DATASET = False


# -----------------------------
# Helpers
# -----------------------------
def list_chunk_files(chunk_dir: str) -> List[str]:
    files = [
        os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)
        if f.startswith(CHUNK_PREFIX) and f.endswith(CHUNK_SUFFIX)
    ]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir} matching {CHUNK_PREFIX}*{CHUNK_SUFFIX}")
    return files


def inspect_chunks(chunk_files: List[str]) -> Dict:
    """
    Inspect each chunk file:
      - check keys exist
      - record shapes and dtypes
      - compute label fraction
    Returns a dict with:
      - chunk_files, chunk_sizes, num_cells, dtype_data, dtype_labels
      - label_fractions list
      - total_samples
    """
    chunk_sizes = []
    label_fracs = []
    num_cells_ref = None
    dtype_data_ref = None
    dtype_labels_ref = None

    for path in chunk_files:
        with np.load(path) as z:
            if "data" not in z.files or "labels" not in z.files:
                raise ValueError(f"{path} does not contain 'data' and 'labels'. Found: {z.files}")

            data = z["data"]
            labels = z["labels"]

            if data.ndim != 2:
                raise ValueError(f"{path}: data should be 2D, got shape {data.shape}")
            if labels.ndim != 1:
                raise ValueError(f"{path}: labels should be 1D, got shape {labels.shape}")
            if data.shape[0] != labels.shape[0]:
                raise ValueError(f"{path}: data rows {data.shape[0]} != labels len {labels.shape[0]}")

            chunk_sizes.append(int(data.shape[0]))
            num_cells = int(data.shape[1])

            # Check consistency across chunks
            if num_cells_ref is None:
                num_cells_ref = num_cells
                dtype_data_ref = str(data.dtype)
                dtype_labels_ref = str(labels.dtype)
            else:
                if num_cells != num_cells_ref:
                    raise ValueError(f"{path}: num_cells {num_cells} != reference {num_cells_ref}")
                if str(data.dtype) != dtype_data_ref:
                    raise ValueError(f"{path}: data dtype {data.dtype} != reference {dtype_data_ref}")
                if str(labels.dtype) != dtype_labels_ref:
                    raise ValueError(f"{path}: labels dtype {labels.dtype} != reference {dtype_labels_ref}")

            # label fraction
            # labels are int8 0/1, so mean is fraction of 1s
            frac_static = float(labels.mean())
            label_fracs.append(frac_static)

    total_samples = int(np.sum(chunk_sizes))

    return {
        "chunk_files": chunk_files,
        "chunk_sizes": chunk_sizes,
        "num_cells": num_cells_ref,
        "dtype_data": dtype_data_ref,
        "dtype_labels": dtype_labels_ref,
        "label_fractions_static": label_fracs,
        "total_samples": total_samples
    }


def save_label_diagnostics(meta: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # CSV per chunk
    csv_path = os.path.join(out_dir, "chunk_label_fractions.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_file", "chunk_size", "frac_static(label=1)"])
        for path, n, frac in zip(meta["chunk_files"], meta["chunk_sizes"], meta["label_fractions_static"]):
            w.writerow([os.path.basename(path), n, frac])

    # Overall stats JSON
    overall_frac = float(
        np.average(meta["label_fractions_static"], weights=meta["chunk_sizes"])
    )

    stats = {
        "chunk_dir": CHUNK_DIR,
        "num_chunks": len(meta["chunk_files"]),
        "total_samples": meta["total_samples"],
        "num_cells": meta["num_cells"],
        "dtype_data": meta["dtype_data"],
        "dtype_labels": meta["dtype_labels"],
        "overall_frac_static(label=1)": overall_frac,
        "min_frac_static_across_chunks": float(np.min(meta["label_fractions_static"])),
        "max_frac_static_across_chunks": float(np.max(meta["label_fractions_static"])),
        "mean_frac_static_across_chunks_unweighted": float(np.mean(meta["label_fractions_static"]))
    }

    json_path = os.path.join(out_dir, "chunk_label_stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved per-chunk label fractions: {csv_path}")
    print(f"Saved label summary stats:       {json_path}")
    print(f"Overall frac static (label=1):   {overall_frac:.4f}")


def build_global_index_map(chunk_sizes: List[int]) -> np.ndarray:
    """
    Returns cumulative offsets, length = num_chunks+1, where:
      - chunk k corresponds to global indices [offsets[k], offsets[k+1])
    """
    offsets = np.zeros(len(chunk_sizes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.array(chunk_sizes, dtype=np.int64))
    return offsets


def chunk_row_from_global(global_indices: np.ndarray, offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given global indices, return:
      chunk_ids: which chunk each index belongs to
      row_ids:   row within that chunk
    """
    # chunk_id is the largest k such that offsets[k] <= idx
    chunk_ids = np.searchsorted(offsets, global_indices, side="right") - 1
    row_ids = global_indices - offsets[chunk_ids]
    return chunk_ids.astype(np.int64), row_ids.astype(np.int64)


def materialize_split_arrays(
    meta: Dict,
    offsets: np.ndarray,
    split_indices: np.ndarray,
    out_x_path: str,
    out_y_path: str
) -> None:
    """
    Load only the needed rows from chunks and write X/Y .npy files.
    """
    num_cells = meta["num_cells"]
    n = split_indices.shape[0]

    X = np.empty((n, num_cells), dtype=np.int8)
    y = np.empty((n,), dtype=np.int8)

    chunk_ids, row_ids = chunk_row_from_global(split_indices, offsets)

    # Group requested rows by chunk for efficient I/O
    order = np.argsort(chunk_ids)
    chunk_ids_sorted = chunk_ids[order]
    row_ids_sorted = row_ids[order]

    # We also need to map back to original positions
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)

    start = 0
    while start < n:
        c = int(chunk_ids_sorted[start])
        end = start
        while end < n and int(chunk_ids_sorted[end]) == c:
            end += 1

        rows = row_ids_sorted[start:end]
        chunk_path = meta["chunk_files"][c]

        with np.load(chunk_path) as z:
            data = z["data"]      # (chunk_size, num_cells)
            labels = z["labels"]  # (chunk_size,)
            X_block = data[rows]
            y_block = labels[rows]

        # Place back into X/y according to original split_indices order
        orig_positions = order[start:end]   # positions in X/y to fill
        X[orig_positions, :] = X_block.astype(np.int8)
        y[orig_positions] = y_block.astype(np.int8)

        print(f"  loaded chunk {c:04d}: {os.path.basename(chunk_path)}  rows={end-start}")
        start = end

    np.save(out_x_path, X)
    np.save(out_y_path, y)

    print(f"Saved X: {out_x_path}  shape={X.shape} dtype={X.dtype}")
    print(f"Saved y: {out_y_path}  shape={y.shape} dtype={y.dtype}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Find chunk files
    chunk_files = list_chunk_files(CHUNK_DIR)
    print(f"Found {len(chunk_files)} chunk files in {CHUNK_DIR}")

    # 2) Inspect chunks and compute per-chunk label fractions
    meta = inspect_chunks(chunk_files)
    print(f"Total samples across chunks: {meta['total_samples']}")
    print(f"num_cells: {meta['num_cells']} (expected 196 for 14x14)")

    save_label_diagnostics(meta, OUT_DIR)

    total = meta["total_samples"]
    if TEST_SIZE + VAL_SIZE >= total:
        raise ValueError(f"TEST_SIZE + VAL_SIZE = {TEST_SIZE + VAL_SIZE} >= total_samples = {total}")

    # 3) Build global index offsets (map global index -> (chunk,row))
    offsets = build_global_index_map(meta["chunk_sizes"])

    # 4) Create deterministic split indices
    all_indices = np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(RNG_SEED)

    if SHUFFLE:
        rng.shuffle(all_indices)

    test_indices = all_indices[:TEST_SIZE]
    val_indices = all_indices[TEST_SIZE:TEST_SIZE + VAL_SIZE]
    train_indices = all_indices[TEST_SIZE + VAL_SIZE:]

    # Save index files
    np.save(os.path.join(OUT_DIR, "test_indices.npy"), test_indices)
    np.save(os.path.join(OUT_DIR, "val_indices.npy"), val_indices)
    np.save(os.path.join(OUT_DIR, "train_indices.npy"), train_indices)

    # Save split metadata
    split_meta = {
        "chunk_dir": CHUNK_DIR,
        "out_dir": OUT_DIR,
        "rng_seed": RNG_SEED,
        "shuffle": SHUFFLE,
        "total_samples": total,
        "num_chunks": len(meta["chunk_files"]),
        "num_cells": meta["num_cells"],
        "test_size": int(TEST_SIZE),
        "val_size": int(VAL_SIZE),
        "train_size": int(train_indices.shape[0]),
        "chunk_files": [os.path.basename(p) for p in meta["chunk_files"]],
        "chunk_sizes": meta["chunk_sizes"],
    }
    with open(os.path.join(OUT_DIR, "split_metadata.json"), "w") as f:
        json.dump(split_meta, f, indent=2)

    print("Saved split indices + metadata.")

    # 5) Materialize frozen test/val arrays
    print("\nMaterializing TEST set...")
    materialize_split_arrays(
        meta=meta,
        offsets=offsets,
        split_indices=test_indices,
        out_x_path=os.path.join(OUT_DIR, "X_test.npy"),
        out_y_path=os.path.join(OUT_DIR, "y_test.npy")
    )

    print("\nMaterializing VAL set...")
    materialize_split_arrays(
        meta=meta,
        offsets=offsets,
        split_indices=val_indices,
        out_x_path=os.path.join(OUT_DIR, "X_val.npy"),
        out_y_path=os.path.join(OUT_DIR, "y_val.npy")
    )

    # Optional: save full dataset (rarely needed)
    if SAVE_FULL_DATASET:
        print("\nSAVE_FULL_DATASET=True: materializing full X/y...")
        materialize_split_arrays(
            meta=meta,
            offsets=offsets,
            split_indices=np.arange(total, dtype=np.int64),
            out_x_path=os.path.join(OUT_DIR, "X_all.npy"),
            out_y_path=os.path.join(OUT_DIR, "y_all.npy")
        )

    print("\nDone.")


if __name__ == "__main__":
    main()