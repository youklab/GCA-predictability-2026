import os
import json
import numpy as np

SPLIT_DIR = "ml_splits_v251209"
CHUNK_DIR = "labeled_data_2025_parallel"

OUT_X = os.path.join(SPLIT_DIR, "X_train_pool.npy")
OUT_Y = os.path.join(SPLIT_DIR, "y_train_pool.npy")


def main():
    meta_path = os.path.join(SPLIT_DIR, "split_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    total = int(meta["total_samples"])
    test_size = int(meta["test_size"])
    val_size = int(meta["val_size"])
    train_size = int(meta["train_size"])
    seed = int(meta["rng_seed"])
    shuffle = bool(meta["shuffle"])

    chunk_files = meta["chunk_files"]
    chunk_sizes = meta["chunk_sizes"]
    num_chunks = int(meta["num_chunks"])
    num_cells = int(meta["num_cells"])

    assert total == sum(chunk_sizes), "chunk_sizes do not sum to total_samples"
    assert test_size + val_size + train_size == total, "split sizes do not sum to total"

    print("Reconstructing split from metadata:")
    print(f"  total={total}  test={test_size}  val={val_size}  train={train_size}")
    print(f"  seed={seed}  shuffle={shuffle}")
    print(f"  num_chunks={num_chunks}  num_cells={num_cells}")

    # Global indices 0..total-1
    idx = np.arange(total, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    test_idx = idx[:test_size]
    val_idx = idx[test_size:test_size + val_size]

    # Boolean mask over GLOBAL indices, True iff in train pool
    is_train = np.ones(total, dtype=bool)
    is_train[test_idx] = False
    is_train[val_idx] = False

    os.makedirs(SPLIT_DIR, exist_ok=True)

# Use open_memmap to write a valid .npy file that can be read back with np.load(mmap_mode="r").
    X = np.lib.format.open_memmap(OUT_X, mode="w+", dtype=np.int8, shape=(train_size, num_cells))
    y = np.lib.format.open_memmap(OUT_Y, mode="w+", dtype=np.int8, shape=(train_size,))

    cursor = 0
    start = 0
    for chunk_id in range(num_chunks):
        n = int(chunk_sizes[chunk_id])
        end = start + n

        chunk_mask = is_train[start:end]
        k = int(chunk_mask.sum())

        if k > 0:
            chunk_path = os.path.join(CHUNK_DIR, chunk_files[chunk_id])
            with np.load(chunk_path, mmap_mode="r") as z:
                X_chunk = z["data"][chunk_mask]
                y_chunk = z["labels"][chunk_mask]

            X[cursor:cursor + k, :] = X_chunk
            y[cursor:cursor + k] = y_chunk
            cursor += k

        if chunk_id % 10 == 0:
            print(f"  chunk {chunk_id:04d}: kept {k:5d} rows   wrote {cursor}/{train_size}")

        start = end

    X.flush()
    y.flush()

    if cursor != train_size:
        raise RuntimeError(f"Expected to write {train_size} rows but wrote {cursor} rows")

    print("Done.")
    print("Wrote:", OUT_X)
    print("Wrote:", OUT_Y)


if __name__ == "__main__":
    main()