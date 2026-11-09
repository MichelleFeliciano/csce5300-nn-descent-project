"""
data_io.py
----------
Loads vector datasets stored in the simple .fbin binary format used
throughout this project:

    [int32 num_vectors] [int32 dimensions] [float32 * num_vectors * dimensions]

Both queries.fbin (real SIFT descriptors, from the paper's dataset family)
and synthetic.fbin (generated comparison dataset) use this same format,
so this one function loads either.
"""

import numpy as np


def load_fbin(path):
    """
    Load a .fbin file into a NumPy array.

    Parameters
    ----------
    path : str
        Path to the .fbin file.

    Returns
    -------
    vectors : np.ndarray, shape (num_vectors, dimensions), dtype float32
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"{path}: could not read header (file too short)")
        num_vectors, dimensions = int(header[0]), int(header[1])

        expected_values = num_vectors * dimensions
        data = np.fromfile(f, dtype=np.float32, count=expected_values)
        if data.size != expected_values:
            raise ValueError(
                f"{path}: expected {expected_values} float32 values "
                f"({num_vectors} x {dimensions}), but read {data.size}"
            )

    return data.reshape(num_vectors, dimensions)


if __name__ == "__main__":
    # Quick self-test / sanity check when run directly.
    import sys

    for test_path in sys.argv[1:]:
        vecs = load_fbin(test_path)
        print(f"{test_path}: shape={vecs.shape}, dtype={vecs.dtype}, "
              f"min={vecs.min():.3f}, max={vecs.max():.3f}, mean={vecs.mean():.3f}")
