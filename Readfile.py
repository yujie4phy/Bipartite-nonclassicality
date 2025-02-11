import os
import numpy as np

# Define the root directory
ROOT_DIR = "Data"


def load_file(file_type, file_name):
    """
    Reads a specific .npy file from Counts or Probabilities.

    :param file_type: "Counts" or "Probabilities"
    :param file_name: Name of the file (e.g., "counts_048.npy")
    :return: NumPy array if the file is found, otherwise None.
    """
    if file_type not in ["Counts", "Probabilities"]:
        print(f"❌ Invalid file type: {file_type}. Choose 'Counts' or 'Probabilities'.")
        return None

    folder_path = os.path.join(ROOT_DIR, file_type)
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        data = np.load(file_path)
        print(f"✅ Loaded {file_name} from {file_type}: {data.shape}")
        return data
    else:
        print(f"❌ File not found: {file_path}")
        return None


def swap1(b):
    """Swap specific row pairs in the array."""
    row_swaps = [(0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10)]
    for row1, row2 in row_swaps:
        b[[row1, row2]] = b[[row2, row1]]
    return b


def swap2(b):
    """Swap specific column pairs in the array."""
    column_swaps = [(0, 18), (1, 15), (2, 16), (3, 11), (4, 10),
                    (5, 14), (6, 12), (7, 17), (8, 19), (9, 13)]
    for col1, col2 in column_swaps:
        b[:, [col1, col2]] = b[:, [col2, col1]]
    return b


def process_data(file_type, file_name):
    """
    Loads the file, applies swap operations, and computes the final processed array P.

    :param file_type: "Counts" or "Probabilities"
    :param file_name: Name of the .npy file to process
    :return: Processed NumPy array P
    """
    data = load_file(file_type, file_name)
    if data is None:
        return None  # Return None if file not found

    # Applying transformations
    t1 = data[0]
    t2 = swap2(swap1(data[1]))
    t3 = swap1(data[2])
    t4 = swap2(data[3])

    # Compute final result
    P = (t1 + t2 + t3 + t4) / 4
    return P