import os
import numpy as np

# Define the root directory


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

def process_data(data):
    """
    Loads the file from "Counts", applies swap operations, and computes the final processed array P.

    :param file_name: Name of the .npy file to process
    :return: Processed NumPy array P
    """
    # Applying transformations
    t1 = data
    t2 = swap2(swap1(data.copy()))
    t3 = swap1(data.copy())
    t4 = swap2(data.copy())

    # Compute final result
    tt = (t1 + t2 + t3 + t4)
    P=t1/tt
    Pu=np.sqrt(P)*np.sqrt(1-P)/np.sqrt(tt)
    return P,Pu