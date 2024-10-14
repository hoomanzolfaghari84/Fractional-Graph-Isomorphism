



import numpy as np


def pad_matrix_with_max(matrix, constant):
    # Get the maximum value from each row and column
    row_max = np.max(matrix, axis=1)
    col_max = np.max(matrix, axis=0)
    
    # Multiply the max values by the constant
    row_pad = (2 - row_max) * constant
    col_pad = (2 - col_max) * constant
    
    # Add a column to the right
    matrix_padded = np.column_stack((matrix, row_pad))
    
    # Add a row to the bottom
    col_pad_with_extra = np.append(col_pad, 0)  # Extra bottom-right element can be zero
    matrix_padded = np.row_stack((matrix_padded, col_pad_with_extra))
    
    return matrix_padded

def rbf_kernel(matrix, gamma=1.0, normalize=False):
    """
    Apply normalized Radial Basis Function (RBF) to each element of a 2D matrix.

    Parameters:
    matrix (np.ndarray): 2D array representing distances.
    gamma (float): Parameter for RBF kernel. Default is 1.0.

    Returns:
    np.ndarray: 2D array with RBF applied and normalized.
    """
    # Normalize the matrix by dividing by the maximum distance
    
    if normalize:
        max_distance = np.max(matrix)
        # Avoid division by zero in case max_distance is zero
        if max_distance != 0:
            matrix = matrix / max_distance
        
        
    # Apply the RBF function: exp(-gamma * (normalized_distance)^2)
    rbf_matrix = np.exp(-gamma * matrix**2)
    
    return rbf_matrix

def cosine_similarity_matrix(A, B):
    # Normalize each row of A and B (vector norm along axis 1)
    A_normalized = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_normalized = B / np.linalg.norm(B, axis=1, keepdims=True)
    
    # Compute the cosine similarity matrix C (dot product of A_normalized and B_normalized.T)
    C = np.dot(A_normalized, B_normalized.T)
    
    return C