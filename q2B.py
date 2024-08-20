import numpy as np
from numpy import abs

def round_off_matrix(matrix,decimals):
    return np.round(matrix,decimals)



def householder_transformation(A):
    n = A.shape[0]
    B = np.copy(A)  # Initialize B as a copy of A

    for k in range(n-2):
        # Select the vector x and form the Householder vector v
        x = B[k+1:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (1 if x[0] == 0 else np.sign(x[0]))
        w = x + e1
        w /= np.linalg.norm(w)

        # Apply the Householder transformation
        H = np.eye(n-k-1) - 2 * np.outer(w, w)
        B[k+1:, k:] = np.dot(H, B[k+1:, k:])
        B[:, k+1:] = np.dot(B[:, k+1:], H.T)

    # Zero out small values to ensure tridiagonal form
    B[np.abs(B) < 1e-20] = 0

    return B

# Example usage


def qr_algorithm(B, max_iterations=1000, tol=1e-10):
    """
    Perform the QR algorithm to find the eigenvalues of a tridiagonal matrix B.

    Args:
        B (numpy.ndarray): The tridiagonal matrix B.
        max_iterations (int): Maximum number of iterations to perform.
        tol (float): Tolerance level for convergence.

    Returns:
        numpy.ndarray: The eigenvalues of matrix B.
    """
    n = B.shape[0]
    B_k = np.copy(B)

    for _ in range(max_iterations):
        # Perform QR decomposition
        Q, R = np.linalg.qr(B_k)
        B_k = np.dot(R, Q)
        
        # Check for convergence by examining the off-diagonal elements
        off_diagonal = np.sum(np.abs(np.diag(B_k, k=-1)))
        if off_diagonal < tol:
            break

    # The eigenvalues are the diagonal elements of the resulting matrix
    eigenvalues = np.diag(B_k)
    
    return eigenvalues




A = np.array([
    [2,1,0],
    [1,3,1],
    [0,1,4]
], dtype=float)

B = round_off_matrix(householder_transformation(A),5)

eigenvalues=qr_algorithm(A)

print(eigenvalues)

