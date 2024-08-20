import numpy as np


def round_off_matrix(matrix,decimals):
    return np.round(matrix,decimals)


def givens_rotation(a, b):
    """ Calculate the Givens rotation matrix for elements a and b. """
    r = np.hypot(a, b)
    c = a / r
    s = -b / r
    return c, s

def apply_givens_rotation(A, i, j, c, s, n):
    """ Apply the Givens rotation to matrix A. """
    for k in range(n):
        temp = c * A[i, k] - s * A[j, k]
        A[j, k] = s * A[i, k] + c * A[j, k]
        A[i, k] = temp
        
    for k in range(n):
        temp = c * A[k, i] - s * A[k, j]
        A[k, j] = s * A[k, i] + c * A[k, j]
        A[k, i] = temp

def tridiagonalize(A):
    """ Perform tridiagonalization using Givens rotations. """
    n = A.shape[0]
    for i in range(n-2):
        for j in range(i+2, n):
            if A[j, i] != 0:
                c, s = givens_rotation(A[i+1, i], A[j, i])
                apply_givens_rotation(A, i+1, j, c, s, n)
    return A

# Define the matrix A
A = np.array([[7, 2, 3, -1],
              [2, 8, 5, 1],
              [3, 5, 12, 9],
              [-1, 1, 9, 7]], dtype=float)

# Perform tridiagonalization
T = round_off_matrix(tridiagonalize(A.copy()),15)



def qr_algorithm(T, max_iter=1000, tol=1e-10):
    n = T.shape[0]
    Q_total = np.eye(n)
    for i in range(max_iter):
        Q, R = np.linalg.qr(T)
        T = R @ Q
        Q_total = Q_total @ Q
        # Check for convergence
        if np.allclose(T - np.diag(np.diagonal(T)), 0, atol=tol):
            break
    eigenvalues = np.diag(T)
    return eigenvalues, Q_total






















print("Tridiagonal matrix T: \n")
print(T)


eigenvalues, eigenvectors = qr_algorithm(T.copy())
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
