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

# Apply QR method
eigenvalues, eigenvectors = qr_algorithm(T.copy())
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
