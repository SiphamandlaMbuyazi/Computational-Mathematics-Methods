import numpy as np
from numpy import abs

def round_off_matrix(matrix,decimals):
    return np.round(matrix,decimals)

def householder_transformation_with_P(A):
    n = A.shape[0]
    C = np.copy(A)
    P = np.eye(n)

    for k in range(n-2):
        # Select the vector x and form the Householder vector v
        x = C[k+1:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * (1 if x[0] == 0 else np.sign(x[0]))
        v = x + e1
        v /= np.linalg.norm(v)

        # Householder matrix for this step
        H_k = np.eye(n-k-1) - 2 * np.outer(v, v)
        
        # Update C by applying H_k to both sides
        C[k+1:, k:] = np.dot(H_k, C[k+1:, k:])
        C[:, k+1:] = np.dot(C[:, k+1:], H_k.T)
        
        # Build the overall transformation matrix P
        H_full = np.eye(n)
        H_full[k+1:, k+1:] = H_k
        P = np.dot(P, H_full)

    # Zero out small values to ensure tridiagonal form
    C[np.abs(C) < 1e-10] = 0

    return C, P

# A=np.array([
#     [7,2,3,-1],
#     [2,8,5,1],
#     [3,5,12,9],

#     [-1,1,9,7]
# ],dtype=float)

A=np.array([
   [3,2,1, 2],
[2,-1, 1, 2],
[1, 1, 4, 3],
[2 ,2 ,3, 1]
],dtype=float)




#question 3A,

C=householder_transformation_with_P(A)[0]
print(C,"\n")
print("Question 3B \n")



P=householder_transformation_with_P(A)[1]
print(P)


# newA=P@C@P.T


# print('\n',newA)