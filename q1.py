import numpy as np


def round_off_matrix(matrix,decimals):
    return np.round(matrix,decimals)

def jacobi_rotation(A, i, j):
    if A[i, i] == A[j, j]: # this is to prevent python from raising a devision by zero error
        theta = np.pi / 4
    else:
        theta = np.abs(0.5 * np.arctan(2 * A[i, j] / (A[i, i] - A[j, j])))
       # print(theta)
    
    R = np.identity(len(A))
    R[i, i] = np.cos(theta)
    R[j, j] = np.cos(theta)
    R[i, j] = np.sin(theta)
    R[j, i] = -1 * np.sin(theta)
    
    return R

def jacobi_method(A, tolerance):
    n = len(A)
    eigenvectors = np.identity(n)
    i=0
    set_of_orthogonals= []

    flag= True

    while flag:
        off_diagonal = np.tril(A, -1)
        off_diag_sum = np.sum(np.abs(off_diagonal))
        i, j = np.unravel_index(np.argmax(np.abs(off_diagonal)), A.shape)
        if off_diag_sum < tolerance:
            flag=False
        
        #update the matrix 
        
        R = jacobi_rotation(A, i, j)

        set_of_orthogonals.append(R)
        A = R.T @ A @ R

        # print(A,"\n")
        i+=1

    for ort in set_of_orthogonals:
        eigenvectors = eigenvectors@ort

    eigenvalues = np.diag(A)
    
    return eigenvalues, eigenvectors,i


A1= np.array([
    [1,1,1],
    [1,1,0],
    [1,0,1]
])

A2=np.array([
    [8,4,2,1],
    [4,8,2,1],
    [2,2,8,1],
    [1,1,1,8]
])

P1= round_off_matrix(jacobi_method(A1,1e-10)[0],10)

D1=round_off_matrix(jacobi_method(A1,1e-10)[1],10)


P2= round_off_matrix(jacobi_method(A2,1e-10)[0],10)

D2=round_off_matrix(jacobi_method(A2,1e-10)[1],10)
print(P1,"\n","\n",D1)


print(P2,"\n","\n",D2)





print(f"iters: {jacobi_method(A1,1e-10)[2]}")
















