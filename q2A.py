import numpy as np

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
       # print(off_diag_sum)
        if off_diag_sum <tolerance:
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

# Example matrix A
A = np.array([
    [1.59,1.69,2.13],
    [1.69,1.31,1.72],
    [2.13,1.72,1.85]
])


print(f"\n {jacobi_method(A,1e-10)[0]} \n \n { jacobi_method(A,1e-10)[1]} \n \n { jacobi_method(A,1e-10)[2]}")


