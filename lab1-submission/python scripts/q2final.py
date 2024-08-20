import numpy as np


"""
README!
To succesfully run this script you have to do the following

install packages: numpy and fpdf(for writting onto the pdf)

to install all the above packages you have to have pip installed on your machine

then run `pip install numpy, fpdf`

after this you can then run the scripts no problem using python3 q2final.py

if there's a problem with this I have pre-ran the script and on this folder there's results folder which already contains the results

"""




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

        i+=1

    for ort in set_of_orthogonals:
        eigenvectors = eigenvectors@ort

    eigenvalues = np.diag(A)
    
    return eigenvalues, eigenvectors



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


def qr_algorithm(B, tol=1e-10):
    n = B.shape[0]
    B_k = np.copy(B)

    for _ in range(1000):
        # Perform QR decomposition
        Q, R = np.linalg.qr(B_k)
        B_k = np.dot(R, Q)
        off_diagonal = np.sum(np.abs(np.diag(B_k, k=-1)))
        if off_diagonal < tol:
            break

    # The eigenvalues are the diagonal elements of the resulting matrix
    eigenvalues = np.diag(B_k)
    
    return eigenvalues





# Example matrix A
A = np.array([
    [1.59,1.69,2.13],
    [1.69,1.31,1.72],
    [2.13,1.72,1.85]
])




#getting the results on the pdf now, to read more about this package fpdf, go to: `https://pypi.org/project/fpdf/`

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()

# Set font
pdf.set_font("Arial", size=12)

# Part (a) - Jacobi Method
jacobi_eigenvalues, jacobi_eigenvectors = jacobi_method(A, 1e-6)

pdf.multi_cell(0, 10, "a. Using the Jacobi method with a tolerance of epsilon = 10^-6,")
pdf.multi_cell(0, 10, f"   the eigenvalues of matrix A are: {jacobi_eigenvalues}")
pdf.multi_cell(0, 10, "   The corresponding eigenvectors are:")
for vec in jacobi_eigenvectors:
    pdf.multi_cell(0, 10, f"   {vec}")

# Part (b) - Householder Method
B = round_off_matrix(householder_transformation(A), 5)

pdf.ln(10)
pdf.multi_cell(0, 10, "b. The matrix A is reduced to a tridiagonal matrix B using the Householder method.")
pdf.multi_cell(0, 10, "   The resulting tridiagonal matrix B is:")
for row in B:
    pdf.multi_cell(0, 10, f"   {row}")

# Part (c) - QR Method
eigenvalues = qr_algorithm(B)
pdf.ln(10)
pdf.multi_cell(0, 10, "c. The eigenvalues of the tridiagonal matrix B obtained in (2b),")
pdf.multi_cell(0, 10, f"   using the QR method, are: {eigenvalues}")

# Save the PDF
pdf.output("Question2_results.pdf")

print("PDF created successfully as Question2_results.pdf.")
