import numpy as np

"""
README!
To succesfully run this script you have to do the following

install packages: numpy and fpdf(for writting onto the pdf)

to install all the above you have to have pip installed on your machine

then run `pip install numpy, fpdf`

after this you can then run the scripts no problem using python3 q4final.py

if there's a problem with this I have pre-ran the script and on this folder there's results folder which already contains the results

"""
def round_off_matrix(matrix,decimals):
    return np.round(matrix,decimals)


def givens_rotation(a, b):
    r = np.hypot(a, b)
    c = a / r
    s = -b / r
    return c, s

def apply_givens_rotation(A, i, j, c, s, n):
    for k in range(n):
        temp = c * A[i, k] - s * A[j, k]
        A[j, k] = s * A[i, k] + c * A[j, k]
        A[i, k] = temp
        
    for k in range(n):
        temp = c * A[k, i] - s * A[k, j]
        A[k, j] = s * A[k, i] + c * A[k, j]
        A[k, i] = temp

def tridiagonalize(A):
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



def qr_algorithm(T, tol=1e-10):
    n = T.shape[0]
    Q_total = np.eye(n)
    for i in range(1000):
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

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()

# Set font
pdf.set_font("Arial", size=12)

# Part (a) - Tridiagonal Matrix T
pdf.multi_cell(0, 10, "4 a.")
pdf.multi_cell(0, 10, "   The resulting tridiagonal matrix T is:")
for row in T:
    pdf.multi_cell(0, 10, f"   {row}")

# Part (b) - Eigenvalues and Eigenvectors
pdf.ln(10)
pdf.multi_cell(0, 10, "4 b.")
pdf.multi_cell(0, 10, "   Eigenvalues:")
pdf.multi_cell(0, 10, f"   {eigenvalues}")
pdf.multi_cell(0, 10, "   Eigenvectors:")
for vec in eigenvectors:
    pdf.multi_cell(0, 10, f"   {vec}")

# Save the PDF as Question4_results.pdf
pdf.output("Question4_results.pdf")

print("PDF created successfully as Question4_results.pdf.")




















# eigenvalues, eigenvectors = qr_algorithm(T.copy())
# print("\nEigenvalues:")
# print(eigenvalues)
# print("\nEigenvectors:")
# print(eigenvectors)
