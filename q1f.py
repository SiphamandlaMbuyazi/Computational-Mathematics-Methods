import numpy as np
from fpdf import FPDF


"""
To succefully run this script you have to do the following

install packages: numpy and fpdf(for writting onto the pdf)

to install all the above you have to have pip installed on your machine

then run `pip install numpy, fpdf`

after this you can then run the scripts no problem using python3 q1final.py

if there's a problem with this I have pre-ran the script and on this folder there's results folder which already contains the results

"""

def round_off_matrix(matrix, decimals):
    return np.round(matrix, decimals)

def jacobi_rotation(A, i, j):
    if A[i, i] == A[j, j]:  # Prevent division by zero
        theta = np.pi / 4
    else:
        theta = np.abs(0.5 * np.arctan(2 * A[i, j] / (A[i, i] - A[j, j])))

    R = np.identity(len(A))
    R[i, i] = np.cos(theta)
    R[j, j] = np.cos(theta)
    R[i, j] = np.sin(theta)
    R[j, i] = -np.sin(theta)

    return R

def jacobi_method(A, tolerance):
    n = len(A)
    eigenvectors = np.identity(n)
    set_of_orthogonals = []

    flag = True
    iteration_count = 0

    while flag:
        off_diagonal = np.tril(A, -1)
        off_diag_sum = np.sum(np.abs(off_diagonal))
        i, j = np.unravel_index(np.argmax(np.abs(off_diagonal)), A.shape)

        if off_diag_sum < tolerance:
            flag = False

        R = jacobi_rotation(A, i, j)
        set_of_orthogonals.append(R)
        A = R.T @ A @ R
        iteration_count += 1

    for ort in set_of_orthogonals:
        eigenvectors = eigenvectors @ ort

    eigenvalues = np.diag(A)

    return eigenvalues, eigenvectors

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Example Matrices
A1 = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 1]
],dtype=float)

A2 = np.array([
    [8, 4, 2, 1],
    [4, 8, 2, 1],
    [2, 2, 8, 1],
    [1, 1, 1, 8]
],dtype=float)

# Run Jacobi Method
P1, D1 = jacobi_method(A1, 1e-4)
P2, D2 = jacobi_method(A2, 1e-4)

# Write results to PDF
def write_matrix_to_pdf(pdf, title, matrix):
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, title, ln=True)
    if matrix.ndim == 1:  # Handle 1D array (eigenvalues)
        matrix_str = "  , ".join([str(val) for val in matrix])
    else:  # Handle 2D array (eigenvectors)
        matrix_str = "\n".join(["\t".join(map(str, row)) for row in matrix])
    pdf.multi_cell(0, 10, matrix_str)

pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.cell(0, 10, 'Jacobi Method Results', ln=True, align='C')

pdf.cell(0, 10, 'Results for Matrix A1 rounded off to 5 decimal places:', ln=True,align='C')
write_matrix_to_pdf(pdf, 'Eigenvalues:', round_off_matrix(round_off_matrix(P1,5), 10))
write_matrix_to_pdf(pdf, 'Eigenvectors:', round_off_matrix(round_off_matrix(D1,5), 10))



pdf.add_page()

pdf.cell(0, 10, 'Results for Matrix A2 rounded off to 5 decimal places:', ln=True,align='C')
write_matrix_to_pdf(pdf, 'Eigenvalues:', round_off_matrix(round_off_matrix(P2,5), 10))
write_matrix_to_pdf(pdf, 'Eigenvectors:', round_off_matrix(round_off_matrix(D2,5), 10))

# Save PDF
pdf.output("Question1 Results.pdf")

print(f"PDF generated successfully as Question1 Results.pdf")
