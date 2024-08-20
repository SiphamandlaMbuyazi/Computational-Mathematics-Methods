import numpy as np


"""
README!
To succesfully run this script you have to do the following

install packages: numpy and fpdf(for writting onto the pdf)

to install all the above you have to have pip installed on your machine

then run `pip install numpy, fpdf`

after this you can then run the scripts no problem using python3 q3final.py

if there's a problem with this I have pre-ran the script and on this folder there's results folder which already contains the results

"""


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
    [7,2,3,-1],
    [2,8,5,1],
    [3,5,12,9],
    [-1,1,9,7]    

],dtype=float)



from fpdf import FPDF

# Assuming the householder_transformation_with_P function is already defined and imported
# Example matrix A


# Initialize PDF
pdf = FPDF()
pdf.add_page()

# Set font
pdf.set_font("Arial", size=12)

# Part (a) - Tridiagonalize Matrix A using Householder’s Method
C = householder_transformation_with_P(A)[0]

pdf.multi_cell(0, 10, "a. Tridiagonalize matrix A using Householder method.")
pdf.multi_cell(0, 10, "   The resulting tridiagonal matrix C is:")
for row in C:
    pdf.multi_cell(0, 10, f"   {row}")

# Part (b) - Transformation Matrix P
P = householder_transformation_with_P(A)[1]

pdf.ln(10)
pdf.multi_cell(0, 10, "b. The transformation matrix P:")
for row in P:
    pdf.multi_cell(0, 10, f"   {row}")

# Save the PDF
pdf.output("Question3_results.pdf")

print("PDF created successfully as Question3_results.pdf.")





#question 3A,

# C=householder_transformation_with_P(A)[0]

# print("Question 3A")
# print(C,"\n")
# print("Question 3B \n")



# P=householder_transformation_with_P(A)[1]
# print(P)


# newA=P@C@P.T

# print(P.T@A@P)
# print('\n',newA)