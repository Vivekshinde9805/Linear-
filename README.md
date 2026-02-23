# Linear Algebra Programs

Python program on invertibility of elementary matrices.
This repository contains linear algebra related programs.
Module 1
1 System of linear equations with examples. 


Ex1 
import numpy as np

A = np.array([[1, 1],
              [1, -1]])

b = np.array([0, 0])

solution = np.linalg.solve(A, b)

print("Solution:")
print(solution)








EX2
import numpy as np

A = np.array([[1, 1, 1],
              [2, 1, -1],
              [1, -1, 1]])

b = np.array([0, 0, 0])

solution = np.linalg.solve(A, b)

print("Trivial solution:")
print(solution)










2.Matrix representation of a system of linear equations.

Matrix Representation in Python (Using NumPy)
import numpy as np

# Coefficient matrix
A = np.array([[2, 3],
              [1, -1]])

# Variable matrix (for representation)
X = np.array(['x', 'y'])

# Constant matrix
B = np.array([8, 2])

print("Coefficient Matrix A:")
print(A)

print("\nVariable Matrix X:")
print(X)

print("\nConstant Matrix B:")
print(B)






3.Matrix representation of a system of linear equations.
import numpy as np
 # Example system of linear equations:
# 2x + 3y - z = 1
# 4x + y + 2z = 2
# 3x + 4y - 5z = 3
 print("Example System of Linear Equations:")
print("2x + 3y - z = 1")
print("4x + y + 2z = 2")
print("3x + 4y - 5z = 3\n")
 # Matrix representation: AX = B
# A = coefficient matrix, X = variable vector, B = constant vector
 A = np.array([[2, 3, -1],
              [4, 1, 2],
              [3, 4, -5]], dtype=float)
 
X = np.array([['x'], ['y'], ['z']])  # Variable symbols
B = np.array([1, 2, 3], dtype=float)
 print("Matrix Representation (AX = B):")
print("A (Coefficient Matrix):")
print(A)
 print("\nX (Variable Vector):")
print(X)
 print("\nB (Constant Vector):")
print(B)
 print("\nAugmented Matrix [A|B]:")
print(np.c_[A, B])
 
# Solve the system
solution = np.linalg.solve(A, B)
print(f"\nSolution: x = {solution[0]:.2f}, y = {solution[1]:.2f}, z = {solution[2]:.2f}")
 














Practical 4

The geometry of solutions of a system of linear equations. 
import numpy as np
import matplotlib.pyplot as plt

# Define the coefficients of the system of equations:
# Example system:
# 2x + y = 5
# x - y = 1

# Coefficients matrix A
A = np.array([[2, 1],
              [1, -1]])

# Constants vector B
B = np.array([5, 1])

# Solve the linear system
solution = np.linalg.solve(A, B)
x_sol, y_sol = solution

print(f"Solution: x = {x_sol}, y = {y_sol}")

# Plotting the lines to show geometry of solution
x = np.linspace(-1, 5, 400)

# Lines from the system
y1 = (5 - 2*x)  # From 2x + y = 5  => y = 5 - 2x
y2 = x - 1      # From x - y = 1 => y = x - 1

plt.plot(x, y1, label="2x + y = 5")
plt.plot(x, y2, label="x - y = 1")

# Plot the solution point
plt.plot(x_sol, y_sol, 'ro', label="Solution")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Geometry of Solutions of Linear Equations')
plt.legend()
plt.grid(True)
plt.show()

output:

Solution: x = 2.0, y = 1.0















5 Simple Python Program: Determinant and Rank of a Matrix
import numpy as np

# Example Matrix
A = np.array([[2, 3, 1],
              [4, 1, -1],
              [1, -2, 5]], dtype=float)

print("Matrix A:")
print(A)

# Determinant
detA = np.linalg.det(A)
print("\nDeterminant of A =", detA)

# Rank
rankA = np.linalg.matrix_rank(A)
print("Rank of A =", rankA)







Practical 6
Python program for solving a system of linear equations using determinants.
import numpy as np

def solve_using_determinants(A, B):
    """
    Solves the system A·x = B using Cramer's Rule.
    A: coefficient matrix (n×n)
    B: constants vector (n)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    det_A = np.linalg.det(A)

    if det_A == 0:
        raise ValueError("Determinant of A is zero. System has no unique solution.")

    n = A.shape[0]
    solutions = []

    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = B  # Replace column i with constants vector
        det_Ai = np.linalg.det(Ai)
        solutions.append(det_Ai / det_A)

    return solutions


# Example: Solve the system
# 2x + 3y - z = 5
# 4x + y  + 2z = 6
# -2x + 5y + 3z = 1

A = [
    [2, 3, -1],
    [4, 1,  2],
    [-2, 5, 3]
    ]

B = [5, 6, 1]

solution = solve_using_determinants(A, B)

print("Solution:")
for i, val in enumerate(solution):
    print(f"x{i+1} = {val}")













7.python program on Elementary matrices and their relations with elementary operations on  matrices.
import numpy as np

# Original matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], float)

print("Original Matrix A:\n", A, "\n")

# --- Elementary Matrix 1: Swap R1 and R2 ---
E1 = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
], float)

print("E1 (Swap R1 and R2):\n", E1)
print("E1 * A:\n", E1 @ A, "\n")

# --- Elementary Matrix 2: Multiply R2 by 3 ---
E2 = np.array([
    [1, 0, 0],
    [0, 3, 0],
    [0, 0, 1]
], float)

print("E2 (Multiply R2 by 3):\n", E2)
print("E2 * A:\n", E2 @ A, "\n")

# --- Elementary Matrix 3: R3 → R3 - 2R1 ---
E3 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [-2, 0, 1]
], float)

print("E3 (R3 = R3 - 2R1):\n", E3)
print("E3 * A:\n", E3 @ A, "\n")














8.Python Program on Invertibility of Elementary Matrices
import numpy as np

# ------- Elementary Matrices -------

# 1. Row swap R1 ↔ R2
E1 = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
], float)

# 2. Multiply R2 by 3
E2 = np.array([
    [1, 0, 0],
    [0, 3, 0],
    [0, 0, 1]
], float)

# 3. Add 2 times R1 to R3  →  R3 = R3 + 2R1
E3 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [2, 0, 1]
], float)


# ------- Check invertibility -------

print("E1:\n", E1)
print("Inverse of E1:\n", np.linalg.inv(E1), "\n")

print("E2:\n", E2)
print("Inverse of E2:\n", np.linalg.inv(E2), "\n")

print("E3:\n", E3)
print("Inverse of E3:\n", np.linalg.inv(E3), "\n")










Module 2

Practical 1
simple python program for Linear transformations and their elementary properties.

import numpy as np

# Define a linear transformation T(x) = A x
A = np.array([[2, 1],
              [0, 3]])

def T(x):
    return A @ x   # Matrix multiplication

# Define vectors
u = np.array([1, 2])
v = np.array([3, 1])
c = 2

# Apply transformation
T_u = T(u)
T_v = T(v)

# Check additivity
left_add = T(u + v)
right_add = T_u + T_v

# Check homogeneity
left_hom = T(c * u)
right_hom = c * T_u

print("T(u) =", T_u)
print("T(v) =", T_v)

print("\nAdditivity check:")
print("T(u + v) =", left_add)
print("T(u) + T(v) =", right_add)

print("\nHomogeneity check:")
print("T(cu) =", left_hom)
print("cT(u) =", right_hom)







Practical 2
python program for Composite of linear transformations.

# First linear transformation
def T1(v):
    x, y = v
    return [2*x, 2*y]

# Second linear transformation
def T2(v):
    x, y = v
    return [x + y, x - y]

# Composite transformation T = T2 ∘ T1
def T_composite(v):
    return T2(T1(v))

# Vector
v = [1, 3]

# Results
print("Vector v =", v)
print("T1(v) =", T1(v))
print("T2(T1(v)) =", T_composite(v))







Practical 3
python program for Rank and nullity of a linear transformation with verification of rank-nullity 
theorem.

import numpy as np

# Define transformation matrix A
A = np.array([[1, 2, 3],
              [2, 4, 6]])

print("Matrix A:\n", A)

# Number of columns (dimension of domain)
n = A.shape[1]

# Rank of A
rank = np.linalg.matrix_rank(A)

# Nullity of A
nullity = n - rank

# Display results
print("\nRank of T =", rank)
print("Nullity of T =", nullity)

# Verify Rank–Nullity Theorem
print("\nVerification:")
print("Rank + Nullity =", rank + nullity)
print("Dimension of domain =", n)

if rank + nullity == n:
    print("Rank–Nullity Theorem Verified")
else:
    print("Rank–Nullity Theorem Not Verified")


Practical 4
Determining linear transformation by knowing its action on basis vectors.
 
# Action of T on basis vectors
Te1 = [2, 1]   # T(1, 0)
Te2 = [1, 3]   # T(0, 1)
 
# Define linear transformation using basis action
def T(v):
    x, y = v
    return [
        x * Te1[0] + y * Te2[0],
        x * Te1[1] + y * Te2[1]
    ]
 
# Vector to transform
v = [3, 2]
 
# Result
print("Vector v =", v)
print("T(v) =", T(v))
 

 
 
Practical 5
python program for Null space and image space of linear transformation.
 
 
import numpy as np
 
# Define transformation matrix A
A = np.array([[1, 2],
              [2, 4]])
 
print("Matrix A:\n", A)
 
# ---- Image Space (Column Space) ----
# Columns of A span the image space
image_space = A[:, 0]  # first column (others are dependent)
 
print("\nImage space is spanned by:")
print(image_space)
 
# ---- Null Space ----
# Solve A x = 0 using SVD
u, s, vh = np.linalg.svd(A)
tolerance = 1e-10
null_space = vh[s <= tolerance]
 
print("\nNull space vectors:")
print(null_space.T)
 
 
 
 
Practical 6
Python program for Computing matrix associated with a linear transformation.
 
 
import numpy as np
 
# Define the linear transformation
def T(v):
    x, y = v
    return np.array([2*x + y, x - y])
 
# Standard basis vectors in R^2
e1 = np.array([1, 0])
e2 = np.array([0, 1])
 
# Apply T to basis vectors
T_e1 = T(e1)
T_e2 = T(e2)
 
# Form the transformation matrix
A = np.column_stack((T_e1, T_e2))
 
print("Matrix associated with the linear transformation:")
print(A)
 
 
 
 
Practical 7
python program for Matrix associated with a composite of two linear transformations.
 
import numpy as np
 
# Matrix of T1: R^2 -> R^2
A = np.array([[1, 2],
              [3, 4]])
 
# Matrix of T2: R^2 -> R^2
B = np.array([[2, 0],
              [1, 5]])
 
# Matrix of composite transformation T = T2 ∘ T1
C = B @ A   # Matrix multiplication
 
print("Matrix of T1:")
print(A)
 
print("\nMatrix of T2:")
print(B)
 
print("\nMatrix of composite transformation (T2 ∘ T1):")
print(C)
 
 
 
Practical 8
 
python program for Effect on change of basis on linear transformation.
 
import numpy as np
 
# Matrix of linear transformation in old basis
A = np.array([[2, 1],
              [1, 3]])
 
# Change-of-basis matrix (new basis vectors as columns)
P = np.array([[1, 1],
              [1, -1]])
 
# Compute inverse of P
P_inv = np.linalg.inv(P)
 
# Matrix in new basis
A_new = P_inv @ A @ P
 
print("Matrix in old basis (A):")
print(A)
 
print("\nChange-of-basis matrix (P):")
print(P)
 
print("\nMatrix in new basis:")
print(A_new)
 







