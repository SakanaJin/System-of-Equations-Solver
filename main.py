import copy

# I'm not error checking this bc it's for testing so don't be a retard
def matmul(A, B):
    if type(B[0]) == list:
        return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
    return [sum(A[i][j]*B[j] for j in range(len(B))) for i in range(len(A))]

def apply_perm(Pvec: list[int], b: list[float]) -> list[float]:
    return [b[Pvec[i]] for i in range(len(b))]

def build_P_from_Pvec(Pvec: list[int]) -> list[list[int]]:
    n = len(Pvec)
    P = [[0]*n]*n
    for i, j in enumerate(Pvec):
        P[i][j] = 1
    return P

def PvecLU(A: list[list[float]]) -> tuple[list[int], list[list[float]], list[list[float]]]:
    n: int = len(A)
    m: int = len(A[0])
    if any(m != len(row) for row in A):
        raise ValueError("All rows must have same number of columns")
    alpha = copy.deepcopy(A)
    Pvec = list(range(n))
    row = 0
    for pivot in range(min(n, m)):
        row = pivot
        while row < n and abs(alpha[row][pivot]) < 1e-12:
            row += 1
        if row == n:
            raise ValueError("No valid pivot point in column {}".format(pivot))
        if row != pivot:
            alpha[pivot], alpha[row] = alpha[row], alpha[pivot]
            Pvec[pivot], Pvec[row] = Pvec[row], Pvec[pivot]
        for i in range(pivot+1, n):
            alpha[i][pivot] /= alpha[pivot][pivot]
            alpha[i][pivot+1:] = [alpha[i][j] - alpha[i][pivot] * alpha[pivot][j] for j in range(pivot+1, m)]
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(min(n, m)):
        for j in range(i, m):
            U[i][j] = alpha[i][j]
    for i in range(n):
        L[i][i] = 1.0
        for j in range(i+1, n):
            L[j][i] = alpha[j][i]
    return (Pvec, L, U)

def eqsolve(A: list[list[float]], b: list[float]) -> list[float]:
    '''A is the coefficient matrix and b is the solution vector'''
    n = len(A)
    if n != len(A[0]) or any(n != len(row) for row in A):
        raise ValueError("Matrix A should be square")
    m = len(b)
    if m != n:
        raise ValueError("Vector b should be same length as number of rows in A")
    pvec, L, U = PvecLU(A)
    b = apply_perm(pvec, b)
    # forward sub
    for i in range(1, n):
        for j in range(0, i):
            L[i][j] = L[i][j] * b[j]
        b[i] = b[i] - sum(L[i][0:i])
    # backward sub
    b[-1] = b[-1] / U[-1][-1]
    for i in range(n-2, -1, -1):
        for j in range(n-1, i, -1):
            U[i][j] = U[i][j] * b[j]
        b[i] = (b[i] - sum(U[i][i+1:n]))/U[i][i]
    b = apply_perm(pvec, b)
    return b