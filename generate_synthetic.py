import numpy as np
import scipy.io


def generate_random_subspace_matrix(m, d):
    # Generate a random d x m matrix with entries sampled from a normal distribution
    random_matrix = np.random.randn(d, m)

    # Perform QR decomposition
    q, r = np.linalg.qr(random_matrix, mode='reduced')

    # Transpose q to obtain a m x d matrix
    subspace_matrix = q.T

    return subspace_matrix


if __name__ == "__main__":
    n = 5*(10**5)
    m = 10
    d = 300
    S = np.random.randn(n, m)
    D = np.diag([1 - (i-1)/m for i in range(m)])
    U = generate_random_subspace_matrix(m, d)

    N = np.random.randn(n, d)
    zeta = 10

    A = S@D@U + N/zeta

    mdic = {"n": n, "m": m, "d": d, "zeta": zeta, "A": A}
    file_path = f"dataset/synthetic_n={n},m={m},d={d},zeta={zeta}.mat"
    scipy.io.savemat(file_path, mdic)
