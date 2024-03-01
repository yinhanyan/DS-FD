import pandas as pd
from scipy.stats import poisson
import scipy
import numpy as np
from itertools import accumulate
import scipy.io

if __name__ == "__main__":
    mu = 0.5
    deltaN = 50000

    df = pd.read_csv("dataset/YearPredictionMSD.txt", delim_whitespace=False)
    A = df.values.astype(np.float64)
    # A = A[-51630:, 1:]
    A = A.astype(np.float64)

    n, d = A.shape

    dt = poisson.rvs(mu, size=n)
    t = list(accumulate(dt))

    i, j = 0, 0
    max_row = 0
    total_row = 0
    total_count = 0
    while j < len(t):
        if t[j] - t[i] >= deltaN:
            rows = j - i
            max_row = max(max_row, rows)
            total_row += rows
            total_count += 1
            i += 1
        else:
            j += 1

    avg_row = total_row / total_count
    print(max_row, avg_row)

    t = np.array(t)
    mdic = {"t": t}
    scipy.io.savemat(f"dataset/year_timestamp,mu={mu},all.mat", mdic)
