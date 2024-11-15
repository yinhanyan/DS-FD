import scipy.io
from scipy import linalg
import numpy as np
from frequent_directions import FasterFrequentDirectionsWithDump
from seq_based_swfd import OptSeqBasedSwfd, SeqBasedSWFD
from lmfd import LMFD
from difd import DIFD
from tqdm import trange
from rowsample import SWR, SWOR
import time
import pickle
from pympler.asizeof import asizeof
import pandas as pd
import argparse

from sliding_windows_frequent_directions import FastSlidingWindowFD

parser = argparse.ArgumentParser()
parser.add_argument("-m")
parser.add_argument("--epochs", default=500000, type=int)
parser.add_argument("-N", default=100000, type=int)
parser.add_argument("-M", default=10, type=int)
parser.add_argument("-d", default=300, type=int)
args = parser.parse_args()


# @profile
def run():
    np.random.seed(0)
    m = args.M
    d = args.d
    A = scipy.io.loadmat(f"dataset/synthetic_n={args.epochs},m={m},d={d},zeta=10.mat")[
        "A"
    ]
    A = A.astype(np.float64)
    # A = A[:10000, :]
    epochs, d = A.shape
    N = args.N
    # print(len(np.linalg.norm(A, axis=1)))
    r = np.min(np.linalg.norm(A, axis=1) ** 2)
    R = np.max(np.linalg.norm(A, axis=1) ** 2)
    # print(R, r)
    # exit(0)
    ls = [50, 100, 150, 200, 250]
    ls = [50]
    query_step = epochs // 500
    # query_step = epochs
    results = {}
    method = args.m
    print(method)

    for l in ls:
        pkl_path = f"logs/synthetic,{method},epochs={epochs},d={d},l={l},N={N},m={m}.pkl"
        with open(
            f"logs/synthetic,{method},epochs={epochs},d={d},l={l},N={N},m={m}.txt", "w"
        ) as f:
            max_error = 0.0
            sum_error = 0.0
            sum_update_time_ms = 0
            sum_query_time_ms = 0
            query_count = 1
            max_size = 0

            match method:
                case "opt":
                    swfd = OptSeqBasedSwfd(N, R, d, l, beta=1.0)
                case "fst":
                    swfd = SeqBasedSWFD(
                        N, R, d, l, beta=1.0, faster=FasterFrequentDirectionsWithDump
                    )
                case "r1a":
                    swfd = SeqBasedSWFD(N, R, d, l, beta=1.0, ty=FastSlidingWindowFD)
                case "ours":
                    swfd = SeqBasedSWFD(N, R, d, l, beta=1.0)
                case "lmfd":
                    swfd = LMFD(N, d, l)
                case "difd":
                    swfd = DIFD(N, d, l, R)
                case "swr":
                    swfd = SWR(N, l, d)
                case "swor":
                    swfd = SWOR(N, l, d)
            # elif method == ""
            # elif method == "best":
            # max_size = asizeof(swfd)
            for t in trange(epochs, desc=pkl_path):
                a = A[t : t + 1, :]
                A_w = A[max(0, t + 1 - N) : t + 1]

                start_time = time.process_time_ns()
                swfd.fit(a)
                # max_size = max(max_size, asizeof(swfd))
                end_time = time.process_time_ns()
                elapsed_time = end_time - start_time
                sum_update_time_ms += elapsed_time // (10**6)

                if (t + 1) % query_step == 0:
                    # if t > N and t % query_step == 0:
                    start_time = time.process_time_ns()
                    B_t, _, _, _ = swfd.get()
                    end_time = time.process_time_ns()
                    elapsed_time = end_time - start_time
                    sum_query_time_ms += elapsed_time // (10**6)

                    A_f = linalg.norm(A_w) ** 2
                    eA_f = A_f / l
                    A_wB_w = linalg.norm(A_w.T @ A_w - B_t.T @ B_t, 2)

                    if eA_f - A_wB_w < 0:
                        print(
                            f"l={l}, t={t}, A_wB_w={A_wB_w}, eA_f={
                            eA_f}, error = {eA_f - A_wB_w}"
                        )
                    avg_update_time = sum_update_time_ms / epochs
                    # print("avg_update_time", avg_update_time)

                    relative_error = A_wB_w / A_f
                    max_error = max(max_error, relative_error)
                    sum_error += relative_error
                    max_size = max(max_size, swfd.get_size())

                    query_count += 1

            avg_error = sum_error / query_count
            avg_update_time = sum_update_time_ms / epochs
            avg_query_time = sum_query_time_ms / query_count
            results[l] = {
                "max_error": max_error,
                "avg_error": avg_error,
                "avg_update_time": avg_update_time,
                "avg_query_time": avg_query_time,
                "max_size": max_size,
            }

            f.write(
                f"l={l}, max_error={max_error}, avg_error={avg_error}, avg_update_time={
                    avg_update_time}, avg_query_time={avg_query_time}, max_size={max_size}\n"
            )

        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    run()
