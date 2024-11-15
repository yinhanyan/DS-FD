import scipy.io
from scipy import linalg
import numpy as np
from frequent_directions import FasterFrequentDirectionsWithDump
from seq_based_swfd import OptSeqBasedSwfd, SeqBasedSWFD
from lmfd import LMFD
from difd import DIFD
from tqdm import tqdm
from rowsample import SWR, SWOR
import time
import pickle
from pympler.asizeof import asizeof
import pandas as pd
import argparse
from sliding_windows_frequent_directions import FastSlidingWindowFD
from utils import sliding_window_agg
import fastfdwithdump
import frequent_directions

parser = argparse.ArgumentParser()
parser.add_argument("-m")
args = parser.parse_args()


# @profile
def run():
    np.random.seed(0)
    name = "pamap_norm"
    df = pd.read_csv(
        f"dataset/PAMAP2_Dataset/Protocol/subject103.dat", delim_whitespace=True
    )
    A = df.values.astype(np.float64)
    A = A[:, 2:]
    A[np.isnan(A)] = 1
    epochs, d = A.shape
    N = 10000
    # print(len(np.linalg.norm(A, axis=1)))
    Rs = np.linalg.norm(A, axis=1) ** 2
    r = np.min(Rs)
    # R = np.max(Rs)
    # print(R, r)
    # print(epochs, d)
    # exit(0)
    A = A / np.sqrt(r)

    Rs = np.linalg.norm(A, axis=1) ** 2
    r = np.min(Rs)
    R = np.max(Rs)
    # print(R, r)
    # exit(0)
    # ls = [50, 100, 150, 200, 250]
    ls = [5, 10, 20, 30, 40]
    ls = [40]
    query_step = epochs // 500
    # query_step = epochs + 1
    results = {}
    method = args.m
    print(method)
    if method == "ours_maxF":
        max_F = sliding_window_agg(Rs, N, max, 0)
        # print(max_F)
        # exit(0)

    for l in ls:
        with open(f"logs/{name},{method},l={l},N={N}.txt", "w") as f:
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
                case "ours_maxF":
                    swfd = SeqBasedSWFD(N, R, d, l, upper_F_norm=max_F)
            # elif method == ""
            # elif method == "best":
            # max_size = asizeof(swfd)
            for t in tqdm(range(epochs)):
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

            match method:
                case "r1a":
                    print(fastfdwithdump.SVD_COUNT, fastfdwithdump.VALID_SVD_COUNT)
                case "ours":
                    print(
                        frequent_directions.SVD_COUNT_OURS,
                        frequent_directions.FLUSH_HIT,
                        frequent_directions.FLUSH_ENTER,
                    )

        with open(f"logs/{name},{method},l={l},N={N}.pkl", "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    run()
