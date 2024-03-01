import scipy.io
from scipy import linalg
import numpy as np
from seq_based_swfd import TimeBasedSWFD
from lmfd import LMFD
from rowsample import SWR, SWOR
from tqdm import tqdm
import time
import pickle
from pympler.asizeof import asizeof
import pandas as pd
import argparse
from utils import time_based_sliding_window_agg

parser = argparse.ArgumentParser()
parser.add_argument('-m')
args = parser.parse_args()


# @profile
def run():
    np.random.seed(0)
    # np.seterr(all="raise")
    name = "year_norm"
    df = pd.read_csv("dataset/YearPredictionMSD.txt", delim_whitespace=False)
    # timestamps = scipy.io.loadmat("dataset/year_timestamp.mat")["t"][0]
    timestamps = scipy.io.loadmat("dataset/year_timestamp,mu=0.5,all.mat")["t"][0]
    A = df.values.astype(np.float64)
    # A = A[-51630:, 1:]
    A = A.astype(np.float64)
    A = A[:200000, 1:]
    timestamps = timestamps[:200000]
    # A = A[:, :50]
    epochs, d = A.shape
    N = 50000
    # print(len(np.linalg.norm(A, axis=1)))
    Rs = np.linalg.norm(A, axis=1)**2
    r = np.min(Rs)
    # R = np.max(Rs)
    # print(r, R)
    A = A / np.sqrt(r)

    Rs = np.linalg.norm(A, axis=1)**2
    r = np.min(Rs)
    R = np.max(Rs)
    # print(r, R)
    # print(epochs, d)
    # exit(0)
    ls = [50, 100, 150, 200, 250]
    ls = [32]
    query_step = epochs // 600
    # query_step = 1
    # query_step = 24192
    results = {}
    method = args.m
    print(method)
    # if method == "ours_maxF" or method == "ours_maxminF":
    if method == "ours":
        max_F = time_based_sliding_window_agg(Rs, timestamps, N, max, 0)
        # min_F = time_based_sliding_window_agg(Rs, timestamps, N, min, max_F)
        # print(np.log(max_F))
        # exit(0)

    for l in ls:
        with open(f"logs/{name},{method},l={l},N={N},qs={query_step}.txt", "w") as f:
            max_error = 0.
            sum_error = 0.
            sum_update_time_ms = 0
            sum_query_time_ms = 0
            query_count = 1
            max_size = 0
            query_layer = {}

            match method:
                case "ours":
                    swfd = TimeBasedSWFD(N, R, d, l, beta=8, upper_F_norm=max_F)
                case "lmfd":
                    swfd = LMFD(N, d, l)
                case "swr":
                    swfd = SWR(N, l, d)
                case "swor":
                    swfd = SWOR(N, l, d)
                case "ours_maxF":
                    swfd = TimeBasedSWFD(
                        N, R, d, l, beta=8, upper_F_norm=max_F)
                case "ours_maxminF":
                    swfd = TimeBasedSWFD(
                        N, R, d, l, beta=4, upper_F_norm=max_F, lower_F_norm=r)
                    # print(swfd.log_eNR)
                    # exit()
            # elif method == "best":
            # max_size = asizeof(swfd)
            for t in tqdm(range(epochs)):
                a = A[t:t+1, :]
                timestamp = timestamps[t]

                start_time = time.process_time_ns()
                swfd.fit(a, t=timestamp)
                # max_size = max(max_size, asizeof(swfd))
                end_time = time.process_time_ns()
                elapsed_time = end_time - start_time
                sum_update_time_ms += elapsed_time//(10**6)

                if t % query_step == 0:
                    # print(max_size)
                    # if t > N and t % query_step == 0:
                    start_time = time.process_time_ns()
                    B_t, _, _, _ = swfd.get()
                    end_time = time.process_time_ns()
                    elapsed_time = end_time - start_time
                    sum_query_time_ms += elapsed_time//(10**6)

                    left_ts = max(0, timestamp - N + 1)
                    left_t = np.searchsorted(timestamps, left_ts)
                    A_w = A[left_t: t+1]

                    A_f = linalg.norm(A_w)**2
                    eA_f = A_f/l
                    A_wB_w = linalg.norm(A_w.T @ A_w - B_t.T@B_t, 2)

                    if eA_f - A_wB_w < 0:
                        n_A_w = linalg.norm(A_w.T @ A_w)
                        n_B_w = linalg.norm(B_t.T@B_t)
                        print(f"l={l}, t={t}, A_wB_w={A_wB_w}, eA_f={eA_f}, error = {
                              eA_f - A_wB_w}, n_A_w={n_A_w}, n_B_w={n_B_w}")

                    relative_error = A_wB_w/A_f
                    max_error = max(max_error, relative_error)
                    sum_error += relative_error
                    max_size = max(max_size, swfd.get_size())

                    query_count += 1

                    # if j in query_layer:
                    #     query_layer[j] += 1
                    # else:
                    #     query_layer[j] = 1

            avg_error = sum_error / query_count
            avg_update_time = sum_update_time_ms / epochs
            avg_query_time = sum_query_time_ms / query_count
            results[l] = {"max_error": max_error, "avg_error": avg_error,
                          "avg_update_time": avg_update_time, "avg_query_time": avg_query_time, "max_size": max_size, }
            #   "query_layer": query_layer}

            f.write(f"l={l}, max_error={max_error}, avg_error={avg_error}, avg_update_time={
                    avg_update_time}, avg_query_time={avg_query_time}, max_size={max_size}\n")

        with open(f"logs/{name},{method},l={l},N={N},qs={query_step}.pkl", "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    run()
