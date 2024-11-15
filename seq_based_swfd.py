from line_profiler import profile
from frequent_directions import FrequentDirectionsWithDump
from sliding_windows_frequent_directions import (
    FastSlidingWindowFD,
    OptSwfd,
    SlidingWindowFD,
)
import numpy as np
import numpy.typing as npt
from typing import Dict
from scipy.io import savemat
from tqdm import tqdm
from scipy import linalg
import pandas as pd


class SeqBasedSWFD:
    def __init__(
        self,
        N: int,
        R: float,
        d: int,
        sketch_dim: int,
        beta: float = 1.0,
        ty=SlidingWindowFD,
        faster=FrequentDirectionsWithDump,
        **kwargs,
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            R (float): Upper bound of square of 2-norm of row vectors.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            beta (float): Additional coefficient of error, default as 1.0.
        """
        self.N = np.uint64(N)
        self.d = d
        self.R = R
        self.logR = int(np.floor(np.log2(R))) + 1
        # print(self.logR)
        # exit(0)
        self.beta = beta
        self.sketch_dim = sketch_dim
        self.ty = ty
        self.faster = faster

        self.levels: Dict[int, ty] = {}

        if "upper_F_norm" in kwargs:
            self.logR = int(np.ceil(np.log2(kwargs["upper_F_norm"] / self.N)))
            base = self.N / self.sketch_dim
            for j in range(self.logR):
                self.levels[j] = ty(
                    self.N,
                    self.d,
                    self.sketch_dim,
                    error_threshold=base * (2**j),
                    faster=faster,
                )

        else:
            for j in range(self.logR):
                self.levels[j] = ty(
                    self.N,
                    self.d,
                    self.sketch_dim,
                    error_threshold=(2**j) * self.N / self.sketch_dim,
                    faster=faster,
                )

        self.time = np.uint64(0)

    @profile
    def fit(self, X: npt.NDArray):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
        """
        self.time += np.uint64(1)

        for j in range(self.logR):
            sq = self.levels[j].fd
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            sq = self.levels[j].fd_aux
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            error = C.get_error()
            if X @ X.T >= error:
                self.levels[j].append(X)
            else:
                self.levels[j].fit(X)

    # @profile
    def get(self):
        j = 0
        rj = 0
        while j < self.logR:
            sq = self.levels[j].fd
            q = sq.queue
            if len(q) != 0:
                head_snapshot = q[0]
                if self.time - head_snapshot.s >= min(self.N - 1, self.time - 1):
                    rj = j
                    break
            else:
                break
            j += 1

        j = rj

        return self.levels[j].get()

    def get_size(self):
        return sum([self.levels[j].get_size() for j in self.levels])


class TimeBasedSWFD:
    def __init__(
        self, N: int, R: float, d: int, sketch_dim: int, beta: float = 4.0, **kwargs
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            R (float): Upper bound of square of 2-norm of row vectors.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            beta (float): Additional coefficient of error, default as 4.0.
        """
        self.N = np.uint64(N)
        self.d = d
        self.R = R
        self.log_eNR = int(np.ceil(np.log2(N * R / sketch_dim)))
        self.beta = beta
        self.sketch_dim = sketch_dim

        # self.levels: Dict[int, FastSlidingWindowFD] = {}
        # for j in range(self.log_eNR):
        #     self.levels[j] = FastSlidingWindowFD(
        #         self.N, self.d, self.sketch_dim, error_threshold=2**j)

        self.levels: Dict[int, SlidingWindowFD] = {}

        if "upper_F_norm" in kwargs:
            self.log_eNR = int(
                np.ceil(np.log2(kwargs["upper_F_norm"] / self.sketch_dim))
            )
            base = 1.0
            if "lower_F_norm" in kwargs:
                self.log_eNR = int(
                    np.ceil(np.log2(kwargs["upper_F_norm"] / kwargs["lower_F_norm"]))
                )
                base = kwargs["lower_F_norm"] / self.sketch_dim
            for j in range(self.log_eNR):
                self.levels[j] = SlidingWindowFD(
                    self.N, self.d, self.sketch_dim, error_threshold=base * (2**j)
                )

        else:
            for j in range(self.log_eNR):
                self.levels[j] = SlidingWindowFD(
                    self.N, self.d, self.sketch_dim, error_threshold=2**j
                )

        self.time = np.uint64(0)

    def fit(self, X: npt.NDArray, t: int):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
            t (int): Arriving time.
        """
        self.time = t

        for j in range(self.log_eNR):
            sq = self.levels[j].fd
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            sq = self.levels[j].fd_aux
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            error = C.get_error()
            if X @ X.T >= error:
                self.levels[j].append(X, t)
            else:
                self.levels[j].fit(X, t)

    def get(self):
        j = 0
        rj = 0
        while j < self.log_eNR:
            sq = self.levels[j].fd
            q = sq.queue
            if len(q) != 0:
                head_snapshot = q[0]
                if self.time - head_snapshot.s >= min(self.N - 1, self.time - 1):
                    # temp = head_snapshot.s
                    rj = j
                    break
            else:
                break
            j += 1

        j = rj

        return self.levels[j].get()

    def get_size(self):
        return sum([self.levels[j].get_size() for j in self.levels])


class OptSeqBasedSwfd:
    def __init__(
        self,
        N: int,
        R: float,
        d: int,
        sketch_dim: int,
        beta: float = 1.0,
        **kwargs,
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            R (float): Upper bound of square of 2-norm of row vectors.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            beta (float): Additional coefficient of error, default as 1.0.
        """
        self.N = np.uint64(N)
        self.d = d
        self.R = R
        self.logR = int(np.floor(np.log2(R))) + 1
        # print(self.logR)
        # exit(0)
        self.beta = beta
        self.sketch_dim = sketch_dim

        self.levels: Dict[int, OptSwfd] = {}

        if "upper_F_norm" in kwargs:
            self.logR = int(np.ceil(np.log2(kwargs["upper_F_norm"] / self.N)))
            base = self.N / self.sketch_dim
            for j in range(self.logR):
                self.levels[j] = OptSwfd(
                    self.N,
                    self.d,
                    self.sketch_dim,
                    error_threshold=base * (2**j),
                )

        else:
            for j in range(self.logR):
                self.levels[j] = OptSwfd(
                    self.N,
                    self.d,
                    self.sketch_dim,
                    error_threshold=(2**j) * self.N / self.sketch_dim,
                )

        self.time = np.uint64(0)

    @profile
    def fit(self, X: npt.NDArray):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
        """
        self.time += np.uint64(1)

        for j in range(self.logR):
            sq = self.levels[j].fd
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            sq = self.levels[j].fd_aux
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            self.levels[j].fit(X)

    # @profile
    def get(self):
        j = 0
        rj = 0
        while j < self.logR:
            sq = self.levels[j].fd
            q = sq.queue
            if len(q) != 0:
                head_snapshot = q[0]
                if self.time - head_snapshot.s >= min(self.N - 1, self.time - 1):
                    rj = j
                    break
            else:
                break
            j += 1

        j = rj

        return self.levels[j].get()

    def get_size(self):
        return sum([self.levels[j].get_size() for j in self.levels])


class OptTimeBasedSwfd:
    def __init__(
        self,
        N: int,
        R: float,
        d: int,
        sketch_dim: int,
        beta: float = 4.0,
        **kwargs,
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            R (float): Upper bound of square of 2-norm of row vectors.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            beta (float): Additional coefficient of error, default as 1.0.
        """

        self.N = np.uint64(N)
        self.d = d
        self.R = R
        self.log_eNR = int(np.ceil(np.log2(N * R / sketch_dim)))
        self.beta = beta
        self.sketch_dim = sketch_dim

        self.levels: Dict[int, OptSwfd] = {}

        if "upper_F_norm" in kwargs:
            self.log_eNR = int(
                np.ceil(np.log2(kwargs["upper_F_norm"] / self.sketch_dim))
            )
            base = 1.0
            if "lower_F_norm" in kwargs:
                self.log_eNR = int(
                    np.ceil(np.log2(kwargs["upper_F_norm"] / kwargs["lower_F_norm"]))
                )
                base = kwargs["lower_F_norm"] / self.sketch_dim
            for j in range(self.log_eNR):
                self.levels[j] = OptSwfd(
                    self.N, self.d, self.sketch_dim, error_threshold=base * (2**j)
                )
        else:
            for j in range(self.log_eNR):
                self.levels[j] = OptSwfd(
                    self.N, self.d, self.sketch_dim, error_threshold=2**j
                )

        self.time = np.uint64(0)

    @profile
    def fit(self, X: npt.NDArray, t: int):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
        """
        self.time = t

        for j in range(self.log_eNR):
            sq = self.levels[j].fd
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            sq = self.levels[j].fd_aux
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            self.levels[j].fit(X, t)

    # @profile
    def get(self):
        j = 0
        rj = 0
        while j < self.log_eNR:
            sq = self.levels[j].fd
            q = sq.queue
            if len(q) != 0:
                head_snapshot = q[0]
                if self.time - head_snapshot.s >= min(self.N - 1, self.time - 1):
                    rj = j
                    break
            else:
                break
            j += 1

        j = rj

        return self.levels[j].get()

    def get_size(self):
        return sum([self.levels[j].get_size() for j in self.levels])


class FastTimeBasedSWFD:
    def __init__(
        self, N: int, R: float, d: int, sketch_dim: int, beta: float = 4.0, **kwargs
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            R (float): Upper bound of square of 2-norm of row vectors.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            beta (float): Additional coefficient of error, default as 4.0.
        """
        self.N = np.uint64(N)
        self.d = d
        self.R = R
        self.log_eNR = int(np.ceil(np.log2(N * R / sketch_dim)))
        self.beta = beta
        self.sketch_dim = sketch_dim

        # self.levels: Dict[int, FastSlidingWindowFD] = {}
        # for j in range(self.log_eNR):
        #     self.levels[j] = FastSlidingWindowFD(
        #         self.N, self.d, self.sketch_dim, error_threshold=2**j)

        self.levels: Dict[int, FastSlidingWindowFD] = {}

        if "upper_F_norm" in kwargs:
            self.log_eNR = int(
                np.ceil(np.log2(kwargs["upper_F_norm"] / self.sketch_dim))
            )
            base = 1.0
            if "lower_F_norm" in kwargs:
                self.log_eNR = int(
                    np.ceil(np.log2(kwargs["upper_F_norm"] / kwargs["lower_F_norm"]))
                )
                base = kwargs["lower_F_norm"] / self.sketch_dim
            for j in range(self.log_eNR):
                self.levels[j] = FastSlidingWindowFD(
                    self.N, self.d, self.sketch_dim, error_threshold=base * (2**j)
                )

        else:
            for j in range(self.log_eNR):
                self.levels[j] = FastSlidingWindowFD(
                    self.N, self.d, self.sketch_dim, error_threshold=2**j
                )

        self.time = np.uint64(0)

    # @profile
    def fit(self, X: npt.NDArray, t: int):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
            t (int): Arriving time.
        """
        self.time = t

        for j in range(self.log_eNR):
            sq = self.levels[j].fd
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            sq = self.levels[j].fd_aux
            q = sq.queue
            C = sq.C
            while len(q) != 0:
                head_snapshot = q[0]
                if (
                    len(q) > (2 + 8 / self.beta) * self.sketch_dim
                    or head_snapshot.t + self.N <= self.time
                ):
                    q.popleft()
                else:
                    break

            error = C.get_error()
            if X @ X.T >= error:
                self.levels[j].append(X, t)
            else:
                self.levels[j].fit(X, t)

    def get(self):
        j = 0
        rj = 0
        while j < self.log_eNR:
            sq = self.levels[j].fd
            q = sq.queue
            if len(q) != 0:
                head_snapshot = q[0]
                if self.time - head_snapshot.s >= min(self.N - 1, self.time - 1):
                    # temp = head_snapshot.s
                    rj = j
                    break
            else:
                break
            j += 1

        j = rj

        return self.levels[j].get()

    def get_size(self):
        return sum([self.levels[j].get_size() for j in self.levels])


def run(l, N, d, R, epochs, name):
    # A = np.empty((epochs, d))
    # sw_fd = SeqBasedSWFD(N, R, d, l)
    sw_fd = SeqBasedSWFD(N, R, d, l, ty=FastSlidingWindowFD)
    # B = np.empty((epochs, l, d))
    # deltas = np.empty((epochs))
    # count = 0
    # A_w = np.empty((epochs, N, d))
    for t in tqdm(range(epochs)):
        random_vector = np.random.randn(1, d)
        if name == "":
            a = np.sqrt(R) * random_vector / np.linalg.norm(random_vector, ord=2)
        if name == "normal":
            a = random_vector * np.sqrt(R) / d
            while linalg.norm(a, ord=2) ** 2 > R:
                random_vector = np.random.randn(1, d)
                a = random_vector * np.sqrt(R) / d
        if name == "cyclical":
            a = np.sqrt(R - 1) * random_vector / np.linalg.norm(random_vector, ord=2)
            if (t // 128) % 2 == 0:
                a /= 100
                count += 1
        if name == "test":
            a = np.sqrt(R / (l)) * np.eye(1, d, k=(t // N) % d)
        # if name == "test2":
        #     a = np.sqrt(R/(l-3)) * np.eye(1, d, k=(t//N) % d)
        # if name == "test3":
        #     a = np.sqrt(R/(l+3)) * np.eye(1, d, k=(t//N) % d)
        # A[t:t+1, :] = a
        # A_w[t, :min(t+1, N)] = A[max(0, t+1-N): t+1]
        # print(t)
        sw_fd.fit(a)
        # B_t, _, _, delta = sw_fd.get()
        # B[t] = B_t
        # print(linalg.norm(A_w[t].T @ A_w[t] - B[t].T@B[t], 2))
        # print(linalg.norm(A_w[t])**2/l)
        # deltas[t] = delta

    # mdic = {"l": l, 'N': N, 'd': d, 'R': R,
    #         'epochs': epochs, 'A': A, 'B': B, 'deltas': deltas}
    # file_path = f"results/result_seq_l={l},N={N},d={d},R={R},epochs={epochs}"
    # if name != "":
    #     file_path += f"_{name}"
    # file_path += ".mat"
    # savemat(file_path, mdic)
    # print(count)


def test_bank():
    df = pd.read_csv("dataset/Bank.csv", header=None)
    A = df.values.astype(np.float64)
    A = A[:, 1:]
    epochs, d = A.shape
    # print(epochs)
    # exit(0)
    # R = np.linalg.norm(A, axis=1)**2
    # print(R.shape)
    R = np.max(np.linalg.norm(A, axis=1) ** 2)
    # print((np.linalg.norm(A, axis=1)**2)[0])
    # print(np.log2(R))
    ls = [15, 25, 45]
    N = 2048
    A_w = np.empty((epochs, N, d))
    for l in ls:
        sw_fd = SeqBasedSWFD(N, R, d, l)
        B = np.empty((epochs, l, d))
        deltas = np.empty((epochs))
        count = 0
        # A_w = np.empty((epochs, N, d))
        for t in tqdm(range(epochs)):
            a = A[t : t + 1, :]
            A_w[t, : min(t + 1, N)] = A[max(0, t + 1 - N) : t + 1]
            # print(t)
            # A_w[t, :min(t, N)] = A[max(0, t-N): t]
            sw_fd.fit(a)
            A_f = linalg.norm(A_w[t]) ** 2
            eA_f = A_f / l
            B_t, _, _, delta = sw_fd.get()
            B[t] = B_t
            A_w2 = linalg.norm(A_w[t].T @ A_w[t], 2)
            B_w2 = linalg.norm(B[t].T @ B[t], 2)
            A_wB_w = linalg.norm(A_w[t].T @ A_w[t] - B[t].T @ B[t], 2)
            # print(t)
            # print(A_wB_w)
            # print(eA_f)
            if 4 * eA_f - A_wB_w < 0:
                print(t)
                print(A_w2)
                print(B_w2)
                print(A_wB_w)
                print(eA_f)
                print(4 * eA_f - A_wB_w)
            deltas[t] = delta

        mdic = {
            "l": l,
            "N": N,
            "d": d,
            "R": R,
            "epochs": epochs,
            "A": A,
            "B": B,
            "deltas": deltas,
        }
        file_path = f"results/result_bank_l={
            l},N={N},d={d},epochs={epochs}"
        file_path += ".mat"
        savemat(file_path, mdic)
        print(file_path)
        print(count)


if __name__ == "__main__":
    # test_bank()
    run(l=400, N=1000, d=6400, R=1, epochs=4000, name="")
    # run(50, 1000, 300, 1, 5000, "")
    # N = 64
    # d = 32
    # R = 2**16 - 1
    # epochs = 2048  # * 4
    # names = ["", "normal", "cyclical", "test"]
    # for name in names:
    #     l = 16
    #     while l < d:
    #         run(l, N, d, R, epochs, name)
    #         l *= 2
    # # l, N, d, R, epochs = l*2, N*2, d*2, R*2, epochs*2
