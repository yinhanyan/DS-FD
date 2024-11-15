from line_profiler import profile
from frequent_directions import (
    FrequentDirectionsWithDump,
    FasterFrequentDirectionsWithDump,
)
from fastfdwithdump import FastFrequentDirectionsWithDump
from dump_fd import DumpFdBase, OptDumpFd
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from scipy.io import savemat
from tqdm import tqdm
from scipy import linalg
from collections import deque
from abc import ABC, abstractmethod


@dataclass
class FDSnapshot:
    v: npt.NDArray
    s: np.uint64
    t: np.uint64


@dataclass
class DumpFDwithSnapshotQueue:
    C: (
        DumpFdBase
        | FrequentDirectionsWithDump
        | FastFrequentDirectionsWithDump
        | FasterFrequentDirectionsWithDump
    )
    queue: deque[FDSnapshot] = field(default_factory=deque)
    last_dump_time: np.uint64 = np.uint64(1)
    size: float = 0.0


class SlidingWindowFdBase(ABC):
    @abstractmethod
    def fit(self, X: npt.NDArray, time: int | None) -> None:
        pass

    @abstractmethod
    def get_sketch(self) -> npt.NDArray:
        pass


class SlidingWindowFD:
    def __init__(
        self,
        N: int,
        d: int,
        sketch_dim: int,
        C: int = 1,
        faster=FrequentDirectionsWithDump,
        **kwargs,
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            error_threshold (float, optional): Dump threshold, default as `1.0`.
        """
        if "error_threshold" in kwargs:
            self.error = kwargs["error_threshold"]
        else:
            self.error = N * 1.0 / sketch_dim

        self.faster = faster

        self.fd = DumpFDwithSnapshotQueue(C=self.faster(d, sketch_dim * C, self.error))
        self.fd_aux = DumpFDwithSnapshotQueue(
            C=self.faster(d, sketch_dim * C, self.error)
        )
        self.period = 0

        self.N = np.uint64(N)
        self.d = d
        self.C = C
        self.sketch_dim = sketch_dim
        self.time = np.uint64(0)
        self.size = 0.0
        self.real_size = 0.0

    def append(self, X: npt.NDArray, t=None):
        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)

        # if self.fd.size > 2 * self.error * self.sketch_dim:
        # if self.time % self.N == 1:
        while self.period < self.time // self.N:
            self.fd = self.fd_aux
            self.fd_aux = DumpFDwithSnapshotQueue(
                C=self.faster(self.d, self.sketch_dim * self.C, self.error),
                last_dump_time=self.time,
            )
            self.period += 1

        s = self.fd.last_dump_time
        self.fd.last_dump_time = self.time + 1
        self.fd.queue.append(FDSnapshot(v=X, s=s, t=self.time))

        s = self.fd_aux.last_dump_time
        self.fd_aux.last_dump_time = self.time + 1
        self.fd_aux.queue.append(FDSnapshot(v=X, s=s, t=self.time))

    @profile
    def fit(self, X: npt.NDArray, t=None):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
        """

        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)
        # self.fd.size += X@X.T
        # self.fd_aux.size += X@X.T

        # if self.fd.size > 2 * self.error * self.sketch_dim:
        # if self.time % self.N == 1:
        while self.period < self.time // self.N:
            self.fd = self.fd_aux
            self.fd_aux = DumpFDwithSnapshotQueue(
                C=self.faster(self.d, self.sketch_dim * self.C, self.error),
                last_dump_time=self.time,
            )
            self.period += 1

        while len(self.fd.queue) != 0:
            head_snapshot = self.fd.queue[0]
            if head_snapshot.t + self.N <= self.time:
                self.fd.queue.popleft()
            else:
                break

        while len(self.fd_aux.queue) != 0:
            head_snapshot = self.fd_aux.queue[0]
            if head_snapshot.t + self.N <= self.time:
                self.fd_aux.queue.popleft()
            else:
                break

        # with energy optimization
        self.fd.C.fit(X)

        dumped = self.fd.C.dump()
        if dumped is not None:
            s = self.fd.last_dump_time
            self.fd.last_dump_time = self.time + 1
            self.fd.queue.append(FDSnapshot(v=dumped, s=s, t=self.time))

        self.fd_aux.C.fit(X)

        dumped = self.fd_aux.C.dump()
        if dumped is not None:
            s = self.fd_aux.last_dump_time
            self.fd_aux.last_dump_time = self.time + 1
            self.fd_aux.queue.append(FDSnapshot(v=dumped, s=s, t=self.time))

        # without energy
        # sketch, sigma_squared, Vt, delta = self.fd.get()

        # if sigma_squared[0] >= (self.N * 1.0 * self.R)/(self.C * self.sketch_dim):
        #     v = np.sqrt(sigma_squared[0]) * Vt[0:1]
        #     t = self.time
        #     self.snapshots.append(FDSnapshot(v, t))
        #     sketch[0, :] = 0
        #     sigma_squared[0] = 0
        #     Vt[0, :] = 0
        #     np.roll(sketch, -1)
        #     np.roll(sigma_squared, -1)
        #     np.roll(Vt, -1)

    def update(self, X: npt.NDArray):
        sq = self.fd
        q = sq.queue
        C = sq.C
        while len(q) != 0:
            head_snapshot = q[0]
            if len(q) > self.queue_capacity or head_snapshot.t + self.N <= self.time:
                q.popleft()
            else:
                break

        sq = self.fd_aux
        q = sq.queue
        C = sq.C
        while len(q) != 0:
            head_snapshot = q[0]
            if len(q) > self.queue_capacity or head_snapshot.t + self.N <= self.time:
                q.popleft()
            else:
                break

        error = C.get_error()
        if X @ X.T >= error:
            self.append(X)
        else:
            self.fit(X)

    def get(self):
        q = self.fd.queue
        C = self.fd.C.get_sketch()
        if len(q) != 0:
            ret = np.vstack([C, *(s.v for s in q)])

            _, s, Vt = linalg.svd(ret, full_matrices=False, lapack_driver="gesvd")
            s = s**2
            if s.shape[0] > self.sketch_dim:
                s = s[: self.sketch_dim] - s[self.sketch_dim]
            Vt = Vt[: self.sketch_dim]
            # else:
            #     s = np.pad(s, (0, self.sketch_dim-s.shape[0]), 'constant')
            #     Vt = np.pad(
            #         Vt, ((0, self.sketch_dim-Vt.shape[0]), (0, 0)), 'constant')
            sketch = Vt * np.sqrt(s).reshape(-1, 1)

            return sketch, s, Vt, 0.0
        else:
            return self.fd.C.get()

    # @profile
    def get_sketch(self):
        q = self.fd.queue
        ret = self.fd.C.sketch
        if len(q) != 0:
            ret = np.vstack([ret, *(s.v for s in q)])

        if self.fd.C.buffer is not None:
            ret = np.vstack([ret, self.fd.C.buffer])

        if len(ret) > self.sketch_dim:
            _, s, Vt = linalg.svd(ret, full_matrices=False, lapack_driver="gesvd")
            s = s**2
            s = s[: self.sketch_dim] - s[self.sketch_dim]
            Vt = Vt[: self.sketch_dim]
            sketch = Vt * np.sqrt(s).reshape(-1, 1)

            return sketch
        else:
            return ret

    def get_size(self):
        return len(self.fd.queue) + len(self.fd_aux.queue) + 2 * self.sketch_dim


class OptSwfd(SlidingWindowFdBase):
    def __init__(
        self,
        N: int,
        d: int,
        sketch_dim: int,
        **kwargs,
    ):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            error_threshold (float, optional): Dump threshold, default as `1.0`.
        """
        if "error_threshold" in kwargs:
            self.error = kwargs["error_threshold"]
        else:
            self.error = N * 1.0 / sketch_dim

        self.fd = DumpFDwithSnapshotQueue(C=OptDumpFd(d, sketch_dim, self.error))
        self.fd_aux = DumpFDwithSnapshotQueue(C=OptDumpFd(d, sketch_dim, self.error))
        self.period = 0

        self.N = np.uint64(N)
        self.d = d
        # self.C = C
        self.sketch_dim = sketch_dim
        self.time = np.uint64(0)
        self.size = 0.0
        self.real_size = 0.0

    def fit(self, X: npt.NDArray, t=None):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
        """

        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)
        # self.fd.size += X@X.T
        # self.fd_aux.size += X@X.T

        # if self.fd.size > 2 * self.error * self.sketch_dim:
        # if self.time % self.N == 1:
        while self.period < self.time // self.N:
            self.fd = self.fd_aux
            self.fd_aux = DumpFDwithSnapshotQueue(
                C=OptDumpFd(self.d, self.sketch_dim, self.error),
                last_dump_time=self.time,
            )
            self.period += 1

        while len(self.fd.queue) != 0:
            head_snapshot = self.fd.queue[0]
            if head_snapshot.t + self.N <= self.time:
                self.fd.queue.popleft()
            else:
                break

        while len(self.fd_aux.queue) != 0:
            head_snapshot = self.fd_aux.queue[0]
            if head_snapshot.t + self.N <= self.time:
                self.fd_aux.queue.popleft()
            else:
                break

        # with energy optimization
        dumped = self.fd.C.fit(X)

        if dumped is not None:
            s = self.fd.last_dump_time
            self.fd.last_dump_time = self.time + 1
            self.fd.queue.append(FDSnapshot(v=dumped, s=s, t=self.time))

        dumped = self.fd_aux.C.fit(X)

        if dumped is not None:
            s = self.fd_aux.last_dump_time
            self.fd_aux.last_dump_time = self.time + 1
            self.fd_aux.queue.append(FDSnapshot(v=dumped, s=s, t=self.time))

    def get(self):
        q = self.fd.queue
        ret = self.fd.C.get_sketch()
        if len(q) != 0:
            ret = np.vstack([ret, *(s.v for s in q)])

        return ret, None, None, None

    # @profile
    def get_sketch(self):
        q = self.fd.queue
        ret = self.fd.C.sketch
        if len(q) != 0:
            ret = np.vstack([ret, *(s.v for s in q)])

        return ret

    def get_size(self):
        return len(self.fd.queue) + len(self.fd_aux.queue) + 2 * self.sketch_dim


class FastSlidingWindowFD:
    def __init__(self, N: int, d: int, sketch_dim: int, C: int = 1, **kwargs):
        """Sliding Window on Frequent Directions

        Args:
            N (int): Sliding window size.
            d (int): Vector dimension.
            sketch_dim (int): Sketch dimension.
            error_threshold (float, optional): Dump threshold, default as `1.0`.
        """
        if "error_threshold" in kwargs:
            self.error = kwargs["error_threshold"]
        else:
            self.error = N * 1.0 / sketch_dim

        self.fd = DumpFDwithSnapshotQueue(
            C=FastFrequentDirectionsWithDump(d, sketch_dim * C, self.error)
        )
        self.fd_aux = DumpFDwithSnapshotQueue(
            C=FastFrequentDirectionsWithDump(d, sketch_dim * C, self.error)
        )
        self.period = 0

        self.N = np.uint64(N)
        self.d = d
        self.C = C
        self.sketch_dim = sketch_dim
        self.time = np.uint64(0)

    def append(self, X: npt.NDArray, t=None):
        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)

        # self.fd.size += X@X.T
        # self.fd_aux.size += X@X.T

        # if self.fd.size > 2 * self.error * self.sketch_dim:
        while self.period < self.time // self.N:
            self.fd = self.fd_aux
            self.fd_aux = DumpFDwithSnapshotQueue(
                C=FastFrequentDirectionsWithDump(
                    self.d, self.sketch_dim * self.C, self.error
                ),
                last_dump_time=self.time,
            )
            self.period += 1

        s = self.fd.last_dump_time
        self.fd.last_dump_time = self.time + 1
        self.fd.queue.append(FDSnapshot(v=X, s=s, t=self.time))

        s = self.fd_aux.last_dump_time
        self.fd_aux.last_dump_time = self.time + 1
        self.fd_aux.queue.append(FDSnapshot(v=X, s=s, t=self.time))

    @profile
    def fit(self, X: npt.NDArray, t=None):
        """Handle the input vector

        Args:
            X (npt.NDArray): Arriving vector at the time. (row vector: (1, n)-shape)
        """

        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)

        # self.fd.size += X@X.T
        # self.fd_aux.size += X@X.T

        # if self.fd.size > 2 * self.error * self.sketch_dim:
        while self.period < self.time // self.N:
            self.fd = self.fd_aux
            self.fd_aux = DumpFDwithSnapshotQueue(
                C=FastFrequentDirectionsWithDump(
                    self.d, self.sketch_dim * self.C, self.error
                ),
                last_dump_time=self.time,
            )
            self.period += 1

        while len(self.fd.queue) != 0:
            head_snapshot = self.fd.queue[0]
            if head_snapshot.t + self.N <= self.time:
                self.fd.queue.popleft()
            else:
                break

        while len(self.fd_aux.queue) != 0:
            head_snapshot = self.fd_aux.queue[0]
            if head_snapshot.t + self.N <= self.time:
                self.fd_aux.queue.popleft()
            else:
                break

        # with energy optimization
        dumped = self.fd.C.fit(X)

        if dumped is not None:
            s = self.fd.last_dump_time
            self.fd.last_dump_time = self.time + 1
            self.fd.queue.append(FDSnapshot(v=dumped, s=s, t=self.time))

        dumped = self.fd_aux.C.fit(X)

        if dumped is not None:
            s = self.fd_aux.last_dump_time
            self.fd_aux.last_dump_time = self.time + 1
            self.fd_aux.queue.append(FDSnapshot(v=dumped, s=s, t=self.time))

    def get(self):
        q = self.fd.queue
        C = self.fd.C.get_sketch()
        if len(q) != 0:
            ret = np.vstack([C, *(s.v for s in q)])

            _, s, Vt = linalg.svd(ret, full_matrices=False, lapack_driver="gesvd")
            s = s**2
            s = s[: self.sketch_dim] - s[self.sketch_dim]
            Vt = Vt[: self.sketch_dim]
            sketch = Vt * np.sqrt(s).reshape(-1, 1)

            return sketch, s, Vt, 0.0
        else:
            return self.fd.C.get()

    # @profile
    def get_sketch(self):
        q = self.fd.queue
        ret = self.fd.C.sketch
        if len(q) != 0:
            ret = np.vstack([ret, *(s.v for s in q)])

        if self.fd.C.buffer is not None:
            ret = np.vstack([ret, self.fd.C.buffer])

        if len(ret) > self.sketch_dim:
            _, s, Vt = linalg.svd(ret, full_matrices=False, lapack_driver="gesvd")
            s = s**2
            s = s[: self.sketch_dim] - s[self.sketch_dim]
            Vt = Vt[: self.sketch_dim]
            sketch = Vt * np.sqrt(s).reshape(-1, 1)

            return sketch
        else:
            return ret

    def get_size(self):
        return len(self.fd.queue) + len(self.fd_aux.queue) + 2 * self.sketch_dim


def run(l, N, d, epochs):
    A = np.empty((epochs, d))
    sw_fd = SlidingWindowFD(N, d, l)
    B = np.empty((epochs, l, d))
    deltas = np.empty((epochs))
    count = 0
    for t in tqdm(range(epochs)):
        random_vector = np.random.randn(1, d)
        a = random_vector / np.linalg.norm(random_vector, ord=2)
        # a = np.sqrt(R/(l-3)) * np.eye(1, d, k=(t//N) % d)
        # if (t // 128) % 2 == 0:
        #     a /= 100
        #     count += 1
        # a = random_vector * np.sqrt(R)/d
        # while linalg.norm(a, ord=2) ** 2 > R:
        #     random_vector = np.random.randn(1, d)
        #     a = random_vector * np.sqrt(R)/d
        A[t : t + 1, :] = a
        sw_fd.fit(a)
        B_t, _, _, delta = sw_fd.get()
        B[t] = B_t
        deltas[t] = delta

    mdic = {
        "l": l,
        "N": N,
        "d": d,
        "R": 1.0,
        "epochs": epochs,
        "A": A,
        "B": B,
        "deltas": deltas,
    }
    mat_path = f"results/result_l={l},N={N},d={d},epochs={epochs}.mat"
    savemat(mat_path, mdic)
    print(mat_path)
    print(count)


if __name__ == "__main__":
    l = 16
    N = 16**3
    d = 32
    epochs = N * 16
    for i in range(1):
        run(l, N, d, epochs)
        l, N, d, epochs = l * 2, N * 2, d * 2, epochs * 2
