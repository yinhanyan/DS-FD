import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from collections import deque
from typing import Dict
from scipy import linalg
from copy import deepcopy
from frequent_directions import RobustFrequentDirections


@dataclass
class LMBlock():
    sketch: RobustFrequentDirections
    buffer: npt.NDArray | None = None
    count: int = 0
    start: int = 1
    end: int = 0
    size: float = 0.

    def append(self, X):
        self.count += 1
        if self.buffer is None:
            self.buffer = X
        else:
            self.buffer = np.vstack([self.buffer, X])

        if len(self.buffer) >= self.sketch.sketch_dim:
            self.flush()

    def flush(self):
        if self.buffer is not None:
            self.sketch.fit(self.buffer, batch_size=self.sketch.sketch_dim)
            self.buffer = None

    def get_sketch(self):
        if self.buffer is not None:
            self.flush()
        return self.sketch.get_sketch()


class DIFD():
    def __init__(self, N, d, sketch_dim, R) -> None:
        self.N = N
        self.d = d
        self.sketch_dim = sketch_dim
        self.R = R

        self.L_f = np.log2(R*sketch_dim)
        self.L = int(np.ceil(self.L_f))

        self.levels: Dict[int, deque[LMBlock]] = {
            i: deque() for i in range(1, self.L+1)}
        self.Bs: Dict[int, LMBlock] = {i: LMBlock(RobustFrequentDirections(self.d, min(self.d, self.L * (2**i))))
                                       for i in range(1, self.L+1)}
        self.time = 0
        self.base_len = 0

    def fit(self, X):
        self.time += 1

        for l in range(1, self.L+1):
            q = self.levels[l]
            while len(q) != 0:
                if q[0].start < self.time-self.N+1:
                    q.popleft()
                else:
                    break

            self.Bs[l].size += X@X.T
            self.Bs[l].end = self.time
            self.Bs[l].append(X)

        if self.Bs[1].size > self.N / self.sketch_dim:
            self.base_len += 1
            v = self.base_len
            v = (v & -v).bit_length()
            v = min(v, self.L+1)
            for i in range(1, v):
                if self.Bs[i].start >= self.time-self.N+1:
                    self.Bs[i].flush()
                    self.levels[i].append(self.Bs[i])
                self.Bs[i] = LMBlock(sketch=RobustFrequentDirections(
                    self.d, min(self.d, self.L*(2**i))), start=self.time+1, end=self.time)

    def get(self):
        a = self.time
        b = max(1, int(self.time - self.N + 1))
        ret = self.Bs[1].get_sketch()

        for i in range(self.L, 0, -1):
            if len(self.levels[i]) != 0:
                lb = self.levels[i][0]
                if lb.start >= self.time - self.N + 1:
                    if lb.end <= a:
                        ret = np.vstack([ret, lb.get_sketch()])
                        a = lb.start
                    if len(self.levels[i]) != 1:
                        rb = self.levels[i][-1]
                        if rb.start >= b:
                            ret = np.vstack([ret, rb.get_sketch()])
                            b = rb.end
                    else:
                        b = lb.end

            # if self.Bs[i].sketch is not None:
            #     lb = self.Bs[i] if len(
            #         self.levels[i]) == 0 else self.levels[i][0]
            #     rb = self.Bs[i]

            #     if lb.start >= int(self.time - self.N + 1) and lb.end <= a:
            #         ret = np.vstack([ret, lb.sketch])
            #         a = lb.start - 1
            #         b = rb.end + 1
            #     if len(self.levels[i]) != 0:
            #         if rb.start >= b:
            #             ret = np.vstack([ret, rb.sketch])
            #             a = rb.start - 1
            #             b = rb.end + 1

        _, s, Vt = linalg.svd(
            ret, full_matrices=False, lapack_driver='gesvd')
        s = s ** 2

        if len(s) > self.sketch_dim:
            s = s[:self.sketch_dim] - s[self.sketch_dim]
        Vt = Vt[:self.sketch_dim]

        ret = Vt * np.sqrt(s).reshape(-1, 1)

        return ret, None, None, None

    def get_size(self):
        res = 0
        # if self.B.sketch is not None:
        #     res += self.B.sketch.shape[0]
        for l in self.levels:
            for b in self.levels[l]:
                if b.sketch is not None:
                    res += min(b.count, self.sketch_dim)

        return res
        # return sum([len(self.levels[l]) * self.sketch_dim for l in self.levels]) + self.L * self.sketch_dim
