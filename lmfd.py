import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from collections import deque
from typing import Dict
from scipy import linalg
from copy import deepcopy


@dataclass
class LMBlock():
    sketch: npt.NDArray | None
    start: np.uint64
    end: np.uint64
    size: float


class LMFD():
    def __init__(self, N, d, sketch_dim) -> None:
        self.L = 1
        self.levels: Dict[int, deque[LMBlock]] = {1: deque()}
        self.N = N
        self.d = d
        self.sketch_dim = sketch_dim
        self.b = self.sketch_dim
        self.time = np.uint64(0)
        self.B: LMBlock = LMBlock(None, self.time, self.time, 0.)

    def __merge(self, B1, B2):
        if B1 is not None and B2 is not None:
            stacked = np.vstack([B1, B2])
            _, s, Vt = linalg.svd(
                stacked, full_matrices=False, lapack_driver='gesvd')
            s = s ** 2
            if len(s) > self.sketch_dim:
                s = s[:self.sketch_dim] - s[self.sketch_dim]
            Vt = Vt[:self.sketch_dim]

            merged = Vt * np.sqrt(s).reshape(-1, 1)

            return merged
        elif B1 is not None:
            return B1
        elif B2 is not None:
            return B2
        else:
            return np.zeros((self.sketch_dim, self.d), dtype=np.float64)

    # @profile
    def fit(self, X, t=None):
        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)

        for l in list(reversed(self.levels)):
            q = self.levels[l]
            while len(q) != 0:
                if q[0].end+self.N <= self.time:
                    q.popleft()
                else:
                    break
            else:
                if l != 1:
                    del self.levels[l]
                    self.L = l-1

        self.B.size += X@X.T
        self.B.end = self.time
        if self.B.sketch is None:
            self.B.sketch = X
            self.B.start = self.time
        else:
            self.B.sketch = np.vstack([self.B.sketch, X])

        # or len(self.B.sketch) >= self.sketch_dim:
        if self.B.size >= self.sketch_dim:
            self.levels[1].append(self.B)
            self.B = LMBlock(None, self.time, self.time, 0.)

        for l in range(1, self.L+1):
            if len(self.levels[l]) >= self.b+1:
                if l+1 not in self.levels:
                    self.levels[l+1] = deque()
                    self.L = l+1

                B1 = self.levels[l].popleft()
                if B1.size + self.levels[l][0].size >= (2**(l+1))*self.sketch_dim:
                    self.levels[l+1].append(B1)
                else:
                    B2 = self.levels[l].popleft()
                    merged = self.__merge(B1.sketch, B2.sketch)

                    self.levels[l+1].append(LMBlock(merged, start=B1.start,
                                            end=B2.end, size=B1.size+B2.size))

    # @profile
    def get(self):
        ret = deepcopy(self.B.sketch)
        for i in range(self.L, 0, -1):
            for B in self.levels[i]:
                ret = self.__merge(ret, B.sketch)

        return ret, None, None, None

    def get_size(self):
        res = 0
        if self.B.sketch is not None:
            res += self.B.sketch.shape[0]
        for l in self.levels:
            for b in self.levels[l]:
                if b.sketch is not None:
                    res += b.sketch.shape[0]

        return res

        # return sum([b.sketch.shape[0] if b.sketch is not None else 0 for l in self.levels for b in self.levels[l]]) + self.B.sketch.shape[0] if self.B.sketch is not None else 0
