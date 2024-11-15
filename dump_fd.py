import numpy as np
import scipy.linalg as linalg
import numpy.typing as npt
from abc import ABC, abstractmethod
import heapq
from typing import List, Tuple


class DumpFdBase(ABC):
    @abstractmethod
    def fit(self, X: npt.NDArray) -> npt.NDArray:
        pass

    @abstractmethod
    def get_sketch(self) -> npt.NDArray:
        pass


class OptDumpFd(DumpFdBase):
    def __init__(self, d: int, sketch_dim: int, error: float):
        self.d: int = d
        self.sketch: npt.NDArray = np.zeros((2 * sketch_dim, d))
        self.sketch_dim: int = sketch_dim
        self.max_heap: List[Tuple[float, int]] = []
        self.error: float = error
        self.row: int = 0

        self.delta: float = 0.0

    def fit(self, X: npt.NDArray) -> npt.NDArray | None:
        norm = linalg.norm(X) ** 2
        if norm > self.error:
            ret = X
        else:
            if self.max_heap:
                neg_norm, ind_max = self.max_heap[0]
                norm_max = -neg_norm
                if norm + norm_max > self.error:
                    Xv = np.vstack((X, self.sketch[ind_max]))
                    _, S, Vt = linalg.svd(Xv, full_matrices=False, overwrite_a=True)
                    if S[0] ** 2 > self.error:
                        heapq.heappop(self.max_heap)
                        self.sketch[ind_max] = S[1] * Vt[1]
                        heapq.heappush(self.max_heap, (-S[1] ** 2, ind_max))
                        ret = S[0] * Vt[0]
                    else:
                        self.sketch[self.row] = X
                        heapq.heappush(self.max_heap, (-norm, self.row))
                        self.row += 1
                        ret = None
                else:
                    self.sketch[self.row] = X
                    heapq.heappush(self.max_heap, (-norm, self.row))
                    self.row += 1
                    ret = None
            else:
                self.sketch[self.row] = X
                heapq.heappush(self.max_heap, (-norm, self.row))
                self.row += 1
                ret = None

            if self.row >= 2 * self.sketch_dim:
                _, S, Vt = linalg.svd(
                    self.sketch, full_matrices=False, overwrite_a=True
                )
                sigma_squared = S**2
                if len(sigma_squared) > self.sketch_dim:
                    self.delta += sigma_squared[self.sketch_dim] / 2.0
                    sigma_squared = (
                        sigma_squared[: self.sketch_dim]
                        - sigma_squared[self.sketch_dim]
                    )
                    Vt = Vt[: self.sketch_dim]

                l = len(sigma_squared)
                self.sketch[:l] = Vt * np.sqrt(sigma_squared).reshape(-1, 1)
                self.sketch[l:] = 0.0
                self.row = l

                self.max_heap = list(zip(-sigma_squared, range(l)))

        return ret

    def get_sketch(self) -> npt.NDArray:
        return self.sketch
