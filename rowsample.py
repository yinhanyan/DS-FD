from dataclasses import dataclass
from typing import Deque, List
import numpy as np
import numpy.typing as npt
from collections import deque


@dataclass
class SWRSampled:
    a: npt.NDArray
    t: int
    rho: float


@dataclass
class FNorm:
    norm: float
    t: int


class SWR():
    def __init__(self, N: int, sketch_dim: int, d: int) -> None:
        self.time: int = 0
        self.N: int = N
        self.sketch_dim = sketch_dim
        self.d = d
        self.l = sketch_dim**2

        self.queues: List[Deque[SWRSampled]] = [deque()
                                                for _ in range(self.l)]
        self.norm_queue: Deque[FNorm] = deque()

    # @ profile
    def fit(self, X, t=None):
        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)

        while len(self.norm_queue) != 0:
            if self.norm_queue[0].t <= self.time - self.N:
                self.norm_queue.popleft()
            else:
                break

        norm = float(X@X.T)
        if norm < 1:
            return
        self.norm_queue.append(FNorm(norm, self.time))

        for queue in self.queues:
            while len(queue) != 0:
                if queue[0].t <= self.time - self.N:
                    queue.popleft()
                else:
                    break

            u_t = np.random.uniform()
            rho_t = u_t**(1/(norm))

            while len(queue) != 0:
                if queue[-1].rho < rho_t:
                    queue.pop()
                else:
                    break

            queue.append(SWRSampled(a=X, t=self.time, rho=rho_t))

    def get(self):
        A_F = sum(norm.norm for norm in self.norm_queue)
        stack = []
        for queue in self.queues:
            if len(queue):
                a = queue[0].a
                a = a/np.sqrt(self.l*(a@a.T)/A_F)
                stack.append(a)
        B = np.vstack(stack)
        return B, None, None, None

    def get_size(self):
        return sum([len(queue) for queue in self.queues])


@dataclass
class SWORSampled:
    a: npt.NDArray
    t: int
    rho: float
    k: int


class SWOR():
    def __init__(self, N: int, sketch_dim: int, d: int) -> None:
        self.time: int = 0
        self.N: int = N
        self.sketch_dim = sketch_dim
        self.d = d
        self.l = sketch_dim**2

        self.queue: Deque[SWORSampled] = deque()
        self.norm_queue: Deque[FNorm] = deque()

    # @ profile
    def fit(self, X, t=None):
        if t != None:
            self.time = np.uint64(t)
        else:
            self.time += np.uint64(1)

        while len(self.norm_queue) != 0:
            if self.norm_queue[0].t <= self.time - self.N:
                self.norm_queue.popleft()
            else:
                break

        norm = float(X@X.T)
        if norm < 1:
            return
        self.norm_queue.append(FNorm(norm, self.time))

        queue = self.queue
        while len(queue) != 0:
            if queue[0].t <= self.time - self.N:
                queue.popleft()
            else:
                break

        u_t = np.random.uniform()
        rho_t = u_t**(1/(norm))

        i = 0
        while i < len(queue):
            s = queue[i]
            if rho_t > s.rho:
                s.k += 1
                if s.k > self.l:
                    del queue[i]
                    i -= 1
            i += 1

        queue.append(SWORSampled(a=X, t=self.time, rho=rho_t, k=1))

    def get(self):
        A_F = sum(norm.norm for norm in self.norm_queue)
        stack = []
        queue = self.queue
        l = min(self.l, len(queue))
        for i in range(l):
            a = queue[i].a
            a = a/np.sqrt(l*(a@a.T)/A_F)
            stack.append(a)
        B = np.vstack(stack)
        return B, None, None, None

    def get_size(self):
        return len(self.queue)
