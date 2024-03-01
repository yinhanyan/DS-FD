# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import numpy.typing as npt
from scipy import linalg


class FrequentDirections:
    def __init__(self, d, sketch_dim=8):
        """
        Class wrapper for all FD-type methods

        __rotate_and_reduce__ is not defined for the standard FrequentDirections but is for the
        subsequent subclasses which inherit from FrequentDirections.
        """
        self.d = d
        self.delta = 0.0  # For RFD

        if sketch_dim is not None:
            self.sketch_dim = sketch_dim
        self.sketch = np.zeros((self.sketch_dim, self.d), dtype=float)
        self.Vt = np.zeros((self.sketch_dim, self.d), dtype=float)
        self.sigma_squared = np.zeros(self.sketch_dim, dtype=float)

    # @profile
    def fit(self, X, batch_size=1):
        """
        Fits the FD transform to dataset X
        """
        n = X.shape[0]
        for i in range(0, n, batch_size):
            aux = np.zeros((self.sketch_dim + batch_size, self.d))
            batch = X[i : i + batch_size, :]
            # aux = np.concatenate((self.sketch, batch), axis=0)
            aux[0 : self.sketch_dim, :] = self.sketch
            aux[self.sketch_dim : self.sketch_dim + batch.shape[0], :] = batch
            # ! WARNING - SCIPY SEEMS MORE ROBUST THAN NUMPY SO COMMENTING THIS WHICH IS FASTER OVERALL
            # try:
            #     _, s, self.Vt = np.linalg.svd(aux, full_matrices=False)
            # except np.linalg.LinAlgError:
            #     _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver='gesvd')
            _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver="gesvd")
            self.sigma_squared = s**2
            self.__rotate_and_reduce__()
            self.sketch = self.Vt * np.sqrt(self.sigma_squared).reshape(-1, 1)

    def get(self):
        return self.sketch, self.sigma_squared, self.Vt, self.delta

    def get_sketch(self):
        return self.sketch


class FastFrequentDirections(FrequentDirections):
    """
    Implements the fast version of FD by doubling space
    """

    def __rotate_and_reduce__(self):
        self.sigma_squared = (
            self.sigma_squared[: self.sketch_dim] - self.sigma_squared[self.sketch_dim]
        )
        self.Vt = self.Vt[: self.sketch_dim]


class RobustFrequentDirections(FrequentDirections):
    """
    Implements the RFD version of FD by maintaining counter self.delta.
    Still operates in the `fast` regimen by doubling space, as in
    FastFrequentDirections
    """

    def __rotate_and_reduce__(self):
        if len(self.sigma_squared) > self.sketch_dim:
            self.delta += self.sigma_squared[self.sketch_dim] / 2.0
            self.sigma_squared = (
                self.sigma_squared[: self.sketch_dim]
                - self.sigma_squared[self.sketch_dim]
            )
            self.Vt = self.Vt[: self.sketch_dim]


class FrequentDirectionsWithDump(RobustFrequentDirections):
    def __init__(self, d: int, sketch_dim: int, error: float):
        super().__init__(d, min(sketch_dim, d))
        self.max_energy: float = 0.0
        self.buffer = None
        self.error: float = error

    def __flush(self):
        if self.buffer is not None:
            super().fit(
                self.buffer, batch_size=min(self.buffer.shape[0], self.sketch_dim)
            )
            self.max_energy = self.sigma_squared[0]
            self.buffer = None

    def get_error(self) -> float:
        return self.error

    # @profile
    def fit(self, X, batch_size=1):
        self.max_energy += X @ X.T
        if self.buffer is None:
            self.buffer = X
        else:
            self.buffer = np.concatenate([self.buffer, X])

        if self.buffer is not None and len(self.buffer) >= self.sketch_dim:
            self.__flush()
        elif self.max_energy >= self.error:
            self.__flush()

    def dump(self) -> npt.NDArray:
        if self.sigma_squared[0] >= self.error:
            v = np.sqrt(self.sigma_squared[0]) * self.Vt[0:1]
            self.sketch[0, :] = 0
            self.sigma_squared[0] = 0
            self.Vt[0, :] = 0
            np.roll(self.sketch, -1)
            np.roll(self.sigma_squared, -1)
            np.roll(self.Vt, -1)
            self.max_energy = self.sigma_squared[0]
            # self.sketch[:, :] = 0
            # self.sigma_squared[:] = 0
            # self.Vt[:, :] = 0
            return v
        else:
            return None

    def get(self):
        self.__flush()
        return super().get()

    def get_sketch(self):
        self.__flush()
        return self.sketch

    def flush(self):
        self.__flush()

    # def __rotate_and_reduce__(self):
    #     # self.delta += self.sigma_squared[self.sketch_dim] / 2.
    #     self.sigma_squared = self.sigma_squared[:self.sketch_dim]
    #     # self.sigma_squared[self.sketch_dim]
    #     self.Vt = self.Vt[:self.sketch_dim]
