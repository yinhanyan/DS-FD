import numpy as np
import numpy.typing as npt
from scipy import linalg
from fbpca import pca, eigenn


class FastFrequentDirectionsWithDump:
    def __init__(self, d: int, sketch_dim: int, error: float, approx: bool = False):
        self.d = d
        self.sketch_dim = sketch_dim

        self.sketch = np.zeros((self.sketch_dim, self.d), dtype=float)
        self.sigma_squared = np.zeros(self.sketch_dim, dtype=float)

        self.max_energy: float = 0.0
        self.error: float = error
        self.cov = np.zeros((self.sketch_dim, self.sketch_dim), dtype=float)
        self.approx = approx

    # @profile
    def __flush(self):
        sketch = self.sketch[:-1, :]
        batch = self.sketch[-1:, :]
        upper_right = sketch @ batch.T
        lower_left = upper_right.T
        lower_right = batch @ batch.T
        self.cov = np.block([[self.cov, upper_right], [lower_left, lower_right]])
        # gap = np.max(np.abs(self.sketch@self.sketch.T-self.cov))
        # ret = []

        # while self.max_energy >= self.error:
            # U, s, _ = linalg.svd(self.cov, overwrite_a=False, lapack_driver="gesdd")
            # U, s, _ = pca(self.cov, k=1)
            # self.max_energy = float(sigma_squared[0])

        # s, U = eigenn(self.cov, k=1)
        if self.max_energy >= self.error:
            # U, sigma_squared, _ = pca(self.cov, k=1)
            U, sigma_squared, _ = linalg.svd(self.cov, overwrite_a=False, lapack_driver="gesvd")
            # U = U[:, 0:1]
            # sigma_squared = s**2
            i = 0
            while i<len(sigma_squared):
                self.max_energy = float(sigma_squared[i])
                if sigma_squared[i] < self.error:
                    break
                i += 1
            # while sigma_squared[0] >= self.error:
            if i == 0:
                return None

            sigma_vt = U[:, :i].T @ self.sketch
            # ret.append(sigma_vt)
            vt = sigma_vt / np.sqrt(sigma_squared[:i])[:,None]
            # vt = sigma_vt / linalg.norm(sigma_vt)
            # if np.abs(linalg.norm(vt) - 1) >0.1:
            #     raise Exception('error')
            Dv = self.sketch @ vt.T
            self.sketch = self.sketch - Dv @ vt
            self.cov = self.cov - Dv @ Dv.T
                # self.max_energy = float(sigma_squared[1])
                # ret = np.vstack([ret, sigma_vt])

                # s, U = eigenn(self.cov, k=1)
                # U, sigma_squared, _ = pca(self.cov, k=1)
                # U, sigma_squared, _ = linalg.svd(self.cov, overwrite_a=False, lapack_driver="gesvd")
                # U = U[:, 0:1]
                # sigma_squared = s**2
                # self.max_energy = float(sigma_squared[0])
                # return sigma_vt
            # else:
                # U, sigma_squared, _ = pca(self.cov, k=1)
                # U, sigma_squared, _ = linalg.svd(self.cov, overwrite_a=False, lapack_driver="gesdd")
                # U = U[:, 0:1]
                # self.max_energy = float(sigma_squared[0])
                # return None
            return sigma_vt
        
        # if len(ret) == 0:
        #     return None
        # else:
        #     return np.vstack(ret)
        return None
        


    def get_error(self) -> float:
        return self.error

    # @profile
    def fit(self, X):
        """
        Fits the FD transform to dataset X
        """
        self.max_energy += float(X @ X.T)
        self.sketch = np.vstack([self.sketch, X])

        if len(self.sketch) >= 2 * self.sketch_dim:
            _, s, Vt = linalg.svd(self.sketch, overwrite_a=True, lapack_driver="gesvd")
            self.sigma_squared = s**2
            if len(self.sigma_squared) > self.sketch_dim:
                self.sigma_squared = (
                    self.sigma_squared[: self.sketch_dim]
                    - self.sigma_squared[self.sketch_dim]
                )
                Vt = Vt[: self.sketch_dim]
            self.sketch = Vt * np.sqrt(self.sigma_squared).reshape(-1, 1)
            # self.cov = self.sketch @ self.sketch.T
            ret = self.dump()
        else:
            ret = self.__flush()
        
        return ret

    # @profile
    def dump(self) -> npt.NDArray:
        idx = np.where(self.sigma_squared >= self.error)
        if len(idx[0]) == 0:
            ret = None
        else:
            ret = self.sketch[idx]
            self.sketch[idx, :] = 0.
            self.sigma_squared[idx] = 0.
            # np.roll(self.sketch, -1)
            # np.roll(self.sigma_squared, -1)
            self.max_energy = float(max(self.sigma_squared))
            # self.sketch[:, :] = 0
            # self.sigma_squared[:] = 0
            # self.Vt[:, :] = 0

        self.cov = self.sketch @ self.sketch.T
        return ret

    def get(self):
        return self.sketch, None, None, None

    def get_sketch(self):
        return self.sketch
