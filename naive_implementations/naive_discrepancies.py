import numpy as np

from distributions import BaseDistribution
from kernels import BaseKernel, SteinKernel


def naive_mmd(kernel: BaseKernel, x: np.ndarray, y: np.ndarray) -> float:
    m, n = x.shape[0], y.shape[0]

    xx = np.array(
        [[kernel.k(x[i, :].T, x[j, :].T) for j in range(m) if i != j] for i in range(m)]
    )
    xy = np.array(
        [[kernel.k(x[i, :].T, y[j, :].T) for j in range(n)] for i in range(m)]
    )
    yy = np.array(
        [[kernel.k(y[i, :].T, y[j, :].T) for j in range(n) if i != j] for i in range(n)]
    )
    return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)


def naive_ksd(stein_kernel: SteinKernel, x: np.ndarray) -> float:
    n = x.shape[0]
    return np.mean(
        np.array(
            [
                [stein_kernel.k(x[i, :].T, x[j, :].T) for j in range(n) if i != j]
                for i in range(n)
            ]
        )
    )


def naive_fisher_divergence(p: BaseDistribution, x: np.ndarray) -> float:
    d = x.shape[1]
    n = x.shape[0]
    return np.mean(
        np.array(
            [
                np.sum(
                    np.diag(p.d_score_dx(x[i, :]).reshape(d, d))
                    + 0.5 * np.square(p.score(x[i, :]))
                )
                for i in range(n)
            ]
        )
    )
