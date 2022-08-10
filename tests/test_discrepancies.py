import numpy as np
import pytest

from discrepancies import ksd, mmd
from distributions import Gaussian
from kernels import (
    BaseKernel,
    GaussianKernel,
    InverseMultiQuadraticKernel,
    LaplacianKernel,
    PolynomialKernel,
    SteinKernel,
)
from naive_implementations.naive_discrepancies import naive_ksd, naive_mmd


@pytest.mark.parametrize(
    "kernel,n_dimensions,x_n_samples,y_n_samples",
    [
        [PolynomialKernel(p=2), 1, 20, 34],
        [GaussianKernel(sigma=0.1), 2, 3, 10],
        [LaplacianKernel(sigma=0.1), 3, 31, 39],
        [InverseMultiQuadraticKernel(c=0.1, beta=-0.4), 2, 10, 10],
    ],
)
def test_mmd(kernel: BaseKernel, n_dimensions: int, x_n_samples, y_n_samples):
    x = np.random.rand(
        x_n_samples,
        n_dimensions,
    )
    y = np.random.rand(
        y_n_samples,
        n_dimensions,
    )
    np.testing.assert_allclose(mmd(kernel, x, y), naive_mmd(kernel, x, y), rtol=1e-02)


@pytest.mark.parametrize(
    "kernel,n_dimensions,n_samples",
    [
        [PolynomialKernel(p=3), 1, 5],
        [GaussianKernel(sigma=0.1), 2, 3],
        [LaplacianKernel(sigma=0.1), 3, 6],
        [InverseMultiQuadraticKernel(c=0.1, beta=-0.4), 2, 4],
    ],
)
def test_ksd(kernel: BaseKernel, n_dimensions: int, n_samples: int):
    gaussian = Gaussian(
        mu=np.random.rand(
            n_dimensions,
        ),
        covariance=np.diag(np.random.rand(n_dimensions)),
    )
    stein_kernel = SteinKernel(
        kernel=kernel,
        distribution=gaussian,
    )
    x = stein_kernel.distribution.sample(n_samples)
    np.testing.assert_allclose(
        naive_ksd(stein_kernel, x), ksd(stein_kernel, x), rtol=1e-05
    )
