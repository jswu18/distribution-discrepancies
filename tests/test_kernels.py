import numpy as np
import pytest
from sklearn.metrics.pairwise import laplacian_kernel, polynomial_kernel, rbf_kernel

from distributions import Gaussian
from kernels import (
    GaussianKernel,
    InverseMultiQuadraticKernel,
    LaplacianKernel,
    PolynomialKernel,
    SteinKernel,
)
from naive_implementations.naive_distributions import NaiveGaussian
from naive_implementations.naive_kernels import NaiveInverseMultiQuadraticKernel


@pytest.mark.parametrize(
    "p,n_dimensions",
    [
        [1, 2],
        [2, 3],
        [3, 5],
    ],
)
def test_polynomial_kernel(p: int, n_dimensions: int):
    x = np.random.rand(
        n_dimensions,
    )
    y = np.random.rand(
        n_dimensions,
    )
    kernel = PolynomialKernel(p)
    np.testing.assert_allclose(
        kernel.k(x, y),
        polynomial_kernel(X=x.reshape(1, -1), Y=y.reshape(1, -1), degree=p),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "sigma,n_dimensions",
    [
        [0.1, 2],
        [0.2, 3],
        [0.3, 1],
    ],
)
def test_gaussian_kernel(sigma: float, n_dimensions: int):
    x = np.random.rand(
        n_dimensions,
    )
    y = np.random.rand(
        n_dimensions,
    )
    kernel = GaussianKernel(sigma)
    np.testing.assert_allclose(
        kernel.k(x, y),
        rbf_kernel(X=x.reshape(1, -1), Y=y.reshape(1, -1), gamma=sigma),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "sigma,n_dimensions",
    [
        [0.1, 2],
        [0.2, 3],
        [0.3, 5],
    ],
)
def test_laplacian_kernel(sigma: float, n_dimensions: int):
    x = np.random.rand(
        n_dimensions,
    )
    y = np.random.rand(
        n_dimensions,
    )
    kernel = LaplacianKernel(sigma)
    np.testing.assert_allclose(
        kernel.k(x, y),
        laplacian_kernel(X=x.reshape(1, -1), Y=y.reshape(1, -1), gamma=sigma),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "c,beta,n_dimensions",
    [
        [0.1, -0.5, 2],
        [0.2, -0.1, 3],
        [0.3, -0.2, 5],
    ],
)
def test_inverse_multi_quadratic_kernel(
    c: float,
    beta: float,
    n_dimensions: int,
):
    x = np.random.rand(
        n_dimensions,
    )
    y = np.random.rand(
        n_dimensions,
    )
    kernel = InverseMultiQuadraticKernel(c, beta)
    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    np.testing.assert_allclose(kernel.k(x, y), naive_kernel.k(x, y), rtol=1e-6)


@pytest.mark.parametrize(
    "c,beta,n_dimensions",
    [
        [0.1, -0.5, 2],
        [0.2, -0.1, 3],
        [0.3, -0.2, 5],
    ],
)
def test_stein_kernel(
    c: float,
    beta: float,
    n_dimensions: int,
):
    mu = np.random.rand(
        n_dimensions,
    )
    covariance = np.diag(
        np.random.rand(
            n_dimensions,
        )
    )

    kernel = InverseMultiQuadraticKernel(c, beta)
    distribution = Gaussian(mu, covariance)
    stein_kernel = SteinKernel(distribution, kernel)

    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    naive_distribution = NaiveGaussian(mu, covariance)
    naive_stein_kernel = SteinKernel(naive_distribution, naive_kernel)

    x = np.random.rand(
        n_dimensions,
    )
    y = np.random.rand(
        n_dimensions,
    )
    np.testing.assert_allclose(
        stein_kernel.k(x, y), naive_stein_kernel.k(x, y), rtol=1e-5
    )
