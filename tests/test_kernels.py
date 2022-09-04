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
from naive_implementations.naive_kernels import (
    NaiveInverseMultiQuadraticKernel,
    NaiveSteinKernel,
)


@pytest.mark.parametrize(
    "p,x,y,k",
    [
        [1, np.arange(4), np.arange(1, 5), 6],
        [3, np.array([1]), np.array([3]), 64],
        [5, np.array([3, 2, 3]), np.array([-4, 2, -1]), -134.84776],
    ],
)
def test_polynomial_kernel(p: int, x: np.ndarray, y: np.ndarray, k: float):
    kernel = PolynomialKernel(p)
    assert kernel.k(x, y) == k


@pytest.mark.parametrize(
    "p,x,y",
    [
        [1, np.arange(4), np.arange(1, 5)],
        [3, np.array([1, 1]), np.array([3, 3])],
        [5, np.array([3]), np.array([-4])],
    ],
)
def test_sklearn_polynomial_kernel(p: int, x: np.ndarray, y: np.ndarray):
    kernel = PolynomialKernel(p)
    np.testing.assert_almost_equal(
        kernel.k(x, y),
        polynomial_kernel(x.reshape(1, -1), y.reshape(1, -1), degree=p),
    )


@pytest.mark.parametrize(
    "sigma,x,y,k",
    [
        [1, np.arange(4), np.arange(1, 5), 0.01831564],
        [3, np.array([1]), np.array([3]), 6.1442124e-06],
        [5, np.array([3, 2, 3]), np.array([-4, 2, -1]), 0],
    ],
)
def test_gaussian_kernel(sigma: float, x: np.ndarray, y: np.ndarray, k: float):
    kernel = GaussianKernel(sigma)
    assert kernel.k(x, y) == k


@pytest.mark.parametrize(
    "sigma,x,y",
    [
        [1, np.arange(4), np.arange(1, 5)],
        [3, np.array([1]), np.array([3])],
        [5, np.array([3, 2, 3]), np.array([-4, 2, -1])],
    ],
)
def test_sklearn_gaussian_kernel(sigma: float, x: np.ndarray, y: np.ndarray):
    kernel = GaussianKernel(sigma)
    np.testing.assert_almost_equal(
        kernel.k(x, y),
        rbf_kernel(X=x.reshape(1, -1), Y=y.reshape(1, -1), gamma=sigma),
    )


@pytest.mark.parametrize(
    "sigma,x,y,k",
    [
        [1, np.arange(4), np.arange(1, 5), 0.01831564],
        [3, np.array([1]), np.array([3]), 0.0024787524],
        [5, np.array([3, 2, 3]), np.array([-4, 2, -1]), 1.2995815e-24],
    ],
)
def test_laplacian_kernel(sigma: float, x: np.ndarray, y: np.ndarray, k: float):
    kernel = LaplacianKernel(sigma)
    assert kernel.k(x, y) == k


@pytest.mark.parametrize(
    "sigma,x,y",
    [
        [1, np.arange(4), np.arange(1, 5)],
        [3, np.array([1]), np.array([3])],
        [5, np.array([3, 2, 3]), np.array([-4, 2, -1])],
    ],
)
def test_sklearn_laplacian_kernel(sigma: float, x: np.ndarray, y: np.ndarray):
    kernel = LaplacianKernel(sigma)
    np.testing.assert_almost_equal(
        kernel.k(x, y),
        laplacian_kernel(X=x.reshape(1, -1), Y=y.reshape(1, -1), gamma=sigma),
    )


@pytest.mark.parametrize(
    "c,beta,x,y,k",
    [
        [0.1, -0.5, np.arange(4), np.arange(1, 5), 0.49937614],
        [0.2, -0.1, np.array([1]), np.array([3]), 0.86968475],
        [0.3, -0.9, np.array([3, 2, 3]), np.array([-4, 2, -1]), 0.023325837],
    ],
)
def test_inverse_multi_quadratic_kernel(
    c: float, beta: float, x: np.ndarray, y: np.ndarray, k
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    assert kernel.k(x, y) == k


@pytest.mark.parametrize(
    "c,beta,x,y",
    [
        [0.1, -0.5, np.arange(4), np.arange(1, 5)],
        [0.2, -0.1, np.array([1]), np.array([3])],
        [0.3, -0.9, np.array([3, 2, 3]), np.array([-4, 2, -1])],
    ],
)
def test_naive_inverse_multi_quadratic_kernel(
    c: float, beta: float, x: np.ndarray, y: np.ndarray
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    np.testing.assert_almost_equal(kernel.k(x, y), naive_kernel.k(x, y))


@pytest.mark.parametrize(
    "c,beta,mu,covariance,x,y,k",
    [
        [
            0.1,
            -0.5,
            np.array([3, 2, 5, 6]),
            np.diag([5, 7, 3, 1]),
            np.arange(4).astype(float),
            np.arange(1, 5).astype(float),
            125.65655,
        ],
        [
            0.2,
            -0.1,
            np.array([1.5]),
            np.array([[0.2]]),
            np.array([-0.5]),
            np.array([3.5]),
            -0.9134231,
        ],
        [
            0.3,
            -0.9,
            np.array([2, 9, 2]),
            np.diag(np.array([1, 29, 2])),
            np.array([3, 2, 3]).astype(float),
            np.array([-4, 2, -1]).astype(float),
            -3.4014258,
        ],
    ],
)
def test_stein_kernel(
    c: float,
    beta: float,
    mu: np.ndarray,
    covariance: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k: float,
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    distribution = Gaussian(mu, covariance)
    stein_kernel = SteinKernel(distribution, kernel)
    assert stein_kernel.k(x, y) == k


@pytest.mark.parametrize(
    "c,beta,mu,covariance,x,y",
    [
        [
            100,
            -0.5,
            np.array([2, 6]),
            np.diag([7, 1]),
            np.arange(0, 2).astype(float),
            np.arange(2, 4).astype(float),
        ],
        [
            50,
            -0.3,
            np.array([1.5]),
            np.array([[0.2]]),
            np.array([-0.5]),
            np.array([3.5]),
        ],
        [
            30,
            -0.2,
            np.array([2, 9, 2]),
            np.diag(np.array([1, 29, 2])),
            np.array([3, 2, 3]).astype(float),
            np.array([-4, 2, -1]).astype(float),
        ],
    ],
)
def test_naive_stein_kernel(
    c: float,
    beta: float,
    mu: np.ndarray,
    covariance: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    distribution = Gaussian(mu, covariance)
    stein_kernel = SteinKernel(distribution, kernel)

    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    naive_distribution = NaiveGaussian(mu, covariance)
    naive_stein_kernel = NaiveSteinKernel(naive_distribution, naive_kernel)

    np.testing.assert_almost_equal(
        stein_kernel.k(x, y),
        naive_stein_kernel.k(x, y),
        decimal=5,
    )
