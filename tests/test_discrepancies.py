import numpy as np
import pytest

from discrepancies import (
    FisherDivergence,
    KernelSteinDiscrepancy,
    MaximumMeanDiscrepancy,
)
from distributions import BaseDistribution, Gaussian
from kernels import (
    BaseKernel,
    GaussianKernel,
    InverseMultiQuadraticKernel,
    LaplacianKernel,
    PolynomialKernel,
    SteinKernel,
)
from naive_implementations.naive_discrepancies import (
    naive_fisher_divergence,
    naive_ksd,
    naive_mmd,
)


@pytest.mark.parametrize(
    "kernel,x,y,mmd_val",
    [
        [
            PolynomialKernel(p=2),
            np.array([[2, 3, 1], [3, 1, 5]]),
            np.array([[1, 0, 4], [3, 6, 1]]),
            -71.77779,
        ],
        [
            GaussianKernel(sigma=0.1),
            np.array([[2, 3], [3, 5], [4, 6]]),
            np.array([[1, 0], [3, 6], [1, 0]]),
            0.23467976,
        ],
        [
            LaplacianKernel(sigma=0.1),
            np.array([[2, 3], [3, 5]]),
            np.array([[1, 0], [3, 6]]),
            -0.18088424,
        ],
        [
            InverseMultiQuadraticKernel(c=0.1, beta=-0.4),
            np.array([[2, 1], [3, 5]]),
            np.array([[1, 0], [3, 6]]),
            -0.59146684,
        ],
    ],
)
def test_mmd(kernel: BaseKernel, x: np.ndarray, y: np.ndarray, mmd_val: float):
    mmd = MaximumMeanDiscrepancy(kernel)
    assert mmd.compute(x, y) == mmd_val


@pytest.mark.parametrize(
    "kernel,x,y,t,f_witness_x",
    [
        [
            GaussianKernel(sigma=0.1),
            np.array([[2, 3], [3, 5], [4, 6]]),
            np.array([[1, 0], [3, 6], [1, 0]]),
            np.array([[-1, -1], [0, 0], [1, 1]]),
            np.array([-0.375450656, -0.503120768, -0.363152992]),
        ],
        [
            LaplacianKernel(sigma=0.1),
            np.array([[2, 3], [3, 5]]),
            np.array([[1, 0], [3, 6]]),
            np.array([[-1, -1], [0, 0]]),
            np.array([-0.104612232, -0.127773760]),
        ],
        [
            InverseMultiQuadraticKernel(c=0.1, beta=-0.4),
            np.array([[2, 1], [3, 5]]),
            np.array([[1, 0], [3, 6]]),
            np.array([[-1, -1], [0, 0], [1, 1]]),
            np.array([-0.074487896, -0.222635264, 0.020822048]),
        ],
    ],
)
def test_mmd_witness_function(
    kernel: BaseKernel,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    f_witness_x: np.ndarray,
):
    mmd = MaximumMeanDiscrepancy(kernel)
    np.testing.assert_almost_equal(mmd.witness_function(x, y, t), f_witness_x)


@pytest.mark.parametrize(
    "kernel,n_dimensions,x_n_samples,y_n_samples",
    [
        [PolynomialKernel(p=2), 1, 3, 2],
        [GaussianKernel(sigma=0.1), 2, 3, 10],
        [LaplacianKernel(sigma=0.1), 3, 4, 2],
        [InverseMultiQuadraticKernel(c=0.1, beta=-0.4), 2, 2, 2],
    ],
)
def test_naive_mmd(kernel: BaseKernel, n_dimensions: int, x_n_samples, y_n_samples):
    np.random.seed(0)
    x = np.random.rand(
        x_n_samples,
        n_dimensions,
    )
    y = np.random.rand(
        y_n_samples,
        n_dimensions,
    )
    mmd = MaximumMeanDiscrepancy(kernel)
    np.testing.assert_almost_equal(
        mmd.compute(x, y), naive_mmd(kernel, x, y), decimal=6
    )


@pytest.mark.parametrize(
    "kernel,mu,covariance,x, ksd_val",
    [
        [
            PolynomialKernel(p=2),
            np.array([4, 3]).astype(float),
            np.array([[1, 0], [0, 1]]).astype(float),
            np.array([[2, 3], [1, 5]]).astype(float),
            540.5,
        ],
        [
            GaussianKernel(sigma=0.1),
            np.array([4, 3]).astype(float),
            np.array([[1, 0.1], [0, 1]]).astype(float),
            np.array([[2, 3], [3, 5], [4, 6]]).astype(float),
            1.8579469,
        ],
        [
            LaplacianKernel(sigma=0.1),
            np.array([4, 3]).astype(float),
            np.array([[1, 0.1], [0, 1]]).astype(float),
            np.array([[2, 3], [3, 5]]).astype(float),
            1.5557182,
        ],
        [
            InverseMultiQuadraticKernel(c=0.1, beta=-0.4),
            np.array([4]).astype(float),
            np.array([[1]]).astype(float),
            np.array([[2], [3]]).astype(float),
            -0.19508517,
        ],
    ],
)
def test_ksd(
    kernel: BaseKernel, mu: np.ndarray, covariance: np.ndarray, x: np.ndarray, ksd_val
):
    np.random.seed(0)
    gaussian = Gaussian(
        mu=mu,
        covariance=covariance,
    )
    stein_kernel = SteinKernel(
        kernel=kernel,
        distribution=gaussian,
    )
    ksd = KernelSteinDiscrepancy(stein_kernel)
    assert ksd.compute(x) == ksd_val


@pytest.mark.parametrize(
    "kernel,p,x,t,f_witness_x",
    [
        [
            GaussianKernel(sigma=0.1),
            Gaussian(
                mu=np.zeros(
                    2,
                ),
                covariance=np.eye(2),
            ),
            np.array([[2, 3], [3, 5], [4, 6]]).astype(float),
            np.array([[-1, -1], [0, 0], [1, 1]]).astype(float),
            np.array([0.3325325, 0.3556344, -1.036464]),
        ],
        [
            LaplacianKernel(sigma=0.1),
            Gaussian(
                mu=np.zeros(
                    2,
                ),
                covariance=np.eye(2),
            ),
            np.array([[2, 3], [3, 5]]).astype(float),
            np.array([[-1, -1], [0, 0]]).astype(float),
            np.array([3.0793705, 0.34192288]),
        ],
        [
            InverseMultiQuadraticKernel(c=0.1, beta=-0.4),
            Gaussian(
                mu=np.zeros(
                    2,
                ),
                covariance=np.eye(2),
            ),
            np.array([[2, 1], [3, 5]]).astype(float),
            np.array([[-1, -1], [0, 0], [1, 1]]).astype(float),
            np.array([1.596536, 0.34268397, -1.8761171]),
        ],
    ],
)
def test_ksd_witness_function(
    kernel: BaseKernel,
    p: BaseDistribution,
    x: np.ndarray,
    t: np.ndarray,
    f_witness_x: np.ndarray,
):
    ksd = KernelSteinDiscrepancy(
        stein_kernel=SteinKernel(distribution=p, kernel=kernel)
    )
    np.testing.assert_almost_equal(ksd.witness_function(x, t), f_witness_x)


@pytest.mark.parametrize(
    "kernel,n_dimensions,n_samples",
    [
        [PolynomialKernel(p=3), 1, 2],
        [GaussianKernel(sigma=0.1), 2, 3],
        [LaplacianKernel(sigma=0.1), 3, 6],
        [InverseMultiQuadraticKernel(c=0.1, beta=-0.4), 2, 4],
    ],
)
def test_naive_ksd(kernel: BaseKernel, n_dimensions: int, n_samples: int):
    np.random.seed(0)
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
    ksd = KernelSteinDiscrepancy(stein_kernel)
    np.testing.assert_almost_equal(ksd.compute(x), naive_ksd(stein_kernel, x))


@pytest.mark.parametrize(
    "n_dimensions,n_samples",
    [
        [1, 2],
        [2, 3],
        [3, 6],
        [2, 4],
    ],
)
def test_naive_fisher_divergence(n_dimensions: int, n_samples: int):
    np.random.seed(0)
    gaussian = Gaussian(
        mu=np.random.rand(
            n_dimensions,
        ),
        covariance=np.diag(np.random.rand(n_dimensions)),
    )
    x = gaussian.sample(n_samples)
    fisher_divergence = FisherDivergence(gaussian)
    np.testing.assert_almost_equal(
        fisher_divergence.compute(x), naive_fisher_divergence(gaussian, x), decimal=6
    )
