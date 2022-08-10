import numpy as np
import pytest
from scipy.stats import multivariate_normal

from distributions import Gaussian
from naive_implementations.naive_distributions import NaiveGaussian


@pytest.mark.parametrize(
    "n_dimensions",
    [34, 10, 39, 2, 1],
)
def test_gaussian_pdf(n_dimensions: int):
    mu = np.random.rand(
        n_dimensions,
    )
    covariance = np.diag(
        np.random.rand(
            n_dimensions,
        )
    )
    gaussian = Gaussian(mu, covariance)
    scipy_gaussian = multivariate_normal(mean=mu, cov=covariance)
    x = np.random.rand(
        n_dimensions,
    )
    np.testing.assert_almost_equal(gaussian.p(x), scipy_gaussian.pdf(x))


@pytest.mark.parametrize(
    "n_dimensions",
    [6, 10, 3, 2, 1],
)
def test_gaussian_dlog_p_dx(n_dimensions: int):
    mu = np.random.rand(
        n_dimensions,
    )
    covariance = np.diag(
        np.random.rand(
            n_dimensions,
        )
    )
    gaussian = Gaussian(mu, covariance)
    naive_gaussian = NaiveGaussian(mu, covariance)
    x = np.random.rand(
        n_dimensions,
    )
    np.testing.assert_allclose(
        gaussian.dlog_p_dx(x), naive_gaussian.dlog_p_dx(x), rtol=1e-4
    )
