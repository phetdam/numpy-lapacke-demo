__doc__ = """pytest test fixtures for regression.tests subpackage.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest
from sklearn.datasets import make_regression

@pytest.fixture(scope="session")
def lr_single(global_seed):
    """Toy single-output linear regression problem.

    Problem has 100 data points with 3 features and a single output. All
    features are informative and are standard Gaussian, identity covariance.
    Standard deviation of the Gaussian noise applied to the response is 0.1.
    The intercept applied to each response is 7.

    Parameters
    ----------
    global_seed : int
        pytest fixture. See top-level package conftest.py.

    Returns
    -------
    X : numpy.ndarray
        Input matrix shape (100, 3)
    y : numpy.ndarray
        Response vector shape (100,)
    coef : numpy.ndarray
        True linear coefficients, shape (3,)
    bias : float
        The bias term of the model
    """
    bias = 7
    X, y, coef = make_regression(
        n_features=3, n_informative=3, bias=bias, noise=0.1,
        coef=True, random_state=global_seed
    )
    return X, y, coef, bias


@pytest.fixture(scope="session")
def lr_multi(global_seed):
    """Toy multi-output linear regression problem.

    Problem has 100 data points with 3 features and 2 outputs. All features
    are informative and are standard Gaussian, identity covariance. Standard
    deviation of the Gaussian noise applied to the response is 0.1.
    The intercept applied to each response is 7.

    Parameters
    ----------
    global_seed : int
        pytest fixture. See top-level package conftest.py.

    Returns
    -------
    X : numpy.ndarray
        Input matrix shape (100, 3)
    y : numpy.ndarray
        Response vector shape (100, 2)
    coef : numpy.ndarray
        True linear coefficients, shape (2, 3), i.e. (n_targets, n_features).
        Transpoe the output from sklearn.datasets.make_regression in order to
        allow comparing with LinearRegression.coef_ without transpose.
    bias : float
        The bias term of the model
    """
    bias = 7
    X, y, coef = make_regression(
        n_features=3, n_informative=3, n_targets=2, bias=bias,
        noise=0.1, coef=True, random_state=global_seed
    )
    return X, y, coef.T, bias