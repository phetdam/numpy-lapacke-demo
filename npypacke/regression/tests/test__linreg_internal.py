"""Tests for internal _linreg functions.

The internal C extension functions in _linreg are exposed using their
respective Python-accessible wrappers in _linreg_internal.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from .. import _linreg_internal


def test_npy_vector_matrix_mean(default_rng):
    """Test the internal npy_vector_matrix_mean function on model inputs.

    Parameters
    ----------
    default_rng : np.random.Generator
        pytest fixture. See top-level package conftest.py.
    """
    # 1D ndarray to compute the mean for
    ar_1d = default_rng.integers(0, 10, size=50)
    # 2D ndarray to compute the mean for
    ar_2d = default_rng.integers(0, 10, size=(10, 20))
    # check that 1D result is same when using ndarray.mean
    np.testing.assert_allclose(
        _linreg_internal.npy_vector_matrix_mean(ar_1d), ar_1d.mean()
    )
    # check that the 2D result is the same when using ndarray.mean
    np.testing.assert_allclose(
        _linreg_internal.npy_vector_matrix_mean(ar_2d), ar_2d.mean(axis=0)
    )
    # check that ValueError is raised when passed higher-dimension ndarray
    with pytest.raises(ValueError, match="ar must be 1D or 2D only"):
        _linreg_internal.npy_vector_matrix_mean(
            default_rng.random(size=(3, 2, 5))
        )


def test_compute_intercept_single(lr_single):
    """Test the internal compute_intercept function on single-response data.

    Parameters
    ----------
    lr_single : tuple
        pytest fixture. See local conftest.py.
    """
    # get X, y, coefficients, intercept
    X, y, coef_, intercept_, _, _ = lr_single
    # compute row-wise (axis=0) means for X, y
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    # compute the intercept and check that it is close to the true intercept
    np.testing.assert_allclose(
        _linreg_internal.compute_intercept(coef_, X_mean, y_mean),
        intercept_, rtol=1e-2
    )


def test_compute_intercept_multi(lr_multi):
    """Test the internal compute_intercept function on multi-response data.

    Parameters
    ----------
    lr_multi : tuple
        pytest fixture. See local conftest.py.
    """
    # get X, y, coefficients, intercept
    X, y, coef_, intercept_, _, _ = lr_multi
    # compute row-wise means for X< y
    X_mean = X.mean(axis=0)
    y_mean = y.mean(axis=0)
    # compute the intercept and check that it is close to the true intercept
    np.testing.assert_allclose(
        _linreg_internal.compute_intercept(coef_, X_mean, y_mean),
        intercept_, rtol=1e-2
    )


@pytest.mark.parametrize("shape", [(5,), (3, 3)])
def test_weighted_r2_edge(default_rng, shape):
    """Test the internal weighted_r2 function on some edge cases.

    Test in particular when y_true is constant and when y_pred == y_true,
    including the case where y_true is constant.

    Parameters
    ----------
    default_rng : np.random.Generator
        pytest fixture. See top-level package conftest.py.
    shape : tuple
        Shape of y_true, y_pred
    """
    # constant y_true, random y_pred
    y_true = np.ones(shape=shape)
    y_pred = default_rng.normal(size=shape)
    # returns 1 in the case y_pred and y_true are equal, even if constant
    assert _linreg_internal.weighted_r2(y_true, y_true) == 1.
    # returns -np.inf if y_true != y_pred, y_true constant
    assert _linreg_internal.weighted_r2(y_true, y_pred) == -np.inf


@pytest.mark.parametrize("shape", [(5,), (3, 3)])
def test_weighted_r2_noweight(default_rng, shape):
    """Test the internal weighted_r2 function without sample weights.

    Parameters
    ----------
    default_rng : np.random.Generator
        pytest fixture. See top-level package conftest.py.
    shape : tuple
        Shape of y_true, y_pred
    """
    # random y_true, y_pred
    y_true = default_rng.normal(size=shape)
    y_pred = default_rng.normal(size=shape)
    # compute predicted R^2 score and compare against actual. note that we
    # handle separate cases depending on the shape of y_true, y_pred
    r2_pred = _linreg_internal.weighted_r2(y_true, y_pred)
    if len(shape) == 1:
        r2_true = r2_score(y_true, y_pred)
    else:
        r2_true = r2_score(y_true[:, 0], y_pred[:, 0])
    np.testing.assert_allclose(r2_pred, r2_true)


@pytest.mark.parametrize("shape", [(5,), (3, 3)])
def test_weighted_r2_yesweight(default_rng, shape):
    """Test the internal weighted_r2 function with sample weights.

    Parameters
    ----------
    default_rng : np.random.Generator
        pytest fixture. See top-level package conftest.py.
    shape : tuple
        Shape of y_true, y_pred
    """
    # random y_true, y_pred
    y_true = default_rng.normal(size=shape)
    y_pred = default_rng.normal(size=shape)
    # random nonnegative weights
    weights = default_rng.lognormal(size=shape[0])
    # compute predicted R^2 score and compare against actual. handle shapes.
    r2_pred = _linreg_internal.weighted_r2(y_true, y_pred, weights=weights)
    if len(shape) == 1:
        r2_true = r2_score(y_true, y_pred, sample_weight=weights)
    else:
        r2_true = r2_score(y_true[:, 0], y_pred[:, 0], sample_weight=weights)
    np.testing.assert_allclose(r2_pred, r2_true)
