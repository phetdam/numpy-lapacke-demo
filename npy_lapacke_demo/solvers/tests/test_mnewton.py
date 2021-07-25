"""Tests for the mnewton function provided by _mnewton.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from functools import partial
import numpy as np
import pytest
import scipy.linalg

# pylint: disable=no-name-in-module
from .._mnewton import mnewton


def test_mnewton_sanity(qp_noargs):
    """Test input checking sanity of mnewton.

    For testing that the input checks work as intended.

    Parameters
    ----------
    qp_noargs : tuple
        pytest fixture. See local conftest.py.
    """
    # get objective, initial guess, gradient, Hessian from qp_noargs
    f_obj, x0, f_grad, f_hess = qp_noargs
    # objective must be callable
    with pytest.raises(TypeError, match="fun must be callable"):
        mnewton("not callable", x0)
    # x0 must be convertible to ndarray of type double. PyArray_FROM_OTF will
    # raise a ValueError here and has a message saying "could not convert."
    with pytest.raises(ValueError, match="could not convert"):
        mnewton(f_obj, "not convertible to type double")
    # can't have empty x0
    with pytest.raises(ValueError, match="x0 must be nonempty"):
        mnewton(f_obj, np.empty(0))
    # x0 must be 1D ndarray
    with pytest.raises(ValueError, match=r"x0 must have shape \(n_features,\)"):
        mnewton(f_obj, np.array([[1, 2], [3, 4]]))
    # args must be tuple if provided
    with pytest.raises(TypeError, match="tuple"):
        mnewton(f_obj, x0, args=[])
    # pytest.raises context for validating jac
    jac_raises = partial(
        pytest.raises, TypeError, match="jac must be callable or True"
    )
    # jac must be provided and must be callable or True
    with jac_raises():
        mnewton(f_obj, x0)
    with jac_raises():
        mnewton(f_obj, x0, jac="not callable")
    with jac_raises():
        mnewton(f_obj, x0, jac=False)
    # pytest.raises context for validating hess
    hess_raises = partial(
        pytest.raises, TypeError, match="hess must be provided and be callable"
    )
    # hess must be provided and must be callable
    with hess_raises():
        mnewton(f_obj, x0, jac=f_grad)
    with hess_raises():
        mnewton(f_obj, x0, jac=f_grad, hess="not callable")
    # wrapped pytest.raises context for validating gtol, maxiter, beta positive
    pos_raises = lambda x: pytest.raises(
        ValueError, match=rf"{x} must be positive"
    )
    # gtol, maxiter, beta must be positive if provided
    with pos_raises("gtol"):
        mnewton(f_obj, x0, jac=f_grad, hess=f_hess, gtol=0)
    with pos_raises("maxiter"):
        mnewton(f_obj, x0, jac=f_grad, hess=f_hess, maxiter=0)
    with pos_raises("beta"):
        mnewton(f_obj, x0, jac=f_grad, hess=f_hess, beta=0)
    # wrapped pytest.raises context for validating alpha, gamma in (0, 1)
    unit_raises = lambda x: pytest.raises(
        ValueError, match=rf"{x} must be in \(0, 1\)"
    )
    # alpha and gamma must be in (0, 1)
    with unit_raises("alpha"):
        mnewton(f_obj, x0, jac=f_grad, hess=f_hess, alpha=0)
    with unit_raises("alpha"):
        mnewton(f_obj, x0, jac=f_grad, hess=f_hess, alpha=1)
    with unit_raises("gamma"):
        mnewton(f_obj, x0, jac=f_grad, hess=f_hess, gamma=0)
    with unit_raises("gamma"):
        mnewton(f_obj, x0, jac=f_grad, hess=f_hess, gamma=1)
    # tau_factor must be 2 or greater
    with pytest.raises(ValueError, match=r"tau_factor must be greater than 1"):
        mnewton(f_obj, x0, jac=f_grad, hess=f_hess, tau_factor=1)


def test_mnewton_qp_noargs(qp_hess_a, qp_noargs):
    """Test mnewton on convex unconstrained QP with no function arguments.

    Parameters
    ----------
    qp_hess_a : tuple
        pytest fixture. See local conftest.py.
    qp_noargs : tuple
        pytest fixture. See local conftest.py.
    """
    # get f_obj, x0, f_grad, f_hess from qp_noargs
    f_obj, x0, f_grad, f_hess = qp_noargs
    # get Hessian and linear terms from qp_hess_a + compute optimal point
    hess, a, _ = qp_hess_a
    qp_sol = scipy.linalg.solve(hess, -a, assume_a="pos")
    # call mnewton and check that solution is close to qp_sol
    res = mnewton(f_obj, x0, jac=f_grad, hess=f_hess, gtol=1e-6)
    np.testing.assert_allclose(res.x, qp_sol)


def test_mnewton_qp_yesargs(qp_hess_a, qp_yesargs):
    """Test mnewton on convex unconstrained QP with function arguments.

    Parameters
    ----------
    qp_hess_a : tuple
        pytest fixture. See local conftest.py.
    qp_yesargs : tuple
        pytest fixture. See local conftest.py.
    """
    # get f_obj, x0, f_grad, f_hess, f_args from qp_noargs
    f_obj, x0, f_grad, f_hess, f_args = qp_yesargs
    # get Hessian and linear terms from qp_hess_a + compute optimal point
    hess, a, _ = qp_hess_a
    qp_sol = scipy.linalg.solve(hess, -a, assume_a="pos")
    # call mnewton and check that solution is close to qp_sol
    res = mnewton(f_obj, x0, args=f_args, jac=f_grad, hess=f_hess, gtol=1e-6)
    np.testing.assert_allclose(res.x, qp_sol)