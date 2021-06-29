__doc__ = """pytest test fixtures for solvers.tests subpackage.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest
from sklearn.datasets import make_spd_matrix


@pytest.fixture(scope="session")
def kwargs_keylist():
    """A list of strings to drop/keep in a kwargs dict.

    Returns
    -------
    list
    """
    # see the lyrics of Lost One's Weeping by Neru
    return ["oi", "dare", "nan", "da", "yo"]


@pytest.fixture
def empty_kwargs(kwargs_keylist):
    """An empty kwargs dict with non-empty list of string keys to drop.

    Parameters
    ----------
    kwargs_keylist : list
        pytest fixture. See local conftest.py.

    Returns
    -------
    kwargs : dict
        An empty dict.
    kwargs_keylist : list
        List of string keys to drop.
    """
    return {}, kwargs_keylist


@pytest.fixture
def full_kwargs(kwargs_keylist):
    """A nonempty kwargs dict with non-empty list of string keys to drop.

    Parameters
    ----------
    kwargs_keylist : list
        pytest fixture. See local conftest.py.

    Returns
    -------
    kwargs : dict
        An nonempty dict with more keys than those provided in kwargs_keylist
        but containing only a subset of the keys in kwargs_keylist.
    kwargs_keylist : list
        List of string keys to drop.
    """
    # only oi, dare, da are names in kwargs_keylist. both
    # EXPOSED_remove_specified_kwargs, EXPOSED_remove_unspecified_kwargs with
    # kwargs_keylist as the droplist/keeplist will raise warnings if warn=True.
    # for drop4, drop5, see the lyrics of Aspirin by MuryokuP
    kwargs = dict(
        oi="yo", dare="da", da="dare", drop1="oh no", drop2="we shouldn't",
        drop3="be here", drop4="tooku no kage", drop5="ao no senaka"
    )
    return kwargs, kwargs_keylist


@pytest.fixture(scope="session")
def qp_hess_a(default_seed):
    """Returns the Hessian and linear terms for a convex quadratic objective.

    The objective is 0.5 * x @ hess @ x + a @ x, so hess is the Hessian.

    Parameters
    ----------
    default_seed : int
        pytest fixture. See top-level package conftest.py.

    Returns
    -------
    hess : numpy.ndarray
        Hessian matrix of the objective, shape (5, 5)
    a : numpy.ndarray
        Linear terms of the objective, shape (5,)
    n_features : int
        Number of features in the corresponding quadratic optimization problem
    """
    # number of features (low-dimensionality problem)
    n_features = 5
    # PRNG used to generate the linear terms. can't use default_rng since this
    # fixture is session scope, while default_rng is function scope.
    rng = np.random.default_rng(default_seed)
    # compute hessian, guaranteed to be positive definite
    hess = make_spd_matrix(n_features, random_state=default_seed)
    hess += 1e-4 * np.eye(n_features)
    # use spectral condition of hess for range of uniform linear terms + return
    cond = np.linalg.cond(hess)
    a = cond * (-1 + 2 * rng.random(n_features))
    return hess, a, n_features


@pytest.fixture(scope="session", params=["separate", "together"])
def qp_noargs(qp_hess_a, request):
    """An unconstrained convex quadratic problem to minimize using mnewton.

    Objective, gradient, and Hessian do not take any arguments.

    All tests depending on this fixture are run twice, first with separate
    objective and gradient, next with an objective that returns (loss, grad)
    together, where the gradient is simply True. Initial parameter guess is 0.

    Parameters
    ----------
    qp_hess_a : tuple
        pytest fixture. See _qp_hess_a.
    request : _pytest.fixtures.FixtureRequest
        Built-in pytest fixture. See the pytest documentation for details.

    Returns
    -------
    f_obj : function
        The objective to minimize. If request.param == "separate", returns a
        scalar objective value, while if request.param == "together", returns
        (loss, grad). A convex, quadratic function of 5 variables.
    x0 : numpy.ndarray
        The initial parameter guess, essentially numpy.zeros(5)
    f_grad : function or bool
        The gradient of the objective. If request.param == "separate", returns
        a numpy.ndarray shape (5,), while if request.param == "together", is
        not a callable and simply set to True.
    f_hess : function
        The Hessian of the objective, returning numpy.ndarray shape (5, 5).
    """
    # get Hessian, linear terms, n_features from _qp_hess_a
    hess, a, n_features = qp_hess_a
    # define objective, gradient. depends on request.param
    if request.param == "together":
        f_obj = lambda x: (0.5 * x @ hess @ x + a @ x, hess @ x + a)
        f_grad = True
    else:
        f_obj = lambda x: 0.5 * x @ hess @ x + a @ x
        f_grad = lambda x: hess @ x + a
    # define hessian + return f_obj, initial guess, f_grad, f_hess
    f_hess = lambda x: hess
    return f_obj, np.zeros(n_features), f_grad, f_hess


@pytest.fixture(scope="session", params=["separate", "together"])
def qp_yesargs(qp_hess_a, request):
    """An unconstrained convex quadratic problem to minimize using mnewton.

    Objective, gradient, and Hessian are functions of their last two arguments,
    which are the Hessian matrix and linear terms of the objective. Essentially
    identical to the optimization problem defined by qp_noargs.

    All tests depending on this fixture are run twice, first with separate
    objective and gradient, next with an objective that returns (loss, grad)
    together, where the gradient is simply True. Initial parameter guess is 0.

    Parameters
    ----------
    qp_hess_a : tuple
        pytest fixture. See _qp_hess_a.
    request : _pytest.fixtures.FixtureRequest
        Built-in pytest fixture. See the pytest documentation for details.

    Returns
    -------
    f_obj : function
        The objective to minimize. If request.param == "separate", returns a
        scalar objective value, while if request.param == "together", returns
        (loss, grad). A convex, quadratic function of 5 variables.
    x0 : numpy.ndarray
        The initial parameter guess, essentially numpy.zeros(5)
    f_grad : function or bool
        The gradient of the objective. If request.param == "separate", returns
        a numpy.ndarray shape (5,), while if request.param == "together", is
        not a callable and simply set to True.
    f_hess : function
        The Hessian of the objective, returning numpy.ndarray shape (5, 5).
    f_args : tuple
        The additional positional arguments passed to the objective, gradient,
        Hessian functions. First element is the Hessian matrix, second
    """
    # get Hessian, linear terms, n_features from _qp_hess_a
    hess, a, n_features = qp_hess_a
    # form tuple holding hess, a. args to pass to f_obj, f_grad, f_hess
    f_args = (hess, a)
    # define objective, gradient. depends on request.param. note that f_obj,
    # f_grad, f_hess all depend on hess, a.
    if request.param == "together":
        f_obj = lambda x, hess, a: (0.5 * x @ hess @ x + a @ x, hess @ x + a)
        f_grad = True
    else:
        f_obj = lambda x, hess, a: 0.5 * x @ hess @ x + a @ x
        f_grad = lambda x, hess, a: hess @ x + a
    # define hessian + return f_obj, initial guess, f_grad, f_hess
    f_hess = lambda x, hess, a: hess
    return f_obj, np.zeros(n_features), f_grad, f_hess, f_args