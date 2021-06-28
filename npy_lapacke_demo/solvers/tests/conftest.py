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
    dict
        An empty dict.
    list
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
    dict
        An nonempty dict with more keys than those provided in kwargs_keylist
        but containing only a subset of the keys in kwargs_keylist.
    list
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


@pytest.fixture(scope="session", params=["separate", "together"])
def qp_noargs(default_seed, request):
    """An unconstrained convex quadratic problem to minimize using mnewton.

    Objective, gradient, and Hessian do not take any arguments.

    All tests depending on this fixture are run twice, first with separate
    objective and gradient, next with an objective that returns (loss, grad)
    together, where the gradient is simply True. Initial parameter guess is 0.

    Parameters
    ----------
    default_seed : int
        pytest fixture. See top-level package conftest.py.
    request : _pytest.fixtures.FixtureRequest
        Built-in pytest fixture. See the pytest documentation for details.

    Returns
    -------
    function
        The objective to minimize. If request.param == "separate", returns a
        scalar objective value, while if request.param == "together", returns
        (loss, grad). A convex, quadratic function of 5 variables.
    numpy.ndarray
        The initial parameter guess, essentially numpy.zeros(5)
    function or bool
        The gradient of the objective. If request.param == "separate", returns
        a numpy.ndarray shape (5,), while if request.param == "together", is
        not a callable and simply set to True.
    function
        The Hessian of the objective, returning numpy.ndarray shape (5, 5).
    """
    # low-dimensionality problem
    n_features = 5
    # PRNG used to generate the linear terms. can't use default_rng since this
    # fixture is session scope, while default_rng is function scope.
    rng = np.random.default_rng(default_seed)
    # compute hessian, guaranteed to be positive definite
    hess = make_spd_matrix(n_features, random_state=default_seed)
    hess += 1e-4 * np.eye(n_features)
    # use spectral condition number of hess for range of uniform linear terms
    cond = np.linalg.cond(hess)
    a = cond * (-1 + 2 * rng.random(n_features))
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