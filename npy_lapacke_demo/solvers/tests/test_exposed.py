__doc__ = """Tests for extension functions requiring -DEXPOSE_INTERNAL.

The C extension functions that are tested here expose Python-accessible
wrappers only when the extensions are compiled with -DEXPOSE_INTERNAL. These
wrappers all start with EXPOSED_ and usually include some input checking.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from functools import partial
import numpy as np
import pytest

from .. import _mnewton

# skip module tests if _mnewton not built with -DEXTERNAL_EXPOSE. pytestmark
# applies mark to all tests in module, skip_internal_exposed a custom mark.
pytestmark = pytest.mark.skip_internal_exposed(_mnewton)

# patterns to match for warnings issued by remove_[un]specified_kwargs
_specified_match = ".+not in kwargs$"
_unspecified_match = ".+removed from kwargs$"


# use filterwarnings mark to turn warnings into test failure if warn=False
@pytest.mark.filterwarnings(f"error:{_specified_match}:UserWarning")
@pytest.mark.parametrize("warn", [True, False])
def test_remove_specified_kwargs_empty(empty_kwargs, warn):
    """Test the internal remove_specified_kwargs function on empty kwargs.

    Parameters
    ----------
    empty_kwargs : tuple
        pytest fixture. See local conftest.py.
    warn : bool
        ``True`` to warn if a specified string key not in kwargs, else silence.
    """
    # get kwargs dict and list of keys to drop
    kwargs, droplist = empty_kwargs
    # callable with kwargs, droplist, warn already filled in
    test_callable = partial(
        _mnewton.EXPOSED_remove_specified_kwargs, kwargs, droplist, warn=warn
    )
    # if warn, expect warnings to be raised. save number of dropped keys.
    if warn:
        with pytest.warns(UserWarning, match=_specified_match):
            drops = test_callable()
    # else expect no warnings to be raised. raised warnings fail the test
    else:
        drops = test_callable()
    # no keys should be dropped from empty_kwargs since it is empty
    assert drops == 0


# use filterwarnings mark to turn warnings into test failure if warn=False
@pytest.mark.filterwarnings(f"error:{_specified_match}:UserWarning")
@pytest.mark.parametrize("warn", [True, False])
def test_remove_specified_kwargs_full(full_kwargs, warn):
    """Test the internal remove_specified_kwargs function on full kwargs.

    Parameters
    ----------
    full_kwargs : tuple
        pytest fixture. See local conftest.py.
    warn : bool
        ``True`` to warn if a specified string key not in kwargs, else silence.
    """
    # get kwargs dict and list of keys to drop
    kwargs, droplist = full_kwargs
    # callable with kwargs, droplist, warn already filled in
    test_callable = partial(
        _mnewton.EXPOSED_remove_specified_kwargs, kwargs, droplist, warn=warn
    )
    # if warn, expect warnings to be raised. save number of dropped keys.
    if warn:
        with pytest.warns(UserWarning, match=_specified_match):
            drops = test_callable()
    # else expect no warnings to be raised. raised warnings fail the test
    else:
        drops = test_callable()
    # 3 of the keys should be dropped from full_kwargs
    assert drops == 3


# use filterwarnings mark to turn any warnings into test failure
@pytest.mark.filterwarnings(f"error:{_unspecified_match}:UserWarning")
@pytest.mark.parametrize("warn", [True, False])
def test_remove_unspecified_kwargs_empty(empty_kwargs, warn):
    """Test the internal remove_unspecified_kwargs function on empty kwargs.

    Parameters
    ----------
    empty_kwargs : tuple
        pytest fixture. See local conftest.py.
    warn : bool
        ``True`` to warn if a specified string key not in kwargs, else silence.
    """
    # get kwargs dict and list of keys to keep
    kwargs, keeplist = empty_kwargs
    # callable with kwargs, droplist, warn already filled in
    test_callable = partial(
        _mnewton.EXPOSED_remove_unspecified_kwargs, kwargs, keeplist, warn=warn
    )
    # for empty kwargs, no warnings should ever be raised + no keys dropped
    assert test_callable() == 0


# use filterwarnings mark to turn warnings into test failure if warn=False
@pytest.mark.filterwarnings(f"error:{_unspecified_match}:UserWarning")
@pytest.mark.parametrize("warn", [True, False])
def test_remove_unspecified_kwargs_full(full_kwargs, warn):
    """Test the internal remove_unspecified_kwargs function on full kwargs.

    Parameters
    ----------
    full_kwargs : tuple
        pytest fixture. See local conftest.py.
    warn : bool
        ``True`` to warn if a specified string key not in kwargs, else silence.
    """
    # get kwargs dict and list of keys to keep
    kwargs, keeplist = full_kwargs
    # callable with kwargs, droplist, warn already filled in
    test_callable = partial(
        _mnewton.EXPOSED_remove_unspecified_kwargs, kwargs, keeplist, warn=warn
    )
    # if warn, expect warnings to be raised. save number of dropped keys.
    if warn:
        with pytest.warns(UserWarning, match=_unspecified_match):
            drops = test_callable()
    # else expect no warnings to be raised. raised warnings fail the test
    else:
        drops = test_callable()
    # 5 of the keys should be dropped from full_kwargs
    assert drops == 5


@pytest.mark.parametrize("fortran", [False, True])
@pytest.mark.parametrize("shape", [(0,), (5, 6, 2, 4)])
def test_npy_frob_norm(default_rng, shape, fortran):
    """Test the internal npy_frob_norm function on different NumPy arrays.

    Parameters
    ----------
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    shape : tuple
        Shape of the ndarray to send to npy_frob_norm.
    fortran : bool
        True for column-major ordering, False for row-major, i.e. C ordering.
    """
    # compute random high-dimensional ndarray using shape. if fortran, then
    # store in a column-major format, i.e. Fortran-style.
    if fortran:
        ar = np.empty(shape=shape, order="F")
        default_rng.random(size=shape, out=ar)
    else:
        ar = default_rng.random(size=shape)
    # check that npy_frob_norm has same result as np.linalg.norm
    np.testing.assert_allclose(
        _mnewton.EXPOSED_npy_frob_norm(ar), np.linalg.norm(ar)
    )


@pytest.mark.parametrize("with_optional", [True, False])
def test_populate_OptimizeResult(default_rng, with_optional):
    """Test the internal populate_OptimizeResult function on model inputs.

    Checks both cases where optional arguments are and aren't provided.

    Parameters
    ----------
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    with_optional : bool
        Whether to include the optional parameters 
    """
    # number of features in the output
    n_features = 5
    # draw x from shifted standard lognormal distribution
    x = default_rng.lognormal(size=n_features) - 1.
    # success, status, message
    success, status = True, 0
    message = "Iteration limit reached"
    # value of the objective, number of function evals, number of solver iters
    fun_x, n_fev, n_iter = 0.1, 1001, 1000
    # draw jac_x, hess_x, hess_inv from same distribution as x
    jac_x = default_rng.lognormal(size=n_features) - 1.
    hess_x = default_rng.lognormal(size=(n_features, n_features)) - 1.
    hess_inv = default_rng.lognormal(size=(n_features, n_features)) - 1.
    # get number of gradient, hessian evaluations
    n_jev, n_hev = n_fev, n_fev
    # maximum constraint violation
    maxcv = 0.
    # collect all the required arguments into a tuple
    req_args = (x, success, status, message, fun_x, n_fev, n_iter)
    # if with_optional, pass the optional arguments as well
    if with_optional:
        res = _mnewton.EXPOSED_populate_OptimizeResult(
            *req_args, jac_x=jac_x, n_jev=n_jev, hess_x=hess_x, n_hev=n_hev,
            hess_inv=hess_inv, maxcv=maxcv
        )
    else:
        res = _mnewton.EXPOSED_populate_OptimizeResult(*req_args)
    # check that the required arguments are present as attributes in the
    # return OptimizeResult and that their value has not been changed. note we
    # directly test for equality with floats since no computation is done.
    assert np.array_equal(res.x, x)
    assert res.success == success
    assert res.status == status
    assert res.message == message
    assert res.fun == fun_x
    assert res.nfev == n_fev
    assert res.nit == n_iter
    # if with_optional is provided, also check them
    if with_optional:
        assert np.array_equal(res.jac, jac_x)
        assert res.njev == n_jev
        assert np.array_equal(res.hess, hess_x)
        assert res.nhev == n_hev
        assert np.array_equal(res.hess_inv, hess_inv)
        assert res.maxcv == maxcv