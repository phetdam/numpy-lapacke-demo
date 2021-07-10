__doc__ = """Tests for internal _mnewton functions.

The internal C extension functions in _mnewton are exposed using their
respective Python-accessible wrappers in _mnewton_internal.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from functools import partial
import numpy as np
import pytest
import scipy.linalg

from .. import _mnewton_internal

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
        _mnewton_internal.remove_specified_kwargs, kwargs, droplist, warn=warn
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
        _mnewton_internal.remove_specified_kwargs, kwargs, droplist, warn=warn
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
        _mnewton_internal.remove_unspecified_kwargs, kwargs, keeplist, warn=warn
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
        _mnewton_internal.remove_unspecified_kwargs, kwargs, keeplist, warn=warn
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
        _mnewton_internal.npy_frob_norm(ar), np.linalg.norm(ar)
    )


def test_tuple_prepend_single():
    """Test the internal tuple_prepend_single function on model inputs."""
    # can be any Python object
    x = "arbitrary Python object"
    # can be any arbitrary tuple as well
    old_tp = ("arbitrary", "tuple")
    # shortens the invocation
    test_func = _mnewton_internal.tuple_prepend_single
    # check that (x,) is returned if old_tp not provided
    assert test_func(x) == (x,)
    # check that the expected result is returned
    assert test_func(x, old_tp=old_tp) == (x, *old_tp)


def test_loss_only_fun_call_noargs(qp_noargs, default_rng):
    """Test the internal loss_only_fun_call function on model inputs.

    Tests the case where the objective takes no args.

    Parameters
    ----------
    qp_noargs : tuple
        pytest fixture. See local conftest.py.
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    """
    # get objective, initial guess (unused), gradient from qp_noargs
    f_obj, x0, f_grad, _ = qp_noargs
    # use x0's shape to get random value to evaluate f_obj at
    x = default_rng.uniform(size=x0.shape)
    # compute expected loss actual loss. f_grad == True => f_obj returns a
    # tuple, so we drop the second element (grad) of the tuple.
    if f_grad == True:
        loss, _ = f_obj(x)
    else:
        loss = f_obj(x)
    loss_hat = _mnewton_internal.loss_only_fun_call(f_obj, x)
    # check that losses are essentially the same
    np.testing.assert_allclose(loss_hat, loss)


def test_loss_only_fun_call_yesargs(qp_yesargs, default_rng):
    """Test the internal loss_only_fun_call function on model inputs.

    Tests the case where the objective takes no args.

    Parameters
    ----------
    qp_yesargs : tuple
        pytest fixture. See local conftest.py.
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    """
    # get objective, initial guess (unused), gradient, args from qp_yesargs
    f_obj, x0, f_grad, _, f_args = qp_yesargs
    # use x0's shape to get random value to evaluate f_obj at
    x = default_rng.uniform(size=x0.shape)
    # compute expected loss actual loss. f_grad == True => f_obj returns a
    # tuple, so we drop the second element (grad) of the tuple.
    if f_grad == True:
        loss, _ = f_obj(x, *f_args)
    else:
        loss = f_obj(x, *f_args)
    loss_hat = _mnewton_internal.loss_only_fun_call(f_obj, x, args=f_args)
    # check that losses are essentially the same
    np.testing.assert_allclose(loss_hat, loss)


def test_compute_loss_grad_noargs(qp_noargs, default_rng):
    """Test the internal compute_loss_grad function on model inputs.

    Tests the case where the objective and gradient take no args.

    Parameters
    ----------
    qp_noargs : tuple
        pytest fixture. See local conftest.py.
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    """
    # get objective, initial guess (unused), gradient from qp_noargs
    f_obj, x0, f_grad, _ = qp_noargs
    # use x0's shape to get random value to evaluate f_obj, f_grad at
    x = default_rng.uniform(size=x0.shape)
    # compute expected loss and gradient and actual loss and gradient. when
    # f_grad is True, then f_obj returns both loss and grad.
    if f_grad == True:
        loss, grad = f_obj(x)
    else:
        loss, grad = f_obj(x), f_grad(x)
    loss_hat, grad_hat = _mnewton_internal.compute_loss_grad(f_obj, f_grad, x)
    # check that losses and grads are essentially the same
    np.testing.assert_allclose(loss_hat, loss)
    np.testing.assert_allclose(grad_hat, grad)


def test_compute_loss_grad_yesargs(qp_yesargs, default_rng):
    """Test the internal compute_loss_grad function on model inputs.

    Tests the case where the objective and gradient take args.

    Parameters
    ----------
    qp_yesargs : tuple
        pytest fixture. See local conftest.py.
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    """
    # get objective, initial guess (unused), gradient, args from qp_yesargs
    f_obj, x0, f_grad, _, f_args = qp_yesargs
    # use x0's shape to get random value to evaluate f_obj, f_grad at
    x = default_rng.uniform(size=x0.shape)
    # compute expected loss and gradient and actual loss and gradient, with
    # args. when f_grad is True, then f_obj returns both loss and grad.
    if f_grad == True:
        loss, grad = f_obj(x, *f_args)
    else:
        loss, grad = f_obj(x, *f_args), f_grad(x, *f_args)
    res = _mnewton_internal.compute_loss_grad(f_obj, f_grad, x, args=f_args)
    # check that losses and grads are essentially the same
    np.testing.assert_allclose(res[0], loss)
    np.testing.assert_allclose(res[1], grad)


def test_compute_hessian_noargs(qp_noargs, default_rng):
    """Test the internal compute_hessian function on model inputs.

    Tests the case where the Hessian function does not take args.

    Parameters
    ----------
    qp_noargs : tuple
        pytest fixture. See local conftest.py.
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    """
    # get initial guess (unused) and hessian function from qp_noargs
    _, x0, _, f_hess = qp_noargs
    # use x0's shape to get random value to evaluate f_obj, f_grad at
    x = default_rng.uniform(size=x0.shape)
    # compute expected Hessian and actual Hessian
    hess = f_hess(x)
    hess_hat = _mnewton_internal.compute_hessian(f_hess, x)
    # check that Hessians are essentially the same
    np.testing.assert_allclose(hess_hat, hess)


def test_compute_hessian_yesargs(qp_yesargs, default_rng):
    """Test the internal compute_hessian function on model inputs.

    Tests the case where the Hessian function does take args.

    Parameters
    ----------
    qp_yesargs : tuple
        pytest fixture. See local conftest.py.
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    """
    # get initial guess (unused) and hessian function from qp_yesargs
    _, x0, _, f_hess, f_args = qp_yesargs
    # use x0's shape to get random value to evaluate f_obj, f_grad at
    x = default_rng.uniform(size=x0.shape)
    # compute expected Hessian and actual Hessian, with args
    hess = f_hess(x, *f_args)
    hess_hat = _mnewton_internal.compute_hessian(f_hess, x, args=f_args)
    # check that Hessians are essentially the same
    np.testing.assert_allclose(hess_hat, hess)


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
        res = _mnewton_internal.populate_OptimizeResult(
            *req_args, jac_x=jac_x, n_jev=n_jev, hess_x=hess_x, n_hev=n_hev,
            hess_inv=hess_inv, maxcv=maxcv
        )
    else:
        res = _mnewton_internal.populate_OptimizeResult(*req_args)
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


def test_lower_packed_copy():
    """Test the internal lower_packed_copy function on a model input."""
    # arbitrary square matrix shape (3, 3) and its packed lower triangle
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    lmatp = np.array([1, 4, 5, 7, 8, 9])
    # compute packed lower triangle of mat using lower_packed_copy
    lmatp_hat = _mnewton_internal.lower_packed_copy(mat)
    # check that lower_packed_copy gives the same result as expected
    np.testing.assert_allclose(lmatp_hat, lmatp)


def test_compute_mnewton_descent_nom(qp_hess_a, default_rng):
    """Test the internal compute_mnewton_descent function on model input.

    Consider only the case where the Hessian is already positive definite.

    Parameters
    ----------
    qp_hess_a : tuple
        pytest fixture. See local conftest.py.
    default_rng : numpy.random.Generator
        pytest fixture. See top-level package conftest.py.
    """
    # get Hessian, linear terms, n_features for convex quadratic function.
    # ensure that hess is positive definite by bumping the diagonal.
    hess, a, n_features = qp_hess_a
    hess += 1e-3 * np.eye(n_features)
    # evaluate gradient of the function at a random point (shifted lognormal)
    grad = hess @ default_rng.lognormal(size=n_features) - 1. + a
    # compute the expected standard Newton descent direction manually using a
    # Cholesky factorization to emulate the expected output
    dvec = scipy.linalg.solve(hess, -grad, assume_a="pos")
    # call compute_mnewton_descent wrapper and get actual descent direction
    dvec_hat = _mnewton_internal.compute_mnewton_descent(hess, grad)
    # check that the actual and expected results are close
    np.testing.assert_allclose(dvec_hat, dvec)