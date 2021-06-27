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


@pytest.mark.parametrize("shape", [(0,), (5, 6, 2, 4)])
def test_npy_frob_norm_ccont(global_seed, shape):
    """Test the interal npy_frob_norm function on row-major ndarray.

    Parameters
    ----------
    global_seed : int
        pytest fixture. See top-level package conftest.py.
    shape : tuple
        Shape of the ndarray to send to npy_frob_norm.
    """
    # PRNG to use
    rng = np.random.default_rng(global_seed)
    # compute random high-dimensional ndarray using shape
    ar = rng.random(size=shape)
    # check that npy_frob_norm has same result as np.linalg.norm. results
    # should be exactly identical, so we don't use assert_allclose
    assert _mnewton.EXPOSED_npy_frob_norm(ar) == np.linalg.norm(ar)