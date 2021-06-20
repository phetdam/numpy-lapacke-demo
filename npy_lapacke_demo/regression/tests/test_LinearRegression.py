__doc__ = """Tests for the LinearRegression class provided by _linreg.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from functools import partial
import pytest

# pylint: disable=no-name-in-module
from .._linreg import LinearRegression


def test_defaults(lr_default):
    """Test that LinearRegression defaults are as expected.

    Parameters
    ----------
    lr_default : LinearRegression
        pytest fixture. See local conftest.py.
    """
    assert lr_default.fit_intercept == True
    assert lr_default.solver == "svd"


def test_init_sanity():
    """Test sanity of LinearRegression.__init__."""
    # no positional arguments are accepted
    with pytest.raises(TypeError, match="no positional"):
        LinearRegression(False, "qr")
    # only solvers allowed are qr, svd. no checking fit_intercept since any
    # Python expression will be converted into a boolean expression
    with pytest.raises(ValueError, match="solver must be one of"):
        LinearRegression(solver="magic_solver")


def test_members(lr_default):
    """Members of the LinearRegression instance are read-only.

    Parameters
    ----------
    lr_default : LinearRegression
        pytest fixture. See local conftest.py.
    """
    # functools.partial facatory for pytest context manager
    mgr_gen = partial(pytest.raises, AttributeError, match="readonly")
    # no type checking done since attributes are read-only.
    with mgr_gen():
        lr_default.fit_intercept = (3 + 5 == 6)
    with mgr_gen():
        lr_default.solver = "magic_solver"


def test_unfitted_getset(lr_default):
    """Test LinearRegression getsets, not available before fitting.

    Parameters
    ----------
    lr_default : LinearRegression
        pytest fixture. See local conftest.py.
    """
    # functools.partial facatory for pytest context manager
    mgr_gen = partial(pytest.raises, AttributeError, match="after fitting")
    # need each access in its own block
    with mgr_gen():
        lr_default.coef_
    with mgr_gen():
        lr_default.intercept_
    with mgr_gen():
        lr_default.rank_
    with mgr_gen():
        lr_default.singular_