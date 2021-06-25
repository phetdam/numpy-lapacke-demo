__doc__ = """pytest test fixtures for solvers.tests subpackage.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest


@pytest.fixture
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