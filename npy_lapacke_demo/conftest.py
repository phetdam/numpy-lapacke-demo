__doc__ = """Top-level conftest.py for global pytest fixtures.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def default_seed():
    """Default seed value to be used for any PRNG seeding.

    Returns
    -------
    int
    """
    return 7


@pytest.fixture
def default_rng(default_seed):
    """Default numpy.random.Generator, seeded with default_seed.

    Function scope means that the PRNG instance is fresh per test.

    Parameters
    ----------
    default_seed : int
        pytest fixture. See default_seed.

    Returns
    -------
    numpy.random.Generator
    """
    return np.random.default_rng(default_seed)