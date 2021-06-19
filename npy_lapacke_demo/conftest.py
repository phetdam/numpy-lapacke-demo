__doc__ = """Top-level conftest.py for global pytest fixtures.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import pytest

@pytest.fixture(scope="session")
def global_seed():
    """Global seed value to be used for any PRNG seeding.

    Returns
    -------
    int
    """
    return 7