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


@pytest.fixture(autouse=True)
def _skip_internal_exposed(request):
    """Skips tests marked with skip_internal_exposed.

    Do not use directly on tests. The skip_internal_exposed mark, passed an
    appropriate module, should be used instead on individual tests.

    autouse=True means all tests will include this fixture. Can be used to set
    individual tests or set to module's pytestmark to skip whole module.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Built-in pytest fixture. See the pytest documentation for details.

    Returns
    -------
    None
    """
    # get the module passed to skip_internal_exposed
    module = request.node.get_closest_marker("skip_internal_exposed").args[0]
    # get name and members of the module
    name = module.__name__
    members = dir(module)
    # look for any EXPOSED_* members. if so, don't skip
    for member in members:
        if member.startswith("EXPOSED_"):
            return
    # else skip, not compiled with -DEXPOSE_INTERNAL
    pytest.skip(msg=f"{name} not compiled with -DEXPOSE_INTERNAL")