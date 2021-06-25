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


# use filterwarnings mark to turn warnings into errors if warn=False
@pytest.mark.filterwarnings("error:.+not in kwargs$:UserWarning")
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
        with pytest.warns(UserWarning, match="not in kwargs"):
            drops = test_callable()
    # else expect no warnings to be raised. raised warnings fail the test
    else:
        drops = test_callable()
    # no keys should be dropped from empty_kwargs since it is empty
    assert drops == 0