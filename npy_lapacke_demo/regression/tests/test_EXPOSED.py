__doc__ = """Tests for extension functions requiring -DEXPOSE_INTERNAL.

The C extension functions that are tested here expose Python-accessible
wrappers only when the extensions are compiled with -DEXPOSE_INTERNAL. These
wrappers all start with EXPOSED_ and usually include some input checking.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import numpy as np
import pytest

from .. import _linreg


def test_npy_vector_matrix_mean(global_seed):
    """Test the internal npy_vector_matrix_mean function on model inputs.

    Parameters
    ----------
    global_seed : int
        pytest fixture. See top-level package conftest.py.
    """
    # seeded random number generator
    rng = np.random.default_rng(global_seed)
    # 1D ndarray to compute the mean for
    ar_1d = rng.integers(0, 10, size=50)
    # 2D ndarray to compute the mean for
    ar_2d = rng.integers(0, 10, size=(10, 20))
    # check that 1D result is same when using ndarray.mean
    np.testing.assert_allclose(
        _linreg.EXPOSED_npy_vector_matrix_mean(ar_1d), ar_1d.mean()
    )
    # check that the 2D result is the same when using ndarray.mean
    np.testing.assert_allclose(
        _linreg.EXPOSED_npy_vector_matrix_mean(ar_2d), ar_2d.mean(axis=0)
    )
    # check that ValueError is raised when passed higher-dimension ndarray
    with pytest.raises(ValueError, match="ar must be 1D or 2D only"):
        _linreg.EXPOSED_npy_vector_matrix_mean(rng.random(size=(3, 2, 5)))