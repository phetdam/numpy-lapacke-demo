__doc__ = """Top-level __init__.py for numpy-lapacke-demo.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

import os
import os.path
import platform
import sys

__version__ = "0.1.0.dev0"

# on Windows, since there is no standardized repair command for wheels,
# redistributable wheels must package copies of the DLLs they depend on. that
# means that these DLLs must also be in the DLL search path, i.e. part of
# %PATH%, in order for the extension modules to work correctly. solution here
# is from answer to question 65334494 on StackOverflow, itself based on NumPy's
# solution. the conditional code only runs on Windows platforms.
_blap_dll_dir = os.path.dirname(os.path.abspath(__file__))
# do PATH adjustments for Windows only
if platform.system() == "Windows":
    # os.add_dll_directory only available for Python 3.8+
    if sys.version_info >= (3, 8):
        os.add_dll_directory(_blap_dll_dir)
    # else manipulate PATH variable itself
    else:
        # just in case PATH doesn't exist (it should usually)
        os.environ.setdefault("PATH", "")
        os.environ["PATH"] += f"{os.pathsep}{_blap_dll_dir}"