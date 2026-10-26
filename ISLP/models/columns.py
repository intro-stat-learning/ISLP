"""
Moved to the modelmatrix package, as `modelmatrix.columns`: this module
is `modelmatrix.columns` itself, so that imports from it keep working.
"""

import sys

from modelmatrix import columns as _module

sys.modules[__name__] = _module
