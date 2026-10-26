"""
Moved to the modelmatrix package, as `modelmatrix.transforms`: this module
is `modelmatrix.transforms` itself, so that imports from it keep working.
"""

import sys

from modelmatrix import transforms as _module

sys.modules[__name__] = _module
