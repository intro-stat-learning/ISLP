"""
Moved to the modelmatrix package, as `modelmatrix.model_spec`: this module
is `modelmatrix.model_spec` itself, so that imports from it keep working.
"""

import sys

from modelmatrix import model_spec as _module

sys.modules[__name__] = _module
