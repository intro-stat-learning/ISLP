# use a non-interactive matplotlib backend: the tests only draw, and
# Tk is not always usable on CI (e.g. Tcl in uv's Python on Windows)
import matplotlib

matplotlib.use('Agg')
