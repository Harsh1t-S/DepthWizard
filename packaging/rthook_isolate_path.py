"""Import only from the bundle, whatever the launching environment sets.

A frozen build still inherits PYTHONPATH, so a developer machine that points it
at an unrelated package directory can shadow the bundled copies of PIL,
fastapi or typing_extensions -- with extension modules built for a different
Python version, which fails at import with no useful message. Nothing outside
the bundle should ever satisfy an import here, so drop every search path that
does not live under the extraction directory before any application module is
imported.
"""

import os
import sys

_meipass = getattr(sys, "_MEIPASS", None)
if _meipass:
    _root = os.path.abspath(_meipass) + os.sep
    _kept = []
    for _entry in sys.path:
        try:
            _resolved = os.path.abspath(_entry)
        except (OSError, ValueError):
            continue
        if _resolved + os.sep == _root or _resolved.startswith(_root):
            _kept.append(_entry)
    sys.path[:] = _kept
    # Subprocesses and any late importer must not pick it back up either.
    os.environ.pop("PYTHONPATH", None)
