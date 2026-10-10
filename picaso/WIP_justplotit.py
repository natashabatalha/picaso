"""
Deprecated: the multi-backend plotting tools now live in ``picaso.justplotit``.
Use ``from picaso import justplotit as jpi`` and the ``backend`` keyword or ``jpi.Analyzer``.
"""
import warnings

from .justplotit import *
from .justplotit import Analyzer

warnings.warn("picaso.WIP_justplotit is deprecated and will be removed. Use picaso.justplotit, "
              "which now supports backend='bokeh', 'matplotlib' or 'plotly'.",
              DeprecationWarning, stacklevel=2)
