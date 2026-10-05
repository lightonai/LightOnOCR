"""Post-processing of LightOnOCR markdown. Standard library only.

Both modules are the scripts used to evaluate the models, unchanged (see their docstrings):

* ``fix_escaped_dollars``: dollar-delimiter repair only (``\\$x\\$`` -> ``$x$``). Applied to every
  LightOnOCR output, in the client and in the benchmarks.
* ``postprocess_ocr_markdown``: PP3 = that same dollar repair, then markdown formatting rules tuned on
  olmOCR-bench. Used only for olmOCR-bench with omit-margins.
"""

from .fix_escaped_dollars import fix_escaped_dollars, repair_dollar_delimiters
from .postprocess_ocr_markdown import cleanup


def pp3(text: str) -> str:
    """Dollar repair followed by the PP3 formatting steps."""
    return cleanup(text, profile="pp3")[0]


__all__ = ["cleanup", "fix_escaped_dollars", "pp3", "repair_dollar_delimiters"]
