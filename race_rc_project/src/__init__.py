"""src package init

Avoid importing heavy training scripts at package import time. Users should
import specific modules explicitly, e.g. `from src import inference`.
"""

__all__ = ["preprocessing", "inference", "evaluate"]
