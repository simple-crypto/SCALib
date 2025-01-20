__all__ = [
    "preprocessing",
    "config",
    "attacks",
    "metrics",
    "modeling",
    "postprocessing",
    "tools",
    "ScalibError",
]


from .build_config import REQUIRE_X86_64_V3
from .version import version as __version__  # noqa: F401

from ._cpu_check import support_x86_64_v3

if REQUIRE_X86_64_V3 and not all(s for _, s in support_x86_64_v3()):
    # mf = ", ".join(f for f in X86_64_V3_FEATURES if f not in cpu["flags"])
    raise ImportError(
        "This SCALib build requires x86-64-v3 level processor features, which "
        + "are not all supported by this CPU or OS."
        + "See https://github.com/simple-crypto/SCALib/blob/main/README.rst "
        + "for compiling a SCALib adapted to your CPU."
        + f"Missing features: {[f for f, a in support_x86_64_v3() if not a]}."
        #    + f"(Missing features: {mf})."
    )

from scalib._scalib_ext import ScalibError
