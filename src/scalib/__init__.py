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

import cpuinfo

from .build_config import REQUIRE_X86_64_V3
from .version import version as __version__

cpu = cpuinfo.get_cpu_info()

X86_64_V3_FEATURES = [
    "AVX",
    "AVX2",
    "BMI1",
    "BMI2",
    # Some features not tested, shouldn't be an issue.
    # "f16c",
    # "fma",
    # "lzcnt",
    # "movbe",
    # "osxsave",
]

if (
    REQUIRE_X86_64_V3
    and cpu["arch"] == "X86_64"
    and any(f.lower() not in cpu["flags"] for f in X86_64_V3_FEATURES)
):
    mf = ", ".join(f for f in X86_64_V3_FEATURES if f not in cpu["flags"])
    raise ImportError(
        "This SCALib build requires x86-64-v3 level processor features, which "
        + "are not all supported by this CPU or OS."
        + "See https://github.com/simple-crypto/SCALib/blob/main/README.rst "
        + "for compiling a SCALib adapted to your CPU."
        + f"(Missing features: {mf})."
    )

from scalib._scalib_ext import ScalibError
