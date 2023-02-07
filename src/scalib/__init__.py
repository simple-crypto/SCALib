__all__ = ["metrics", "attacks", "modeling", "postprocessing", "config", "ScalibError"]

import cpuinfo

from .build_config import REQUIRE_AVX2
from .version import version as __version__

cpu = cpuinfo.get_cpu_info()


if REQUIRE_AVX2 and cpu["arch"] == "X68_64" and "avx2" not in cpu["flags"]:
    raise ImportError(
        "SCALib has been compiled with AVX2 instructions, which are not "
        + "supported by your CPU or OS. See "
        + "https://github.com/simple-crypto/SCALib/blob/main/README.rst "
        + "for compiling without AVX2 instructions."
    )

from scalib._scalib_ext import ScalibError
