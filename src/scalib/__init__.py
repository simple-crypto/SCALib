__all__ = ["metrics", "attacks", "modeling", "postprocessing", "config"]

import cpufeature

import ._build_config
from .version import version as __version__

if (
        _build_config.REQUIRE_AVX2 and not
        (cpufeature.CPUFeature["AVX2"] and cpufeature.CPUFeature["OS_AVX"])
        ):
        raise ImportError(
            "SCALib has been compiled with AVX2 instructions, which are not " +
            "supported by your CPU or OS. See "+
            "https://github.com/simple-crypto/SCALib/blob/main/DEVELOP.rst#local-build " +
            "for compiling without AVX2 instructions."
            )
