#pragma once
/**
 * @file denoise_libraw.h
 * @brief LibRaw subclass exposing the protected pre/post-demosaic denoise
 *        stages so the custom RAW pipeline can reuse LibRaw's OWN compiled
 *        denoise code (green_matching / wavelet / FBDD / chroma median).
 *
 * LibRaw's denoise methods (libraw.h:457-468) are `protected`, so they are
 * unreachable from outside the class. Public inheritance + `using`-declarations
 * lifts only those members into a public interface, without re-implementing a
 * single line of the algorithm. `DenoiseLibRaw` IS-A `LibRaw`, so it is also
 * accepted by any helper taking `LibRaw&`.
 *
 * Exposed members:
 *   - green_matching()  — G1/G3 equalization (pre-demosaic, CFA domain).
 *   - scale_colors()    — camera WB multiply; internally runs wavelet_denoise()
 *                         when imgdata.params.threshold > 0.
 *   - fbdd(int)         — CFA-domain chroma denoise (pre-demosaic).
 *   - median_filter()   — post-demosaic 3x3 chroma median on (R-G)/(B-G),
 *                         gated by imgdata.params.med_passes.
 *
 * Already public on LibRaw (no `using` needed, listed for reference):
 *   raw2image_ex(int), subtract_black_internal(), adjust_maximum().
 */

#include <libraw/libraw.h>

namespace rawalchemy {

class DenoiseLibRaw : public LibRaw {
public:
    using LibRaw::green_matching;
    using LibRaw::scale_colors;
    using LibRaw::fbdd;
    using LibRaw::median_filter;
};

} // namespace rawalchemy
