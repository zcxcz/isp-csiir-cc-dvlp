//==============================================================================
// ISP-CSIIR HLS Backend Abstraction
//------------------------------------------------------------------------------
// This header is intentionally limited to datatype / channel abstraction.
// Tool scheduling, interface, unroll, and memory directives should be managed
// in the HLS tool Tcl, not scattered in C++ source.
//==============================================================================

#ifndef CSIIR_HLS_BACKEND_HPP
#define CSIIR_HLS_BACKEND_HPP

#if !defined(CSIIR_HLS_BACKEND_VIVADO) && !defined(CSIIR_HLS_BACKEND_CATAPULT)
#define CSIIR_HLS_BACKEND_CATAPULT 1
#endif

#if defined(CSIIR_HLS_BACKEND_VIVADO) && defined(CSIIR_HLS_BACKEND_CATAPULT)
#error "Select only one HLS backend"
#endif

#if defined(CSIIR_HLS_BACKEND_VIVADO)
#include <ap_int.h>
#include <hls_stream.h>

namespace csiir_hls {
template <int W>
using uint_t = ap_uint<W>;

template <int W>
using int_t = ap_int<W>;

template <typename T>
using stream_t = hls::stream<T>;
}  // namespace csiir_hls

#else
#include "third_party/ac_types/include/ac_int.h"
#include "third_party/ac_types/include/ac_channel.h"

namespace csiir_hls {
template <int W>
using uint_t = ac_int<W, false>;

template <int W>
using int_t = ac_int<W, true>;

template <typename T>
using stream_t = ac_channel<T>;
}  // namespace csiir_hls
#endif

#endif  // CSIIR_HLS_BACKEND_HPP
