//==============================================================================
// ISP-CSIIR HLS Backend Abstraction
//==============================================================================

#ifndef CSIIR_HLS_BACKEND_HPP
#define CSIIR_HLS_BACKEND_HPP

#if !defined(CSIIR_HLS_BACKEND_VIVADO) && !defined(CSIIR_HLS_BACKEND_CATAPULT)
#define CSIIR_HLS_BACKEND_VIVADO 1
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

#define CSIIR_HLS_PRAGMA(x) _Pragma(#x)
#define CSIIR_HLS_INTERFACE_AXIS(port) CSIIR_HLS_PRAGMA(HLS INTERFACE axis port=port)
#define CSIIR_HLS_INTERFACE_AXILITE(port, bundle) CSIIR_HLS_PRAGMA(HLS INTERFACE s_axilite port=port bundle=bundle)
#define CSIIR_HLS_INTERFACE_CTRL_HS(port) CSIIR_HLS_PRAGMA(HLS INTERFACE ap_ctrl_hs port=port)
#define CSIIR_HLS_UNROLL CSIIR_HLS_PRAGMA(HLS UNROLL)
#define CSIIR_HLS_ARRAY_PARTITION_COMPLETE(var, dim) CSIIR_HLS_PRAGMA(HLS ARRAY_PARTITION variable=var dim=dim complete)
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

#define CSIIR_HLS_INTERFACE_AXIS(port)
#define CSIIR_HLS_INTERFACE_AXILITE(port, bundle)
#define CSIIR_HLS_INTERFACE_CTRL_HS(port)
#define CSIIR_HLS_UNROLL
#define CSIIR_HLS_ARRAY_PARTITION_COMPLETE(var, dim)
#endif

#endif  // CSIIR_HLS_BACKEND_HPP
