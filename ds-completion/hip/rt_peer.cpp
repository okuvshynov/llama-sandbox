// A range of layers' stream to and from the host (README step 110): what --teacher with
// --stream-in / --stream-out uses. (Until step 178 this file also held the two-machine link.)
#include "rt.h"

#include "rt_internal.h"

void Runtime::stream_from_host(int M) {
  const size_t nh = (size_t)hp.n_embd * hp.hc_mult, np = hp.hc_mult;
  if (!stream_in || stream_in->h.size() < (stream_in_tok + M) * nh || stream_in->pre.size() < (stream_in_tok + M) * np)
    throw std::runtime_error("runtime: layers from " + std::to_string(L0) + " need the stream of every token (stream_in)");
  CHECK(hipMemcpyAsync(h, stream_in->h.data() + stream_in_tok * nh, (size_t)M * nh * 4, hipMemcpyHostToDevice, s));
  CHECK(hipMemcpyAsync(pre, stream_in->pre.data() + stream_in_tok * np, (size_t)M * np * 4, hipMemcpyHostToDevice, s));
  stream_in_tok += M;
}

void Runtime::stream_to_host(int M) {
  const size_t nh = (size_t)hp.n_embd * hp.hc_mult, np = hp.hc_mult;
  stream_out.h.resize((size_t)M * nh);
  stream_out.pre.resize((size_t)M * np);
  CHECK(hipMemcpyAsync(stream_out.h.data(), h, (size_t)M * nh * 4, hipMemcpyDeviceToHost, s));
  CHECK(hipMemcpyAsync(stream_out.pre.data(), pre, (size_t)M * np * 4, hipMemcpyDeviceToHost, s));
  CHECK(hipStreamSynchronize(s));
}
