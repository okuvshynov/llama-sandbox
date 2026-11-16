// The runtime as one half of a model on two machines (README step 111; peer.h has the messages):
// the front (--remote) sends the back its tokens and stream and uses its answers as its own
// layers' results; the back (--serve) answers one front. The forward they wrap is in rt.cpp.
#include "rt.h"

#include "rt_internal.h"

void Runtime::peer_send_tokens(Peer::Type type, const std::vector<int32_t>& toks, int32_t arg) {
  const size_t n = toks.size(), nh = (size_t)hp.n_embd * hp.hc_mult, np = hp.hc_mult;
  if (stream_out.h.size() != n * nh || stream_out.pre.size() != n * np) throw std::runtime_error("peer: the stream does not match the tokens");
  const auto t0 = Clock::now();
  peer->send_msg(type, (uint32_t)n, arg, {{toks.data(), n * 4}, {stream_out.h.data(), n * nh * 4}, {stream_out.pre.data(), n * np * 4}});
  trace_host((std::string("send to the back: ") + Peer::name(type) + " " + std::to_string(n)).c_str(), t0, Clock::now(), 3);
}

// waits for the back's answer header; its payload (`bytes`, known to the caller) is read after
void Runtime::peer_answer(Peer::Type type, size_t bytes) {
  const auto t0 = Clock::now();
  Peer::Header hd;
  peer->get(&hd, sizeof hd);
  if (hd.type != type || hd.bytes != bytes)
    throw std::runtime_error(std::string("peer: expected ") + Peer::name(type) + " with " + std::to_string(bytes) + " bytes, got " + Peer::name(hd.type) + " with " +
                             std::to_string(hd.bytes) + " (the two halves are different builds?)");
  if (!trace_path.empty() && (type == Peer::HELLO || type == Peer::SYNC || type == Peer::DRAFT))
    peer->clock_samples.push_back({peer->last_sent_us, hd.t_recv, hd.t_send, Peer::now_us()});
  peer->wait_ms += ms_since(t0);
  peer->round_trips++;
  trace_host("waiting for the back", t0, Clock::now(), 3);
}

// a few clock samples (tracing only): at the start and the end of a run, so the merge can follow the clocks' drift
void Runtime::peer_sync(int n) {
  if (trace_path.empty()) return;
  for (int i = 0; i < n; i++) {
    peer->send_msg(Peer::SYNC, 0, 0);
    peer_answer(Peer::SYNC, 0);
  }
}

void Runtime::connect_peer(const std::string& host_port) {
  if (L0 != 0 || has_head()) throw std::runtime_error("peer: the front is a range from layer 0 without the last layer (--layers 0-k)");
  peer.reset(new Peer);
  peer->connect_to(host_port);
  peer->send_msg(Peer::HELLO, 0, Peer::kVersion);
  peer_answer(Peer::HELLO, 4 * 4);
  const std::vector<int32_t> v = peer->get_vec<int32_t>(4);
  if (v[0] != hp.n_layer || v[1] != L1 || v[2] != hp.n_layer)
    throw std::runtime_error("peer: the back runs layers " + std::to_string(v[1]) + "-" + std::to_string(v[2] - 1) + ", this front ends at " + std::to_string(L1 - 1));
  peer_block = v[3];
  peer_sync(8);
  fprintf(stderr, "# peer: layers %d-%d and the head at %s, draft block %d\n", v[1], v[2] - 1, host_port.c_str(), peer_block);
}

// The back: every message is one call of this runtime with the front's stream as its input
void Runtime::serve(int port) {
  if (L0 == 0 || !has_head()) throw std::runtime_error("peer: the back is a range up to the last layer (--layers k-39)");
  Peer p;
  fprintf(stderr, "# peer: layers %d-%d waiting for the front on port %d\n", L0, L1 - 1, port);
  p.accept_on(port);
  const size_t nh = (size_t)hp.n_embd * hp.hc_mult, np = hp.hc_mult;
  Stream in;
  stream_in = &in;
  long msgs = 0;
  double work_ms = 0;
  for (Peer::Header hd; p.get(&hd, sizeof hd, true) && hd.type != Peer::BYE; msgs++) {
    std::vector<int32_t> toks;
    const auto t_hd = Clock::now();
    if (hd.type == Peer::CHUNK || hd.type == Peer::DECODE || hd.type == Peer::VERIFY) {
      if (hd.n < 1 || hd.n > (uint32_t)kChunk || hd.bytes != hd.n * (4 + (nh + np) * 4)) throw std::runtime_error("peer: a message of the wrong size");
      toks = p.get_vec<int32_t>(hd.n);
      in.h.resize(hd.n * nh);
      in.pre.resize(hd.n * np);
      p.get(in.h.data(), in.h.size() * 4);
      p.get(in.pre.data(), in.pre.size() * 4);
      stream_in_tok = 0;
      trace_host((std::string("receive the stream: ") + Peer::name(hd.type) + " " + std::to_string(hd.n)).c_str(), t_hd, Clock::now(), 3);
    } else if (hd.bytes != 0) {
      throw std::runtime_error(std::string("peer: ") + Peer::name(hd.type) + " with a payload");
    }
    p.reply_t_recv = Peer::now_us();
    const auto t0 = Clock::now();
    switch (hd.type) {
      case Peer::SYNC: p.send_msg(Peer::SYNC, 0, 0); break;
      case Peer::HELLO: {
        if (hd.arg != Peer::kVersion) throw std::runtime_error("peer: the front is another build (its HELLO says " + std::to_string(hd.arg) + ")");
        const int32_t v[4] = {hp.n_layer, L0, L1, dspark ? dspark->block_size : 0};
        p.send_msg(Peer::HELLO, 0, 0, {{v, sizeof v}});
        break;
      }
      case Peer::CHUNK:
        in_prefill = hd.arg & Peer::kInPrefill;  // as on the front: what is profiled, traced and logged follows it
        prefill_chunk(toks);
        in_prefill = false;
        break;
      case Peer::DECODE: {
        if (hd.n != 1) throw std::runtime_error("peer: DECODE takes one token");
        in_prefill = hd.arg & Peer::kInPrefill;
        const std::vector<float> logits = decode(toks[0], hd.arg & Peer::kWantLogits);
        in_prefill = false;
        if (hd.arg & Peer::kWantLogits) p.send_msg(Peer::DECODE, 0, 0, {{logits.data(), logits.size() * 4}});
        break;
      }
      case Peer::VERIFY: {
        const std::vector<int32_t> next = verify(toks, hd.arg != 0);
        p.send_msg(Peer::VERIFY, hd.n, 0, {{next.data(), next.size() * 4}});
        break;
      }
      case Peer::COMMIT: commit(hd.arg); break;
      case Peer::MARK: mark(); break;
      case Peer::REWIND: rewind(); break;
      case Peer::DRAFT: {
        const DraftResult r = draft(hd.arg);
        p.send_msg(Peer::DRAFT, 0, 0, {{r.ids.data(), r.ids.size() * 4}, {r.conf.data(), r.conf.size() * 4}});
        break;
      }
      default: throw std::runtime_error("peer: unknown message " + std::to_string(hd.type));
    }
    work_ms += ms_since(t0);
    trace_host((std::string("serve ") + Peer::name(hd.type)).c_str(), t_hd, Clock::now(), 4);
  }
  stream_in = nullptr;
  fprintf(stderr, "# peer: the front left after %ld messages; %.1f MB received, %.1f MB sent; %.2f s in this range's work\n", msgs,
          p.recv_bytes / 1e6, p.sent_bytes / 1e6, work_ms / 1e3);
}

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
