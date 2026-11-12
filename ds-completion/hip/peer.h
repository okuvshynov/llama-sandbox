// The link between two layer ranges of the model in two processes (README step 111).
//
// The front (layers 0 .. k-1) drives: it runs its layers for some tokens, then sends them with
// their residual stream; the back (layers k .., the head, the draft) runs its layers on that
// stream and answers where an answer exists. Every message mirrors one Runtime call, so both
// sides run the code a single process runs, in the same order:
//
//   message   payload to the back                     answer
//   HELLO     arg = kVersion                          n_layer, the back's first layer, its end, draft block
//   CHUNK     n tokens, their stream, arg = flags     -            (prompt tokens: no head)
//   DECODE    1 token, its stream, arg = flags        the logits, if wanted
//   VERIFY    n tokens, their stream, arg = want ids  n next ids
//   COMMIT    arg = tokens kept                       -
//   DRAFT     arg = the anchor token                  block + 1 ids, block confidences
//   MARK, REWIND                                      -
//   SYNC      -                                       an empty SYNC (a clock sample only)
//   BYE                                               -
//
// flags: kWantLogits, kInPrefill (the front's in_prefill, so the back's profile, trace and routing
// log treat a token as the front's do). Sizes are checked on both ends: an answer of another size
// than the one asked for, or a request of another size than its type's, is an error, not a guess.
//
// Messages without an answer are not waited for: the front goes on with its next chunk while
// the back works (TCP orders them). Plain blocking TCP with TCP_NODELAY; floats and ints as
// they are in memory (both ends are the same machine type), so the stream arrives bit for bit.
//
// One trace of both machines (README step 112): every answer carries the back's clock when the
// request had arrived and when the answer left. With the front's own times around it that is one
// sample of the two clocks' offset (as NTP takes them). Only the small symmetric exchanges are
// kept as samples (HELLO, SYNC, DRAFT: a verify's 64 KB per token up would bias it); each
// process writes its trace on its own clock and scripts/trace_merge.py puts the back's on the
// front's.
#pragma once

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct Peer {
  enum Type : uint32_t { HELLO = 1, CHUNK, DECODE, VERIFY, COMMIT, DRAFT, MARK, REWIND, BYE, SYNC };
  enum : int32_t { kVersion = 2 };                    // HELLO's arg: both ends must be the same build of this file
  enum : int32_t { kWantLogits = 1, kInPrefill = 2 };  // CHUNK and DECODE's arg
  static const char* name(uint32_t t) {
    static const char* n[] = {"?", "HELLO", "CHUNK", "DECODE", "VERIFY", "COMMIT", "DRAFT", "MARK", "REWIND", "BYE", "SYNC"};
    return t <= SYNC ? n[t] : n[0];
  }
  // this machine's monotonic clock, microseconds (what the traces' clockEpochUs is in)
  static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
  struct Header {
    uint32_t type = 0;
    uint32_t n = 0;    // tokens in the payload
    int32_t arg = 0;
    uint32_t bytes = 0;  // of the payload that follows
    double t_recv = 0, t_send = 0;  // answers: the back's clock at the request's arrival and at this answer's start
  };
  int fd = -1;
  long sent_bytes = 0, recv_bytes = 0, round_trips = 0;
  double wait_ms = 0;  // the front's time waiting for answers' headers (the back's work; not the answers' payloads)
  double last_sent_us = 0;   // the front: its clock when its last message was handed to the socket
  double reply_t_recv = 0;   // the back: its clock when the request being answered had arrived
  std::vector<std::array<double, 4>> clock_samples;  // the front: {sent, the back's arrival, the back's answer, received} per sample

  ~Peer() { if (fd >= 0) close(fd); }

  // the front: connect to "host:port"
  void connect_to(const std::string& host_port) {
    const size_t c = host_port.rfind(':');
    if (c == std::string::npos) throw std::runtime_error("peer: expected host:port, got " + host_port);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host_port.substr(0, c).c_str(), host_port.substr(c + 1).c_str(), &hints, &res) != 0 || !res)
      throw std::runtime_error("peer: cannot resolve " + host_port);
    fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    const int rc = fd < 0 ? -1 : connect(fd, res->ai_addr, res->ai_addrlen);
    freeaddrinfo(res);
    if (rc != 0) throw std::runtime_error("peer: cannot connect to " + host_port + ": " + strerror(errno));
    tune();
  }

  // the back: wait for one front on `port`
  void accept_on(int port) {
    const int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) throw std::runtime_error(std::string("peer: no socket: ") + strerror(errno));
    const int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(ls, (sockaddr*)&a, sizeof a) != 0 || listen(ls, 1) != 0)
      throw std::runtime_error("peer: cannot listen on port " + std::to_string(port) + ": " + strerror(errno));
    fd = accept(ls, nullptr, nullptr);
    close(ls);
    if (fd < 0) throw std::runtime_error("peer: accept failed");
    tune();
  }

  void tune() {
    const int one = 1, buf = 8 << 20;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);  // a machine that vanished ends the wait (after the system's keepalive time)
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
  }

  void put(const void* p, size_t n) {
    for (const char* c = (const char*)p; n > 0;) {
      const ssize_t w = ::send(fd, c, n, MSG_NOSIGNAL);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) throw std::runtime_error(std::string("peer: the connection closed while sending") + (w < 0 ? std::string(": ") + strerror(errno) : ""));
      c += w, n -= (size_t)w, sent_bytes += w;
    }
  }
  bool get(void* p, size_t n, bool eof_ok = false) {
    for (char* c = (char*)p; n > 0;) {
      const ssize_t r = ::recv(fd, c, n, 0);
      if (r == 0 && eof_ok && c == (char*)p) return false;
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) throw std::runtime_error(std::string("peer: the connection closed while receiving") + (r < 0 ? std::string(": ") + strerror(errno) : ""));
      c += r, n -= (size_t)r, recv_bytes += r;
    }
    return true;
  }

  // one message: the header, then the parts (each may be empty)
  struct Part { const void* p; size_t bytes; };
  void send_msg(Type type, uint32_t n, int32_t arg, std::initializer_list<Part> parts = {}) {
    Header hd;
    hd.type = type, hd.n = n, hd.arg = arg;
    size_t bytes = 0;
    for (const Part& x : parts) bytes += x.bytes;
    if (bytes > UINT32_MAX) throw std::runtime_error("peer: a message over 4 GB");
    hd.bytes = (uint32_t)bytes;
    hd.t_recv = reply_t_recv, hd.t_send = now_us();
    put(&hd, sizeof hd);
    for (const Part& x : parts)
      if (x.bytes) put(x.p, x.bytes);
    last_sent_us = now_us();
  }
  template <class T>
  std::vector<T> get_vec(size_t n) {
    std::vector<T> v(n);
    if (n) get(v.data(), n * sizeof(T));
    return v;
  }
};
