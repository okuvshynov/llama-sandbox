// The link between two layer ranges of the model in two processes (README step 111).
//
// The front (layers 0 .. k-1) drives: it runs its layers for some tokens, then sends them with
// their residual stream; the back (layers k .., the head, the draft) runs its layers on that
// stream and answers where an answer exists. Every message mirrors one Runtime call, so both
// sides run the code a single process runs, in the same order:
//
//   message   payload to the back                     answer
//   HELLO     -                                       n_layer, the back's first layer, draft block
//   CHUNK     n tokens, their stream                  -            (prompt tokens: no head)
//   DECODE    1 token, its stream, arg = want logits  the logits, if wanted
//   VERIFY    n tokens, their stream                  n next ids
//   COMMIT    arg = tokens kept                       -
//   DRAFT     arg = the anchor token                  block + 1 ids, block confidences
//   MARK, REWIND                                      -
//   BYE                                               -
//
// Messages without an answer are not waited for: the front goes on with its next chunk while
// the back works (TCP orders them). Plain blocking TCP with TCP_NODELAY; floats and ints as
// they are in memory (both ends are the same machine type), so the stream arrives bit for bit.
#pragma once

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct Peer {
  enum Type : uint32_t { HELLO = 1, CHUNK, DECODE, VERIFY, COMMIT, DRAFT, MARK, REWIND, BYE };
  struct Header {
    uint32_t type = 0;
    uint32_t n = 0;    // tokens in the payload
    int32_t arg = 0;
    uint32_t bytes = 0;  // of the payload that follows
  };
  int fd = -1;
  long sent_bytes = 0, recv_bytes = 0, round_trips = 0;
  double wait_ms = 0;  // the front's time between a request's last byte and its answer's last

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
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf);
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
  }

  void put(const void* p, size_t n) {
    for (const char* c = (const char*)p; n > 0;) {
      const ssize_t w = ::send(fd, c, n, MSG_NOSIGNAL);
      if (w <= 0) throw std::runtime_error("peer: the connection closed while sending");
      c += w, n -= (size_t)w, sent_bytes += w;
    }
  }
  bool get(void* p, size_t n, bool eof_ok = false) {
    for (char* c = (char*)p; n > 0;) {
      const ssize_t r = ::recv(fd, c, n, 0);
      if (r == 0 && eof_ok && c == (char*)p) return false;
      if (r <= 0) throw std::runtime_error("peer: the connection closed while receiving");
      c += r, n -= (size_t)r, recv_bytes += r;
    }
    return true;
  }

  // one message: the header, then the parts (each may be empty)
  struct Part { const void* p; size_t bytes; };
  void send_msg(Type type, uint32_t n, int32_t arg, std::initializer_list<Part> parts = {}) {
    Header hd;
    hd.type = type, hd.n = n, hd.arg = arg;
    for (const Part& x : parts) hd.bytes += (uint32_t)x.bytes;
    put(&hd, sizeof hd);
    for (const Part& x : parts)
      if (x.bytes) put(x.p, x.bytes);
  }
  template <class T>
  std::vector<T> get_vec(size_t n) {
    std::vector<T> v(n);
    if (n) get(v.data(), n * sizeof(T));
    return v;
  }
};
