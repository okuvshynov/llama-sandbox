#pragma once

// Wire protocol between the nano-glm trunk and the MoE backend, plus the
// minimum portable TCP needed to speak it. See ../PLAN.md "Protocol v1".
//
// One request is one MoE layer for one batch: the trunk sends the post-norm
// activation, the backend routes it, evaluates the selected experts, combines
// them, and sends back one row per token. Expert ids and routing weights never
// cross the wire — the router lives on the backend.
//
// Discipline borrowed from the lkldtopk format: magic, version, explicit dims
// in every header, and hard errors on mismatch. A protocol that silently
// misparses is indistinguishable from a model that is subtly wrong, and this
// whole project exists to tell those apart.
//
// NOTE: include this BEFORE nano_model.h. winsock2.h must precede windows.h,
// and nano_model.h pulls the latter in.

#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <winsock2.h>
#   include <ws2tcpip.h>
    using moe_socket = SOCKET;
#   define MOE_INVALID_SOCKET INVALID_SOCKET
#else
#   include <arpa/inet.h>
#   include <netinet/in.h>
#   include <netinet/tcp.h>
#   include <sys/socket.h>
#   include <unistd.h>
    using moe_socket = int;
#   define MOE_INVALID_SOCKET (-1)
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

static constexpr uint32_t MOE_MAGIC   = 0x454F4D4Eu;  // "NMOE" little-endian
static constexpr uint32_t MOE_VERSION = 2;            // v2 added the hello handshake

enum moe_msg_type : uint32_t {
    MOE_MSG_REQUEST  = 1,
    MOE_MSG_RESPONSE = 2,
    MOE_MSG_HELLO    = 3,   // client -> server on connect, server replies in kind
};

enum moe_return_mode : uint32_t {
    MOE_RET_COMBINED = 0,  // one row per token: experts weighted and summed
    MOE_RET_PER_SLOT = 1,  // n_used rows per token + the expert ids (debug)
};

enum moe_status : uint32_t {
    MOE_OK             = 0,
    MOE_ERR_VERSION    = 1,
    MOE_ERR_DIMS       = 2,
    MOE_ERR_LAYER      = 3,
    MOE_ERR_MODE       = 4,
    MOE_ERR_INTERNAL   = 5,
};

// ---------------------------------------------------------------------------
// hello: who is on the other end of this socket
//
// The trunk and the backend each hold half of one model and neither can see
// the other's half. Per-request validation catches only n_embd and the layer
// range, which two entirely different models can share — so without this,
// pointing --moe-addr at the wrong server produces fluent, confident, wrong
// output, which is the worst failure mode this project has.
//
// A mismatch means three different things, and the split is the point:
//
//   structural (below, binary)  the client's graph *assumes* these. Always
//                               fatal; no flag permits continuing.
//   reproducibility (in the     valid to run, but bit-exactness is void.
//   text payload)               Fatal only under --strict, because Q4_K
//                               experts against a Q6_K trunk is a planned
//                               configuration (PLAN.md step 3), not an error.
//   informational               printed, never enforced.
//
// Enforcement is client-side only: the server logs what connected and serves,
// since it cannot know what the operator intended.
//
// Structural fields are binary rather than text so they cannot be misparsed
// into agreement. Everything else is `key=value` lines from build_info.h,
// which keeps adding a field from being another protocol version.

struct moe_hello_request {
    uint32_t magic;
    uint32_t version;
    uint32_t msg_type;      // MOE_MSG_HELLO
    uint32_t reserved;
    uint64_t payload_bytes; // client fingerprint, key=value lines
};

struct moe_hello_response {
    uint32_t magic;
    uint32_t version;
    uint32_t msg_type;      // MOE_MSG_HELLO
    uint32_t status;
    char     arch[16];      // "glm-dsa"
    uint32_t n_embd;
    uint32_t n_layer;
    uint32_t n_dense_lead;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_ff_exp;
    float    expert_scale;
    uint32_t expert_norm;
    uint32_t reserved[2];
    uint64_t payload_bytes; // server fingerprint, key=value lines
};

static_assert(sizeof(moe_hello_request)  == 24, "hello request layout changed");
static_assert(sizeof(moe_hello_response) == 80, "hello response layout changed");

// Both headers are fixed-size and fully explicit; payload_bytes is what
// follows immediately after.
struct moe_request_header {
    uint32_t magic;
    uint32_t version;
    uint32_t msg_type;      // MOE_MSG_REQUEST
    uint32_t layer;         // model layer index
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t return_mode;
    uint32_t reserved;
    uint64_t payload_bytes; // n_embd * n_tokens * sizeof(float)
};

struct moe_response_header {
    uint32_t magic;
    uint32_t version;
    uint32_t msg_type;      // MOE_MSG_RESPONSE
    uint32_t status;        // moe_status; payload is an error string if != OK
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t n_slots;       // 0 when combined, n_expert_used when per-slot
    uint32_t reserved;
    // Durations measured on the server's own clock. Never absolute timestamps:
    // the client subtracts server_total from its RTT to get network+queueing,
    // so the two machines need no clock sync.
    uint32_t t_parse_us;
    uint32_t t_route_us;
    uint32_t t_compute_us;
    uint32_t t_serialize_us;
    uint64_t payload_bytes;
};

static_assert(sizeof(moe_request_header)  == 40, "request header layout changed");
static_assert(sizeof(moe_response_header) == 56, "response header layout changed");

// ---------------------------------------------------------------------------
// transport

inline bool moe_net_init() {
#if defined(_WIN32)
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#else
    return true;
#endif
}

inline void moe_close(moe_socket s) {
    if (s == MOE_INVALID_SOCKET) return;
#if defined(_WIN32)
    closesocket(s);
#else
    close(s);
#endif
}

inline std::string moe_net_error() {
#if defined(_WIN32)
    return std::to_string(WSAGetLastError());
#else
    return std::string(strerror(errno));
#endif
}

// Latency, not throughput, is what this protocol is judged on: 75 strictly
// sequential round trips per token. Nagle would add up to 40ms to each.
inline void moe_set_nodelay(moe_socket s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
}

inline bool moe_send_all(moe_socket s, const void * data, size_t n) {
    const char * p = (const char *) data;
    while (n > 0) {
        int chunk = (int) (n > (1 << 20) ? (1 << 20) : n);
        int sent = send(s, p, chunk, 0);
        if (sent <= 0) return false;
        p += sent;
        n -= (size_t) sent;
    }
    return true;
}

inline bool moe_recv_all(moe_socket s, void * data, size_t n) {
    char * p = (char *) data;
    while (n > 0) {
        int chunk = (int) (n > (1 << 20) ? (1 << 20) : n);
        int got = recv(s, p, chunk, 0);
        if (got <= 0) return false;   // 0 = peer closed
        p += got;
        n -= (size_t) got;
    }
    return true;
}

// Blocking connect to host:port. Returns MOE_INVALID_SOCKET on failure.
inline moe_socket moe_connect(const std::string & host, uint16_t port) {
    moe_socket s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == MOE_INVALID_SOCKET) return MOE_INVALID_SOCKET;

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        moe_close(s);
        return MOE_INVALID_SOCKET;
    }
    if (connect(s, (sockaddr *) &addr, sizeof(addr)) != 0) {
        moe_close(s);
        return MOE_INVALID_SOCKET;
    }
    moe_set_nodelay(s);
    return s;
}

// Listening socket bound to host:port.
inline moe_socket moe_listen(const std::string & host, uint16_t port) {
    moe_socket s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == MOE_INVALID_SOCKET) return MOE_INVALID_SOCKET;

    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        moe_close(s);
        return MOE_INVALID_SOCKET;
    }
    if (bind(s, (sockaddr *) &addr, sizeof(addr)) != 0 || listen(s, 4) != 0) {
        moe_close(s);
        return MOE_INVALID_SOCKET;
    }
    return s;
}
