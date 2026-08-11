#pragma once

// The MoE shape a client and a backend must agree on, independent of which
// architecture supplied it.
//
// Not an abstraction invented for the sake of one: this is *exactly* the field
// set `moe_hello_response` already carries over the wire (moe_proto.h). The
// protocol committed to it before a second model existed; naming it here just
// stops the generic halves of lib/ — the RPC client and the routing trace —
// from having to include one model's hparams to read six integers.
//
// Each model provides a `moe_shape_of(its hparams)`. Nothing else is shared:
// the graph, the KV layout and the tensor names stay per-model, as
// lib/README.md argues they should.

#include <cstdint>
#include <string>

struct moe_shape {
    std::string arch;

    uint32_t n_embd        = 0;
    uint32_t n_layer       = 0;
    // Leading layers the backend does NOT serve, which the client evaluates
    // itself. For glm-dsa those are its dense layers; for deepseek4 they are
    // the hash-routed ones, whose experts come from a token-id lookup the
    // wire protocol does not carry. Same field, same meaning to the client.
    uint32_t n_dense_lead  = 0;
    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_ff_exp      = 0;

    float    expert_scale  = 1.0f;
    bool     expert_norm   = false;
};
