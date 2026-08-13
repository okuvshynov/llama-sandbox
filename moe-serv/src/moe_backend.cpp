// moeserv — a ggml backend that llama.cpp loads at runtime.
//
// `handshake` (PLAN.md): register, expose a buffer type called `MoE`, and claim
// nothing. `supports_op` is false for every op, so a run with this library
// loaded must be indistinguishable from one without it. That is the whole test
// — a backend that changes behaviour before it claims anything has a side
// effect in its registration, and everything after would be built on it.
//
// Loaded by exporting `ggml_backend_init` and `ggml_backend_score`, which
// `GGML_BACKEND_DL_IMPL` generates from the two functions at the bottom. The
// loader (`ggml/src/ggml-backend-reg.cpp`) reads `GGML_BACKEND_PATH`, dlopens
// the library, checks the score and calls init.
//
// The three tiers ggml wants are all here and all minimal:
//
//   reg     names the backend and enumerates its devices (one)
//   device  answers "what are you" and "can you do this op" (currently: no)
//   buffer  owns memory; for now plain host malloc, so the CPU backend can read
//           it directly and `passthrough` can delegate arithmetic to it
//
// Nothing here is llama.cpp-specific and nothing patches llama.cpp. If that
// ever stops being true it is a finding to write down, not a patch to carry.

#include "ggml-backend-impl.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define MOESERV_NAME "MoE"

// ---------------------------------------------------------------------------
// buffer — plain host memory
//
// Deliberately the simplest thing that can hold a weight: malloc, memcpy,
// `is_host` true. Two reasons it is not a placeholder. The loader writes
// weights in through `set_tensor` and a host buffer makes that a memcpy from
// the mmap; and `passthrough` computes by handing the subgraph to a
// `ggml_backend_cpu` instance, which can only read our tensors if they are
// ordinary host memory in the standard layout. Device-resident weights are a
// later decision (`dies`), and the buffer type is where it will be made.

struct moeserv_buffer {
    void * data = nullptr;
    size_t size = 0;
};

static void moeserv_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (moeserv_buffer *) buffer->context;
    free(ctx->data);
    delete ctx;
}

static void * moeserv_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((moeserv_buffer *) buffer->context)->data;
}

static void moeserv_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                         uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

static void moeserv_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                      const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void moeserv_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                      void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

// Accept a copy from any host buffer. Returning false is legal and makes the
// caller fall back to get/set through a staging buffer, so this is an
// optimisation rather than a requirement — but the model loader copies a lot,
// and "we are host memory" is exactly the case where it is free.
static bool moeserv_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void moeserv_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (moeserv_buffer *) buffer->context;
    memset(ctx->data, value, ctx->size);
}

static const ggml_backend_buffer_i moeserv_buffer_i = {
    /* .free_buffer   = */ moeserv_buffer_free,
    /* .get_base      = */ moeserv_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ moeserv_buffer_memset_tensor,
    /* .set_tensor    = */ moeserv_buffer_set_tensor,
    /* .get_tensor    = */ moeserv_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ moeserv_buffer_cpy_tensor,
    /* .clear         = */ moeserv_buffer_clear,
    /* .reset         = */ nullptr,
};

// ---------------------------------------------------------------------------
// buffer type — this is the name `-ot` matches
//
// `-ot <regex>=<name>` resolves <name> against every registered device's buffer
// type (`common/arg.cpp`, and llama-bench's own copy of the same loop), so this
// string is the entire public interface for placing tensors here:
//
//     -ot "\.ffn_(up|down|gate|gate_up)_(ch|)exps=MoE"
//
// which is `-cmoe`'s regex pointed at us instead of the CPU.

static const char * moeserv_buft_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return MOESERV_NAME;
}

static ggml_backend_buffer_t moeserv_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new moeserv_buffer;
    // ggml asks for zero-sized buffers for empty contexts; malloc(0) may return
    // null, which would read as failure.
    ctx->data = size ? malloc(size) : nullptr;
    ctx->size = size;
    if (size && !ctx->data) {
        fprintf(stderr, "%s: failed to allocate %zu bytes\n", MOESERV_NAME, size);
        delete ctx;
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, moeserv_buffer_i, ctx, size);
}

static size_t moeserv_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;   // enough for every vector unit ggml's CPU kernels use
}

static bool moeserv_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static ggml_backend_buffer_type_t moeserv_buffer_type();

// ---------------------------------------------------------------------------
// device

static const char * moeserv_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return MOESERV_NAME;
}

static const char * moeserv_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "MoE expert block (moe-serv)";
}

static void moeserv_device_get_memory(ggml_backend_dev_t dev, size_t * free_, size_t * total) {
    GGML_UNUSED(dev);
    // Host memory, and we do not track a budget yet. Reporting 0/0 is the
    // convention for "no device memory to report" (see ggml-backend-impl.h) and
    // is what BLAS does.
    *free_ = 0;
    *total = 0;
}

// ACCEL, not GPU. The distinction decides where llama.cpp puts us: ACCEL
// backends are added to the scheduler's list after the GPUs and before the CPU
// (`llama-context.cpp`), which is the priority we want, while a GPU device
// would invite `-ngl` to assign whole layers to something that cannot run a
// trunk. BLAS makes the same choice for the same reason.
// `enum` is required, not stylistic: ggml-backend.h declares both an enum
// `ggml_backend_dev_type` and a *function* of the same name. C keeps those in
// separate namespaces; C++ lets the function hide the type, so the bare name
// does not compile. ggml's own backends write it this way for the same reason.
static enum ggml_backend_dev_type moeserv_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void moeserv_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = moeserv_device_get_name(dev);
    props->description = moeserv_device_get_description(dev);
    props->type        = moeserv_device_get_type(dev);
    moeserv_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

// `handshake`: claim nothing at all.
//
// This is the load-bearing line of the increment. With it returning false, a
// weight can be routed here by `-ot` and llama.cpp will still refuse to place
// it (`weight_buft_supported` asks whether the owning device can run the op),
// so the run falls back to the CPU and must be bit-identical to not loading
// this library. `passthrough` narrows it to the MoE block.
static bool moeserv_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false;
}

static bool moeserv_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft == moeserv_buffer_type();
}

static ggml_backend_t moeserv_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t moeserv_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return moeserv_buffer_type();
}

static const ggml_backend_device_i moeserv_device_i = {
    /* .get_name             = */ moeserv_device_get_name,
    /* .get_description      = */ moeserv_device_get_description,
    /* .get_memory           = */ moeserv_device_get_memory,
    /* .get_type             = */ moeserv_device_get_type,
    /* .get_props            = */ moeserv_device_get_props,
    /* .init_backend         = */ moeserv_device_init_backend,
    /* .get_buffer_type      = */ moeserv_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ moeserv_device_supports_op,
    /* .supports_buft        = */ moeserv_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---------------------------------------------------------------------------
// backend (stream)

static const char * moeserv_backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return MOESERV_NAME;
}

static void moeserv_backend_free(ggml_backend_t backend) {
    delete backend;
}

// Unreachable while `supports_op` is false — the scheduler never assigns us a
// node, so it never builds a split for us. Aborting rather than returning an
// error because reaching here means the scheduler and `supports_op` disagree,
// and a wrong answer computed quietly is worse than a stop.
static enum ggml_status moeserv_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    fprintf(stderr, "%s: graph_compute called with %d nodes, but this build claims no ops\n",
            MOESERV_NAME, cgraph ? ggml_graph_n_nodes(cgraph) : 0);
    return GGML_STATUS_FAILED;
}

static const ggml_backend_i moeserv_backend_i = {
    /* .get_name             = */ moeserv_backend_get_name,
    /* .free                 = */ moeserv_backend_free,
    /* .set_tensor_async     = */ nullptr,
    /* .get_tensor_async     = */ nullptr,
    /* .set_tensor_2d_async  = */ nullptr,
    /* .get_tensor_2d_async  = */ nullptr,
    /* .cpy_tensor_async     = */ nullptr,
    /* .synchronize          = */ nullptr,
    /* .graph_plan_create    = */ nullptr,
    /* .graph_plan_free      = */ nullptr,
    /* .graph_plan_update    = */ nullptr,
    /* .graph_plan_compute   = */ nullptr,
    /* .graph_compute        = */ moeserv_backend_graph_compute,
    /* .event_record         = */ nullptr,
    /* .event_wait           = */ nullptr,
    /* .graph_optimize       = */ nullptr,
};

static ggml_guid_t moeserv_guid() {
    // Arbitrary but fixed: ggml identifies a backend instance by guid.
    static ggml_guid guid = { 0x6d, 0x6f, 0x65, 0x73, 0x65, 0x72, 0x76, 0x00,
                              0x4d, 0x6f, 0x45, 0x62, 0x6b, 0x6e, 0x64, 0x01 };
    return &guid;
}

// ---------------------------------------------------------------------------
// reg — one device, and the singletons everything above refers to

static const char * moeserv_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return MOESERV_NAME;
}

static size_t moeserv_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_reg_t moeserv_reg();

static ggml_backend_dev_t moeserv_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    GGML_UNUSED(index);
    static ggml_backend_device dev = {
        /* .iface   = */ moeserv_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    dev.reg = reg;
    return &dev;
}

static const ggml_backend_reg_i moeserv_reg_i = {
    /* .get_name         = */ moeserv_reg_get_name,
    /* .get_device_count = */ moeserv_reg_get_device_count,
    /* .get_device       = */ moeserv_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

static ggml_backend_reg_t moeserv_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ moeserv_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

static ggml_backend_buffer_type_t moeserv_buffer_type() {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ moeserv_buft_get_name,
            /* .alloc_buffer   = */ moeserv_buft_alloc_buffer,
            /* .get_alignment  = */ moeserv_buft_get_alignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ moeserv_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    buft.device = moeserv_reg_get_device(moeserv_reg(), 0);
    return &buft;
}

static ggml_backend_t moeserv_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    return new ggml_backend {
        /* .guid    = */ moeserv_guid(),
        /* .iface   = */ moeserv_backend_i,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
}

// ---------------------------------------------------------------------------
// entry points
//
// `ggml_backend_score` gates loading: 0 means "not usable on this system" and
// the loader skips the library without an error. There is nothing to detect yet
// — host memory always works — so this is unconditional, and it becomes a real
// probe when `dies` needs a Vulkan device to exist.

static int moeserv_score() {
    return 1;
}

extern "C" GGML_BACKEND_API ggml_backend_reg_t moeserv_backend_reg(void) {
    return moeserv_reg();
}

GGML_BACKEND_DL_IMPL(moeserv_backend_reg)
GGML_BACKEND_DL_SCORE_IMPL(moeserv_score)
