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

// Is this tensor's storage ours? During a real graph that means the model
// loader put it here via `-ot`; during `weight_buft_supported`'s probe it means
// llama.cpp attached a temporary zero-size buffer of our type to ask the
// question (`llama-model-loader.cpp`). Both should answer yes.
static bool moeserv_is_ours(const ggml_tensor * t) {
    if (!t) return false;
    const ggml_tensor * s = t->view_src ? t->view_src : t;
    return s->buffer != nullptr && s->buffer->buft == moeserv_buffer_type();
}

// Does this op's value descend from a matmul against weights we own?
//
// The elementwise ops of the expert block — the SwiGLU clamps, the gate, the
// router-weight multiply — carry no weights, so "is the weight ours" cannot
// answer for them and claiming them unconditionally would claim every `mul` and
// `clamp` in the model. Walking a few links up the source chain is precise
// instead: `clamp(up)` where `up = mul_mat_id(our_weights, ...)` is ours, and a
// `mul` in the attention block is not.
//
// Depth 4 covers the longest chain in the block, mul_mat_id -> clamp -> glu ->
// mul_mat_id -> mul, and bounds the walk on a graph where a source chain can
// otherwise be long.
static bool moeserv_derives_from_ours(const ggml_tensor * t, int depth) {
    if (!t || depth <= 0) return false;
    if (t->op == GGML_OP_MUL_MAT_ID && moeserv_is_ours(t->src[0])) return true;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (moeserv_derives_from_ours(t->src[i], depth - 1)) return true;
    }
    return false;
}

// `passthrough`: claim the routed expert block and nothing else.
//
// The set is the expert half of llama.cpp's `build_moe_ffn`, minus the expert
// sum. `ADD` is deliberately absent: it is also the residual add, and pass 2 of
// the scheduler does not reset its running backend id when it meets an op it
// cannot place (`ggml_backend_sched_set_if_supported`), so a claim on `ADD`
// could reach past the block into the trunk. The sum is `n_expert_used - 1`
// adds of [n_embd, n_tokens] and costs little on the CPU.
//
// Shapes neither of our two models uses are refused rather than mishandled:
// fused `gate_up_exps` (caught by the src0 check, since a fused block's weights
// would still be ours — so it is caught by the ne check below instead), expert
// bias `ADD_ID`, and the non-SwiGLU gates. An unported model falls back to the
// CPU and is slow, not wrong.
static bool moeserv_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (!op) return false;

    switch (op->op) {
        // Free structural ops, as BLAS does: claiming them keeps a view from
        // splitting a run of nodes that is otherwise ours. Pass 2 skips view
        // ops entirely, so this cannot pull in anything distant.
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT_ID:
            return moeserv_is_ours(op->src[0]);

        case GGML_OP_CLAMP:
        case GGML_OP_MUL:
            return moeserv_derives_from_ours(op, 4);

        // ggml_swiglu_split and friends are all GGML_OP_GLU; the variant is in
        // op_params. Only SwiGLU is claimed — the others are untested here.
        case GGML_OP_GLU:
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
                   moeserv_derives_from_ours(op, 4);

        default:
            return false;
    }
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

// The CPU backend we delegate to, taken from the *host's* registry rather than
// linked. Linking ggml-cpu would put a second copy of the kernels in this
// library; asking the registry gets the same instance type llama.cpp is using,
// built with the same flags, which is what makes delegation bit-identical
// rather than merely correct.
static ggml_backend_t moeserv_cpu() {
    static ggml_backend_t cpu = nullptr;
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!cpu) {
            fprintf(stderr, "%s: no CPU backend in the registry to delegate to\n", MOESERV_NAME);
        }
    }
    return cpu;
}

// `passthrough` computes by handing the split to that CPU backend. Every tensor
// involved is host memory — ours by `-ot`, everything else llama.cpp's — so the
// CPU kernels read them in place with no copy, and the arithmetic is the same
// arithmetic in the same order as a run without this library.
//
// Reported once, because the split's contents are the whole claim of this
// increment: if the block did not arrive as one piece, or if something outside
// it did, this line says so without needing GGML_SCHED_DEBUG.
static enum ggml_status moeserv_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    if (!cgraph) return GGML_STATUS_FAILED;

    static bool reported = false;
    if (!reported) {
        reported = true;
        int counts[GGML_OP_COUNT] = { 0 };
        const int n = ggml_graph_n_nodes(cgraph);
        for (int i = 0; i < n; i++) counts[ggml_graph_node(cgraph, i)->op]++;
        fprintf(stderr, "%s: first split has %d nodes:", MOESERV_NAME, n);
        for (int o = 0; o < GGML_OP_COUNT; o++) {
            if (counts[o]) fprintf(stderr, " %s x%d", ggml_op_name((enum ggml_op) o), counts[o]);
        }
        fprintf(stderr, "\n");
    }

    ggml_backend_t cpu = moeserv_cpu();
    if (!cpu) return GGML_STATUS_FAILED;
    return ggml_backend_graph_compute(cpu, cgraph);
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

// Thread count, forwarded to the CPU backend we delegate to.
//
// This matters for correctness, not tidiness. ggml partitions matmul work by
// `n_threads`, so the summation structure — and therefore the rounding —
// changes with it (repo CLAUDE.md). A delegated expert matmul running on a
// different thread count than the rest of the model would not be bit-identical
// to a stock run, and the whole point of `passthrough` is that it is.
//
// llama.cpp asks every backend's reg for this symbol and calls it with the
// same value it uses everywhere (`llama-context.cpp`), so exposing it is how we
// learn a number we otherwise could not see.
static void moeserv_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_UNUSED(backend);
    ggml_backend_t cpu = moeserv_cpu();
    if (!cpu) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) return;
    auto fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(cpu, n_threads);
}

static void * moeserv_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (name && strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) moeserv_set_n_threads;
    }
    return nullptr;
}

static const ggml_backend_reg_i moeserv_reg_i = {
    /* .get_name         = */ moeserv_reg_get_name,
    /* .get_device_count = */ moeserv_reg_get_device_count,
    /* .get_device       = */ moeserv_reg_get_device,
    /* .get_proc_address = */ moeserv_reg_get_proc_address,
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
