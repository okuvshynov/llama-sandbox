#pragma once

// TP-within-expert execution of the decode block on the four dies, inside the
// backend. MOESERV_TP=1 enables it; off, nothing here runs.
//
// The design is E9/e109151, measured in the probe at 113 µs per die per layer:
// every die holds columns [I*d, I*(d+1)) of gate|up (fused, one m=2I matmul)
// and the matching k-rows of down, for ALL experts of a resident layer. Per
// call: upload x/ids/router-weights to each die, four dispatches per die,
// read four partial [mout] per slot back, sum on the host into the MUL node's
// output — exact, since the split is along down's reduction dimension.
//
// Scope guards, all of which mean "CPU fallback", never "wrong":
//   - one token per call (decode); anything else falls back
//   - the split must parse as exactly the deepseek4 expert block:
//     mul_mat_id -> clamp -> (x2) -> swiglu_split(not swapped) -> mul_mat_id
//     -> mul(router weights); anything else falls back
//   - a layer whose slices did not fit the per-die budget falls back
//
// Raw Vulkan, own instance — the host's ggml-vulkan instance is unrelated and
// the two coexist. Shaders are loaded from a `shaders/` directory next to
// moeserv.dll (override: MOESERV_SHADERS).

#include "ggml.h"

#include <vulkan/vulkan.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define MOE_TP_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "MoE-TP: vulkan error %d at %s:%d\n", (int) r_, __FILE__, __LINE__); return false; } } while (0)

struct moe_tp_buf {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t         size = 0;
};

struct moe_tp_pipeline {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipe = VK_NULL_HANDLE;
};

// Per-(layer, die) resources: the weight planes, their descriptor sets, and a
// command buffer recorded once — per call only the staging contents change.
struct moe_tp_layer_die {
    moe_tp_buf gu_plane, gu_scale, dn_plane, dn_scale;
    VkDescriptorSet ds_gu = VK_NULL_HANDLE, ds_dn = VK_NULL_HANDLE,
                    ds_mid = VK_NULL_HANDLE, ds_fin = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
};

struct moe_tp_layer {
    bool  resident = false;
    float gmin = 0, gmax = 0, umin = 0, umax = 0;   // baked into the CBs
    std::vector<moe_tp_layer_die> per_die;
};

struct moe_tp_die {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice         dev = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    uint32_t         qfam = 0;
    VkCommandPool    pool = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkFence          fence = VK_NULL_HANDLE;
    moe_tp_pipeline  p1, pmid, pfin;
    // Per-call scratch, shared by all layers of this die.
    moe_tp_buf io;                                  // host-visible: x | ids | wts | y readback
    void *     io_ptr = nullptr;                    // persistently mapped — vkMapMemory per
                                                    // call cost ~2 ms/layer on this driver
    moe_tp_buf x_dev, ids_dev, wts_dev, h, part_gu, part_dn, y;
    moe_tp_buf staging;                             // for weight uploads
    size_t     budget = 0;                          // device-local bytes still allowed
};

struct moe_tp {
    bool checked = false, on = false, alive = false, dims_set = false;
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<moe_tp_die> dies;
    std::map<int, moe_tp_layer> layers;
    int64_t K = 0, INTER = 0, MOUT = 0, SLOTS = 0, ISL = 0;
    int64_t tile_k = 128;
    uint64_t n_calls = 0, n_fallback = 0;
    // Per-call border accounting, µs — the probe timed the pipeline inside one
    // submission and could not see any of this.
    double t_stage = 0, t_submit = 0, t_wait = 0, t_sum = 0;
    std::string shader_dir;
    // io buffer offsets
    size_t off_x = 0, off_ids = 0, off_wts = 0, off_y = 0, io_size = 0;
};

// What one received split parses into. Everything is a host pointer into
// llama.cpp's (or our) buffers — the split's inputs and outputs are host
// tensors by construction.
struct moe_tp_call {
    const ggml_tensor * gate_w, * up_w, * down_w;
    const ggml_tensor * x, * ids, * wts;
    ggml_tensor       * out;                        // the MUL node; host sum target
    float gmin, gmax, umin, umax;
    int   layer;
    int64_t n_slots;
};

// ---------------------------------------------------------------------------
// small helpers (the probe's, trimmed)

static inline uint32_t moe_tp_mem_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return UINT32_MAX;
}

static inline bool moe_tp_make_buf(moe_tp_die & D, size_t size, VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags props, moe_tp_buf & b) {
    b.size = size;
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    MOE_TP_CHECK(vkCreateBuffer(D.dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(D.dev, b.buf, &req);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = moe_tp_mem_type(D.phys, req.memoryTypeBits, props);
    if (ai.memoryTypeIndex == UINT32_MAX) return false;
    MOE_TP_CHECK(vkAllocateMemory(D.dev, &ai, nullptr, &b.mem));
    MOE_TP_CHECK(vkBindBufferMemory(D.dev, b.buf, b.mem, 0));
    return true;
}

static inline void moe_tp_free_buf(moe_tp_die & D, moe_tp_buf & b) {
    if (b.buf) vkDestroyBuffer(D.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(D.dev, b.mem, nullptr);
    b = moe_tp_buf{};
}

// Chunked staged upload for weight planes (the staging buffer is smaller than
// a plane).
static inline bool moe_tp_upload(moe_tp_die & D, moe_tp_buf & dst, const void * data, size_t size) {
    size_t done = 0;
    while (done < size) {
        const size_t n = size - done < D.staging.size ? size - done : D.staging.size;
        void * p = nullptr;
        MOE_TP_CHECK(vkMapMemory(D.dev, D.staging.mem, 0, n, 0, &p));
        memcpy(p, (const char *) data + done, n);
        vkUnmapMemory(D.dev, D.staging.mem);

        VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = D.pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb;
        MOE_TP_CHECK(vkAllocateCommandBuffers(D.dev, &cai, &cb));
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        MOE_TP_CHECK(vkBeginCommandBuffer(cb, &bi));
        VkBufferCopy region = { 0, done, n };
        vkCmdCopyBuffer(cb, D.staging.buf, dst.buf, 1, &region);
        MOE_TP_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        MOE_TP_CHECK(vkQueueSubmit(D.queue, 1, &si, VK_NULL_HANDLE));
        MOE_TP_CHECK(vkQueueWaitIdle(D.queue));
        vkFreeCommandBuffers(D.dev, D.pool, 1, &cb);
        done += n;
    }
    return true;
}

static inline bool moe_tp_load_pipeline(moe_tp_die & D, const std::string & path,
                                        uint32_t n_bindings, uint32_t push_bytes,
                                        moe_tp_pipeline & P) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "MoE-TP: cannot open %s\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code((n + 3) / 4);
    if (fread(code.data(), 1, n, f) != (size_t) n) { fclose(f); return false; }
    fclose(f);
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = (size_t) n;
    ci.pCode = code.data();
    VkShaderModule mod;
    MOE_TP_CHECK(vkCreateShaderModule(D.dev, &ci, nullptr, &mod));

    std::vector<VkDescriptorSetLayoutBinding> binds(n_bindings);
    for (uint32_t i = 0; i < n_bindings; i++) {
        binds[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    VkDescriptorSetLayoutCreateInfo di = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    di.bindingCount = n_bindings;
    di.pBindings = binds.data();
    MOE_TP_CHECK(vkCreateDescriptorSetLayout(D.dev, &di, nullptr, &P.dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.setLayoutCount = 1;
    li.pSetLayouts = &P.dsl;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    MOE_TP_CHECK(vkCreatePipelineLayout(D.dev, &li, nullptr, &P.layout));
    VkComputePipelineCreateInfo pi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pi.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
    pi.layout = P.layout;
    MOE_TP_CHECK(vkCreateComputePipelines(D.dev, VK_NULL_HANDLE, 1, &pi, nullptr, &P.pipe));
    vkDestroyShaderModule(D.dev, mod, nullptr);
    return true;
}

// ggml's block_mxfp4, bit for bit.
struct moe_tp_block17 {
    uint8_t e;
    uint8_t qs[16];
};

// Row-range slice into a fused plane (gate|up). Same packing as the probe's.
static inline void moe_tp_repack_rows(const uint8_t * blocks, int64_t k,
                                      int64_t row0, int64_t nrows,
                                      int64_t m_total, int64_t dst_row0,
                                      uint32_t * plane, uint8_t * scales) {
    const int64_t kb = k / 32;
    for (int64_t r = 0; r < nrows; r++) {
        const int64_t mm = dst_row0 + r;
        const moe_tp_block17 * row = (const moe_tp_block17 *) (blocks + (row0 + r) * kb * 17);
        for (int64_t b = 0; b < kb; b++) {
            scales[b * m_total + mm] = row[b].e;
            for (int j = 0; j < 16; j++) {
                const uint32_t lo = row[b].qs[j] & 0x0F;
                const uint32_t hi = row[b].qs[j] >> 4;
                const int64_t k_lo = b * 32 + j;
                const int64_t k_hi = b * 32 + j + 16;
                plane[(((k_lo >> 1) * (m_total / 8) + mm / 8) << 1) | (k_lo & 1)] |= lo << (4 * (mm % 8));
                plane[(((k_hi >> 1) * (m_total / 8) + mm / 8) << 1) | (k_hi & 1)] |= hi << (4 * (mm % 8));
            }
        }
    }
}

// Block-aligned k-range slice of every row (down's split).
static inline void moe_tp_repack_krange(const uint8_t * blocks, int64_t k_src, int64_t m,
                                        int64_t kb0, int64_t nkb,
                                        uint32_t * plane, uint8_t * scales) {
    const int64_t kb_src = k_src / 32;
    for (int64_t mm = 0; mm < m; mm++) {
        const moe_tp_block17 * row = (const moe_tp_block17 *) (blocks + mm * kb_src * 17);
        for (int64_t b = 0; b < nkb; b++) {
            const moe_tp_block17 & blk = row[kb0 + b];
            scales[b * m + mm] = blk.e;
            for (int j = 0; j < 16; j++) {
                const uint32_t lo = blk.qs[j] & 0x0F;
                const uint32_t hi = blk.qs[j] >> 4;
                const int64_t k_lo = b * 32 + j;
                const int64_t k_hi = b * 32 + j + 16;
                plane[(((k_lo >> 1) * (m / 8) + mm / 8) << 1) | (k_lo & 1)] |= lo << (4 * (mm % 8));
                plane[(((k_hi >> 1) * (m / 8) + mm / 8) << 1) | (k_hi & 1)] |= hi << (4 * (mm % 8));
            }
        }
    }
}

// ---------------------------------------------------------------------------

static inline bool moe_tp_enabled(moe_tp & T) {
    if (!T.checked) {
        T.checked = true;
        T.on = getenv("MOESERV_TP") != nullptr;
    }
    return T.on;
}

// Parse the received split. Returns false for anything that is not exactly the
// one-token deepseek4 expert block — the caller falls back, which is slow and
// right.
// First few parse rejections are named: a fallback that cannot say why is how
// this project has shipped four checks that tested nothing.
static inline bool moe_tp_why(const char * why) {
    static int shown = 0;
    if (shown < 3) { shown++; fprintf(stderr, "MoE-TP: parse fallback: %s" , why); fputc(10, stderr); }
    return false;
}

static inline bool moe_tp_parse(ggml_cgraph * gf, moe_tp_call & c, const char * (*layer_name)(const ggml_tensor *)) {
    (void) layer_name;
    const int n = ggml_graph_n_nodes(gf);
    ggml_tensor * glu = nullptr, * mul = nullptr;
    for (int i = 0; i < n; i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_GLU) glu = t;
        if (t->op == GGML_OP_MUL) mul = t;
    }
    if (!glu || !mul) return moe_tp_why("no GLU or MUL");
    int32_t glu_op = 0, swapped = 0;
    memcpy(&glu_op,   (const char *) glu->op_params + 0, 4);
    memcpy(&swapped,  (const char *) glu->op_params + 4, 4);
    if (glu_op != GGML_GLU_OP_SWIGLU || swapped != 0) return moe_tp_why("glu variant");

    ggml_tensor * gc = glu->src[0], * uc = glu->src[1];
    if (!gc || !uc || gc->op != GGML_OP_CLAMP || uc->op != GGML_OP_CLAMP) return moe_tp_why("glu srcs not clamps");
    ggml_tensor * gmm = gc->src[0], * umm = uc->src[0];
    if (!gmm || !umm || gmm->op != GGML_OP_MUL_MAT_ID || umm->op != GGML_OP_MUL_MAT_ID) return moe_tp_why("clamp srcs not mmid");

    // down: the mul_mat_id feeding MUL (possibly through nothing — the capture
    // shows MUL's src[0] is the down matmul directly).
    ggml_tensor * dmm = mul->src[0];
    if (!dmm || dmm->op != GGML_OP_MUL_MAT_ID) return moe_tp_why("mul src0 not mmid");
    const ggml_tensor * dsrc = dmm->src[1];
    if (dsrc != glu) return moe_tp_why("down input not glu");

    c.gate_w = gmm->src[0];
    c.up_w   = umm->src[0];
    c.down_w = dmm->src[0];
    c.x      = gmm->src[1];
    c.ids    = gmm->src[2];
    c.wts    = mul->src[1];
    c.out    = mul;
    if (umm->src[1] != c.x || umm->src[2] != c.ids || dmm->src[2] != c.ids) return moe_tp_why("shared x/ids mismatch");
    if (!c.x->data || !c.ids->data || !c.wts->data || !c.out->data) return moe_tp_why("missing host data");
    if (c.x->type != GGML_TYPE_F32 || c.ids->type != GGML_TYPE_I32 || c.wts->type != GGML_TYPE_F32) return moe_tp_why("io types");
    if (c.gate_w->type != GGML_TYPE_MXFP4 || c.up_w->type != GGML_TYPE_MXFP4 ||
        c.down_w->type != GGML_TYPE_MXFP4) return moe_tp_why("weight types");
    if (c.x->ne[1] != 1 || c.x->ne[2] != 1) return moe_tp_why("more than one token");
    if (!ggml_is_contiguous(c.out)) return moe_tp_why("out not contiguous");

    memcpy(&c.gmin, (const char *) gc->op_params + 0, 4);
    memcpy(&c.gmax, (const char *) gc->op_params + 4, 4);
    memcpy(&c.umin, (const char *) uc->op_params + 0, 4);
    memcpy(&c.umax, (const char *) uc->op_params + 4, 4);
    c.n_slots = c.ids->ne[0];

    // Layer index from the gate tensor's name (blk.N....).
    const char * name = ggml_get_name(c.gate_w);
    if (!name || strncmp(name, "blk.", 4) != 0) return moe_tp_why("weight name");
    c.layer = atoi(name + 4);
    return true;
}

static inline void moe_tp_free(moe_tp & T) {
    if (!T.alive) return;
    for (auto & [l, L] : T.layers) {
        for (size_t d = 0; d < L.per_die.size(); d++) {
            moe_tp_layer_die & LD = L.per_die[d];
            moe_tp_die & D = T.dies[d];
            moe_tp_free_buf(D, LD.gu_plane);
            moe_tp_free_buf(D, LD.gu_scale);
            moe_tp_free_buf(D, LD.dn_plane);
            moe_tp_free_buf(D, LD.dn_scale);
        }
    }
    T.layers.clear();
    for (moe_tp_die & D : T.dies) {
        if (!D.dev) continue;
        vkDeviceWaitIdle(D.dev);
        if (D.io_ptr) { vkUnmapMemory(D.dev, D.io.mem); D.io_ptr = nullptr; }
        for (moe_tp_buf * b : { &D.io, &D.x_dev, &D.ids_dev, &D.wts_dev, &D.h,
                                &D.part_gu, &D.part_dn, &D.y, &D.staging }) moe_tp_free_buf(D, *b);
        for (moe_tp_pipeline * P : { &D.p1, &D.pmid, &D.pfin }) {
            if (P->pipe)   vkDestroyPipeline(D.dev, P->pipe, nullptr);
            if (P->layout) vkDestroyPipelineLayout(D.dev, P->layout, nullptr);
            if (P->dsl)    vkDestroyDescriptorSetLayout(D.dev, P->dsl, nullptr);
        }
        if (D.fence) vkDestroyFence(D.dev, D.fence, nullptr);
        if (D.dpool) vkDestroyDescriptorPool(D.dev, D.dpool, nullptr);
        if (D.pool)  vkDestroyCommandPool(D.dev, D.pool, nullptr);
        vkDestroyDevice(D.dev, nullptr);
    }
    T.dies.clear();
    if (T.instance) vkDestroyInstance(T.instance, nullptr);
    T.instance = VK_NULL_HANDLE;
    T.alive = false;
    T.dims_set = false;
}

static inline bool moe_tp_init(moe_tp & T, const char * tag) {
    if (T.alive) return true;

    // Shaders sit next to the DLL.
    if (const char * e = getenv("MOESERV_SHADERS")) {
        T.shader_dir = e;
    } else {
#ifdef _WIN32
        HMODULE mod = nullptr;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR) &moe_tp_init, &mod);
        char path[1024] = { 0 };
        GetModuleFileNameA(mod, path, sizeof(path));
        std::string p = path;
        const size_t cut = p.find_last_of("/\\");
        T.shader_dir = (cut == std::string::npos ? std::string(".") : p.substr(0, cut)) + "\\shaders";
#else
        T.shader_dir = "shaders";
#endif
    }

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "moeserv-tp";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    MOE_TP_CHECK(vkCreateInstance(&ici, nullptr, &T.instance));

    uint32_t n_dev = 0;
    MOE_TP_CHECK(vkEnumeratePhysicalDevices(T.instance, &n_dev, nullptr));
    std::vector<VkPhysicalDevice> phys(n_dev);
    MOE_TP_CHECK(vkEnumeratePhysicalDevices(T.instance, &n_dev, phys.data()));
    for (VkPhysicalDevice p : phys) {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        moe_tp_die D;
        D.phys = p;
        uint32_t n_q = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(p, &n_q, nullptr);
        std::vector<VkQueueFamilyProperties> qs(n_q);
        vkGetPhysicalDeviceQueueFamilyProperties(p, &n_q, qs.data());
        D.qfam = UINT32_MAX;
        for (uint32_t i = 0; i < n_q; i++) {
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { D.qfam = i; break; }
        }
        if (D.qfam == UINT32_MAX) continue;
        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qi.queueFamilyIndex = D.qfam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        MOE_TP_CHECK(vkCreateDevice(p, &di, nullptr, &D.dev));
        vkGetDeviceQueue(D.dev, D.qfam, 0, &D.queue);
        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.queueFamilyIndex = D.qfam;
        MOE_TP_CHECK(vkCreateCommandPool(D.dev, &pci, nullptr, &D.pool));
        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        MOE_TP_CHECK(vkCreateFence(D.dev, &fci, nullptr, &D.fence));

        const std::string sd = T.shader_dir + "/";
        if (!moe_tp_load_pipeline(D, sd + "mxfp4_pass1.spv", 5, 7 * 4, D.p1) ||
            !moe_tp_load_pipeline(D, sd + "tp_mid.spv",     2, 6 * 4, D.pmid) ||
            !moe_tp_load_pipeline(D, sd + "tp_final.spv",   3, 2 * 4, D.pfin)) {
            fprintf(stderr, "%s-TP: shader load failed from %s\n", tag, T.shader_dir.c_str());
            return false;
        }

        VkDescriptorPoolSize psz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 };
        VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpi.maxSets = 1024;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &psz;
        MOE_TP_CHECK(vkCreateDescriptorPool(D.dev, &dpi, nullptr, &D.dpool));

        size_t budget_mb = 28000;
        if (const char * b = getenv("MOESERV_TP_BUDGET_MB")) budget_mb = (size_t) atoll(b);
        D.budget = budget_mb << 20;

        // 64 MB staging for the chunked weight uploads.
        if (!moe_tp_make_buf(D, 64ull << 20,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                D.staging)) return false;

        T.dies.push_back(D);
        if (T.dies.size() == 4) break;
    }
    if (T.dies.empty()) {
        fprintf(stderr, "%s-TP: no discrete GPUs\n", tag);
        return false;
    }
    fprintf(stderr, "%s-TP: %zu die(s), shaders from %s\n", tag, T.dies.size(), T.shader_dir.c_str());
    T.alive = true;
    return true;
}

// Fix the shape-dependent scratch once, from the first parsed call.
static inline bool moe_tp_set_dims(moe_tp & T, const moe_tp_call & c) {
    if (T.dims_set) {
        return T.K == c.x->ne[0] && T.SLOTS == c.n_slots &&
               T.INTER == c.gate_w->ne[1] && T.MOUT == c.down_w->ne[1];
    }
    T.K = c.x->ne[0];
    T.INTER = c.gate_w->ne[1];
    T.MOUT = c.down_w->ne[1];
    T.SLOTS = c.n_slots;
    const int64_t n_die = (int64_t) T.dies.size();
    if (T.INTER % (n_die * 32) != 0 || T.K % T.tile_k != 0) return false;
    T.ISL = T.INTER / n_die;
    if (T.ISL % T.tile_k != 0) return false;

    const int64_t GU_M = 2 * T.ISL;
    const uint32_t gu_tiles = (uint32_t) (T.K / T.tile_k);
    const uint32_t dn_tiles = (uint32_t) (T.ISL / T.tile_k);

    T.off_x = 0;
    T.off_ids = (size_t) T.K * 4;
    T.off_wts = T.off_ids + (size_t) T.SLOTS * 4;
    T.off_y   = T.off_wts + (size_t) T.SLOTS * 4;
    T.io_size = T.off_y + (size_t) T.SLOTS * T.MOUT * 4;

    const VkBufferUsageFlags dev_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    for (moe_tp_die & D : T.dies) {
        // HOST_CACHED matters more than it looks: the plain VISIBLE|COHERENT
        // type on AMD is write-combined — fine to write, ~150 MB/s to read —
        // and this buffer is where the CPU reads 98 KB of partials per die per
        // call. Uncached reads cost ~2 ms/layer; the cached type removes it.
        // (lyrae hit the Metal spelling of this: storageModeShared, 8 GB/s.)
        if (!moe_tp_make_buf(D, T.io_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_HOST_CACHED_BIT, D.io)) {
            if (!moe_tp_make_buf(D, T.io_size,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, D.io)) return false;
        }
        MOE_TP_CHECK(vkMapMemory(D.dev, D.io.mem, 0, VK_WHOLE_SIZE, 0, &D.io_ptr));
        if (!moe_tp_make_buf(D, (size_t) T.K * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, D.x_dev)) return false;
        if (!moe_tp_make_buf(D, (size_t) T.SLOTS * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, D.ids_dev)) return false;
        if (!moe_tp_make_buf(D, (size_t) T.SLOTS * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, D.wts_dev)) return false;
        if (!moe_tp_make_buf(D, (size_t) T.SLOTS * T.ISL * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, D.h)) return false;
        if (!moe_tp_make_buf(D, (size_t) T.SLOTS * gu_tiles * GU_M * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, D.part_gu)) return false;
        if (!moe_tp_make_buf(D, (size_t) T.SLOTS * dn_tiles * T.MOUT * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, D.part_dn)) return false;
        if (!moe_tp_make_buf(D, (size_t) T.SLOTS * T.MOUT * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, D.y)) return false;
    }
    T.dims_set = true;
    return true;
}

// Make one layer resident: repack this layer's three tensors into per-die
// slices, upload, build descriptor sets, record the per-die command buffer.
// Runs once per layer, at its first decode call — seconds of host repack per
// layer, paid lazily.
static inline bool moe_tp_setup_layer(moe_tp & T, const moe_tp_call & c, moe_tp_layer & L, const char * tag) {
    const int64_t n_die = (int64_t) T.dies.size();
    const int64_t E = c.gate_w->ne[2];
    const int64_t GU_M = 2 * T.ISL;
    const uint32_t gu_tiles = (uint32_t) (T.K / T.tile_k);
    const uint32_t dn_tiles = (uint32_t) (T.ISL / T.tile_k);

    const size_t gu_plane_sz = (size_t) E * T.K * GU_M / 2;
    const size_t gu_scale_sz = (size_t) E * (T.K / 32) * GU_M;
    const size_t dn_plane_sz = (size_t) E * T.ISL * T.MOUT / 2;
    const size_t dn_scale_sz = (size_t) E * (T.ISL / 32) * T.MOUT;
    const size_t need = gu_plane_sz + gu_scale_sz + dn_plane_sz + dn_scale_sz;

    for (moe_tp_die & D : T.dies) {
        if (D.budget < need) {
            fprintf(stderr, "%s-TP: layer %d does not fit (%zu MB needed); CPU fallback\n",
                    tag, c.layer, need >> 20);
            return false;
        }
    }

    L.gmin = c.gmin; L.gmax = c.gmax; L.umin = c.umin; L.umax = c.umax;
    L.per_die.resize(T.dies.size());

    const size_t gate_expert_bytes = (size_t) T.INTER * (T.K / 32) * 17;
    const size_t down_expert_bytes = (size_t) T.MOUT * (T.INTER / 32) * 17;

    std::vector<uint32_t> gup(gu_plane_sz / 4), dnp(dn_plane_sz / 4);
    std::vector<uint8_t>  gus(gu_scale_sz),  dns(dn_scale_sz);

    for (int64_t d = 0; d < n_die; d++) {
        moe_tp_die & D = T.dies[d];
        moe_tp_layer_die & LD = L.per_die[d];

        memset(gup.data(), 0, gup.size() * 4);
        memset(dnp.data(), 0, dnp.size() * 4);
        for (int64_t e = 0; e < E; e++) {
            const uint8_t * gb = (const uint8_t *) c.gate_w->data + e * gate_expert_bytes;
            const uint8_t * ub = (const uint8_t *) c.up_w->data   + e * gate_expert_bytes;
            const uint8_t * db = (const uint8_t *) c.down_w->data + e * down_expert_bytes;
            uint32_t * pl = gup.data() + e * (T.K * GU_M / 8);
            uint8_t  * sc = gus.data() + e * ((T.K / 32) * GU_M);
            moe_tp_repack_rows(gb, T.K, d * T.ISL, T.ISL, GU_M, 0,     pl, sc);
            moe_tp_repack_rows(ub, T.K, d * T.ISL, T.ISL, GU_M, T.ISL, pl, sc);
            uint32_t * pl2 = dnp.data() + e * (T.ISL * T.MOUT / 8);
            uint8_t  * sc2 = dns.data() + e * ((T.ISL / 32) * T.MOUT);
            moe_tp_repack_krange(db, T.INTER, T.MOUT, d * T.ISL / 32, T.ISL / 32, pl2, sc2);
        }

        const VkBufferUsageFlags dev_usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!moe_tp_make_buf(D, gu_plane_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, LD.gu_plane) ||
            !moe_tp_make_buf(D, gu_scale_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, LD.gu_scale) ||
            !moe_tp_make_buf(D, dn_plane_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, LD.dn_plane) ||
            !moe_tp_make_buf(D, dn_scale_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, LD.dn_scale)) return false;
        if (!moe_tp_upload(D, LD.gu_plane, gup.data(), gu_plane_sz) ||
            !moe_tp_upload(D, LD.gu_scale, gus.data(), gu_scale_sz) ||
            !moe_tp_upload(D, LD.dn_plane, dnp.data(), dn_plane_sz) ||
            !moe_tp_upload(D, LD.dn_scale, dns.data(), dn_scale_sz)) return false;
        D.budget -= need;

        // Descriptor sets.
        VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = D.dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &D.p1.dsl;   MOE_TP_CHECK(vkAllocateDescriptorSets(D.dev, &ai, &LD.ds_gu));
        MOE_TP_CHECK(vkAllocateDescriptorSets(D.dev, &ai, &LD.ds_dn));
        ai.pSetLayouts = &D.pmid.dsl; MOE_TP_CHECK(vkAllocateDescriptorSets(D.dev, &ai, &LD.ds_mid));
        ai.pSetLayouts = &D.pfin.dsl; MOE_TP_CHECK(vkAllocateDescriptorSets(D.dev, &ai, &LD.ds_fin));
        struct bind_t { VkDescriptorSet set; uint32_t bind; VkBuffer buf; };
        const bind_t binds[] = {
            { LD.ds_gu, 0, LD.gu_plane.buf }, { LD.ds_gu, 1, LD.gu_scale.buf },
            { LD.ds_gu, 2, D.x_dev.buf },     { LD.ds_gu, 3, D.part_gu.buf },
            { LD.ds_gu, 4, D.ids_dev.buf },
            { LD.ds_dn, 0, LD.dn_plane.buf }, { LD.ds_dn, 1, LD.dn_scale.buf },
            { LD.ds_dn, 2, D.h.buf },         { LD.ds_dn, 3, D.part_dn.buf },
            { LD.ds_dn, 4, D.ids_dev.buf },
            { LD.ds_mid, 0, D.part_gu.buf },  { LD.ds_mid, 1, D.h.buf },
            { LD.ds_fin, 0, D.part_dn.buf },  { LD.ds_fin, 1, D.wts_dev.buf },
            { LD.ds_fin, 2, D.y.buf },
        };
        const size_t nb = sizeof(binds) / sizeof(binds[0]);
        std::vector<VkDescriptorBufferInfo> infos(nb);
        std::vector<VkWriteDescriptorSet> writes(nb);
        for (size_t i = 0; i < nb; i++) {
            infos[i] = { binds[i].buf, 0, VK_WHOLE_SIZE };
            writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = binds[i].set;
            writes[i].dstBinding = binds[i].bind;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(D.dev, (uint32_t) writes.size(), writes.data(), 0, nullptr);

        // Record the per-call command buffer once: stage-in copies, the four
        // dispatches, the read-back copy. Per call only the io contents change.
        VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = D.pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        MOE_TP_CHECK(vkAllocateCommandBuffers(D.dev, &cai, &LD.cb));
        VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        MOE_TP_CHECK(vkBeginCommandBuffer(LD.cb, &cbi));

        VkBufferCopy cx = { T.off_x, 0, (size_t) T.K * 4 };
        VkBufferCopy ci = { T.off_ids, 0, (size_t) T.SLOTS * 4 };
        VkBufferCopy cw = { T.off_wts, 0, (size_t) T.SLOTS * 4 };
        vkCmdCopyBuffer(LD.cb, D.io.buf, D.x_dev.buf, 1, &cx);
        vkCmdCopyBuffer(LD.cb, D.io.buf, D.ids_dev.buf, 1, &ci);
        vkCmdCopyBuffer(LD.cb, D.io.buf, D.wts_dev.buf, 1, &cw);
        VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        auto barrier = [&](VkPipelineStageFlags src, VkPipelineStageFlags dst) {
            vkCmdPipelineBarrier(LD.cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
        };
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        const uint32_t pc_gu[7] = { (uint32_t) T.K, (uint32_t) GU_M, (uint32_t) T.tile_k, gu_tiles,
                                    (uint32_t) (T.K * GU_M / 16), (uint32_t) ((T.K / 32) * GU_M / 4), 0 };
        const uint32_t pc_dn[7] = { (uint32_t) T.ISL, (uint32_t) T.MOUT, (uint32_t) T.tile_k, dn_tiles,
                                    (uint32_t) (T.ISL * T.MOUT / 16), (uint32_t) ((T.ISL / 32) * T.MOUT / 4),
                                    (uint32_t) T.ISL };
        struct { uint32_t inter, n_tiles; float gmin, gmax, umin, umax; } pc_mid =
            { (uint32_t) T.ISL, gu_tiles, c.gmin, c.gmax, c.umin, c.umax };
        const uint32_t pc_fin[2] = { (uint32_t) T.MOUT, dn_tiles };

        vkCmdBindPipeline(LD.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.p1.pipe);
        vkCmdBindDescriptorSets(LD.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.p1.layout, 0, 1, &LD.ds_gu, 0, nullptr);
        vkCmdPushConstants(LD.cb, D.p1.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_gu), pc_gu);
        vkCmdDispatch(LD.cb, (uint32_t) ((GU_M / 8 * 2 + 255) / 256), gu_tiles, (uint32_t) T.SLOTS);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(LD.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pmid.pipe);
        vkCmdBindDescriptorSets(LD.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pmid.layout, 0, 1, &LD.ds_mid, 0, nullptr);
        vkCmdPushConstants(LD.cb, D.pmid.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_mid), &pc_mid);
        vkCmdDispatch(LD.cb, (uint32_t) ((T.ISL + 255) / 256), 1, (uint32_t) T.SLOTS);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(LD.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.p1.pipe);
        vkCmdBindDescriptorSets(LD.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.p1.layout, 0, 1, &LD.ds_dn, 0, nullptr);
        vkCmdPushConstants(LD.cb, D.p1.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_dn), pc_dn);
        vkCmdDispatch(LD.cb, (uint32_t) ((T.MOUT / 8 * 2 + 255) / 256), dn_tiles, (uint32_t) T.SLOTS);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(LD.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pfin.pipe);
        vkCmdBindDescriptorSets(LD.cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.pfin.layout, 0, 1, &LD.ds_fin, 0, nullptr);
        vkCmdPushConstants(LD.cb, D.pfin.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_fin), pc_fin);
        vkCmdDispatch(LD.cb, (uint32_t) ((T.MOUT + 255) / 256), 1, (uint32_t) T.SLOTS);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy cy = { 0, T.off_y, (size_t) T.SLOTS * T.MOUT * 4 };
        vkCmdCopyBuffer(LD.cb, D.y.buf, D.io.buf, 1, &cy);
        MOE_TP_CHECK(vkEndCommandBuffer(LD.cb));
    }

    L.resident = true;
    fprintf(stderr, "%s-TP: layer %d resident, %zu MB per die\n", tag, c.layer, need >> 20);
    return true;
}

// The per-call path: stage inputs, submit every die, wait, sum partials into
// the MUL node's host output.
static inline bool moe_tp_compute(moe_tp & T, ggml_cgraph * gf, const char * tag) {
    moe_tp_call c;
    if (!moe_tp_parse(gf, c, nullptr)) return false;
    if (!moe_tp_init(T, tag)) { T.on = false; return false; }
    if (!moe_tp_set_dims(T, c)) return false;

    moe_tp_layer & L = T.layers[c.layer];
    if (!L.resident) {
        if (!L.per_die.empty()) return false;      // tried before and failed
        if (!moe_tp_setup_layer(T, c, L, tag)) { L.per_die.resize(1); return false; }
    }
    // Clamp params are baked into the recorded CBs; a mismatch means the model
    // changed them per call, which deepseek4 does not do — fall back if so.
    if (L.gmin != c.gmin || L.gmax != c.gmax || L.umin != c.umin || L.umax != c.umax) return false;

    const auto tt0 = std::chrono::steady_clock::now();
    // Stage inputs into every die's persistently mapped io buffer.
    for (moe_tp_die & D : T.dies) {
        memcpy((char *) D.io_ptr + T.off_x,   c.x->data,   (size_t) T.K * 4);
        memcpy((char *) D.io_ptr + T.off_ids, c.ids->data, (size_t) T.SLOTS * 4);
        memcpy((char *) D.io_ptr + T.off_wts, c.wts->data, (size_t) T.SLOTS * 4);
    }
    const auto tt1 = std::chrono::steady_clock::now();
    // Submit all dies, then wait all — the concurrency the split probe priced
    // at ~60-100 us of fixed floor.
    for (size_t d = 0; d < T.dies.size(); d++) {
        moe_tp_die & D = T.dies[d];
        MOE_TP_CHECK(vkResetFences(D.dev, 1, &D.fence));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &L.per_die[d].cb;
        MOE_TP_CHECK(vkQueueSubmit(D.queue, 1, &si, D.fence));
    }
    const auto tt2 = std::chrono::steady_clock::now();
    float * out = (float *) c.out->data;
    const size_t n_out = (size_t) T.SLOTS * T.MOUT;
    std::chrono::steady_clock::time_point tt3;
    for (size_t d = 0; d < T.dies.size(); d++) {
        moe_tp_die & D = T.dies[d];
        // Poll instead of vkWaitForFences: the blocking wait pays a kernel
        // wake-up per fence, and vkGetFenceStatus on this driver is a
        // user-mode read. The CPU is idle here anyway — the trunk is blocked
        // on this block's output.
        VkResult f_;
        while ((f_ = vkGetFenceStatus(D.dev, D.fence)) == VK_NOT_READY) {
#if defined(_WIN32)
            YieldProcessor();
#endif
        }
        MOE_TP_CHECK(f_);
        if (d == 0) tt3 = std::chrono::steady_clock::now();
        const float * yp = (const float *) ((const char *) D.io_ptr + T.off_y);
        if (d == 0) {
            memcpy(out, yp, n_out * 4);
        } else {
            for (size_t i = 0; i < n_out; i++) out[i] += yp[i];
        }
    }
    const auto tt4 = std::chrono::steady_clock::now();
    using us_t = std::chrono::duration<double, std::micro>;
    T.t_stage  += us_t(tt1 - tt0).count();
    T.t_submit += us_t(tt2 - tt1).count();
    T.t_wait   += us_t(tt3 - tt2).count();
    T.t_sum    += us_t(tt4 - tt3).count();
    if (T.n_calls++ == 0) {
        fprintf(stderr, "%s-TP: computed a split on %zu die(s) (layer %d, %lld slots)\n",
                tag, T.dies.size(), c.layer, (long long) T.SLOTS);
    }
    return true;
}

static inline void moe_tp_report(const moe_tp & T, const char * tag) {
    if (T.n_calls || T.n_fallback) {
        fprintf(stderr, "%s-TP: %llu splits on the dies, %llu fell back\n",
                tag, (unsigned long long) T.n_calls, (unsigned long long) T.n_fallback);
    }
    if (T.n_calls) {
        const double n = (double) T.n_calls;
        fprintf(stderr, "%s-TP: per call us: stage %.1f  submit %.1f  wait-first %.1f  wait-rest+sum %.1f%c",
                tag, T.t_stage / n, T.t_submit / n, T.t_wait / n, T.t_sum / n, 10);
    }
}
