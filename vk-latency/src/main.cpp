// vk-latency: what does it cost to ask a GPU to do nothing?
//
// Submits a null compute shader (one workgroup, empty main) through
// pre-recorded command buffers and times two host-visible quantities:
//   submit        — how long vkQueueSubmit takes to return
//   submit->fence — from submit to the fence reading signaled
// per die, per wait mode (poll vs blocking), and across 1/2/4 dies with the
// same submit-all-then-wait-all shape moe-serv uses. Deliberately no ggml,
// no memory traffic, no descriptors: the floor this measures is the border
// itself, so moe-serv's numbers have a machine baseline to sit on.
//
// Setup mirrors moe-serv/src/moe_tp.h where the choice could matter: discrete
// GPUs only, first compute-capable queue family, one queue, fences created
// unsignaled, poll = vkGetFenceStatus + YieldProcessor.

#include <vulkan/vulkan.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s:%d: %s -> VkResult %d\n", __FILE__, __LINE__, #x, (int)r_); \
    exit(1); } } while (0)

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

static const int N_CB   = 3;                 // command buffer variants
static const int CB_DISPATCHES[N_CB] = { 0, 1, 4 };
static const int WARM  = 50;
static const int ITERS = 500;

struct Die {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    char             name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {};
    uint32_t         qfam = 0;
    VkDevice         dev = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    VkCommandPool    pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline       pipe = VK_NULL_HANDLE;
    VkCommandBuffer  cb[N_CB] = {};
    VkFence          fence = VK_NULL_HANDLE;
    bool             has_calib = false;   // VK_EXT_calibrated_timestamps enabled
    float            ts_period_ns = 0.0f; // limits.timestampPeriod
};

static std::string exe_dir() {
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string s(buf);
    return s.substr(0, s.find_last_of("\\/"));
}

static std::vector<char> read_file(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> data(n);
    if (fread(data.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read on %s\n", path.c_str()); exit(1); }
    fclose(f);
    return data;
}

static void wait_fence(Die &d, bool poll) {
    if (poll) {
        VkResult r;
        while ((r = vkGetFenceStatus(d.dev, d.fence)) == VK_NOT_READY) {
            YieldProcessor();
        }
        CHECK(r);
    } else {
        CHECK(vkWaitForFences(d.dev, 1, &d.fence, VK_TRUE, UINT64_MAX));
    }
}

struct Stat { double med, mn, p90; };

static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stat s;
    s.mn  = v.front();
    s.med = v[v.size() / 2];
    s.p90 = v[(v.size() * 9) / 10];
    return s;
}

// One iteration of the single-die case; returns (submit us, total us).
static void run_single(Die &d, int cb_idx, bool poll, int iters,
                       std::vector<double> &submit_us, std::vector<double> &total_us) {
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d.cb[cb_idx];
    for (int i = 0; i < iters; i++) {
        CHECK(vkResetFences(d.dev, 1, &d.fence));
        double t0 = now_us();
        CHECK(vkQueueSubmit(d.queue, 1, &si, d.fence));
        double t1 = now_us();
        wait_fence(d, poll);
        double t2 = now_us();
        submit_us.push_back(t1 - t0);
        total_us.push_back(t2 - t0);
    }
}

// ===== TP-shaped extension ==================================================
// Everything below grows the null case toward moe-serv's TP call, one
// ingredient per rung, to find which ingredient prices the border
// (moe-serv/src/moe_tp.h `moe_tp_setup_layer` records the real cb this
// mirrors; dimensions are DeepSeek-V4-Flash's at 4 dies). The null scenarios
// above are untouched. The ladder:
//   +desc    one dispatch, 5-binding descriptor set, 28 B push constants
//   +copies  the TP copy pattern: 3 copy-ins, barriers, 96 KiB copy-out
//   +4disp   node-for-node the TP cb: 4 dispatches, 4 sets, barriers between
//   +bigref  plane/scale bindings point at real-sized buffers (512/32/256/16 MiB)
//   ballast  +33 x 816 MiB resident per die (the other resident layers), remeasure

struct Buf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; };

static const size_t TP_K = 4096, TP_SLOTS = 6, TP_MOUT = 4096, TP_ISL = 512;
static const size_t TP_GU_M = 2 * TP_ISL;
static const size_t TP_E = 256, TP_TILE_K = 128;
static const size_t TP_GU_TILES = TP_K / TP_TILE_K;
static const size_t TP_DN_TILES = TP_ISL / TP_TILE_K;

// io layout, exactly moe_tp_set_dims': x | ids | wts | y
static const size_t TP_OFF_X   = 0;
static const size_t TP_OFF_IDS = TP_K * 4;
static const size_t TP_OFF_WTS = TP_OFF_IDS + TP_SLOTS * 4;
static const size_t TP_OFF_Y   = TP_OFF_WTS + TP_SLOTS * 4;
static const size_t TP_IO_SIZE = TP_OFF_Y + TP_SLOTS * TP_MOUT * 4;

// real per-die weight-slice sizes for one layer (816 MiB total)
static const size_t TP_GU_PLANE = TP_E * TP_K * TP_GU_M / 2;       // 512 MiB
static const size_t TP_GU_SCALE = TP_E * (TP_K / 32) * TP_GU_M;    //  32 MiB
static const size_t TP_DN_PLANE = TP_E * TP_ISL * TP_MOUT / 2;     // 256 MiB
static const size_t TP_DN_SCALE = TP_E * (TP_ISL / 32) * TP_MOUT;  //  16 MiB
static const size_t TP_LAYER_BYTES = TP_GU_PLANE + TP_GU_SCALE + TP_DN_PLANE + TP_DN_SCALE;

struct Pipe { VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkPipeline pipe; };

struct TpDie {
    Die *d = nullptr;
    bool io_cached = false;
    Buf io; void *io_ptr = nullptr;
    Buf x, ids, wts, h, part_gu, part_dn, y, small_;
    Buf bgu_plane, bgu_scale, bdn_plane, bdn_scale;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    Pipe p5, p2, p3;
    VkDescriptorSet gu_s, dn_s, gu_b, dn_b, mid, fin;
    VkCommandBuffer cb_desc, cb_copy, cb_4disp, cb_bigref;
    std::vector<Buf> ballast;
};

static uint32_t find_mtype(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return UINT32_MAX;
}

static bool make_buf(Die &d, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, Buf &out) {
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(d.dev, &bci, nullptr, &out.buf) != VK_SUCCESS) { out.buf = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(d.dev, out.buf, &mr);
    uint32_t mt = find_mtype(d.phys, mr.memoryTypeBits, props);
    if (mt == UINT32_MAX) { vkDestroyBuffer(d.dev, out.buf, nullptr); out = Buf{}; return false; }
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(d.dev, &mai, nullptr, &out.mem) != VK_SUCCESS) {
        vkDestroyBuffer(d.dev, out.buf, nullptr);
        out = Buf{};
        return false;
    }
    CHECK(vkBindBufferMemory(d.dev, out.buf, out.mem, 0));
    return true;
}

static void free_buf(Die &d, Buf &b) {
    if (b.buf) vkDestroyBuffer(d.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(d.dev, b.mem, nullptr);
    b = Buf{};
}

// Fill once so the allocation is committed (never-touched DEVICE_LOCAL memory
// may not be resident, and residency is one of the suspects under test).
static void fill_buf(Die &d, Buf &b, size_t size) {
    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(d.dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdFillBuffer(cb, b.buf, 0, size, 0x01010101u);
    CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    CHECK(vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(d.queue));
    vkFreeCommandBuffers(d.dev, d.pool, 1, &cb);
}

static Pipe make_pipe(Die &d, const std::vector<char> &spv, uint32_t n_bind) {
    Pipe p;
    std::vector<VkDescriptorSetLayoutBinding> binds(n_bind);
    for (uint32_t i = 0; i < n_bind; i++) {
        binds[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dli.bindingCount = n_bind;
    dli.pBindings = binds.data();
    CHECK(vkCreateDescriptorSetLayout(d.dev, &dli, nullptr, &p.dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 28 };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    CHECK(vkCreatePipelineLayout(d.dev, &pli, nullptr, &p.layout));
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv.size();
    smci.pCode = (const uint32_t *)spv.data();
    VkShaderModule sm;
    CHECK(vkCreateShaderModule(d.dev, &smci, nullptr, &sm));
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = p.layout;
    CHECK(vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipe));
    vkDestroyShaderModule(d.dev, sm, nullptr);
    return p;
}

static VkDescriptorSet alloc_set(Die &d, VkDescriptorPool pool, VkDescriptorSetLayout dsl,
                                 const std::vector<VkBuffer> &bufs) {
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &dsl;
    VkDescriptorSet set;
    CHECK(vkAllocateDescriptorSets(d.dev, &ai, &set));
    std::vector<VkDescriptorBufferInfo> infos(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); i++) {
        infos[i] = { bufs[i], 0, VK_WHOLE_SIZE };
        writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(d.dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    return set;
}

// Record one variant. Copies, barrier masks and dispatch order are exactly
// moe_tp_setup_layer's; only the grids are cut to (1,1,1) — the border is
// under test, not the arithmetic.
static void record_tp(TpDie &t, VkCommandBuffer cb, bool copies, bool four, bool big,
                      VkQueryPool qp = VK_NULL_HANDLE) {
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CHECK(vkBeginCommandBuffer(cb, &bi));
    if (qp) {
        vkCmdResetQueryPool(cb, qp, 0, 4);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    }
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    auto barrier = [&](VkPipelineStageFlags src, VkPipelineStageFlags dst) {
        vkCmdPipelineBarrier(cb, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    const uint32_t pc[7] = { 1, 2, 3, 4, 5, 6, 7 };
    auto disp = [&](Pipe &p, VkDescriptorSet ds) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cb, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cb, 1, 1, 1);
    };
    if (copies) {
        VkBufferCopy cx = { TP_OFF_X, 0, TP_K * 4 };
        VkBufferCopy ci = { TP_OFF_IDS, 0, TP_SLOTS * 4 };
        VkBufferCopy cw = { TP_OFF_WTS, 0, TP_SLOTS * 4 };
        vkCmdCopyBuffer(cb, t.io.buf, t.x.buf, 1, &cx);
        vkCmdCopyBuffer(cb, t.io.buf, t.ids.buf, 1, &ci);
        vkCmdCopyBuffer(cb, t.io.buf, t.wts.buf, 1, &cw);
        barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        if (qp) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    }
    disp(t.p5, big ? t.gu_b : t.gu_s);
    if (four) {
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        disp(t.p2, t.mid);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        disp(t.p5, big ? t.dn_b : t.dn_s);
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        disp(t.p3, t.fin);
    }
    if (copies) {
        barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (qp) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 2);
        VkBufferCopy cy = { 0, TP_OFF_Y, TP_SLOTS * TP_MOUT * 4 };
        vkCmdCopyBuffer(cb, t.y.buf, t.io.buf, 1, &cy);
    }
    if (qp) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 3);
    CHECK(vkEndCommandBuffer(cb));
}

static bool setup_tp(Die &d, const std::vector<char> &spv5, const std::vector<char> &spv2,
                     const std::vector<char> &spv3, TpDie &t) {
    t.d = &d;
    const VkBufferUsageFlags dev_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkBufferUsageFlags io_usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    t.io_cached = make_buf(d, TP_IO_SIZE, io_usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT, t.io);
    if (!t.io_cached) {
        if (!make_buf(d, TP_IO_SIZE, io_usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, t.io)) return false;
    }
    CHECK(vkMapMemory(d.dev, t.io.mem, 0, VK_WHOLE_SIZE, 0, &t.io_ptr));
    memset(t.io_ptr, 1, TP_IO_SIZE);

    struct { Buf *b; size_t sz; } devbufs[] = {
        { &t.x, TP_K * 4 }, { &t.ids, TP_SLOTS * 4 }, { &t.wts, TP_SLOTS * 4 },
        { &t.h, TP_SLOTS * TP_ISL * 4 },
        { &t.part_gu, TP_SLOTS * TP_GU_TILES * TP_GU_M * 4 },
        { &t.part_dn, TP_SLOTS * TP_DN_TILES * TP_MOUT * 4 },
        { &t.y, TP_SLOTS * TP_MOUT * 4 }, { &t.small_, 64 * 1024 },
        { &t.bgu_plane, TP_GU_PLANE }, { &t.bgu_scale, TP_GU_SCALE },
        { &t.bdn_plane, TP_DN_PLANE }, { &t.bdn_scale, TP_DN_SCALE },
    };
    for (auto &db : devbufs) {
        if (!make_buf(d, db.sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, *db.b)) return false;
        fill_buf(d, *db.b, db.sz);
    }

    VkDescriptorPoolSize psz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = 8;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &psz;
    CHECK(vkCreateDescriptorPool(d.dev, &dpi, nullptr, &t.dpool));

    t.p5 = make_pipe(d, spv5, 5);
    t.p2 = make_pipe(d, spv2, 2);
    t.p3 = make_pipe(d, spv3, 3);

    // binding tables mirror moe_tp_setup_layer's ds_gu/ds_dn/ds_mid/ds_fin
    t.gu_s = alloc_set(d, t.dpool, t.p5.dsl, { t.small_.buf, t.small_.buf, t.x.buf, t.part_gu.buf, t.ids.buf });
    t.dn_s = alloc_set(d, t.dpool, t.p5.dsl, { t.small_.buf, t.small_.buf, t.h.buf, t.part_dn.buf, t.ids.buf });
    t.gu_b = alloc_set(d, t.dpool, t.p5.dsl, { t.bgu_plane.buf, t.bgu_scale.buf, t.x.buf, t.part_gu.buf, t.ids.buf });
    t.dn_b = alloc_set(d, t.dpool, t.p5.dsl, { t.bdn_plane.buf, t.bdn_scale.buf, t.h.buf, t.part_dn.buf, t.ids.buf });
    t.mid  = alloc_set(d, t.dpool, t.p2.dsl, { t.part_gu.buf, t.h.buf });
    t.fin  = alloc_set(d, t.dpool, t.p3.dsl, { t.part_dn.buf, t.wts.buf, t.y.buf });

    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 4;
    VkCommandBuffer cbs[4];
    CHECK(vkAllocateCommandBuffers(d.dev, &cai, cbs));
    t.cb_desc = cbs[0]; t.cb_copy = cbs[1]; t.cb_4disp = cbs[2]; t.cb_bigref = cbs[3];
    record_tp(t, t.cb_desc,   false, false, false);
    record_tp(t, t.cb_copy,   true,  false, false);
    record_tp(t, t.cb_4disp,  true,  true,  false);
    record_tp(t, t.cb_bigref, true,  true,  true);
    return true;
}

static volatile double g_sink;  // keeps the readback sum from being optimized out

// One TP-shaped iteration set: stage (write x+ids+wts into io), submit, poll
// the fence, then read and sum the 96 KiB y region — the four phases of
// moe_tp_compute.
static void run_tp(TpDie &t, VkCommandBuffer cb, int iters, std::vector<double> &stage_us,
                   std::vector<double> &submit_us, std::vector<double> &wait_us, std::vector<double> &sum_us) {
    Die &d = *t.d;
    static std::vector<char> src(TP_OFF_Y, 2);
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    for (int i = 0; i < iters; i++) {
        CHECK(vkResetFences(d.dev, 1, &d.fence));
        double t0 = now_us();
        memcpy(t.io_ptr, src.data(), TP_OFF_Y);
        double t1 = now_us();
        CHECK(vkQueueSubmit(d.queue, 1, &si, d.fence));
        double t2 = now_us();
        wait_fence(d, true);
        double t3 = now_us();
        const float *y = (const float *)((const char *)t.io_ptr + TP_OFF_Y);
        double acc = 0;
        for (size_t k = 0; k < TP_SLOTS * TP_MOUT; k++) acc += y[k];
        g_sink = acc;
        double t4 = now_us();
        stage_us.push_back(t1 - t0);
        submit_us.push_back(t2 - t1);
        wait_us.push_back(t3 - t2);
        sum_us.push_back(t4 - t3);
    }
}

static void print_stat_row(const char *c0, const char *c1, std::vector<double> *cols[], int n_cols) {
    printf("%-4s %-8s", c0, c1);
    for (int i = 0; i < n_cols; i++) {
        Stat s = stat_of(*cols[i]);
        char b[64];
        snprintf(b, sizeof(b), "%7.1f [%5.1f ..%7.1f]", s.med, s.mn, s.p90);
        printf(" %-28s", b);
    }
    printf("\n");
}

int main() {
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.pApplicationName = "vk-latency";
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t n_phys = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n_phys, nullptr));
    std::vector<VkPhysicalDevice> phys(n_phys);
    CHECK(vkEnumeratePhysicalDevices(inst, &n_phys, phys.data()));

    std::vector<char> spv = read_file(exe_dir() + "\\shaders\\null.spv");

    std::vector<Die> dies;
    for (VkPhysicalDevice p : phys) {
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            printf("skipping non-discrete device: %s\n", pp.deviceName);
            continue;
        }
        Die d;
        d.phys = p;
        memcpy(d.name, pp.deviceName, sizeof(d.name));

        uint32_t n_q = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(p, &n_q, nullptr);
        std::vector<VkQueueFamilyProperties> qs(n_q);
        vkGetPhysicalDeviceQueueFamilyProperties(p, &n_q, qs.data());
        d.qfam = UINT32_MAX;
        for (uint32_t i = 0; i < n_q; i++) {
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { d.qfam = i; break; }
        }
        if (d.qfam == UINT32_MAX) continue;

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qi.queueFamilyIndex = d.qfam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        // Enable calibrated timestamps when the driver has them (this
        // machine's 2.0.204 driver does). Enabling an extension changes no
        // timing by itself; the null scenarios above stay comparable.
        const char *want_ext = "VK_EXT_calibrated_timestamps";
        uint32_t n_ext = 0;
        vkEnumerateDeviceExtensionProperties(p, nullptr, &n_ext, nullptr);
        std::vector<VkExtensionProperties> exts(n_ext);
        vkEnumerateDeviceExtensionProperties(p, nullptr, &n_ext, exts.data());
        for (const VkExtensionProperties &e : exts) {
            if (strcmp(e.extensionName, want_ext) == 0) { d.has_calib = true; break; }
        }
        d.ts_period_ns = pp.limits.timestampPeriod;
        VkDeviceCreateInfo di = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = d.has_calib ? 1 : 0;
        di.ppEnabledExtensionNames = &want_ext;
        CHECK(vkCreateDevice(p, &di, nullptr, &d.dev));
        vkGetDeviceQueue(d.dev, d.qfam, 0, &d.queue);

        VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci.codeSize = spv.size();
        smci.pCode = (const uint32_t *)spv.data();
        VkShaderModule sm;
        CHECK(vkCreateShaderModule(d.dev, &smci, nullptr, &sm));

        VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        CHECK(vkCreatePipelineLayout(d.dev, &plci, nullptr, &d.layout));

        VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = "main";
        cpci.layout = d.layout;
        CHECK(vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &d.pipe));
        vkDestroyShaderModule(d.dev, sm, nullptr);

        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.queueFamilyIndex = d.qfam;
        CHECK(vkCreateCommandPool(d.dev, &pci, nullptr, &d.pool));

        VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = d.pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = N_CB;
        CHECK(vkAllocateCommandBuffers(d.dev, &cai, d.cb));
        for (int v = 0; v < N_CB; v++) {
            VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            CHECK(vkBeginCommandBuffer(d.cb[v], &bi));
            if (CB_DISPATCHES[v] > 0) {
                vkCmdBindPipeline(d.cb[v], VK_PIPELINE_BIND_POINT_COMPUTE, d.pipe);
                for (int k = 0; k < CB_DISPATCHES[v]; k++) {
                    vkCmdDispatch(d.cb[v], 1, 1, 1);
                }
            }
            CHECK(vkEndCommandBuffer(d.cb[v]));
        }

        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        CHECK(vkCreateFence(d.dev, &fci, nullptr, &d.fence));

        dies.push_back(d);
    }

    printf("vk-latency: null-dispatch submission baseline, %d iters (+%d warmup) per row\n", ITERS, WARM);
    for (size_t i = 0; i < dies.size(); i++) {
        printf("die %zu: %s (queue family %u)\n", i, dies[i].name, dies[i].qfam);
    }
    if (dies.empty()) { fprintf(stderr, "no discrete GPU with a compute queue\n"); return 1; }
    printf("\n");

    // Warm every die once: first submit after device creation is not
    // representative (lazy driver state, no pipeline is compiled here but the
    // queue itself warms up).
    for (Die &d : dies) {
        std::vector<double> s, t;
        run_single(d, 1, true, WARM, s, t);
    }

    // --- per die, per dispatch count, poll wait ---------------------------
    printf("per die, poll wait (us, median [min .. p90]):\n");
    printf("%-4s %-10s %-24s %-24s\n", "die", "dispatches", "submit", "submit->fence");
    for (size_t i = 0; i < dies.size(); i++) {
        for (int v = 0; v < N_CB; v++) {
            std::vector<double> s, t;
            run_single(dies[i], v, true, ITERS, s, t);
            Stat ss = stat_of(s), ts = stat_of(t);
            char sb[64], tb[64];
            snprintf(sb, sizeof(sb), "%6.1f [%5.1f ..%6.1f]", ss.med, ss.mn, ss.p90);
            snprintf(tb, sizeof(tb), "%6.1f [%5.1f ..%6.1f]", ts.med, ts.mn, ts.p90);
            printf("%-4zu %-10d %-24s %-24s\n", i, CB_DISPATCHES[v], sb, tb);
        }
    }
    printf("\n");

    // --- wait mode: poll vs blocking, 1 dispatch --------------------------
    printf("wait mode, 1 dispatch (submit->fence us, median [min .. p90]):\n");
    printf("%-4s %-24s %-24s\n", "die", "poll", "block");
    for (size_t i = 0; i < dies.size(); i++) {
        std::vector<double> s1, tp, s2, tb;
        run_single(dies[i], 1, true,  ITERS, s1, tp);
        run_single(dies[i], 1, false, ITERS, s2, tb);
        Stat p = stat_of(tp), b = stat_of(tb);
        char pb[64], bb[64];
        snprintf(pb, sizeof(pb), "%6.1f [%5.1f ..%6.1f]", p.med, p.mn, p.p90);
        snprintf(bb, sizeof(bb), "%6.1f [%5.1f ..%6.1f]", b.med, b.mn, b.p90);
        printf("%-4zu %-24s %-24s\n", i, pb, bb);
    }
    printf("\n");

    // --- multi-die rounds: submit all, then wait all (moe-serv's shape) ---
    printf("multi-die round, 1 dispatch per die, poll (us, median [min .. p90]):\n");
    printf("%-4s %-24s %-24s %-24s\n", "dies", "submit-all", "round total", "serial total");
    std::vector<size_t> counts;
    for (size_t n = 1; n <= dies.size(); n *= 2) counts.push_back(n);
    if (counts.back() != dies.size()) counts.push_back(dies.size());
    for (size_t n : counts) {
        std::vector<double> sub, tot, ser;
        for (int it = 0; it < ITERS; it++) {
            // submit-all-then-wait-all
            for (size_t i = 0; i < n; i++) CHECK(vkResetFences(dies[i].dev, 1, &dies[i].fence));
            double t0 = now_us();
            for (size_t i = 0; i < n; i++) {
                VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
                si.commandBufferCount = 1;
                si.pCommandBuffers = &dies[i].cb[1];
                CHECK(vkQueueSubmit(dies[i].queue, 1, &si, dies[i].fence));
            }
            double t1 = now_us();
            for (size_t i = 0; i < n; i++) wait_fence(dies[i], true);
            double t2 = now_us();
            sub.push_back(t1 - t0);
            tot.push_back(t2 - t0);

            // fully serial: submit die, wait die, next die
            double t3 = now_us();
            for (size_t i = 0; i < n; i++) {
                CHECK(vkResetFences(dies[i].dev, 1, &dies[i].fence));
                VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
                si.commandBufferCount = 1;
                si.pCommandBuffers = &dies[i].cb[1];
                CHECK(vkQueueSubmit(dies[i].queue, 1, &si, dies[i].fence));
                wait_fence(dies[i], true);
            }
            double t4 = now_us();
            ser.push_back(t4 - t3);
        }
        Stat s = stat_of(sub), t = stat_of(tot), e = stat_of(ser);
        char sb[64], tb[64], eb[64];
        snprintf(sb, sizeof(sb), "%6.1f [%5.1f ..%6.1f]", s.med, s.mn, s.p90);
        snprintf(tb, sizeof(tb), "%6.1f [%5.1f ..%6.1f]", t.med, t.mn, t.p90);
        snprintf(eb, sizeof(eb), "%6.1f [%5.1f ..%6.1f]", e.med, e.mn, e.p90);
        printf("%-4zu %-24s %-24s %-24s\n", n, sb, tb, eb);
    }

    // ===== TP-shaped ladder ===============================================
    {
        std::vector<char> spv5 = read_file(exe_dir() + "\\shaders\\touch5.spv");
        std::vector<char> spv2 = read_file(exe_dir() + "\\shaders\\touch2.spv");
        std::vector<char> spv3 = read_file(exe_dir() + "\\shaders\\touch3.spv");
        std::vector<TpDie> tp(dies.size());
        for (size_t i = 0; i < dies.size(); i++) {
            if (!setup_tp(dies[i], spv5, spv2, spv3, tp[i])) {
                fprintf(stderr, "TP-shape setup failed on die %zu\n", i);
                return 1;
            }
        }
        if (!tp[0].io_cached) {
            printf("note: io buffer fell back to write-combined (no HOST_CACHED type); sum column reads slow memory\n");
        }
        for (TpDie &t : tp) {
            VkCommandBuffer warm_cbs[4] = { t.cb_desc, t.cb_copy, t.cb_4disp, t.cb_bigref };
            for (VkCommandBuffer cb : warm_cbs) {
                std::vector<double> a, b, c, e;
                run_tp(t, cb, WARM, a, b, c, e);
            }
        }

        printf("TP-shaped ladder, per die, poll wait (us, median [min .. p90]):\n");
        printf("%-4s %-8s %-28s %-28s %-28s %-28s\n",
               "die", "variant", "stage", "submit", "submit->fence", "sum 96KB");
        const char *vnames[4] = { "+desc", "+copies", "+4disp", "+bigref" };
        for (size_t i = 0; i < tp.size(); i++) {
            VkCommandBuffer vcbs[4] = { tp[i].cb_desc, tp[i].cb_copy, tp[i].cb_4disp, tp[i].cb_bigref };
            for (int v = 0; v < 4; v++) {
                std::vector<double> a, b, c, e;
                run_tp(tp[i], vcbs[v], ITERS, a, b, c, e);
                std::vector<double> *cols[] = { &a, &b, &c, &e };
                char dnum[8];
                snprintf(dnum, sizeof(dnum), "%zu", i);
                print_stat_row(dnum, vnames[v], cols, 4);
            }
        }
        printf("\n");

        // The moe_tp_compute shape: stage all dies, submit all, then wait and
        // sum die by die in submission order.
        std::vector<char> round_src(TP_OFF_Y, 2);
        auto tp_round = [&](VkCommandBuffer TpDie::*cbm, std::vector<double> &stage,
                            std::vector<double> &sub, std::vector<double> &waitsum,
                            std::vector<double> &tot) {
            const size_t n = tp.size();
            for (int it = 0; it < ITERS; it++) {
                for (size_t i = 0; i < n; i++) CHECK(vkResetFences(tp[i].d->dev, 1, &tp[i].d->fence));
                double t0 = now_us();
                for (size_t i = 0; i < n; i++) memcpy(tp[i].io_ptr, round_src.data(), TP_OFF_Y);
                double t1 = now_us();
                for (size_t i = 0; i < n; i++) {
                    VkCommandBuffer cb = tp[i].*cbm;
                    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    si.commandBufferCount = 1;
                    si.pCommandBuffers = &cb;
                    CHECK(vkQueueSubmit(tp[i].d->queue, 1, &si, tp[i].d->fence));
                }
                double t2 = now_us();
                double acc = 0;
                for (size_t i = 0; i < n; i++) {
                    wait_fence(*tp[i].d, true);
                    const float *y = (const float *)((const char *)tp[i].io_ptr + TP_OFF_Y);
                    for (size_t k = 0; k < TP_SLOTS * TP_MOUT; k++) acc += y[k];
                }
                g_sink = acc;
                double t3 = now_us();
                stage.push_back(t1 - t0);
                sub.push_back(t2 - t1);
                waitsum.push_back(t3 - t2);
                tot.push_back(t3 - t0);
            }
        };

        printf("multi-die TP round, all %zu dies, poll (us, median [min .. p90]):\n", tp.size());
        printf("%-4s %-8s %-28s %-28s %-28s %-28s\n",
               "dies", "variant", "stage", "submit-all", "wait+sum", "total");
        struct { const char *name; VkCommandBuffer TpDie::*cbm; } rvars[] = {
            { "+4disp", &TpDie::cb_4disp }, { "+bigref", &TpDie::cb_bigref },
        };
        char nbuf[8];
        snprintf(nbuf, sizeof(nbuf), "%zu", tp.size());
        for (auto &rv : rvars) {
            std::vector<double> a, b, c, e;
            tp_round(rv.cbm, a, b, c, e);
            std::vector<double> *cols[] = { &a, &b, &c, &e };
            print_stat_row(nbuf, rv.name, cols, 4);
        }
        printf("(moe-serv full-model comparator, per layer: stage 4.5 / submit ~125 / wait ~310 of which ~113 GPU / total ~440)\n\n");

        // Ballast: the rest of moe-serv's residency — 33 more layer-sized
        // allocations per die, alive but never referenced by any submission.
        printf("ballast: allocating 33 x %zu MiB per die...\n", TP_LAYER_BYTES >> 20);
        for (size_t i = 0; i < tp.size(); i++) {
            while (tp[i].ballast.size() < 33) {
                Buf b;
                if (!make_buf(*tp[i].d, TP_LAYER_BYTES,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b)) break;
                fill_buf(*tp[i].d, b, TP_LAYER_BYTES);
                tp[i].ballast.push_back(b);
            }
            printf("die %zu: %zu chunks (%.1f GiB) resident\n", i, tp[i].ballast.size(),
                   tp[i].ballast.size() * (double)TP_LAYER_BYTES / (1ull << 30));
        }
        printf("\nwith ballast, per die, poll (us, median [min .. p90]):\n");
        printf("%-4s %-8s %-28s %-28s\n", "die", "variant", "submit", "submit->fence");
        for (size_t i = 0; i < tp.size(); i++) {
            std::vector<double> s, t2;
            run_single(dies[i], 1, true, ITERS, s, t2);
            std::vector<double> *cols[] = { &s, &t2 };
            char dnum[8];
            snprintf(dnum, sizeof(dnum), "%zu", i);
            print_stat_row(dnum, "null-1d", cols, 2);
        }
        for (size_t i = 0; i < tp.size(); i++) {
            std::vector<double> a, b, c, e;
            run_tp(tp[i], tp[i].cb_bigref, ITERS, a, b, c, e);
            std::vector<double> *cols[] = { &b, &c };
            char dnum[8];
            snprintf(dnum, sizeof(dnum), "%zu", i);
            print_stat_row(dnum, "+bigref", cols, 2);
        }
        printf("\nmulti-die TP round with ballast (us, median [min .. p90]):\n");
        printf("%-4s %-8s %-28s %-28s %-28s %-28s\n",
               "dies", "variant", "stage", "submit-all", "wait+sum", "total");
        {
            std::vector<double> a, b, c, e;
            tp_round(&TpDie::cb_bigref, a, b, c, e);
            std::vector<double> *cols[] = { &a, &b, &c, &e };
            print_stat_row(nbuf, "+bigref", cols, 4);
        }

        // Host CPU contention: moe-serv's submit/poll thread lives in a
        // process where ggml's threadpool (16 threads on this machine)
        // busy-spins between graph nodes. Mimic that and remeasure.
        // Cumulative with ballast, like the real process state.
        const int N_SPIN = 16;
        std::atomic<bool> spin_stop{false};
        std::vector<std::thread> spinners;
        for (int i = 0; i < N_SPIN; i++) {
            spinners.emplace_back([&spin_stop] {
                volatile uint64_t x = 0;
                while (!spin_stop.load(std::memory_order_relaxed)) x = x + 1;
            });
        }
        printf("\nwith %d spinning host threads (+ ballast still resident), per die, poll:\n", N_SPIN);
        printf("%-4s %-8s %-28s %-28s\n", "die", "variant", "submit", "submit->fence");
        for (size_t i = 0; i < tp.size(); i++) {
            std::vector<double> s, t2;
            run_single(dies[i], 1, true, ITERS, s, t2);
            std::vector<double> *cols[] = { &s, &t2 };
            char dnum[8];
            snprintf(dnum, sizeof(dnum), "%zu", i);
            print_stat_row(dnum, "null-1d", cols, 2);
        }
        for (size_t i = 0; i < tp.size(); i++) {
            std::vector<double> a, b, c, e;
            run_tp(tp[i], tp[i].cb_bigref, ITERS, a, b, c, e);
            std::vector<double> *cols[] = { &b, &c };
            char dnum[8];
            snprintf(dnum, sizeof(dnum), "%zu", i);
            print_stat_row(dnum, "+bigref", cols, 2);
        }
        printf("\nmulti-die TP round with ballast + %d spinning threads:\n", N_SPIN);
        printf("%-4s %-8s %-28s %-28s %-28s %-28s\n",
               "dies", "variant", "stage", "submit-all", "wait+sum", "total");
        {
            std::vector<double> a, b, c, e;
            tp_round(&TpDie::cb_bigref, a, b, c, e);
            std::vector<double> *cols[] = { &a, &b, &c, &e };
            print_stat_row(nbuf, "+bigref", cols, 4);
        }
        spin_stop = true;
        for (std::thread &th : spinners) th.join();

        // ===== calibrated timestamps ======================================
        // VK_EXT_calibrated_timestamps reads (GPU tick, QPC) as one pair, so
        // GPU timestamps convert onto the host clock and submit->fence splits
        // into: launch (submit returned -> GPU cb start), GPU execution
        // (interior timestamps: copy-in / dispatches / copy-out), and signal
        // (GPU done -> host sees the fence). State: ballast resident,
        // spinners stopped.
        bool all_calib = true;
        for (Die &d : dies) all_calib = all_calib && d.has_calib;
        if (!all_calib) {
            printf("\ncalibrated timestamps: VK_EXT_calibrated_timestamps missing; section skipped\n");
        } else {
            LARGE_INTEGER qf;
            QueryPerformanceFrequency(&qf);
            const double qpc_to_us = 1e6 / (double)qf.QuadPart;
            auto qpc_now_us = [qpc_to_us]() {
                LARGE_INTEGER c;
                QueryPerformanceCounter(&c);
                return (double)c.QuadPart * qpc_to_us;
            };

            struct CalDie {
                PFN_vkGetCalibratedTimestampsEXT pfn = nullptr;
                VkQueryPool qp = VK_NULL_HANDLE;
                VkCommandBuffer cb_null = VK_NULL_HANDLE, cb_tp = VK_NULL_HANDLE;
            };
            std::vector<CalDie> cal(tp.size());
            for (size_t i = 0; i < tp.size(); i++) {
                Die &d = dies[i];
                cal[i].pfn = (PFN_vkGetCalibratedTimestampsEXT)
                    vkGetDeviceProcAddr(d.dev, "vkGetCalibratedTimestampsEXT");
                VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
                qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qpi.queryCount = 4;
                CHECK(vkCreateQueryPool(d.dev, &qpi, nullptr, &cal[i].qp));
                VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
                cai.commandPool = d.pool;
                cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cai.commandBufferCount = 2;
                VkCommandBuffer tcbs[2];
                CHECK(vkAllocateCommandBuffers(d.dev, &cai, tcbs));
                cal[i].cb_null = tcbs[0];
                cal[i].cb_tp = tcbs[1];
                // null 1-dispatch with q0..q3 (q1/q2 written back to back so
                // every query is valid; only q3-q0 is meaningful here)
                VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                CHECK(vkBeginCommandBuffer(cal[i].cb_null, &bi));
                vkCmdResetQueryPool(cal[i].cb_null, cal[i].qp, 0, 4);
                vkCmdWriteTimestamp(cal[i].cb_null, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, cal[i].qp, 0);
                vkCmdWriteTimestamp(cal[i].cb_null, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, cal[i].qp, 1);
                vkCmdBindPipeline(cal[i].cb_null, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipe);
                vkCmdDispatch(cal[i].cb_null, 1, 1, 1);
                vkCmdWriteTimestamp(cal[i].cb_null, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, cal[i].qp, 2);
                vkCmdWriteTimestamp(cal[i].cb_null, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, cal[i].qp, 3);
                CHECK(vkEndCommandBuffer(cal[i].cb_null));
                record_tp(tp[i], cal[i].cb_tp, true, true, true, cal[i].qp);
            }

            struct CalRes {
                std::vector<double> submit, launch, cin, disp, cout, gpu, signal, total, dev;
            };
            // scrub: stream this many bytes through the cache before each
            // iteration, evicting the driver's submit-path code and data the
            // way ~8 ms of trunk compute does between moe-serv's calls.
            std::vector<char> scrub_a(64 << 20, 1), scrub_b(64 << 20, 2);
            auto run_cal = [&](size_t i, VkCommandBuffer cb, int iters, CalRes &r, size_t scrub = 0) {
                Die &d = dies[i];
                const double tick_us = d.ts_period_ns / 1000.0;
                VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cb;
                for (int it = 0; it < iters; it++) {
                    if (scrub) {
                        memcpy(it & 1 ? scrub_a.data() : scrub_b.data(),
                               it & 1 ? scrub_b.data() : scrub_a.data(), scrub);
                    }
                    VkCalibratedTimestampInfoEXT ci[2] = {
                        { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_DEVICE_EXT },
                        { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT },
                    };
                    uint64_t base[2], max_dev;
                    CHECK(cal[i].pfn(d.dev, 2, ci, base, &max_dev));
                    auto gpu_us = [&](uint64_t g) {
                        return (double)base[1] * qpc_to_us + (double)(int64_t)(g - base[0]) * tick_us;
                    };
                    CHECK(vkResetFences(d.dev, 1, &d.fence));
                    double t0 = qpc_now_us();
                    CHECK(vkQueueSubmit(d.queue, 1, &si, d.fence));
                    double t1 = qpc_now_us();
                    wait_fence(d, true);
                    double t2 = qpc_now_us();
                    uint64_t q[4];
                    CHECK(vkGetQueryPoolResults(d.dev, cal[i].qp, 0, 4, sizeof(q), q, 8, VK_QUERY_RESULT_64_BIT));
                    double h0 = gpu_us(q[0]), h1 = gpu_us(q[1]), h2 = gpu_us(q[2]), h3 = gpu_us(q[3]);
                    r.submit.push_back(t1 - t0);
                    r.launch.push_back(h0 - t1);
                    r.cin.push_back(h1 - h0);
                    r.disp.push_back(h2 - h1);
                    r.cout.push_back(h3 - h2);
                    r.gpu.push_back(h3 - h0);
                    r.signal.push_back(t2 - h3);
                    r.total.push_back(t2 - t0);
                    r.dev.push_back((double)max_dev / 1000.0);
                }
            };

            for (size_t i = 0; i < tp.size(); i++) {
                CalRes w;
                run_cal(i, cal[i].cb_null, WARM, w);
                run_cal(i, cal[i].cb_tp, WARM, w);
            }

            printf("\ncalibrated timestamps, null 1-dispatch, per die (us, median [min .. p90]):\n");
            printf("%-4s %-8s %-28s %-28s %-28s %-28s %-28s\n",
                   "die", "variant", "submit", "launch", "gpu", "signal", "total");
            std::vector<CalRes> res_tp(tp.size());
            for (size_t i = 0; i < tp.size(); i++) {
                CalRes r;
                run_cal(i, cal[i].cb_null, ITERS, r);
                run_cal(i, cal[i].cb_tp, ITERS, res_tp[i]);
                std::vector<double> *cols[] = { &r.submit, &r.launch, &r.gpu, &r.signal, &r.total };
                char dnum[8];
                snprintf(dnum, sizeof(dnum), "%zu", i);
                print_stat_row(dnum, "null-1d", cols, 5);
            }
            printf("\ncalibrated timestamps, TP +bigref, per die (us, median [min .. p90]):\n");
            printf("%-4s %-8s %-28s %-28s %-28s %-28s %-28s %-28s\n",
                   "die", "variant", "launch", "gpu copy-in", "gpu disp", "gpu copy-out", "signal", "total");
            for (size_t i = 0; i < tp.size(); i++) {
                CalRes &r = res_tp[i];
                std::vector<double> *cols[] = { &r.launch, &r.cin, &r.disp, &r.cout, &r.signal, &r.total };
                char dnum[8];
                snprintf(dnum, sizeof(dnum), "%zu", i);
                print_stat_row(dnum, "+bigref", cols, 6);
            }
            {
                std::vector<double> all_dev;
                for (CalRes &r : res_tp) all_dev.insert(all_dev.end(), r.dev.begin(), r.dev.end());
                Stat sd = stat_of(all_dev);
                printf("calibration max-deviation: median %.2f us, p90 %.2f us (bound on cross-clock error)\n",
                       sd.med, sd.p90);
            }

            // Cold-cache rung: same calibrated measurement, but each
            // iteration first streams 64 MB through the cache — the state
            // moe-serv's submit thread is actually in after ~8 ms of trunk
            // compute. If the in-process ~21-24 us submits are cache
            // eviction, they should reappear here.
            const int COLD_ITERS = 200;
            printf("\ncold cache (64 MB streamed before every iteration), per die (us, median [min .. p90]):\n");
            printf("%-4s %-8s %-28s %-28s %-28s %-28s %-28s\n",
                   "die", "variant", "submit", "launch", "gpu", "signal", "total");
            for (size_t i = 0; i < tp.size(); i++) {
                CalRes r;
                run_cal(i, cal[i].cb_null, COLD_ITERS, r, scrub_a.size());
                std::vector<double> *cols[] = { &r.submit, &r.launch, &r.gpu, &r.signal, &r.total };
                char dnum[8];
                snprintf(dnum, sizeof(dnum), "%zu", i);
                print_stat_row(dnum, "null-1d", cols, 5);
            }
            for (size_t i = 0; i < tp.size(); i++) {
                CalRes r;
                run_cal(i, cal[i].cb_tp, COLD_ITERS, r, scrub_a.size());
                std::vector<double> *cols[] = { &r.submit, &r.launch, &r.gpu, &r.signal, &r.total };
                char dnum[8];
                snprintf(dnum, sizeof(dnum), "%zu", i);
                print_stat_row(dnum, "+bigref", cols, 5);
            }
            for (size_t i = 0; i < tp.size(); i++) {
                vkDestroyQueryPool(dies[i].dev, cal[i].qp, nullptr);
            }
        }

        for (size_t i = 0; i < tp.size(); i++) {
            Die &d = dies[i];
            for (Buf &b : tp[i].ballast) free_buf(d, b);
            vkUnmapMemory(d.dev, tp[i].io.mem);
            free_buf(d, tp[i].io);
            Buf *bufs[] = { &tp[i].x, &tp[i].ids, &tp[i].wts, &tp[i].h, &tp[i].part_gu,
                            &tp[i].part_dn, &tp[i].y, &tp[i].small_, &tp[i].bgu_plane,
                            &tp[i].bgu_scale, &tp[i].bdn_plane, &tp[i].bdn_scale };
            for (Buf *b : bufs) free_buf(d, *b);
            vkDestroyDescriptorPool(d.dev, tp[i].dpool, nullptr);
            Pipe *pipes[] = { &tp[i].p5, &tp[i].p2, &tp[i].p3 };
            for (Pipe *p : pipes) {
                vkDestroyPipeline(d.dev, p->pipe, nullptr);
                vkDestroyPipelineLayout(d.dev, p->layout, nullptr);
                vkDestroyDescriptorSetLayout(d.dev, p->dsl, nullptr);
            }
        }
    }

    for (Die &d : dies) {
        vkDestroyFence(d.dev, d.fence, nullptr);
        vkDestroyCommandPool(d.dev, d.pool, nullptr);
        vkDestroyPipeline(d.dev, d.pipe, nullptr);
        vkDestroyPipelineLayout(d.dev, d.layout, nullptr);
        vkDestroyDevice(d.dev, nullptr);
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}
