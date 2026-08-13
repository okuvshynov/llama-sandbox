// moe-probe --custom — the 2-pass MXFP4 vecmat prototype, raw Vulkan.
//
// Standalone by design: its own instance, its own device, its own pipelines,
// no ggml on the GPU side at all. ggml appears exactly once, as the CPU
// reference the output is compared against — the kernel is new code, so it
// must be checked against the implementation it intends to replace, on the
// same quantized bytes.
//
// What it measures: GPU time (timestamp queries around all reps) for
// pass1+pass2 over n_used experts at our decode shapes. Race it against the
// per-format spans in PLAN.md (`decode-kernel`): ggml's mxfp4 span is 163.8 µs
// at k=4096 m=2048 6 experts; the byte bound is ~27 µs at 1 TB/s.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define VK_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "vulkan error %d at %s:%d\n", (int) r_, __FILE__, __LINE__); exit(2); } } while (0)

namespace {

struct vk_ctx {
    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice         dev = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    uint32_t         qfam = 0;
    float            ts_period = 0.0f;
    VkCommandPool    pool = VK_NULL_HANDLE;
    std::string      name;
};

struct vk_buf {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t         size = 0;
};

static uint32_t find_mem_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    fprintf(stderr, "no suitable memory type\n");
    exit(2);
}

static vk_buf make_buf(const vk_ctx & C, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    vk_buf b;
    b.size = size;
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(C.dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(C.dev, b.buf, &req);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_mem_type(C.phys, req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(C.dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(C.dev, b.buf, b.mem, 0));
    return b;
}

static void free_buf(const vk_ctx & C, vk_buf & b) {
    if (b.buf) vkDestroyBuffer(C.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(C.dev, b.mem, nullptr);
    b = vk_buf{};
}

// Staged upload: map the staging buffer, memcpy, one-shot copy command.
static void upload(const vk_ctx & C, vk_buf & staging, vk_buf & dst, const void * data, size_t size) {
    void * p = nullptr;
    VK_CHECK(vkMapMemory(C.dev, staging.mem, 0, size, 0, &p));
    memcpy(p, data, size);
    vkUnmapMemory(C.dev, staging.mem);

    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = C.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(C.dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy region = { 0, 0, size };
    vkCmdCopyBuffer(cb, staging.buf, dst.buf, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(C.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(C.queue));
    vkFreeCommandBuffers(C.dev, C.pool, 1, &cb);
}

static VkShaderModule load_spv(const vk_ctx & C, const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code((n + 3) / 4);
    if (fread(code.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
    fclose(f);
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = (size_t) n;
    ci.pCode = code.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(C.dev, &ci, nullptr, &m));
    return m;
}

struct pipeline {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipe = VK_NULL_HANDLE;
};

static pipeline make_pipeline(const vk_ctx & C, VkShaderModule mod, uint32_t n_bindings, uint32_t push_bytes) {
    pipeline P;
    std::vector<VkDescriptorSetLayoutBinding> binds(n_bindings);
    for (uint32_t i = 0; i < n_bindings; i++) {
        binds[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    }
    VkDescriptorSetLayoutCreateInfo di = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    di.bindingCount = n_bindings;
    di.pBindings = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(C.dev, &di, nullptr, &P.dsl));

    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes };
    VkPipelineLayoutCreateInfo li = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    li.setLayoutCount = 1;
    li.pSetLayouts = &P.dsl;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(C.dev, &li, nullptr, &P.layout));

    VkComputePipelineCreateInfo pi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pi.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                 VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
    pi.layout = P.layout;
    VK_CHECK(vkCreateComputePipelines(C.dev, VK_NULL_HANDLE, 1, &pi, nullptr, &P.pipe));
    return P;
}

// ggml's block_mxfp4, bit for bit (ggml-common.h): one e8m0 scale byte, then
// 16 bytes of nibbles where qs[j] holds elements j (low) and j+16 (high).
struct block_mxfp4_raw {
    uint8_t e;
    uint8_t qs[16];
};

// The repack: ggml's 17-byte interleaved blocks -> two planes.
//   nibble plane u32[K][M/8]: entry (k, t) = columns 8t..8t+7 at row k
//   scale plane  u8[K/32][M]: (k-block, column)
static void repack(const uint8_t * blocks, int64_t k, int64_t m,
                   std::vector<uint32_t> & plane, std::vector<uint8_t> & scales) {
    const int64_t kb = k / 32;
    plane.assign((size_t) (k * m / 8), 0);
    scales.assign((size_t) (kb * m), 0);
    for (int64_t mm = 0; mm < m; mm++) {
        const block_mxfp4_raw * row = (const block_mxfp4_raw *) (blocks + mm * kb * 17);
        for (int64_t b = 0; b < kb; b++) {
            scales[(size_t) (b * m + mm)] = row[b].e;
            for (int j = 0; j < 16; j++) {
                const uint32_t lo = row[b].qs[j] & 0x0F;
                const uint32_t hi = row[b].qs[j] >> 4;
                const int64_t k_lo = b * 32 + j;
                const int64_t k_hi = b * 32 + j + 16;
                // k-paired for uvec2 loads (E2): u32 index = ((k/2)*(m/8) +
                // col_group)*2 + (k&1), so rows k and k+1 sit in one 8-byte
                // load for a given column group.
                plane[(size_t) ((((k_lo >> 1) * (m / 8) + mm / 8) << 1) | (k_lo & 1))] |= lo << (4 * (mm % 8));
                plane[(size_t) ((((k_hi >> 1) * (m / 8) + mm / 8) << 1) | (k_hi & 1))] |= hi << (4 * (mm % 8));
            }
        }
    }
}

// Reference: ggml's own dequantisation of the same blocks into f32, then an
// f32 mul_mat_id. Not ggml's *quantized* CPU path on purpose: that path
// quantizes the activations to q8 before the dot product, which puts ~0.4%
// noise into the reference itself — the first run of this comparison "failed"
// with exactly that signature (max abs ~0.3 on 4096-term sums). Dequant into
// f32 is exact, the semantics are still ggml's (to_float is
// dequantize_row_mxfp4), and both sides then differ only in summation order.
static std::vector<float> ggml_reference(const std::vector<std::vector<uint8_t>> & expert_blocks,
                                         const std::vector<float> & xv,
                                         int64_t k, int64_t m, int64_t n_used) {
    const size_t overhead = 8 * ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 16);
    ggml_init_params ip = { overhead, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_used);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * dst = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(cpu));

    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_MXFP4);
    std::vector<float> dq((size_t) (k * m));
    for (int64_t e = 0; e < n_used; e++) {
        tr->to_float(expert_blocks[e].data(), dq.data(), k * m);
        ggml_backend_tensor_set(as, dq.data(), e * dq.size() * sizeof(float),
                                dq.size() * sizeof(float));
    }
    for (int64_t s = 0; s < n_used; s++) {
        ggml_backend_tensor_set(b, xv.data(), s * k * sizeof(float), k * sizeof(float));
    }
    std::vector<int32_t> iv(n_used);
    for (int64_t i = 0; i < n_used; i++) iv[i] = (int32_t) i;
    ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> out((size_t) (m * n_used));
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return out;
}

} // namespace

int run_custom_kernel(int64_t k, int64_t m, int64_t n_used, int reps, int64_t tile_k,
                      const char * argv0) {
    if (k % 32 || m % 8 || tile_k % 32 || tile_k > 128) {
        fprintf(stderr, "need k%%32==0, m%%8==0, tile_k%%32==0 and <=128\n");
        return 2;
    }

    // Shaders sit next to the executable.
    std::string dir = argv0 ? argv0 : "";
    const size_t cut = dir.find_last_of("/\\");
    dir = cut == std::string::npos ? "." : dir.substr(0, cut);

    // --- data: quantize once, use the same bytes for GPU and reference ------
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<uint8_t>> expert_blocks(n_used);
    std::vector<uint32_t> plane_all;
    std::vector<uint8_t> scales_all;
    {
        std::vector<float> src((size_t) (k * m));
        std::vector<uint32_t> plane;
        std::vector<uint8_t> scales;
        for (int64_t e = 0; e < n_used; e++) {
            for (auto & v : src) v = dist(rng);
            expert_blocks[e].resize((size_t) (m * (k / 32) * 17));
            ggml_quantize_chunk(GGML_TYPE_MXFP4, src.data(), expert_blocks[e].data(), 0, m, k, nullptr);
            repack(expert_blocks[e].data(), k, m, plane, scales);
            plane_all.insert(plane_all.end(), plane.begin(), plane.end());
            scales_all.insert(scales_all.end(), scales.begin(), scales.end());
        }
    }
    std::vector<float> xv((size_t) k);
    for (auto & v : xv) v = dist(rng);

    const uint32_t n_tiles = (uint32_t) (k / tile_k);
    const size_t partials_sz = (size_t) n_used * n_tiles * m * sizeof(float);
    const size_t y_sz = (size_t) n_used * m * sizeof(float);

    // --- vulkan ---------------------------------------------------------------
    vk_ctx C;
    {
        VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "moe-probe-custom";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &C.instance));

        uint32_t n_dev = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(C.instance, &n_dev, nullptr));
        std::vector<VkPhysicalDevice> devs(n_dev);
        VK_CHECK(vkEnumeratePhysicalDevices(C.instance, &n_dev, devs.data()));
        for (VkPhysicalDevice d : devs) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                C.phys = d;
                C.name = p.deviceName;
                C.ts_period = p.limits.timestampPeriod;
                break;
            }
        }
        if (!C.phys) { fprintf(stderr, "no discrete GPU\n"); return 2; }

        uint32_t n_q = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(C.phys, &n_q, nullptr);
        std::vector<VkQueueFamilyProperties> qs(n_q);
        vkGetPhysicalDeviceQueueFamilyProperties(C.phys, &n_q, qs.data());
        C.qfam = UINT32_MAX;
        for (uint32_t i = 0; i < n_q; i++) {
            if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qs[i].timestampValidBits > 0) {
                C.qfam = i;
                break;
            }
        }
        if (C.qfam == UINT32_MAX) { fprintf(stderr, "no compute queue with timestamps\n"); return 2; }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qi.queueFamilyIndex = C.qfam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        VK_CHECK(vkCreateDevice(C.phys, &di, nullptr, &C.dev));
        vkGetDeviceQueue(C.dev, C.qfam, 0, &C.queue);

        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.queueFamilyIndex = C.qfam;
        VK_CHECK(vkCreateCommandPool(C.dev, &pci, nullptr, &C.pool));
    }

    const VkBufferUsageFlags dev_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vk_buf b_plane  = make_buf(C, plane_all.size() * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_scale  = make_buf(C, scales_all.size(),    dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_x      = make_buf(C, xv.size() * 4,        dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_part   = make_buf(C, partials_sz,          dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_y      = make_buf(C, y_sz,                 dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_ids    = make_buf(C, n_used * 4,           dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf staging  = make_buf(C, plane_all.size() * 4,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    upload(C, staging, b_plane, plane_all.data(), plane_all.size() * 4);
    upload(C, staging, b_scale, scales_all.data(), scales_all.size());
    upload(C, staging, b_x, xv.data(), xv.size() * 4);
    {
        std::vector<int32_t> idv(n_used);
        for (int64_t i = 0; i < n_used; i++) idv[i] = (int32_t) i;   // identity slots
        upload(C, staging, b_ids, idv.data(), idv.size() * 4);
    }

    pipeline p1, p2;
    {
        VkShaderModule m1 = load_spv(C, dir + "/shaders/mxfp4_pass1.spv");
        VkShaderModule m2 = load_spv(C, dir + "/shaders/mxfp4_pass2.spv");
        p1 = make_pipeline(C, m1, 5, 7 * sizeof(uint32_t));
        p2 = make_pipeline(C, m2, 2, 2 * sizeof(uint32_t));
        vkDestroyShaderModule(C.dev, m1, nullptr);
        vkDestroyShaderModule(C.dev, m2, nullptr);
    }

    VkDescriptorPoolSize psz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = 2;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &psz;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(C.dev, &dpi, nullptr, &dpool));

    VkDescriptorSet ds1, ds2;
    {
        VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = dpool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &p1.dsl;
        VK_CHECK(vkAllocateDescriptorSets(C.dev, &ai, &ds1));
        ai.pSetLayouts = &p2.dsl;
        VK_CHECK(vkAllocateDescriptorSets(C.dev, &ai, &ds2));

        VkDescriptorBufferInfo infos[7] = {
            { b_plane.buf, 0, VK_WHOLE_SIZE }, { b_scale.buf, 0, VK_WHOLE_SIZE },
            { b_x.buf, 0, VK_WHOLE_SIZE },     { b_part.buf, 0, VK_WHOLE_SIZE },
            { b_ids.buf, 0, VK_WHOLE_SIZE },
            { b_part.buf, 0, VK_WHOLE_SIZE },  { b_y.buf, 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet writes[7];
        for (int i = 0; i < 7; i++) {
            writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = i < 5 ? ds1 : ds2;
            writes[i].dstBinding = i < 5 ? i : i - 5;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(C.dev, 7, writes, 0, nullptr);
    }

    VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 2;
    VkQueryPool qpool;
    VK_CHECK(vkCreateQueryPool(C.dev, &qpi, nullptr, &qpool));

    // --- record: warm-up dispatch, then reps back to back --------------------
    const uint32_t pc1[7] = { (uint32_t) k, (uint32_t) m, (uint32_t) tile_k, n_tiles,
                              (uint32_t) (k * m / 16), (uint32_t) ((k / 32) * m / 4),
                              0 };   // x_stride 0: one shared vector
    const uint32_t pc2[2] = { (uint32_t) m, n_tiles };
    // E6: two threads per u32 column group (4 columns each).
    const uint32_t gx1 = (uint32_t) ((m / 8 * 2 + 255) / 256);
    const uint32_t gx2 = (uint32_t) ((m + 255) / 256);

    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = C.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(C.dev, &cai, &cb));

    VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
    vkCmdResetQueryPool(cb, qpool, 0, 2);

    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    auto barrier = [&]() {
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    auto one_rep = [&]() {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p1.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p1.layout, 0, 1, &ds1, 0, nullptr);
        vkCmdPushConstants(cb, p1.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), pc1);
        vkCmdDispatch(cb, gx1, n_tiles, (uint32_t) n_used);
        barrier();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p2.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p2.layout, 0, 1, &ds2, 0, nullptr);
        vkCmdPushConstants(cb, p2.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), pc2);
        vkCmdDispatch(cb, gx2, 1, (uint32_t) n_used);
        barrier();
    };

    one_rep();   // warm-up: first-use costs, excluded from the timed span
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qpool, 0);
    for (int r = 0; r < reps; r++) one_rep();
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qpool, 1);
    VK_CHECK(vkEndCommandBuffer(cb));

    const auto t0 = std::chrono::steady_clock::now();
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(C.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(C.queue));
    const double wall_us = std::chrono::duration<double, std::micro>(
                               std::chrono::steady_clock::now() - t0).count() / reps;

    uint64_t ts[2];
    VK_CHECK(vkGetQueryPoolResults(C.dev, qpool, 0, 2, sizeof(ts), ts, 8,
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    const double gpu_us = (double) (ts[1] - ts[0]) * C.ts_period / 1000.0 / reps;

    // --- read back and check against ggml ------------------------------------
    std::vector<float> y((size_t) n_used * m);
    {
        VkCommandBuffer rb;
        VK_CHECK(vkAllocateCommandBuffers(C.dev, &cai, &rb));
        VK_CHECK(vkBeginCommandBuffer(rb, &cbi));
        VkBufferCopy region = { 0, 0, y_sz };
        vkCmdCopyBuffer(rb, b_y.buf, staging.buf, 1, &region);
        VK_CHECK(vkEndCommandBuffer(rb));
        VkSubmitInfo s2 = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        s2.commandBufferCount = 1;
        s2.pCommandBuffers = &rb;
        VK_CHECK(vkQueueSubmit(C.queue, 1, &s2, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(C.queue));
        void * p = nullptr;
        VK_CHECK(vkMapMemory(C.dev, staging.mem, 0, y_sz, 0, &p));
        memcpy(y.data(), p, y_sz);
        vkUnmapMemory(C.dev, staging.mem);
        vkFreeCommandBuffers(C.dev, C.pool, 1, &rb);
    }

    // Combined tolerance: |diff| <= atol + rtol*|ref|. A pure relative gate
    // fails at near-zero sums where both sides hold rounding dust, and a pure
    // absolute one scales wrongly with |y|. Both sides compute exact-dequant
    // f32 sums differing only in order, so the slack needed is tiny.
    const std::vector<float> ref = ggml_reference(expert_blocks, xv, k, m, n_used);
    double max_abs = 0.0, max_excess = 0.0, at_y = 0.0, at_ref = 0.0;
    int64_t at = -1;
    for (size_t i = 0; i < y.size(); i++) {
        const double d = fabs((double) y[i] - (double) ref[i]);
        if (d > max_abs) { max_abs = d; at = (int64_t) i; at_y = y[i]; at_ref = ref[i]; }
        const double excess = d - (1e-4 + 1e-3 * fabs((double) ref[i]));
        if (excess > max_excess) max_excess = excess;
    }
    const bool ok = max_excess <= 0.0;

    const double bytes = (double) n_used * m * ggml_row_size(GGML_TYPE_MXFP4, k);
    printf("%s: custom 2-pass mxfp4, k=%lld m=%lld experts=%lld tile_k=%lld (%u tiles), %d reps\n",
           C.name.c_str(), (long long) k, (long long) m, (long long) n_used,
           (long long) tile_k, n_tiles, reps);
    printf("  gpu  %10.1f us  %8.1f GB/s   (wall %.1f us)\n",
           gpu_us, bytes / (gpu_us * 1e-6) / 1e9, wall_us);
    printf("  check vs ggml dequant+f32: max abs %.3e  %s\n",
           max_abs, ok ? "ok" : "MISMATCH");
    if (!ok) {
        printf("  worst at %lld: got %+.6e  ref %+.6e\n", (long long) at, at_y, at_ref);
    }

    vkDestroyQueryPool(C.dev, qpool, nullptr);
    vkDestroyDescriptorPool(C.dev, dpool, nullptr);
    for (pipeline * P : { &p1, &p2 }) {
        vkDestroyPipeline(C.dev, P->pipe, nullptr);
        vkDestroyPipelineLayout(C.dev, P->layout, nullptr);
        vkDestroyDescriptorSetLayout(C.dev, P->dsl, nullptr);
    }
    for (vk_buf * b : { &b_plane, &b_scale, &b_x, &b_part, &b_y, &b_ids, &staging }) free_buf(C, *b);
    vkDestroyCommandPool(C.dev, C.pool, nullptr);
    vkDestroyDevice(C.dev, nullptr);
    vkDestroyInstance(C.instance, nullptr);
    return ok ? 0 : 1;
}

// --- TP-within-expert block probe ------------------------------------------
//
// The E9 design, executed end to end for one layer: each of four die-slices
// holds columns [I*d, I*(d+1)) of gate and up (fused into one m=2I matmul) and
// the matching k-rows of down, for every expert. Four dispatches per die:
// pass1(GU) -> reduce+clamp+SwiGLU -> pass1(down) -> reduce x router weight.
// The host sums the four partial outputs per slot, which is exact because the
// split is along down's reduction dimension.
//
// Correctness: against a ggml graph of the whole block (mul_mat_id x3, clamps,
// swiglu_split, mul) on f32-dequantized copies of the same quantized bytes.
// The four slices run sequentially on one physical die — the arithmetic of the
// scheme does not care, and the timing of interest is one die's pipeline.

// Slice nrows rows starting at row0 of a quantized [k, m_src] matrix into a
// plane laid out for m_total columns, placing them at column dst_row0.
static void repack_slice(const uint8_t * blocks, int64_t k, int64_t m_src,
                         int64_t row0, int64_t nrows,
                         int64_t m_total, int64_t dst_row0,
                         std::vector<uint32_t> & plane, std::vector<uint8_t> & scales) {
    const int64_t kb = k / 32;
    (void) m_src;
    for (int64_t r = 0; r < nrows; r++) {
        const int64_t mm = dst_row0 + r;
        const block_mxfp4_raw * row = (const block_mxfp4_raw *) (blocks + (row0 + r) * kb * 17);
        for (int64_t b = 0; b < kb; b++) {
            scales[(size_t) (b * m_total + mm)] = row[b].e;
            for (int j = 0; j < 16; j++) {
                const uint32_t lo = row[b].qs[j] & 0x0F;
                const uint32_t hi = row[b].qs[j] >> 4;
                const int64_t k_lo = b * 32 + j;
                const int64_t k_hi = b * 32 + j + 16;
                plane[(size_t) ((((k_lo >> 1) * (m_total / 8) + mm / 8) << 1) | (k_lo & 1))] |= lo << (4 * (mm % 8));
                plane[(size_t) ((((k_hi >> 1) * (m_total / 8) + mm / 8) << 1) | (k_hi & 1))] |= hi << (4 * (mm % 8));
            }
        }
    }
}

// Slice a k-range (block-aligned) of every row: down's split, where die d owns
// input rows [I*d, I*(d+1)) = blocks [I*d/32, ...).
static void repack_kslice(const uint8_t * blocks, int64_t k_src, int64_t m,
                          int64_t kb0, int64_t nkb,
                          std::vector<uint32_t> & plane, std::vector<uint8_t> & scales) {
    const int64_t kb_src = k_src / 32;
    for (int64_t mm = 0; mm < m; mm++) {
        const block_mxfp4_raw * row = (const block_mxfp4_raw *) (blocks + mm * kb_src * 17);
        for (int64_t b = 0; b < nkb; b++) {
            const block_mxfp4_raw & blk = row[kb0 + b];
            scales[(size_t) (b * m + mm)] = blk.e;
            for (int j = 0; j < 16; j++) {
                const uint32_t lo = blk.qs[j] & 0x0F;
                const uint32_t hi = blk.qs[j] >> 4;
                const int64_t k_lo = b * 32 + j;
                const int64_t k_hi = b * 32 + j + 16;
                plane[(size_t) ((((k_lo >> 1) * (m / 8) + mm / 8) << 1) | (k_lo & 1))] |= lo << (4 * (mm % 8));
                plane[(size_t) ((((k_hi >> 1) * (m / 8) + mm / 8) << 1) | (k_hi & 1))] |= hi << (4 * (mm % 8));
            }
        }
    }
}

// ggml reference of the whole block on f32-dequantized weights.
static std::vector<float> tp_reference(const std::vector<std::vector<uint8_t>> & gate_b,
                                       const std::vector<std::vector<uint8_t>> & up_b,
                                       const std::vector<std::vector<uint8_t>> & down_b,
                                       const std::vector<float> & xv, const std::vector<float> & wv,
                                       int64_t k, int64_t inter, int64_t mout, int64_t slots,
                                       float gmin, float gmax, float umin, float umax) {
    const size_t overhead = 24 * ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 16);
    ggml_init_params ip = { overhead, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, inter, slots);
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, inter, slots);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, inter, mout, slots);
    ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, slots, 1);
    ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, slots, 1);
    ggml_tensor * wt   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, slots, 1);

    ggml_tensor * g  = ggml_mul_mat_id(ctx, gate, b, ids);
    ggml_tensor * gc = ggml_clamp(ctx, g, gmin, gmax);
    ggml_tensor * u  = ggml_mul_mat_id(ctx, up, b, ids);
    ggml_tensor * uc = ggml_clamp(ctx, u, umin, umax);
    ggml_tensor * h  = ggml_swiglu_split(ctx, gc, uc);
    ggml_tensor * d  = ggml_mul_mat_id(ctx, down, h, ids);
    ggml_tensor * y  = ggml_mul(ctx, d, wt);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(cpu));

    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_MXFP4);
    std::vector<float> dq((size_t) (k * inter));
    for (int64_t e = 0; e < slots; e++) {
        tr->to_float(gate_b[e].data(), dq.data(), k * inter);
        ggml_backend_tensor_set(gate, dq.data(), e * dq.size() * 4, dq.size() * 4);
        tr->to_float(up_b[e].data(), dq.data(), k * inter);
        ggml_backend_tensor_set(up, dq.data(), e * dq.size() * 4, dq.size() * 4);
    }
    std::vector<float> dq2((size_t) (inter * mout));
    for (int64_t e = 0; e < slots; e++) {
        tr->to_float(down_b[e].data(), dq2.data(), inter * mout);
        ggml_backend_tensor_set(down, dq2.data(), e * dq2.size() * 4, dq2.size() * 4);
    }
    for (int64_t s2 = 0; s2 < slots; s2++) {
        ggml_backend_tensor_set(b, xv.data(), s2 * k * 4, k * 4);
    }
    std::vector<int32_t> idv(slots);
    for (int64_t i = 0; i < slots; i++) idv[i] = (int32_t) i;
    ggml_backend_tensor_set(ids, idv.data(), 0, idv.size() * 4);
    ggml_backend_tensor_set(wt, wv.data(), 0, wv.size() * 4);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_alloc_graph(galloc, gf);
    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> out((size_t) (mout * slots));
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return out;
}

int run_tp_block(int reps, int64_t tile_k, const char * argv0) {
    const int64_t K = 4096, INTER = 2048, MOUT = 4096, SLOTS = 6, DIES = 4;
    const int64_t ISL = INTER / DIES;                 // 512: die slice of intermediate
    const float gmin = -1e30f, gmax = 10.0f;          // representative deepseek4 clamps;
    const float umin = -10.0f,  umax = 10.0f;         // integration reads them from the graph

    std::string dir = argv0 ? argv0 : "";
    const size_t cut = dir.find_last_of("/\\");
    dir = cut == std::string::npos ? "." : dir.substr(0, cut);

    // Six distinct experts, quantized once, shared by GPU and reference.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<uint8_t>> gate_b(SLOTS), up_b(SLOTS), down_b(SLOTS);
    {
        std::vector<float> src((size_t) (K * INTER));
        std::vector<float> src2((size_t) (INTER * MOUT));
        for (int64_t e = 0; e < SLOTS; e++) {
            gate_b[e].resize((size_t) (INTER * (K / 32) * 17));
            up_b[e].resize(gate_b[e].size());
            down_b[e].resize((size_t) (MOUT * (INTER / 32) * 17));
            for (auto & v : src) v = dist(rng);
            ggml_quantize_chunk(GGML_TYPE_MXFP4, src.data(), gate_b[e].data(), 0, INTER, K, nullptr);
            for (auto & v : src) v = dist(rng);
            ggml_quantize_chunk(GGML_TYPE_MXFP4, src.data(), up_b[e].data(), 0, INTER, K, nullptr);
            for (auto & v : src2) v = dist(rng);
            ggml_quantize_chunk(GGML_TYPE_MXFP4, src2.data(), down_b[e].data(), 0, MOUT, INTER, nullptr);
        }
    }
    std::vector<float> xv((size_t) K), wv((size_t) SLOTS);
    for (auto & v : xv) v = dist(rng);
    for (auto & v : wv) v = 0.1f + 0.15f * dist(rng);

    const std::vector<float> ref = tp_reference(gate_b, up_b, down_b, xv, wv,
                                                K, INTER, MOUT, SLOTS, gmin, gmax, umin, umax);

    // --- vulkan ----------------------------------------------------------------
    vk_ctx C;
    {
        VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "moe-probe-tp";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &C.instance));
        uint32_t n_dev = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(C.instance, &n_dev, nullptr));
        std::vector<VkPhysicalDevice> devs(n_dev);
        VK_CHECK(vkEnumeratePhysicalDevices(C.instance, &n_dev, devs.data()));
        for (VkPhysicalDevice d : devs) {
            VkPhysicalDeviceProperties p2;
            vkGetPhysicalDeviceProperties(d, &p2);
            if (p2.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                C.phys = d; C.name = p2.deviceName; C.ts_period = p2.limits.timestampPeriod;
                break;
            }
        }
        if (!C.phys) { fprintf(stderr, "no discrete GPU\n"); return 2; }
        uint32_t n_q = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(C.phys, &n_q, nullptr);
        std::vector<VkQueueFamilyProperties> qs(n_q);
        vkGetPhysicalDeviceQueueFamilyProperties(C.phys, &n_q, qs.data());
        C.qfam = UINT32_MAX;
        for (uint32_t i = 0; i < n_q; i++) {
            if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qs[i].timestampValidBits > 0) { C.qfam = i; break; }
        }
        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qi.queueFamilyIndex = C.qfam; qi.queueCount = 1; qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
        VK_CHECK(vkCreateDevice(C.phys, &di, nullptr, &C.dev));
        vkGetDeviceQueue(C.dev, C.qfam, 0, &C.queue);
        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.queueFamilyIndex = C.qfam;
        VK_CHECK(vkCreateCommandPool(C.dev, &pci, nullptr, &C.pool));
    }

    pipeline p1, pmid, pfin;
    {
        VkShaderModule m1 = load_spv(C, dir + "/shaders/mxfp4_pass1.spv");
        VkShaderModule mm = load_spv(C, dir + "/shaders/tp_mid.spv");
        VkShaderModule mf = load_spv(C, dir + "/shaders/tp_final.spv");
        p1   = make_pipeline(C, m1, 5, 7 * 4);
        pmid = make_pipeline(C, mm, 2, 6 * 4);
        pfin = make_pipeline(C, mf, 3, 2 * 4);
        vkDestroyShaderModule(C.dev, m1, nullptr);
        vkDestroyShaderModule(C.dev, mm, nullptr);
        vkDestroyShaderModule(C.dev, mf, nullptr);
    }

    VkDescriptorPoolSize psz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.maxSets = 16; dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(C.dev, &dpi, nullptr, &dpool));

    const int64_t GU_M = 2 * ISL;                        // 1024
    const size_t gu_plane_sz  = (size_t) SLOTS * K * GU_M / 2;
    const size_t gu_scale_sz  = (size_t) SLOTS * (K / 32) * GU_M;
    const size_t dn_plane_sz  = (size_t) SLOTS * ISL * MOUT / 2;
    const size_t dn_scale_sz  = (size_t) SLOTS * (ISL / 32) * MOUT;
    const uint32_t gu_tiles = (uint32_t) (K / tile_k);
    const uint32_t dn_tiles = (uint32_t) (ISL / tile_k);
    const size_t part_gu_sz = (size_t) SLOTS * gu_tiles * GU_M * 4;
    const size_t part_dn_sz = (size_t) SLOTS * dn_tiles * MOUT * 4;

    const VkBufferUsageFlags dev_usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vk_buf staging = make_buf(C, gu_plane_sz,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vk_buf b_x   = make_buf(C, (size_t) K * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_ids = make_buf(C, (size_t) SLOTS * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_wt  = make_buf(C, (size_t) SLOTS * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_h   = make_buf(C, (size_t) SLOTS * ISL * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_pgu = make_buf(C, part_gu_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_pdn = make_buf(C, part_dn_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_y   = make_buf(C, (size_t) SLOTS * MOUT * 4, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_gup = make_buf(C, gu_plane_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_gus = make_buf(C, gu_scale_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_dnp = make_buf(C, dn_plane_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_buf b_dns = make_buf(C, dn_scale_sz, dev_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    upload(C, staging, b_x, xv.data(), xv.size() * 4);
    upload(C, staging, b_wt, wv.data(), wv.size() * 4);
    {
        std::vector<int32_t> idv(SLOTS);
        for (int64_t i = 0; i < SLOTS; i++) idv[i] = (int32_t) i;
        upload(C, staging, b_ids, idv.data(), idv.size() * 4);
    }

    // Descriptor sets: pass1 twice (GU, down), mid, final.
    VkDescriptorSet ds_gu, ds_dn, ds_mid, ds_fin;
    {
        VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = dpool; ai.descriptorSetCount = 1;
        ai.pSetLayouts = &p1.dsl;   VK_CHECK(vkAllocateDescriptorSets(C.dev, &ai, &ds_gu));
        VK_CHECK(vkAllocateDescriptorSets(C.dev, &ai, &ds_dn));
        ai.pSetLayouts = &pmid.dsl; VK_CHECK(vkAllocateDescriptorSets(C.dev, &ai, &ds_mid));
        ai.pSetLayouts = &pfin.dsl; VK_CHECK(vkAllocateDescriptorSets(C.dev, &ai, &ds_fin));
        struct bind_t { VkDescriptorSet set; uint32_t bind; VkBuffer buf; };
        const bind_t binds[] = {
            { ds_gu, 0, b_gup.buf }, { ds_gu, 1, b_gus.buf }, { ds_gu, 2, b_x.buf },
            { ds_gu, 3, b_pgu.buf }, { ds_gu, 4, b_ids.buf },
            { ds_dn, 0, b_dnp.buf }, { ds_dn, 1, b_dns.buf }, { ds_dn, 2, b_h.buf },
            { ds_dn, 3, b_pdn.buf }, { ds_dn, 4, b_ids.buf },
            { ds_mid, 0, b_pgu.buf }, { ds_mid, 1, b_h.buf },
            { ds_fin, 0, b_pdn.buf }, { ds_fin, 1, b_wt.buf }, { ds_fin, 2, b_y.buf },
        };
        const size_t nb = sizeof(binds) / sizeof(binds[0]);
        std::vector<VkDescriptorBufferInfo> infos(nb);
        std::vector<VkWriteDescriptorSet> writes(nb);
        for (size_t i = 0; i < nb; i++) {
            infos[i] = { binds[i].buf, 0, VK_WHOLE_SIZE };
            writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = binds[i].set; writes[i].dstBinding = binds[i].bind;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(C.dev, (uint32_t) writes.size(), writes.data(), 0, nullptr);
    }

    const uint32_t pc_gu[7] = { (uint32_t) K, (uint32_t) GU_M, (uint32_t) tile_k, gu_tiles,
                                (uint32_t) (K * GU_M / 16), (uint32_t) ((K / 32) * GU_M / 4), 0 };
    const uint32_t pc_dn[7] = { (uint32_t) ISL, (uint32_t) MOUT, (uint32_t) tile_k, dn_tiles,
                                (uint32_t) (ISL * MOUT / 16), (uint32_t) ((ISL / 32) * MOUT / 4),
                                (uint32_t) ISL };
    struct { uint32_t inter, n_tiles; float gmin, gmax, umin, umax; } pc_mid =
        { (uint32_t) ISL, gu_tiles, gmin, gmax, umin, umax };
    const uint32_t pc_fin[2] = { (uint32_t) MOUT, dn_tiles };

    VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
    VkQueryPool qpool;
    VK_CHECK(vkCreateQueryPool(C.dev, &qpi, nullptr, &qpool));

    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = C.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

    std::vector<float> y_sum((size_t) SLOTS * MOUT, 0.0f);
    double die0_us = 0.0;

    for (int64_t die = 0; die < DIES; die++) {
        // This die's planes.
        std::vector<uint32_t> gup(gu_plane_sz / 4, 0), dnp(dn_plane_sz / 4, 0);
        std::vector<uint8_t>  gus(gu_scale_sz, 0),  dns(dn_scale_sz, 0);
        for (int64_t e = 0; e < SLOTS; e++) {
            std::vector<uint32_t> pl((size_t) (K * GU_M / 8), 0);
            std::vector<uint8_t>  sc((size_t) ((K / 32) * GU_M), 0);
            repack_slice(gate_b[e].data(), K, INTER, die * ISL, ISL, GU_M, 0,   pl, sc);
            repack_slice(up_b[e].data(),   K, INTER, die * ISL, ISL, GU_M, ISL, pl, sc);
            memcpy(gup.data() + e * (K * GU_M / 8), pl.data(), pl.size() * 4);
            memcpy(gus.data() + e * ((K / 32) * GU_M), sc.data(), sc.size());

            std::vector<uint32_t> pl2((size_t) (ISL * MOUT / 8), 0);
            std::vector<uint8_t>  sc2((size_t) ((ISL / 32) * MOUT), 0);
            repack_kslice(down_b[e].data(), INTER, MOUT, die * ISL / 32, ISL / 32, pl2, sc2);
            memcpy(dnp.data() + e * (ISL * MOUT / 8), pl2.data(), pl2.size() * 4);
            memcpy(dns.data() + e * ((ISL / 32) * MOUT), sc2.data(), sc2.size());
        }
        upload(C, staging, b_gup, gup.data(), gup.size() * 4);
        upload(C, staging, b_gus, gus.data(), gus.size());
        upload(C, staging, b_dnp, dnp.data(), dnp.size() * 4);
        upload(C, staging, b_dns, dns.data(), dns.size());

        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(C.dev, &cai, &cb));
        VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
        vkCmdResetQueryPool(cb, qpool, 0, 2);
        auto barrier = [&]() {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        };
        auto one_layer = [&]() {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p1.pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p1.layout, 0, 1, &ds_gu, 0, nullptr);
            vkCmdPushConstants(cb, p1.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_gu), pc_gu);
            vkCmdDispatch(cb, (uint32_t) ((GU_M / 8 * 2 + 255) / 256), gu_tiles, (uint32_t) SLOTS);
            barrier();
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pmid.pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pmid.layout, 0, 1, &ds_mid, 0, nullptr);
            vkCmdPushConstants(cb, pmid.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_mid), &pc_mid);
            vkCmdDispatch(cb, (uint32_t) ((ISL + 255) / 256), 1, (uint32_t) SLOTS);
            barrier();
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p1.pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p1.layout, 0, 1, &ds_dn, 0, nullptr);
            vkCmdPushConstants(cb, p1.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_dn), pc_dn);
            vkCmdDispatch(cb, (uint32_t) ((MOUT / 8 * 2 + 255) / 256), dn_tiles, (uint32_t) SLOTS);
            barrier();
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pfin.pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pfin.layout, 0, 1, &ds_fin, 0, nullptr);
            vkCmdPushConstants(cb, pfin.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_fin), pc_fin);
            vkCmdDispatch(cb, (uint32_t) ((MOUT + 255) / 256), 1, (uint32_t) SLOTS);
            barrier();
        };
        one_layer();   // warm-up / the correctness pass
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qpool, 0);
        for (int r = 0; r < reps; r++) one_layer();
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, qpool, 1);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(C.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(C.queue));
        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(C.dev, qpool, 0, 2, sizeof(ts), ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        if (die == 0) die0_us = (double) (ts[1] - ts[0]) * C.ts_period / 1000.0 / reps;
        vkFreeCommandBuffers(C.dev, C.pool, 1, &cb);

        // Read this die's partial and accumulate on the host — the real
        // integration does exactly this into the terminal views.
        std::vector<float> yp((size_t) SLOTS * MOUT);
        {
            VkCommandBuffer rb;
            VK_CHECK(vkAllocateCommandBuffers(C.dev, &cai, &rb));
            VK_CHECK(vkBeginCommandBuffer(rb, &cbi));
            VkBufferCopy region = { 0, 0, yp.size() * 4 };
            vkCmdCopyBuffer(rb, b_y.buf, staging.buf, 1, &region);
            VK_CHECK(vkEndCommandBuffer(rb));
            VkSubmitInfo s2 = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
            s2.commandBufferCount = 1; s2.pCommandBuffers = &rb;
            VK_CHECK(vkQueueSubmit(C.queue, 1, &s2, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(C.queue));
            void * p2 = nullptr;
            VK_CHECK(vkMapMemory(C.dev, staging.mem, 0, yp.size() * 4, 0, &p2));
            memcpy(yp.data(), p2, yp.size() * 4);
            vkUnmapMemory(C.dev, staging.mem);
            vkFreeCommandBuffers(C.dev, C.pool, 1, &rb);
        }
        for (size_t i = 0; i < yp.size(); i++) y_sum[i] += yp[i];
    }

    double max_abs = 0.0, max_excess = 0.0, at_y = 0.0, at_ref = 0.0;
    int64_t at = -1;
    for (size_t i = 0; i < y_sum.size(); i++) {
        const double d = fabs((double) y_sum[i] - (double) ref[i]);
        if (d > max_abs) { max_abs = d; at = (int64_t) i; at_y = y_sum[i]; at_ref = ref[i]; }
        const double excess = d - (1e-4 + 1e-3 * fabs((double) ref[i]));
        if (excess > max_excess) max_excess = excess;
    }
    const bool ok = max_excess <= 0.0;

    printf("%s: TP block, K=%lld inter=%lld (x%lld dies, slice %lld) mout=%lld slots=%lld tile=%lld, %d reps\n",
           C.name.c_str(), (long long) K, (long long) INTER, (long long) DIES, (long long) ISL,
           (long long) MOUT, (long long) SLOTS, (long long) tile_k, reps);
    printf("  per-die pipeline (4 dispatches): %10.1f us\n", die0_us);
    printf("  check vs ggml block (dequant+f32): max abs %.3e  %s\n",
           max_abs, ok ? "ok" : "MISMATCH");
    if (!ok) printf("  worst at %lld: got %+.6e  ref %+.6e\n", (long long) at, at_y, at_ref);

    vkDestroyQueryPool(C.dev, qpool, nullptr);
    vkDestroyDescriptorPool(C.dev, dpool, nullptr);
    for (pipeline * P : { &p1, &pmid, &pfin }) {
        vkDestroyPipeline(C.dev, P->pipe, nullptr);
        vkDestroyPipelineLayout(C.dev, P->layout, nullptr);
        vkDestroyDescriptorSetLayout(C.dev, P->dsl, nullptr);
    }
    for (vk_buf * b : { &staging, &b_x, &b_ids, &b_wt, &b_h, &b_pgu, &b_pdn, &b_y,
                        &b_gup, &b_gus, &b_dnp, &b_dns }) free_buf(C, *b);
    vkDestroyCommandPool(C.dev, C.pool, nullptr);
    vkDestroyDevice(C.dev, nullptr);
    vkDestroyInstance(C.instance, nullptr);
    return ok ? 0 : 1;
}
