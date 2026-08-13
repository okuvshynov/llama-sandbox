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
    vk_buf staging  = make_buf(C, plane_all.size() * 4,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    upload(C, staging, b_plane, plane_all.data(), plane_all.size() * 4);
    upload(C, staging, b_scale, scales_all.data(), scales_all.size());
    upload(C, staging, b_x, xv.data(), xv.size() * 4);

    pipeline p1, p2;
    {
        VkShaderModule m1 = load_spv(C, dir + "/shaders/mxfp4_pass1.spv");
        VkShaderModule m2 = load_spv(C, dir + "/shaders/mxfp4_pass2.spv");
        p1 = make_pipeline(C, m1, 4, 6 * sizeof(uint32_t));
        p2 = make_pipeline(C, m2, 2, 2 * sizeof(uint32_t));
        vkDestroyShaderModule(C.dev, m1, nullptr);
        vkDestroyShaderModule(C.dev, m2, nullptr);
    }

    VkDescriptorPoolSize psz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 };
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

        VkDescriptorBufferInfo infos[6] = {
            { b_plane.buf, 0, VK_WHOLE_SIZE }, { b_scale.buf, 0, VK_WHOLE_SIZE },
            { b_x.buf, 0, VK_WHOLE_SIZE },     { b_part.buf, 0, VK_WHOLE_SIZE },
            { b_part.buf, 0, VK_WHOLE_SIZE },  { b_y.buf, 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet writes[6];
        for (int i = 0; i < 6; i++) {
            writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = i < 4 ? ds1 : ds2;
            writes[i].dstBinding = i < 4 ? i : i - 4;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(C.dev, 6, writes, 0, nullptr);
    }

    VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 2;
    VkQueryPool qpool;
    VK_CHECK(vkCreateQueryPool(C.dev, &qpi, nullptr, &qpool));

    // --- record: warm-up dispatch, then reps back to back --------------------
    const uint32_t pc1[6] = { (uint32_t) k, (uint32_t) m, (uint32_t) tile_k, n_tiles,
                              (uint32_t) (k * m / 16), (uint32_t) ((k / 32) * m / 4) };
    const uint32_t pc2[2] = { (uint32_t) m, n_tiles };
    const uint32_t gx1 = (uint32_t) ((m / 8 + 255) / 256);
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
    for (vk_buf * b : { &b_plane, &b_scale, &b_x, &b_part, &b_y, &staging }) free_buf(C, *b);
    vkDestroyCommandPool(C.dev, C.pool, nullptr);
    vkDestroyDevice(C.dev, nullptr);
    vkDestroyInstance(C.instance, nullptr);
    return ok ? 0 : 1;
}
