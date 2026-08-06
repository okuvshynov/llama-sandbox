// Minimal Vulkan compute scaffolding, adapted from ../../vk-test/src/common/
// vk_common.h (same author, same machine). Trimmed to what this harness needs
// and extended with fence-based submission, because measuring *when* the CPU
// regains control is the whole point here — vkQueueWaitIdle cannot separate
// "submit returned" from "GPU finished".
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                        \
    do {                                                                   \
        VkResult err_ = (x);                                               \
        if (err_ != VK_SUCCESS) {                                          \
            std::fprintf(stderr, "Vulkan error %d at %s:%d\n  %s\n",       \
                         (int)err_, __FILE__, __LINE__, #x);               \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

namespace vkc {

inline VkDeviceSize round4(VkDeviceSize n) { return (n + 3) & ~(VkDeviceSize)3; }

inline std::vector<char> readFile(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf((size_t) size);
    if (std::fread(buf.data(), 1, (size_t) size, f) != (size_t) size) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        std::exit(1);
    }
    std::fclose(f);
    return buf;
}

struct Buffer {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
};

struct DeviceInfo {
    std::string name;
    std::string driver;
    uint32_t    apiVersion         = 0;
    uint64_t    maxAllocationSize  = 0;   // the 2 GiB cap on the AMD/Windows driver
    uint32_t    subgroupSize       = 0;   // 64 on AMD, 32 elsewhere — kernels assume this
};

struct Context {
    VkInstance       inst  = VK_NULL_HANDLE;
    VkPhysicalDevice pd    = VK_NULL_HANDLE;
    VkDevice         dev   = VK_NULL_HANDLE;
    VkQueue          queue = VK_NULL_HANDLE;
    uint32_t         qf    = 0;
    VkCommandPool    pool  = VK_NULL_HANDLE;
    VkCommandBuffer  cmd   = VK_NULL_HANDLE;
    VkFence          fence = VK_NULL_HANDLE;
    DeviceInfo       info;

    Buffer staging;
    void * stagingMapped = nullptr;

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want) const {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        }
        std::fprintf(stderr, "no suitable memory type\n");
        std::exit(1);
    }

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags props) const {
        Buffer b;
        b.size = round4(size);

        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size        = b.size;
        bi.usage       = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, b.buf, &req);

        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
        VkResult r = vkAllocateMemory(dev, &ai, nullptr, &b.mem);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr,
                "vkAllocateMemory failed for %.2f MiB (error %d).\n"
                "  Device reports maxMemoryAllocationSize = %.2f MiB — a single buffer\n"
                "  cannot exceed it no matter how much VRAM is free. Reduce --experts.\n",
                (double) b.size / (1 << 20), (int) r,
                (double) info.maxAllocationSize / (1 << 20));
            std::exit(1);
        }
        VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
        return b;
    }

    Buffer createStorageBuffer(VkDeviceSize size) const {
        return createBuffer(size,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    void destroyBuffer(Buffer & b) const {
        if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
        if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
        b = Buffer{};
    }

    static std::vector<DeviceInfo> listDevices() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "moe-offload";
        app.apiVersion       = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
#if defined(__APPLE__)
        // MoltenVK is a non-conformant (portability) implementation; without
        // this the loader hides it and enumeration comes back empty.
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        const char * exts[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
        ici.enabledExtensionCount   = 1;
        ici.ppEnabledExtensionNames = exts;
#endif
        VkInstance inst;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

        uint32_t count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(inst, &count, nullptr));
        std::vector<VkPhysicalDevice> devs(count);
        if (count) VK_CHECK(vkEnumeratePhysicalDevices(inst, &count, devs.data()));

        std::vector<DeviceInfo> out;
        for (auto pd : devs) out.push_back(queryInfo(pd));
        vkDestroyInstance(inst, nullptr);
        return out;
    }

    static DeviceInfo queryInfo(VkPhysicalDevice pd) {
        DeviceInfo d;

        VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
        m3.pNext = &sub;
        VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        drv.pNext = &m3;
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &drv;
        vkGetPhysicalDeviceProperties2(pd, &p2);

        d.name              = p2.properties.deviceName;
        d.driver            = drv.driverName[0] ? drv.driverName : "(unknown)";
        d.apiVersion        = p2.properties.apiVersion;
        d.maxAllocationSize = m3.maxMemoryAllocationSize;
        d.subgroupSize      = sub.subgroupSize;
        return d;
    }

    static Context create(uint32_t deviceIndex) {
        Context c;

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "moe-offload";
        app.apiVersion       = VK_API_VERSION_1_2;

        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
#if defined(__APPLE__)
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        const char * iexts[] = { VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME };
        ici.enabledExtensionCount   = 1;
        ici.ppEnabledExtensionNames = iexts;
#endif
        VK_CHECK(vkCreateInstance(&ici, nullptr, &c.inst));

        uint32_t count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(c.inst, &count, nullptr));
        if (count == 0) { std::fprintf(stderr, "no Vulkan physical devices\n"); std::exit(1); }
        std::vector<VkPhysicalDevice> devices(count);
        VK_CHECK(vkEnumeratePhysicalDevices(c.inst, &count, devices.data()));
        if (deviceIndex >= count) {
            std::fprintf(stderr, "device %u out of range (%u present)\n", deviceIndex, count);
            std::exit(1);
        }
        c.pd   = devices[deviceIndex];
        c.info = queryInfo(c.pd);

        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &qfCount, qfs.data());

        bool found = false;
        for (uint32_t i = 0; i < qfCount && !found; ++i) {
            if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                c.qf = i; found = true;
            }
        }
        for (uint32_t i = 0; i < qfCount && !found; ++i) {
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qf = i; found = true; }
        }
        if (!found) { std::fprintf(stderr, "no compute queue family\n"); std::exit(1); }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = c.qf;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos    = &qci;
#if defined(__APPLE__)
        // MoltenVK advertises VK_KHR_portability_subset and the spec requires
        // enabling it on any device that exposes it.
        const char * dexts[] = { "VK_KHR_portability_subset" };
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> props(extCount);
        vkEnumerateDeviceExtensionProperties(c.pd, nullptr, &extCount, props.data());
        for (const auto & p : props) {
            if (std::strcmp(p.extensionName, dexts[0]) == 0) {
                dci.enabledExtensionCount   = 1;
                dci.ppEnabledExtensionNames = dexts;
                break;
            }
        }
#endif
        VK_CHECK(vkCreateDevice(c.pd, &dci, nullptr, &c.dev));
        vkGetDeviceQueue(c.dev, c.qf, 0, &c.queue);

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.queueFamilyIndex = c.qf;
        cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(c.dev, &cpi, nullptr, &c.pool));

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool        = c.pool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(c.dev, &cbai, &c.cmd));

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &c.fence));

        return c;
    }

    VkCommandBuffer beginCmd() const {
        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        return cmd;
    }

    void endCmd() const { VK_CHECK(vkEndCommandBuffer(cmd)); }

    // Submit without blocking. Returns as soon as the driver accepts the work,
    // so the caller can do CPU work (the shared expert) before waitFence().
    void submit() const {
        VK_CHECK(vkResetFences(dev, 1, (VkFence *) &fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
    }

    void waitFence() const {
        VK_CHECK(vkWaitForFences(dev, 1, (const VkFence *) &fence, VK_TRUE, UINT64_MAX));
    }

    void endSubmitWait() const { endCmd(); submit(); waitFence(); }

    void computeBarrier(VkCommandBuffer c) const {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void ensureStaging(VkDeviceSize need) {
        need = round4(need);
        if (staging.buf && staging.size >= need) return;
        if (staging.buf) { vkUnmapMemory(dev, staging.mem); destroyBuffer(staging); stagingMapped = nullptr; }
        staging = createBuffer(need,
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkMapMemory(dev, staging.mem, 0, staging.size, 0, &stagingMapped));
    }

    // Submits and waits before returning, so successive uploads cannot alias
    // the one staging buffer (the bug vk-test's docs/staging-aliasing-bug.md
    // documents — do not "optimize" this into a batched recording).
    void upload(const Buffer & dst, const void * data, VkDeviceSize bytes, VkDeviceSize dstOffset = 0) {
        VkDeviceSize copySize = round4(bytes);
        ensureStaging(copySize);
        std::memcpy(stagingMapped, data, (size_t) bytes);
        if (copySize > bytes) std::memset((char *) stagingMapped + bytes, 0, (size_t)(copySize - bytes));
        VkCommandBuffer c = beginCmd();
        VkBufferCopy copy{0, dstOffset, copySize};
        vkCmdCopyBuffer(c, staging.buf, dst.buf, 1, &copy);
        endSubmitWait();
    }

    void download(const Buffer & src, void * data, VkDeviceSize bytes, VkDeviceSize srcOffset = 0) {
        ensureStaging(bytes);
        VkCommandBuffer c = beginCmd();
        VkBufferCopy copy{srcOffset, 0, round4(bytes)};
        vkCmdCopyBuffer(c, src.buf, staging.buf, 1, &copy);
        endSubmitWait();
        std::memcpy(data, stagingMapped, (size_t) bytes);
    }

    void destroy() {
        if (staging.buf) { vkUnmapMemory(dev, staging.mem); destroyBuffer(staging); stagingMapped = nullptr; }
        if (fence) vkDestroyFence(dev, fence, nullptr);
        if (pool)  vkDestroyCommandPool(dev, pool, nullptr);
        if (dev)   vkDestroyDevice(dev, nullptr);
        if (inst)  vkDestroyInstance(inst, nullptr);
        *this = Context{};
    }
};

// One compute shader plus one descriptor set: N storage buffers at bindings
// 0..N-1 and an optional push-constant block.
struct ComputeJob {
    const Context *       ctx    = nullptr;
    VkShaderModule        shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl    = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipe   = VK_NULL_HANDLE;
    VkDescriptorPool      dpool  = VK_NULL_HANDLE;
    VkDescriptorSet       dset   = VK_NULL_HANDLE;
    uint32_t              pcSize = 0;

    static ComputeJob create(const Context & c, const std::string & spvPath,
                             uint32_t numBindings, uint32_t pushConstBytes) {
        ComputeJob j;
        j.ctx    = &c;
        j.pcSize = pushConstBytes;

        std::vector<char> spirv = readFile(spvPath);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spirv.size();
        smci.pCode    = (const uint32_t *) spirv.data();
        VK_CHECK(vkCreateShaderModule(c.dev, &smci, nullptr, &j.shader));

        std::vector<VkDescriptorSetLayoutBinding> bindings(numBindings);
        for (uint32_t i = 0; i < numBindings; ++i) {
            bindings[i] = {};
            bindings[i].binding         = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = numBindings;
        dslci.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(c.dev, &dslci, nullptr, &j.dsl));

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstBytes};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &j.dsl;
        plci.pushConstantRangeCount = pushConstBytes ? 1 : 0;
        plci.pPushConstantRanges    = pushConstBytes ? &pcr : nullptr;
        VK_CHECK(vkCreatePipelineLayout(c.dev, &plci, nullptr, &j.layout));

        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = j.shader;
        cpci.stage.pName  = "main";
        cpci.layout       = j.layout;
        VK_CHECK(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &j.pipe));

        VkDescriptorPoolSize psize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, numBindings};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets       = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &psize;
        VK_CHECK(vkCreateDescriptorPool(c.dev, &dpci, nullptr, &j.dpool));

        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool     = j.dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &j.dsl;
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &dsai, &j.dset));
        return j;
    }

    void bind(uint32_t binding, const Buffer & b) const {
        VkDescriptorBufferInfo info{b.buf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = dset;
        w.dstBinding      = binding;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &info;
        vkUpdateDescriptorSets(ctx->dev, 1, &w, 0, nullptr);
    }

    // Exact workgroup count: matmul_i8 uses barrier(), so an out-of-range early
    // return would be non-uniform control flow.
    void recordGroups(VkCommandBuffer c, uint32_t groups, const void * pc) const {
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
        if (pcSize) vkCmdPushConstants(c, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize, pc);
        vkCmdDispatch(c, groups, 1, 1);
    }

    void destroy() {
        if (!ctx) return;
        if (pipe)   vkDestroyPipeline(ctx->dev, pipe, nullptr);
        if (shader) vkDestroyShaderModule(ctx->dev, shader, nullptr);
        if (layout) vkDestroyPipelineLayout(ctx->dev, layout, nullptr);
        if (dsl)    vkDestroyDescriptorSetLayout(ctx->dev, dsl, nullptr);
        if (dpool)  vkDestroyDescriptorPool(ctx->dev, dpool, nullptr);
        *this = ComputeJob{};
    }
};

}  // namespace vkc
