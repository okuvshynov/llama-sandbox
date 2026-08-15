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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
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
        VkDeviceCreateInfo di = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
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
