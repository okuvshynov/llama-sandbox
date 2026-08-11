#pragma once

// Generic GGUF loading: metadata helpers, read-only file mapping, shard
// enumeration, and a tensor map. Nothing here knows what a model is.
//
// Split out of nano_model.h when a second architecture arrived. The line is
// the one lib/README.md predicted: this file is tier one ("any program"),
// while hparams, tensor names, the graph and the KV layout are tier three
// ("one model") and live under models/<arch>/.
//
// A model's own struct derives from `gguf_store`, so `M.tensors`,
// `M.bytes_mapped` and friends keep working unqualified at every call site.

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"


// File mapping for the weight shards (map_file_ro below). Guarded because
// cpu_topology.h may have pulled windows.h in already; NOMINMAX has to be set
// before the first include of it either way, or the min/max macros eat the
// std::min / std::max calls further down.
#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <fcntl.h>
#   include <sys/mman.h>
#   include <sys/stat.h>
#   include <unistd.h>
#endif

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <functional>
#include <vector>

#define NANO_ABORT(...) do { fprintf(stderr, "nano-glm: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

// ---------------------------------------------------------------------------
// GGUF metadata helpers (hard error on missing key unless *_opt)

static int64_t kv_id(const gguf_context * g, const char * key, bool required) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 && required) NANO_ABORT("missing GGUF key '%s'", key);
    return id;
}

static uint32_t kv_u32(const gguf_context * g, const char * key) {
    int64_t id = kv_id(g, key, true);
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, id);
        case GGUF_TYPE_INT16:  return (uint32_t) gguf_get_val_i16(g, id);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(g, id);
        default: NANO_ABORT("GGUF key '%s' is not an integer", key);
    }
}

static uint32_t kv_u32_opt(const gguf_context * g, const char * key, uint32_t dflt) {
    return gguf_find_key(g, key) < 0 ? dflt : kv_u32(g, key);
}

static float kv_f32(const gguf_context * g, const char * key) {
    return gguf_get_val_f32(g, kv_id(g, key, true));
}

static float kv_f32_opt(const gguf_context * g, const char * key, float dflt) {
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? dflt : gguf_get_val_f32(g, id);
}

static bool kv_bool_opt(const gguf_context * g, const char * key, bool dflt) {
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? dflt : gguf_get_val_bool(g, id);
}

static std::string kv_str_opt(const gguf_context * g, const char * key, const std::string & dflt) {
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? dflt : gguf_get_val_str(g, id);
}


// What every model has once its shards are mapped: the mmap'd regions, the
// contexts owning the tensor structs, and a name -> tensor map. A model's own
// struct derives from this and adds its hparams, its layer array and its
// named tensors.
struct gguf_store {
    std::string desc;

    // Summed as shards are mapped, so the handshake can report which model is
    // loaded without stat'ing the directory a second time.
    uint64_t bytes_mapped = 0;
    uint32_t n_shards     = 0;

    std::vector<ggml_context *>          meta_ctxs;  // own the tensor structs
    std::vector<ggml_backend_buffer_t>   map_bufs;   // wrap the mmap'd data regions
    std::map<std::string, ggml_tensor *> tensors;
};

// Read-only whole-file mapping. Weights are used straight from the mapping
// (ggml_backend_cpu_buffer_from_ptr), so it must outlive the model; nothing
// unmaps — the process exits and the OS reclaims.
static void * map_file_ro(const std::string & path, size_t * size_out) {
#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) NANO_ABORT("cannot open '%s' (err %lu)", path.c_str(), GetLastError());
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) NANO_ABORT("cannot size '%s' (err %lu)", path.c_str(), GetLastError());
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) NANO_ABORT("CreateFileMapping failed for '%s' (err %lu)", path.c_str(), GetLastError());
    void * addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    // the view keeps the file and section alive on its own
    CloseHandle(mh);
    CloseHandle(fh);
    if (!addr) NANO_ABORT("MapViewOfFile failed for '%s' (err %lu)", path.c_str(), GetLastError());
    *size_out = (size_t) sz.QuadPart;
    return addr;
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) NANO_ABORT("cannot open '%s'", path.c_str());
    struct stat st;
    fstat(fd, &st);
    void * addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) NANO_ABORT("mmap failed for '%s'", path.c_str());
    *size_out = (size_t) st.st_size;
    return addr;
#endif
}

static void load_shard(gguf_store & M, const std::string & path, const gguf_context ** meta_out) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ &ctx_meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) NANO_ABORT("failed to read GGUF '%s'", path.c_str());
    M.meta_ctxs.push_back(ctx_meta);

    size_t file_size = 0;
    void * addr = map_file_ro(path, &file_size);
    M.bytes_mapped += file_size;
    M.n_shards     += 1;

    // A metadata-only shard has no data section to wrap, and asking for one is
    // not merely wasteful: with no tensors `gguf_get_data_offset` returns the
    // *unpadded* end of the header, so the pointer is very likely not 32-byte
    // aligned and `ggml_backend_cpu_buffer_from_ptr` asserts on it.
    //
    // GLM-5.2 never exposed this — its metadata shard happens to end on a
    // multiple of 32. DeepSeek-V4-Flash's ends at 5257394, which is 18 past
    // one. Note this must not early-return: the caller still needs `meta_out`,
    // and shard 1 is exactly the shard it reads hparams from.
    if (gguf_get_n_tensors(g) > 0) {
        const size_t data_off = gguf_get_data_offset(g);
        ggml_backend_buffer_t buf =
            ggml_backend_cpu_buffer_from_ptr((char *) addr + data_off, file_size - data_off);
        M.map_bufs.push_back(buf);

        for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
            const char * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
            if (!t) NANO_ABORT("tensor '%s' missing from meta context", name);
            ggml_backend_tensor_alloc(buf, t, (char *) addr + data_off + gguf_get_tensor_offset(g, i));
            M.tensors[name] = t;
        }
    }

    if (meta_out) {
        *meta_out = g; // caller reads hparams from shard 1 and frees it
    } else {
        gguf_free(g);
    }
}

static ggml_tensor * get_tensor(gguf_store & M, const std::string & name, bool required = true) {
    auto it = M.tensors.find(name);
    if (it == M.tensors.end()) {
        if (required) NANO_ABORT("missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

static std::string blk(int il, const char * suffix) {
    return "blk." + std::to_string(il) + "." + suffix;
}

// Map the first shard, hand its metadata to the caller, then map the rest.
// Shard names follow `...-00001-of-000NN.gguf`; `split.count` says how many.
//
// The caller reads hparams from `meta` and must NOT free it — this does, once
// the remaining shards are mapped.
static void load_shards(gguf_store & M, const std::string & first_shard,
                        const std::function<void(const gguf_context *)> & read_meta) {
    const gguf_context * meta = nullptr;
    load_shard(M, first_shard, &meta);
    read_meta(meta);

    const uint32_t n_split = kv_u32_opt(meta, "split.count", 1);
    gguf_free((gguf_context *) meta);

    if (n_split > 1) {
        const std::string pat = "-00001-of-";
        size_t at = first_shard.find(pat);
        if (at == std::string::npos) NANO_ABORT("split.count=%u but path has no '-00001-of-' pattern", n_split);
        for (uint32_t s = 2; s <= n_split; s++) {
            char idx[16];
            snprintf(idx, sizeof(idx), "-%05u-of-", s);
            std::string path = first_shard;
            path.replace(at, pat.size(), idx);
            load_shard(M, path, nullptr);
        }
    }
}
