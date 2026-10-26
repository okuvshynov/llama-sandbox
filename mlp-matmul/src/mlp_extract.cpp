// mlp-extract: pull one layer's MLP weights out of a GGUF model into a small GGUF.
//
//   mlp-extract list    <model.gguf>
//       model metadata + every FFN tensor with its quant type and shape
//   mlp-extract extract <model.gguf> <layer> <out.gguf>
//       copies blk.<layer>.ffn_{gate,up,down}.weight byte for byte, then re-reads
//       the output and checks every byte against the source
//
// Tensors keep their original names, types and layout, so the benchmark sees
// exactly what llama.cpp would load. Extra keys in the output:
//   mlp.source (file name), mlp.layer, mlp.n_embd, mlp.n_ff, mlp.architecture

#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die(const char* fmt, const char* a = "") {
  fprintf(stderr, fmt, a);
  fprintf(stderr, "\n");
  exit(1);
}

static gguf_context* open_meta(const char* path) {
  gguf_init_params p = {/*no_alloc*/ true, /*ctx*/ nullptr};
  gguf_context* g = gguf_init_from_file(path, p);
  if (!g) die("cannot read GGUF: %s", path);
  return g;
}

static std::string get_str(gguf_context* g, const std::string& key) {
  int64_t id = gguf_find_key(g, key.c_str());
  if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return "";
  return gguf_get_val_str(g, id);
}

// Integer-valued key of any width; -1 if missing.
static int64_t get_int(gguf_context* g, const std::string& key) {
  int64_t id = gguf_find_key(g, key.c_str());
  if (id < 0) return -1;
  switch (gguf_get_kv_type(g, id)) {
    case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
    case GGUF_TYPE_INT32: return gguf_get_val_i32(g, id);
    case GGUF_TYPE_UINT64: return (int64_t)gguf_get_val_u64(g, id);
    case GGUF_TYPE_INT64: return gguf_get_val_i64(g, id);
    case GGUF_TYPE_ARRAY: return -2;  // per-layer array, e.g. feed_forward_length
    default: return -1;
  }
}

static std::string shape(const int64_t* ne) {
  char s[64];
  snprintf(s, sizeof s, "%" PRId64 " x %" PRId64, ne[0], ne[1]);
  return s;
}

static int cmd_list(const char* path) {
  gguf_context* g = open_meta(path);
  std::string arch = get_str(g, "general.architecture");
  printf("file          %s\n", path);
  printf("architecture  %s\n", arch.c_str());
  for (const char* k : {"block_count", "embedding_length", "feed_forward_length"})
    printf("%-13s %" PRId64 "\n", k, get_int(g, arch + "." + k));
  printf("\n%-5s %-28s %-8s %-16s %10s\n", "layer", "tensor", "type", "ne0 x ne1", "MiB");
  int64_t n = gguf_get_n_tensors(g);
  for (int64_t i = 0; i < n; i++) {
    const char* name = gguf_get_tensor_name(g, i);
    if (!strstr(name, ".ffn_")) continue;
    int layer = -1;
    char rest[128] = {0};
    if (sscanf(name, "blk.%d.%127s", &layer, rest) != 2) continue;
    printf("%-5d %-28s %-8s %-16s %10.2f\n", layer, rest,
           ggml_type_name(gguf_get_tensor_type(g, i)), shape(gguf_get_tensor_ne(g, i)).c_str(),
           gguf_get_tensor_size(g, i) / 1048576.0);
  }
  gguf_free(g);
  return 0;
}

static std::vector<uint8_t> read_bytes(const char* path, size_t off, size_t n) {
  FILE* f = fopen(path, "rb");
  if (!f) die("cannot open %s", path);
  std::vector<uint8_t> buf(n);
  if (fseeko(f, (off_t)off, SEEK_SET) != 0 || fread(buf.data(), 1, n, f) != n)
    die("short read from %s", path);
  fclose(f);
  return buf;
}

static int cmd_extract(const char* src, int layer, const char* out) {
  gguf_context* g = open_meta(src);
  std::string arch = get_str(g, "general.architecture");
  const char* parts[3] = {"ffn_gate", "ffn_up", "ffn_down"};

  // A ggml context that owns the tensor data for the writer.
  size_t total = 0;
  int64_t ids[3];
  for (int k = 0; k < 3; k++) {
    std::string name = "blk." + std::to_string(layer) + "." + parts[k] + ".weight";
    ids[k] = gguf_find_tensor(g, name.c_str());
    if (ids[k] < 0) die("tensor not found: %s", name.c_str());
    total += gguf_get_tensor_size(g, ids[k]);
  }
  ggml_init_params ip = {total + 3 * ggml_tensor_overhead() + 4096, nullptr, false};
  ggml_context* ctx = ggml_init(ip);

  gguf_context* o = gguf_init_empty();
  gguf_set_val_str(o, "general.architecture", "mlp-extract");
  gguf_set_val_str(o, "mlp.architecture", arch.c_str());
  const char* base = strrchr(src, '/');
  gguf_set_val_str(o, "mlp.source", base ? base + 1 : src);
  gguf_set_val_u32(o, "mlp.layer", (uint32_t)layer);

  std::vector<std::vector<uint8_t>> src_bytes(3);
  int64_t n_embd = 0, n_ff = 0;
  for (int k = 0; k < 3; k++) {
    const char* name = gguf_get_tensor_name(g, ids[k]);
    const int64_t* ne = gguf_get_tensor_ne(g, ids[k]);
    ggml_type type = gguf_get_tensor_type(g, ids[k]);
    ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne[0], ne[1]);
    ggml_set_name(t, name);
    size_t nbytes = gguf_get_tensor_size(g, ids[k]);
    if (ggml_nbytes(t) != nbytes) die("size mismatch for %s", name);
    src_bytes[k] = read_bytes(src, gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ids[k]), nbytes);
    memcpy(t->data, src_bytes[k].data(), nbytes);
    gguf_add_tensor(o, t);
    printf("  %-24s %-6s %-14s %8.2f MiB\n", name, ggml_type_name(type), shape(ne).c_str(),
           nbytes / 1048576.0);
    if (k == 0) { n_embd = ne[0]; n_ff = ne[1]; }
  }
  gguf_set_val_u32(o, "mlp.n_embd", (uint32_t)n_embd);
  gguf_set_val_u32(o, "mlp.n_ff", (uint32_t)n_ff);
  if (!gguf_write_to_file(o, out, false)) die("failed to write %s", out);

  // Verify: re-open the output and compare every byte with the source.
  gguf_context* chk = open_meta(out);
  for (int k = 0; k < 3; k++) {
    const char* name = gguf_get_tensor_name(g, ids[k]);
    int64_t j = gguf_find_tensor(chk, name);
    if (j < 0) die("verify: %s missing in output", name);
    if (gguf_get_tensor_type(chk, j) != gguf_get_tensor_type(g, ids[k])) die("verify: type differs: %s", name);
    auto got = read_bytes(out, gguf_get_data_offset(chk) + gguf_get_tensor_offset(chk, j),
                          gguf_get_tensor_size(chk, j));
    if (got != src_bytes[k]) die("verify: bytes differ: %s", name);
  }
  printf("wrote %s (layer %d of %s), verified byte-identical\n", out, layer, base ? base + 1 : src);
  gguf_free(chk);
  gguf_free(o);
  ggml_free(ctx);
  gguf_free(g);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 3 && !strcmp(argv[1], "list")) return cmd_list(argv[2]);
  if (argc == 5 && !strcmp(argv[1], "extract")) return cmd_extract(argv[2], atoi(argv[3]), argv[4]);
  fprintf(stderr,
          "usage: mlp-extract list <model.gguf>\n"
          "       mlp-extract extract <model.gguf> <layer> <out.gguf>\n");
  return 2;
}
