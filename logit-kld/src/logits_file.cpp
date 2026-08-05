#include "logits_file.h"

#include <cstdio>

namespace {

bool write_raw(FILE * fp, const void * data, size_t size) {
    return fwrite(data, 1, size, fp) == size;
}

bool write_u32(FILE * fp, uint32_t v)  { return write_raw(fp, &v, sizeof(v)); }
bool write_i32(FILE * fp, int32_t v)   { return write_raw(fp, &v, sizeof(v)); }

bool write_str(FILE * fp, const std::string & s) {
    return write_u32(fp, (uint32_t)s.size()) && write_raw(fp, s.data(), s.size());
}

} // namespace

bool lkld_write(const std::string & path, const lkld_file & f) {
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "lkld_write: cannot open '%s'\n", path.c_str());
        return false;
    }

    bool ok = write_raw(fp, LKLD_MAGIC, 8)
           && write_u32(fp, LKLD_VERSION)
           && write_i32(fp, f.n_vocab)
           && write_i32(fp, f.top_k)
           && write_i32(fp, (int32_t)f.seqs.size())
           && write_str(fp, f.model_desc);

    for (const lkld_seq & s : f.seqs) {
        if (!ok) break;
        ok = write_str(fp, s.label)
          && write_i32(fp, s.n_prompt)
          && write_i32(fp, s.n_total)
          && write_i32(fp, (int32_t)s.positions.size())
          && write_raw(fp, s.tokens.data(), s.tokens.size() * sizeof(int32_t));
        for (const lkld_position & p : s.positions) {
            if (!ok) break;
            ok = write_raw(fp, &p.max_logit, sizeof(float))
              && write_raw(fp, &p.lse_rest, sizeof(float))
              && write_raw(fp, p.ids.data(),    p.ids.size()    * sizeof(int32_t))
              && write_raw(fp, p.logits.data(), p.logits.size() * sizeof(float));
        }
    }

    if (fclose(fp) != 0) ok = false;
    if (!ok) {
        fprintf(stderr, "lkld_write: write failed for '%s'\n", path.c_str());
    }
    return ok;
}
