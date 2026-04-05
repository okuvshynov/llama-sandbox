#include "stats_io.h"

#include <cstdio>
#include <cstring>

static constexpr size_t HEADER_SIZE = 32;

bool stats_write(const std::string & path, const stats_file & s) {
    FILE * fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "stats_write: cannot open '%s'\n", path.c_str());
        return false;
    }

    fwrite(STATS_MAGIC, 1, 8, fp);

    uint32_t version = STATS_VERSION;
    fwrite(&version,   4, 1, fp);
    fwrite(&s.n_gen,   4, 1, fp);
    fwrite(&s.n_prompt, 4, 1, fp);
    fwrite(&s.temp,    4, 1, fp);

    char reserved[8] = {};
    fwrite(reserved, 1, 8, fp);

    fwrite(s.kl.data(), sizeof(float), s.n_gen, fp);
    fwrite(s.top1_match.data(), 1, s.n_gen, fp);

    fclose(fp);
    return true;
}

bool stats_read(const std::string & path, stats_file & s) {
    FILE * fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "stats_read: cannot open '%s'\n", path.c_str());
        return false;
    }

    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, STATS_MAGIC, 8) != 0) {
        fprintf(stderr, "stats_read: bad magic in '%s'\n", path.c_str());
        fclose(fp);
        return false;
    }

    uint32_t version;
    if (fread(&version, 4, 1, fp) != 1 || version != STATS_VERSION) {
        fprintf(stderr, "stats_read: unsupported version %u in '%s'\n", version, path.c_str());
        fclose(fp);
        return false;
    }

    if (fread(&s.n_gen,    4, 1, fp) != 1) { fclose(fp); return false; }
    if (fread(&s.n_prompt,  4, 1, fp) != 1) { fclose(fp); return false; }
    if (fread(&s.temp,     4, 1, fp) != 1) { fclose(fp); return false; }

    // skip reserved
    if (fseek(fp, 8, SEEK_CUR) != 0) { fclose(fp); return false; }

    s.version = version;
    s.kl.resize(s.n_gen);
    s.top1_match.resize(s.n_gen);

    if ((int32_t)fread(s.kl.data(), sizeof(float), s.n_gen, fp) != s.n_gen) {
        fprintf(stderr, "stats_read: truncated kl data in '%s'\n", path.c_str());
        fclose(fp); return false;
    }
    if ((int32_t)fread(s.top1_match.data(), 1, s.n_gen, fp) != s.n_gen) {
        fprintf(stderr, "stats_read: truncated top1 data in '%s'\n", path.c_str());
        fclose(fp); return false;
    }

    fclose(fp);
    return true;
}
