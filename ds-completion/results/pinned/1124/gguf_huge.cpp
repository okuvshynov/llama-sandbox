// scratch: does llama.cpp's GGUF reader take a file that lives on huge pages (a hugetlb memfd, by /proc/self/fd/N)?
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
int main(int argc, char** argv) {
  const int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
  const size_t H = 2ull << 20, padded = (st.st_size + H - 1) / H * H;
  const int mfd = memfd_create("ds-gguf", MFD_HUGETLB);
  if (mfd < 0 || ftruncate(mfd, padded)) { perror("memfd"); return 1; }
  char* w = (char*)mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
  if (w == MAP_FAILED) { perror("mmap"); return 1; }
  if (pread(fd, w, st.st_size, 0) != st.st_size) { perror("pread"); return 1; }
  const std::string path = "/proc/self/fd/" + std::to_string(mfd);
  struct stat hs; stat(path.c_str(), &hs);
  gguf_init_params ip = {true, nullptr};
  gguf_context* a = gguf_init_from_file(argv[1], ip);
  gguf_context* b = gguf_init_from_file(path.c_str(), ip);
  printf("the file: %ld bytes; on huge pages: %ld bytes (padded to 2 MB)\n", (long)st.st_size, (long)hs.st_size);
  if (!b) { printf("the reader REFUSED the copy on huge pages\n"); return 1; }
  int same = gguf_get_n_tensors(a) == gguf_get_n_tensors(b) && gguf_get_n_kv(a) == gguf_get_n_kv(b) && gguf_get_data_offset(a) == gguf_get_data_offset(b);
  for (int64_t i = 0; same && i < gguf_get_n_tensors(a); i++)
    same = !strcmp(gguf_get_tensor_name(a, i), gguf_get_tensor_name(b, i)) && gguf_get_tensor_offset(a, i) == gguf_get_tensor_offset(b, i) && gguf_get_tensor_size(a, i) == gguf_get_tensor_size(b, i);
  printf("the reader on the copy: %ld tensors, %ld keys, data offset %zu: %s the original\n", (long)gguf_get_n_tensors(b), (long)gguf_get_n_kv(b), gguf_get_data_offset(b), same ? "the same as" : "DIFFERENT from");
  char buf[16]; printf("read() of the copy's first bytes: %s\n", pread(mfd, buf, 4, 0) == 4 && !memcmp(buf, "GGUF", 4) ? "works (GGUF)" : "fails");
  return !same;
}
