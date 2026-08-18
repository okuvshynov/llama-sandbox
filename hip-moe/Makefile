HIPCC ?= hipcc
ARCH  ?= gfx906
LLAMA ?= $(HOME)/projects/llama.cpp

all: bin/hip-moe-bench bin/moe-ggml-bench

bin/hip-moe-bench: src/main.hip
	@mkdir -p bin
	$(HIPCC) -O3 -std=c++17 --offload-arch=$(ARCH) -o $@ $<

# One V4-Pro MoE layer through llama.cpp's own HIP kernels (build-hip must
# exist; the ggml backend libs are loaded from there at runtime).
bin/moe-ggml-bench: src/moe_ggml_bench.cpp
	@mkdir -p bin
	g++ -O2 -std=c++17 -I$(LLAMA)/ggml/include -o $@ $< \
	    -L$(LLAMA)/build-hip/bin -lggml -lggml-base -lggml-cpu \
	    -Wl,-rpath,$(LLAMA)/build-hip/bin

clean:
	rm -rf bin

.PHONY: all clean

bin/moe-ep-bench: src/moe_ep_bench.cpp
	@mkdir -p bin
	g++ -O2 -std=c++17 -I$(LLAMA)/ggml/include -o $@ $< \
	    -L$(LLAMA)/build-hip/bin -lggml -lggml-base -lggml-cpu \
	    -Wl,-rpath,$(LLAMA)/build-hip/bin
