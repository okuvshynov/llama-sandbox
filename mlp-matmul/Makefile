# mlp-matmul: one dense MLP layer through llama.cpp's own HIP matmul kernels.
# Links against llama.cpp's HIP build (ROCm 6.3.4, gfx906, GGML_BACKEND_DL=OFF).
LLAMA ?= $(HOME)/projects/llama.cpp
BUILD ?= $(LLAMA)/build-hip
CXXFLAGS = -O2 -std=c++17 -Wall -I$(LLAMA)/ggml/include
LDFLAGS  = -L$(BUILD)/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$(BUILD)/bin

BINS = bin/mlp-extract bin/mm-bench bin/mm-split bin/ceiling

all: $(BINS)

bin/mlp-extract: src/mlp_extract.cpp
	@mkdir -p bin
	g++ $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# -O3 -march=native -fopenmp-simd: the host reference matmul (double accumulation)
bin/mm-bench: src/mm_bench.cpp src/common.h
	@mkdir -p bin
	g++ $(CXXFLAGS) -O3 -march=native -fopenmp-simd -o $@ $< $(LDFLAGS) -lpthread

bin/mm-split: src/mm_split.cpp src/common.h
	@mkdir -p bin
	g++ $(CXXFLAGS) -O3 -march=native -fopenmp-simd -o $@ $< $(LDFLAGS) -lpthread

# Raw HIP, no ggml: memory-bandwidth and FMA ceilings of one die.
bin/ceiling: src/ceiling.hip
	@mkdir -p bin
	/opt/rocm-6.3.4/bin/hipcc -O3 --offload-arch=gfx906 -o $@ $<

clean:
	rm -rf bin

.PHONY: all clean
