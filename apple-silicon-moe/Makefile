# apple-silicon-moe — the hip-moe study replicated on Apple Silicon (M2 Ultra)
# Links against the shared llama.cpp build (GGML_BACKEND_DL=OFF: Metal/CPU/BLAS
# register statically via libggml).

LLAMA ?= $(HOME)/projects/3rd/llama.cpp
CXXFLAGS = -O2 -std=c++17 -I$(LLAMA)/ggml/include
LDFLAGS  = -L$(LLAMA)/build/bin -lggml -lggml-base -lggml-cpu \
           -Wl,-rpath,$(LLAMA)/build/bin

all: bin/moe-ggml-bench bin/ceilings bin/moe-contention-bench

bin/moe-contention-bench: src/moe_contention_bench.cpp
	@mkdir -p bin
	clang++ $(CXXFLAGS) -o $@ $< $(LDFLAGS)

bin/moe-ggml-bench: src/moe_ggml_bench.cpp
	@mkdir -p bin
	clang++ $(CXXFLAGS) -o $@ $< $(LDFLAGS)

bin/ceilings: src/ceilings.mm
	@mkdir -p bin
	clang++ -O2 -std=c++17 -fobjc-arc -o $@ $< -framework Metal -framework Foundation

clean:
	rm -rf bin

.PHONY: all clean
