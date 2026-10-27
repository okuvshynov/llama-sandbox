# ds-completion: links ggml (and, for the tokenizer only, libllama) from the llama.cpp fork that
# carries DeepSeek-V4.1 support (~/forks/llama.cpp).
#
#   make          CPU build: the fork's build/ (static libraries) -> bin/
#   make HIP=1    GPU build: the fork's build-hip/ (ROCm 6.3.4, gfx906, shared libraries) -> bin-hip/
#                 (build-hip needs patches/fork-hip-ballot-sync.patch applied to the fork)
FORK ?= $(HOME)/forks/llama.cpp
ifeq ($(HIP),1)
BUILD = build-hip
BIN   = bin-hip
GGML_LIBS = -L$(FORK)/$(BUILD)/bin -Wl,-rpath,$(FORK)/$(BUILD)/bin -lggml -lggml-base -lggml-cpu -lggml-hip
LLAMA_LIB = -lllama
else
BUILD = build
BIN   = bin
GGML_LIBS = $(FORK)/$(BUILD)/ggml/src/libggml.a $(FORK)/$(BUILD)/ggml/src/libggml-cpu.a \
            $(FORK)/$(BUILD)/ggml/src/libggml-base.a
LLAMA_LIB = $(FORK)/$(BUILD)/src/libllama.a
endif

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -g -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter \
           -I$(FORK)/ggml/include -I$(FORK)/include -Isrc
LDLIBS   = $(GGML_LIBS) -fopenmp -lpthread -ldl

CORE   = src/model.cpp src/forward.cpp src/fakequant.cpp
CORE_H = src/model.h src/forward.h src/fakequant.h

all: $(BIN)/ds-completion $(BIN)/ds-inspect $(BIN)/ds-test $(BIN)/test-fakequant $(BIN)/llama-logits $(BIN)/fp8pack
ifeq ($(HIP),1)
all: $(BIN)/test-matmul-prec $(BIN)/bench-fp8 $(BIN)/bench-ggml-bf16
endif

# GPU only: ggml's bf16 matmul on the dense shapes (the step-4 baseline for bench-fp8)
$(BIN)/bench-ggml-bf16: src/bench_ggml_bf16.cpp
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

# our own HIP kernels (PLAN.md B1 onwards), compiled with the ROCm the fork was built against
HIPCC    = /opt/rocm-6.3.4/bin/hipcc
HIPFLAGS = -std=c++17 -O3 --offload-arch=gfx906 -Ihip -Isrc -I$(FORK)/ggml/include

$(BIN)/bench-fp8: hip/bench_fp8.cpp hip/fp8_gemv.hip hip/fp8_gemv.h src/fakequant.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ hip/bench_fp8.cpp hip/fp8_gemv.hip src/fakequant.cpp \
	    -L$(FORK)/$(BUILD)/bin -Wl,-rpath,$(FORK)/$(BUILD)/bin -lggml -lggml-base

# GPU only: precision of ggml's bf16 matmul by batch size
$(BIN)/test-matmul-prec: src/test_matmul_prec.cpp
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

# the CLI: libllama only for the tokenizer (vocab-only load)
$(BIN)/ds-completion: src/ds_completion.cpp $(CORE) $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/ds_completion.cpp $(CORE) $(LLAMA_LIB) $(LDLIBS)

$(BIN)/ds-test: src/ds_test.cpp $(CORE) $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/ds_test.cpp $(CORE) $(LDLIBS)

$(BIN)/ds-inspect: src/ds_inspect.cpp src/model.cpp src/model.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/ds_inspect.cpp src/model.cpp $(LDLIBS)

$(BIN)/fp8pack: src/fp8pack.cpp src/model.cpp src/model.h src/fakequant.cpp src/fakequant.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/fp8pack.cpp src/model.cpp src/fakequant.cpp $(LDLIBS)

$(BIN)/test-fakequant: src/test_fakequant.cpp src/fakequant.cpp src/fakequant.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/test_fakequant.cpp src/fakequant.cpp $(LDLIBS)

# the fork's own llama.cpp pipeline: the "second opinion" for the reference
$(BIN)/llama-logits: ref/llama_logits.cpp
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LLAMA_LIB) $(LDLIBS)

clean:
	rm -rf bin bin-hip
.PHONY: all clean
