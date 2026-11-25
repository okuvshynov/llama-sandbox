# ds-completion: links ggml (and, for the tokenizer only, libllama) from a llama.cpp checkout
# (LLAMA_CPP, default ~/projects/llama.cpp): upstream, at the commit QUICKSTART.md names (README steps 168, 175).
#
# src/ is the engine (the files that need ROCm are compiled only with HIP=1); apps/, tests/ and tools/ the programs.
#
#   make          CPU build -> bin/
#   make HIP=1    also the own GPU runtime (hip/, ROCm 6.3.4, gfx906) -> bin-hip/
#   make product, make tests, make tools (each also with HIP=1): one group of programs, below
# Both link llama.cpp's CPU build (build/, static libraries); ggml's own GPU backend is not used.
LLAMA_CPP ?= $(HOME)/projects/llama.cpp
# the variable's name until step 176, in the commands of older steps
ifdef FORK
LLAMA_CPP := $(FORK)
endif
BUILD = build
GGML_LIBS = $(LLAMA_CPP)/$(BUILD)/ggml/src/libggml.a $(LLAMA_CPP)/$(BUILD)/ggml/src/libggml-cpu.a \
            $(LLAMA_CPP)/$(BUILD)/ggml/src/libggml-base.a
LLAMA_LIB = $(LLAMA_CPP)/$(BUILD)/src/libllama.a
ifeq ($(HIP),1)
BIN = bin-hip
else
BIN = bin
endif

CXX      = g++
CXXFLAGS = -std=c++17 -O3 -g -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter \
           -I$(LLAMA_CPP)/ggml/include -I$(LLAMA_CPP)/include -Isrc
LDLIBS   = $(GGML_LIBS) -fopenmp -lpthread -ldl

CORE   = src/model.cpp src/engram.cpp src/fakequant.cpp src/cpu_moe.cpp src/canon_dot.cpp src/sampler.cpp
CORE_H = src/model.h src/engram.h src/taps.h src/options.h src/fakequant.h src/cpu_moe.h src/canon_dot.h src/sampler.h

# HIP builds: ds-completion (built only here) and ds-test carry the own decode runtime (src/rt.cpp and what it uses), compiled by
# hipcc into objects that the g++ link picks up
ROCM = /opt/rocm-6.3.4
ifeq ($(HIP),1)
RT_OBJ   = $(BIN)/obj/rt.o $(BIN)/obj/rt_trace.o $(BIN)/obj/kernels.o $(BIN)/obj/fp8_gemv.o $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o \
           $(BIN)/obj/expert_loader.o $(BIN)/obj/expert_cache.o $(BIN)/obj/dspark.o
RT_FLAGS = -D__HIP_PLATFORM_AMD__ -I$(ROCM)/include
RT_LIBS  = $(RT_OBJ) -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64
endif

# Three groups (README step 169; docs/overview.md "Code map"). `make` builds all three.
#   make product   what serves: ds-completion (HIP=1 only: it is the GPU runtime's program), and fp8pack,
#                  which makes dense-fp8.gguf once
#   make tests     what checks it: the unit tests and, with HIP=1, ds-test (the runtime against the golden sets)
#   Not in a group: llama-logits (make bin/llama-logits), llama.cpp's own pipeline on the model. It gives
#   right logits only when LLAMA_CPP is a checkout of the DeepSeek-V4.1 PR's commit (docs/llama-cpp-upstream.md): upstream
#   reads this model's Engram table as another type, without an error (README step 175).
#   make tools     what measures and inspects: ds-inspect, the benchmarks
# With HIP=1 each is the bin-hip/ build of the same group (and the programs only it has).
PRODUCT = $(BIN)/fp8pack
TESTS   = $(BIN)/test-fakequant $(BIN)/test-sampler
TOOLS   = $(BIN)/ds-inspect $(BIN)/bench-cpu-moe $(BIN)/bench-fp4-decode
ifeq ($(HIP),1)
PRODUCT += $(BIN)/ds-completion
TESTS  += $(BIN)/ds-test $(BIN)/test-topk
TOOLS  += $(BIN)/bench-fp8 $(BIN)/bench-kernels $(BIN)/bench-fp4-moe $(BIN)/bench-expert-pool $(BIN)/bench-expert-load \
          $(BIN)/bench-register $(BIN)/reg-probe $(BIN)/bench-launch $(BIN)/bench-loader-exit $(BIN)/bench-row-stride $(BIN)/bench-wave-slots
endif

all: product tests tools
product: $(PRODUCT)
tests: $(TESTS)
tools: $(TOOLS)

# our own HIP kernels (PLAN.md B1 onwards)
HIPCC    = $(ROCM)/bin/hipcc
HIPFLAGS = -std=c++17 -O3 -fPIC --offload-arch=gfx906 -Isrc -I$(LLAMA_CPP)/ggml/include

$(BIN)/bench-fp8: tools/bench_fp8.cpp src/fp8_gemv.hip src/fp8_gemv.h src/fakequant.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -fopenmp -o $@ tools/bench_fp8.cpp src/fp8_gemv.hip src/fakequant.cpp \
	    -L$(LLAMA_CPP)/$(BUILD)/ggml/src -lggml -lggml-cpu -lggml-base -lpthread -ldl

# the CLI, the product: the GPU runtime, no CPU path (README step 173). libllama only for the
# tokenizer (vocab-only load). Without HIP=1 there is no such program.
$(BIN)/ds-completion: apps/ds_completion.cpp apps/record.h apps/think_view.h $(CORE) $(CORE_H) $(RT_OBJ)
ifneq ($(HIP),1)
	$(error ds-completion is the GPU runtime's program: make HIP=1)
endif
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -o $@ apps/ds_completion.cpp $(CORE) $(RT_LIBS) $(LLAMA_LIB) $(LDLIBS)

# the runtime against the golden sets (HIP=1 only; its CPU mode went with the reference path, README step 194)
$(BIN)/ds-test: tests/ds_test.cpp $(CORE) $(CORE_H) $(RT_OBJ)
ifneq ($(HIP),1)
	$(error ds-test tests the GPU runtime: make HIP=1)
endif
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -o $@ tests/ds_test.cpp $(CORE) $(RT_LIBS) $(LDLIBS)

$(BIN)/bench-kernels: tools/bench_kernels.cpp src/kernels.hip src/fp8_gemv.hip src/kernels.h src/fp8_gemv.h
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ tools/bench_kernels.cpp src/kernels.hip src/fp8_gemv.hip

$(BIN)/bench-row-stride: tools/bench_row_stride.hip
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ tools/bench_row_stride.hip

$(BIN)/bench-wave-slots: tools/bench_wave_slots.hip
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ tools/bench_wave_slots.hip

$(BIN)/bench-loader-exit: tools/bench_loader_exit.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ tools/bench_loader_exit.cpp -lpthread

$(BIN)/bench-launch: tools/bench_launch.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ tools/bench_launch.cpp

$(BIN)/bench-register: tools/bench_register.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ tools/bench_register.cpp

$(BIN)/reg-probe: tools/reg_probe.cpp src/model.cpp src/model.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -o $@ tools/reg_probe.cpp src/model.cpp -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/test-topk: tests/test_topk.cpp src/kernels.hip src/kernels.h
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ tests/test_topk.cpp src/kernels.hip

$(BIN)/bench-fp4-moe: tools/bench_fp4_moe.cpp $(BIN)/obj/fp4_moe.o src/model.cpp src/cpu_moe.cpp src/fakequant.cpp $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -O3 -o $@ tools/bench_fp4_moe.cpp src/model.cpp src/cpu_moe.cpp src/canon_dot.cpp src/fakequant.cpp \
	    $(BIN)/obj/fp4_moe.o -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/bench-expert-load: tools/bench_expert_load.cpp $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o $(BIN)/obj/expert_loader.o src/model.cpp src/cpu_moe.cpp src/fakequant.cpp $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -O3 -o $@ tools/bench_expert_load.cpp src/model.cpp src/cpu_moe.cpp src/canon_dot.cpp src/fakequant.cpp \
	    $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o $(BIN)/obj/expert_loader.o -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/bench-expert-pool: tools/bench_expert_pool.cpp $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o src/model.cpp src/cpu_moe.cpp src/fakequant.cpp $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -O3 -o $@ tools/bench_expert_pool.cpp src/model.cpp src/cpu_moe.cpp src/canon_dot.cpp src/fakequant.cpp \
	    $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/obj/%.o: src/%.hip src/kernels.h src/fp8_gemv.h src/fp4_moe.h src/expert_pool.h
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/expert_loader.o: src/expert_loader.cpp src/expert_loader.h src/expert_pool.h src/fp4_moe.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/expert_cache.o: src/expert_cache.cpp src/expert_cache.h src/expert_loader.h src/expert_pool.h src/kernels.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/rt.o: src/rt.cpp src/rt.h src/rt_internal.h src/expert_cache.h src/expert_loader.h src/dspark.h src/kernels.h src/fp8_gemv.h src/fp4_moe.h src/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/rt_trace.o: src/rt_trace.cpp src/rt.h src/rt_internal.h src/expert_cache.h src/expert_loader.h src/dspark.h src/kernels.h src/fp8_gemv.h src/fp4_moe.h src/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/dspark.o: src/dspark.cpp src/rt.h src/dspark.h src/kernels.h src/fp8_gemv.h src/fp4_moe.h src/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/ds-inspect: tools/ds_inspect.cpp src/model.cpp src/model.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ tools/ds_inspect.cpp src/model.cpp $(LDLIBS)

$(BIN)/fp8pack: apps/fp8pack.cpp src/model.cpp src/model.h src/fakequant.cpp src/fakequant.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ apps/fp8pack.cpp src/model.cpp src/fakequant.cpp $(LDLIBS)

$(BIN)/bench-cpu-moe: tools/bench_cpu_moe.cpp $(CORE) $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -O3 -o $@ tools/bench_cpu_moe.cpp $(CORE) $(LDLIBS)

$(BIN)/bench-fp4-decode: tools/bench_fp4_decode.cpp src/canon_dot.cpp src/canon_dot.h src/fakequant.cpp src/fakequant.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ tools/bench_fp4_decode.cpp src/canon_dot.cpp src/fakequant.cpp $(LDLIBS)

# the sampler alone: no model, no GPU (README step 148)
$(BIN)/test-sampler: tests/test_sampler.cpp src/sampler.cpp src/sampler.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_sampler.cpp src/sampler.cpp

$(BIN)/test-fakequant: tests/test_fakequant.cpp src/fakequant.cpp src/fakequant.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_fakequant.cpp src/fakequant.cpp

# llama.cpp's own pipeline: the "second opinion" for the reference (steps 2-3). Not in `make`: with
# upstream llama.cpp it builds and runs but its logits are wrong (the Engram table's type, above);
# it needs LLAMA_CPP=<a checkout of the PR's commit>, docs/llama-cpp-upstream.md.
$(BIN)/llama-logits: ref/llama_logits.cpp
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LLAMA_LIB) $(LDLIBS)

clean:
	rm -rf bin bin-hip bin-asan
.PHONY: all product tests tools clean
