# ds-completion: links ggml (and, for the tokenizer only, libllama) from a llama.cpp checkout
# (FORK, default ~/forks/llama.cpp): upstream, at the commit QUICKSTART.md names (README steps 168, 175).
#
# src/ and hip/ are the libraries (CPU side, GPU runtime); apps/, tests/ and bench/ the programs.
#
#   make          CPU build -> bin/
#   make HIP=1    also the own GPU runtime (hip/, ROCm 6.3.4, gfx906) -> bin-hip/
#   make product, make tests, make tools (each also with HIP=1): one group of programs, below
# Both link llama.cpp's CPU build (build/, static libraries); ggml's own GPU backend is not used.
FORK ?= $(HOME)/forks/llama.cpp
BUILD = build
GGML_LIBS = $(FORK)/$(BUILD)/ggml/src/libggml.a $(FORK)/$(BUILD)/ggml/src/libggml-cpu.a \
            $(FORK)/$(BUILD)/ggml/src/libggml-base.a
LLAMA_LIB = $(FORK)/$(BUILD)/src/libllama.a
ifeq ($(HIP),1)
BIN = bin-hip
else
BIN = bin
endif

CXX      = g++
CXXFLAGS = -std=c++17 -O3 -g -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter \
           -I$(FORK)/ggml/include -I$(FORK)/include -Isrc
LDLIBS   = $(GGML_LIBS) -fopenmp -lpthread -ldl

CORE   = src/model.cpp src/engram.cpp src/fakequant.cpp src/cpu_moe.cpp src/canon_dot.cpp src/sampler.cpp
# the CPU reference path (the model as ggml graphs): ds-ref, ds-test and a benchmark link it; ds-completion does not
REF    = src/forward.cpp src/fakequant_graph.cpp
CORE_H = src/model.h src/engram.h src/taps.h src/options.h src/forward.h src/fakequant.h src/fakequant_graph.h src/cpu_moe.h src/canon_dot.h src/sampler.h

# HIP builds: ds-completion and ds-test also carry the own decode runtime (hip/), compiled by
# hipcc into objects that the g++ link picks up
ROCM = /opt/rocm-6.3.4
ifeq ($(HIP),1)
RT_OBJ   = $(BIN)/obj/rt.o $(BIN)/obj/rt_trace.o $(BIN)/obj/rt_peer.o $(BIN)/obj/kernels.o $(BIN)/obj/fp8_gemv.o $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o \
           $(BIN)/obj/expert_loader.o $(BIN)/obj/expert_cache.o $(BIN)/obj/dspark.o
RT_FLAGS = -DDS_RUNTIME -D__HIP_PLATFORM_AMD__ -Ihip -I$(ROCM)/include
RT_LIBS  = $(RT_OBJ) -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64
endif

# Three groups (README step 169; docs/overview.md "Code map"). `make` builds all three.
#   make product   what serves: ds-completion (HIP=1 only: it is the GPU runtime's program), and fp8pack,
#                  which makes dense-fp8.gguf once
#   make tests     what checks it: ds-ref (the CPU reference path as a program; without HIP=1), ds-test and
#                  the unit tests
#   Not in a group: llama-logits (make bin/llama-logits), llama.cpp's own pipeline on the model. It gives
#   right logits only when FORK is the DeepSeek-V4.1 PR's commit (docs/llama-cpp-upstream.md): upstream
#   reads this model's Engram table as another type, without an error (README step 175).
#   make tools     what measures and inspects: ds-inspect, the benchmarks
# With HIP=1 each is the bin-hip/ build of the same group (and the programs only it has).
PRODUCT = $(BIN)/fp8pack
TESTS   = $(BIN)/ds-test $(BIN)/test-fakequant $(BIN)/test-sampler
TOOLS   = $(BIN)/ds-inspect $(BIN)/bench-cpu-moe $(BIN)/bench-fp4-decode
ifeq ($(HIP),1)
PRODUCT += $(BIN)/ds-completion
TESTS  += $(BIN)/test-topk
TOOLS  += $(BIN)/bench-fp8 $(BIN)/bench-kernels $(BIN)/bench-fp4-moe $(BIN)/bench-expert-pool $(BIN)/bench-expert-load \
          $(BIN)/bench-register $(BIN)/bench-launch $(BIN)/bench-loader-exit $(BIN)/bench-row-stride $(BIN)/bench-wave-slots
else
TESTS  += $(BIN)/ds-ref
endif

all: product tests tools
product: $(PRODUCT)
tests: $(TESTS)
tools: $(TOOLS)

# our own HIP kernels (PLAN.md B1 onwards)
HIPCC    = $(ROCM)/bin/hipcc
HIPFLAGS = -std=c++17 -O3 -fPIC --offload-arch=gfx906 -Ihip -Isrc -I$(FORK)/ggml/include

$(BIN)/bench-fp8: bench/bench_fp8.cpp hip/fp8_gemv.hip hip/fp8_gemv.h src/fakequant.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -fopenmp -o $@ bench/bench_fp8.cpp hip/fp8_gemv.hip src/fakequant.cpp \
	    -L$(FORK)/$(BUILD)/ggml/src -lggml -lggml-cpu -lggml-base -lpthread -ldl

# the CLI, the product: the GPU runtime, no CPU path (README step 173). libllama only for the
# tokenizer (vocab-only load). Without HIP=1 there is no such program: the CPU path is ds-ref.
$(BIN)/ds-completion: apps/ds_completion.cpp apps/record.h apps/think_view.h $(CORE) $(CORE_H) $(RT_OBJ)
ifneq ($(HIP),1)
	$(error ds-completion is the GPU runtime's program: make HIP=1 (the CPU reference path: make bin/ds-ref))
endif
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -o $@ apps/ds_completion.cpp $(CORE) $(RT_LIBS) $(LLAMA_LIB) $(LDLIBS)

# the same CLI over the CPU reference path (the ggml Engine), no runtime: what bin/ds-completion
# was until step 173
$(BIN)/ds-ref: apps/ds_completion.cpp apps/record.h apps/think_view.h $(CORE) $(REF) $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ apps/ds_completion.cpp $(CORE) $(REF) $(LLAMA_LIB) $(LDLIBS)

$(BIN)/ds-test: tests/ds_test.cpp $(CORE) $(REF) $(CORE_H) $(RT_OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -o $@ tests/ds_test.cpp $(CORE) $(REF) $(RT_LIBS) $(LDLIBS)

$(BIN)/bench-kernels: bench/bench_kernels.cpp hip/kernels.hip hip/fp8_gemv.hip hip/kernels.h hip/fp8_gemv.h
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ bench/bench_kernels.cpp hip/kernels.hip hip/fp8_gemv.hip

$(BIN)/bench-row-stride: bench/bench_row_stride.hip
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ bench/bench_row_stride.hip

$(BIN)/bench-wave-slots: bench/bench_wave_slots.hip
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ bench/bench_wave_slots.hip

$(BIN)/bench-loader-exit: bench/bench_loader_exit.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ bench/bench_loader_exit.cpp -lpthread

$(BIN)/bench-launch: bench/bench_launch.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ bench/bench_launch.cpp

$(BIN)/bench-register: bench/bench_register.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ bench/bench_register.cpp

$(BIN)/test-topk: tests/test_topk.cpp hip/kernels.hip hip/kernels.h
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ tests/test_topk.cpp hip/kernels.hip

$(BIN)/bench-fp4-moe: bench/bench_fp4_moe.cpp $(BIN)/obj/fp4_moe.o src/model.cpp src/cpu_moe.cpp src/fakequant.cpp $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -O3 -o $@ bench/bench_fp4_moe.cpp src/model.cpp src/cpu_moe.cpp src/canon_dot.cpp src/fakequant.cpp \
	    $(BIN)/obj/fp4_moe.o -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/bench-expert-load: bench/bench_expert_load.cpp $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o $(BIN)/obj/expert_loader.o src/model.cpp src/cpu_moe.cpp src/fakequant.cpp $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -O3 -o $@ bench/bench_expert_load.cpp src/model.cpp src/cpu_moe.cpp src/canon_dot.cpp src/fakequant.cpp \
	    $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o $(BIN)/obj/expert_loader.o -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/bench-expert-pool: bench/bench_expert_pool.cpp $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o src/model.cpp src/cpu_moe.cpp src/fakequant.cpp $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -O3 -o $@ bench/bench_expert_pool.cpp src/model.cpp src/cpu_moe.cpp src/canon_dot.cpp src/fakequant.cpp \
	    $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/obj/%.o: hip/%.hip hip/kernels.h hip/fp8_gemv.h hip/fp4_moe.h hip/expert_pool.h
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/expert_loader.o: hip/expert_loader.cpp hip/expert_loader.h hip/expert_pool.h hip/fp4_moe.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/expert_cache.o: hip/expert_cache.cpp hip/expert_cache.h hip/expert_loader.h hip/expert_pool.h hip/kernels.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/rt.o: hip/rt.cpp hip/rt.h hip/rt_internal.h hip/peer.h hip/expert_cache.h hip/expert_loader.h hip/dspark.h hip/kernels.h hip/fp8_gemv.h hip/fp4_moe.h hip/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/rt_trace.o: hip/rt_trace.cpp hip/rt.h hip/rt_internal.h hip/peer.h hip/expert_cache.h hip/expert_loader.h hip/dspark.h hip/kernels.h hip/fp8_gemv.h hip/fp4_moe.h hip/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/rt_peer.o: hip/rt_peer.cpp hip/rt.h hip/rt_internal.h hip/peer.h hip/expert_cache.h hip/expert_loader.h hip/dspark.h hip/kernels.h hip/fp8_gemv.h hip/fp4_moe.h hip/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/dspark.o: hip/dspark.cpp hip/rt.h hip/peer.h hip/dspark.h hip/kernels.h hip/fp8_gemv.h hip/fp4_moe.h hip/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/ds-inspect: apps/ds_inspect.cpp src/model.cpp src/model.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ apps/ds_inspect.cpp src/model.cpp $(LDLIBS)

$(BIN)/fp8pack: apps/fp8pack.cpp src/model.cpp src/model.h src/fakequant.cpp src/fakequant.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ apps/fp8pack.cpp src/model.cpp src/fakequant.cpp $(LDLIBS)

$(BIN)/bench-cpu-moe: bench/bench_cpu_moe.cpp $(CORE) $(REF) $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -O3 -o $@ bench/bench_cpu_moe.cpp $(CORE) $(REF) $(LDLIBS)

$(BIN)/bench-fp4-decode: bench/bench_fp4_decode.cpp src/canon_dot.cpp src/canon_dot.h src/fakequant.cpp src/fakequant.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ bench/bench_fp4_decode.cpp src/canon_dot.cpp src/fakequant.cpp $(LDLIBS)

# the sampler alone: no model, no GPU (README step 148)
$(BIN)/test-sampler: tests/test_sampler.cpp src/sampler.cpp src/sampler.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_sampler.cpp src/sampler.cpp

$(BIN)/test-fakequant: tests/test_fakequant.cpp src/fakequant.cpp src/fakequant_graph.cpp src/fakequant.h src/fakequant_graph.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_fakequant.cpp src/fakequant.cpp src/fakequant_graph.cpp $(LDLIBS)

# llama.cpp's own pipeline: the "second opinion" for the reference (steps 2-3). Not in `make`: with
# upstream llama.cpp it builds and runs but its logits are wrong (the Engram table's type, above);
# it needs FORK=<a checkout of the PR's commit>, docs/llama-cpp-upstream.md.
$(BIN)/llama-logits: ref/llama_logits.cpp
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LLAMA_LIB) $(LDLIBS)

clean:
	rm -rf bin bin-hip bin-asan
.PHONY: all product tests tools clean
