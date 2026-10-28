# ds-completion: links ggml (and, for the tokenizer only, libllama) from the llama.cpp fork that
# carries DeepSeek-V4.1 support (~/forks/llama.cpp).
#
#   make          CPU build -> bin/
#   make HIP=1    also the own GPU runtime (hip/, ROCm 6.3.4, gfx906) -> bin-hip/
# Both link the fork's CPU build (build/, static libraries); ggml's own GPU backend is not used.
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
CXXFLAGS = -std=c++17 -O2 -g -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter \
           -I$(FORK)/ggml/include -I$(FORK)/include -Isrc
LDLIBS   = $(GGML_LIBS) -fopenmp -lpthread -ldl

CORE   = src/model.cpp src/forward.cpp src/fakequant.cpp src/cpu_moe.cpp
CORE_H = src/model.h src/forward.h src/fakequant.h src/cpu_moe.h

# HIP builds: ds-completion and ds-test also carry the own decode runtime (hip/), compiled by
# hipcc into objects that the g++ link picks up
ROCM = /opt/rocm-6.3.4
ifeq ($(HIP),1)
RT_OBJ   = $(BIN)/obj/rt.o $(BIN)/obj/kernels.o $(BIN)/obj/fp8_gemv.o $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o \
           $(BIN)/obj/dspark.o
RT_FLAGS = -DDS_RUNTIME -D__HIP_PLATFORM_AMD__ -Ihip -I$(ROCM)/include
RT_LIBS  = $(RT_OBJ) -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64
endif

all: $(BIN)/ds-completion $(BIN)/ds-inspect $(BIN)/ds-test $(BIN)/test-fakequant $(BIN)/llama-logits $(BIN)/fp8pack $(BIN)/bench-cpu-moe
ifeq ($(HIP),1)
all: $(BIN)/bench-fp8 $(BIN)/bench-kernels $(BIN)/bench-fp4-moe $(BIN)/bench-expert-pool
endif

# our own HIP kernels (PLAN.md B1 onwards)
HIPCC    = $(ROCM)/bin/hipcc
HIPFLAGS = -std=c++17 -O3 -fPIC --offload-arch=gfx906 -Ihip -Isrc -I$(FORK)/ggml/include

$(BIN)/bench-fp8: hip/bench_fp8.cpp hip/fp8_gemv.hip hip/fp8_gemv.h src/fakequant.cpp
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -fopenmp -o $@ hip/bench_fp8.cpp hip/fp8_gemv.hip src/fakequant.cpp \
	    -L$(FORK)/$(BUILD)/ggml/src -lggml -lggml-cpu -lggml-base -lpthread -ldl

# the CLI: libllama only for the tokenizer (vocab-only load)
$(BIN)/ds-completion: src/ds_completion.cpp $(CORE) $(CORE_H) $(RT_OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -o $@ src/ds_completion.cpp $(CORE) $(RT_LIBS) $(LLAMA_LIB) $(LDLIBS)

$(BIN)/ds-test: src/ds_test.cpp $(CORE) $(CORE_H) $(RT_OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -o $@ src/ds_test.cpp $(CORE) $(RT_LIBS) $(LDLIBS)

$(BIN)/bench-kernels: hip/bench_kernels.cpp hip/kernels.hip hip/fp8_gemv.hip hip/kernels.h hip/fp8_gemv.h
	@mkdir -p $(BIN)
	$(HIPCC) $(HIPFLAGS) -o $@ hip/bench_kernels.cpp hip/kernels.hip hip/fp8_gemv.hip

$(BIN)/bench-fp4-moe: hip/bench_fp4_moe.cpp $(BIN)/obj/fp4_moe.o src/model.cpp src/cpu_moe.cpp src/fakequant.cpp $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -O3 -o $@ hip/bench_fp4_moe.cpp src/model.cpp src/cpu_moe.cpp src/fakequant.cpp \
	    $(BIN)/obj/fp4_moe.o -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/bench-expert-pool: hip/bench_expert_pool.cpp $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o src/model.cpp src/cpu_moe.cpp src/fakequant.cpp $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(RT_FLAGS) -O3 -o $@ hip/bench_expert_pool.cpp src/model.cpp src/cpu_moe.cpp src/fakequant.cpp \
	    $(BIN)/obj/fp4_moe.o $(BIN)/obj/expert_pool.o -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lamdhip64 $(LDLIBS)

$(BIN)/obj/%.o: hip/%.hip hip/kernels.h hip/fp8_gemv.h hip/fp4_moe.h hip/expert_pool.h
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/rt.o: hip/rt.cpp hip/rt.h hip/dspark.h hip/kernels.h hip/fp8_gemv.h hip/fp4_moe.h hip/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/obj/dspark.o: hip/dspark.cpp hip/rt.h hip/dspark.h hip/kernels.h hip/fp8_gemv.h hip/fp4_moe.h hip/expert_pool.h $(CORE_H)
	@mkdir -p $(BIN)/obj
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(BIN)/ds-inspect: src/ds_inspect.cpp src/model.cpp src/model.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/ds_inspect.cpp src/model.cpp $(LDLIBS)

$(BIN)/fp8pack: src/fp8pack.cpp src/model.cpp src/model.h src/fakequant.cpp src/fakequant.h
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/fp8pack.cpp src/model.cpp src/fakequant.cpp $(LDLIBS)

$(BIN)/bench-cpu-moe: src/bench_cpu_moe.cpp $(CORE) $(CORE_H)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -O3 -o $@ src/bench_cpu_moe.cpp $(CORE) $(LDLIBS)

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
