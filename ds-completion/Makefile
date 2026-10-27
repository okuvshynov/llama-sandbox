# ds-completion: links ggml from the llama.cpp fork that carries DeepSeek-V4.1 support
# (~/forks/llama.cpp, CPU build in build/, static libraries).
FORK  ?= $(HOME)/forks/llama.cpp
BUILD ?= build
GGML_LIBS = $(FORK)/$(BUILD)/ggml/src/libggml.a $(FORK)/$(BUILD)/ggml/src/libggml-cpu.a \
            $(FORK)/$(BUILD)/ggml/src/libggml-base.a

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -g -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter \
           -I$(FORK)/ggml/include -Isrc
LDLIBS   = $(GGML_LIBS) -fopenmp -lpthread -ldl

BINS = bin/ds-completion bin/ds-inspect bin/llama-logits bin/ds-test bin/test-fakequant
all: $(BINS)

bin/ds-inspect: src/ds_inspect.cpp src/model.cpp src/model.h
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ src/ds_inspect.cpp src/model.cpp $(LDLIBS)

CORE = src/model.cpp src/forward.cpp src/fakequant.cpp
CORE_H = src/model.h src/forward.h src/fakequant.h

# the CLI: libllama only for the tokenizer (vocab-only load)
bin/ds-completion: src/ds_completion.cpp $(CORE) $(CORE_H)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -I$(FORK)/include -o $@ src/ds_completion.cpp $(CORE) $(FORK)/$(BUILD)/src/libllama.a $(LDLIBS)

bin/ds-test: src/ds_test.cpp $(CORE) $(CORE_H)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ src/ds_test.cpp $(CORE) $(LDLIBS)

bin/test-fakequant: src/test_fakequant.cpp src/fakequant.cpp src/fakequant.h
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ src/test_fakequant.cpp src/fakequant.cpp $(LDLIBS)

# the fork's libllama, CPU only: the "second opinion" for the reference
bin/llama-logits: ref/llama_logits.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -I$(FORK)/include -o $@ $< $(FORK)/$(BUILD)/src/libllama.a $(LDLIBS)

clean:
	rm -rf bin
.PHONY: all clean
