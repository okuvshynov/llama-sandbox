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

BINS = bin/ds-inspect
all: $(BINS)

bin/ds-inspect: src/ds_inspect.cpp src/model.cpp src/model.h
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ src/ds_inspect.cpp src/model.cpp $(LDLIBS)

clean:
	rm -rf bin
.PHONY: all clean
