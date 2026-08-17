HIPCC ?= hipcc
ARCH  ?= gfx906

bin/hip-moe-bench: src/main.hip
	@mkdir -p bin
	$(HIPCC) -O3 -std=c++17 --offload-arch=$(ARCH) -o $@ $<

clean:
	rm -rf bin

.PHONY: clean
