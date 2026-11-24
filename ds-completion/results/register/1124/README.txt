# Probes and outputs of the registration and stall study of 2026-11-24 (docs/registration-and-stalls.md).
# Commit c6c42fb (engine unchanged; a scratch build with scratch-build.diff for the profiled runs); sclk 6 (manual, all four cards);
# booted 2026-11-24 08:26; kernel 6.10.5-1-t2-jammy; ROCm 6.3.4. README step 201.
#
# Build lines:
#   g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I/opt/rocm-6.3.4/include <file>.cpp -o <name> -L/opt/rocm-6.3.4/lib -Wl,-rpath,/opt/rocm-6.3.4/lib -lamdhip64 -lhsa-runtime64 -lpthread
#   g++ -O2 -std=c++17 -fPIC -shared -I/opt/rocm-6.3.4/include shim.cpp -o shim.so -ldl          (LD_PRELOAD=shim.so <probe>)
#   /opt/rocm-6.3.4/bin/hipcc -std=c++17 -O3 --offload-arch=gfx906 -o <name> <die_*.cpp> -lpthread
# $M3 / $M4 = shards 3 and 4 of ~/llms/UD-Q8_K_XL.
#
# Registration
#   fill_faults.cpp   drops a file from the page cache and brings G GiB back by 16 threads' page faults (6.27 MB blocks, random order)
#   reg_hsa.cpp       hsa_amd_memory_lock for 1 / 2 / 4 dies over page cache and anonymous memory     out/reg_hsa.out (cat-filled), out/pieces.out (fault-filled, by piece size; its row labels say "read by cat": they are stale)
#   shim.cpp          counts the HSA lock calls HIP makes                                               out/hip.out (reg_flags of step 189 under it)
#   reg_copy.cpp      lock call shapes; HIP copies from registered / HSA-locked / pageable memory       out/copy.out (cat-filled), out/faults2.out (fault-filled)
#   reg_pre.cpp       one HSA lock, then hipHostRegister                                                out/pre.out
#   lock_copy.cpp     a lock per expert for one die + hsa_amd_memory_async_copy into HIP's memory       out/lc.out
#   vram.py           each card's used memory (sysfs) sampled through a command                         (numbers in the report)
#   busy_die.cpp      a lock for one die while die 0 is idle / copying                                  out/busy4.out
#   out/faults.out    reg_flags after fill_faults; out/long.err: the engine's lines of the 2,731-token traced run
#
# Stalls
#   die_load.cpp      400 small kernels + one 1 GiB kernel on die 0 under background patterns           out/dl.out
#   die_dir.cpp       the same, copies by direction and die                                             out/dir.out
#   die_stall.cpp     tiny / mid / long kernels, one background mode per process                        out/stall.out is the FIRST build's
#                     (its mid and long kernels came out at 656 us: only the tiny column and the modes' copy rates are of use)
#   scratch-build.diff  the three environment switches of the profiled engine: DS_HOME_ADAPT (step 86's rule off),
#                     DS_LOAD_NO_UNPACK and DS_LOAD_PINNED (die 0's loads only)
#   out/prof-*.err    the engine's lines of the four profiled runs:
#                     DS_HOME_ADAPT=1 [DS_LOAD_NO_UNPACK=1 | DS_LOAD_PINNED=1] rocprofv3 --kernel-trace --memory-copy-trace -d prof -o <name> --output-format csv --
#                       bin-hip/ds-completion -m $M --rt dense-fp8.gguf --placement placements/study30.txt --dspark ... --spec --canon
#                       --cache-loads 800 --cache-period 8 --cache-wait -p "<cpp-matrix>" --chat -n 150
#   analyze2.py <name>  die 0's main-stream kernels by the load copies in flight (reads <name>_kernel_trace.csv, <name>_memory_copy_trace.csv)
#   The profiler's csv files (108 MB) are not kept here: the runs above write them again in about two minutes each.
