# Profiling the GPU side with rocprofv3

`--profile` and `--trace` show where a step's time goes as the host sees it. `rocprofv3`
(AMD's profiler for ROCm 6) adds what they cannot: the time each kernel runs on the device,
every HIP call the runtime makes, and every memory copy. It needs no rebuild.

First tried on 2026-11-16 (ROCm 6.3.4, gfx906, a 64-token plain run); the numbers below are
from that one run and are examples, not measurements to quote.

## Install

    sudo apt install rocprofiler-sdk6.3.4        # -> /opt/rocm-6.3.4/bin/rocprofv3

The `rocprof` and `rocprofv2` in `/usr/bin` belong to ROCm 5.7.1 and do not fit our build.

## Run

Put `rocprofv3 [options] --` in front of the usual command. Keep the run short (tens of
tokens): the trace has one line per kernel launch, about 2,200 per generated token here.

    RP=/opt/rocm-6.3.4/bin/rocprofv3

    # kernels only: the cheapest useful mode
    $RP --kernel-trace --stats -d prof -o k --output-format csv -- \
        bin-hip/ds-completion -m $M -p "..." --chat -n 64 --rt $F8 --canon \
        --placement placements/study30.txt

    # kernels + HIP calls + memory copies, and a timeline to look at
    $RP --runtime-trace --stats -d prof -o r --output-format csv pftrace -- \
        bin-hip/ds-completion ...

- `-d DIR -o NAME`: the files are `DIR/NAME_*.csv` (and `DIR/NAME_results.pftrace`). Without
  them they go to `<hostname>/<pid>/`.
- `--stats` writes the per-kernel and per-call totals; without it only the raw traces.
- `-S` prints a summary table to stderr at exit. With `--summary-output-file` give a bare
  name: the path is taken inside `DIR`.
- The profiler writes its own lines to stderr ("Opened result file ..."), mixed with ours.

What it costs, one run each:

| mode | generation | wall time |
|---|---|---|
| no profiler | 24.3 t/s | 17 s |
| `--kernel-trace` | 19.3 t/s (-21%) | 18 s |
| `--runtime-trace` with `pftrace` | 18.0 t/s (-26%) | 31 s (writing ~160 MB at exit) |

The output's bytes do not change. The speed does, so a profiled run is never a speed
measurement.

## The files

| file | what | read it for |
|---|---|---|
| `NAME_kernel_stats.csv` | per kernel: launches, total, average, min, max, share | the ranking of kernels by device time |
| `NAME_kernel_trace.csv` | per launch: kernel, die (`Agent_Id`), stream (`Queue_Id`), start and end, grid size | counts per step or per die, a kernel's time against its size |
| `NAME_hip_api_stats.csv` | per HIP call: count, total, average | how many launches, event and copy calls the runtime makes |
| `NAME_hip_api_trace.csv` | per HIP call: thread, start, end | which thread calls what, and when |
| `NAME_memory_copy_stats.csv`, `_trace.csv` | copies by direction; per copy its size and time | the load at startup, the per-step copies |
| `NAME_domain_stats.csv` | one line per kind of data | totals |
| `NAME_agent_info.csv` | the devices | `Agent_Id` 1-4 are the four dies in device order; 1 is the home die |
| `NAME_results.pftrace` | everything on a timeline | looking at it |

The stats files are small; a spreadsheet or a few lines of Python read them:

    python3 -c "
    import csv, re, sys
    rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: -float(r['TotalDurationNs']))
    for r in rows[:15]:
        print('%-40s %7s launches  avg %8.1f us  %5.1f%%' % (re.sub(r'\(.*', '', r['Name'])[:40],
              r['Calls'], float(r['AverageNs']) / 1e3, float(r['Percentage'])))
    " prof/k_kernel_stats.csv

## The timeline

Open `NAME_results.pftrace` at https://ui.perfetto.dev ("Open trace file"; the file is read by
the browser and not uploaded). There are tracks per thread for HIP calls and per die and
stream for kernels and copies. W/S zoom, A/D move, a click on a span shows its details. Our
own `--trace` file opens in the same viewer, in another tab; the two are not merged and their
clocks differ.

## Reading the numbers

- **Everything covers the whole process:** startup, the prompt and the generation. The
  16.5 s of host-to-device copies in the example is the load of the weights, not a cost of a
  step. To look at generation only, cut the kernel trace at the output head's launches
  (`gemv_partial_kernel<unsigned short, ...>`, one per token): in the example 27,562 launches
  come before the first one (startup and the prompt), then 2,197-2,242 per token.
- **A kernel's duration is its time on the device** (start to end of the dispatch), without
  the launch on the host and without the wait before it starts. This is the column our own
  trace does not have. The sum over kernels (2.9 s in the example) is over four dies working
  at once, so it is not comparable with wall time.
- **`Min` against `Average` of one kernel** says whether it does the same work every time
  (`unpack_blocks`: 17.9 and 21.7 us) or is one name for different sizes
  (`canon_gate_up`: 3.7 and 44.7 us; the grid size in the trace tells them apart).
- **Host-side durations include the profiler.** `hipLaunchKernel` averaged 8.5 us here against
  2.2-2.5 us measured without a profiler (README step 49). Use the HIP files for *counts*
  (launches, `hipEventRecord`, `hipStreamWaitEvent`, copies), not for times.
- **Per die:** in the example the home die took 119,876 of 167,441 launches and each other
  die 15,855: the dense part against the experts of one die.
- **A share is not a saving.** A kernel with 16% of device time on a die that waits for the
  CPU's experts anyway may gain nothing when made faster; `--trace` and
  `scripts/trace_critpath.py` say what the step waits for. Use rocprofv3 to rank and to count,
  then measure a change with the ordinary runs.

## A die's busy share

`results/rocprof/busy.py DIR/NAME_kernel_trace.csv [TOKENS]` (README steps 143-144; TOKENS, the
number generated, is needed for a speculative run): per die, over the
generation, the time a kernel is running, the idle gaps by length, and for the home die
between which kernels the long gaps fall. It answers "is the die waiting, and for what" from
the device's side; the run is traced, so the gaps include the profiler's host cost.

## Hardware counters (`--pmc`)

One more package, which `rocprofiler-sdk` does not pull in (without it: `undefined symbol:
aqlprofile_register_agent`):

    sudo apt install hsa-amd-aqlprofile6.3.4

    $RP -L                                   # the counters: 86 per die on gfx906
    timeout 1500 $RP --pmc MemUnitBusy MemUnitStalled -d prof -o p --output-format csv -- \
        bin-hip/ds-completion ... -n 16

`prof/p_counter_collection.csv` has one line per launch and counter: the kernel, the die, the
grid, the counter's value, start and end.

- **Few counters per run.** A set that needs more hardware counters than the card has is
  refused ("error code 38: Request exceeds the capabilities of the hardware to collect"):
  the five of the table below together are refused, and so is `L2CacheHit` with `FetchSize`.
  Accepted here: each alone, `MemUnitBusy MemUnitStalled`, `VALUBusy VALUUtilization`,
  `VALUBusy VALUInsts VFetchInsts`. One run per set.
- **Always under `timeout`.** A refused set aborted the model at once, but left a small test
  program hanging. `bin-hip/bench-launch` is the cheap way to ask whether a set is accepted.
- **Very slow:** a 1.6 us launch takes 57-146 us; the model generates at 4-7 t/s where it
  does 26.6. Sixteen tokens are enough. The output's bytes do not change.

Example (`results/rocprof/counters.txt`; generation only, mean over a kernel's launches):

| kernel | MemUnitBusy | MemUnitStalled | L2CacheHit | FetchSize, KB | VALUBusy |
|---|---|---|---|---|---|
| `fp8_gemv_kernel<64,1,1>` | 48.9% | 1.5% | 12.9% | 18,777 | 44.2% |
| `fp8_gemv_kernel<8,1,1>` | 79.4% | 2.3% | 3.0% | 41,024 | 52.8% |
| `canon_gate_up_kernel` | 36.3% | 0.1% | 41.1% | 13,791 | 21.2% |
| `canon_down_kernel` | 31.9% | 0.1% | 43.9% | 6,761 | 22.3% |
| `quant_act_kernel` | 2.0% | 0.2% | 32.6% | 37 | 1.5% |

How to read them, after a check against kernels whose reads are known (`bench-fp8`, one byte
per weight; `results/rocprof/calibration.txt`, README step 142):

- **`FetchSize` is exact:** the bytes read from video memory in KB of 1,024. Eight fp8
  tensors from 2.5 to 150 MB each read their weights once plus 0.1-0.4% (activations,
  scales). Against the size of a kernel's weights it says whether it reads them once: the
  model's `fp8_gemv_kernel<8,1,1>` at 41,024 KB is `attn_q_b` (40,960 KB), once.
- **`TCC_MISS_sum` counts 64-byte lines** (misses x 64 bytes = `FetchSize` to 0.3%), and
  `L2CacheHit` is hits / (hits + misses). A kernel streaming weights it reads once sits at
  3-18%.
- **`MemUnitBusy` and the other "share of time" counters are not comparable between kernels
  of different length:** the same kernel on the same kind of data read 11.7% at 7 us and 91%
  at 209 us. The share is of an interval with a fixed part around the kernel, so for a
  kernel of a few microseconds it is mostly that part (`quant_act`: 2%). Compare them only
  between runs of one kernel at one size.
- The definitions are the expressions in
  `/opt/rocm-6.3.4/share/rocprofiler-sdk/counter_defs.yaml`. The instruction counters
  (`VALUInsts`, `VFetchInsts`) are not checked.

## Not tried yet

`--hsa-trace`, the kernel filters (`--kernel-include-regex`, which should make a counter run on
one kernel cheap), and the timeline in a viewer.
