# Registration and stalls: a study of how experts get onto the GPUs (2026-11-24)

A report on one day of measurements. Two questions were asked:

1. Why does "registering" the model's expert weights with the GPUs take 1 to 5 minutes at every
   start, and can it be made faster or avoided?
2. Why did the engine get 8% faster when the expert cache stopped loading into GPU 0 (step 86)?
   That change was kept because it helped; the mechanism was never found.

Nothing in the engine was changed. The probes, their outputs and the build lines are in
`results/register/1124/` (see its `README.txt`). Numbers are from one boot (commit `c6c42fb`,
clock level 6, kernel 6.10.5, ROCm 6.3.4) and mostly from one or two runs each; where a figure is
an extrapolation or an inference, the text says so.

## Summary

**Registration**

- Registering memory means writing the GPU's own page tables so that it can read pages of system
  RAM directly. It costs one 8-byte entry per 4 KiB page, per GPU: for our 289 GB up to 565 MB of
  page tables on each of the four GPUs, which is most of the 0.75 GB the engine keeps free on a GPU.
- It is slow for three reasons that multiply: the work is done once per GPU; HIP asks for it in
  a wasteful way (four separate requests per piece, costing 2.5 to 3.4 times one request for four
  GPUs); and the physical layout of the page cache changes the rate by a factor of 3 or more.
- One GPU alone registers at 27 to 170 GB/s. Through HIP, four GPUs register at 1 to 10 GB/s.
- HIP has no setting that fixes this. Calling the runtime below HIP does: about 10 to 31 s for
  the 289 GB instead of 39 to 100 s (extrapolated), and the copies run as fast with less CPU.
- Registration is an optimization, not a requirement. The engine is correct without it.

**Stalls**

- While a bulk copy from system RAM is in flight to GPU 0, GPU 0's kernels take 2.2 times as
  long. Copies to the other GPU of the same card cost about 6%; copies to the other card nothing.
- The slowdown is not uniform. About every 90 to 96 µs one kernel is held up by 50 to 70 µs.
- It does not depend on the loader's unpack kernel, on its separate stream, or on whether the
  copy's source is registered or pinned memory. Only not copying to GPU 0 avoids it, which is
  what step 86 did.
- A benchmark of plain kernels with the same copies does not reproduce the stalls, so the
  engine has some ingredient on GPU 0 that is still unidentified.
- Separately, every bulk copy to a card adds a few microseconds to each kernel launch on both
  GPUs of that card. That part is understood and reproduced.

**Recommended**, in order: read the model sequentially before runs (already known, worth 3x);
measure the engine with registration switched off; make GPU 1 the last choice for cache loads;
trace the kernel driver during a copy to GPU 0. Details in "Recommendations".

## Background for readers new to the project

The engine runs one model, DeepSeek-V4.1-Flash, on one machine: a Mac Pro 2019 with 755 GB of
RAM and two Radeon Pro Vega II Duo cards. Each card carries two GPUs ("dies") with 32 GB each,
behind one PCIe 3.0 x16 link per card. Dies 0 and 1 are one card, dies 2 and 3 the other. Die 0
is the "home die": it runs the dense part of the model and the draft model for speculative
decoding. The software stack is ROCm 6.3.4: our code calls HIP (AMD's CUDA-like API), HIP calls
a runtime called ROCr (the `hsa_*` functions), and ROCr calls the kernel driver (amdgpu).

The model is a mixture of experts. Its 289 GB of expert weights do not fit on the GPUs (128 GB
in all), so they stay in system RAM, in a memory-mapped file, and about 6,100 of the most useful
experts are copied into "slots" on the four dies at startup. An expert that is not in a slot is
computed on the CPU. While the engine generates, an **expert cache** watches which experts are
used and swaps them: a background **loader** thread per die copies an expert (18.8 MB, as three
matrices of 6.27 MB) from RAM into a slot. This is a "load". The engine tracks which expert is in
which slot itself; that bookkeeping has nothing to do with registration.

Two facts about the page cache matter below. The model file's bytes sit in RAM as 4 KiB pages
of the kernel's page cache. Those pages can end up physically consecutive (after one sequential
read of the file into free memory: "cat-filled") or scattered (when 16 threads touched them in
the order the model happened to need them: "fault-filled").

## Part 1: registration

### What problem it solves

A copy from RAM to a GPU is done by the GPU: its copy engine reads system memory over PCIe.
It can only read pages that are mapped into its own address space and that will not move while
it reads. Ordinary memory, including a memory-mapped file, is neither. There are three ways to
get an expert's bytes in front of the copy engine:

| path | how | cost |
|---|---|---|
| pinned buffer | the CPU copies the expert into a small permanently mapped buffer, the GPU copies from there | the CPU reads and writes 18.8 MB per load, on the memory bus the CPU's own expert computation is using |
| pageable | give HIP the file mapping's pointer; it maps the pages for the GPU for that one copy | slower on first use; crashed the copy engine with four dies loading (note in `src/expert_pool.h`) |
| registered | map the file's pages for the GPUs once (`hipHostRegister`); every later copy reads the page cache directly | the minutes this report is about; afterwards about 1.4 ms of CPU per load and no CPU memory traffic |

Step 30 chose "registered" for the cache's loads because the pinned buffer slowed the CPU's
expert computation (17% at 100 loads/s against none) and the whole engine by 9% against 1%. The
startup upload of the 6,100 experts uses the pinned buffer.

The engine registers the 289 GB on a background thread after startup, in 64 MB pieces, and the
cache starts no load until it is done. That took 52 s today, 75 s yesterday after a sequential
read, 116 to 139 s on earlier boots, and 285 s on the boot of step 198.

### What a registration is: page tables on the GPU

Each die has its own memory-management unit, and each process has its own virtual address
space on each die, described by page tables that live in the die's memory. They have the same
shape as a CPU's. The driver's boot log says: `vm size is 262144 GB, 4 levels, block size is
9-bit, fragment size is 9-bit`.

Registering a range does this, per die:

1. The driver walks the process's CPU page tables to find the physical page behind every 4 KiB
   of the range, and subscribes to be told if any of them is later moved or dropped.
2. It writes entries into the die's page tables that point at those physical pages. The GPU
   virtual address is made equal to the CPU virtual address, so one pointer means the same bytes
   on both sides.

No data moves and nothing leaves the page cache. Afterwards the copy engine is given a source
address, translates it through these tables like a CPU would, and reads RAM itself.

The fact "this range is registered" is kept at four levels: HIP has a table of registered ranges
by address (which is why one copy may not span two registrations); ROCr has its own record; the
kernel driver has an object with the list of pages; and the die has the page-table entries.

**The unit is the 4 KiB page, and an entry is 8 bytes.** When 512 consecutive pages are also
physically consecutive and aligned, the driver writes one 2 MB entry a level up instead of 512
small ones (the "fragment size" in the log).

**How much memory the page tables take.** 289 GB is 70.5 million pages: 565 MB of entries per
die if every page needs its own. Measured, by sampling each card's used memory (the driver's
counter in sysfs) while 24 GiB were registered on all four dies:

| page cache | growth per die, 24 GiB | scaled to 289 GB |
|---|---|---|
| fault-filled | 45 to 47 MB (96% of the all-4-KiB size) | about 540 MB |
| cat-filled (after an hour of other work) | 27 to 29 MB | about 330 MB |

The engine sizes its expert pools to leave 0.75 GB free on each die, and its memory table is
printed before the registration ends. So after it, a die has 0.2 to 0.4 GB really free. The
reserve is mostly page tables, on all four dies, including die 0, which never reads through
them (below).

### Why it is slow

All figures: 6 GiB of one model shard in 64 MB pieces, wall time.

| | cat-filled | fault-filled |
|---|---|---|
| one request to ROCr, one die | 0.04 to 0.11 s (60 to 170 GB/s) | 0.24 s (27 GB/s) |
| one request, four dies | 0.26 s (25 GB/s) | 0.89 s (7.2 GB/s) |
| `hipHostRegister` (what the engine calls) | 0.8 to 0.9 s (7 to 8 GB/s) | 2.24 s (2.9 GB/s) |

- **The work is per die.** 0.24, 0.47 and 0.91 s for one, two and four dies (fault-filled).
- **HIP's request shape.** A small library placed under HIP showed that `hipHostRegister`
  calls ROCr once per device for each piece, with one die each time. Four such requests for the
  same piece cost 2.5 to 3.4 times one request that names four dies. Making the same four calls
  by hand reproduces HIP's time (0.80 against 0.90 s, 2.22 against 2.24 s).
- **The layout of the page cache.** A factor of 3.4 on this boot, as step 199 found. In the
  scattered case half of the time is waiting rather than CPU work.
- **A likely contributor to the waiting.** The cards ask for a 32 GB window (BAR) through which
  the CPU could address all of a die's memory; the firmware gives 256 MB and the kernel cannot
  enlarge it (`BAR 0 [mem size 0x800000000]: can't assign; no space`). By the driver's parameter
  description it then writes page tables through the GPU's own command queue instead of from the
  CPU. Two observations fit: kernel worker threads named after that queue are busy during a
  registration, and a registration for one die waits longer (0.26 to 0.30 s instead of 0.10 s per
  6 GiB) while the *other die of the same card* is receiving a bulk copy, but not while a die of
  the other card is. This was not confirmed by tracing the kernel.
- **Not the cause:** the flags passed, the number of threads, pieces of 64 MB or larger, or the
  0.5 ms pause the loader takes between pieces (2 s in all).

One thing is unexplained. Yesterday's boot, up for 7 hours, gave 1.0 GB/s fault-filled and 2.4
to 4.2 cat-filled through HIP, about 2.5 times slower than today in both states. It did not
reproduce on a fresh boot.

A larger BAR would at best halve the scattered case and do nothing for the consecutive one
(waiting is 0.46 of 0.89 s and 0.04 of 0.26 s). It would not remove the need to register: the
BAR is about the CPU reaching GPU memory, and registration is about the GPU reaching RAM.

### What was tried

| idea | result |
|---|---|
| one ROCr request for all dies, then `hipHostRegister` | 0.44 against 0.86 s (cat-filled), 1.31 against 1.71 s (scattered): HIP still pays per device afterwards |
| HIP copies from memory locked below HIP | HIP does not know the memory and maps every copy again: 5.6 GB/s and 2.4 ms of CPU per load on first use |
| no registration up front, lock each expert for its die when it is first loaded | 3.2 ms per expert, one at a time across dies: about 240 to 300 loads/s in all |
| lock once with ROCr, and copy with ROCr's own copy call into memory HIP allocated | works, see below |

The last one, measured in a probe (`lock_copy.cpp`):

| | through HIP (today's loader) | below HIP |
|---|---|---|
| copy to one die | 13 GB/s | 12 to 12.7 GB/s |
| copy to four dies at once | 27 GB/s (step 30) | 25 GB/s |
| CPU per load | 1.43 ms | 0.63 to 0.67 ms |
| registering 289 GB, cat-filled | about 39 s | about 10 s |
| registering 289 GB, fault-filled | about 100 s | about 31 s |

The copied bytes read back equal. The 289 GB times are extrapolated from 6 GiB rates, and the
"below HIP" ones lock for dies 1 to 3 only. Caveats: it mixes two API levels, so the unpack
kernel on a HIP stream has to wait for the copy's completion signal by hand; the probe ran 200
loads per die, not hours; and the pageable path's old crash with four dies loading says this
needs a long run before it is trusted.

### Die 0 is registered for nothing

Since step 86 the cache never loads into die 0. HIP does not know that: it maps every piece for
every device. Die 0 gets the same 330 to 540 MB of page tables and a quarter of the per-die work,
and it is the die with the least room. Below HIP the registration can name dies 1 to 3 only:
0.69 against 0.89 s (fault-filled), 0.21 against 0.26 s (cat-filled).

## Part 2: stalls on die 0

### What was known

Step 86 saw that a draft (8 ms of work on die 0) took 14 to 18 ms whenever the cache was
loading an expert into die 0, and stopped loading into die 0. Step 122 measured that rule as
worth 8%. Two guesses, the host thread and the loader's unpack kernel, had been ruled out; the
mechanism stayed open.

### Effect one: a copy taxes every launch on its card (understood)

A microbenchmark: 400 small kernels in a row on die 0 take 0.98 ms with nothing else running.
With 6.27 MB copies running back to back in the background (13 GB/s):

| background copies | host to device | device to host |
|---|---|---|
| die 0 | 2.27 ms | 1.20 ms |
| die 1 (same card) | 2.28 ms | 1.10 ms |
| die 2 (other card) | 0.97 ms | 0.97 ms |

A single kernel reading 1 GiB (32 to 36 ms) is unaffected in every case. So this is a cost per
launch of about 3 µs, shared by the two dies of a card, for copies in the direction RAM to GPU.
It scales with how much of the time a copy is in flight (one copy per ms: 1.30 ms; one per 4 ms:
1.01 ms). A second stream running tiny kernels costs nothing by itself.

The reading that fits: a die fetches each launch's packet and arguments from system memory over
the card's PCIe link, and a bulk copy nearly fills that link (about 13.5 GB/s per card). That the
launch queues live in system memory is from knowledge of the runtime, not checked here. The
sharing per card and the direction are measured.

In the engine this is the 6% that loads into die 1 cost die 0.

### Effect two: periodic stalls when the copy lands on die 0 (not understood)

The engine was built with a switch that turns step 86's rule off and run under AMD's profiler
(`rocprofv3`, kernel and memory-copy traces; cpp-matrix prompt, 150 tokens, about 310 loads
into die 0). Each of the 72,000 kernels on die 0's main stream during generation was classed by
which load-sized copy was in flight when it started. Mean time per kernel:

| run | copy to die 0 | copy to die 1 only | none on this card |
|---|---|---|---|
| the engine's normal load | 47.8 µs | 22.7 µs | 21.4 µs |
| the same, repeated | 47.9 µs | 23.0 µs | 21.6 µs |
| unpack kernel skipped on die 0 | 47.3 µs | 22.2 µs | 21.5 µs |
| pinned path instead of registered on die 0 | 48.9 µs | 25.4 µs | 21.5 µs |

Load copies to die 0 were in flight 15 to 16% of the time. The slowdown is the same with the
unpack kernel removed and with the other source memory, and kernels close to an unpack kernel
are no slower than the rest (49 to 50 against 47 µs).

**The shape.** Kernel run times with a copy to die 0 in flight split into two groups:

| run time | copy to die 0 in flight | no such copy |
|---|---|---|
| under 20 µs | 58% | 74% |
| 20 to 60 µs | 3% | 17% |
| 60 to 100 µs | 32% | 7% |
| over 100 µs | 8% | 3% |

One stretch of die 0's main stream during a single 918 µs copy (µs from the copy's start):

| start | run | kernel |
|---|---|---|
| 216.8 | 88.5 | `copy16` |
| 305.3 | 71.7 | `moe_combine` |
| 377.0 | 8.0 | `hc_post` |
| 385.0 | 7.4 | a device-to-device copy |
| 392.4 | 73.0 | `gemv_partial_m` |
| 465.3 | 10.4 | `split_sum_m` |
| 475.7 | 78.2 | `hc_split` |

Over the four runs, the time from the end of one held-up kernel to the end of the next is
sharply peaked: 57% of 8,426 intervals fall between 80 and 100 µs (quartiles 91, 96, 114). So
while the copy runs, something recurs about every 93 µs and holds up whichever kernel is at the
head of the queue by 50 to 70 µs.

**Which kernels are hit** (pooled over the four runs, a hold-up counted as 40 µs or more above
the kernel's normal time):

| kernel's normal time | held up | mean added |
|---|---|---|
| under 6 µs | 6% | 8 µs |
| 6 to 12 µs | 48% | 33 µs |
| 12 to 25 µs | 52% | 33 µs |
| 25 to 50 µs | 77% | 47 µs |
| 50 to 120 µs | 21% | 20 µs |

Per kernel type the rates are consistent and far apart: `router` 94%, `hc_split` 83%, `copy16`
81%, most fp8 projections 60 to 95%; `hc_post`, `swiglu`, `rope` and `quant_act` 0 to 5%; the
long `attn_partial` (81 µs) 13% against 9% without a copy. The kernels run in a fixed order
within a layer, so a kernel's rate may say more about where it sits in that order than about
what it computes.

**Ruled out**, each by a measurement: the loader's unpack kernel and stream; registered against
pinned source memory; die 0's own small copies waiting behind the load's (one ends inside only
1% of held-up kernels); a queue waking from idle (95% of held-up kernels start when the previous
one ends); kernels of die 0's other streams running at the same time (3% overlap); a message
about queue oversubscription in the kernel log (none).

**Not reproduced outside the engine.** A benchmark with tiny, 23 µs and 75 µs kernels and the
same copies shows only the per-launch tax of effect one (a few µs, equal for die 0 and die 1),
with the kernels' reads packed or spread over 28 GiB. Its test kernel's timing shifted between
builds, so its figures are rough, but there is no sign of 70 µs hold-ups.

A caveat on the profiler's times: a kernel's reported start equals the previous kernel's end in
over 90% of cases, with no gap at all. So a "run time" here is probably the time from a kernel
reaching the head of its queue to its completion, launch latency included, and a hold-up may be
a delay in starting a kernel rather than in running it.

### How this fits step 86

Loads into die 0 make die 0's roughly 2,000 kernels per verify step take 2.2 times as long for
15 to 16% of the time, which is the order of the 8% of step 122. The earlier puzzle that the bus
tax alone was too small is resolved: most of it is effect two. The pinned path makes no
difference to it, in either direction.

## A side measurement: how busy the CPU's experts are

This bears on loading without registration, because pinned loads share the memory bus with the
CPU's expert computation. From the stored traces and one new one:

| request | CPU experts running | CPU's own reads, averaged | cache loads |
|---|---|---|---|
| chat, generation | 26% of the time | 16.6 GB/s | 362/s |
| code, generation | 39% | 25.6 GB/s | 592/s |
| generation right after a 2,731-token prompt | 46% | 31 GB/s | none yet |
| the 2,731-token prompt itself | 92% | not measured | none yet |

Within a run both fall together: in the chat trace from 46% busy and 736 loads/s in the first
eighth to 21% and 197 in the last. As pinned loads, the loader in those runs would read and
write 13.6 to 22.3 GB/s on average, the same order as the CPU kernel's own reads. The gaps
between CPU calls are short (median 0.76 ms, about one matrix's copy).

Busy share is exposure, not predicted loss: step 30 measured +17% on the CPU kernel from only
3.8 GB/s of loader traffic, far more than a share of bandwidth explains.

## Recommendations

1. **Read the model sequentially before runs** (plan item 36, step C1). It is worth about 3x on
   the registration and makes the page tables smaller. Today's whole-model read took 193 s.
2. **Measure the engine with registration off.** `ExpertLoader` already takes a flag for
   pinned-only loads; with it and the cache's wait lifted, compare a chat prompt, a code prompt
   and a long prompt against the baseline. If the loss is small, the registration, its wait, the
   unregister at exit, the memory-map stalls of step 43 and 330 to 540 MB of page tables per die
   all go away with a small change. If it is not, try a lower load rate first.
3. **If registration stays, do it below HIP for dies 1 to 3 only.** About a third of today's
   time, less CPU per load, and die 0 gets its page-table memory back. Turn the probe into a tool
   beside `bench-expert-load` and give it a long four-die run before the loader uses it.
4. **Make die 1 the last choice for a load.** Its copies tax die 0 by about 6% while they fly,
   about a third of the time in these runs. One speed run with victims preferred on dies 2 and 3
   says whether the placement's loss outweighs it.
5. **Do not lower the 0.75 GB reserve** while registration exists. It is page tables, and how
   much depends on the page cache's layout.
6. **Keep step 86's rule.** Nothing about how a load is issued changes the stalls.
7. **Trace the kernel driver during a copy to die 0.** A recurring 93 µs is specific enough that
   the driver's own trace events and the interrupts should show what it is.
8. Apart from this study: the loader threads create their GPU objects at the same moment,
   which is where step 198's crash was. One lock around that, before any loader change.

## Speculation

Clearly marked as such: none of this is measured.

**What if we avoid registration altogether?** Three designs, from least to most work:

- *Pinned buffer only.* The simplest engine: one load path shared by startup and cache, loads
  from the first token, no page tables. The risk is the CPU's expert computation, most of all
  during long prompts, where it runs 92% of the time and there are no gaps to copy in. Policies
  to consider: a lower load rate, no loads during prompt chunks, or copying in small slices only
  while the CPU kernel is idle.
- *Read from the disk.* `pread` with `O_DIRECT` into the pinned buffer lets the NVMe drive
  place the bytes by its own DMA, so the CPU moves nothing. The drive gives about 2.7 GB/s:
  roughly 140 loads/s. That fits only if a low load rate turns out to be enough.
- *Experts on huge pages we own.* Memory of our own on 2 MB pages registers at 30 GB/s for four
  dies below HIP, whatever the page cache looks like, with page tables of a few MB. A private
  copy does not fit in RAM beside the page cache (step 200); it would need the file read without
  the page cache, or a shared-memory copy that outlives the process.

**What if the dense part were a few large kernels instead of about 2,000 small ones per step?**

- The per-launch tax (effect one) is per launch, so it would shrink in proportion. So would the
  launch overhead itself, which the timing page puts at 7 to 12% of a step.
- For the stalls the evidence points the same way, without settling it. The hold-ups come on a
  clock of about 93 µs, and what they cost depends on the kernel: 8 to 50 µs kernels gain 33 to
  47 µs each, the under-6 µs ones mostly slip through, and several 70 to 110 µs kernels are
  barely touched (`attn_partial` +7 µs, two fp8 projections +7 to +9 µs). If a hold-up delays the
  *start* of a kernel and leaves a running one alone, then one long kernel pays once where twenty
  short ones pay ten times, and fusion would remove most of the cost. If instead the die's
  computation is paused, a fused kernel would be paused just the same. Other 75 to 81 µs kernels
  are held up 44 to 74% of the time, so the first reading is not safe yet.
- If fusion did remove the stalls, die 0 could take cache loads again. Step 122 put the price
  of die 0's fixed set at 0.1 points of expert uses on GPUs, so the prize there is small; the
  larger prize would be the launch overhead and the tax from the other dies' loads.
- Plan item 14 already estimates fusion at 37 to 22 launches per layer for 2 to 3 ms per step.
  This study adds a second reason, not a new estimate.

**What might the 93 µs be?** A copy of 6.27 MB at 7 to 10 GB/s moves 0.65 to 0.9 MB in 93 µs,
so it could be the copy engine working in chunks of about that size, with something on the die
(an address-translation flush, a queue hand-over) at each chunk. That is a guess.

## Open questions

- What recurs every 93 µs during a copy to die 0, and what does the engine have on die 0 that
  the benchmark lacks? Kernel-side tracing, or removing die 0's other work piece by piece in a
  scratch build, would answer it.
- Is the period tied to bytes copied or to time? The pinned and registered runs can be compared
  by copy rate.
- Why did yesterday's boot register 2.5 times slower in both layouts?
- What does the pinned-only loader cost today, per kind of request?
- Does the page-table write really go through the GPU's queue, and is that the waiting?

## How it was measured, and what to distrust

- Probes: `results/register/1124/`, with the build lines and which output belongs to which
  probe. The profiler's raw traces (108 MB) were not kept; the commands that write them are in
  the `README.txt` there, about two minutes per run.
- The profiled engine is a scratch copy with three environment switches
  (`scratch-build.diff`). The repository's engine was not changed.
- The profiler slows generation (37.6 to 38.1 t/s against 54.5 unprofiled in step 122), so no total from those
  runs is a speed measurement. In the run with the unpack kernel skipped, die 0's slots held the
  wrong experts: its timing was used, its text is not the model's.
- No kernel-side profile was available (no root; the installed `perf` does not match this
  kernel). Every statement about what the driver does inside is inferred from timing.
- Two early readings were wrong and are corrected above: that a copy on die 0 stops during a
  registration (it does not), and that a second stream's kernels slow the die (the test kernel was
  simply heavy).
