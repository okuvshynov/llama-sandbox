// The runtime's observability: the startup memory ledger, --profile's groups and the CPU's misses
// folded into them, and --trace (README step 51). The forward itself is in rt.cpp; it calls
// mark() at each group's start and the trace_* hooks around a step.
#include "rt.h"

#include "rt_internal.h"

void Runtime::mem_mark(const char* phase) {
  std::vector<size_t> used;
  int cur;
  CHECK(hipGetDevice(&cur));
  if (mem_total.empty()) mem_total.assign(mem_devs.size(), 0);
  for (size_t i = 0; i < mem_devs.size(); i++) {
    size_t fr, tot;
    CHECK(hipSetDevice(mem_devs[i]));
    CHECK(hipMemGetInfo(&fr, &tot));
    used.push_back(tot - fr);
    mem_total[i] = tot;
  }
  CHECK(hipSetDevice(cur));
  mem_snaps.push_back({phase, used});
}

// one line per phase: GB per device (and, for the expert dies, what it would hold in expert slots)
void Runtime::mem_report() const {
  if (mem_snaps.empty()) return;
  const double slot = 3.0 * hp.n_embd * hp.n_ff_exp / 32 * 17;  // an expert slot: gate + up + down in mxfp4
  fprintf(stderr, "# GPU memory (GB; the driver's used memory after each startup phase; expert slot = %.1f MB):\n", slot / 1e6);
  fprintf(stderr, "#   %-62s", "phase");
  for (int d : mem_devs) fprintf(stderr, "  GPU %d", d);
  fprintf(stderr, "\n");
  for (size_t p = 0; p < mem_snaps.size(); p++) {
    fprintf(stderr, "#   %-62s", mem_snaps[p].first.c_str());
    for (size_t i = 0; i < mem_devs.size(); i++) {
      const double b = (double)mem_snaps[p].second[i] - (p ? (double)mem_snaps[p - 1].second[i] : 0.0);
      fprintf(stderr, " %6.2f", b / 1e9);
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "#   %-62s", "free");
  for (size_t i = 0; i < mem_devs.size(); i++) fprintf(stderr, " %6.2f", (double)(mem_total[i] - mem_snaps.back().second[i]) / 1e9);
  fprintf(stderr, "\n#   %-62s", "total");
  for (size_t i = 0; i < mem_devs.size(); i++) fprintf(stderr, " %6.2f", (double)mem_total[i] / 1e9);
  fprintf(stderr, "\n");
}

void Runtime::fold_misses() {
  for (const MissRec& r : miss_recs) {
    float w = 0;
    if (r.mark >= 0 && r.mark + 1 < (int)marks.size()) CHECK(hipEventElapsedTime(&w, marks[r.mark].second, marks[r.mark + 1].second));
    const int mb = r.M == 1 ? 0 : r.M <= 3 ? 1 : r.M <= 6 ? 2 : 3;
    const int db = r.distinct <= 4 ? r.distinct : r.distinct <= 7 ? 5 : 6;
    MissCell& c = miss_hist[mb][db];
    c.n++;
    c.pairs += r.pairs;
    c.cpu_ms += r.cpu_ms;
    c.wait_ms += w;
  }
  miss_recs.clear();
}

void Runtime::set_trace(const std::string& path) {
  trace_path = path;
  profile = true;
  trace_epoch = Clock::now();
  if (cache && cache->loader_ptr()) cache->loader_ptr()->trace_on = true;  // README step 86
  CHECK(hipEventCreate(&tr_staged));
  die_trace.resize(dies.size());
  for (size_t i = 0; i < dies.size(); i++) {
    DieTrace& t = die_trace[i];
    t.st = dies[i].st ? dies[i].st : s_local;
    CHECK(hipSetDevice(dies[i].dev));
    CHECK(hipEventCreate(&t.base));
  }
  CHECK(hipSetDevice(opt.gpu));
}

void Runtime::trace_host(const char* name, Clock::time_point a, Clock::time_point b, int tid, const std::string& args) {
  if (trace_path.empty()) return;
  const double ts = std::chrono::duration<double, std::micro>(a - trace_epoch).count();
  trace_evs.push_back({name, tid, ts, std::chrono::duration<double, std::micro>(b - a).count(), args});
}

hipEvent_t Runtime::trace_ev(int die) {
  DieTrace& t = die_trace[die];
  if (t.used == t.pool.size()) {
    hipEvent_t e;
    CHECK(hipEventCreate(&e));
    t.pool.push_back(e);
  }
  return t.pool[t.used++];
}

// at a step's start: a reference event on every die's expert stream. The other devices' streams
// are idle then, so the host waits for the event and reads its clock (exact to the wake-up); the
// home die's second stream shares die 0's clock and is placed from die 0's anchor at the end.
void Runtime::trace_step_begin() {
  if (trace_path.empty()) return;
  step_anchor_us = 1e300;
  for (size_t i = 0; i < die_trace.size(); i++) {
    DieTrace& t = die_trace[i];
    t.used = 0;
    t.spans.clear();
    CHECK(hipSetDevice(dies[i].dev));
    CHECK(hipEventRecord(t.base, t.st));
  }
  for (size_t i = 0; i < die_trace.size(); i++) {
    if (!dies[i].st) continue;
    CHECK(hipEventSynchronize(die_trace[i].base));
    die_trace[i].base_us = std::chrono::duration<double, std::micro>(Clock::now() - trace_epoch).count();
  }
  CHECK(hipSetDevice(opt.gpu));
}

// at a step's end (its events have completed): die 0's groups per layer, the dies' expert spans
void Runtime::trace_step_end() {
  if (trace_path.empty() || marks.empty()) return;
  // each group ends where the next begins, so a layer's span (its groups in a row; the prompt's
  // two halves interleave, giving two per layer) contains them exactly and Perfetto nests them
  const double t0 = step_anchor_us < 1e299 ? step_anchor_us : step_t0_us;
  std::vector<double> at(marks.size());
  for (size_t i = 0; i < marks.size(); i++) {
    float a;
    CHECK(hipEventElapsedTime(&a, marks[0].second, marks[i].second));
    at[i] = t0 + a * 1000.0;
  }
  for (size_t i = 0; i < die_trace.size(); i++)
    if (!dies[i].st) {
      float a;
      CHECK(hipEventElapsedTime(&a, marks[0].second, die_trace[i].base));
      die_trace[i].base_us = t0 + a * 1000.0;
    }
  if (step_anchor_us < 1e299)  // generation (anchored clocks): the issue-to-start lags
    for (size_t i = 0; i + 1 < marks.size(); i++) {
      const double dur = at[i + 1] - at[i], lag = at[i + 1] - mark_host[i + 1];
      gen_gpu_us += dur;
      const int b = lag < 0 ? 0 : lag < 30 ? 1 : lag < 100 ? 2 : lag < 300 ? 3 : lag < 1000 ? 4 : 5;
      lag_hist[b]++;
      if (lag < 30) {
        hb_total_us += dur;
        hb_by_next[marks[i + 1].first] += dur;
      }
    }
  for (size_t i = 0; i + 1 < marks.size(); i++) {
    const int L = mark_layer[i];
    if (L >= 0 && (i == 0 || mark_layer[i - 1] != L)) {
      size_t j = i;
      while (j + 1 < marks.size() && mark_layer[j] == L) j++;
      trace_evs.push_back({"layer " + std::to_string(L), 2, at[i], at[j] - at[i], ""});
    }
    std::string name = marks[i].first;
    if (L >= 0) name += " L" + std::to_string(L);
    trace_evs.push_back({name, 2, at[i], at[i + 1] - at[i], ""});
  }
  for (size_t i = 0; i < die_trace.size(); i++) {
    DieTrace& t = die_trace[i];
    CHECK(hipSetDevice(dies[i].dev));
    for (const DieTrace::Span& sp : t.spans) {
      float a, d;
      CHECK(hipEventSynchronize(sp.b));
      CHECK(hipEventElapsedTime(&a, t.base, sp.a));
      CHECK(hipEventElapsedTime(&d, sp.a, sp.b));
      trace_evs.push_back({"experts L" + std::to_string(sp.L), 10 + (int)i, t.base_us + a * 1000.0, d * 1000.0, sp.args});
    }
  }
  CHECK(hipSetDevice(opt.gpu));
}

void Runtime::trace_write() {
  if (trace_path.empty()) return;
  FILE* f = fopen(trace_path.c_str(), "w");
  if (!f) return;
  fprintf(f, "{\"traceEvents\":[\n");
  auto meta = [&](int tid, const std::string& name, bool first = false) {
    fprintf(f, "%s{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%d,\"args\":{\"name\":\"%s\"}}", first ? "" : ",\n", tid, name.c_str());
  };
  meta(1, "host thread", true);
  meta(3, "steps (host)");
  if (L0 > 0) meta(4, "messages from the front");
  meta(2, "GPU " + std::to_string(opt.gpu) + " main stream (dense part)");
  for (size_t i = 0; i < dies.size(); i++)
    meta(10 + (int)i, "GPU " + std::to_string(dies[i].dev) + (dies[i].st ? " experts" : " experts (second stream)"));
  if (cache && cache->loader_ptr()) {  // README step 86: the cache's background loads, per die
    for (size_t i = 0; i < dies.size(); i++) meta(20 + (int)i, "GPU " + std::to_string(dies[i].dev) + " cache loads (host view)");
    for (const ExpertLoader::LoadSpan& l : cache->loader_ptr()->take_trace())
      trace_host(("load L" + std::to_string(l.layer) + " e" + std::to_string(l.expert)).c_str(), l.a, l.b, 20 + l.die);
  }
  // start and end rounded to 0.1 us each, so a span that ends where its parent ends still nests
  for (const TraceEv& e : trace_evs) {
    const double a = std::round(e.ts * 10) / 10, b = std::round((e.ts + e.dur) * 10) / 10;
    fprintf(f, ",\n{\"name\":\"%s\",\"ph\":\"X\",\"pid\":1,\"tid\":%d,\"ts\":%.1f,\"dur\":%.1f%s%s%s}", e.name.c_str(), e.tid, a,
            b - a, e.args.empty() ? "" : ",\"args\":{", e.args.c_str(), e.args.empty() ? "" : "}");
  }
  // for scripts/trace_merge.py (README step 112): this trace's zero on this machine's monotonic
  // clock, the layers, and the front's clock samples against the back
  fprintf(f, "\n],\n\"clockEpochUs\":%.1f,\"layers\":[%d,%d],\"peerSync\":[",
          std::chrono::duration<double, std::micro>(trace_epoch.time_since_epoch()).count(), L0, L1 - 1);
  if (peer)
    for (size_t i = 0; i < peer->clock_samples.size(); i++)
      fprintf(f, "%s[%.1f,%.1f,%.1f,%.1f]", i ? "," : "", peer->clock_samples[i][0], peer->clock_samples[i][1], peer->clock_samples[i][2], peer->clock_samples[i][3]);
  fprintf(f, "]}\n");
  fclose(f);
  fprintf(stderr, "# trace: %zu events in %s (open in ui.perfetto.dev)\n", trace_evs.size(), trace_path.c_str());
  if (gen_gpu_us > 0) {  // README step 65
    fprintf(stderr, "# GPU 0 issue-to-start lag per group (generation): <0 %ld, 0-30 us %ld, 30-100 %ld, 100-300 %ld, 300-1000 %ld, >1 ms %ld\n",
            lag_hist[0], lag_hist[1], lag_hist[2], lag_hist[3], lag_hist[4], lag_hist[5]);
    fprintf(stderr, "# host-bound estimate: %.1f of %.1f ms on GPU 0's stream (%.1f%%) in groups whose next group started within 30 us of being issued; by that group:",
            hb_total_us / 1e3, gen_gpu_us / 1e3, 100 * hb_total_us / gen_gpu_us);
    std::vector<std::pair<double, std::string>> v;
    for (auto& [k, x] : hb_by_next) v.push_back({x, k});
    std::sort(v.rbegin(), v.rend());
    for (size_t i = 0; i < v.size() && i < 8; i++) fprintf(stderr, " %s %.1f", v[i].second.c_str(), v[i].first / 1e3);
    fprintf(stderr, " ms\n");
  }
}

void Runtime::mark(const char* group) {
  if (!profile) return;
  if (marks.size() == ev_pool.size()) {
    hipEvent_t e;
    CHECK(hipEventCreate(&e));
    ev_pool.push_back(e);
  }
  hipEvent_t e = ev_pool[marks.size()];
  if (marks.empty()) step_t0_us = std::chrono::duration<double, std::micro>(Clock::now() - trace_epoch).count();
  CHECK(hipEventRecord(e, s));
  marks.emplace_back(group, e);
  if (mark_layer.size() < marks.size()) mark_layer.resize(marks.size());
  mark_layer[marks.size() - 1] = cur_layer;
  if (!trace_path.empty()) {
    if (mark_host.size() < marks.size()) mark_host.resize(marks.size());
    mark_host[marks.size() - 1] = std::chrono::duration<double, std::micro>(Clock::now() - trace_epoch).count();
  }
}

// after a decode step's end_step(), with or without the head: the profile's groups, the misses
// folded into them, the step's spans into the trace, the lookahead study's step
void Runtime::step_stats() {
  if (profile) {
    last_groups.clear();
    for (size_t i = 0; i + 1 < marks.size(); i++) {
      float ms;
      CHECK(hipEventElapsedTime(&ms, marks[i].second, marks[i + 1].second));
      last_groups[marks[i].first] += ms;
    }
    fold_misses();
  }
  trace_step_end();
}
