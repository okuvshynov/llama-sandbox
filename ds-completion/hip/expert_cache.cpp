#include "expert_cache.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
  } while (0)

ExpertCache::ExpertCache(const Model& m, std::vector<ExpertPool*> pools_, std::vector<int>& table_, int* d_table_, int n_expert_,
                         Options o)
    : n_expert(n_expert_), n_pairs((int)table_.size()), n_dies((int)pools_.size()), opt(o), table(table_), d_table(d_table_),
      pools(pools_) {
  slot_owner.resize(n_dies);
  for (int d = 0; d < n_dies; d++) slot_owner[d].assign(pools[d]->n_slots, -1);
  for (int k = 0; k < n_pairs; k++)
    if (table[k] >= 0) slot_owner[table[k] >> 16][table[k] & 0xFFFF] = k;
  loading.assign(n_pairs, 0);
  count.assign(n_pairs, 0);
  last_use.assign(n_pairs, -1);
  last_tick = std::chrono::steady_clock::now();
  loader.reset(new ExpertLoader(m, pools, true));  // direct loads once the tensors are registered
  if (o.wait && !loader->wait_direct()) throw std::runtime_error("expert cache: registering the expert tensors failed");
}

ExpertCache::~ExpertCache() { loader.reset(); }

void ExpertCache::observe(int L, const MoeList* lists, int M, int nd) {
  for (int t = 0; t < M; t++)
    for (int p = 0; p <= nd; p++) {
      const MoeList& l = lists[(size_t)t * kMoeListStride + p];
      for (int j = 0; j < l.count; j++) {
        const int key = p < nd ? slot_owner[p][l.ids[j]] : L * n_expert + l.ids[j];
        if (key >= 0) cur.push_back(key);
      }
    }
}

void ExpertCache::rebalance() {
  st.rebalances++;
  std::vector<int> cand, res;
  for (int k = 0; k < n_pairs; k++) {
    if (loading[k]) continue;
    if (table[k] >= 0) res.push_back(k);
    else if (count[k] >= 2) cand.push_back(k);
  }
  // newcomers: most used first (ties: most recent); victims: least used first (ties: least recent)
  std::sort(cand.begin(), cand.end(), [&](int a, int b) { return count[a] != count[b] ? count[a] > count[b] : last_use[a] > last_use[b]; });
  std::sort(res.begin(), res.end(), [&](int a, int b) { return count[a] != count[b] ? count[a] < count[b] : last_use[a] < last_use[b]; });
  plan.clear();
  for (size_t i = 0; i < cand.size() && i < res.size(); i++) {
    if (count[cand[i]] <= count[res[i]]) break;
    plan.push_back({cand[i], res[i]});
  }
  st.swaps_planned += (long)plan.size();
}

void ExpertCache::step(int M) {
  // the forward's tokens enter the window, the oldest leave it
  for (int k : cur) {
    count[k]++;
    last_use[k] = tokens;
  }
  hist.emplace_back(M, std::move(cur));
  cur.clear();
  hist_tokens += M;
  tokens += M;
  while (!hist.empty() && hist_tokens - hist.front().first >= opt.window) {
    for (int k : hist.front().second) count[k]--;
    hist_tokens -= hist.front().first;
    hist.pop_front();
  }
  // 1. publish finished loads
  for (const ExpertLoader::Request& r : loader->take_done()) {
    const int key = r.layer * n_expert + r.expert;
    slot_owner[r.die][r.slot] = key;
    loading[key] = 0;
    table[key] = r.die << 16 | r.slot;
    table_dirty = true;
    st.loads_published++;
  }
  // 2. rebalance when due
  if (opt.rate > 0 && tokens >= next_rebalance) {
    rebalance();
    next_rebalance = tokens + opt.period;
  }
  // 3. start swaps within the rate: victims leave the table first
  const auto now = std::chrono::steady_clock::now();
  bucket = std::min(bucket + opt.rate * std::chrono::duration<double>(now - last_tick).count(), std::max(1.0, opt.rate * 0.2));
  last_tick = now;
  std::vector<ExpertLoader::Request> start;
  while (!plan.empty() && bucket >= 1.0 && loader->direct_ready()) {
    const Swap s = plan.front();
    plan.pop_front();
    if (table[s.victim] < 0 || table[s.pair] >= 0 || loading[s.pair]) continue;  // overtaken
    const int die = table[s.victim] >> 16, slot = table[s.victim] & 0xFFFF;
    table[s.victim] = -1;
    slot_owner[die][slot] = -1;
    loading[s.pair] = 1;
    start.push_back({die, slot, s.pair / n_expert, s.pair % n_expert});
    bucket -= 1.0;
  }
  if (table_dirty || !start.empty()) {
    CHECK(hipMemcpy(d_table, table.data(), table.size() * sizeof(int), hipMemcpyHostToDevice));
    table_dirty = false;
  }
  // 4. only now may the victims' slots be overwritten
  for (const ExpertLoader::Request& r : start) loader->submit(r);
  st.loads_started += (long)start.size();
}
