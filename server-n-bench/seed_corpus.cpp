// seed_corpus.cpp
//
// Reference implementations of a few canonical data structures, used as
// seed text for prompt-length-controlled benchmarking by bench_n.py. The
// file is read verbatim at startup, POSTed to /tokenize, the resulting
// token list is repeated/truncated to a target length, and then sent
// through /detokenize to produce chat-completion input. The whole point
// of using code (rather than prose) is to make the token distribution
// look like a real coding workload so that MoE routing decisions match
// what would happen in production.
//
// Nothing in this file is invoked by the benchmark; it exists purely to
// have its bytes tokenized. Code is intentionally written in a normal,
// production-style way (templates, iterators, RAII, error handling) so
// the experts that fire are the ones a coding session would actually
// activate.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace seed {

// ============================================================================
// LRU cache
// ============================================================================
//
// Bounded least-recently-used cache. `get()` promotes the touched entry to
// the front of an intrusive doubly-linked list; `put()` evicts the tail when
// the cache is full. Both operations are amortized O(1) thanks to a side
// hash map of node pointers.
//
// Thread safety: none. A single-mutex wrapper is fine for low-contention
// callers; for hot paths consider a sharded variant keyed on hash(key) %
// shard_count, or replace the hash map with a concurrent map.

template <typename K, typename V>
class LRUCache {
public:
    explicit LRUCache(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("LRUCache capacity must be positive");
        }
        index_.reserve(capacity_ * 2);
    }

    LRUCache(const LRUCache&) = delete;
    LRUCache& operator=(const LRUCache&) = delete;

    ~LRUCache() { clear(); }

    bool get(const K& key, V& out) {
        auto it = index_.find(key);
        if (it == index_.end()) {
            return false;
        }
        promote(it->second);
        out = it->second->value;
        return true;
    }

    void put(const K& key, V value) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            it->second->value = std::move(value);
            promote(it->second);
            return;
        }
        if (size_ == capacity_) {
            evict_tail();
        }
        Node* node = new Node{key, std::move(value), nullptr, nullptr};
        push_front(node);
        index_.emplace(key, node);
        ++size_;
    }

    bool contains(const K& key) const {
        return index_.find(key) != index_.end();
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }

    void clear() noexcept {
        Node* cur = head_;
        while (cur) {
            Node* next = cur->next;
            delete cur;
            cur = next;
        }
        head_ = tail_ = nullptr;
        index_.clear();
        size_ = 0;
    }

private:
    struct Node {
        K key;
        V value;
        Node* prev;
        Node* next;
    };

    void promote(Node* node) {
        if (node == head_) return;
        unlink(node);
        push_front(node);
    }

    void push_front(Node* node) {
        node->prev = nullptr;
        node->next = head_;
        if (head_) head_->prev = node;
        head_ = node;
        if (!tail_) tail_ = node;
    }

    void unlink(Node* node) {
        if (node->prev) node->prev->next = node->next;
        else            head_ = node->next;
        if (node->next) node->next->prev = node->prev;
        else            tail_ = node->prev;
    }

    void evict_tail() {
        if (!tail_) return;
        Node* victim = tail_;
        unlink(victim);
        index_.erase(victim->key);
        delete victim;
        --size_;
    }

    std::size_t capacity_;
    std::size_t size_ = 0;
    Node* head_ = nullptr;
    Node* tail_ = nullptr;
    std::unordered_map<K, Node*> index_;
};

// ============================================================================
// Open-addressing hash map (robin hood probing)
// ============================================================================
//
// Linear-probing hash map with the robin hood trick: when inserting, if the
// probe distance of the candidate slot's existing occupant is less than the
// would-be inserter's, swap them and continue probing with the displaced
// pair. This bounds variance in probe distance and keeps lookups fast even
// at high load factors.
//
// Tombstones are avoided by backshifting on erase. Capacity is always a
// power of two so the modulo reduces to a bit-mask.

template <typename K, typename V, typename Hash = std::hash<K>>
class HashMap {
public:
    explicit HashMap(std::size_t initial_capacity = 16,
                     double max_load_factor = 0.75)
        : max_load_(max_load_factor) {
        if (max_load_factor <= 0.0 || max_load_factor >= 1.0) {
            throw std::invalid_argument("max_load_factor must be in (0, 1)");
        }
        cap_ = round_up_pow2(std::max<std::size_t>(initial_capacity, 4));
        table_.assign(cap_, Slot{});
    }

    void insert(K key, V value) {
        if (static_cast<double>(size_ + 1) >= cap_ * max_load_) {
            rehash(cap_ * 2);
        }
        insert_unchecked(std::move(key), std::move(value));
    }

    bool find(const K& key, V& out) const {
        const std::size_t mask = cap_ - 1;
        std::size_t i = Hash{}(key) & mask;
        std::size_t dist = 0;
        while (true) {
            const Slot& s = table_[i];
            if (!s.occupied) return false;
            if (s.psl < dist)  return false;       // robin hood: would have
                                                   //   swapped earlier
            if (s.key == key) {
                out = s.value;
                return true;
            }
            i = (i + 1) & mask;
            ++dist;
        }
    }

    bool erase(const K& key) {
        const std::size_t mask = cap_ - 1;
        std::size_t i = Hash{}(key) & mask;
        std::size_t dist = 0;
        while (true) {
            Slot& s = table_[i];
            if (!s.occupied)  return false;
            if (s.psl < dist) return false;
            if (s.key == key) {
                backshift(i);
                --size_;
                return true;
            }
            i = (i + 1) & mask;
            ++dist;
        }
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return cap_; }

private:
    struct Slot {
        K   key{};
        V   value{};
        std::uint32_t psl = 0;
        bool occupied     = false;
    };

    static std::size_t round_up_pow2(std::size_t n) {
        std::size_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    void insert_unchecked(K key, V value) {
        const std::size_t mask = cap_ - 1;
        Slot incoming{std::move(key), std::move(value), 0, true};
        std::size_t i = Hash{}(incoming.key) & mask;
        while (true) {
            Slot& s = table_[i];
            if (!s.occupied) {
                s = std::move(incoming);
                ++size_;
                return;
            }
            if (s.psl < incoming.psl) {
                std::swap(s, incoming);
            }
            i = (i + 1) & mask;
            ++incoming.psl;
        }
    }

    void backshift(std::size_t hole) {
        const std::size_t mask = cap_ - 1;
        std::size_t next = (hole + 1) & mask;
        while (table_[next].occupied && table_[next].psl > 0) {
            table_[hole] = std::move(table_[next]);
            table_[hole].psl -= 1;
            table_[next].occupied = false;
            hole = next;
            next = (next + 1) & mask;
        }
        table_[hole].occupied = false;
    }

    void rehash(std::size_t new_cap) {
        std::vector<Slot> old = std::move(table_);
        cap_ = new_cap;
        size_ = 0;
        table_.assign(cap_, Slot{});
        for (auto& s : old) {
            if (s.occupied) {
                insert_unchecked(std::move(s.key), std::move(s.value));
            }
        }
    }

    std::vector<Slot> table_;
    std::size_t       cap_       = 0;
    std::size_t       size_      = 0;
    double            max_load_  = 0.75;
};

// ============================================================================
// Dijkstra shortest-path on an adjacency-list graph
// ============================================================================
//
// Standard binary-heap variant. Edge weights must be non-negative; for
// graphs with negative edges use Bellman-Ford instead. Returns a pair
// (dist, prev) so callers can both read distances and reconstruct paths.

struct Edge {
    std::uint32_t to;
    double        weight;
};

class Graph {
public:
    explicit Graph(std::size_t n) : adj_(n) {}

    void add_edge(std::uint32_t u, std::uint32_t v, double w) {
        if (w < 0.0) {
            throw std::invalid_argument("Dijkstra requires non-negative weights");
        }
        adj_[u].push_back({v, w});
    }

    std::size_t num_nodes() const noexcept { return adj_.size(); }
    const std::vector<Edge>& neighbors(std::uint32_t u) const { return adj_[u]; }

private:
    std::vector<std::vector<Edge>> adj_;
};

struct DijkstraResult {
    std::vector<double>        dist;
    std::vector<std::uint32_t> prev;
};

DijkstraResult dijkstra(const Graph& g, std::uint32_t source) {
    const auto INF = std::numeric_limits<double>::infinity();
    const std::size_t n = g.num_nodes();
    if (source >= n) {
        throw std::out_of_range("source out of range");
    }

    DijkstraResult out;
    out.dist.assign(n, INF);
    out.prev.assign(n, std::numeric_limits<std::uint32_t>::max());
    out.dist[source] = 0.0;

    using HeapEntry = std::pair<double, std::uint32_t>;
    std::priority_queue<HeapEntry,
                        std::vector<HeapEntry>,
                        std::greater<HeapEntry>> pq;
    pq.emplace(0.0, source);

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > out.dist[u]) {
            continue;                              // stale entry; skip
        }
        for (const Edge& e : g.neighbors(u)) {
            const double nd = d + e.weight;
            if (nd < out.dist[e.to]) {
                out.dist[e.to] = nd;
                out.prev[e.to] = u;
                pq.emplace(nd, e.to);
            }
        }
    }
    return out;
}

std::vector<std::uint32_t> reconstruct_path(const DijkstraResult& r,
                                            std::uint32_t target) {
    std::vector<std::uint32_t> path;
    const auto NONE = std::numeric_limits<std::uint32_t>::max();
    if (r.prev[target] == NONE && r.dist[target] != 0.0) {
        return path;                               // unreachable
    }
    for (std::uint32_t cur = target; cur != NONE; cur = r.prev[cur]) {
        path.push_back(cur);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

}  // namespace seed

// ============================================================================
// Smoke tests
// ============================================================================

int main() {
    using namespace seed;

    // ---- LRU cache ---------------------------------------------------------
    {
        LRUCache<std::string, int> c(2);
        c.put("a", 1);
        c.put("b", 2);
        int v = -1;
        assert(c.get("a", v) && v == 1);           // 'a' is now most recent
        c.put("c", 3);                             // evicts 'b'
        assert(!c.get("b", v));
        assert(c.get("a", v) && v == 1);
        assert(c.get("c", v) && v == 3);
        assert(c.size() == 2 && c.capacity() == 2);
    }

    // ---- Hash map ----------------------------------------------------------
    {
        HashMap<std::string, int> m;
        for (int i = 0; i < 1000; ++i) {
            m.insert("k" + std::to_string(i), i);
        }
        assert(m.size() == 1000);
        int v = -1;
        assert(m.find("k0", v)   && v == 0);
        assert(m.find("k500", v) && v == 500);
        assert(m.find("k999", v) && v == 999);
        assert(!m.find("k1000", v));
        assert(m.erase("k500"));
        assert(!m.find("k500", v));
        assert(m.size() == 999);
    }

    // ---- Dijkstra ----------------------------------------------------------
    {
        Graph g(6);
        g.add_edge(0, 1, 7.0);
        g.add_edge(0, 2, 9.0);
        g.add_edge(0, 5, 14.0);
        g.add_edge(1, 2, 10.0);
        g.add_edge(1, 3, 15.0);
        g.add_edge(2, 3, 11.0);
        g.add_edge(2, 5, 2.0);
        g.add_edge(3, 4, 6.0);
        g.add_edge(5, 4, 9.0);
        const auto r = dijkstra(g, 0);
        assert(r.dist[0] == 0.0);
        assert(r.dist[1] == 7.0);
        assert(r.dist[2] == 9.0);
        assert(r.dist[5] == 11.0);
        assert(r.dist[4] == 20.0);                 // 0 -> 2 -> 5 -> 4
        const auto path = reconstruct_path(r, 4);
        assert((path == std::vector<std::uint32_t>{0, 2, 5, 4}));
    }

    std::cout << "seed_corpus: all smoke tests passed\n";
    return 0;
}
