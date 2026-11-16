// What ds-completion --record writes about a run beyond its own variables (docs/record.md).
#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// The prompt's identity is its token ids, after the chat template: FNV-1a, 64 bits, over each
// id's four bytes, least significant first.
inline uint64_t fnv1a64_ids(const std::vector<int32_t>& ids) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (int32_t id : ids)
    for (int b = 0; b < 4; b++) h = (h ^ (((uint32_t)id >> (8 * b)) & 0xff)) * 0x100000001b3ull;
  return h;
}

inline std::string json_string(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o + "\"";
}

// The GPU core clock (sclk) setting of every AMD card, in card order, as JSON values: the level
// (0..8) when it is fixed (power_dpm_force_performance_level = manual: the starred line of
// pp_dpm_sclk), else the mode as a string ("auto": the driver changes the level with the load,
// so the level read at any one moment says nothing about the run; after a reboot every card is
// on auto). "unknown": a file could not be read. The level is set outside the program, as root
// (QUICKSTART.md); the record holds it as it was when the run started. `root`: /sys/class/drm, or
// a copy of it for a test.
inline std::vector<std::string> gpu_sclk(const std::string& root = "/sys/class/drm") {
  auto first_line = [](const std::string& path, std::string& out) {
    std::ifstream f(path);
    return (bool)std::getline(f, out);
  };
  std::vector<std::string> r;
  for (int c = 0; c < 16; c++) {
    const std::string dev = root + "/card" + std::to_string(c) + "/device/";
    std::string vendor, mode;
    if (!first_line(dev + "vendor", vendor) || vendor != "0x1002") continue;
    if (!first_line(dev + "power_dpm_force_performance_level", mode)) { r.push_back("\"unknown\""); continue; }
    if (mode != "manual") { r.push_back(json_string(mode)); continue; }
    std::ifstream f(dev + "pp_dpm_sclk");  // lines "6: 1608Mhz *"
    std::string line, level = "\"unknown\"";
    while (std::getline(f, line)) {
      const size_t colon = line.find(':');
      if (line.find('*') == std::string::npos || colon == std::string::npos || colon == 0) continue;
      if (line.find_first_not_of("0123456789") == colon) level = line.substr(0, colon);
    }
    r.push_back(level);
  }
  return r;
}

// Bytes this process has had fetched from storage so far (/proc/self/io, read_bytes): reads of
// file pages that were not in the page cache, by any of its threads, through the model's mmap or
// pread alike; a read served from the page cache adds nothing. -1: not available.
inline long long io_read_bytes() {
  std::ifstream f("/proc/self/io");
  std::string key;
  long long v;
  while (f >> key >> v)
    if (key == "read_bytes:") return v;
  return -1;
}

// a list of JSON values as a JSON array
inline std::string json_array(const std::vector<std::string>& v) {
  std::string o = "[";
  for (size_t i = 0; i < v.size(); i++) o += (i ? ", " : "") + v[i];
  return o + "]";
}
