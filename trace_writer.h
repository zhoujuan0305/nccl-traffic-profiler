// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace fine {
inline unsigned long long now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<unsigned long long>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

inline std::string escape(const char* text) {
  std::string out;
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text ? text : ""); *p; ++p) {
    if (*p == '"' || *p == '\\') { out += '\\'; out += *p; }
    else if (*p < 32 || *p >= 127) {
      char hex[7]; std::snprintf(hex, sizeof(hex), "\\u%04x", *p); out += hex;
    } else out += *p;
  }
  return out;
}

// One buffered file per process avoids a syscall on every proxy callback.
// The file is experiment-specific; callers must never reuse a run directory.
class Writer {
 public:
  std::atomic<unsigned long long> failures{0};
  std::atomic<unsigned long long> allocations_failed{0};
  std::atomic<unsigned long long> anomalies{0};
  int rank = -1;

  bool open() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) return true;
    const char* root = std::getenv("FINE_TRACE_DIR");
    const char* r = std::getenv("RANK");
    if (!root || !r) return false;
    rank = std::atoi(r);
    std::string dir = std::string(root) + "/rank-" + r;
    mkdir(dir.c_str(), 0755);
    const auto path = dir + "/nccl-events.jsonl";
    file_ = std::fopen(path.c_str(), "a");
    if (!file_) return false;
    std::setvbuf(file_, nullptr, _IOFBF, 1024 * 1024);
    return true;
  }

  void line(const char* format, ...) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_) { ++failures; return; }
    va_list args;
    va_start(args, format);
    if (std::vfprintf(file_, format, args) < 0) ++failures;
    va_end(args);
  }

  void flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_ && std::fflush(file_) != 0) ++failures;
  }

  ~Writer() { if (file_) std::fclose(file_); }

 private:
  std::mutex mutex_;
  FILE* file_ = nullptr;
};
}  // namespace fine
