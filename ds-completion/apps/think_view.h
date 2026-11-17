// How a reply is printed (README step 147): with --think the model reasons first, up to its
// </think> token, and then answers. --think-show says what to do with the reasoning:
//   all   printed like the answer, on stdout (the default: stdout is every generated token)
//   line  one line on stderr that moves: a token count and the reasoning's last characters;
//         erased when the answer starts, which goes to stdout alone. If stderr is not a
//         terminal: as none.
//   none  not shown; a note on stderr when it ends
// The tokens, the context and the record do not depend on it.
#pragma once

#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

struct ThinkView {
  enum Mode { All, Line, None } mode = All;
  int32_t think_end = -1;  // the </think> token
  bool thinking = false;   // inside the reasoning of this turn
  long n = 0;              // its tokens so far
  std::string tail;        // its last characters, line breaks as spaces
  bool drawn = false;
  std::chrono::steady_clock::time_point t0;

  static bool parse(const std::string& s, Mode& m) {
    if (s == "all") m = All;
    else if (s == "line") m = Line;
    else if (s == "none") m = None;
    else return false;
    return true;
  }

  // a turn begins; `think`: its reply starts with reasoning
  void begin(bool think) {
    if (mode == Line && !isatty(STDERR_FILENO)) mode = None;
    thinking = think && mode != All;
    n = 0;
    tail.clear();
    drawn = false;
    t0 = std::chrono::steady_clock::now();
  }

  // a generated token and its text
  void token(int32_t id, const std::string& piece) {
    if (!thinking) {
      fwrite(piece.data(), 1, piece.size(), stdout);
      fflush(stdout);
      return;
    }
    if (id == think_end) {
      finish("");
      return;
    }
    n++;
    if (mode != Line) return;
    for (char c : piece) tail += (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
    if (tail.size() > 4096) tail.erase(0, tail.size() - 2048);
    draw();
  }

  // the turn ends; if the reasoning did not, say so
  void end() {
    if (thinking) finish(" (not finished: the reply ended inside it)");
  }

 private:
  static int columns() {
    winsize w{};
    return ioctl(STDERR_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 20 ? w.ws_col : 80;
  }

  void erase() {
    if (drawn) fputs("\r\033[K", stderr);
    drawn = false;
  }

  void draw() {
    char head[48];
    const int hn = snprintf(head, sizeof head, "thinking, %ld tokens: ", n);
    // the last characters that fit: a character outside ASCII is counted as two columns (CJK is;
    // others then leave room), and the cut never falls inside a UTF-8 sequence
    int room = columns() - 1 - hn;
    size_t i = tail.size();
    while (i > 0) {
      size_t j = i - 1;
      while (j > 0 && ((unsigned char)tail[j] & 0xC0) == 0x80) j--;
      const int w = (unsigned char)tail[j] < 0x80 ? 1 : 2;
      if (w > room) break;
      room -= w;
      i = j;
    }
    fprintf(stderr, "\r\033[K\033[2m%s%s\033[0m", head, tail.c_str() + i);
    fflush(stderr);
    drawn = true;
  }

  void finish(const char* how) {
    erase();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "# reasoning: %ld tokens in %.1f s%s\n", n, s, how);
    fflush(stderr);
    thinking = false;
  }
};
