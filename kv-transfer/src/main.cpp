#include "ref.h"
#include "target.h"
#include "handoff.h"
#include "compare.h"
#include "decay.h"
#include "batch.h"

#include <cstdio>
#include <cstring>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  ref      Run reference model (prompt + generation), save logits\n"
        "  target   Run target model on existing tokens, compute KL vs ref inline\n"
        "  handoff  Process prompt with ref model, transfer KV, generate with target, compute KL inline\n"
        "  compare  Summarize a per-token stats file (mean/p95/p99 KL + top-1)\n"
        "  decay    Analyze KL decay across generation position from stats files\n"
        "  batch    Process all prompts with ref+target models loaded together\n",
        prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char * cmd = argv[1];

    if (strcmp(cmd, "ref") == 0) {
        return cmd_ref(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "target") == 0) {
        return cmd_target(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "handoff") == 0) {
        return cmd_handoff(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "compare") == 0) {
        return cmd_compare(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "decay") == 0) {
        return cmd_decay(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "batch") == 0) {
        return cmd_batch(argc - 1, argv + 1);
    }

    fprintf(stderr, "Unknown command: %s\n\n", cmd);
    print_usage(argv[0]);
    return 1;
}
