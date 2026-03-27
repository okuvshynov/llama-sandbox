#include "generate.h"
#include "collect.h"
#include "compare.h"
#include "compare_batch.h"
#include "diff.h"

#include <cstdio>
#include <cstring>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  generate   Run reference model, sample tokens, save logits\n"
        "  collect    Run quantized model on existing tokens, save logits\n"
        "  compare    Compute KL divergence and optimize sampling parameters\n"
        "  compare-batch  Multi-prompt analysis with per-prompt variance tracking\n"
        "  diff           Show top-N logits at positions where top-1 disagrees\n",
        prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char * cmd = argv[1];

    // shift argv so subcommand sees itself as argv[0]
    if (strcmp(cmd, "generate") == 0) {
        return cmd_generate(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "collect") == 0) {
        return cmd_collect(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "compare") == 0) {
        return cmd_compare(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "compare-batch") == 0) {
        return cmd_compare_batch(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "diff") == 0) {
        return cmd_diff(argc - 1, argv + 1);
    }

    fprintf(stderr, "Unknown command: %s\n\n", cmd);
    print_usage(argv[0]);
    return 1;
}
