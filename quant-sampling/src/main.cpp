#include "ref.h"
#include "target.h"
#include "compare.h"
#include "diff.h"

#include <cstdio>
#include <cstring>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  ref            Run reference model, sample tokens, save logits\n"
        "  target         Run target model on existing tokens, save logits\n"
        "  compare        Compute KL divergence and optimize sampling parameters\n"
        "  diff           Show top-N logits at positions where top-1 disagrees\n",
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
    if (strcmp(cmd, "compare") == 0) {
        return cmd_compare(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "diff") == 0) {
        return cmd_diff(argc - 1, argv + 1);
    }

    fprintf(stderr, "Unknown command: %s\n\n", cmd);
    print_usage(argv[0]);
    return 1;
}
