#!/usr/bin/env python
"""Generate horizontal boxplots of best-submission MCC per attempt, grouped by model slug."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_attempt_scores(results_file: Path, task: str, exclude: list[str] | None = None
                        ) -> dict[str, list[float]]:
    """Read results.jsonl → {slug: [max_mcc_per_attempt]}."""
    # First pass: collect best MCC per (attempt_id, slug)
    best: dict[tuple[str, str], float] = {}
    for line in results_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task") != task:
            continue
        slug = r.get("slug", "")
        if exclude and slug in exclude:
            continue
        mcc = r.get("mcc")
        if mcc is None:
            continue
        key = (r["attempt_id"], slug)
        best[key] = max(best.get(key, -1.0), mcc)

    # Group by slug
    by_slug: dict[str, list[float]] = defaultdict(list)
    for (_, slug), mcc in best.items():
        by_slug[slug].append(mcc)
    return dict(by_slug)


FIRST_PARTY_PREFIXES = ("gpt-", "claude-", "o1-", "o3-", "o4-")
COLOR_FIRST_PARTY = "#4C72B0"
COLOR_OSS = "#55A868"


def _is_first_party(slug: str) -> bool:
    return any(slug.startswith(p) for p in FIRST_PARTY_PREFIXES)


def plot_boxplots(scores: dict[str, list[float]], task: str, output: Path):
    slugs = sorted(scores.keys(), key=lambda s: sorted(scores[s])[len(scores[s]) // 2])
    data = [scores[s] for s in slugs]
    labels = [f"{s} (n={len(scores[s])})" for s in slugs]
    colors = [COLOR_FIRST_PARTY if _is_first_party(s) else COLOR_OSS for s in slugs]

    fig, ax = plt.subplots(figsize=(10, max(3, len(slugs) * 0.45)))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="#333", markersize=4),
                    medianprops=dict(color="white", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    # Background shading per row
    for i, c in enumerate(colors):
        ax.axhspan(i + 0.5, i + 1.5, color=c, alpha=0.06)

    ax.set_yticklabels(labels)
    ax.set_xlabel("Best MCC per attempt")
    ax.set_title(f"Model scores — {task}")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.3)

    # Scatter individual points
    for i, (d, c) in enumerate(zip(data, colors)):
        y = [i + 1] * len(d)
        ax.scatter(d, y, color=c, alpha=0.4, s=15, zorder=3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def plot_dumbbell(scores_a: dict[str, list[float]], scores_b: dict[str, list[float]],
                  task_a: str, task_b: str, output: Path):
    """Dumbbell chart comparing mean MCC between two tasks for shared slugs."""
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        print(f"No shared slugs between '{task_a}' and '{task_b}'")
        return

    import statistics
    means_a = {s: statistics.mean(scores_a[s]) for s in shared}
    means_b = {s: statistics.mean(scores_b[s]) for s in shared}
    # Sort by gap (most degradation at bottom, best-performing on top)
    shared.sort(key=lambda s: means_b[s] - means_a[s])

    fig, ax = plt.subplots(figsize=(10, max(3, len(shared) * 0.5)))

    for i, slug in enumerate(shared):
        ma, mb = means_a[slug], means_b[slug]
        color = COLOR_FIRST_PARTY if _is_first_party(slug) else COLOR_OSS
        ax.axhspan(i - 0.4, i + 0.4, color=color, alpha=0.06)
        ax.plot([ma, mb], [i, i], color=color, linewidth=2, alpha=0.5, zorder=2)
        ax.scatter([ma], [i], color=color, s=60, zorder=3, marker="o", edgecolors="white", linewidths=0.5)
        ax.scatter([mb], [i], color=color, s=60, zorder=3, marker="s", edgecolors="white", linewidths=0.5)
        drop_pct = (mb - ma) / ma * 100 if ma != 0 else 0
        label_x = max(ma, mb) + 0.01
        ax.text(label_x, i, f"{drop_pct:+.0f}%", va="center", fontsize=8, color="#555")

    na = {s: len(scores_a[s]) for s in shared}
    nb = {s: len(scores_b[s]) for s in shared}
    labels = [f"{s} ({na[s]}/{nb[s]})" for s in shared]
    ax.set_yticks(range(len(shared)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean best MCC per attempt")
    short_a = task_a.replace("toml-1.0-cpp", "spec").replace("toml-1.1-cpp", "1.1-spec")
    short_b = task_b.replace("toml-1.0-cpp-nospec", "nospec").replace("toml-1.1-cpp-nospec", "1.1-nospec")
    ax.set_title(f"Spec vs no-spec — {short_a} vs {short_b}")
    ax.set_xlim(-0.05, 1.15)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666", markersize=8, label=task_a),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#666", markersize=8, label=task_b),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def load_attempt_times(results_file: Path, slug_prefix: str,
                       exclude: list[str] | None = None
                       ) -> dict[str, dict[str, list[float]]]:
    """Read results.jsonl → {task: {slug: [elapsed_seconds_per_attempt]}}."""
    seen: dict[tuple[str, str, str], float] = {}  # (attempt_id, slug, task) -> seconds
    for line in results_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        slug = r.get("slug", "")
        if not slug.startswith(slug_prefix):
            continue
        if exclude and slug in exclude:
            continue
        elapsed = r.get("attempt_elapsed_seconds")
        if elapsed is None:
            continue
        key = (r["attempt_id"], slug, r["task"])
        seen[key] = elapsed

    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_, slug, task), elapsed in seen.items():
        by_task[task][slug].append(elapsed)
    return dict(by_task)


def plot_timing(times_by_task: dict[str, dict[str, list[float]]],
                slug_prefix: str, output_dir: Path):
    """One image per task, showing attempt elapsed time per slug as horizontal boxplots."""
    for task, slugs_data in sorted(times_by_task.items()):
        slugs = sorted(slugs_data.keys(),
                       key=lambda s: sorted(slugs_data[s])[len(slugs_data[s]) // 2])
        data = [slugs_data[s] for s in slugs]
        labels = [f"{s} (n={len(slugs_data[s])})" for s in slugs]
        colors = [COLOR_FIRST_PARTY if _is_first_party(s) else COLOR_OSS for s in slugs]

        fig, ax = plt.subplots(figsize=(10, max(2, len(slugs) * 0.45)))
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5,
                        showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="#333", markersize=4),
                        medianprops=dict(color="white", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for i, c in enumerate(colors):
            ax.axhspan(i + 0.5, i + 1.5, color=c, alpha=0.06)
        for i, (d, c) in enumerate(zip(data, colors)):
            y = [i + 1] * len(d)
            ax.scatter(d, y, color=c, alpha=0.4, s=15, zorder=3)

        ax.set_yticklabels(labels)
        ax.set_xlabel("Attempt elapsed time (seconds)")
        ax.set_title(f"Attempt timing — {task} ({slug_prefix}*)")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        out = output_dir / f"timing-{slug_prefix}-{task}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot MCC boxplots per model slug")
    subparsers = parser.add_subparsers(dest="command")

    box = subparsers.add_parser("boxplot", help="Horizontal boxplots for a single task")
    box.add_argument("--task", default="toml-1.0-cpp", help="Task to plot (default: toml-1.0-cpp)")
    box.add_argument("--results", default=None, help="Path to results.jsonl")
    box.add_argument("--output", default=None, help="Output image path (default: plots/<task>.png)")
    box.add_argument("--exclude", nargs="*", default=[], help="Slugs to exclude")

    cmp = subparsers.add_parser("compare", help="Dumbbell chart comparing two tasks")
    cmp.add_argument("--task-a", default="toml-1.0-cpp", help="First task (default: toml-1.0-cpp)")
    cmp.add_argument("--task-b", default="toml-1.0-cpp-nospec", help="Second task (default: toml-1.0-cpp-nospec)")
    cmp.add_argument("--results", default=None, help="Path to results.jsonl")
    cmp.add_argument("--output", default=None, help="Output image path")
    cmp.add_argument("--exclude", nargs="*", default=[], help="Slugs to exclude")

    tim = subparsers.add_parser("timing", help="Attempt timing boxplots by slug prefix")
    tim.add_argument("--prefix", default="fireworks", help="Slug prefix to filter (default: fireworks)")
    tim.add_argument("--results", default=None, help="Path to results.jsonl")
    tim.add_argument("--output", default=None, help="Output image path")
    tim.add_argument("--exclude", nargs="*", default=[], help="Slugs to exclude")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    results_file = Path(args.results) if args.results else Path(__file__).parent / "results" / "results.jsonl"

    if args.command == "boxplot":
        output = Path(args.output) if args.output else Path(__file__).parent / "plots" / f"{args.task}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        scores = load_attempt_scores(results_file, args.task, args.exclude or None)
        if not scores:
            print(f"No data for task '{args.task}'")
            return
        plot_boxplots(scores, args.task, output)

    elif args.command == "compare":
        output = Path(args.output) if args.output else Path(__file__).parent / "plots" / f"{args.task_a}_vs_{args.task_b}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        scores_a = load_attempt_scores(results_file, args.task_a, args.exclude or None)
        scores_b = load_attempt_scores(results_file, args.task_b, args.exclude or None)
        if not scores_a or not scores_b:
            print(f"No data for one of the tasks")
            return
        plot_dumbbell(scores_a, scores_b, args.task_a, args.task_b, output)

    elif args.command == "timing":
        output_dir = Path(args.output) if args.output else Path(__file__).parent / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        times = load_attempt_times(results_file, args.prefix, args.exclude or None)
        if not times:
            print(f"No timing data for prefix '{args.prefix}'")
            return
        plot_timing(times, args.prefix, output_dir)


if __name__ == "__main__":
    main()
