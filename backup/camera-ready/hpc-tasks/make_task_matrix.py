#!/usr/bin/env python3
"""Create TSV task matrices for one-GPU ABSA additional comparison baseline jobs."""

import argparse
import csv
from pathlib import Path


LANGUAGES = ("slovenian", "serbian")
VARIANTS = ("unmasked", "masked")
RUN_INDICES = (0, 1, 2)
TASK_SETS = {
    "core": ("longformer", "mdeberta", "mt5"),
    "all": ("longformer", "mdeberta", "mt5", "slavic_specific"),
    "longformer": ("longformer",),
    "mdeberta": ("mdeberta",),
    "mt5": ("mt5",),
    "slavic": ("slavic_specific",),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-set",
        default="all",
        choices=sorted(TASK_SETS),
        help="Which approaches to include.",
    )
    parser.add_argument(
        "--output",
        default="hpc-tasks/tasks_all.tsv",
        help="Output TSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    task_id = 0
    for approach in TASK_SETS[args.task_set]:
        for language in LANGUAGES:
            for variant in VARIANTS:
                for run_index in RUN_INDICES:
                    rows.append(
                        {
                            "task_id": task_id,
                            "approach": approach,
                            "language": language,
                            "variant": variant,
                            "run_index": run_index,
                        }
                    )
                    task_id += 1

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "approach", "language", "variant", "run_index"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote %s tasks to %s" % (len(rows), output))


if __name__ == "__main__":
    main()
