"""The job the MiniJev Kaggle runner notebook executes.

The notebook (notebooks/kaggle_minijev_job.ipynb) only clones the `jev`
branch, installs dependencies, finds the adapter and calls this module, so the
job is chosen in git and a new job needs no notebook edits: change JOB below,
push, then Save & Run All on Kaggle.

Usage: python -m scripts.kaggle_job --adapter-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# Current job: Phase 2 decision-API benchmark (docs/designs/minijev-phase2-plan.md).
# It runs the base Qwen3 backbone, so the Banking77 adapter is not passed on.
JOB = ["scripts.bench_decide"]
JOB_USES_ADAPTER = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    command = [sys.executable, "-m", *JOB, "--output-dir", args.output_dir]
    if JOB_USES_ADAPTER:
        command += ["--adapter-dir", args.adapter_dir]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
