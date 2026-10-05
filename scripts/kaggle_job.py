"""The job the MiniJev Kaggle runner notebook executes.

The notebook (notebooks/kaggle_minijev_job.ipynb) only clones the `jev`
branch, installs dependencies, finds the adapter and calls this module, so the
job is chosen in git and a new job needs no notebook edits: change the job
below, push, then Save & Run All on Kaggle.

Current job: Phase 4 training (docs/designs/minijev-phase4-plan.md).
1. Build the training data from pinned sources into /tmp (not saved as output).
2. Train on every visible GPU with torchrun. Resumes from a previous session's
   checkpoint if that notebook version is attached as input.
3. If training fails in its first 20 minutes (almost always out of memory on the
   largest batch, which runs first), retry once with half the token budget.

Usage: python -m scripts.kaggle_job --adapter-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

DATA_DIR = "/tmp/minijev-data"
TOKEN_BUDGET = 4096
EARLY_FAILURE_MINUTES = 20
TRAIN_ARGS = [
    "--resume-search", "/kaggle/input",
    "--max-hours", "10.5",  # Kaggle stops sessions at 12h and then keeps no output
    "--planned-hours", "21",  # two sessions
]


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, sum(line.startswith("GPU") for line in out.splitlines()))
    except (OSError, subprocess.CalledProcessError):
        return 1


def run(command: list[str], env: dict) -> int:
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, env=env).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter-dir", required=True)  # unused: Phase 4 trains a new adapter
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # the notebook pins GPU 0; training uses all of them
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if run([sys.executable, "-m", "src.minijev.data.build", "--out", DATA_DIR], env):
        sys.exit("data build failed")

    gpus = gpu_count()
    budget = TOKEN_BUDGET
    for attempt in range(2):
        started = time.time()
        code = run(
            [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(gpus),
             "-m", "src.minijev.train", "--data-dir", DATA_DIR, "--output-dir", args.output_dir,
             "--token-budget", str(budget), *TRAIN_ARGS],
            env,
        )
        if code == 0:
            return
        minutes = (time.time() - started) / 60
        if attempt == 0 and minutes < EARLY_FAILURE_MINUTES:
            budget //= 2
            print(f"Training failed after {minutes:.1f} min; retrying with token budget {budget}", flush=True)
            continue
        sys.exit(f"training failed (exit {code})")


if __name__ == "__main__":
    main()
