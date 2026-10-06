"""The job the MiniJev Kaggle runner notebook executes.

The notebook (notebooks/kaggle_minijev_job.ipynb) only clones the `jev`
branch, installs dependencies, finds the adapter and calls this module, so the
job is chosen in git and a new job needs no notebook edits: change the job
below, push, then Save & Run All on Kaggle.

Current job: Phase 4 training (docs/designs/minijev-phase4-plan.md).
1. Build the training data from pinned sources into /tmp (not saved as output).
   Remove Kaggle's preinstalled torchao and check that LoRA can be applied
   (preflight), so an environment problem fails in seconds, not mid-launch.
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


PREFLIGHT = """
import torch, transformers
from peft import LoraConfig, get_peft_model
from src.minijev.train import LORA_TARGETS
config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=16)
model = get_peft_model(transformers.Qwen3Model(config), LoraConfig(task_type="FEATURE_EXTRACTION", r=4, target_modules=LORA_TARGETS))
model(input_ids=torch.tensor([[1, 2, 3]]))
print("preflight ok: LoRA applies and runs", flush=True)
"""


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
    # Kaggle's image ships torchao 0.10; peft 0.21 raises on any torchao below 0.16
    # while applying LoRA (2026-10-06 session 1 failed this way). MiniJev doesn't
    # quantise, and peft treats an absent torchao as fine.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], env)
    if run([sys.executable, "-c", PREFLIGHT], env):
        sys.exit("preflight failed: LoRA can't be applied in this environment")

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
