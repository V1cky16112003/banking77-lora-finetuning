# MiniJev — Architecture

A small, open replication of the *System One model* idea behind TypeSafe's Jev:
**unstructured state in, typed calibrated decisions out, in one forward pass.**

Status: design, not built. Date: 2026-10-03.

> **Honesty note.** TypeSafe has not published Jev's architecture ("no paper,
> not yet" — Almeida, Latent Space, 2026-09-21). This design is reconstructed
> from public clues (below) plus the OpenJev-RLCD paper (arXiv 2609.38850), which
> explicitly claims nothing about Jev's internals. Treat every "Jev does X" here
> as inference, not fact.

## 1. Evidence → design decision

| Public clue | Source | Decision in MiniJev |
|---|---|---|
| "I wouldn't pre-train with $1B"; "Frankensteining" | Interview ~1:49 | Start from an open decoder (Qwen family); no pre-training |
| Model can be coaxed into saying it is "Qwen" | Interview ~1:47 | Qwen3 base is the most plausible backbone |
| "Ingests the state once and evaluates every question in parallel"; limit = state + *longest single* question | docs.typesafe.ai/models | Shared-prefix KV cache + independent question branches |
| Max cardinality 255 | Launch post | Fixed bank of 255 option slots; 2-stage for larger sets |
| Output tokens free; no strings | Launch post | No `generate()`; read logits once per question |
| "A logit bias for each function" | Interview ~2:16 | Probability over options = masked softmax over slot logits |
| RLCD = calibrated, epistemically honest probabilities | Launch post | Proper-scoring-rule training (Brier / log score) |
| All training data synthetic | Interview ~22:00 | Synthetic (state, questions, reference-distribution) generator |
| One set of weights for all customers | Docs | Domain knowledge goes in `state`/question text, not weights |

## 2. Interface (the contract)

```python
decide(
    state: str | dict | list[str],          # serialized to text
    questions: list[Question],               # evaluated in parallel
) -> list[Decision]

Question = Choice(key, prompt, options: list[str])        # 2..255 options
         | Noul(key, statement)                           # P(true), 2 options
         | Score(key, rubric, levels: list[str])          # ordered options

Decision = {key, value: <one of options>, probs: dict[option, float], confidence}
```

Type safety is structural: `value` is always `argmax(probs)` over the declared
options, so an out-of-schema answer cannot be produced.

## 3. Model

```
             ┌──────────────── shared, computed once ────────────────┐
 input  ───► [SYS] <state tokens ...>                                  │ prefill → KV cache
             └────────────────────────────────────────────────────────┘
                 │                 │                    │
                 ▼                 ▼                    ▼
   branch 1: <Q1 text> <opts:A,B,C> [ANS]   branch 2: <Q2 ...> [ANS]   branch k ...
                 │                                      │
        hidden h_[ANS] ──► slot head W ∈ R^{d×255} ──► mask to |options| ──► softmax ──► probs
```

### 3.1 Backbone
- **Qwen3-1.7B** (dev) / Qwen3-4B (stretch). Apache-2.0, fp16-stable on T4
  (see `research_notes/`). LoRA on attention + MLP; base frozen.

### 3.2 Parallel sampler (shared prefix + branches)
- Prefill the state once → KV cache.
- Pack all questions into one sequence after the state, with a **block attention
  mask**: every question token attends to the state and to earlier tokens of its
  *own* question only. Position ids for each branch restart at `len(state)`, so
  each branch is positionally identical to "state + this question alone".
  This is exactly why the context budget is "state + longest question".
- Equivalent serving path: vLLM/SGLang prefix caching with one request per
  question sharing the cached state (simpler; use first).

### 3.3 Option encoding (how the model "sees" choices)
Each option is listed in the question as `[S_i] option text`, where
`[S_1]..[S_255]` are 255 new special tokens (the "slots"). The question ends in
`[ANS]`. The decision is read at `[ANS]`:

```
logits = h_ANS · E[S_1..S_n]^T        # tie the head to slot embeddings
probs  = softmax(logits / T)          # only over the n declared slots
```

Tying the read-out to slot embeddings lets the model bind "slot 7" to "the
option written next to [S_7]" via attention — options are *read*, not memorized,
so unseen label sets work at inference. Shuffle slot assignment during training
so no slot acquires meaning.

- **Noul** = 2 slots (`false`, `true`); `confidence = max(p)`.
- **Score** = ordered levels; also report expected level `Σ p_i · i`.
- **>255 options** (stage 2, like Jev): score options in chunks of ≤255 as
  independent Noul questions ("is X the answer?"), keep top-k, then one Choice.

### 3.4 Optional rationale (off by default)
OpenJev shows that with a proper-score objective the model collapses to *no
rationale* on non-reasoning tasks ("System-One collapse"). MiniJev v1 has no
rationale. A "ReasoningJev" mode (short rationale before `[ANS]`) is future work.

## 4. Training — RLCD-lite

Objective for a question with reference distribution `q` (one-hot or soft):

```
L = Brier(p, q) = ‖p − q‖²       (or log score: −Σ q_i log p_i)
```

Both are **strictly proper**: the unique optimum is `p = q`, so the model is
paid for honest probabilities, not for confidence. With no rationale, OpenJev
shows RL reduces to this supervised gradient, so v1 is plain proper-score
training on soft targets — the cheap, stable core of RLCD.

| Stage | What | Why |
|---|---|---|
| 0 | Add `[S_*]`, `[ANS]` tokens; LoRA init | Interface |
| 1 | Proper-score training on synthetic + real labeled tasks | Calibrated readout (the main gain per OpenJev §5.1) |
| 2 (opt.) | RLCD-RL with rationale, KL β=0.04, LOO baseline, c=0.3 | Only if reasoning-heavy tasks matter |
| 3 | Temperature fit on dev; record `T` (goal: T≈1) | Calibration check, not a crutch |

## 5. Data — synthetic decision tasks

Records: `{state, questions:[{type, prompt, options}], targets:[q_vector]}`.

Sources:
1. **Converted labeled datasets** → one-hot `q`: Banking77, CLINC150, MNLI,
   BoolQ, SST-2 … re-phrased as Choice/Noul over random option subsets.
2. **Soft-label datasets** → aleatoric `q`: ChaosNLI (100 annotators).
3. **Teacher-labeled synthetic states** → soft `q` from a frontier LLM asked for
   option probabilities (averaged over a few samples / two teachers, like
   TypeSafe's eval reference). States = generated tickets, logs, JSON records,
   game states; questions = route/classify/extract/verify.

Augmentations: shuffle options, vary cardinality 2–255, distractor options,
paraphrased prompts, JSON vs prose state. **Hold out Banking77 entirely** for
the zero-shot eval.

## 6. Evaluation

| Metric | Meaning |
|---|---|
| Accuracy / macro-F1 | Correctness (reuse `src/evaluate_core.py`) |
| Brier, NLL | Proper-score quality |
| ECE + reliability diagram | Calibration |
| AURC, Cov@ε | "How much can I automate at ≤ε error?" — the key Jev metric |
| Fitted temperature T | ≈1 means calibrated as trained |
| Latency p50/p95, questions/sec | On T4, k questions per state |

Baselines: current generate-and-parse LoRA classifier in this repo; SFT +
temperature scaling; zero-shot Qwen3 with logit read-out.

## 7. Known limits
- Real Jev is likely much larger and trained far longer; expect a capability gap.
- Slot binding over 255 options at 1.7B is unproven — biggest technical risk.
- Teacher soft labels import teacher biases (TypeSafe notes the same).
- English only; text only.
