# Pointwise — Architecture

A small, open replication of the *System One model* idea behind TypeSafe's Jev:
**unstructured state in, typed calibrated decisions out, in one forward pass.**

Status: Phase 0 (logit read-out baseline) built and measured; Phases 1–7 are
design. Date: 2026-10-03, revised 2026-10-04 after comparison with Kev (§8).

> **Honesty note.** TypeSafe has not published Jev's architecture ("no paper,
> not yet" — Almeida, Latent Space, 2026-09-21). This design is reconstructed
> from public clues (below), the OpenJev-RLCD paper (arXiv 2609.38850, which
> explicitly claims nothing about Jev's internals) and Kev, an open Jev-like
> model family by Jared Palmer (§8). Treat every "Jev does X" here as
> inference, not fact.

## 1. Evidence → design decision

| Public clue | Source | Decision in Pointwise |
|---|---|---|
| "I wouldn't pre-train with $1B"; "Frankensteining" | Interview ~1:49 † | Start from an open decoder (Qwen family); no pre-training |
| Model can be coaxed into saying it is "Qwen" | Interview ~1:47 † | Qwen base is the most plausible backbone (Kev uses Qwen3 / Qwen3.5 bases too) |
| "Ingests the state once and evaluates every question in parallel"; limit = state + *longest single* question | docs.typesafe.ai/models | Shared-prefix KV cache + independent question branches |
| Context: **64k per request = 32k state + longest question** | docs.typesafe.ai/models | Train on short states first; serving limit is a config value, not an architecture change |
| Max cardinality 255 | Launch post | API cap of 255 options; 2-stage for larger sets. Not a fixed bank of slot tokens (§3.3) |
| Output tokens free; no strings | Launch post | No `generate()`; read hidden states once per question |
| "A logit bias for each function" | Interview ~2:16 † | Probability over options = softmax over per-option scores |
| RLCD = calibrated, epistemically honest probabilities | Launch post | Proper-scoring-rule training (log score / Brier). *The link to proper scoring rules is a third-party and OpenJev interpretation; TypeSafe has not confirmed it.* |
| All training data synthetic | Interview ~22:00 † | Synthetic (state, questions, reference-distribution) generator |
| One set of weights for all customers | Docs | Domain knowledge goes in `state`/question text, not weights |
| Eval reference = average of two frontier models' probabilities | Launch post / workflow evals | Teacher soft labels averaged over two teachers (§5) |

† Timestamps and quotes from the interview, and OpenJev section numbers and
hyperparameters cited below (§3.4, §4), were not captured in the 2026-10-02
study notes. Re-check them against the sources before relying on them.

## 2. Interface (the contract)

Match TypeSafe's public `POST /v1/systemone` shape (as reproduced by Kev's
`kev/api.py` against TypeSafe's reference `system-one-adapter` 0.2.1), not a
shape of our own, so a Pointwise server is a drop-in for a Jev client.

```json
request:  {"state": str | object | array,
           "model": "pointwise-latest",
           "questions": {
             "<id>": {"type": "choice", "instructions": ..., "criteria": {"<name>": "<description or null>", ...}},
             "<id>": {"type": "noul",   "instructions": ..., "criteria": {"true": ..., "false": ...}  /* optional */},
             "<id>": {"type": "score",  "instructions": ..., "criteria": ["<level 0>", "<level 1>", ...]}}}

response: {"answers": {
             "<id>": {"type": "choice", "choice": "<name>", "confidence": c, "probabilities": {"<name>": p, ...}},
             "<id>": {"type": "noul",   "noul": p_true},
             "<id>": {"type": "score",  "score": E[level], "confidence": c, "legend": {...}, "probabilities": {"0": p, ...}}}}
```

- Choice: 1..255 options. Noul: 2 options, `[false, true]`. Score: 1..255
  ordered levels.
- `state` objects/arrays are rendered to indented `key: value` text.
- Confidence follows TypeSafe's adapter, not `max(p)`:
  - choice: `(p_max − 1/K) / (1 − 1/K)` — 0 at uniform, 1 at certainty;
  - score: `max(0, 1 − E|level − mode| / D)`, `D` = mean absolute deviation
    of a uniform distribution over the levels.
- Probabilities rounded to 4 decimals (keeps `|Σp − 1| < 0.02` at 255 options).

Type safety is structural: the answer is always `argmax(probs)` over the
declared options, so an out-of-schema answer cannot be produced.

## 3. Model

```
             ┌──────────── shared, computed once ────────────┐
 input  ───► <state> <state tokens ...>                       │ prefill → KV cache
             └────────────────────────────────────────────────┘
                 │                               │
                 ▼                               ▼
   branch 1: <q> instr <opt> A </opt> <opt> B </opt> ... <decide>     branch k ...
                          │            │               │
                  h_</opt>A    h_</opt>B   ...   h_<decide>
                          └─────┬──────┘               │
                                ▼                      ▼
                     k = W_k h_opt_j            q = W_q h_decide
                                └──── z_j = k_j · q / √d_p ────► softmax over the question's options ──► probs
```

### 3.1 Backbone
- **Qwen3-1.7B** (dev) / Qwen3-4B (stretch), Apache-2.0, fp16-stable on T4
  (see `research_notes/`). Use the **Base** model, as Kev does: the
  instruct/chat tuning brings nothing to a read-out head. LoRA on attention +
  MLP; base frozen. The vocabulary output layer is not used.
- **Requires transformers ≥ 4.51** (Qwen3 support). The repo pins 4.46.3 for
  the Qwen2.5 classifier; bump it on the `jev` branch before Phase 2.
- Avoid hybrid backbones (Qwen3.5 Gated DeltaNet) on T4: recurrent layers
  cannot honour the block mask of §3.2, so every question has to run as its
  own row (Kev's "row form").

### 3.2 Parallel sampler (shared prefix + branches)
- Prefill the state once → KV cache.
- Pack all questions into one sequence after the state, with a **block attention
  mask**: every question token attends to the state and to earlier tokens of its
  *own* question only. Position ids for each branch restart at `len(state)`, so
  each branch is positionally identical to "state + this question alone".
  This is exactly why the context budget is "state + longest question".
- Kev's `branch_mask_batch` implements exactly this mask (additive, `finfo.min`
  not `-inf`, padded query rows keep their diagonal so no row is all-masked).
- Equivalent serving path: one causal row per question continuing a cached
  state. Exact by construction; needed for long requests, where the packed
  L×L mask grows with the number of questions. Cap rows per pass by a token
  budget, since each row carries a copy of the state cache.

### 3.3 Option encoding and read-out — pointer head over option spans
Delimiters reuse **existing, rarely used Qwen special tokens** (e.g.
`<|fim_prefix|>` = state, `<|fim_middle|>` = question, `<|box_start|>` /
`<|box_end|>` = option open/close, `<|fim_suffix|>` = decide), so no
embedding rows are added. Each option is written as `<opt> option text </opt>`.

The read-out is a small **pointer head** (two linear maps to `d_p = 256`):

```
q   = W_q · h[<decide>]                   # what this question is asking
k_j = W_k · h[</opt> of option j]         # what option j means, in context
z_j = (k_j · q) / sqrt(d_p)
p   = softmax(z / T)                      # over this question's options only
```

**Why not 255 slot tokens** (the original design: `[S_1]..[S_255]`, read-out
tied to their embeddings)? With slots the model must *learn* that `[S_7]`
means "whatever option is written next to `[S_7]`", which was this design's
biggest risk (§7). With a pointer head each option is represented by the
model's own contextual hidden state at the end of its text, so there is no
binding to learn: the score is attention-like matching between "what is being
asked" and "what this option says". No slot ever carries meaning, unseen label
sets work by construction, and 255 is only an API cap.

- **Unforgeable boundaries.** Caller text must never produce delimiter tokens.
  Rewrite `<|name|>` in user text (e.g. to `<¦name¦>`) before tokenising;
  test that a state or option containing `<|box_end|>` cannot add an option.
- **Option-order robustness.** Shuffle option order in training, add a
  symmetric-KL consistency term between two orderings (§4), and report an
  argmax-flip rate under shuffles. Optional *option isolation*: each option
  span attends only to state + instruction + itself and all spans share
  positions, which makes option representations permutation-invariant by
  construction (packed form only).
- **Noul** = 2 options (`no`, `yes`, with optional descriptions); report p(true).
- **Score** = ordered levels; report expected level `Σ p_i · i`.
- **>255 options** (stage 2, like Jev): score options in chunks of ≤255 as
  independent Noul questions ("is X the answer?"), keep top-k, then one Choice.
- Keep the pointer head and its softmax in fp32 even when the backbone is fp16.

### 3.4 Optional rationale (off by default)
OpenJev † reports that with a proper-score objective the model collapses to *no
rationale* on non-reasoning tasks ("System-One collapse"). Pointwise v1 has no
rationale. A "ReasoningJev" mode (short rationale before `<decide>`) is future
work.

## 4. Training — RLCD-lite

Objective for a question with reference distribution `q` (one-hot or soft):

```
L = −Σ_i q_i log p_i               (log score / soft cross-entropy; default)
  + λ_B ‖p − q‖²                   (optional Brier term)
  + λ_P · ½[KL(p_π‖p) + KL(p‖p_π)] (optional: same question, options permuted by π)
```

Log score and Brier are both **strictly proper**: the unique optimum is
`p = q`, so the model is paid for honest probabilities, not for confidence.
Log score is the default because it gives a stronger gradient on confidently
wrong answers; Kev trains this way, with Brier as an optional extra term.
OpenJev † reports that without a rationale RL reduces to this supervised
gradient, so v1 is plain proper-score training on soft targets.

| Stage | What | Why |
|---|---|---|
| 0 | Delimiter tokens, pointer head, LoRA init | Interface |
| 1 | Proper-score training on synthetic + real labeled tasks | Calibrated read-out |
| 2 (opt.) | RLCD-RL with rationale † | Only if reasoning-heavy tasks matter |
| 3 | Temperature fit on dev; record `T` | Calibration check |

**Expect T > 1.** Kev's adapter checkpoints fit T ≈ 2.2–2.4 (raw
out-of-domain ECE 0.12 → 0.05 after scaling; only the full fine-tune of its
27B model got T ≈ 1.3). One-hot real labels push towards overconfidence. So
the gate is on ECE *after* temperature scaling, and T is reported, not gated.
Store T with the checkpoint and apply it only at inference.

## 5. Data — decision tasks

Records: `{request, targets}`: a `/v1/systemone` request plus a probability
vector per question over its options (one-hot from labels; soft where a source
has annotator distributions). Full list and caps: `pointwise-phase3-plan.md`.

Sources:
1. **Fixed-label datasets** → Choice/Noul/Score over random option subsets:
   MNLI, BoolQ, AG News, SST-5 and Yelp (ordered → Score), CLINC150 without
   the banking and credit-card domains.
2. **Multiple-choice QA** (ARC, OpenBookQA, CommonsenseQA): option texts differ
   per question, so the model must *read* options rather than memorise label
   meanings. The skill zero-shot Banking77 depends on.
3. **Rule-based policy records**: generated orders, loan applications and
   support tickets as JSON or prose, with several questions each (thresholds,
   bands, extraction, multi-condition rules). Labels are computed by code, so
   they are exact and free. Kev trains on similar generated policies; this
   replaces the paid teacher set for v1.

Augmentations: shuffle unordered options, vary cardinality, paraphrased
instructions, JSON vs prose state, 1–8 questions per state.
**Hold out Banking77 entirely** for the zero-shot eval, and hold out whole
*sources* (QNLI, PAWS, Emotion, TweetEval; ChaosNLI in Phase 5) for an
out-of-domain split, as Kev does. Kev itself trains on Banking77, so a Kev
Banking77 score is not zero-shot.

Start with short training contexts (Kev: state ≤ 384, question branch ≤ 1024,
packed ≤ 2048 tokens), which fit a T4.

## 6. Evaluation

| Metric | Meaning |
|---|---|
| Accuracy / macro-F1 | Correctness (reuse `src/evaluate_core.py`) |
| Brier, NLL | Proper-score quality |
| ECE + reliability diagram | Calibration (raw and after T) |
| AURC, Cov@ε | "How much can I automate at ≤ε error?" — the key Jev metric |
| Fitted temperature T | How far raw training is from calibrated |
| Argmax-flip rate under option shuffles | Option-order robustness |
| Latency p50/p95, questions/sec | On T4, k questions per state |

Baselines: current generate-and-parse LoRA classifier in this repo; SFT +
temperature scaling; zero-shot Qwen3 with logit read-out; Kev-0.8B / Kev-4B
on the same held-out sets (open weights, same API).

## 7. Known limits
- Real Jev is likely much larger and trained far longer; expect a capability
  gap. Kev-0.8B is "noticeably weaker out of domain"; a 1.7B Pointwise should
  be expected to sit between Kev-0.8B and Kev-4B, not near Jev.
- Pointer-head read-out removes the slot-binding risk, but option-order bias
  remains and must be measured (§3.3).
- Read-out speed: the options are input tokens after the per-message state,
  so they cannot be prefix-cached across states. A Banking77 question with 77
  options costs ~400 input positions; the gain over generation is the removed
  decode loop, not fewer positions (see `pointwise-phase0c-plan.md`).
- Teacher soft labels import teacher biases (TypeSafe notes the same).
- English only; text only.

## 8. Reference replication: Kev

Kev (Jared Palmer, Apache-2.0; github.com/jaredpalmer/kev, demo at
huggingface.co/spaces/jaredpalmer/kev, weights `jaredpalmer/kev-{0.8b,4b,9b,27b}`)
is an open Jev-like family that reached a similar design independently.
Compared on 2026-10-04 from its `kev/model.py`, `kev/api.py`, `kev/train.py`.

| | Pointwise (this doc) | Kev |
|---|---|---|
| Backbone | Qwen3-1.7B Base, LoRA | Qwen3.5 Base (earlier: Qwen3), LoRA r=16 incl. DeltaNet projections; 27B full fine-tune |
| Parallel questions | Block mask, positions restart after state | Same; row form on hybrid backbones |
| Delimiters | Reused Qwen special tokens (adopted from Kev) | 5 reused Qwen special tokens |
| Read-out | Pointer head over option spans (adopted from Kev) | `PointerHead`, `d_p = 256`, fp32 |
| Loss | Log score + optional Brier + permutation KL | CE / soft CE + optional Brier, focal, anchor KL, permutation KL |
| Calibration | Fit T, gate on post-T ECE | T ≈ 2.2–2.4 stored in `head.pt` |
| API | TypeSafe `/v1/systemone` (adopted) | TypeSafe `/v1/systemone` |
| Hardware | T4 fp16, 1.7B | fp32 training, data-centre GPUs and Apple Silicon |

Kev's self-reported held-out results (community Decision Index, 14 datasets
none of its models trained on): Kev-27B 52.3 vs Jev 54.0; Kev-4B new-source
accuracy ~0.82, Brier ~0.27. Self-reported and not reproduced by us.
