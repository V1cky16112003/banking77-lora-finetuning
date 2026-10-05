# MiniJev Phase 3: Training data (execution plan)

Parent: `minijev-plan.md` Phase 3. Date: 2026-10-06.

## Goal
Training data shaped like `/v1/systemone` requests, with a target probability
vector per question, and evaluation sets that keep the zero-shot Banking77 claim
honest.

## Source review (why the list changed)
The first list (CLINC150, MNLI, BoolQ, SST-2, AG News, ChaosNLI) was checked
against Kev's training data (`jaredpalmer/kev`, `kev/data.py`):
- **Every source had a fixed label set.** A pointer head can do well on those
  by memorising label meanings without reading the options. Added
  multiple-choice QA (ARC, OpenBookQA, CommonsenseQA), whose option texts
  differ per question. Kev trains on the same three.
- **No ordered data for Score.** SST-2 → SST-5, and added Yelp (1–5 stars).
- **No Jev-shaped requests** (structured state, several questions). Added a
  rule-based policy generator: labels computed by code, so exact and free.
  This replaces the optional paid teacher set.
- **CLINC's banking and credit_cards domains** are near-copies of Banking77
  intents. Dropped, plus any intent named exactly like a Banking77 label
  (`exchange_rate`) and out-of-scope: 32 intents in all.
- **ChaosNLI** is built from MNLI/SNLI dev items and exists only as an
  unofficial CC BY-NC mirror. Moved to Phase 5 as a calibration eval.

## Files
| File | What | Imports |
|---|---|---|
| `src/minijev/data/records.py` | record format, choice/noul/score builders, validation, label report | pydantic only |
| `src/minijev/data/policies.py` | 3 domains (orders, loans, tickets) × 6 question templates | pure Python |
| `src/minijev/data/sources.py` | converters (rows in, records out), text cleaning, pinned Hub revisions | `datasets` only in `load_rows` |
| `src/minijev/data/build.py` | CLI: build, leakage checks, report, gate | — |

Record: `{"id", "source", "split", "request": <systemone request>, "targets": {key: [p per option]}}`.
Output `data/minijev/` (gitignored; rebuilt in ~1 minute from pinned revisions).

## Design choices
- **Options are always shuffled** for choice questions; Score levels never are,
  because their order is the scale. The report checks the gold position.
- **Option subsets:** CLINC uses a log-uniform 2–119 options from the remaining
  intents, twice per training message (two different subsets). AG News uses 2–4.
- **Derived questions** make multi-question records: e.g. "Is this article
  mainly about sports?" next to the topic choice, "Is "X" a correct answer?" next
  to a QA choice. Policies have 1–6 questions per record.
- **State form:** 25% of dataset states wrapped in JSON (`{"article": …}`,
  `{"premise", "hypothesis"}`); policies 50/50 JSON vs prose.
- **Policies:** no dates, durations as counts (the backbone can't subtract dates;
  Kev notes the same). Money values carry cents and thresholds are whole dollars,
  so no value sits on a boundary. Fields within one rule are distinct.
- **Held-out sets use fixed instructions**, so evaluation measures the model,
  not instruction paraphrase noise. Banking77 uses this repo's frozen test split
  (3,080), the same set as the fine-tuned classifier.

## Verification
- `tests/test_minijev_data.py`, 27 tests, no network (CI-safe):
  builders, validation, gold-position spread, policy rule semantics (band labels
  re-derived from the level text; decision rules in order; ties), converters on
  hand-made rows, CLINC exclusion, and a full `build()` on a fake loader with a
  Banking77 message and a train/eval duplicate planted (the test fails if the
  leakage checks are switched off).
- Read one real record per source by hand. That caught "a order", a rule
  testing the same field twice, AG News's broken HTML entities and Yelp's
  literal `\n`; all fixed and tested.

## Results (full build, seed 0)
| | train | dev | test | held-out |
|---|---|---|---|---|
| records | 154,707 | 9,320 | 9,348 | 7,080 |
| questions | **239,029** | 13,726 | 13,719 | 7,080 |

| Train source | records | questions | gold first (actual / chance) |
|---|---|---|---|
| policies | 15,000 | 52,289 | 0.311 / 0.311 |
| mnli | 30,000 | 39,096 | 0.334 / 0.333 |
| agnews | 25,000 | 37,555 | 0.364 / 0.361 |
| yelp | 25,000 | 30,956 | score only |
| clinc | 23,796 | 30,821 | 0.122 / 0.121 |
| commonsenseqa | 9,619 | 14,374 | 0.203 / 0.200 |
| sst5 | 8,543 | 11,969 | 0.343 / 0.333 |
| boolq | 9,427 | 9,427 | noul only (62% true) |
| openbookqa | 4,954 | 7,486 | 0.232 / 0.250 |
| arc | 3,368 | 5,056 | 0.243 / 0.250 |

Held out: Banking77 3,080 (77 options each), QNLI, PAWS, Emotion, TweetEval sentiment 1,000 each.

Leakage: 0 Banking77 texts found; 7 train records dropped as duplicates of
eval rows (repeats inside the source datasets); 0 excluded CLINC intents offered.

**Gate: PASS.** 239k ≥ 200k train questions; report written; gold position
within 0.02 of chance for every source with ≥ 1,000 choice questions; no leakage.

## Known limits, carried into Phase 4
- **Few high-cardinality questions:** only ~4% of training questions have more
  than 25 options (all CLINC), but Banking77 has 77. If zero-shot Banking77
  is weak, the first lever is a larger share of big CLINC option sets.
- **Long states:** Yelp and BoolQ texts can exceed the 384-token training
  state budget. Phase 4 must filter or truncate by real token count.
- **Natural label imbalance** is kept (BoolQ 62% true, MNLI 1/3 entailment):
  these are the true base rates, which a calibrated model should learn.
- Licences: fine for this learning project. AG News and ChaosNLI are
  non-commercial; check before any commercial use.
