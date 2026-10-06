"""Pointwise training data (Phase 3, docs/designs/pointwise-phase3-plan.md).

records.py   question builders, augmentation, label report (pure Python)
policies.py  rule-based policy records with code-computed labels (pure Python)
sources.py   public dataset converters (rows in, records out) and loaders
build.py     CLI: build train/dev/test + held-out eval sets, check leakage, write the report
"""
