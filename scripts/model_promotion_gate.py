"""Publish gate: a retrained forecast model is promoted only when it beats the
naive previous-year baseline on dollar MAE on the same training run's holdout.
Missing, non-finite, non-positive or otherwise incomparable metrics are not a
pass: the gate withholds. Reads the training meta json and writes
promote=true/false to GITHUB_OUTPUT (or stdout when run locally).
Never edits the meta file."""
import argparse
import json
import math
import os
import sys


def _finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) and value > 0


def decide(meta):
    if not isinstance(meta, dict):
        return False, "meta is not an object; unknown is not a pass"
    model_mae = (meta.get("metrics") or {}).get("test_dollar_scale_mae")
    base_mae = (meta.get("naive_previous_year_baseline") or {}).get("test_dollar_scale_mae")
    if not _finite_positive(model_mae) or not _finite_positive(base_mae):
        return False, (f"missing/non-finite/non-positive metrics "
                       f"(model_mae={model_mae!r}, baseline_mae={base_mae!r}); unknown is not a pass")
    if model_mae < base_mae:
        return True, f"model dollar MAE ${model_mae:,.0f} beats previous-year baseline ${base_mae:,.0f}"
    return False, f"model dollar MAE ${model_mae:,.0f} does not beat previous-year baseline ${base_mae:,.0f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    args = p.parse_args()
    try:
        with open(args.meta) as f:
            meta = json.load(f)
    except Exception as e:
        promote, reason = False, f"meta unreadable ({e}); unknown is not a pass"
    else:
        promote, reason = decide(meta)
    print(f"[PROMOTION GATE] {'PROMOTE' if promote else 'WITHHOLD'}: {reason}")
    line = f"promote={'true' if promote else 'false'}\n"
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(line)
    else:
        sys.stdout.write(line)


if __name__ == "__main__":
    main()
