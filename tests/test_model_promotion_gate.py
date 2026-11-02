"""Committed regression coverage for scripts/model_promotion_gate.py.

The gate automates the owner's hold decision: a retrained model publishes only
when it beats the same run's naive previous-year baseline on dollar MAE.
Anything missing, non-finite, non-positive or incomparable is NOT a pass.
"""
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATE = os.path.join(REPO, "scripts", "model_promotion_gate.py")


def run_gate(meta_obj=None, raw_text=None):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        path = f.name
        if raw_text is not None:
            f.write(raw_text)
        else:
            json.dump(meta_obj, f)
    try:
        env = dict(os.environ)
        out_file = path + ".ghout"
        env["GITHUB_OUTPUT"] = out_file
        proc = subprocess.run(
            [sys.executable, GATE, "--meta", path],
            capture_output=True, text=True, env=env, timeout=60,
        )
        gh = open(out_file).read() if os.path.exists(out_file) else ""
        return proc, gh
    finally:
        for suffix in ("", ".ghout"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass


def good_meta(model_mae=50_000_000.0, base_mae=58_653_000.0):
    return {
        "metrics": {"test_dollar_scale_mae": model_mae},
        "naive_previous_year_baseline": {"test_dollar_scale_mae": base_mae},
    }


def test_promote_when_model_beats_baseline():
    proc, gh = run_gate(good_meta())
    assert proc.returncode == 0
    assert "PROMOTE" in proc.stdout
    assert "promote=true" in gh


def test_withhold_when_model_loses():
    proc, gh = run_gate(good_meta(model_mae=98_971_744.77))
    assert "WITHHOLD" in proc.stdout
    assert "promote=false" in gh


def test_withhold_on_exact_tie():
    _, gh = run_gate(good_meta(model_mae=58_653_000.0))
    assert "promote=false" in gh


def test_withhold_when_metrics_block_missing():
    _, gh = run_gate({"naive_previous_year_baseline": {"test_dollar_scale_mae": 1.0}})
    assert "promote=false" in gh


def test_withhold_when_baseline_block_missing():
    _, gh = run_gate({"metrics": {"test_dollar_scale_mae": 1.0}})
    assert "promote=false" in gh


def test_withhold_on_zero_model_mae():
    _, gh = run_gate(good_meta(model_mae=0.0))
    assert "promote=false" in gh


def test_withhold_on_negative_baseline():
    _, gh = run_gate(good_meta(base_mae=-5.0))
    assert "promote=false" in gh


def test_withhold_on_nan():
    _, gh = run_gate(raw_text='{"metrics": {"test_dollar_scale_mae": NaN}, '
                              '"naive_previous_year_baseline": {"test_dollar_scale_mae": 100}}')
    assert "promote=false" in gh


def test_withhold_on_infinity():
    _, gh = run_gate(raw_text='{"metrics": {"test_dollar_scale_mae": Infinity}, '
                              '"naive_previous_year_baseline": {"test_dollar_scale_mae": 100}}')
    assert "promote=false" in gh


def test_withhold_on_string_metrics():
    _, gh = run_gate(good_meta(model_mae="50000000"))
    assert "promote=false" in gh


def test_withhold_on_bool_metrics():
    _, gh = run_gate(good_meta(model_mae=True))
    assert "promote=false" in gh


def test_withhold_when_meta_is_not_an_object():
    _, gh = run_gate(raw_text='[1, 2, 3]')
    assert "promote=false" in gh


def test_withhold_when_meta_unreadable():
    env = dict(os.environ)
    import tempfile
    out = tempfile.NamedTemporaryFile(delete=False).name
    env["GITHUB_OUTPUT"] = out
    proc = subprocess.run(
        [sys.executable, GATE, "--meta", "/nonexistent/meta.json"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0  # fail-closed is a decision, not a crash
    assert "WITHHOLD" in proc.stdout
    assert "promote=false" in open(out).read()
    os.unlink(out)
