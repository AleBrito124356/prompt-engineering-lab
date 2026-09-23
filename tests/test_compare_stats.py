"""Tests for position-bias control, statistics and reports in ``compare``."""

import json
import math

import pytest

from promptlab.backends import ScriptedClient
from promptlab.cli import main
from promptlab.compare import (
    JUDGE_SYSTEM,
    NO_DIFFERENCE,
    ComparisonResult,
    PairResult,
    judge_both_orders,
    load_report,
    min_decisive_for_significance,
    render_markdown,
    render_table,
    report_dict,
    results_from_report,
    run_comparison,
    sign_test_p,
    tally,
    wilson_interval,
    write_report,
)

INPUTS = ["q{}".format(i) for i in range(1, 8)]


def _is_judge(messages):
    return messages[0]["content"] == JUDGE_SYSTEM


def _responses(messages):
    user = messages[-1]["content"]
    first = user.split("--- Response 1 ---\n", 1)[1].split("\n\n--- Response 2 ---", 1)[0]
    second = user.split("--- Response 2 ---\n", 1)[1].split("\n\n--- End of responses ---", 1)[0]
    return first, second


def position_biased_judge():
    """Always answers 'Winner: 1' -- pure position bias, no judgement at all."""
    return ScriptedClient(rules=[(_is_judge, "Both are fine, the first reads better.\nWinner: 1")],
                          default="the exact same answer")


def marker_judge(marker="B-MARKER"):
    """Prefers whichever response contains ``marker``, wherever it is shown."""

    def judge(messages, params):
        first, second = _responses(messages)
        if marker in first and marker not in second:
            return "Response 1 has the marker.\nWinner: 1"
        if marker in second and marker not in first:
            return "Response 2 has the marker.\nWinner: 2"
        return "Equivalent.\nWinner: tie"

    return judge


def candidates(messages, params):
    return "answer with B-MARKER" if messages[0]["content"] == "SB" else "plain answer"


# -- the audit scenario ----------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_position_biased_judge_yields_no_winner(seed):
    """Regression: identical answers + an always-'1' judge used to crown a
    winner that flipped with the seed (3-4, 4-3, 2-5)."""
    results = run_comparison("SA", "SB", INPUTS, client=position_biased_judge(), seed=seed)
    counts = tally(results, "v1", "v2")
    assert all(r.winner == "tie" for r in results)
    assert counts["position_bias_rate"] == 1.0
    assert counts["inconsistent"] == len(INPUTS)
    assert counts["verdict"] == NO_DIFFERENCE and not counts["significant"]
    assert "orders disagree: A with A first, B with B first (position bias)" in results[0].reason
    assert "winner: none" in render_table(results, "v1", "v2")


def test_single_order_mode_still_shows_the_old_failure():
    """--single-order keeps the legacy behaviour; the stats refuse to call it."""
    splits = set()
    for seed in (0, 1, 2):
        results = run_comparison("SA", "SB", INPUTS, client=position_biased_judge(), seed=seed, both_orders=False)
        counts = tally(results, "v1", "v2")
        splits.add((counts["v1"], counts["v2"]))
        assert counts["verdict"] == NO_DIFFERENCE  # none of 3-4 / 4-3 / 2-5 is significant
        assert counts["pairs_both_orders"] == 0
    assert len(splits) > 1  # the raw split flips with the seed


def test_consistent_preference_wins_once_n_reaches_six():
    for n, expect_significant in ((5, False), (6, True)):
        client = ScriptedClient(rules=[(_is_judge, marker_judge())], default=candidates)
        results = run_comparison("SA", "SB", ["q"] * n, client=client)
        counts = tally(results, "A", "B")
        assert counts["B"] == n and counts["position_bias_rate"] == 0.0
        assert counts["significant"] is expect_significant
        assert counts["verdict"] == ("B" if expect_significant else NO_DIFFERENCE)
    assert counts["p_value"] == pytest.approx(0.03125)


def test_separate_judge_client_gets_every_judge_call():
    llm = ScriptedClient(default=candidates)
    judge = ScriptedClient(default=marker_judge())
    run_comparison("SA", "SB", ["q1", "q2"], client=llm, judge=judge)
    assert len(llm.calls) == 4 and len(judge.calls) == 4
    assert all(c["messages"][0]["content"] == JUDGE_SYSTEM for c in judge.calls)
    assert all(c["params"]["temperature"] == 0.0 for c in judge.calls)


def test_judge_both_orders_unparseable_is_a_flagged_tie():
    judge = ScriptedClient(["Winner: 1", "I cannot decide between these."])
    pair = judge_both_orders(judge, "q", "a", "b")
    assert pair.winner == "tie" and pair.consistent is False and pair.unparsed
    assert "could not be parsed (B first)" in pair.reason
    counts = tally([ComparisonResult("q", "a", "b", "tie", pairs=[pair])])
    assert counts["unparsed"] == 1 and counts["position_biased"] == 0


def test_criteria_reach_the_judge_prompt():
    client = ScriptedClient(rules=[(_is_judge, "Winner: tie")], default="x")
    run_comparison("SA", "SB", ["q"], client=client, criteria="brevity above all")
    assert "Criteria: brevity above all" in client.calls[-1]["messages"][-1]["content"]


def test_repeats_take_the_majority_per_input():
    verdicts = iter(["Winner: 1", "Winner: 2",   # repeat 1: A both orders -> A
                     "Winner: 2", "Winner: 1",   # repeat 2: B both orders -> B
                     "Winner: 1", "Winner: 2"])  # repeat 3: A -> A
    client = ScriptedClient(rules=[(_is_judge, lambda m, p: next(verdicts))], default="x")
    results = run_comparison("SA", "SB", ["q"], client=client, repeats=3)
    assert len(results) == 1 and len(results[0].pairs) == 3
    assert [p.winner for p in results[0].pairs] == ["A", "B", "A"]
    assert results[0].winner == "A"
    assert tally(results)["total"] == 1  # the sign test counts inputs, not samples
    assert "A2 B1 T0" in render_table(results)


def test_run_comparison_rejects_bad_arguments():
    with pytest.raises(ValueError, match="no inputs"):
        run_comparison("SA", "SB", [], client=ScriptedClient())
    with pytest.raises(ValueError, match="repeats"):
        run_comparison("SA", "SB", ["q"], client=ScriptedClient(), repeats=0)


def test_on_progress_callback():
    seen = []
    client = ScriptedClient(rules=[(_is_judge, "Winner: tie")], default="x")
    run_comparison("SA", "SB", ["a", "b"], client=client, on_progress=lambda i, n, r: seen.append((i, n)))
    assert seen == [(1, 2), (2, 2)]


# -- statistics ------------------------------------------------------------- #
@pytest.mark.parametrize(
    "a,b,expected",
    [
        (6, 0, 0.03125),                                   # 2 * 1/64
        (0, 6, 0.03125),
        (5, 0, 0.0625),                                    # a sweep of 5 is not enough
        (4, 3, 1.0),
        (7, 1, 2 * (1 + 8) / 256),                         # 0.0703125
        (10, 2, 2 * (1 + 12 + 66) / 4096),                 # 0.03857...
        (0, 0, 1.0),
        (1, 1, 1.0),
    ],
)
def test_sign_test_against_hand_computed_binomial(a, b, expected):
    assert sign_test_p(a, b) == pytest.approx(expected)


def test_sign_test_matches_binomial_definition():
    for n in range(1, 15):
        for k in range(n + 1):
            lo = min(k, n - k)
            expected = min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)
            assert sign_test_p(k, n - k) == pytest.approx(expected)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(6, 6)
    assert lo == pytest.approx(0.6097, abs=1e-4) and hi == 1.0
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    narrow = wilson_interval(50, 100, alpha=0.2)
    wide = wilson_interval(50, 100, alpha=0.01)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


def test_min_decisive_for_significance():
    assert min_decisive_for_significance(0.05) == 6
    assert min_decisive_for_significance(0.01) == 8


def test_tally_keeps_legacy_keys_and_adds_stats():
    results = [ComparisonResult("i", "a", "b", w) for w in ("A", "A", "B", "tie")]
    counts = tally(results, "v1", "v2")
    assert (counts["v1"], counts["v2"], counts["ties"], counts["total"], counts["overall"]) == (2, 1, 1, 4, "v1")
    assert counts["decisive"] == 3 and counts["win_rate_a"] == pytest.approx(2 / 3)
    assert counts["p_value"] == 1.0 and counts["verdict"] == NO_DIFFERENCE
    assert counts["position_bias_rate"] is None  # no two-order pairs recorded
    lo_a, hi_a = counts["ci_a"]
    lo_b, hi_b = counts["ci_b"]
    assert lo_a == pytest.approx(1 - hi_b) and hi_a == pytest.approx(1 - lo_b)


def test_render_table_reports_significant_winner():
    results = [ComparisonResult("i", "a", "b", "B", "better") for _ in range(8)]
    table = render_table(results, "v1", "v2")
    assert "->  winner: v2 (p = 0.008 < 0.05)" in table
    assert "v2 win rate (ties excluded): 100%" in table


# -- reports ---------------------------------------------------------------- #
def _sample_results():
    client = ScriptedClient(rules=[(_is_judge, marker_judge())], default=candidates)
    return run_comparison("SA", "SB", ["How do I | pipe?", "second"], client=client)


def test_json_report_round_trip(tmp_path):
    results = _sample_results()
    path = write_report(tmp_path / "out" / "report.json", results, "v1", "v2", config={"judge_model": "j"})
    data, loaded = load_report(path)
    assert data["format"] == "promptlab.compare/1"
    assert data["summary"]["wins_b"] == 2 and data["summary"]["verdict"] == NO_DIFFERENCE
    assert data["config"] == {"judge_model": "j"}
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in results]
    pair = data["results"][0]["pairs"][0]
    assert pair["response_a"] == "plain answer" and pair["response_b"] == "answer with B-MARKER"
    assert [j["order"] for j in pair["judgements"]] == ["AB", "BA"]
    assert tally(loaded, "v1", "v2") == tally(results, "v1", "v2")


def test_markdown_report_contains_both_responses_and_rationales(tmp_path):
    results = _sample_results()
    path = write_report(tmp_path / "report.md", results, "v1", "v2", config={"criteria": "c"})
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# A/B comparison: v1 vs v2")
    assert "**Verdict: no significant difference**" in text
    assert "plain answer" in text and "answer with B-MARKER" in text
    assert "Judge, A shown first: **B**" in text and "Judge, B shown first: **B**" in text
    assert "Response 2 has the marker." in text
    assert "How do I \\| pipe?" in text  # table cells are escaped
    assert "| criteria | c |" in text


def test_markdown_fences_survive_code_blocks_in_answers():
    pair = PairResult("```python\nx = 1\n```", "b", "A", [], True, False, "r")
    md = render_markdown([ComparisonResult("q", pair.response_a, "b", "A", "r", pairs=[pair])])
    assert "````text\n```python" in md


def test_report_errors(tmp_path):
    with pytest.raises(ValueError, match=".json or .md"):
        write_report(tmp_path / "r.txt", [], "a", "b")
    with pytest.raises(ValueError, match="not a promptlab compare report"):
        results_from_report({"format": "other"})
    assert report_dict([], "a", "b")["summary"]["verdict"] == NO_DIFFERENCE


# -- CLI -------------------------------------------------------------------- #
@pytest.fixture
def repo(monkeypatch):
    from pathlib import Path

    import promptlab.client as client_mod

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("PROMPTLAB_BACKEND", raising=False)
    monkeypatch.setattr(client_mod, "_load_dotenv", lambda: None)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])


def test_cli_compare_offline_writes_reports(repo, tmp_path, capsys):
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    code = main([
        "compare", "coding-assistant@v1", "coding-assistant@v2", "--var", "language=Python",
        "--inputs", "examples/coding-questions.txt", "--offline", "--out", str(md), "--out", str(js),
        "--criteria", "correctness", "--judge-model", "judge-x",
    ])
    out = capsys.readouterr()
    assert code == 0
    assert "judge consistency: 4/4 pairs agreed" in out.out
    assert "need at least 6" in out.out
    assert "report written" in out.err
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["config"]["judge_model"] == "judge-x" and data["config"]["criteria"] == "correctness"
    assert data["config"]["backend"] == "mock"
    assert len(data["results"]) == 4
    assert "[offline mock response" in md.read_text(encoding="utf-8")


def test_cli_compare_options(repo, capsys):
    base = ["compare", "sql-expert@v1", "sql-expert@v2", "--var", "dialect=SQLite",
            "--inputs", "examples/coding-questions.txt", "--offline"]
    assert main(base + ["--single-order", "--seed", "3"]) == 0
    assert "not measured (--single-order)" in capsys.readouterr().out
    assert main(base + ["--repeats", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("A0 B2 T0") == 4  # two judged samples per input, majority per input
    assert "8/8 pairs agreed" in out and "(4 inputs, 4 decisive)" in out
    assert main(["compare", "sql-expert", "sql-expert", "--var", "dialect=x",
                 "--inputs", "examples/coding-questions.txt", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "sql-expert@v2 (A)" in out and "ties: 4" in out  # A/A test: no difference


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--out", "report.txt"], "--out must end in .json or .md"),
        (["--repeats", "0"], "--repeats must be at least 1"),
        (["--alpha", "1.5"], "--alpha must be between 0 and 1"),
    ],
)
def test_cli_compare_validation(repo, capsys, extra, message):
    code = main(["compare", "sql-expert@v1", "sql-expert@v2", "--var", "dialect=x",
                 "--inputs", "examples/coding-questions.txt", "--offline"] + extra)
    assert code == 2 and message in capsys.readouterr().err
