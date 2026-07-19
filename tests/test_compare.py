"""Tests for the A/B comparison scoring, judging, and table rendering."""

from promptlab.compare import (
    ComparisonResult,
    JUDGE_SYSTEM,
    judge_pair,
    parse_verdict,
    render_table,
    run_comparison,
    tally,
)


class _FixedRNG:
    """Deterministic stand-in for random with a fixed random() value."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


class _FakeClient:
    """Returns canned answers for A/B and a fixed judge verdict."""

    def __init__(self, system_a, system_b, verdict="1"):
        self.system_a = system_a
        self.system_b = system_b
        self.verdict = verdict

    def complete(self, user, *, system=None, **kwargs):
        if system == JUDGE_SYSTEM:
            return "Response {} is clearer.\nWinner: {}".format(self.verdict, self.verdict)
        if system == self.system_a:
            return "ANSWER-A for {}".format(user)
        if system == self.system_b:
            return "ANSWER-B for {}".format(user)
        return "unknown"


def test_parse_verdict_variants():
    assert parse_verdict("...\nWinner: 1") == "1"
    assert parse_verdict("blah\nWinner: 2\n") == "2"
    assert parse_verdict("Winner: tie") == "tie"
    assert parse_verdict("no verdict") == "tie"


def test_parse_verdict_uses_last_winner_line():
    assert parse_verdict("Winner: 1\nactually\nWinner: 2") == "2"


def test_judge_pair_maps_shuffled_order_back_to_a():
    # rng < 0.5 -> A shown first; judge picks "1" -> winner is A.
    client = _FakeClient("SA", "SB", verdict="1")
    winner, reason = judge_pair(client, "q", "respA", "respB", rng=_FixedRNG(0.1))
    assert winner == "A"
    assert reason


def test_judge_pair_maps_shuffled_order_back_to_b():
    # rng >= 0.5 -> B shown first as "Response 1"; judge picks "1" -> winner is B.
    client = _FakeClient("SA", "SB", verdict="1")
    winner, _ = judge_pair(client, "q", "respA", "respB", rng=_FixedRNG(0.9))
    assert winner == "B"


def test_judge_pair_tie():
    client = _FakeClient("SA", "SB", verdict="tie")
    winner, _ = judge_pair(client, "q", "a", "b", rng=_FixedRNG(0.1))
    assert winner == "tie"


def test_tally_counts_and_overall():
    results = [
        ComparisonResult("i1", "a", "b", "A"),
        ComparisonResult("i2", "a", "b", "A"),
        ComparisonResult("i3", "a", "b", "B"),
        ComparisonResult("i4", "a", "b", "tie"),
    ]
    counts = tally(results, "v1", "v2")
    assert counts["v1"] == 2
    assert counts["v2"] == 1
    assert counts["ties"] == 1
    assert counts["overall"] == "v1"


def test_render_table_contains_rows_and_summary():
    results = [ComparisonResult("input one", "a", "b", "A", "clearer answer")]
    table = render_table(results, "v1", "v2")
    assert "input one" in table
    assert "winner:" in table
    assert "v1" in table


def test_run_comparison_end_to_end_with_fake_client():
    client = _FakeClient("SYS-A", "SYS-B", verdict="1")
    results = run_comparison(
        "SYS-A", "SYS-B", ["q1", "q2"], client=client, seed=0
    )
    assert len(results) == 2
    assert all(r.winner in ("A", "B", "tie") for r in results)
    assert results[0].response_a.startswith("ANSWER-A")
    assert results[0].response_b.startswith("ANSWER-B")
