"""Checks of the interaction algebra on hand-built games (no model needed)."""

import math

import numpy as np

from ria import RegionPlayer, mixed_difference, powerset, relational_triple


def game_from(f):
    """f(present: set) -> score; returns a value_fn over removed subsets."""
    players = {"a", "b", "p"}
    return lambda subsets: {s: f(players - s) for s in subsets}


def test_purely_relational_score_is_all_third_order():
    # Score exists only when a, b and p are all present.
    r = relational_triple(game_from(lambda P: float({"a", "b", "p"} <= P)), "a", "b", "p")
    assert math.isclose(r.triple, 1.0)
    assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in r.pair.values())
    assert math.isclose(r.rho, 1.0)
    # Local pairwise terms absorb the triple, which is why rho uses dividends.
    assert math.isclose(r.pair_local["a|p"], 1.0)


def test_dividends_sum_to_score_gap():
    rng = np.random.default_rng(1)
    table = {s: rng.normal() for s in powerset("abp")}
    vals = {s: table[s] for s in powerset("abp")}
    from ria.interactions import dividend
    total = sum(dividend(vals, "abp", t) for t in powerset("abp") if t)
    assert math.isclose(total, vals[frozenset()] - vals[frozenset("abp")])


def test_object_wise_score_has_no_third_order_term():
    # Phrase supported by each object separately: s = [a & p] + [b & p].
    r = relational_triple(game_from(lambda P: float({"a", "p"} <= P) + float({"b", "p"} <= P)), "a", "b", "p")
    assert math.isclose(r.triple, 0.0, abs_tol=1e-12)
    assert math.isclose(r.pair["a|p"], 1.0) and math.isclose(r.pair["b|p"], 1.0)
    assert math.isclose(r.pair_local["a|p"], 1.0)
    assert math.isclose(r.rho, 0.0)


def test_additive_score_has_no_interactions():
    r = relational_triple(game_from(lambda P: 2.0 * ("a" in P) + 3.0 * ("b" in P) + 5.0 * ("p" in P)), "a", "b", "p")
    assert math.isclose(r.effect["a"], 2.0) and math.isclose(r.effect["p"], 5.0)
    assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in r.pair.values())
    assert math.isclose(r.triple, 0.0, abs_tol=1e-12)


def test_harsanyi_decomposition_sums_to_score():
    # Local mixed differences over all subsets reconstruct v(empty) - v(all removed).
    rng = np.random.default_rng(0)
    table = {s: rng.normal() for s in powerset("abp")}
    v = lambda subsets: {s: table[s] for s in subsets}
    vals = v(powerset("abp"))
    total = sum(mixed_difference(vals, t) * (-1) ** (len(t) + 1) for t in powerset("abp") if t)
    assert math.isclose(total, vals[frozenset()] - vals[frozenset("abp")])


def test_context_region_is_complement():
    a = RegionPlayer.from_box("a", (0, 0, 4, 4), 8)
    b = RegionPlayer.from_box("b", (4, 4, 8, 8), 8)
    c = RegionPlayer.complement("c", [a, b])
    assert c.mask.sum() == 64 - 32 and not (c.mask & a.mask).any()


# ---- noise floor (T03) --------------------------------------------------------------

from ria.interactions import robust_triple  # noqa: E402


def relational_game(scale, offset=0.0):
    """G3 = scale; the pair dividends are 0 apart from a shared offset on a&p."""
    return game_from(lambda P: scale * float({"a", "b", "p"} <= P) + offset * float({"a", "p"} <= P))


def test_consistent_terms_clear_the_floor():
    games = {f"cfg{i}": relational_game(1.0 + 0.01 * i) for i in range(4)}
    r = robust_triple(games, "a", "b", "p", k=2.0)
    assert r.relational and r.rho is not None and r.rho > 0.95


def test_sign_flipping_terms_make_rho_undefined():
    # Same small magnitude, sign depends on the method choice: pure noise.
    games = {f"cfg{i}": relational_game(0.01 * (-1) ** i, offset=0.01 * (-1) ** (i // 2)) for i in range(4)}
    r = robust_triple(games, "a", "b", "p", k=2.0)
    assert not any(r.significant.values())
    assert r.rho is None and not r.relational


def test_object_wise_signal_gives_defined_low_rho():
    games = {f"cfg{i}": game_from(lambda P, s=1.0 + 0.02 * i: s * (float({"a", "p"} <= P) + float({"b", "p"} <= P)))
             for i in range(3)}
    r = robust_triple(games, "a", "b", "p")
    assert r.significant["a|p"] and r.significant["b|p"] and not r.relational
    assert r.rho is not None and r.rho < 0.05


def test_single_configuration_relies_on_min_abs():
    games = {"only": relational_game(0.003)}
    assert robust_triple(games, "a", "b", "p", min_abs=0.0).rho is not None
    assert robust_triple(games, "a", "b", "p", min_abs=0.01).rho is None
