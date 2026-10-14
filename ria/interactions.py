"""Interaction indices over removal games.

A *game* maps a set of removed players to a score: v(S) = s(x with S removed).
All interaction terms here are local (every other player stays present) and are
discrete mixed differences, i.e. Harsanyi-style terms anchored at the full input:

    Gamma(T) = sum_{S subset of T} (-1)^{|S|} v(S)

|T| = 1 gives the removal effect, |T| = 2 the pairwise interaction, and
|T| = 3 the relational (region, region, phrase) term this project is about.

Local pairwise terms absorb higher-order structure: with b present, Gamma({a, p})
equals the a-p dividend plus the a-b-p dividend. To split a phrase's support into
object-wise and relational parts we therefore use Harsanyi dividends within the
triple U = {a, b, p} (players outside U stay present):

    d(T) = sum_{S subset of T} (-1)^{|T| - |S|} f(S),   f(S) = v(U \\ S)

For |T| = 3 the dividend equals the local Gamma(U).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations
from typing import Callable, Dict, FrozenSet, Iterable, Sequence

Subset = FrozenSet[str]
ValueFn = Callable[[Sequence[Subset]], Dict[Subset, float]]


def powerset(items: Iterable[str]) -> list[Subset]:
    items = list(items)
    return [frozenset(c) for c in chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))]


def memoize(value_fn: ValueFn) -> ValueFn:
    """Cache a game's values; evaluate the full powerset once, then any triple is free."""
    cache: Dict[Subset, float] = {}

    def wrapped(subsets: Sequence[Subset]) -> Dict[Subset, float]:
        missing = [s for s in dict.fromkeys(subsets) if s not in cache]
        if missing:
            cache.update(value_fn(missing))
        return {s: cache[s] for s in subsets}

    return wrapped


def mixed_difference(values: Dict[Subset, float], players: Iterable[str]) -> float:
    """Discrete mixed derivative of the removal game over `players`."""
    return sum((-1) ** len(s) * values[s] for s in powerset(players))


def dividend(values: Dict[Subset, float], universe: Iterable[str], coalition: Iterable[str]) -> float:
    """Harsanyi dividend of `coalition` within `universe` (outside players stay present)."""
    universe, coalition = frozenset(universe), frozenset(coalition)
    return sum((-1) ** (len(coalition) - len(s)) * values[universe - s] for s in powerset(coalition))


@dataclass
class TripleResult:
    a: str
    b: str
    p: str
    score: float
    effect: Dict[str, float]          # local first-order removal effects
    pair_local: Dict[str, float]      # local second-order terms (include the triple)
    pair: Dict[str, float]            # pairwise dividends within the triple, keyed "a|p" etc.
    triple: float                     # third-order relational term (dividend = local)
    rho: float                        # relational share of the phrase's support

    def as_row(self) -> dict:
        return {"a": self.a, "b": self.b, "p": self.p, "score": self.score,
                **{f"d[{k}]": v for k, v in self.effect.items()},
                **{f"G2[{k}]": v for k, v in self.pair.items()},
                "G3": self.triple, "rho": self.rho}


def _rho(g3: float, d_ap: float, d_bp: float) -> float:
    denom = abs(d_ap) + abs(d_bp) + abs(g3)
    return abs(g3) / denom if denom > 0 else 0.0


def relational_triple(value_fn: ValueFn, a: str, b: str, p: str) -> TripleResult:
    """All terms of the (a, b, p) decomposition from the 8 removal configurations."""
    u = (a, b, p)
    v = value_fn(powerset(u))
    effect = {x: mixed_difference(v, [x]) for x in u}
    pairs = ((a, p), (b, p), (a, b))
    pair_local = {f"{x}|{y}": mixed_difference(v, [x, y]) for x, y in pairs}
    pair = {f"{x}|{y}": dividend(v, u, [x, y]) for x, y in pairs}
    g3 = mixed_difference(v, u)
    rho = _rho(g3, pair[f"{a}|{p}"], pair[f"{b}|{p}"])
    return TripleResult(a, b, p, v[frozenset()], effect, pair_local, pair, g3, rho)


def pair_terms(value_fn: ValueFn, a: str, b: str) -> Dict[str, float]:
    """Removal effects of a and b and their interaction, e.g. on a contrastive margin game."""
    v = value_fn(powerset([a, b]))
    return {"score": v[frozenset()], f"d[{a}]": mixed_difference(v, [a]), f"d[{b}]": mixed_difference(v, [b]),
            "G2": mixed_difference(v, [a, b])}


# ---- noise floor ------------------------------------------------------------------
# The model is deterministic, so the uncertainty in each term comes from arbitrary
# method choices: how a region is removed (operator) and what replaces the phrase
# (substitute). Each choice gives one estimate; a term counts as signal only when
# its mean is k standard deviations from zero across those estimates.

RHO_TERMS = ("a|p", "b|p", "G3")


@dataclass
class RobustTriple:
    a: str
    b: str
    p: str
    labels: list[str]                 # one label per method configuration
    estimates: list[TripleResult]
    mean: Dict[str, float]            # over configurations, keys RHO_TERMS
    sd: Dict[str, float]
    significant: Dict[str, bool]
    rho: float | None                 # None = undefined (no term above the floor)
    rho_sd: float                     # spread of per-configuration rho
    k: float

    @property
    def relational(self) -> bool:
        """True when the third-order term itself clears the floor."""
        return self.significant["G3"]

    def as_row(self) -> dict:
        row = {"a": self.a, "b": self.b, "p": self.p, "n_configs": len(self.labels), "k": self.k,
               "rho": self.rho, "rho_sd": self.rho_sd, "relational": self.relational}
        for t in RHO_TERMS:
            row.update({f"{t}_mean": self.mean[t], f"{t}_sd": self.sd[t], f"{t}_sig": self.significant[t]})
        return row


def _terms(r: TripleResult) -> Dict[str, float]:
    return {"a|p": r.pair[f"{r.a}|{r.p}"], "b|p": r.pair[f"{r.b}|{r.p}"], "G3": r.triple}


def robust_triple(value_fns: Dict[str, ValueFn], a: str, b: str, p: str,
                  k: float = 2.0, min_abs: float = 0.0) -> RobustTriple:
    """Evaluate the triple under several method configurations and apply the noise floor.

    A term is significant when |mean| > k * sd and |mean| > min_abs. With a single
    configuration sd is 0, so only `min_abs` guards against noise. rho is computed
    from the mean terms and is None unless at least one term is significant.
    """
    import numpy as np

    labels = list(value_fns)
    est = [relational_triple(value_fns[lab], a, b, p) for lab in labels]
    vals = {t: np.array([_terms(r)[t] for r in est]) for t in RHO_TERMS}
    mean = {t: float(v.mean()) for t, v in vals.items()}
    sd = {t: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for t, v in vals.items()}
    sig = {t: abs(mean[t]) > k * sd[t] and abs(mean[t]) > min_abs for t in RHO_TERMS}
    rho = _rho(mean["G3"], mean["a|p"], mean["b|p"]) if any(sig.values()) else None
    rhos = np.array([r.rho for r in est])
    rho_sd = float(rhos.std(ddof=1)) if len(rhos) > 1 else 0.0
    return RobustTriple(a, b, p, labels, est, mean, sd, sig, rho, rho_sd, k)
