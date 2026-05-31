"""Simplified psychological/neurocognitive agent model.

This module intentionally removes biological life-cycle variables from the
working branch: no age, sex, reproduction, gestation, fertility, mortality or
sexual selection. Agents are defined only by psychological/neurocognitive
traits, memory, reputation, wealth/status and interaction outcomes.

The purpose is exploratory: observe how peculiar profiles react to events and
to one another under repeated probabilistic interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


TRAIT_KEYS = (
    "attn_selective",
    "attn_flex",
    "hyperfocus",
    "impulsivity",
    "risk_aversion",
    "sociality",
    "language",
    "reasoning",
    "emotional_impulsivity",
    "resilience",
    "empathy",
    "dominance",
    "affect_reg",
    "aggression",
    "moral_prosocial",
    "moral_common_good",
    "moral_honesty",
    "moral_spite",
    "dark_narc",
    "dark_mach",
    "dark_psycho",
)

ALIASES = {
    "attnselective": "attn_selective",
    "attnflex": "attn_flex",
    "riskaversion": "risk_aversion",
    "emotionalimpulsivity": "emotional_impulsivity",
    "affectreg": "affect_reg",
    "moralprosocial": "moral_prosocial",
    "moralcommongood": "moral_common_good",
    "moralhonesty": "moral_honesty",
    "moralspite": "moral_spite",
    "darknarc": "dark_narc",
    "darkmach": "dark_mach",
    "darkpsycho": "dark_psycho",
}


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def gini(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(arr.mean())
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.abs(np.subtract.outer(arr, arr)).sum() / (2 * arr.size**2 * mean))


def _normalise_key(key: str) -> str:
    raw = str(key).strip()
    return ALIASES.get(raw, raw)


def _clean_traits(raw: Dict[str, float] | None) -> Dict[str, float]:
    traits = {key: 0.5 for key in TRAIT_KEYS}
    if not raw:
        return traits
    for key, value in raw.items():
        norm = _normalise_key(key)
        if norm in TRAIT_KEYS:
            traits[norm] = clamp01(float(value))
    return traits


def load_profiles(path: str = "profiles.json") -> Dict[str, Dict[str, object]]:
    if not os.path.exists(path):
        return {
            "neutral": {
                "name": "Neutral",
                "description": "Fallback neutral profile",
                "traits": _clean_traits({}),
                "biological_bias": {},
                "spectrum_ranges": {},
            }
        }

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    profiles: Dict[str, Dict[str, object]] = {}
    for item in data.get("profiles", []):
        pid = str(item.get("id", "")).strip()
        if not pid:
            continue
        profiles[pid] = {
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "traits": _clean_traits(item.get("traits") or item.get("latents") or {}),
            "biological_bias": item.get("biological_bias", {}) or {},
            "spectrum_ranges": item.get("spectrum_ranges", {}) or {},
        }
    return profiles


def sample_profile_traits(
    rng: np.random.Generator,
    profile: Dict[str, object],
    jitter: float = 0.05,
    spectrum_level: int | None = None,
) -> Dict[str, float]:
    traits = dict(profile.get("traits", {}) or {})
    for key, default in _clean_traits({}).items():
        traits.setdefault(key, default)

    # Add moral/emotional ranges from profile metadata without treating them as biological facts.
    for key, bounds in (profile.get("biological_bias", {}) or {}).items():
        norm = _normalise_key(key)
        if norm not in TRAIT_KEYS:
            continue
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            lo, hi = float(bounds[0]), float(bounds[1])
            traits[norm] = float(rng.uniform(min(lo, hi), max(lo, hi)))

    # Optional intensity scaling for spectrum ranges.
    if spectrum_level is not None:
        level = max(1, min(3, int(spectrum_level)))
        scale = level / 3.0
        for key, bounds in (profile.get("spectrum_ranges", {}) or {}).items():
            norm = _normalise_key(key)
            if norm not in TRAIT_KEYS:
                continue
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lo, hi = float(bounds[0]), float(bounds[1])
                traits[norm] = float(rng.uniform(lo, lo + (hi - lo) * scale))

    return {key: clamp01(float(value) + rng.normal(0, jitter)) for key, value in traits.items()}


@dataclass
class PsychAgent:
    unique_id: int
    profile_id: str
    traits: Dict[str, float]
    wealth: float = 1.0
    reputation_coop: float = 0.5
    reputation_fear: float = 0.0
    status: float = 0.5
    last_action: str = "none"
    alive: bool = True
    memory: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def dark_core(self) -> float:
        dark = 0.34 * self.traits["dark_narc"] + 0.33 * self.traits["dark_mach"] + 0.33 * self.traits["dark_psycho"]
        return clamp01(
            0.45 * dark
            + 0.25 * (1.0 - self.traits["empathy"])
            + 0.20 * self.traits["dominance"]
            + 0.15 * self.traits["impulsivity"]
            - 0.20 * self.traits["affect_reg"]
            - 0.15 * self.traits["moral_prosocial"]
        )

    def memory_for(self, other_id: int) -> Dict[str, float]:
        return self.memory.setdefault(other_id, {"trust": 0.5, "fear": 0.0, "interactions": 0.0})

    def decide_action(self, other: "PsychAgent", rng: np.random.Generator) -> str:
        mem = self.memory_for(other.unique_id)
        trust = mem["trust"]
        fear = mem["fear"]

        empathy = self.traits["empathy"]
        prosocial = self.traits["moral_prosocial"]
        honesty = self.traits["moral_honesty"]
        common = self.traits["moral_common_good"]
        dominance = self.traits["dominance"]
        aggression = self.traits["aggression"]
        impulsivity = self.traits["impulsivity"]
        reasoning = self.traits["reasoning"]
        risk_aversion = self.traits["risk_aversion"]
        regulation = self.traits["affect_reg"]
        dark = self.dark_core()

        avoid = clamp01(0.35 * risk_aversion + 0.35 * fear + 0.20 * reasoning * (1.0 - trust))
        violence = clamp01(
            0.30 * aggression
            + 0.25 * dominance
            + 0.20 * impulsivity
            + 0.25 * dark
            - 0.25 * empathy
            - 0.20 * regulation
            - 0.15 * risk_aversion
        )
        cooperate = clamp01(
            0.30 * empathy
            + 0.25 * prosocial
            + 0.20 * honesty
            + 0.15 * common
            + 0.10 * trust
            + 0.10 * self.reputation_coop
            - 0.15 * dark
        )
        defect = clamp01(
            0.25 * dark
            + 0.20 * dominance
            + 0.20 * impulsivity
            + 0.15 * (1.0 - honesty)
            + 0.10 * (1.0 - empathy)
            - 0.10 * fear
        )
        support = clamp01(0.25 * empathy + 0.25 * common + 0.20 * self.traits["language"] + 0.15 * self.traits["sociality"])

        weights = np.array([avoid, violence, cooperate, defect, support], dtype=float)
        weights = np.maximum(weights, 0.001)
        weights = weights / weights.sum()
        return str(rng.choice(["avoid", "violence", "cooperate", "defect", "support"], p=weights))

    def plasticity(self, context: str, rng: np.random.Generator, rate: float = 0.025) -> None:
        resilience = self.traits["resilience"]
        regulation = self.traits["affect_reg"]
        sensitivity = clamp01(1.0 - 0.5 * resilience - 0.3 * regulation)
        delta = abs(float(rng.normal(rate, rate * 0.5))) * (0.5 + sensitivity)

        if context == "good_interaction":
            for key in ("empathy", "moral_prosocial", "moral_common_good", "moral_honesty"):
                self.traits[key] = clamp01(self.traits[key] + delta)
            self.traits["moral_spite"] = clamp01(self.traits["moral_spite"] - delta)
        elif context == "betrayal":
            self.traits["risk_aversion"] = clamp01(self.traits["risk_aversion"] + delta)
            self.traits["moral_spite"] = clamp01(self.traits["moral_spite"] + delta)
            self.traits["empathy"] = clamp01(self.traits["empathy"] - delta * 0.6)
        elif context == "violence_success":
            self.traits["aggression"] = clamp01(self.traits["aggression"] + delta)
            self.traits["dominance"] = clamp01(self.traits["dominance"] + delta * 0.7)
            self.traits["empathy"] = clamp01(self.traits["empathy"] - delta * 0.5)
        elif context == "violence_received":
            self.traits["risk_aversion"] = clamp01(self.traits["risk_aversion"] + delta)
            self.traits["resilience"] = clamp01(self.traits["resilience"] - delta * 0.4)


class PsychNeuroSociety:
    POP_SCALES = {"tiny": 10, "small": 50, "tribe": 150, "city": 500}

    def __init__(
        self,
        seed: int | None = 42,
        profiles_path: str = "profiles.json",
        population_scale: str = "small",
        profile1: str | None = None,
        profile2: str | None = None,
        profile3: str | None = None,
        weight1: float = 0.6,
        weight2: float = 0.3,
        weight3: float = 0.1,
        steps_per_agent: int = 1,
        jitter: float = 0.05,
        spectrum_level: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.profiles = load_profiles(profiles_path)
        self.population_scale = population_scale
        self.n_agents = int(self.POP_SCALES.get(population_scale, self.POP_SCALES["small"]))
        self.steps_per_agent = max(1, int(steps_per_agent))
        self.step_count = 0
        self.history: List[Dict[str, object]] = []

        selected = [profile1, profile2, profile3]
        available = list(self.profiles.keys())
        if not any(selected):
            selected = available[:3]
        selected = [(p or "").strip() for p in selected if (p or "").strip()]
        selected = [p for p in selected if p in self.profiles] or available[:1]

        weights = np.array([weight1, weight2, weight3], dtype=float)[: len(selected)]
        if weights.size == 0 or weights.sum() <= 0:
            weights = np.ones(len(selected), dtype=float)
        weights = weights / weights.sum()

        self.agents: List[PsychAgent] = []
        for uid in range(self.n_agents):
            pid = str(self.rng.choice(selected, p=weights))
            traits = sample_profile_traits(self.rng, self.profiles[pid], jitter=jitter, spectrum_level=spectrum_level)
            self.agents.append(PsychAgent(unique_id=uid, profile_id=pid, traits=traits))

    def alive_agents(self) -> List[PsychAgent]:
        return [a for a in self.agents if a.alive]

    def _choose_pair(self) -> Tuple[PsychAgent, PsychAgent] | None:
        alive = self.alive_agents()
        if len(alive) < 2:
            return None
        i, j = self.rng.choice(len(alive), 2, replace=False)
        return alive[int(i)], alive[int(j)]

    def _apply_interaction(self, a: PsychAgent, b: PsychAgent, action_a: str, action_b: str) -> Dict[str, float]:
        event = {"cooperation": 0.0, "violence": 0.0, "defection": 0.0, "support": 0.0, "avoidance": 0.0}

        def update_memory(actor: PsychAgent, other: PsychAgent, trust_delta: float, fear_delta: float) -> None:
            mem = actor.memory_for(other.unique_id)
            mem["trust"] = clamp01(mem["trust"] + trust_delta)
            mem["fear"] = clamp01(mem["fear"] + fear_delta)
            mem["interactions"] += 1.0

        if "avoid" in (action_a, action_b):
            event["avoidance"] = 1.0
            a.last_action, b.last_action = action_a, action_b
            return event

        if "violence" in (action_a, action_b):
            attacker, victim = (a, b) if action_a == "violence" else (b, a)
            event["violence"] = 1.0
            power_att = attacker.traits["dominance"] + attacker.traits["aggression"] + 0.4 * attacker.dark_core()
            power_vic = victim.traits["resilience"] + victim.traits["reasoning"] + victim.traits["risk_aversion"]
            success_p = clamp01(power_att / (power_att + power_vic + 1e-9))
            if self.rng.random() < success_p:
                transfer = max(0.0, victim.wealth * 0.15)
                victim.wealth -= transfer
                attacker.wealth += transfer
                attacker.reputation_fear = clamp01(attacker.reputation_fear + 0.08)
                victim.reputation_coop = clamp01(victim.reputation_coop - 0.03)
                attacker.plasticity("violence_success", self.rng)
                victim.plasticity("violence_received", self.rng)
                update_memory(victim, attacker, trust_delta=-0.20, fear_delta=0.25)
                update_memory(attacker, victim, trust_delta=-0.05, fear_delta=0.02)
            else:
                attacker.wealth -= 0.05
                attacker.reputation_fear = clamp01(attacker.reputation_fear - 0.04)
            a.last_action, b.last_action = action_a, action_b
            return event

        if action_a == "cooperate" and action_b == "cooperate":
            event["cooperation"] = 1.0
            gain = 0.08
            a.wealth += gain
            b.wealth += gain
            a.reputation_coop = clamp01(a.reputation_coop + 0.05)
            b.reputation_coop = clamp01(b.reputation_coop + 0.05)
            update_memory(a, b, 0.08, -0.04)
            update_memory(b, a, 0.08, -0.04)
            a.plasticity("good_interaction", self.rng)
            b.plasticity("good_interaction", self.rng)
        elif "defect" in (action_a, action_b):
            event["defection"] = 1.0
            defector, target = (a, b) if action_a == "defect" else (b, a)
            transfer = max(0.0, target.wealth * 0.08)
            target.wealth -= transfer
            defector.wealth += transfer
            defector.reputation_coop = clamp01(defector.reputation_coop - 0.06)
            target.plasticity("betrayal", self.rng)
            update_memory(target, defector, -0.15, 0.10)
        elif "support" in (action_a, action_b):
            event["support"] = 1.0
            supporter, target = (a, b) if action_a == "support" else (b, a)
            target.wealth += 0.04
            supporter.reputation_coop = clamp01(supporter.reputation_coop + 0.04)
            update_memory(target, supporter, 0.10, -0.04)
        else:
            event["cooperation"] = 0.5

        a.last_action, b.last_action = action_a, action_b
        return event

    def step(self) -> Dict[str, object]:
        events = {"cooperation": 0.0, "violence": 0.0, "defection": 0.0, "support": 0.0, "avoidance": 0.0}
        interactions = max(1, len(self.alive_agents()) * self.steps_per_agent)

        for _ in range(interactions):
            pair = self._choose_pair()
            if pair is None:
                break
            a, b = pair
            action_a = a.decide_action(b, self.rng)
            action_b = b.decide_action(a, self.rng)
            outcome = self._apply_interaction(a, b, action_a, action_b)
            for key in events:
                events[key] += outcome.get(key, 0.0)

        self.step_count += 1
        denom = max(1.0, float(interactions))
        metrics = {
            "step": self.step_count,
            "population": len(self.alive_agents()),
            "coop_rate": events["cooperation"] / denom,
            "violence_rate": events["violence"] / denom,
            "defection_rate": events["defection"] / denom,
            "support_rate": events["support"] / denom,
            "avoidance_rate": events["avoidance"] / denom,
            "gini_wealth": gini(a.wealth for a in self.alive_agents()),
            "wealth_mean": float(np.mean([a.wealth for a in self.alive_agents()])) if self.alive_agents() else 0.0,
            "empathy_mean": self.trait_mean("empathy"),
            "reasoning_mean": self.trait_mean("reasoning"),
            "dominance_mean": self.trait_mean("dominance"),
            "dark_core_mean": float(np.mean([a.dark_core() for a in self.alive_agents()])) if self.alive_agents() else 0.0,
            "regime": self.classify_regime(),
        }
        self.history.append(metrics)
        return metrics

    def run(self, steps: int = 100) -> pd.DataFrame:
        for _ in range(int(steps)):
            self.step()
        return pd.DataFrame(self.history)

    def trait_mean(self, key: str) -> float:
        alive = self.alive_agents()
        if not alive:
            return 0.0
        return float(np.mean([a.traits.get(key, 0.5) for a in alive]))

    def classify_regime(self) -> str:
        alive = self.alive_agents()
        if len(alive) < 2:
            return "collapsed"
        wealth_gini = gini(a.wealth for a in alive)
        fear = float(np.mean([a.reputation_fear for a in alive]))
        coop = float(np.mean([a.reputation_coop for a in alive]))
        dominance = self.trait_mean("dominance")
        empathy = self.trait_mean("empathy")
        reasoning = self.trait_mean("reasoning")
        dark = float(np.mean([a.dark_core() for a in alive]))

        if fear > 0.55 and dominance > 0.60 and wealth_gini > 0.45:
            return "dominance_hierarchy"
        if coop > 0.62 and empathy > 0.58 and wealth_gini < 0.35:
            return "cooperative_pluralism"
        if reasoning > 0.62 and coop > 0.50 and dark < 0.45:
            return "deliberative_meritocracy"
        if dark > 0.55 or wealth_gini > 0.55:
            return "competitive_fragmentation"
        return "mixed_adaptive"

    def per_profile_stats(self) -> pd.DataFrame:
        rows = []
        for a in self.alive_agents():
            rows.append(
                {
                    "profile": a.profile_id,
                    "wealth": a.wealth,
                    "reputation_coop": a.reputation_coop,
                    "reputation_fear": a.reputation_fear,
                    "dark_core": a.dark_core(),
                    "last_action": a.last_action,
                    **{f"trait_{key}": value for key, value in a.traits.items()},
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return (
            df.groupby("profile")
            .agg(
                count=("profile", "count"),
                wealth_mean=("wealth", "mean"),
                wealth_gini=("wealth", gini),
                coop_rep_mean=("reputation_coop", "mean"),
                fear_rep_mean=("reputation_fear", "mean"),
                dark_core_mean=("dark_core", "mean"),
                empathy_mean=("trait_empathy", "mean"),
                reasoning_mean=("trait_reasoning", "mean"),
                dominance_mean=("trait_dominance", "mean"),
            )
            .reset_index()
        )
