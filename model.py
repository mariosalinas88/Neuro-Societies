"""Modelo Neuro Societies con reputación dual y regímenes emergentes (Mesa 3.3+)."""

from __future__ import annotations

import json
import math
import os
import random
import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    from culture import cultural_transmission_step
except Exception:  # pragma: no cover
    cultural_transmission_step = None  # type: ignore

try:
    from calibration import load_calibration_data, sample_calibrated_traits, calibration_quality_report
except Exception:  # pragma: no cover
    load_calibration_data = None  # type: ignore
    sample_calibrated_traits = None  # type: ignore
    calibration_quality_report = None  # type: ignore

try:
    import networkx as nx  # optional dependency for social topology
    from network import build_social_graph, compute_network_metrics, get_graph_neighbors, NetworkMetrics
except Exception:  # pragma: no cover - optional
    nx = None  # type: ignore
    build_social_graph = None  # type: ignore
    compute_network_metrics = None  # type: ignore
    get_graph_neighbors = None  # type: ignore
    NetworkMetrics = None  # type: ignore

import subprocess
from mesa import Agent, DataCollector, Model
from mesa.space import MultiGrid


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def gauss_sample(
    rng: np.random.Generator,
    mean: float,
    std: float | None = None,
    relative_std: float = 0.15,
    lo: float = 0.0,
    hi: float = 1.0,
    integer: bool = False,
) -> float:
    """
    Universal Gaussian sampler for stochastic values (clipped).

    GAUSSIANIZED: All numeric constants should use this sampler.
    """
    if std is None:
        std = abs(mean * relative_std)
    value = rng.normal(mean, std)
    clipped = np.clip(value, lo, hi)
    if integer:
        return int(round(float(clipped)))
    return float(clipped)


def gauss_weights(
    rng: np.random.Generator, means: List[float], relative_std: float = 0.15, normalize: bool = True
) -> List[float]:
    """Sample multiple weights from Gaussians and optionally normalize to 1."""
    weights = [gauss_sample(rng, m, relative_std=relative_std, lo=0.0, hi=1.0) for m in means]
    if normalize:
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
    return weights


def gini(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = arr.mean()
    if mean == 0:
        return 0.0
    diff_sum = np.abs(np.subtract.outer(arr, arr)).sum()
    return float(diff_sum / (2 * arr.size**2 * mean))


def load_profiles(path: str = "profiles.json") -> Dict[str, Dict[str, object]]:
    if not os.path.exists(path):
        return {}
    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        result = {}
        for p in data.get("profiles", []):
            pid = str(p.get("id", "")).strip()
            raw_traits = p.get("traits") or p.get("latents") or {}
            traits = {k: clamp01(float(v)) for k, v in raw_traits.items()}
            result[pid] = {
                "traits": traits,
                "name": p.get("name", ""),
                "description": p.get("description", ""),
                "biological_bias": p.get("biological_bias", {}),
                "spectrum_ranges": p.get("spectrum_ranges", {}),
            }
        return result
    except Exception:
        return {}


@dataclass
class EmergentNorm:
    rule: str
    strength: float
    enforcement: float
    cognitive_origin: Dict[str, float]
    supporters: List[int]
    behavior: str
    direction: str
    contextual_modifiers: Dict[str, float] = field(default_factory=dict)
    applies_to: str = "all"


@dataclass
class LegalSystem:
    formalism_index: float
    norms: List[EmergentNorm] = field(default_factory=list)
    enforcement_style: str = "universal"
    discretion_level: float = 0.5


@dataclass
class PoliticalSystem:
    participation_structure: str = "democracy"
    decision_makers: List[int] = field(default_factory=list)
    decision_weight_gini: float = 0.0
    benefit_orientation: str = "meritocratic"
    benefit_distribution_gini: float = 0.0
    legitimacy: float = 0.5
    stability: float = 0.5
    merit_criteria: Dict[str, float] = field(default_factory=dict)


@dataclass
class GovernanceDecision:
    decision_type: str
    initiator: int
    supporters: List[int]
    beneficiaries: List[int]
    cost_to_society: float
    benefit_to_beneficiaries: float


PROFILE_MAP = load_profiles()
TRACE_LATENTS = os.getenv("TRACE_LATENTS", "0") == "1"
USED_LATENT_KEYS: set[str] = set()


class TraceDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _touch(self, key):
        try:
            USED_LATENT_KEYS.add(str(key))
        except Exception:
            pass

    def get(self, key, default=None):
        self._touch(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self._touch(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._touch(key)
        return super().__contains__(key)

    def setdefault(self, key, default=None):
        self._touch(key)
        return super().setdefault(key, default)


def load_rare_variants(path: str = "profiles.json") -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        variants = []
        for v in data.get("rare_variants", []):
            vid = str(v.get("id", "")).strip()
            if not vid:
                continue
            variants.append(
                {
                    "id": vid,
                    "name": v.get("name", ""),
                    "probability": clamp01(float(v.get("probability", 0.0))),
                    "trait_mods": {k: float(vv) for k, vv in (v.get("trait_mods", {}) or {}).items()},
                    "imagination_boost": float(v.get("imagination_boost", 0.0) or 0.0),
                    "chaos_innovation": float(v.get("chaos_innovation", 0.0) or 0.0),
                    "narrative": v.get("narrative", "") or "",
                }
            )
        return variants
    except Exception:
        return []


RARE_VARIANTS = load_rare_variants()


def gauss_clip(rng: np.random.Generator, mean: float, std: float = 0.15, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(rng.normal(mean, std), lo, hi))


class Citizen(Agent):
    def __init__(
        self,
        model: "SocietyModel",
        latent: Dict[str, float],
        bias_ranges: Dict[str, List[float]] | None = None,
        spectrum_ranges: Dict[str, List[float]] | None = None,
        spectrum_level: int | None = None,
    ):
        super().__init__(model)
        # GAUSSIANIZED: initialization parameters sampled from Gaussians.
        base_latent = {k: clamp01(v) for k, v in latent.items()}
        self.latent = TraceDict(base_latent) if TRACE_LATENTS else dict(base_latent)
        self.bias_ranges = bias_ranges or {}
        self.spectrum_ranges = spectrum_ranges or {}
        if spectrum_level:
            self.spectrum_level = spectrum_level
        else:
            self.spectrum_level = int(self.model.model_gauss_sample(2, std=0.8, lo=1, hi=3, integer=True))
        self.last_p_coop = 0.0
        self.last_p_violence = 0.0
        self.last_p_support = 0.0
        self.last_delta = 0.0
        self.rare_variant: Dict[str, object] | None = None
        gender_p = self.model.model_gauss_sample(0.5, relative_std=0.1, lo=0.3, hi=0.7)
        self.gender = "Female" if self.model.rng.random() < gender_p else "Male"
        # ciclo vital y reproducción
        self.age = 0
        self.max_age = max(90, int(self.model.model_gauss_sample(260, std=45, lo=90, hi=400, integer=True)))
        base_fertility = self.model.model_gauss_sample(0.35, std=0.12, lo=0.1, hi=0.7)
        empathy_bonus_rate = self.model.model_gauss_sample(0.1, relative_std=0.3, lo=0.05, hi=0.2)
        self.fertile_prob = clamp01(base_fertility + empathy_bonus_rate * self.latent.get("empathy", 0.5))
        self.gestation_steps = max(10, int(self.model.model_gauss_sample(24, std=6, lo=10, hi=40, integer=True)))
        self.gestation_timer = 0
        self.fertility_cooldown = 0
        self.mate_history: List[int] = []
        self.offspring_count = 0
        self.mating_success = 0.0
        self.dominance_rank = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.status_score = self.model.model_gauss_sample(0.0, std=0.2, lo=-0.4, hi=0.4)
        self.current_partner: int | None = None
        # gender biases
        if self.gender == "Female":
            g_delta = self.model.model_gauss_sample(0.05, std=0.05, lo=0.0, hi=0.15)
            self.latent["language"] = clamp01(self.latent.get("language", 0.5) + g_delta)
            self.latent["sociality"] = clamp01(self.latent.get("sociality", 0.5) + g_delta)
            self.latent["risk_aversion"] = clamp01(self.latent.get("risk_aversion", 0.5) - g_delta)
            self.latent["emotional_impulsivity"] = clamp01(self.latent.get("emotional_impulsivity", 0.5) - g_delta)
        else:
            g_delta = self.model.model_gauss_sample(0.05, std=0.05, lo=0.0, hi=0.15)
            self.latent["language"] = clamp01(self.latent.get("language", 0.5) - g_delta)
            self.latent["sociality"] = clamp01(self.latent.get("sociality", 0.5) - g_delta)
            self.latent["risk_aversion"] = clamp01(self.latent.get("risk_aversion", 0.5) + g_delta)
            self.latent["emotional_impulsivity"] = clamp01(self.latent.get("emotional_impulsivity", 0.5) + g_delta)

        def level_scaled_range(base_range: List[float] | tuple[float, float]) -> tuple[float, float]:
            if not isinstance(base_range, (list, tuple)) or len(base_range) != 2:
                return (0.4, 0.6)
            low, high = float(base_range[0]), float(base_range[1])
            level = max(1, min(3, int(self.spectrum_level)))
            scale = level / 3.0
            adjusted_high = low + (high - low) * scale
            return clamp01(low), clamp01(max(low, adjusted_high))

        def sample_trait(key: str, default_range: tuple[float, float] = (0.4, 0.6)) -> float:
            if key in self.latent:
                return clamp01(self.latent[key])
            base_range = self.bias_ranges.get(key, default_range)
            if isinstance(base_range, (list, tuple)) and len(base_range) == 2:
                low, high = float(base_range[0]), float(base_range[1])
            else:
                low, high = default_range
            if key in self.spectrum_ranges:
                low, high = level_scaled_range(self.spectrum_ranges[key])
            if high < low:
                high = low
            mean = (low + high) / 2.0
            std = max(0.01, (high - low) / 2.0)
            sampled = self.model.model_gauss_sample(mean, std=std, lo=low, hi=high)
            noise = float(self.model.rng.normal(0, 0.05))
            return clamp01(sampled + noise)

        moral_emotional_keys = (
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
            "emotional_impulsivity",
            "resilience",
            "guilt",
            "shame",
        )
        for key in moral_emotional_keys:
            self.latent[key] = sample_trait(key)
        # baseline benevolence shift
        prosocial_shift = self.model.model_gauss_sample(0.05, std=0.03, lo=0.0, hi=0.12)
        spite_shift = self.model.model_gauss_sample(0.02, std=0.02, lo=0.0, hi=0.08)
        self.latent["moral_prosocial"] = clamp01(self.latent.get("moral_prosocial", 0.5) + prosocial_shift)
        self.latent["moral_spite"] = clamp01(self.latent.get("moral_spite", 0.5) - spite_shift)
        if "sexual_impulsivity" not in self.latent:
            w_imp, w_eimp = gauss_weights(self.model.rng, [0.5, 0.5], relative_std=0.2, normalize=True)
            base_imp = w_imp * self.latent.get("impulsivity", 0.5) + w_eimp * self.latent.get("emotional_impulsivity", 0.5)
            self.latent["sexual_impulsivity"] = clamp01(base_imp)

        self.guilt = clamp01(self.latent.get("guilt", 0.0))
        self.shame = clamp01(self.latent.get("shame", 0.0))
        self.norm_adherence = clamp01(0.5 * self.latent.get("moral_prosocial", 0.5) + 0.5 * self.latent.get("moral_common_good", 0.5))
        self.ostracism_timer = 0

        self.rare_variant = self._maybe_apply_rare_variant()
        if self.rare_variant:
            for k, delta in self.rare_variant.get("trait_mods", {}).items():
                self.latent[k] = clamp01(self.latent.get(k, 0.5) + float(delta))

        self.original_latent = dict(self.latent)
        e = self.latent.get("empathy", 0.5)
        d = self.latent.get("dominance", 0.5)
        imp = self.latent.get("impulsivity", 0.5)
        reg = self.latent.get("affect_reg", 0.5)
        narc = self.latent.get("dark_narc", 0.0)
        mach = self.latent.get("dark_mach", 0.0)
        psy = self.latent.get("dark_psycho", 0.0)
        prosocial = self.latent.get("moral_prosocial", 0.5)
        w_narc, w_mach, w_psy = gauss_weights(self.model.rng, [0.35, 0.35, 0.30], relative_std=0.2, normalize=True)
        dark_tri = clamp01(w_narc * narc + w_mach * mach + w_psy * psy)
        w_dark, w_empathy, w_dom, w_imp, w_reg, w_pro = gauss_weights(
            self.model.rng, [0.5, 0.3, 0.2, 0.2, 0.3, 0.2], relative_std=0.25, normalize=True
        )
        self.dark_core = clamp01(
            w_dark * dark_tri
            + w_empathy * (1.0 - e)
            + w_dom * d
            + w_imp * imp
            - w_reg * reg
            - w_pro * prosocial
        )
        reasoning = self.latent.get("reasoning", 0.5)
        self.conscious_core = self._init_conscious_core()
        w_r, w_lang, w_soc = gauss_weights(self.model.rng, [0.5, 0.3, 0.2], relative_std=0.2, normalize=True)
        self.conscious_core["self_model"]["agency"] = clamp01(
            w_r * reasoning + w_lang * self.latent.get("language", 0.5) + w_soc * self.latent.get("sociality", 0.5)
        )
        prosocial_thr = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.3, hi=0.7)
        self.resource_generation = reasoning * (e if prosocial > prosocial_thr else d)
        contrib_scale = self.model.model_gauss_sample(0.3, relative_std=0.3, lo=0.1, hi=0.6)
        self.nd_contribution = reasoning * e * contrib_scale
        self.nd_cost = self.model.model_gauss_sample(0.0, std=0.05, lo=0.0, hi=0.2)
        cap_base = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.7)
        cap_weight = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.7)
        self.violence_cap = clamp01(gauss_clip(self.model.rng, cap_base + cap_weight * dark_tri, 0.2))
        self.wealth = self.model.model_gauss_sample(1.0, relative_std=0.2, lo=0.5, hi=1.5)
        self.reproduction_cooldown = 0
        self.bonding_timer = 0
        self.mates_lifetime: set[int] = set()
        self.children_ids: List[int] = []
        # neuromoduladores
        self.dopamine = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.oxytocin = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.serotonin = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.endorphin = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.happiness = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.health = self.model.model_gauss_sample(1.0, relative_std=0.15, lo=0.6, hi=1.2)
        self.alive = True
        # reputaciones duales
        self.reputation_coop = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.reputation_fear = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.6)
        self.memory: Dict[int, Dict[str, object]] = {}
        self.social_memory: Dict[str, Dict[str, float]] = {}
        self.last_action: str | None = None
        self.alliance_id: str | None = None

    def reputation_total(self) -> float:
        high = max(self.reputation_coop, self.reputation_fear)
        low = min(self.reputation_coop, self.reputation_fear)
        low_weight = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
        return clamp01(high + low_weight * low)

    def get_status_score(self) -> float:
        wealth_term = math.log1p(max(self.wealth, 0.0) + 1.0)
        w_fear = self.model.model_gauss_sample(4.0, relative_std=0.25, lo=2.0, hi=6.0)
        w_coop = self.model.model_gauss_sample(2.0, relative_std=0.25, lo=1.0, hi=3.5)
        w_dom = self.model.model_gauss_sample(2.0, relative_std=0.25, lo=1.0, hi=3.5)
        w_reason = self.model.model_gauss_sample(0.8, relative_std=0.25, lo=0.3, hi=1.2)
        fear_term = w_fear * self.reputation_fear
        coop_term = w_coop * self.reputation_coop
        dom_term = w_dom * self.latent.get("dominance", 0.5)
        reason_term = w_reason * self.latent.get("reasoning", 0.5)
        noise = self.model.model_gauss_sample(0.0, std=0.2, lo=-0.4, hi=0.4)
        score = wealth_term + fear_term + coop_term + dom_term + reason_term + noise
        return max(0.0, score)

    def _update_status(self):
        self.status_score = self.get_status_score()
        w_dom, w_stat = gauss_weights(self.model.rng, [0.5, 0.5], relative_std=0.2, normalize=True)
        scale = self.model.model_gauss_sample(6.0, relative_std=0.2, lo=3.0, hi=9.0)
        self.dominance_rank = clamp01(w_dom * self.latent.get("dominance", 0.5) + w_stat * math.tanh(self.status_score / scale))

    def _decay_neurochem(self):
        k = clamp01(self.model.model_gauss_sample(self.model.neuro_decay_k, relative_std=0.2, lo=0.01, hi=0.5))
        for attr in ("dopamine", "oxytocin", "serotonin", "endorphin"):
            val = getattr(self, attr, 0.5)
            val = clamp01(val + k * (0.5 - val))
            setattr(self, attr, val)

    def reward(self, event_type: str, intensity: float = 1.0):
        intensity = clamp01(intensity)
        if event_type == "reproduction":
            d_boost = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.4)
            o_boost = self.model.model_gauss_sample(0.3, relative_std=0.4, lo=0.08, hi=0.5)
            e_boost = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.4)
            self.dopamine = clamp01(self.dopamine + d_boost * intensity)
            self.oxytocin = clamp01(self.oxytocin + o_boost * intensity)
            self.endorphin = clamp01(self.endorphin + e_boost * intensity)
        elif event_type == "reproduction_partner":
            d_boost = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.4)
            o_boost = self.model.model_gauss_sample(0.1, relative_std=0.5, lo=0.02, hi=0.25)
            self.dopamine = clamp01(self.dopamine + d_boost * intensity)
            self.oxytocin = clamp01(self.oxytocin + o_boost * intensity)
        elif event_type == "alliance":
            d_boost = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
            o_boost = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
            self.dopamine = clamp01(self.dopamine + d_boost * intensity)
            self.oxytocin = clamp01(self.oxytocin + o_boost * intensity)
        elif event_type == "conflict_win":
            delta = self.model.model_gauss_sample(0.08, relative_std=0.4, lo=0.02, hi=0.18) * intensity
            self.dopamine = clamp01(self.dopamine + delta)
            self.serotonin = clamp01(self.serotonin + delta)
        elif event_type == "conflict_lose":
            delta = self.model.model_gauss_sample(0.06, relative_std=0.4, lo=0.02, hi=0.15) * intensity
            self.dopamine = clamp01(self.dopamine - delta)
            self.serotonin = clamp01(self.serotonin - delta)

    def is_fertile(self) -> bool:
        if not self.model.enable_reproduction or self.gender != "Female":
            return False
        if self.gestation_timer > 0 or self.fertility_cooldown > 0:
            return False
        roll = self.model.rng.random()
        target_std = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
        target = clamp01(self.model.model_gauss_sample(self.fertile_prob, std=target_std, lo=0.0, hi=1.0))
        return roll < target

    def female_preference_score(self, male: "Citizen") -> float:
        wealth_score = self.model.normalized_wealth(male.wealth)
        dom = male.latent.get("dominance", 0.5)
        health_signal = clamp01(gauss_clip(self.model.rng, male.health, 0.1))
        # edad: pico en adultez
        adult_center = male.max_age * 0.35
        width = max(1.0, male.max_age * 0.25)
        age_factor = math.exp(-((male.age - adult_center) / width) ** 2)

        # pesos base globales
        w_wealth = self.model.model_gauss_sample(self.model.mate_weight_wealth, relative_std=0.2, lo=0.05, hi=0.8)
        w_dom = self.model.model_gauss_sample(self.model.mate_weight_dom, relative_std=0.2, lo=0.05, hi=0.8)
        w_health = self.model.model_gauss_sample(self.model.mate_weight_health, relative_std=0.2, lo=0.05, hi=0.8)
        w_age = self.model.model_gauss_sample(self.model.mate_weight_age, relative_std=0.2, lo=0.02, hi=0.6)

        # variaciones individuales
        dom_scale = self.model.model_gauss_sample(0.8, relative_std=0.2, lo=0.5, hi=1.1)
        dom_weight = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.8)
        wealth_scale = self.model.model_gauss_sample(0.9, relative_std=0.2, lo=0.6, hi=1.2)
        wealth_weight = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.6)
        health_scale = self.model.model_gauss_sample(0.9, relative_std=0.2, lo=0.6, hi=1.2)
        health_weight = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.6)
        w_dom *= clamp01(dom_scale + dom_weight * self.latent.get("dominance", 0.5))
        w_wealth *= clamp01(wealth_scale + wealth_weight * self.latent.get("risk_aversion", 0.5))
        w_health *= clamp01(health_scale + health_weight * self.latent.get("empathy", 0.5))

        score = w_wealth * wealth_score + w_dom * dom + w_health * health_signal + w_age * age_factor
        risk_scale = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
        risk_penalty = clamp01(gauss_clip(self.model.rng, male.latent.get("aggression", 0.5), 0.1)) * risk_scale
        score -= risk_penalty
        score += self.model.model_gauss_sample(0.0, std=0.05, lo=-0.1, hi=0.1)
        return score

    def _start_gestation(self, father: "Citizen"):
        self.gestation_timer = max(8, int(self.model.model_gauss_sample(self.gestation_steps, std=4, lo=8, hi=60, integer=True)))
        self.fertility_cooldown = max(1, int(self.model.model_gauss_sample(self.model.female_repro_cooldown, std=3, lo=1, hi=40, integer=True)))
        self.current_partner = father.unique_id
        self.mate_history.append(father.unique_id)
        father.mating_success += 1.0
        father.reproduction_cooldown = max(1, int(self.model.model_gauss_sample(self.model.male_repro_cooldown, std=2, lo=1, hi=20, integer=True)))
        if self.model.rng.random() < clamp01(self.dopamine):
            father.reproduction_cooldown = max(1, father.reproduction_cooldown - int(self.model.model_gauss_sample(1, std=0.3, lo=1, hi=2, integer=True)))
        cooldown_f = max(1, int(self.model.model_gauss_sample(self.model.female_repro_cooldown, std=3, lo=1, hi=40, integer=True)))
        self.reproduction_cooldown = max(self.reproduction_cooldown, cooldown_f)
        self.model.step_events["mating_attempts"] += 1

    def _apply_reproduction_costs(self):
        drain = self.model.model_gauss_sample(self.model.reproduction_costs, std=0.05, lo=0.0, hi=1.0)
        self.wealth -= drain
        health_hit = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
        happy_hit = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
        self.health = clamp01(self.health - health_hit * drain)
        self.happiness = clamp01(self.happiness - happy_hit * drain)

    def _give_birth(self):
        father = None
        for a in self.model.agents_alive():
            if a.unique_id == self.current_partner:
                father = a
                break
        if father is None:
            return
        child_traits: Dict[str, float] = {}
        for k in self.latent.keys():
            avg = 0.5 * (self.latent.get(k, 0.5) + father.latent.get(k, 0.5))
            child_traits[k] = clamp01(avg + self.model.rng.normal(0, 0.05))
        child = Citizen(self.model, child_traits, bias_ranges={}, spectrum_ranges={}, spectrum_level=None)
        child.profile_id = getattr(self, "profile_id", None) or "unknown"
        self.model.agents.add(child)
        self.model.grid.place_agent(child, self.pos)
        self.offspring_count += 1
        father.offspring_count += 1
        self.children_ids.append(child.unique_id)
        father.children_ids.append(child.unique_id)
        self.mating_success += 1.0
        self.mates_lifetime.add(father.unique_id)
        father.mates_lifetime.add(self.unique_id)
        self.model.births_total += 1
        self.model.step_events["births"] += 1
        self.reward("reproduction", intensity=1.0)
        father.reward("reproduction_partner", intensity=1.0)
        self._apply_reproduction_costs()
        self.gestation_timer = 0
        self.current_partner = None
        self.bonding_timer = self.model.bonding_steps

    def _bonding_tick(self):
        if self.bonding_timer <= 0 or self.current_partner is None:
            return
        partner = None
        for a in self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=1):
            if isinstance(a, Citizen) and a.unique_id == self.current_partner:
                partner = a
                break
        if partner:
            bond_delta = self.model.model_gauss_sample(self.model.bonding_delta, relative_std=0.4, lo=0.005, hi=0.08)
            self.oxytocin = clamp01(self.oxytocin + bond_delta)
            partner.oxytocin = clamp01(partner.oxytocin + 0.5 * bond_delta)
        self.bonding_timer -= 1

    def male_initiation(self):
        if not self.model.enable_reproduction or self.gender != "Male" or self.reproduction_cooldown > 0:
            return
        base_init = self.model.model_gauss_sample(self.model.male_initiation_base, relative_std=0.3, lo=0.01, hi=0.2)
        desire_scale = self.model.model_gauss_sample(self.model.male_desire_scale, relative_std=0.3, lo=0.05, hi=0.8)
        p = clamp01(base_init + desire_scale * self.latent.get("sexual_impulsivity", 0.5))
        p = clamp01(self.model.model_gauss_sample(p, std=0.05, lo=0.0, hi=1.0))
        if self.model.rng.random() > p:
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=2)
        females = [
            f
            for f in neighbors
            if isinstance(f, Citizen) and f.gender == "Female" and f.alive and f.gestation_timer == 0 and f.reproduction_cooldown == 0
        ]
        if not females:
            return
        target = self.model.rng.choice(females)
        score = target.female_preference_score(self)
        beta = self.model.model_gauss_sample(self.model.mate_choice_beta, relative_std=0.2, lo=0.2, hi=2.5)
        exp_accept = math.exp(beta * score)
        exp_reject = math.exp(0.0)
        accept_p = clamp01(exp_accept / (exp_accept + exp_reject))
        base_offset = self.model.model_gauss_sample(self.model.repro_base_offset, relative_std=0.3, lo=0.0, hi=0.6)
        desire_scale = self.model.model_gauss_sample(self.model.repro_desire_scale, relative_std=0.3, lo=0.05, hi=0.8)
        success_base = self.model.last_metrics.get("coop_rate", 0.0) - self.model.last_metrics.get("violence_rate", 0.0) + base_offset
        desire = desire_scale * target.latent.get("sexual_impulsivity", 0.5)
        success_p = clamp01(success_base + desire)
        w_accept, w_success = gauss_weights(self.model.rng, [0.5, 0.5], relative_std=0.2, normalize=True)
        final_p = clamp01(w_accept * accept_p + w_success * success_p)
        if self.model.rng.random() < final_p:
            target._start_gestation(self)
            target._apply_reproduction_costs()

    def _male_competition(self):
        if not self.model.enable_reproduction or self.gender != "Male":
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1)
        rivals = [m for m in neighbors if isinstance(m, Citizen) and m.gender == "Male" and m.alive]
        if not rivals:
            return
        dominance = self.latent.get("dominance", 0.5)
        aggression = self.latent.get("aggression", 0.5)
        empathy = self.latent.get("empathy", 0.5)
        base_drive = self.model.model_gauss_sample(0.05, relative_std=0.6, lo=0.01, hi=0.15)
        w_dom = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        w_aggr = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        w_emp = self.model.model_gauss_sample(0.1, relative_std=0.3, lo=0.02, hi=0.3)
        competition_drive = clamp01(base_drive + w_dom * dominance + w_aggr * aggression - w_emp * empathy)
        if self.model.rng.random() < competition_drive:
            self.model.step_events["male_male_conflicts"] += 1

    def attempt_mating(self):
        if not self.is_fertile():
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=2)
        candidates = [
            m
            for m in neighbors
            if isinstance(m, Citizen)
            and m.gender == "Male"
            and m.alive
            and m.age > 18
            and m.reproduction_cooldown == 0
        ]
        if not candidates:
            return
        scored = []
        for m in candidates:
            score = self.female_preference_score(m)
            gate_base = self.model.model_gauss_sample(1.0, relative_std=0.1, lo=0.7, hi=1.3)
            gate_threshold = self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.3, hi=0.9)
            resource_gate = clamp01(gate_base - self.model.resource_constraint * max(0.0, gate_threshold - self.wealth))
            score *= resource_gate
            scored.append((score, m))
        if not scored:
            return
        beta = self.model.model_gauss_sample(self.model.mate_choice_beta, relative_std=0.2, lo=0.2, hi=2.5)
        max_score = max(s for s, _ in scored)
        exps = [math.exp(beta * (s - max_score)) for s, _ in scored]
        total = sum(exps) or 1e-6
        pick_r = self.model.rng.random() * total
        accum = 0.0
        chosen = scored[0][1]
        for (expv, (_, cand)) in zip(exps, scored):
            accum += expv
            if pick_r <= accum:
                chosen = cand
                break
        base_offset = self.model.model_gauss_sample(self.model.repro_base_offset, relative_std=0.3, lo=0.0, hi=0.6)
        desire_scale = self.model.model_gauss_sample(self.model.repro_desire_scale, relative_std=0.3, lo=0.05, hi=0.8)
        success_base = self.model.last_metrics.get("coop_rate", 0.0) - self.model.last_metrics.get("violence_rate", 0.0) + base_offset
        desire = desire_scale * self.latent.get("sexual_impulsivity", 0.5)
        success_p = clamp01(success_base + desire)
        if self.model.rng.random() < success_p:
            self._start_gestation(chosen)
            self._apply_reproduction_costs()
        else:
            self.model.step_events["female_indirect_competition"] += 1

    def _mem_entry(self, other: "Citizen") -> Dict[str, object]:
        trust_init = self.model.model_gauss_sample(0.5, relative_std=0.25, lo=0.1, hi=0.9)
        return self.memory.setdefault(other.unique_id, {"trust": trust_init, "last_outcome": None, "interactions": 0})

    def predicted_coop(self, other: "Citizen") -> float:
        entry = self._mem_entry(other)
        trust = float(entry["trust"])
        empathy = self.latent.get("empathy", 0.5)
        # percepción social del otro
        coop_other = other.reputation_coop
        fear_other = other.reputation_fear
        status_div = self.model.model_gauss_sample(15.0, relative_std=0.2, lo=8.0, hi=25.0)
        status_boost = clamp01(other.get_perceived_status() / status_div)
        w_emp, w_coop, w_fear, w_status = gauss_weights(self.model.rng, [0.6, 0.25, 0.1, 0.05], relative_std=0.2, normalize=True)
        perceived = clamp01(w_emp * empathy + w_coop * coop_other + w_fear * (1 - fear_other) + w_status * status_boost)
        w_trust, w_perc = gauss_weights(self.model.rng, [0.5, 0.5], relative_std=0.2, normalize=True)
        base = w_trust * trust + w_perc * perceived
        return clamp01(base)

    def decide_action(self, other: "Citizen") -> str:
        entry = self._mem_entry(other)
        trust = float(entry["trust"])
        empathy = self.latent.get("empathy", 0.5)
        dominance = self.latent.get("dominance", 0.5)
        impulsivity = self.latent.get("impulsivity", 0.5)
        risk_aversion = self.latent.get("risk_aversion", 0.5)
        reasoning = self.latent.get("reasoning", 0.5)
        aggression = self.latent.get("aggression", 0.5)
        emo_imp = self.latent.get("emotional_impulsivity", 0.5)
        moral_prosocial = self.latent.get("moral_prosocial", 0.5)
        moral_common_good = self.latent.get("moral_common_good", 0.5)
        moral_honesty = self.latent.get("moral_honesty", 0.5)
        moral_spite = self.latent.get("moral_spite", 0.5)
        resilience = self.latent.get("resilience", 0.5)
        attn_flex = self.latent.get("attn_flex", 0.5)
        dark = self.dark_core
        std_tiny = self.model.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.15)
        std_small = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.2)
        std_med = self.model.model_gauss_sample(0.15, relative_std=0.4, lo=0.05, hi=0.3)

        predicted_coop = self.predicted_coop(other)
        reputation_other_coop = other.reputation_coop
        reputation_other_fear = other.reputation_fear
        # perception bias modulation
        threat_sens = self.latent.get("perception_threat", self.latent.get("perception_bias", {}).get("threat_sensitivity", 0.5))
        detail_orient = self.latent.get("perception_detail", self.latent.get("perception_bias", {}).get("detail_orientation", 0.5))
        social_cue = self.latent.get("perception_social", self.latent.get("perception_bias", {}).get("social_cue_weight", 0.5))
        threat_weight = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
        perceived_threat = threat_sens * (1.0 - reputation_other_coop) + (1 - social_cue) * reputation_other_fear + detail_orient * (1 - predicted_coop)
        predicted_coop = clamp01(predicted_coop - perceived_threat * threat_weight)
        self._update_conscious_perception(other, perceived_threat)

        # RAG-style episodic memory retrieval to bias action selection.
        context_query = f"interaction {getattr(other, 'profile_id', '')} {other.unique_id} {self.last_action or ''}"
        relevant_memories = self.retrieve_relevant_memory(context_query)
        if relevant_memories:
            mem_valence = float(np.mean([self._memory_valence(m.get("event", "")) for m in relevant_memories]))
            mem_weight = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
        else:
            mem_valence = 0.0
            mem_weight = 0.0

        w_rep = self.model.model_gauss_sample(0.3, std=0.15, lo=0.05, hi=0.6)
        base_coop = empathy + trust + gauss_clip(self.model.rng, w_rep, std_med) * reputation_other_coop - risk_aversion
        w_pro = self.model.model_gauss_sample(0.4, std=0.15, lo=0.05, hi=0.8)
        w_common = self.model.model_gauss_sample(0.2, std=0.15, lo=0.03, hi=0.6)
        w_hon = self.model.model_gauss_sample(0.1, std=0.15, lo=0.01, hi=0.5)
        w_spite = self.model.model_gauss_sample(0.1, std=0.15, lo=0.01, hi=0.5)
        base_coop += gauss_clip(self.model.rng, w_pro, std_med) * moral_prosocial
        base_coop += gauss_clip(self.model.rng, w_common, std_med) * moral_common_good
        base_coop += gauss_clip(self.model.rng, w_hon, std_med) * moral_honesty
        base_coop -= gauss_clip(self.model.rng, w_spite, std_med) * moral_spite
        dark_weight = self.model.model_gauss_sample(0.7, std=0.15, lo=0.3, hi=1.1)
        base_coop *= (1.0 - gauss_clip(self.model.rng, dark_weight, std_med) * dark)
        res_boost = self.model.model_gauss_sample(0.2, std=0.1, lo=0.02, hi=0.4)
        base_coop += gauss_clip(self.model.rng, res_boost * resilience * empathy, std_small)
        attn_weight = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
        base_coop += attn_weight * attn_flex
        reason_weight = self.model.model_gauss_sample(0.15, relative_std=0.4, lo=0.03, hi=0.3)
        base_coop += reason_weight * reasoning * max(0.0, 1.0 - self.model.last_metrics.get("violence_rate", 0.0))
        if getattr(self.model, "enable_guilt", False):
            guilt_w = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
            base_coop += guilt_w * self.guilt
        if getattr(self.model, "enable_cultural_transmission", False):
            adherence_w = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
            base_coop += adherence_w * self.norm_adherence
        base_coop += mem_weight * mem_valence
        if self.alliance_id and self.alliance_id == getattr(other, "alliance_id", None):
            ally_boost = self.model.model_gauss_sample(1.15, relative_std=0.1, lo=1.02, hi=1.35)
            base_coop *= ally_boost
            base_support_bias = self.model.model_gauss_sample(0.08, relative_std=0.4, lo=0.01, hi=0.2)
        else:
            base_support_bias = self.model.model_gauss_sample(0.0, std=0.02, lo=-0.05, hi=0.05)
        base_coop *= gauss_clip(self.model.rng, self.model.model_gauss_sample(1.7, std=0.2, lo=1.2, hi=2.2), std_med)
        base_coop += self._imagine_outcome(other)
        prob_coop = clamp01(gauss_clip(self.model.rng, base_coop, self.model.model_gauss_sample(0.15, relative_std=0.4, lo=0.05, hi=0.3)))

        support_base = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.3, std=0.1, lo=0.1, hi=0.6), 0.1)
        w_emp, w_reason = gauss_weights(self.model.rng, [0.7, 0.3], relative_std=0.2, normalize=True)
        base_support = support_base * (empathy * w_emp + reasoning * w_reason)
        prosocial_thr = self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        if moral_prosocial > prosocial_thr:
            base_support += gauss_clip(self.model.rng, self.model.model_gauss_sample(0.2, std=0.1, lo=0.05, hi=0.4), std_small)
        base_support += base_support_bias
        base_support += self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12) * attn_flex
        ox_weight = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
        base_support *= 1.0 + ox_weight * self.oxytocin
        prob_support = clamp01(gauss_clip(self.model.rng, base_support, self.model.model_gauss_sample(0.15, relative_std=0.4, lo=0.05, hi=0.3)))

        imp_bias = self.model.model_gauss_sample(0.1, relative_std=0.5, lo=0.02, hi=0.25)
        base_violence = dominance * (1.0 - empathy) * (impulsivity + imp_bias) * (1.0 - predicted_coop)
        dark_base = self.model.model_gauss_sample(0.3, relative_std=0.3, lo=0.1, hi=0.6)
        dark_weight = self.model.model_gauss_sample(0.7, relative_std=0.2, lo=0.3, hi=1.0)
        base_violence *= (dark_base + dark_weight * dark)
        strategic = reasoning
        strat_base = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        strat_weight = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        base_violence *= (strat_base + strat_weight * strategic * (1.0 - reputation_other_coop))
        aggr_w = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.5, std=0.15, lo=0.2, hi=0.9), std_med)
        emo_w = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.4, std=0.15, lo=0.1, hi=0.8), std_med)
        base_violence *= 1.0 + aggr_w * aggression + emo_w * emo_imp
        if getattr(self.model, "enable_guilt", False):
            guilt_w = self.model.model_gauss_sample(0.4, relative_std=0.4, lo=0.1, hi=0.8)
            base_violence *= max(0.0, 1.0 - guilt_w * self.guilt)
        if getattr(self.model, "enable_cultural_transmission", False):
            adherence_w = self.model.model_gauss_sample(0.3, relative_std=0.4, lo=0.05, hi=0.7)
            base_violence *= max(0.0, 1.0 - adherence_w * self.norm_adherence)
        base_violence *= 1.0 + mem_weight * (-mem_valence)
        w_pro = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.4, std=0.15, lo=0.1, hi=0.8), std_med)
        w_common = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.2, std=0.15, lo=0.05, hi=0.6), std_med)
        w_hon = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.15, std=0.1, lo=0.05, hi=0.5), std_small)
        moral_brake = clamp01(w_pro * moral_prosocial + w_common * moral_common_good + w_hon * moral_honesty)
        brake_w = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.6, std=0.15, lo=0.2, hi=0.9), std_med)
        base_violence *= max(0.0, 1.0 - brake_w * moral_brake)
        low_v = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.2, std=0.05, lo=0.05, hi=0.4), std_tiny)
        high_v = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.5, std=0.1, lo=0.2, hi=0.9), std_small)
        if high_v < low_v:
            high_v = low_v
        base_violence += self.model.rng.uniform(low_v, high_v) * emo_imp
        floor = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.5, std=0.1, lo=0.2, hi=0.8), std_small)
        res_w = gauss_clip(self.model.rng, self.model.model_gauss_sample(0.4, std=0.15, lo=0.1, hi=0.8), std_med)
        base_violence *= max(floor, 1.0 - res_w * resilience)
        base_violence += gauss_clip(self.model.rng, self.model.model_gauss_sample(0.2, std=0.1, lo=0.05, hi=0.4) * dark * reasoning, std_small)
        if self.alliance_id and self.alliance_id != getattr(other, "alliance_id", None):
            base_violence *= self.model.model_gauss_sample(0.9, relative_std=0.1, lo=0.75, hi=1.05)
        if self._calculate_malice(other):
            base_violence += self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9)
        attn_floor = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        attn_w = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        base_violence *= max(attn_floor, 1.0 - attn_w * attn_flex)
        reason_floor = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.3)
        reason_w = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.7)
        base_violence *= max(reason_floor, 1.0 - reason_w * reasoning)
        oxy_floor = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        oxy_w = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.7)
        base_violence *= max(oxy_floor, 1.0 - oxy_w * self.oxytocin)
        sero_base = self.model.model_gauss_sample(1.0, relative_std=0.1, lo=0.7, hi=1.3)
        sero_w = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        sero_thr = self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.3, hi=0.9)
        base_violence *= sero_base + sero_w * max(0.0, sero_thr - self.serotonin)
        prob_violence = clamp01(gauss_clip(self.model.rng, base_violence, self.model.model_gauss_sample(0.15, relative_std=0.4, lo=0.05, hi=0.3)))
        self.last_p_coop = prob_coop
        self.last_p_violence = prob_violence
        self.last_p_support = prob_support

        prob_defect = max(0.0, 1.0 - min(1.0, prob_coop + prob_violence + prob_support))

        total = prob_coop + prob_defect + prob_violence + prob_support
        if total <= 0:
            return "defect"
        r = self.model.rng.random() * total
        if r < prob_coop:
            return "coop"
        if r < prob_coop + prob_defect:
            return "defect"
        if r < prob_coop + prob_defect + prob_violence:
            return "violence"
        return "support"

    def neuroplasticity(self, context: str):
        affect_reg = self.latent.get("affect_reg", 0.5)
        dark = self.dark_core
        emo_imp = self.latent.get("emotional_impulsivity", 0.5)
        resilience = self.latent.get("resilience", 0.5)

        def rand_delta(base_min: float = 0.2, base_max: float = 1.0) -> float:
            base_min_g = self.model.model_gauss_sample(base_min, relative_std=0.3, lo=0.05, hi=2.0)
            base_max_g = self.model.model_gauss_sample(base_max, relative_std=0.3, lo=0.1, hi=2.5)
            bias_scale = self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=1.2) + emo_imp
            lo = base_min_g * bias_scale
            hi = base_max_g * bias_scale
            mid = (lo + hi) / 2
            std = self.model.model_gauss_sample(0.3, relative_std=0.4, lo=0.05, hi=0.8)
            return float(np.clip(self.model.rng.normal(mid, std), lo, hi))

        def should_change() -> bool:
            base = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
            weight = self.model.model_gauss_sample(0.6, relative_std=0.3, lo=0.2, hi=0.9)
            base_p = base + weight * (1.0 - resilience)
            return self.model.rng.random() < clamp01(base_p)

        def apply_delta(key: str, sign: float, magnitude: float):
            self.latent[key] = clamp01(self.latent.get(key, 0.5) + sign * magnitude)

        trauma_threshold = self.model.model_gauss_sample(-0.2, std=0.2, lo=-0.8, hi=0.2)
        traumatic_loss = self.wealth < trauma_threshold
        trauma_hi = self.model.model_gauss_sample(2.0, relative_std=0.3, lo=1.2, hi=3.5)
        trauma_lo = self.model.model_gauss_sample(1.0, relative_std=0.2, lo=0.7, hi=1.3)
        trauma_boost = trauma_hi if traumatic_loss else trauma_lo

        if context == "coop_success":
            if should_change():
                delta = rand_delta(0.2, 1.0) * trauma_boost
                apply_delta("empathy", +1, delta)
                apply_delta("moral_prosocial", +1, delta)
                apply_delta("moral_common_good", +1, delta * self.model.model_gauss_sample(0.8, relative_std=0.3, lo=0.3, hi=1.2))
                apply_delta("aggression", -1, delta * self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9))
                apply_delta("moral_spite", -1, delta * self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9))
                apply_delta("dark_mach", -1, delta * self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.8))
                apply_delta("dark_psycho", -1, delta * self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.8))
                apply_delta("affect_reg", +1, delta * self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9))
            change_p = self.model.model_gauss_sample(0.3, relative_std=0.4, lo=0.1, hi=0.6)
            if self.model.rng.random() < change_p:
                apply_delta("trust", +1, rand_delta(0.1, 0.4))

        elif context == "betrayed":
            if should_change():
                delta = rand_delta(0.2, 1.0) * trauma_boost
                apply_delta("empathy", -1, delta)
                apply_delta("moral_prosocial", -1, delta)
                apply_delta("moral_honesty", -1, delta * self.model.model_gauss_sample(0.8, relative_std=0.3, lo=0.3, hi=1.2))
                apply_delta("aggression", +1, delta)
                apply_delta("moral_spite", +1, delta)
                apply_delta("dark_mach", +1, delta * self.model.model_gauss_sample(0.6, relative_std=0.3, lo=0.2, hi=1.0))
                apply_delta("emotional_impulsivity", +1, delta * self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9))
                apply_delta("risk_aversion", +1, delta * self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9))
            dark_thr = self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
            if dark > dark_thr:
                apply_delta("dominance", +1, rand_delta(0.1, 0.5))

        elif context == "violence_success":
            if should_change():
                delta = rand_delta(0.2, 1.0) * trauma_boost
                apply_delta("aggression", +1, delta)
                apply_delta("moral_spite", +1, delta * self.model.model_gauss_sample(0.8, relative_std=0.3, lo=0.3, hi=1.2))
                apply_delta("dark_psycho", +1, delta * self.model.model_gauss_sample(0.7, relative_std=0.3, lo=0.3, hi=1.1))
                apply_delta("dark_narc", +1, delta * self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9))
                apply_delta("moral_prosocial", -1, delta * self.model.model_gauss_sample(0.8, relative_std=0.3, lo=0.3, hi=1.2))
                apply_delta("empathy", -1, delta * self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9))
            emp_thr = self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
            empathic_p = self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.8)
            if self.latent.get("empathy", 0.5) > emp_thr and self.model.rng.random() < empathic_p:
                apply_delta("moral_spite", -1, rand_delta(0.1, 0.4))
            self.reputation_fear = clamp01(self.reputation_fear + 0.05 - 0.08 * self.model.institution_pressure)

        # fluid reversion toward original if new rentable outcome arises
        revert_low = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.3)
        revert_high = self.model.model_gauss_sample(0.3, relative_std=0.3, lo=0.1, hi=0.6)
        if revert_high < revert_low:
            revert_low, revert_high = revert_high, revert_low
        if self.model.rng.random() < self.model.rng.uniform(revert_low, revert_high):
            for key, orig in self.original_latent.items():
                revert_rate = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
                self.latent[key] = clamp01(self.latent[key] + revert_rate * (orig - self.latent[key]))

        noise_base = self.model.model_gauss_sample(0.005, relative_std=0.5, lo=0.001, hi=0.02)
        noise_sigma = noise_base * (1.0 + (1.0 - affect_reg))
        for key in (
            "empathy",
            "dominance",
            "reasoning",
            "risk_aversion",
            "affect_reg",
            "impulsivity",
            "language",
            "aggression",
            "emotional_impulsivity",
            "resilience",
        ):
            self.latent[key] = clamp01(self.latent.get(key, 0.5) + self.model.rng.normal(0, noise_sigma))
        self.last_delta = delta if 'delta' in locals() else 0.0
        reasoning = self.latent.get("reasoning", 0.5)
        meta_thr = self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        reason_thr = self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        wealth_thr = self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.1, hi=1.0)
        if self.conscious_core["self_model"]["metacognition"] > meta_thr and reasoning > reason_thr and context == "violence_success" and self.wealth < wealth_thr:
            aggr_drop = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
            self.latent["aggression"] = clamp01(self.latent.get("aggression", 0.5) - aggr_drop)

        e = self.latent.get("empathy", 0.5)
        d = self.latent.get("dominance", 0.5)
        imp = self.latent.get("impulsivity", 0.5)
        reg = self.latent.get("affect_reg", 0.5)
        narc = self.latent.get("dark_narc", 0.0)
        mach = self.latent.get("dark_mach", 0.0)
        psy = self.latent.get("dark_psycho", 0.0)
        prosocial = self.latent.get("moral_prosocial", 0.5)
        dark_tri = clamp01(0.35 * narc + 0.35 * mach + 0.30 * psy)
        self.dark_core = clamp01(
            0.5 * dark_tri
            + 0.3 * (1.0 - e)
            + 0.2 * d
            + 0.2 * imp
            - 0.3 * reg
            - 0.2 * prosocial
        )

    def update_memory(self, other: "Citizen", outcome: str, trust_delta: float):
        entry = self._mem_entry(other)
        entry["interactions"] += self.model.model_gauss_sample(1.0, relative_std=0.1, lo=0.7, hi=1.3)
        entry["last_outcome"] = outcome
        entry["trust"] = clamp01(entry["trust"] + trust_delta)

    def remember_behavior(self, behavior: str, outcome: float):
        entry = self.social_memory.setdefault(behavior, {"score": 0.0, "count": 0.0})
        entry["score"] += float(outcome)
        entry["count"] += 1.0

    def interact(self):
        if not self.alive:
            return
        # GAUSSIANIZED: interaction payoffs, penalties, and thresholds.
        raw_neighbors = self.model.get_social_neighbors(self, radius=1, include_center=False)
        if not raw_neighbors:
            return
        reasoning = self.latent.get("reasoning", 0.5)
        filtered = []
        for n in raw_neighbors:
            w_fear, w_coop = gauss_weights(self.model.rng, [0.6, 0.4], relative_std=0.2, normalize=True)
            risk = w_fear * n.reputation_fear + w_coop * (1.0 - n.reputation_coop)
            dominance = self.latent.get("dominance", 0.5)
            impulsivity = self.latent.get("impulsivity", 0.5)
            risk_aversion = self.latent.get("risk_aversion", 0.5)
            w_dom, w_imp, w_risk = gauss_weights(self.model.rng, [0.5, 0.3, 0.2], relative_std=0.2, normalize=True)
            bravery = clamp01(w_dom * dominance + w_imp * impulsivity + w_risk * (1 - risk_aversion))
            avoid_p = clamp01(reasoning * risk * (1 - bravery))
            seek_base = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
            seek_w = self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.9)
            seek_p = clamp01(bravery * (seek_base + seek_w * (1 - risk_aversion)))
            reason_thr = self.model.model_gauss_sample(0.7, relative_std=0.2, lo=0.4, hi=0.9)
            risk_thr = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.3, hi=0.8)
            pred_base = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.7)
            pred_w = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.7)
            if reasoning > reason_thr and self.model.rng.random() < gauss_clip(self.model.rng, pred_base + pred_w * max(0, reasoning - 0.5), 0.15) and risk > risk_thr:
                continue  # predicted negative interaction, skip
            if self.model.rng.random() < avoid_p and self.model.rng.random() > seek_p:
                continue
            filtered.append(n)
        neighbors = filtered or raw_neighbors
        max_interactions = max(1, int(self.model.model_gauss_sample(2, std=0.6, lo=1, hi=4, integer=True)))
        interactions = min(len(neighbors), max_interactions)
        for other in self.model.rng.choice(neighbors, interactions, replace=False):
            my_action = self.decide_action(other)
            other_action = other.decide_action(self)
            self.last_action = my_action
            other.last_action = other_action

            if my_action == "violence" or other_action == "violence":
                attacker, victim = (self, other) if my_action == "violence" else (other, self)
                special = attacker._attack(victim)
                if special == "SUCCESS":
                    self._log_episode(attacker, victim, "violence")
                    self.model.record_behavior("violence", attacker, victim, actor_outcome=1.0, target_outcome=-1.0)
                    self.model.apply_decentralized_enforcement("violence", attacker, victim)
                    if getattr(self.model, "enable_guilt", False):
                        guilt_gain = self.model.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.12)
                        guilt_gain *= attacker.latent.get("empathy", 0.5) * attacker.latent.get("moral_prosocial", 0.5)
                        guilt_drop = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
                        guilt_gain -= guilt_drop * attacker.latent.get("dark_mach", 0.0)
                        attacker._adjust_guilt(guilt_gain)
                    drag_base = self.model.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.15)
                    drag = drag_base * (1.0 - self.model.last_metrics.get("coop_rate", 0.0))
                    attacker.wealth -= drag
                    victim.wealth -= drag
                    self.model.total_wealth -= drag * 2
                    self.model.register_violence(attacker, victim)
                    continue
                if special == "FAILED":
                    self._log_episode(attacker, victim, "violence_fail")
                    self.model.record_behavior("violence", attacker, victim, actor_outcome=-0.2, target_outcome=0.1)
                    drag = self.model.model_gauss_sample(0.03, relative_std=0.5, lo=0.005, hi=0.1)
                    attacker.wealth -= drag
                    self.model.total_wealth -= drag
                    self.model.register_violence(attacker, victim)
                    continue
                gain_base = self.model.model_gauss_sample(0.8, std=0.15, lo=0.3, hi=1.2)
                gain_bonus = self.model.model_gauss_sample(0.2, std=0.05, lo=0.05, hi=0.4)
                gain = gain_base * (attacker.latent.get("dominance", 0.5) + gain_bonus)
                attacker.wealth += gain
                victim.wealth -= gain * 1.2
                cost_base = self.model.model_gauss_sample(0.3, std=0.1, lo=0.1, hi=0.6)
                cost = self.model.legal_formalism * cost_base
                attacker.wealth -= cost
                self.model.total_wealth -= cost
                happy_boost = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
                happy_hit = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.4)
                attacker.happiness = clamp01(attacker.happiness + happy_boost)
                victim.happiness = clamp01(victim.happiness - happy_hit)
                # Rare lethality only under extreme conditions.
                aggr = self.model.model_gauss_sample(attacker.latent.get("aggression", 0.5), relative_std=0.2, lo=0.0, hi=1.0)
                dom = self.model.model_gauss_sample(attacker.latent.get("dominance", 0.5), relative_std=0.2, lo=0.0, hi=1.0)
                res = self.model.model_gauss_sample(victim.latent.get("resilience", 0.5), relative_std=0.2, lo=0.0, hi=1.0)
                emp = self.model.model_gauss_sample(victim.latent.get("empathy", 0.5), relative_std=0.2, lo=0.0, hi=1.0)
                extreme = max(0.0, aggr + dom - res - emp)
                lethality_base = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.05)
                lethality_prob = clamp01(lethality_base * extreme * max(0.1, self.model.lethality_multiplier))
                extreme_thr = self.model.model_gauss_sample(0.8, relative_std=0.3, lo=0.6, hi=0.95)
                allow_gate = False
                formal_thr = self.model.model_gauss_sample(0.7, relative_std=0.2, lo=0.5, hi=0.9)
                leg_thr = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.2, hi=0.6)
                allow_gate = (
                    self.model.legal_formalism > formal_thr
                    and self.model.political_system.participation_structure in ("tyranny", "oligarchy")
                    and self.model.political_system.legitimacy < leg_thr
                )
                allow_gate = allow_gate and self.model.rng.random() < self.model.model_gauss_sample(0.5, std=0.2, lo=0.2, hi=0.8)
                pop_guard = self.model.model_gauss_sample(40, std=10, lo=15, hi=70)
                if len(self.model.agents_alive()) < pop_guard:
                    allow_gate = False
                if allow_gate and extreme > extreme_thr and victim.model.rng.random() < lethality_prob:
                    victim.alive = False
                    self.model.step_events["violence_deaths"] += 1.0
                else:
                    victim.alive = victim.wealth > -2.0
                attacker.neuroplasticity("violence_success")
                victim.neuroplasticity("betrayed")
                attacker.reward("conflict_win", intensity=1.0)
                victim.reward("conflict_lose", intensity=1.0)
                trust_hit = self.model.model_gauss_sample(0.12, relative_std=0.4, lo=0.03, hi=0.25)
                attacker.update_memory(victim, "violence", trust_delta=-trust_hit)
                victim.update_memory(attacker, "violence", trust_delta=-trust_hit)
                fear_boost = self.model.model_gauss_sample(0.12, relative_std=0.4, lo=0.02, hi=0.25)
                coop_penalty = self.model.model_gauss_sample(0.15, relative_std=0.4, lo=0.03, hi=0.3)
                attacker.reputation_fear = clamp01(attacker.reputation_fear + fear_boost)
                attacker.reputation_coop = clamp01(attacker.reputation_coop - coop_penalty)
                nd_cost = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
                victim.nd_cost += nd_cost
                self.model.total_wealth -= nd_cost
                self._log_episode(attacker, victim, "violence")
                self.model.record_behavior("violence", attacker, victim, actor_outcome=1.0, target_outcome=-1.0)
                self.model.apply_decentralized_enforcement("violence", attacker, victim)
                if getattr(self.model, "enable_guilt", False):
                    guilt_gain = self.model.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.12)
                    guilt_gain *= attacker.latent.get("empathy", 0.5) * attacker.latent.get("moral_prosocial", 0.5)
                    guilt_drop = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
                    guilt_gain -= guilt_drop * attacker.latent.get("dark_mach", 0.0)
                    attacker._adjust_guilt(guilt_gain)
                # Negative gossip and anti-aggressor alliance as non-lethal feedback.
                neighbors = self.model.get_social_neighbors(victim, radius=2, include_center=True)
                gossip_p = self.model.model_gauss_sample(0.5, relative_std=0.4, lo=0.1, hi=0.9)
                if victim.alive and self.model.rng.random() < gossip_p:
                    victim.gossip(neighbors=neighbors, target=attacker, force_negative=True)
                alliance_p = self.model.model_gauss_sample(0.3, relative_std=0.4, lo=0.05, hi=0.7)
                if victim.alive and self.model.rng.random() < alliance_p:
                    victim._form_punishment_alliance(attacker)
                continue

            if my_action == "coop" and other_action == "coop":
                bonus_base = self.model.model_gauss_sample(0.6, std=0.15, lo=0.2, hi=1.0)
                bonus_raw = bonus_base * (self.latent.get("empathy", 0.5) + other.latent.get("empathy", 0.5))
                efficiency = self._calculate_trade_efficiency(other)
                bonus = bonus_raw * efficiency
                self.wealth += bonus
                other.wealth += bonus
                happy_boost = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.2)
                self.happiness = clamp01(self.happiness + happy_boost)
                other.happiness = clamp01(other.happiness + happy_boost)
                # development: high reasoning + empathetic resource generation boosts total wealth
                for actor in (self, other):
                    actor.resource_generation = actor.latent.get("reasoning", 0.5) * (actor.latent.get("empathy", 0.5) if actor.latent.get("moral_prosocial", 0.5) > 0.5 else actor.latent.get("dominance", 0.5))
                    if actor.latent.get("moral_prosocial", 0.5) > 0.5 and actor.resource_generation > 0.4:
                        low_dev = self.model.model_gauss_sample(0.2, std=0.05, lo=0.05, hi=0.4)
                        high_dev = self.model.model_gauss_sample(0.5, std=0.1, lo=0.2, hi=0.9)
                        if high_dev < low_dev:
                            high_dev = low_dev
                        dev = float(self.model.rng.uniform(low_dev, high_dev) * actor.resource_generation)
                        self.model.total_wealth += dev
                        share_rate = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.2)
                        actor.wealth += share_rate * dev
                        other.wealth += share_rate * dev
                        actor.nd_contribution += dev
                self.neuroplasticity("coop_success")
                other.neuroplasticity("coop_success")
                trust_boost = self.model.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.12)
                self.update_memory(other, "coop", trust_delta=trust_boost)
                other.update_memory(self, "coop", trust_delta=trust_boost)
                coop_boost = self.model.model_gauss_sample(0.06, relative_std=0.4, lo=0.01, hi=0.12)
                self.reputation_coop = clamp01(self.reputation_coop + coop_boost)
                other.reputation_coop = clamp01(other.reputation_coop + coop_boost)
                self._log_episode(self, other, "coop")
                self.model.record_behavior("coop", self, other, actor_outcome=1.0, target_outcome=1.0)
                self.model.apply_normative_rewards("coop", self, other)
                if getattr(self.model, "enable_guilt", False):
                    guilt_drop = self.model.model_gauss_sample(0.03, relative_std=0.5, lo=0.005, hi=0.08)
                    self._adjust_guilt(-guilt_drop)
                    other._adjust_guilt(-guilt_drop)
            elif my_action == "support" or other_action == "support":
                supporter = self if my_action == "support" else other
                receiver = other if supporter is self else self
                low_s = self.model.model_gauss_sample(0.2, std=0.05, lo=0.05, hi=0.4)
                high_s = self.model.model_gauss_sample(0.6, std=0.1, lo=0.2, hi=0.9)
                if high_s < low_s:
                    high_s = low_s
                dev = float(self.model.rng.uniform(low_s, high_s) * supporter.latent.get("reasoning", 0.5))
                self.model.total_wealth += dev
                support_share = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
                receive_share = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.2)
                supporter.wealth += support_share * dev
                receiver.wealth += receive_share * dev
                support_boost = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
                supporter.reputation_coop = clamp01(supporter.reputation_coop + support_boost)
                empathy_bump = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.05)
                supporter.latent["empathy"] = clamp01(supporter.latent.get("empathy", 0.5) + empathy_bump)
                supporter.nd_contribution += dev
                self._log_episode(supporter, receiver, "support")
                self.model.record_behavior("support", supporter, receiver, actor_outcome=0.8, target_outcome=0.4)
                self.model.apply_normative_rewards("support", supporter, receiver)
                if getattr(self.model, "enable_guilt", False):
                    guilt_drop = self.model.model_gauss_sample(0.04, relative_std=0.5, lo=0.01, hi=0.1)
                    supporter._adjust_guilt(-guilt_drop)
            elif my_action == "coop" and other_action == "defect":
                steal = self.model.model_gauss_sample(0.9, relative_std=0.1, lo=0.6, hi=1.1)
                other.wealth += steal
                self.wealth -= steal
                happy_boost = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
                happy_hit = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.2)
                other.happiness = clamp01(other.happiness + happy_boost)
                self.happiness = clamp01(self.happiness - happy_hit)
                self.neuroplasticity("betrayed")
                other.neuroplasticity("violence_success")
                trust_hit = self.model.model_gauss_sample(0.12, relative_std=0.4, lo=0.03, hi=0.25)
                self.update_memory(other, "defect", trust_delta=-trust_hit)
                neutral_trust = self.model.model_gauss_sample(0.0, std=0.02, lo=-0.05, hi=0.05)
                other.update_memory(self, "defect", trust_delta=neutral_trust)
                fear_boost = self.model.model_gauss_sample(0.08, relative_std=0.4, lo=0.01, hi=0.18)
                coop_penalty = self.model.model_gauss_sample(0.10, relative_std=0.4, lo=0.02, hi=0.25)
                other.reputation_fear = clamp01(other.reputation_fear + fear_boost)
                other.reputation_coop = clamp01(other.reputation_coop - coop_penalty)
                self._log_episode(other, self, "defect")
                self.model.record_behavior("defect", other, self, actor_outcome=0.9, target_outcome=-0.9)
                self.model.apply_decentralized_enforcement("defect", other, self)
                if getattr(self.model, "enable_guilt", False):
                    guilt_gain = self.model.model_gauss_sample(0.04, relative_std=0.5, lo=0.01, hi=0.1)
                    guilt_gain *= other.latent.get("empathy", 0.5) * other.latent.get("moral_prosocial", 0.5)
                    other._adjust_guilt(guilt_gain)
            elif my_action == "defect" and other_action == "coop":
                steal = self.model.model_gauss_sample(0.9, relative_std=0.1, lo=0.6, hi=1.1)
                self.wealth += steal
                other.wealth -= steal
                happy_boost = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
                happy_hit = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.2)
                self.happiness = clamp01(self.happiness + happy_boost)
                other.happiness = clamp01(other.happiness - happy_hit)
                self.neuroplasticity("violence_success")
                other.neuroplasticity("betrayed")
                neutral_trust = self.model.model_gauss_sample(0.0, std=0.02, lo=-0.05, hi=0.05)
                self.update_memory(other, "defect", trust_delta=neutral_trust)
                trust_hit = self.model.model_gauss_sample(0.12, relative_std=0.4, lo=0.03, hi=0.25)
                other.update_memory(self, "defect", trust_delta=-trust_hit)
                fear_boost = self.model.model_gauss_sample(0.08, relative_std=0.4, lo=0.01, hi=0.18)
                coop_penalty = self.model.model_gauss_sample(0.10, relative_std=0.4, lo=0.02, hi=0.25)
                self.reputation_fear = clamp01(self.reputation_fear + fear_boost)
                self.reputation_coop = clamp01(self.reputation_coop - coop_penalty)
                self._log_episode(self, other, "defect")
                self.model.record_behavior("defect", self, other, actor_outcome=0.9, target_outcome=-0.9)
                self.model.apply_decentralized_enforcement("defect", self, other)
                if getattr(self.model, "enable_guilt", False):
                    guilt_gain = self.model.model_gauss_sample(0.04, relative_std=0.5, lo=0.01, hi=0.1)
                    guilt_gain *= self.latent.get("empathy", 0.5) * self.latent.get("moral_prosocial", 0.5)
                    self._adjust_guilt(guilt_gain)
            else:
                happy_hit = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
                trust_hit = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
                self.happiness = clamp01(self.happiness - happy_hit)
                other.happiness = clamp01(other.happiness - happy_hit)
                self.update_memory(other, "defect", trust_delta=-trust_hit)
                other.update_memory(self, "defect", trust_delta=-trust_hit)
                self._log_episode(self, other, "standoff")
                self.model.record_behavior("standoff", self, other, actor_outcome=-0.1, target_outcome=-0.1)

            # psicópata estratégico: simulación de empatía para reputación cooperativa
            if self.latent.get("reasoning", 0.5) > 0.8 and self.latent.get("empathy", 0.5) < 0.3:
                mask_boost = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
                self.reputation_coop = clamp01(self.reputation_coop + mask_boost)
            if other.latent.get("reasoning", 0.5) > 0.8 and other.latent.get("empathy", 0.5) < 0.3:
                mask_boost = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
                other.reputation_coop = clamp01(other.reputation_coop + mask_boost)
        if raw_neighbors:
            self._perform_community_service(raw_neighbors)

    def gossip(
        self,
        neighbors: List["Citizen"] | None = None,
        target: "Citizen" | None = None,
        force_negative: bool = False,
    ):
        """
        Gossip about a third party when it benefits the gossiper (or a high-reputation influencer).

        GAUSSIANIZED: convenience thresholds, weights, and intensity vary per use.
        """
        if not self.alive:
            return
        if neighbors is None:
            neighbors = self.model.get_social_neighbors(self, radius=2, include_center=False)
        if len(neighbors) < 2 and target is None:
            return
        listener = self.model.rng.choice(neighbors) if neighbors else None
        if target is None:
            candidates = [n for n in neighbors if n is not listener]
            if not candidates:
                return
            target = self.model.rng.choice(candidates)

        # Alignment and motive determination.
        aligned = bool(self.alliance_id and self.alliance_id == getattr(target, "alliance_id", None))
        dark_mach = self.latent.get("dark_mach", 0.0)
        empathy = self.latent.get("empathy", 0.5)
        moral_prosocial = self.latent.get("moral_prosocial", 0.5)
        benefit_bias = empathy + moral_prosocial - dark_mach
        bias_thr = self.model.model_gauss_sample(0.1, relative_std=0.5, lo=-0.1, hi=0.3)

        benefit = aligned or (benefit_bias > bias_thr)
        if dark_mach > self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85) and self.wealth < target.wealth:
            benefit = False
        if force_negative:
            benefit = False

        # Convenience score determines whether gossip is executed.
        w_wealth = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.1, hi=0.7)
        w_rep = self.model.model_gauss_sample(0.3, relative_std=0.3, lo=0.05, hi=0.6)
        w_emp = self.model.model_gauss_sample(0.3, relative_std=0.3, lo=0.05, hi=0.6)
        convenience_score = (
            w_wealth * (self.wealth - target.wealth)
            + w_rep * (self.reputation_coop - target.reputation_coop)
            + w_emp * empathy * (1.0 if benefit else -1.0)
        )
        conv_thr = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.45)
        if convenience_score <= conv_thr:
            return

        # Influence intensity increases if the gossiper has higher cooperative reputation.
        base_intensity = self.model.model_gauss_sample(0.04, relative_std=0.5, lo=0.01, hi=0.12)
        rep_adv = max(0.0, self.reputation_coop - (listener.reputation_coop if listener else 0.0))
        rep_amp = 1.0 + self.model.model_gauss_sample(0.8, relative_std=0.3, lo=0.3, hi=1.3) * rep_adv
        intensity = base_intensity * rep_amp

        if benefit:
            target.reputation_coop = clamp01(target.reputation_coop + intensity)
            fear_drop = self.model.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
            target.reputation_fear = clamp01(target.reputation_fear - fear_drop * intensity)
            self.model.step_events["positive_gossip"] += 1.0
            self.model.record_behavior("gossip_positive", self, target, +intensity, +intensity)
        else:
            target.reputation_coop = clamp01(target.reputation_coop - intensity)
            fear_boost = self.model.model_gauss_sample(0.02, relative_std=0.6, lo=0.005, hi=0.06)
            target.reputation_fear = clamp01(target.reputation_fear + fear_boost * intensity)
            # Opportunistic gain for dark Machiavellians.
            if dark_mach > self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85):
                gain = self.model.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.12)
                self.wealth += gain
            self.model.step_events["negative_gossip"] += 1.0
            self.model.record_behavior("gossip_negative", self, target, -intensity, -intensity)

        # Small cost to gossiper to reflect effort/risk.
        cost = self.model.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
        self.happiness = clamp01(self.happiness - cost)
        self.model.step_events["gossip_total"] += 1.0

    def move(self):
        if not self.alive:
            return
        # GAUSSIANIZED: movement desire weights.
        impulsivity = self.latent.get("impulsivity", 0.5)
        attn_flex = self.latent.get("attn_flex", 0.5)
        base_desire = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        w_imp = self.model.model_gauss_sample(0.5, relative_std=0.25, lo=0.2, hi=0.8)
        w_happy = self.model.model_gauss_sample(0.4, relative_std=0.25, lo=0.2, hi=0.7)
        w_attn = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
        w_wealth = self.model.model_gauss_sample(0.3, relative_std=0.3, lo=0.1, hi=0.6)
        desire = base_desire + impulsivity * w_imp + max(0.0, w_happy - self.happiness) + w_attn * attn_flex
        desire -= w_wealth * min(1.0, max(0.0, self.wealth))
        if self.model.rng.random() > desire:
            return
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empties = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if empties:
            idx = int(self.model.rng.integers(len(empties)))
            self.model.grid.move_agent(self, tuple(empties[idx]))

    def production_and_consumption(self):
        # GAUSSIANIZED: production/consumption coefficients.
        reasoning = self.latent.get("reasoning", 0.5)
        hyperfocus = self.latent.get("hyperfocus", 0.5)
        base_prod = self.model.model_gauss_sample(1.0, relative_std=0.15, lo=0.6, hi=1.5)
        w_reason = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        w_focus = self.model.model_gauss_sample(0.3, relative_std=0.25, lo=0.1, hi=0.6)
        prod = base_prod + w_reason * reasoning + w_focus * hyperfocus
        prod *= self.model.production_multiplier
        self.wealth += prod * self.model.scale_factor
        cons_base = self.model.model_gauss_sample(0.8, relative_std=0.2, lo=0.4, hi=1.2)
        cons_w = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        consumption = cons_base + cons_w * (1 - reasoning)
        self.wealth -= consumption
        death_threshold = self.model.model_gauss_sample(-1.5, std=0.3, lo=-3.0, hi=-0.5)
        if self.wealth < death_threshold:
            self.alive = False

    def epidemic_effect(self):
        if self.model.external_factor != "epidemic":
            return
        # GAUSSIANIZED: epidemic risk.
        affect_reg = self.latent.get("affect_reg", 0.5)
        risk_base = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
        risk = risk_base * (1 - affect_reg)
        if self.model.rng.random() < risk:
            self.alive = False

    def step(self):
        if not self.alive:
            return
        self._decay_neurochem()
        self.age += 1
        self.production_and_consumption()
        # reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        if self.gestation_timer > 0 and self.gender == "Female":
            self.gestation_timer -= 1
            if self.gestation_timer == 0:
                self._give_birth()
        if self.gender == "Female":
            if self.fertility_cooldown > 0:
                self.fertility_cooldown -= 1
            self.attempt_mating()
        else:
            self.male_initiation()
            self._male_competition()
        if self.bonding_timer > 0:
            self._bonding_tick()
        if not self.alive:
            return
        self.epidemic_effect()
        if not self.alive:
            return
        mortality_base = self.model.model_gauss_sample(0.03, relative_std=0.5, lo=0.005, hi=0.08)
        age_limit = int(self.model.model_gauss_sample(320, std=40, lo=200, hi=450, integer=True))
        if self.model.rng.random() < mortality_base * self.model.mortality_multiplier or self.age > age_limit:
            self.alive = False
            return
        self.move()
        if self.ostracism_timer > 0:
            self.ostracism_timer -= 1
            penalty = self.model.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
            self.happiness = clamp01(self.happiness - penalty)
            return
        self.interact()
        # Gossip after direct interaction to influence third-party reputations.
        self.gossip()
        if self.model.innovation_boost > 0:
            self.latent["reasoning"] = clamp01(self.latent.get("reasoning", 0.5) + self.model.innovation_boost)
        self._reflective_cycle()
        if self.health <= 0:
            self.alive = False

    def _maybe_apply_rare_variant(self) -> Dict[str, object] | None:
        if not RARE_VARIANTS:
            return None
        pick = None
        for variant in RARE_VARIANTS:
            if self.model.rng.random() < variant.get("probability", 0.0):
                pick = variant
                break
        return pick

    def _init_conscious_core(self) -> Dict[str, object]:
        base = self.model.model_gauss_sample(0.4, relative_std=0.2, lo=0.2, hi=0.7)
        w_lang = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        w_reason = self.model.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        w_soc = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.4)
        awareness = clamp01(base + w_lang * self.latent.get("language", 0.5) + w_reason * self.latent.get("reasoning", 0.5) + w_soc * self.latent.get("sociality", 0.5))
        if self.rare_variant:
            awareness = clamp01(awareness + self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12))
        goals = []
        if self.latent.get("reasoning", 0.5) > self.model.model_gauss_sample(0.65, relative_std=0.2, lo=0.4, hi=0.85):
            goals.append("orden_formal")
        if self.latent.get("empathy", 0.5) > self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85):
            goals.append("cuidado_comunidad")
        if self.latent.get("impulsivity", 0.5) > self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85):
            goals.append("innovacion_caotica")
        if self.latent.get("hyperfocus", 0.5) > self.model.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85):
            goals.append("precision_leibniziana")
        narrative = f"{getattr(self, 'profile_id', 'anon')}|{self.rare_variant['name'] if self.rare_variant else 'base'}"
        return {
            "awareness": awareness,
            "perception_map": {"bias": {}, "last": None},
            "self_model": {"agency": awareness, "goals": goals, "metacognition": clamp01(self.latent.get("reasoning", 0.5))},
            # Vector-based episodic memory store for similarity retrieval.
            "memory_store": [],
            "imagination_buffer": [],
            "identity_narrative": narrative,
        }

    def _update_conscious_perception(self, other: "Citizen", perceived_threat: float):
        bias = {
            "threat": perceived_threat,
            "detail": self.latent.get("attn_selective", 0.5),
            "social": self.latent.get("sociality", 0.5),
        }
        base_collapse = self.model.model_gauss_sample(0.02, relative_std=0.6, lo=0.005, hi=0.06)
        w_focus = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
        w_res = self.model.model_gauss_sample(0.02, relative_std=0.6, lo=0.005, hi=0.06)
        collapse_p = clamp01(base_collapse + w_focus * self.latent.get("hyperfocus", 0.5) - w_res * self.latent.get("resilience", 0.5))
        if self.rare_variant:
            collapse_p += self.model.model_gauss_sample(0.03, relative_std=0.5, lo=0.005, hi=0.08)
        if self.model.rng.random() < collapse_p:
            bias["collapse"] = clamp01(self.model.rng.random())
        self.conscious_core["perception_map"] = {"bias": bias, "last": getattr(other, "unique_id", None)}

    def _interpretation_style(self) -> str:
        if self.latent.get("reasoning", 0.5) >= self.latent.get("emotional_impulsivity", 0.5):
            return "logico"
        return "afectivo"

    def _adjust_guilt(self, delta: float):
        self.guilt = clamp01(self.guilt + delta)
        self.latent["guilt"] = self.guilt

    def _adjust_shame(self, delta: float):
        self.shame = clamp01(self.shame + delta)
        self.latent["shame"] = self.shame

    def _memory_embed(self, text: str, dim: int = 64) -> np.ndarray:
        """Create a lightweight semantic embedding via hashed token bins."""
        vec = np.zeros(dim, dtype=float)
        tokens = [t for t in re.split(r"[^a-z0-9_]+", text.lower()) if t]
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).hexdigest()
            idx = int(h, 16) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _memory_store_add(self, actor: "Citizen", target: "Citizen", label: str):
        memory = self.conscious_core["memory_store"]
        max_mem = int(self.model.model_gauss_sample(60, std=15, lo=30, hi=120, integer=True))
        trim_n = int(self.model.model_gauss_sample(10, std=3, lo=5, hi=20, integer=True))
        if len(memory) > max_mem:
            del memory[:trim_n]
        reason_thr = self.model.model_gauss_sample(0.65, relative_std=0.2, lo=0.4, hi=0.85)
        interp = "rigido" if self.latent.get("reasoning", 0.5) > reason_thr else ("visionario" if self.rare_variant else "contextual")
        interpretation = f"{self._interpretation_style()}_{interp}"
        text = f"{label} {interpretation}"
        memory.append(
            {
                "actor": actor.unique_id,
                "target": target.unique_id,
                "event": label,
                "interpretation": interpretation,
                "embedding": self._memory_embed(text),
                "step": int(getattr(self.model, "step_count", 0)),
            }
        )

    def retrieve_relevant_memory(self, context_query: str, top_k: int = 3) -> List[Dict[str, object]]:
        """
        Retrieve episodic memories by semantic similarity (RAG-style).
        """
        memory = self.conscious_core.get("memory_store", [])
        if not memory:
            return []
        query_vec = self._memory_embed(context_query)
        if np.linalg.norm(query_vec) == 0:
            return []
        scored = []
        for mem in memory:
            emb = mem.get("embedding")
            if emb is None:
                continue
            sim = float(np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-9))
            recency = 1.0 / (1.0 + max(0, int(getattr(self.model, "step_count", 0)) - int(mem.get("step", 0))))
            score = sim * (0.7 + 0.3 * recency)
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def _memory_valence(self, event: str) -> float:
        """Map memory events to a coarse valence in [-1, 1]."""
        positive = {"coop", "support"}
        negative = {"violence", "violence_fail", "defect", "attack_success", "attack_fail"}
        if event in positive:
            return 1.0
        if event in negative:
            return -1.0
        return 0.0

    def _log_episode(self, actor: "Citizen", target: "Citizen", label: str):
        self._memory_store_add(actor, target, label)

    def _imagine_outcome(self, other: "Citizen") -> float:
        buf = self.conscious_core["imagination_buffer"]
        max_buf = int(self.model.model_gauss_sample(30, std=8, lo=15, hi=60, integer=True))
        trim_n = int(self.model.model_gauss_sample(10, std=3, lo=4, hi=18, integer=True))
        if len(buf) > max_buf:
            del buf[:trim_n]
        vision = self.model.rng.normal(0, self.model.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.12))
        vision_p = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
        vision_boost = self.model.model_gauss_sample(0.15, relative_std=0.4, lo=0.03, hi=0.35)
        if self.rare_variant and self.rare_variant.get("imagination_boost", 0.0) > 0 and self.model.rng.random() < vision_p:
            vision += vision_boost
            buf.append({"type": "vision", "target": other.unique_id, "delta": vision})
        else:
            buf.append({"type": "project", "target": other.unique_id, "delta": vision})
        return vision

    def _reflective_cycle(self):
        awareness = self.conscious_core["awareness"]
        aw_weight = self.model.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.12)
        awareness = clamp01(awareness + aw_weight * (self.latent.get("language", 0.5) + self.latent.get("affect_reg", 0.5) - 0.5))
        if self.rare_variant and self.rare_variant.get("chaos_innovation", 0.0) > 0:
            chaos_boost = self.model.model_gauss_sample(0.03, relative_std=0.5, lo=0.005, hi=0.08)
            awareness = clamp01(awareness + chaos_boost)
        self.conscious_core["awareness"] = awareness
        reason_thr = self.model.model_gauss_sample(0.7, relative_std=0.2, lo=0.4, hi=0.85)
        emp_thr = self.model.model_gauss_sample(0.7, relative_std=0.2, lo=0.4, hi=0.85)
        if self.latent.get("reasoning", 0.5) > reason_thr and self.last_action == "violence":
            imp_drop = self.model.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
            self.latent["impulsivity"] = clamp01(self.latent.get("impulsivity", 0.5) - imp_drop)
        if self.latent.get("empathy", 0.5) > emp_thr and self.last_action == "violence":
            reg_boost = self.model.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
            self.latent["affect_reg"] = clamp01(self.latent.get("affect_reg", 0.5) + reg_boost)
        if self.last_action == "violence":
            health_hit = self.model.model_gauss_sample(0.02, relative_std=0.6, lo=0.005, hi=0.06)
            self.health = clamp01(self.health - health_hit)
        focus_thr = self.model.model_gauss_sample(0.8, relative_std=0.2, lo=0.5, hi=0.95)
        reason_thr2 = self.model.model_gauss_sample(0.8, relative_std=0.2, lo=0.5, hi=0.95)
        if self.latent.get("hyperfocus", 0.5) > focus_thr and self.latent.get("reasoning", 0.5) > reason_thr2:
            inv_p = self.model.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
            if self.model.rng.random() < inv_p * self.latent.get("reasoning", 0.5):
                inv_scale = self.model.model_gauss_sample(5.0, relative_std=0.3, lo=2.0, hi=9.0)
                invention_value = self.latent.get("reasoning", 0.5) * inv_scale
                self.model.total_wealth += invention_value
                rep_boost = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
                self.reputation_coop = clamp01(self.reputation_coop + rep_boost * self.latent.get("language", 0.5))

    def _calculate_malice(self, other: "Citizen") -> bool:
        spite = self.latent.get("moral_spite", 0.0)
        envy_trigger = 1.0 if other.wealth > self.wealth else 0.0
        prob_sabotage = spite * envy_trigger * self.latent.get("impulsivity", 0.5)
        return self.model.rng.random() < prob_sabotage

    def _perform_community_service(self, neighbors: List["Citizen"]):
        if self.wealth <= 0 or not neighbors:
            return
        altruism_score = clamp01(self.latent.get("moral_common_good", 0.0) * (1.0 - self.latent.get("dark_mach", 0.0)))
        if self.model.rng.random() < altruism_score:
            donation_rate = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.15)
            donation = self.wealth * donation_rate * altruism_score
            poorest = min(neighbors, key=lambda a: a.wealth)
            self.wealth -= donation
            poorest.wealth += donation
            happy_boost = self.model.model_gauss_sample(0.02, relative_std=0.6, lo=0.005, hi=0.06)
            self.happiness = clamp01(self.happiness + happy_boost * altruism_score)

    def get_perceived_status(self) -> float:
        rep_weight = self.model.model_gauss_sample(10.0, relative_std=0.2, lo=5.0, hi=15.0)
        base_status = self.wealth + rep_weight * self.reputation_total()
        infl_base = self.model.model_gauss_sample(1.0, relative_std=0.1, lo=0.7, hi=1.3)
        infl_weight = self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.1, hi=0.9)
        inflation = infl_base + infl_weight * self.latent.get("dark_narc", 0.0)
        return base_status * inflation

    def _calculate_trade_efficiency(self, other: "Citizen") -> float:
        avg_language = (self.latent.get("language", 0.5) + other.latent.get("language", 0.5)) / 2.0
        base = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        weight = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        total = base + weight
        if total > 0:
            base, weight = base / total, weight / total
        return base + weight * avg_language

    def _attack(self, target: "Citizen"):
        # GAUSSIANIZED: attack thresholds and effects (non-lethal by default).
        wealth_drive = clamp01(self.wealth / 2.0)
        expected_gain = target.wealth * self.model.model_gauss_sample(0.4, relative_std=0.25, lo=0.2, hi=0.7)
        w_dark = self.model.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        w_wealth = self.model.model_gauss_sample(0.2, relative_std=0.25, lo=0.05, hi=0.5)
        risk_factor = clamp01((1.0 - self.latent.get("risk_aversion", 0.5)) + self.latent.get("dark_psycho", 0.0) * w_dark + wealth_drive * w_wealth)
        spite_threshold = self.model.model_gauss_sample(0.8, relative_std=0.15, lo=0.6, hi=0.95)
        spite_attack = self.latent.get("moral_spite", 0.0) > spite_threshold and target.wealth > self.wealth
        if self.model.rng.random() < clamp01(risk_factor) or spite_attack:
            w_aggr = self.model.model_gauss_sample(0.7, relative_std=0.2, lo=0.4, hi=0.9)
            w_res = self.model.model_gauss_sample(0.3, relative_std=0.25, lo=0.1, hi=0.6)
            attack_success_chance = clamp01(self.latent.get("aggression", 0.5) * w_aggr - target.latent.get("resilience", 0.5) * w_res)
            if self.model.rng.random() < attack_success_chance:
                steal_rate = self.model.model_gauss_sample(0.5, relative_std=0.25, lo=0.2, hi=0.8)
                stolen_wealth = target.wealth * steal_rate
                self.wealth += stolen_wealth
                target.wealth -= stolen_wealth
                fear_boost = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
                risk_drop = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
                self.reputation_fear = clamp01(self.reputation_fear + fear_boost)
                self.latent["risk_aversion"] = clamp01(self.latent.get("risk_aversion", 0.5) - risk_drop)

                # Non-lethal default penalties on victim.
                wealth_hit = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.1, hi=0.3)
                rep_drop = self.model.model_gauss_sample(0.1, relative_std=0.5, lo=0.05, hi=0.15)
                target.wealth -= wealth_hit
                target.reputation_coop = clamp01(target.reputation_coop - rep_drop)

                # Rare lethality gate (extreme traits + permissive norms).
                aggr = self.model.model_gauss_sample(self.latent.get("aggression", 0.5), relative_std=0.2, lo=0.0, hi=1.0)
                dom = self.model.model_gauss_sample(self.latent.get("dominance", 0.5), relative_std=0.2, lo=0.0, hi=1.0)
                res = self.model.model_gauss_sample(target.latent.get("resilience", 0.5), relative_std=0.2, lo=0.0, hi=1.0)
                emp = self.model.model_gauss_sample(target.latent.get("empathy", 0.5), relative_std=0.2, lo=0.0, hi=1.0)
                extreme = max(0.0, aggr + dom - res - emp)
                lethality_base = self.model.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.05)
                lethality_prob = clamp01(lethality_base * extreme * max(0.1, self.model.lethality_multiplier))
                extreme_thr = self.model.model_gauss_sample(0.8, relative_std=0.3, lo=0.6, hi=0.95)
                allow_gate = False
                formal_thr = self.model.model_gauss_sample(0.7, relative_std=0.2, lo=0.5, hi=0.9)
                leg_thr = self.model.model_gauss_sample(0.4, relative_std=0.3, lo=0.2, hi=0.6)
                allow_gate = (
                    self.model.legal_formalism > formal_thr
                    and self.model.political_system.participation_structure in ("tyranny", "oligarchy")
                    and self.model.political_system.legitimacy < leg_thr
                )
                allow_gate = allow_gate and self.model.rng.random() < self.model.model_gauss_sample(0.5, std=0.2, lo=0.2, hi=0.8)
                pop_guard = self.model.model_gauss_sample(40, std=10, lo=15, hi=70)
                if len(self.model.agents_alive()) < pop_guard:
                    allow_gate = False
                if allow_gate and extreme > extreme_thr and self.model.rng.random() < lethality_prob:
                    target.alive = False
                    self.model.step_events["violence_deaths"] += 1.0

                # Post-attack gossip and anti-aggressor alliance triggers.
                neighbors = self.model.get_social_neighbors(target, radius=2, include_center=True)
                gossip_p = self.model.model_gauss_sample(0.5, relative_std=0.4, lo=0.1, hi=0.9)
                if target.alive and self.model.rng.random() < gossip_p:
                    target.gossip(neighbors=neighbors, target=self, force_negative=True)
                alliance_p = self.model.model_gauss_sample(0.3, relative_std=0.4, lo=0.05, hi=0.7)
                if target.alive and self.model.rng.random() < alliance_p:
                    target._form_punishment_alliance(self)

                self._log_episode(self, target, "attack_success")
                return "SUCCESS"
            else:
                health_hit = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.3)
                fear_drop = self.model.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
                self.health = clamp01(self.health - health_hit * self.latent.get("aggression", 0.5))
                self.reputation_fear = clamp01(self.reputation_fear - fear_drop)
                self._log_episode(self, target, "attack_fail")
                return "FAILED"
        return "NO_ATTACK"

    def _form_punishment_alliance(self, target: "Citizen"):
        # GAUSSIANIZED: coalition sizes, penalties, and rewards.
        neighbors = self.model.get_social_neighbors(self, radius=2, include_center=False)
        potential_allies = [
            n
            for n in neighbors
            if isinstance(n, Citizen)
            and n.alive
            and n is not self
            and self.model.rng.random() < clamp01(n.latent.get("empathy", 0.5) * (1.0 - n.latent.get("dark_psycho", 0.0)))
        ]
        if not potential_allies:
            return
        size_rate = self.model.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.8)
        size = max(1, int(len(potential_allies) * size_rate))
        alliance_members = list(self.model.rng.choice(potential_allies, size, replace=False))
        alliance_members.append(self)
        total_coop_power = sum(a.latent.get("empathy", 0.5) * a.latent.get("language", 0.5) for a in alliance_members)
        violence_mult = self.model.model_gauss_sample(1.5, relative_std=0.2, lo=0.8, hi=2.2)
        target_violence_power = target.latent.get("dominance", 0.5) * target.latent.get("aggression", 0.5) * violence_mult
        success_p = total_coop_power / (total_coop_power + target_violence_power + 1e-6)
        if self.model.rng.random() < success_p:
            steal_rate = self.model.model_gauss_sample(0.6, relative_std=0.25, lo=0.3, hi=0.9)
            stolen_wealth = max(0.0, target.wealth * steal_rate)
            target.wealth -= stolen_wealth
            self.model.redistribute_wealth_to_allies(stolen_wealth, alliance_members)
            fear_drop = self.model.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.4)
            target.reputation_fear = clamp01(target.reputation_fear - fear_drop)
            for ally in alliance_members:
                coop_boost = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
                ally.reputation_coop = clamp01(ally.reputation_coop + coop_boost)
            self.model.log_event("CASTIGO_EXITOSO", getattr(target, "profile_id", "n/a"))
        else:
            self.model.log_event("CASTIGO_FALLIDO", getattr(target, "profile_id", "n/a"))
            for ally in alliance_members:
                health_hit = self.model.model_gauss_sample(0.3, relative_std=0.35, lo=0.1, hi=0.6)
                wealth_hit = self.model.model_gauss_sample(0.3, relative_std=0.35, lo=0.1, hi=0.6)
                rep_drop = self.model.model_gauss_sample(0.1, relative_std=0.4, lo=0.02, hi=0.25)
                ally.health = clamp01(ally.health - health_hit)
                ally.wealth -= ally.wealth * wealth_hit
                ally.reputation_coop = clamp01(ally.reputation_coop - rep_drop)


class SocietyModel(Model):
    POP_SCALES = {"tiny": 10, "tribe": 1000, "city": 100000, "nation": 10000000}

    def __init__(
        self,
        seed: int | None = None,
        climate: str = "stable",
        external_factor: str = "none",
        population_scale: str = "tribe",
        profile1: str | None = None,
        profile2: str | None = None,
        profile3: str | None = None,
        weight1: float = 0.6,
        weight2: float = 0.3,
        weight3: float = 0.1,
        jitter: float = 0.05,
        spectrum_level: int | None = None,
        initial_moral_bias: str | None = None,
        resilience_bias: str | None = None,
        emotional_bias: str | None = None,
        enable_reproduction: bool = True,
        enable_sexual_selection: bool = True,
        male_violence_multiplier: float = 1.2,
        female_violence_multiplier: float = 0.35,
        female_target_protection: float = 0.6,
        coalition_enabled: bool = True,
        coalition_power_weight: float = 0.6,
        coalition_strategy_weight: float = 0.4,
        sneaky_strategy_enabled: bool = True,
        sneaky_success_weight: float = 0.6,
        reproduction_costs: float = 0.3,
        resource_constraint: float = 0.4,
        mate_weight_wealth: float = 0.4,
        mate_weight_dom: float = 0.3,
        mate_weight_health: float = 0.2,
        mate_weight_age: float = 0.1,
        mate_choice_beta: float = 1.0,
        female_repro_cooldown: int = 10,
        male_repro_cooldown: int = 2,
        repro_base_offset: float = 0.2,
        repro_desire_scale: float = 0.3,
        male_initiation_base: float = 0.05,
        male_desire_scale: float = 0.3,
        neuro_decay_k: float = 0.1,
        bonding_steps: int = 5,
        bonding_delta: float = 0.02,
        enable_coercion: bool = False,
        interaction_topology: str = "grid_local",
        topology_params: Dict[str, float] | None = None,
        interaction_radius: int = 1,
        enable_cultural_transmission: bool = False,
        cultural_learning_rate: float = 0.05,
        imitation_bias: str = "prestige",
        conformity_bias: float = 0.2,
        innovation_rate: float = 0.02,
        enable_guilt: bool = False,
        enable_ostracism: bool = False,
        enable_fermi_update: bool = False,
        fermi_beta: float = 1.0,
        calibrate_traits: bool = False,
        calibration_data_paths: List[str] | None = None,
        calibration_mapping_path: str = "calibration/mapping.json",
        policy_mode: str = "none",
        policy_schedule: Dict[int, str] | None = None,
        **kwargs,
    ):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.climate = climate
        self.external_factor = external_factor
        self.stochasticity_level = 1.0
        self.spectrum_level = spectrum_level
        self.initial_moral_bias = (initial_moral_bias or "").strip().lower() or None
        self.resilience_bias = (resilience_bias or "").strip().lower() or None
        self.emotional_bias = (emotional_bias or "").strip().lower() or None
        self.total_wealth = 0.0
        self.nd_contribution_log: Dict[str, float] = {}
        self.population_scale_key = population_scale
        self.profile_weights = {
            "profile1": (profile1 or "").strip(),
            "profile2": (profile2 or "").strip(),
            "profile3": (profile3 or "").strip(),
            "weight1": float(weight1),
            "weight2": float(weight2),
            "weight3": float(weight3),
        }
        w_sum = self.profile_weights["weight1"] + self.profile_weights["weight2"] + self.profile_weights["weight3"]
        self.weight_warning = False
        self.weight_sum_original = w_sum
        if w_sum <= 0:
            self.profile_weights["weight1"] = 1.0
            self.profile_weights["weight2"] = 0.0
            self.profile_weights["weight3"] = 0.0
            self.weight_warning = True
        elif abs(w_sum - 1.0) > 1e-6:
            self.profile_weights["weight1"] /= w_sum
            self.profile_weights["weight2"] /= w_sum
            self.profile_weights["weight3"] /= w_sum
            self.weight_warning = True

        target = self.POP_SCALES.get(population_scale, 1000)
        self.max_agents = 5000
        self.actual_agents = min(target, self.max_agents)
        self.scale_factor = target / self.actual_agents if self.actual_agents else 1.0
        self.institution_pressure = min(1.0, math.log10(target) / 7.0)
        self.initial_population_scaled = self.actual_agents * self.scale_factor

        side = max(8, int(math.sqrt(self.actual_agents) * 1.4))
        self.grid = MultiGrid(side, side, torus=True)
        self.alliances: Dict[str, Dict[str, object]] = {}
        self.step_count = 0
        self.behavior_memory: List[Dict[str, object]] = []
        self.legal_system = LegalSystem(formalism_index=0.5)
        self.political_system = PoliticalSystem()
        self.norm_count = 0
        self.norm_density = 0.0
        self.norm_consistency = 1.0
        self.norm_centralization = 0.0
        self.economic_mechanism = "mixed"
        self.power_concentration = 0.0
        self.conflict_resolution = "consensus"
        self.moral_framework = "prosocial"
        self.cognitive_profile_top20: Dict[str, float] = {}

        self.production_multiplier = self._compute_production_multiplier()
        self.mortality_multiplier = self._compute_mortality_multiplier()
        self.innovation_boost = 0.001 if self.external_factor == "technological" else 0.0
        self.recent_violence_events: List[Tuple[int, int]] = []
        self.enable_reproduction = bool(enable_reproduction)
        self.enable_sexual_selection = bool(enable_sexual_selection)
        self.male_violence_multiplier = float(male_violence_multiplier)
        self.female_violence_multiplier = float(female_violence_multiplier)
        self.female_target_protection = float(female_target_protection)
        self.coalition_enabled = bool(coalition_enabled)
        self.coalition_power_weight = float(coalition_power_weight)
        self.coalition_strategy_weight = float(coalition_strategy_weight)
        self.sneaky_strategy_enabled = bool(sneaky_strategy_enabled)
        self.sneaky_success_weight = float(sneaky_success_weight)
        self.reproduction_costs = float(reproduction_costs)
        self.resource_constraint = clamp01(float(resource_constraint))
        self.mate_weight_wealth = float(mate_weight_wealth)
        self.mate_weight_dom = float(mate_weight_dom)
        self.mate_weight_health = float(mate_weight_health)
        self.mate_weight_age = float(mate_weight_age)
        self.mate_choice_beta = float(mate_choice_beta)
        self.female_repro_cooldown = max(1, int(female_repro_cooldown))
        self.male_repro_cooldown = max(1, int(male_repro_cooldown))
        self.repro_base_offset = float(repro_base_offset)
        self.repro_desire_scale = float(repro_desire_scale)
        self.male_initiation_base = float(male_initiation_base)
        self.male_desire_scale = float(male_desire_scale)
        self.neuro_decay_k = clamp01(float(neuro_decay_k))
        self.bonding_steps = max(0, int(bonding_steps))
        self.bonding_delta = float(bonding_delta)
        self.enable_coercion = bool(enable_coercion)
        self.births_total = 0
        self.step_events: Dict[str, float] = {}
        self.interaction_topology = interaction_topology
        self.topology_params = topology_params or {}
        self.interaction_radius = max(1, int(interaction_radius))
        self.social_graph = None
        self._agent_by_id: Dict[int, Citizen] = {}
        self.enable_cultural_transmission = bool(enable_cultural_transmission)
        self.cultural_learning_rate = float(cultural_learning_rate)
        self.imitation_bias = str(imitation_bias)
        self.conformity_bias = float(conformity_bias)
        self.innovation_rate = float(innovation_rate)
        self.enable_guilt = bool(enable_guilt)
        self.enable_ostracism = bool(enable_ostracism)
        self.enable_fermi_update = bool(enable_fermi_update)
        self.fermi_beta = float(fermi_beta)
        self.calibrate_traits = bool(calibrate_traits)
        self.calibration_data_paths = calibration_data_paths or []
        self.calibration_mapping_path = calibration_mapping_path
        self.calibration_data = None
        self.calibration_report: Dict[str, float] = {}
        self.policy_mode = str(policy_mode)
        self.policy_schedule = policy_schedule or {}
        self.enforcement_multiplier = 1.0
        self.reward_multiplier = 1.0

        if self.calibrate_traits and load_calibration_data is not None:
            self.calibration_data = load_calibration_data(self.calibration_data_paths, self.calibration_mapping_path)
            if self.calibration_data and calibration_quality_report is not None:
                self.calibration_report = calibration_quality_report(self.calibration_data, self.calibration_data_paths)

        for _ in range(self.actual_agents):
            latent, bias_ranges, spectrum_ranges, profile_id = self._compose_latent(jitter=jitter)
            agent = Citizen(
                self,
                latent,
                bias_ranges=bias_ranges,
                spectrum_ranges=spectrum_ranges,
                spectrum_level=self.spectrum_level,
            )
            agent.profile_id = profile_id or "unknown"
            agent.conscious_core["identity_narrative"] = f"{agent.profile_id or 'anon'}|{agent.rare_variant['name'] if agent.rare_variant else 'base'}"
            self.agents.add(agent)
            self.grid.place_agent(agent, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))
            self._agent_by_id[agent.unique_id] = agent

        # Optional social interaction topology (separate from spatial grid).
        if self.interaction_topology != "grid_local":
            if nx is None or build_social_graph is None:
                # Fallback to grid_local if networkx is unavailable.
                self.interaction_topology = "grid_local"
            else:
                base_graph = build_social_graph(
                    self.interaction_topology, len(self._agent_by_id), self.rng, params=self.topology_params
                )
                # Relabel nodes to agent unique_ids for stable lookup.
                id_map = {i: uid for i, uid in enumerate(self._agent_by_id.keys())}
                self.social_graph = nx.relabel_nodes(base_graph, id_map)

        self.running = True
        self.total_wealth = sum(a.wealth for a in self.agents if isinstance(a, Citizen))
        self.regime = "Inicial"
        self.legal_formalism = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.liberty_index = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        self.gini_wealth = self.model_gauss_sample(0.0, std=0.05, lo=0.0, hi=0.2)
        self.lethality_multiplier = 1.0
        self.cultural_convergence = 0.0
        self.last_metrics: Dict[str, float] = {}
        self.run_metadata = {
            "seed": seed,
            "population_scale": population_scale,
            "interaction_topology": self.interaction_topology,
            "topology_params": self.topology_params,
            "enable_cultural_transmission": self.enable_cultural_transmission,
            "cultural_learning_rate": self.cultural_learning_rate,
            "imitation_bias": self.imitation_bias,
            "conformity_bias": self.conformity_bias,
            "innovation_rate": self.innovation_rate,
            "enable_guilt": self.enable_guilt,
            "enable_ostracism": self.enable_ostracism,
            "enable_fermi_update": self.enable_fermi_update,
            "fermi_beta": self.fermi_beta,
            "calibrate_traits": self.calibrate_traits,
            "calibration_loaded": bool(self.calibration_data),
            "policy_mode": self.policy_mode,
            "git_hash": self._git_hash(),
        }
        self.datacollector = DataCollector(
            model_reporters={
                "population": lambda m: len(m.agents_alive()) * m.scale_factor,
                "coop_rate": lambda m: m.last_metrics.get("coop_rate", 0.0),
                "violence_rate": lambda m: m.last_metrics.get("violence_rate", 0.0),
                "male_violence_rate": lambda m: m.last_metrics.get("male_violence_rate", 0.0),
                "female_violence_rate": lambda m: m.last_metrics.get("female_violence_rate", 0.0),
                "gini_wealth": lambda m: m.gini_wealth,
                "life_expectancy": lambda m: m.last_metrics.get("life_expectancy", 0.0),
                "legal_formalism": lambda m: m.legal_formalism,
                "liberty_index": lambda m: m.liberty_index,
                "regime": lambda m: m.regime,
                "avg_reasoning": lambda m: m.last_metrics.get("avg_reasoning", 0.0),
                "avg_empathy": lambda m: m.last_metrics.get("avg_empathy", 0.0),
                "avg_dominance": lambda m: m.last_metrics.get("avg_dominance", 0.0),
                "reputation_coop_mean": lambda m: float(np.mean([a.reputation_coop for a in m.agents_alive()])) if m.agents_alive() else 0.0,
                "reputation_fear_mean": lambda m: float(np.mean([a.reputation_fear for a in m.agents_alive()])) if m.agents_alive() else 0.0,
                "top5_wealth_share": lambda m: m._top5_power()[0],
                "total_wealth": lambda m: m.total_wealth,
                "nd_contribution_mean": lambda m: float(np.mean([a.nd_contribution for a in m.agents_alive()])) if m.agents_alive() else 0.0,
                "alliances_count": lambda m: len(m.alliances),
                "allied_share": lambda m: (sum(1 for a in m.agents_alive() if getattr(a, "alliance_id", None)) / len(m.agents_alive())) if m.agents_alive() else 0.0,
                "conscious_awareness_mean": lambda m: float(np.mean([a.conscious_core.get("awareness", 0.0) for a in m.agents_alive()])) if m.agents_alive() else 0.0,
                "births": lambda m: m.last_metrics.get("births", 0.0),
                "sex_ratio": lambda m: m.last_metrics.get("sex_ratio", 0.0),
                "coalition_count": lambda m: m.last_metrics.get("coalition_count", 0.0),
                "coalition_wins": lambda m: m.last_metrics.get("coalition_wins", 0.0),
                "sneaky_success_rate": lambda m: m.last_metrics.get("sneaky_success_rate", 0.0),
                "male_male_conflicts": lambda m: m.last_metrics.get("male_male_conflicts", 0.0),
                "female_indirect_competition": lambda m: m.last_metrics.get("female_indirect_competition", 0.0),
                "mating_inequality": lambda m: m.last_metrics.get("mating_inequality", 0.0),
                "mean_harem_size": lambda m: m.last_metrics.get("mean_harem_size", 0.0),
                "repro_gini_males": lambda m: m.last_metrics.get("repro_gini_males", 0.0),
                "male_childless_share": lambda m: m.last_metrics.get("male_childless_share", 0.0),
                "mean_partners_male": lambda m: m.last_metrics.get("mean_partners_male", 0.0),
                "mean_partners_female": lambda m: m.last_metrics.get("mean_partners_female", 0.0),
                "gossip_rate": lambda m: m.last_metrics.get("gossip_rate", 0.0),
                "positive_gossip_rate": lambda m: m.last_metrics.get("positive_gossip_rate", 0.0),
                "negative_gossip_rate": lambda m: m.last_metrics.get("negative_gossip_rate", 0.0),
                "lethality_rate": lambda m: m.last_metrics.get("lethality_rate", 0.0),
                "network_clustering": lambda m: m.last_metrics.get("network_clustering", 0.0),
                "network_avg_path_len": lambda m: m.last_metrics.get("network_avg_path_len", 0.0),
                "network_degree_mean": lambda m: m.last_metrics.get("network_degree_mean", 0.0),
                "network_degree_std": lambda m: m.last_metrics.get("network_degree_std", 0.0),
                "network_degree_gini": lambda m: m.last_metrics.get("network_degree_gini", 0.0),
                "cultural_convergence": lambda m: m.last_metrics.get("cultural_convergence", 0.0),
                "norm_count": lambda m: m.norm_count,
                "norm_density": lambda m: m.norm_density,
                "norm_consistency": lambda m: m.norm_consistency,
                "economic_mechanism": lambda m: m.economic_mechanism,
                "power_concentration": lambda m: m.power_concentration,
                "conflict_resolution": lambda m: m.conflict_resolution,
                "moral_framework": lambda m: m.moral_framework,
                "cognitive_profile_top20": lambda m: m.cognitive_profile_top20,
                "participation_structure": lambda m: m.political_system.participation_structure,
                "benefit_orientation": lambda m: m.political_system.benefit_orientation,
                "political_legitimacy": lambda m: m.political_system.legitimacy,
            }
        )
        self._update_metrics()
        self.datacollector.collect(self)

    def _reset_step_events(self):
        self.step_events = {
            "male_violence": 0.0,
            "female_violence": 0.0,
            "violence_events": 0.0,
            "violence_deaths": 0.0,
            "male_male_conflicts": 0.0,
            "female_indirect_competition": 0.0,
            "coalition_count": 0.0,
            "coalition_wins": 0.0,
            "sneaky_attempts": 0.0,
            "sneaky_success": 0.0,
            "mating_attempts": 0.0,
            "births": 0.0,
            "redistribution_rate": 0.0,
            "gossip_total": 0.0,
            "positive_gossip": 0.0,
            "negative_gossip": 0.0,
        }

    def model_gauss_sample(self, mean: float, **kwargs) -> float:
        """Gaussian sampler honoring stochasticity_level. # GAUSSIANIZED"""
        if "relative_std" in kwargs and kwargs["relative_std"] is not None:
            kwargs["relative_std"] = float(kwargs["relative_std"]) * self.stochasticity_level
        if "std" in kwargs and kwargs["std"] is not None:
            kwargs["std"] = float(kwargs["std"]) * self.stochasticity_level
        return gauss_sample(self.rng, mean, **kwargs)

    def record_behavior(self, behavior: str, actor: Citizen, target: Citizen, actor_outcome: float, target_outcome: float):
        """Store behavior outcomes for emergent norm detection."""
        actor.remember_behavior(behavior, actor_outcome)
        target.remember_behavior(behavior, target_outcome)
        self.behavior_memory.append(
            {
                "behavior": behavior,
                "actor": actor.unique_id,
                "target": target.unique_id,
                "outcome": float(actor_outcome + target_outcome) * 0.5,
            }
        )
        if len(self.behavior_memory) > 500:
            del self.behavior_memory[:100]

    def wealth_percentile(self, wealth: float) -> float:
        alive = self.agents_alive()
        if not alive:
            return 0.5
        vals = sorted([a.wealth for a in alive])
        idx = 0
        for v in vals:
            if wealth >= v:
                idx += 1
        return idx / max(1, len(vals))

    def get_social_neighbors(
        self,
        agent: Citizen,
        radius: int | None = None,
        include_center: bool = False,
    ) -> List[Citizen]:
        """Return neighbors from social topology if enabled, else spatial grid."""
        radius = self.interaction_radius if radius is None else max(1, int(radius))
        if self.interaction_topology == "grid_local" or self.social_graph is None:
            neighbors = self.grid.get_neighbors(agent.pos, moore=True, include_center=include_center, radius=radius)
            return [n for n in neighbors if isinstance(n, Citizen) and n.alive]
        if get_graph_neighbors is None:
            return []
        ids = get_graph_neighbors(self.social_graph, agent.unique_id, radius=radius)
        result = []
        for uid in ids:
            other = self._agent_by_id.get(uid)
            if other is None or not other.alive:
                continue
            if not include_center and other is agent:
                continue
            result.append(other)
        return result

    def _power_scores(self, alive: List[Citizen]) -> List[Tuple[int, float]]:
        scores = []
        for a in alive:
            wealth_p = self.wealth_percentile(a.wealth)
            rep = a.reputation_total()
            dom = a.latent.get("dominance", 0.5)
            alliance = 1.0 if getattr(a, "alliance_id", None) else 0.0
            w_w, w_r, w_d, w_a = gauss_weights(self.rng, [0.4, 0.3, 0.2, 0.1], relative_std=0.2, normalize=True)
            power = w_w * wealth_p + w_r * rep + w_d * dom + w_a * alliance
            scores.append((a.unique_id, power))
        return scores

    def get_elite_agents(self, percentile: float = 0.8) -> List[Citizen]:
        alive = self.agents_alive()
        if not alive:
            return []
        scores = self._power_scores(alive)
        scores.sort(key=lambda x: x[1], reverse=True)
        cutoff = max(1, int(len(scores) * (1.0 - percentile)))
        elite_ids = {uid for uid, _ in scores[:cutoff]}
        return [a for a in alive if a.unique_id in elite_ids]

    def calculate_formalism_index(self) -> float:
        """Estimate formalism vs casuistry based on elite traits and power concentration."""
        # GAUSSIANIZED: coefficients sampled per call.
        elite = self.get_elite_agents()
        if not elite:
            return 0.5
        reasoning = float(np.mean([a.latent.get("reasoning", 0.5) for a in elite]))
        empathy = float(np.mean([a.latent.get("empathy", 0.5) for a in elite]))
        moral_honesty = float(np.mean([a.latent.get("moral_honesty", 0.5) for a in elite]))
        dark_mach = float(np.mean([a.latent.get("dark_mach", 0.5) for a in elite]))
        sociality = float(np.mean([a.latent.get("sociality", 0.5) for a in elite]))
        power_gini = self.power_concentration
        w_reasoning = self.model_gauss_sample(0.4, relative_std=0.2, lo=0.05, hi=0.8)
        w_empathy = self.model_gauss_sample(0.3, relative_std=0.2, lo=0.05, hi=0.6)
        w_honesty = self.model_gauss_sample(0.2, relative_std=0.2, lo=0.05, hi=0.5)
        w_mach = self.model_gauss_sample(0.2, relative_std=0.2, lo=0.05, hi=0.5)
        w_power = self.model_gauss_sample(0.15, relative_std=0.2, lo=0.03, hi=0.4)
        w_sociality = self.model_gauss_sample(0.15, relative_std=0.2, lo=0.03, hi=0.4)
        raw = (
            w_reasoning * reasoning
            - w_empathy * empathy
            + w_honesty * moral_honesty
            - w_mach * dark_mach
            + w_power * (1.0 - power_gini)
            - w_sociality * sociality
        )
        base = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        return clamp01(base + raw)

    def generate_contextual_modifiers(self) -> Dict[str, float]:
        formalism = self.legal_system.formalism_index
        if formalism > self.model_gauss_sample(0.7, relative_std=0.15, lo=0.5, hi=0.9):
            return {}
        elite = self.get_elite_agents()
        modifiers: Dict[str, float] = {}
        if elite:
            avg_sociality = float(np.mean([a.latent.get("sociality", 0.5) for a in elite]))
            avg_dark = float(np.mean([a.dark_core for a in elite]))
            avg_empathy = float(np.mean([a.latent.get("empathy", 0.5) for a in elite]))
            if avg_sociality > self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8):
                modifiers["coalition_member"] = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.3, hi=0.9)
            if avg_dark > self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8):
                modifiers["elite_protection"] = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
            if avg_empathy > self.model_gauss_sample(0.7, relative_std=0.2, lo=0.5, hi=0.9):
                modifiers["high_reputation"] = self.model_gauss_sample(0.7, relative_std=0.2, lo=0.4, hi=0.9)
        return modifiers

    def apply_sanction_multiplier(self, agent: Citizen, norm: EmergentNorm) -> float:
        formal_high = self.model_gauss_sample(0.7, relative_std=0.15, lo=0.5, hi=0.9)
        formal_low = self.model_gauss_sample(0.3, relative_std=0.2, lo=0.1, hi=0.5)
        if self.legal_system.formalism_index > formal_high:
            return 1.0
        if self.legal_system.formalism_index < formal_low:
            return self._contextual_multiplier(agent, norm)
        base = self.model_gauss_sample(1.0, relative_std=0.1, lo=0.7, hi=1.3)
        contextual = self._contextual_multiplier(agent, norm)
        w = self.legal_system.formalism_index
        return w * base + (1.0 - w) * contextual

    def _contextual_multiplier(self, agent: Citizen, norm: EmergentNorm) -> float:
        mult = self.model_gauss_sample(1.0, relative_std=0.1, lo=0.7, hi=1.3)
        mods = norm.contextual_modifiers or {}
        if mods.get("high_reputation") and agent.reputation_coop > self.model_gauss_sample(0.7, relative_std=0.2, lo=0.5, hi=0.9):
            mult *= mods["high_reputation"]
        if mods.get("elite_protection") and self.wealth_percentile(agent.wealth) > self.model_gauss_sample(0.8, relative_std=0.15, lo=0.6, hi=0.95):
            mult *= mods["elite_protection"]
        if mods.get("coalition_member") and getattr(agent, "alliance_id", None):
            mult *= mods["coalition_member"]
        return mult

    def update_legal_system(self):
        """Bottom-up emergence of norms from repeated interaction outcomes."""
        # GAUSSIANIZED: norm thresholds derived from stochastic memory.
        alive = self.agents_alive()
        pop = len(alive)
        if pop == 0:
            self.legal_system.norms = []
            return
        window = int(self.model_gauss_sample(200, std=40, lo=80, hi=300, integer=True))
        recent = self.behavior_memory[-window:] if self.behavior_memory else []
        behavior_stats: Dict[str, Dict[str, float]] = {}
        for event in recent:
            behavior = str(event.get("behavior", ""))
            if not behavior:
                continue
            entry = behavior_stats.setdefault(behavior, {"count": 0.0, "score": 0.0})
            entry["count"] += self.model_gauss_sample(1.0, relative_std=0.1, lo=0.7, hi=1.3)
            entry["score"] += float(event.get("outcome", 0.0))
        norms: List[EmergentNorm] = []
        for behavior, stat in behavior_stats.items():
            freq = stat["count"] / max(1.0, float(len(recent)))
            freq_thr = self.model_gauss_sample(0.1, relative_std=0.4, lo=0.03, hi=0.25)
            if freq < freq_thr:
                continue
            avg_score = stat["score"] / max(1.0, stat["count"])
            score_thr = self.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.2)
            if abs(avg_score) < score_thr:
                continue
            direction = "encourage" if avg_score > 0 else "discourage"
            supporters = []
            for a in alive:
                mem = a.social_memory.get(behavior)
                if not mem:
                    continue
                score = mem.get("score", 0.0)
                pos_thr = self.model_gauss_sample(0.1, relative_std=0.5, lo=0.02, hi=0.3)
                neg_thr = self.model_gauss_sample(-0.1, std=0.05, lo=-0.3, hi=-0.02)
                if direction == "encourage" and score > pos_thr:
                    supporters.append(a.unique_id)
                if direction == "discourage" and score < neg_thr:
                    supporters.append(a.unique_id)
            strength = len(supporters) / pop
            supporters_agents = [a for a in alive if a.unique_id in set(supporters)]
            if supporters_agents:
                cognitive_origin = {
                    "reasoning": float(np.mean([a.latent.get("reasoning", 0.5) for a in supporters_agents])),
                    "empathy": float(np.mean([a.latent.get("empathy", 0.5) for a in supporters_agents])),
                    "sociality": float(np.mean([a.latent.get("sociality", 0.5) for a in supporters_agents])),
                    "dark_mach": float(np.mean([a.latent.get("dark_mach", 0.5) for a in supporters_agents])),
                    "moral_common_good": float(np.mean([a.latent.get("moral_common_good", 0.5) for a in supporters_agents])),
                }
                base = self.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
                w_score = self.model_gauss_sample(0.6, relative_std=0.3, lo=0.2, hi=0.9)
                w_common = self.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
                enforcement = clamp01(base + w_score * abs(avg_score) + w_common * cognitive_origin["moral_common_good"])
            else:
                cognitive_origin = {
                    "reasoning": self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8),
                    "empathy": self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8),
                    "sociality": self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8),
                    "dark_mach": self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8),
                    "moral_common_good": self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8),
                }
                enforcement = self.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
            rule = f"{direction}_{behavior}"
            norms.append(
                EmergentNorm(
                    rule=rule,
                    strength=clamp01(strength),
                    enforcement=enforcement,
                    cognitive_origin=cognitive_origin,
                    supporters=supporters,
                    behavior=behavior,
                    direction=direction,
                    contextual_modifiers=self.generate_contextual_modifiers(),
                    applies_to="all",
                )
            )
        # Add an anti-gossip norm when negative gossip is frequent and empathy is high.
        neg_rate = self.step_events.get("negative_gossip", 0.0) / max(1, pop)
        neg_thr = self.model_gauss_sample(0.15, relative_std=0.4, lo=0.05, hi=0.35)
        avg_empathy = self.latent_mean("empathy")
        emp_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85)
        if neg_rate > neg_thr and avg_empathy > emp_thr:
            supporters = [a.unique_id for a in alive if a.latent.get("empathy", 0.5) > emp_thr]
            cognitive_origin = {
                "reasoning": float(np.mean([a.latent.get("reasoning", 0.5) for a in alive])) if alive else 0.5,
                "empathy": avg_empathy,
                "sociality": float(np.mean([a.latent.get("sociality", 0.5) for a in alive])) if alive else 0.5,
                "dark_mach": float(np.mean([a.latent.get("dark_mach", 0.5) for a in alive])) if alive else 0.5,
                "moral_common_good": float(np.mean([a.latent.get("moral_common_good", 0.5) for a in alive])) if alive else 0.5,
            }
            enforcement = self.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
            norms.append(
                EmergentNorm(
                    rule="discourage_gossip",
                    strength=clamp01(len(supporters) / max(1, pop)),
                    enforcement=enforcement,
                    cognitive_origin=cognitive_origin,
                    supporters=supporters,
                    behavior="gossip_negative",
                    direction="discourage",
                    contextual_modifiers=self.generate_contextual_modifiers(),
                    applies_to="all",
                )
            )
        # Non-lethal punishment norm when violence is frequent but deaths are rare.
        violence_freq = self.step_events.get("violence_events", 0.0) / max(1, pop)
        death_rate = self.step_events.get("violence_deaths", 0.0) / max(1.0, self.step_events.get("violence_events", 1.0))
        freq_thr = self.model_gauss_sample(0.1, relative_std=0.3, lo=0.03, hi=0.25)
        death_thr = self.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.12)
        if violence_freq > freq_thr and death_rate < death_thr and avg_empathy > emp_thr:
            supporters = [a.unique_id for a in alive if a.latent.get("empathy", 0.5) > emp_thr]
            enforcement = self.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
            norms.append(
                EmergentNorm(
                    rule="non_lethal_punishment",
                    strength=clamp01(len(supporters) / max(1, pop)),
                    enforcement=enforcement,
                    cognitive_origin={
                        "reasoning": float(np.mean([a.latent.get("reasoning", 0.5) for a in alive])) if alive else 0.5,
                        "empathy": avg_empathy,
                        "sociality": float(np.mean([a.latent.get("sociality", 0.5) for a in alive])) if alive else 0.5,
                        "dark_mach": float(np.mean([a.latent.get("dark_mach", 0.5) for a in alive])) if alive else 0.5,
                        "moral_common_good": float(np.mean([a.latent.get("moral_common_good", 0.5) for a in alive])) if alive else 0.5,
                    },
                    supporters=supporters,
                    behavior="violence",
                    direction="discourage",
                    contextual_modifiers=self.generate_contextual_modifiers(),
                    applies_to="all",
                )
            )
            self.lethality_multiplier = self.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.8)
        else:
            self.lethality_multiplier = 1.0

        # Authoritarian regimes with low legitimacy may allow lethal punishment.
        if self.political_system.participation_structure in ("tyranny", "oligarchy"):
            leg_thr = self.model_gauss_sample(0.4, relative_std=0.3, lo=0.2, hi=0.6)
            formal_thr = self.model_gauss_sample(0.7, relative_std=0.2, lo=0.5, hi=0.9)
            if self.political_system.legitimacy < leg_thr and self.legal_formalism > formal_thr:
                self.lethality_multiplier *= self.model_gauss_sample(2.0, relative_std=0.4, lo=1.5, hi=2.5)
        self.lethality_multiplier = float(np.clip(self.lethality_multiplier, 0.1, 3.0))
        self.legal_system.norms = norms
        self.legal_system.formalism_index = self.calculate_formalism_index()
        self.legal_system.discretion_level = clamp01(1.0 - self.legal_system.formalism_index)
        if self.legal_system.formalism_index > self.model_gauss_sample(0.7, relative_std=0.15, lo=0.5, hi=0.9):
            self.legal_system.enforcement_style = "universal"
        elif self.legal_system.formalism_index < self.model_gauss_sample(0.3, relative_std=0.2, lo=0.1, hi=0.5):
            self.legal_system.enforcement_style = "selective"
        else:
            self.legal_system.enforcement_style = "mixed"
        self.legal_formalism = self.legal_system.formalism_index

    def _calculate_norm_consistency(self) -> float:
        if not self.legal_system.norms:
            return 1.0
        contradictions = 0
        behaviors = {}
        for n in self.legal_system.norms:
            behaviors.setdefault(n.behavior, set()).add(n.direction)
        for dirs in behaviors.values():
            if len(dirs) > 1:
                contradictions += 1
        return clamp01(1.0 - contradictions / max(1.0, float(len(behaviors))))

    def _calculate_norm_centralization(self) -> float:
        norms = self.legal_system.norms
        if not norms:
            return 0.0
        elite_ids = {a.unique_id for a in self.get_elite_agents()}
        central = 0.0
        for norm in norms:
            if not norm.supporters:
                continue
            elite_share = sum(1 for s in norm.supporters if s in elite_ids) / max(1, len(norm.supporters))
            elite_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85)
            if elite_share > elite_thr:
                central += 1.0
        return central / max(1.0, float(len(norms)))

    def apply_decentralized_enforcement(self, behavior: str, violator: Citizen, victim: Citizen):
        """Apply peer punishment based on endorsed norms with coalition cost sharing."""
        norms = [n for n in self.legal_system.norms if n.behavior == behavior and n.direction == "discourage"]
        if not norms:
            return
        applicable = norms[0]
        supporters = set(applicable.supporters)
        radius = int(self.model_gauss_sample(2, std=0.6, lo=1, hi=3, integer=True))
        neighbors = self.get_social_neighbors(victim, radius=radius, include_center=True)
        candidates = [a for a in neighbors if isinstance(a, Citizen) and a.alive and a.unique_id in supporters and a is not violator]
        enforcers = []
        for a in candidates:
            base = self.model_gauss_sample(0.3, relative_std=0.4, lo=0.05, hi=0.6)
            w_common = self.model_gauss_sample(0.7, relative_std=0.3, lo=0.2, hi=1.0)
            p = clamp01(applicable.enforcement * (base + w_common * a.latent.get("moral_common_good", 0.5)))
            if self.rng.random() < p:
                enforcers.append(a)
        if not enforcers:
            return
        coalition_size = len(enforcers)
        sanction = applicable.enforcement * self.apply_sanction_multiplier(violator, applicable)
        base = self.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.8)
        scale = self.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.8)
        coalition_scale = self.model_gauss_sample(3.0, relative_std=0.3, lo=1.5, hi=5.0)
        sanction *= (base + scale * min(1.0, coalition_size / coalition_scale)) * self.enforcement_multiplier
        violator.wealth -= sanction
        coop_drop = self.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
        fear_drop = self.model_gauss_sample(0.02, relative_std=0.6, lo=0.005, hi=0.06)
        violator.reputation_coop = clamp01(violator.reputation_coop - coop_drop)
        violator.reputation_fear = clamp01(violator.reputation_fear - fear_drop)
        if getattr(self, "enable_ostracism", False):
            ostracism_steps = int(self.model_gauss_sample(10, std=4, lo=3, hi=20, integer=True))
            violator.ostracism_timer = max(violator.ostracism_timer, ostracism_steps)
        cost_base = self.model_gauss_sample(0.04, relative_std=0.5, lo=0.01, hi=0.12)
        cost = cost_base / max(1, coalition_size)
        for enforcer in enforcers:
            enforcer.wealth -= cost
            rep_boost = self.model_gauss_sample(0.02, relative_std=0.6, lo=0.005, hi=0.06)
            enforcer.reputation_coop = clamp01(enforcer.reputation_coop + rep_boost)
            enforcer.nd_cost += cost

    def apply_normative_rewards(self, behavior: str, actor: Citizen, partner: Citizen):
        """Apply peer rewards for norm-consistent prosocial behaviors."""
        norms = [n for n in self.legal_system.norms if n.behavior == behavior and n.direction == "encourage"]
        if not norms:
            return
        applicable = norms[0]
        supporters = set(applicable.supporters)
        radius = int(self.model_gauss_sample(2, std=0.6, lo=1, hi=3, integer=True))
        neighbors = self.get_social_neighbors(actor, radius=radius, include_center=True)
        candidates = [a for a in neighbors if isinstance(a, Citizen) and a.alive and a.unique_id in supporters]
        rewarders = []
        for a in candidates:
            base = self.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
            w_common = self.model_gauss_sample(0.6, relative_std=0.3, lo=0.2, hi=0.9)
            p = clamp01(applicable.enforcement * (base + w_common * a.latent.get("moral_common_good", 0.5)))
            if self.rng.random() < p:
                rewarders.append(a)
        if not rewarders:
            return
        reward_rate = self.model_gauss_sample(0.02, relative_std=0.6, lo=0.005, hi=0.06)
        reward = reward_rate * applicable.enforcement * self.reward_multiplier
        actor.wealth += reward
        partner_share = self.model_gauss_sample(0.5, relative_std=0.3, lo=0.2, hi=0.8)
        partner.wealth += reward * partner_share
        cost_base = self.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
        cost = cost_base / max(1, len(rewarders))
        for r in rewarders:
            r.wealth -= cost
            rep_boost = self.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
            r.reputation_coop = clamp01(r.reputation_coop + rep_boost)

    def detect_economic_mechanism(self) -> str:
        alive = self.agents_alive()
        if not alive:
            return "collapsed"
        wealth = np.array([a.wealth for a in alive], dtype=float)
        contrib = np.array([a.nd_contribution for a in alive], dtype=float)
        if wealth.size > 1 and contrib.std() > 1e-6:
            corr = float(np.corrcoef(wealth, contrib)[0, 1])
        else:
            corr = 0.0
        top_share, _, _ = self._top5_power()
        avg_common = float(np.mean([a.latent.get("moral_common_good", 0.5) for a in alive]))
        top_thr = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.4, hi=0.8)
        power_thr = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.3, hi=0.7)
        corr_thr = self.model_gauss_sample(0.35, relative_std=0.2, lo=0.2, hi=0.6)
        gini_market = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.4, hi=0.8)
        common_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        gini_red = self.model_gauss_sample(0.45, relative_std=0.2, lo=0.25, hi=0.65)
        if top_share > top_thr and self.power_concentration > power_thr:
            return "hierarchy"
        if corr > corr_thr and self.gini_wealth < gini_market:
            return "market"
        if avg_common > common_thr and self.gini_wealth < gini_red:
            return "redistribution"
        return "mixed"

    def detect_conflict_resolution(self) -> str:
        violence = self.last_metrics.get("violence_rate", 0.0)
        violence_thr = self.model_gauss_sample(0.25, relative_std=0.2, lo=0.1, hi=0.45)
        formal_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        if violence > violence_thr:
            return "violence"
        if self.legal_system.formalism_index > formal_thr and self.legal_system.norms:
            return "arbitration"
        return "consensus"

    def detect_moral_framework(self) -> str:
        alive = self.agents_alive()
        if not alive:
            return "collapsed"
        empathy = self.latent_mean("empathy")
        prosocial = self.latent_mean("moral_prosocial")
        reasoning = self.latent_mean("reasoning")
        dominance = self.latent_mean("dominance")
        emp_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        pro_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        reason_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        emp_low = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.3, hi=0.7)
        dom_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        if empathy > emp_thr and prosocial > pro_thr:
            return "prosocial"
        if reasoning > reason_thr and empathy < emp_low:
            return "contractual"
        if dominance > dom_thr:
            return "hierarchical"
        return "mixed"

    def update_political_system(self):
        """Update political structure and benefit orientation from power distribution."""
        # GAUSSIANIZED: thresholds and weights sampled per update.
        alive = self.agents_alive()
        if not alive:
            return
        scores = self._power_scores(alive)
        scores.sort(key=lambda x: x[1], reverse=True)
        power_vals = [p for _, p in scores]
        total_power = sum(power_vals) or 1e-6
        self.political_system.decision_weight_gini = gini(power_vals)
        self.power_concentration = self.political_system.decision_weight_gini
        top1 = power_vals[0] / total_power
        top_k = max(1, int(self.model_gauss_sample(5, std=1.5, lo=3, hi=10, integer=True)))
        top_frac = self.model_gauss_sample(0.2, relative_std=0.3, lo=0.1, hi=0.35)
        top5 = sum(power_vals[:top_k]) / total_power
        top20 = sum(power_vals[: max(1, int(len(power_vals) * top_frac))]) / total_power
        threshold_tyranny = self.model_gauss_sample(0.5, relative_std=0.15, lo=0.4, hi=0.65)
        threshold_oligarchy = self.model_gauss_sample(0.7, relative_std=0.12, lo=0.6, hi=0.8)
        threshold_concentrated = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.5, hi=0.75)
        merit_gap = self.model_gauss_sample(0.2, relative_std=0.3, lo=0.1, hi=0.35)
        min_pop = int(self.model_gauss_sample(10, std=3, lo=6, hi=20, integer=True))
        if top1 > threshold_tyranny:
            self.political_system.participation_structure = "tyranny"
            self.political_system.decision_makers = [scores[0][0]]
        elif top5 > threshold_oligarchy and len(alive) > min_pop:
            self.political_system.participation_structure = "oligarchy"
            self.political_system.decision_makers = [uid for uid, _ in scores[:top_k]]
        elif top20 > threshold_concentrated:
            elite_ids = [uid for uid, _ in scores[: max(1, int(len(scores) * top_frac))]]
            elite_agents = [a for a in alive if a.unique_id in elite_ids]
            avg_reasoning_elite = float(np.mean([a.latent.get("reasoning", 0.5) for a in elite_agents]))
            avg_empathy_elite = float(np.mean([a.latent.get("empathy", 0.5) for a in elite_agents]))
            avg_reasoning_all = self.latent_mean("reasoning")
            avg_empathy_all = self.latent_mean("empathy")
            if avg_reasoning_elite > avg_reasoning_all + merit_gap or avg_empathy_elite > avg_empathy_all + merit_gap:
                self.political_system.participation_structure = "meritocracy"
                self.political_system.merit_criteria = {
                    "reasoning": max(0.0, avg_reasoning_elite - avg_reasoning_all),
                    "empathy": max(0.0, avg_empathy_elite - avg_empathy_all),
                }
            else:
                self.political_system.participation_structure = "oligarchy"
                self.political_system.merit_criteria = {}
            self.political_system.decision_makers = elite_ids
        else:
            self.political_system.participation_structure = "democracy"
            self.political_system.decision_makers = [a.unique_id for a in alive]

        decision_makers = [a for a in alive if a.unique_id in set(self.political_system.decision_makers)]
        if not decision_makers:
            return
        avg_empathy_rulers = float(np.mean([a.latent.get("empathy", 0.5) for a in decision_makers]))
        avg_dark_rulers = float(np.mean([a.dark_core for a in decision_makers]))
        avg_prosocial_rulers = float(np.mean([a.latent.get("moral_prosocial", 0.5) for a in decision_makers]))
        recent_redistribution_rate = self.step_events.get("redistribution_rate", 0.0)
        dark_threshold = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.5, hi=0.75)
        empathy_low_threshold = self.model_gauss_sample(0.4, relative_std=0.2, lo=0.3, hi=0.55)
        empathy_high_threshold = self.model_gauss_sample(0.7, relative_std=0.12, lo=0.6, hi=0.8)
        prosocial_threshold = self.model_gauss_sample(0.7, relative_std=0.12, lo=0.6, hi=0.8)
        redist_threshold = self.model_gauss_sample(0.3, relative_std=0.25, lo=0.2, hi=0.45)
        if avg_dark_rulers > dark_threshold and avg_empathy_rulers < empathy_low_threshold:
            self.political_system.benefit_orientation = "extractive"
        elif avg_empathy_rulers > empathy_high_threshold and avg_prosocial_rulers > prosocial_threshold:
            if self.political_system.participation_structure in {"tyranny", "meritocracy"}:
                self.political_system.benefit_orientation = "paternalist"
            else:
                self.political_system.benefit_orientation = "redistributive"
        elif recent_redistribution_rate > redist_threshold:
            self.political_system.benefit_orientation = "redistributive"
        else:
            self.political_system.benefit_orientation = "meritocratic"

        avg_coop_rep = float(np.mean([a.reputation_coop for a in decision_makers]))
        avg_fear_rep = float(np.mean([a.reputation_fear for a in decision_makers]))
        violence_rate = self.last_metrics.get("violence_rate", 0.0)
        w_coop = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.1, hi=0.8)
        w_fear = self.model_gauss_sample(0.2, relative_std=0.25, lo=0.05, hi=0.5)
        w_gini = self.model_gauss_sample(0.2, relative_std=0.25, lo=0.05, hi=0.5)
        w_violence = self.model_gauss_sample(0.1, relative_std=0.3, lo=0.02, hi=0.4)
        total_w = w_coop + w_fear + w_gini + w_violence
        if total_w > 0:
            w_coop, w_fear, w_gini, w_violence = w_coop / total_w, w_fear / total_w, w_gini / total_w, w_violence / total_w
        self.political_system.legitimacy = clamp01(
            w_coop * avg_coop_rep + w_fear * (1.0 - avg_fear_rep) + w_gini * (1.0 - self.gini_wealth) + w_violence * (1.0 - violence_rate)
        )

    def execute_governance_decision(self):
        """Execute redistribution decisions based on political orientation."""
        # GAUSSIANIZED: redistribution parameters inside each method.
        decision_makers = [a for a in self.agents_alive() if a.unique_id in set(self.political_system.decision_makers)]
        if not decision_makers:
            return
        if self.political_system.benefit_orientation == "extractive":
            self.redistribute_extractive(decision_makers)
        elif self.political_system.benefit_orientation == "redistributive":
            self.redistribute_egalitarian()
        elif self.political_system.benefit_orientation == "paternalist":
            self.redistribute_paternalist(decision_makers)
        else:
            self.redistribute_meritocratic()

    def redistribute_extractive(self, elite: List[Citizen]):
        """
        Elite extracts wealth from non-elite population.

        GAUSSIANIZED: Extraction rates sampled from Gaussian distributions.
        """
        all_alive = self.agents_alive()
        non_elite = [a for a in all_alive if a not in elite]
        if not non_elite:
            return
        base_extraction = self.model_gauss_sample(0.05, relative_std=0.3, lo=0.01, hi=0.12)
        extraction_rate = base_extraction * float(np.mean([a.dark_core for a in elite]))
        total_extracted = sum(a.wealth * extraction_rate for a in non_elite)
        for agent in non_elite:
            agent.wealth *= (1 - extraction_rate)
        share_per_elite = total_extracted / max(1, len(elite))
        for agent in elite:
            agent.wealth += share_per_elite
        self.step_events["redistribution_rate"] = -extraction_rate

    def redistribute_egalitarian(self):
        """
        Egalitarian redistribution.

        GAUSSIANIZED: Redistribution intensity sampled per step.
        """
        all_alive = self.agents_alive()
        if not all_alive:
            return
        total_wealth = sum(a.wealth for a in all_alive)
        equal_share = total_wealth / max(1, len(all_alive))
        redistribution_intensity = self.model_gauss_sample(0.1, relative_std=0.4, lo=0.03, hi=0.25)
        for agent in all_alive:
            delta = (equal_share - agent.wealth) * redistribution_intensity
            agent.wealth += delta
        self.step_events["redistribution_rate"] = redistribution_intensity

    def redistribute_paternalist(self, rulers: List[Citizen]):
        """
        Empathetic rulers optimize welfare.

        GAUSSIANIZED: Transfer rate sampled per step.
        """
        avg_empathy = float(np.mean([a.latent.get("empathy", 0.5) for a in rulers]))
        if avg_empathy < 0.7:
            return
        all_alive = self.agents_alive()
        if not all_alive:
            return
        sorted_by_wealth = sorted(all_alive, key=lambda a: a.wealth)
        transfer_rate = self.model_gauss_sample(0.05, relative_std=0.3, lo=0.02, hi=0.12)
        transfer_amount = transfer_rate * sum(a.wealth for a in sorted_by_wealth[-max(1, len(sorted_by_wealth) // 5) :])
        bottom_half = sorted_by_wealth[: max(1, len(sorted_by_wealth) // 2)]
        for agent in bottom_half:
            agent.wealth += transfer_amount / max(1, len(bottom_half))
        for agent in sorted_by_wealth[-max(1, len(sorted_by_wealth) // 5) :]:
            agent.wealth -= transfer_amount / max(1, len(sorted_by_wealth) // 5)
        self.step_events["redistribution_rate"] = 0.05

    def redistribute_meritocratic(self):
        """
        Reward by contribution.

        GAUSSIANIZED: Reward pool rate sampled per step.
        """
        all_alive = self.agents_alive()
        if not all_alive:
            return
        total_contribution = sum(a.nd_contribution for a in all_alive)
        if total_contribution == 0:
            return
        reward_pool_rate = self.model_gauss_sample(0.1, relative_std=0.35, lo=0.05, hi=0.2)
        reward_pool = reward_pool_rate * sum(a.wealth for a in all_alive)
        for agent in all_alive:
            contribution_share = agent.nd_contribution / total_contribution
            reward = reward_pool * contribution_share
            agent.wealth += reward
        for agent in all_alive:
            if agent.nd_contribution < 0.01:
                agent.wealth *= 0.98

    def get_cognitive_profile_top20(self) -> Dict[str, float]:
        elite = self.get_elite_agents()
        if not elite:
            return {}
        traits = ["reasoning", "empathy", "sociality", "dark_mach", "moral_common_good", "moral_prosocial", "dominance"]
        return {t: float(np.mean([a.latent.get(t, 0.5) for a in elite])) for t in traits}

    def update_institutional_metrics(self):
        """Refresh institutional architecture metrics for data collection."""
        alive = self.agents_alive()
        pop = len(alive)
        self.norm_count = len(self.legal_system.norms)
        self.norm_density = self.norm_count / max(1, pop)
        self.norm_consistency = self._calculate_norm_consistency()
        self.norm_centralization = self._calculate_norm_centralization()
        self.economic_mechanism = self.detect_economic_mechanism()
        self.conflict_resolution = self.detect_conflict_resolution()
        self.moral_framework = self.detect_moral_framework()
        self.cognitive_profile_top20 = self.get_cognitive_profile_top20()

    def analyze_cognitive_institutional_causality(self) -> Dict[str, object]:
        """Correlate elite cognitive profile with institutional architecture."""
        elite = self.get_elite_agents()
        cognitive_profile = self.get_cognitive_profile_top20()
        legal_architecture = {
            "formalism": self.legal_system.formalism_index,
            "norms": len(self.legal_system.norms),
            "consistency": self.norm_consistency,
            "centralization": self.norm_centralization,
        }
        political_architecture = {
            "participation": self.political_system.participation_structure,
            "benefit": self.political_system.benefit_orientation,
            "legitimacy": self.political_system.legitimacy,
        }
        narrative = "ANALISIS CAUSAL:\n\n"
        formal_high = self.model_gauss_sample(0.7, relative_std=0.15, lo=0.5, hi=0.9)
        formal_low = self.model_gauss_sample(0.3, relative_std=0.2, lo=0.1, hi=0.5)
        reason_thr = self.model_gauss_sample(0.7, relative_std=0.2, lo=0.5, hi=0.9)
        emp_low = self.model_gauss_sample(0.4, relative_std=0.2, lo=0.2, hi=0.6)
        soc_thr = self.model_gauss_sample(0.7, relative_std=0.2, lo=0.5, hi=0.9)
        emp_high = self.model_gauss_sample(0.7, relative_std=0.2, lo=0.5, hi=0.9)
        if legal_architecture["formalism"] > formal_high:
            narrative += f"Sistema legal FORMALISTA (indice={legal_architecture['formalism']:.2f}). "
            if cognitive_profile.get("reasoning", 0.5) > reason_thr:
                narrative += f"El razonamiento alto en la elite (R={cognitive_profile['reasoning']:.2f}) favorece reglas universales. "
            if cognitive_profile.get("empathy", 0.5) < emp_low:
                narrative += f"La empatia baja (E={cognitive_profile['empathy']:.2f}) reduce excepciones contextuales. "
        elif legal_architecture["formalism"] < formal_low:
            narrative += f"Sistema legal CASUISTICO (indice={legal_architecture['formalism']:.2f}). "
            if cognitive_profile.get("sociality", 0.5) > soc_thr:
                narrative += f"La sociabilidad alta (S={cognitive_profile['sociality']:.2f}) favorece juicios personalizados. "
            if cognitive_profile.get("empathy", 0.5) > emp_high:
                narrative += f"La empatia alta (E={cognitive_profile['empathy']:.2f}) aumenta consideracion contextual. "
        else:
            narrative += f"Sistema legal MIXTO (indice={legal_architecture['formalism']:.2f}). "
        narrative += "\n\n"
        narrative += f"Sistema politico {political_architecture['participation'].upper()} con orientacion {political_architecture['benefit'].upper()}. "
        if political_architecture["participation"] == "tyranny":
            if political_architecture["benefit"] == "paternalist":
                narrative += "Tirania benevolente basada en empatía. "
            elif political_architecture["benefit"] == "extractive":
                narrative += "Tirania extractiva favoreciendo al dominante. "
        narrative += f"\n\nLegitimidad estimada: {political_architecture['legitimacy']:.2f}/1.0"
        return {
            "cognitive_profile": cognitive_profile,
            "legal_architecture": legal_architecture,
            "political_architecture": political_architecture,
            "elite_count": len(elite),
            "causal_narrative": narrative,
        }

    def _compute_production_multiplier(self) -> float:
        # GAUSSIANIZED: environment multipliers.
        base = self.model_gauss_sample(1.0, relative_std=0.1, lo=0.8, hi=1.3)
        if self.climate == "scarce":
            scarcity_mult = self.model_gauss_sample(0.4, relative_std=0.25, lo=0.2, hi=0.6)
            base *= scarcity_mult
        elif self.climate == "abundant":
            abundance_mult = self.model_gauss_sample(1.2, relative_std=0.15, lo=1.0, hi=1.5)
            base *= abundance_mult
        if self.external_factor == "disaster":
            disaster_mult = self.model_gauss_sample(0.4, relative_std=0.3, lo=0.2, hi=0.7)
            base *= disaster_mult
        return base

    def _compute_mortality_multiplier(self) -> float:
        # GAUSSIANIZED: environment multipliers.
        mult = self.model_gauss_sample(1.0, relative_std=0.1, lo=0.8, hi=1.3)
        if self.climate == "scarce":
            mult *= self.model_gauss_sample(3.0, relative_std=0.2, lo=2.0, hi=4.0)
        if self.external_factor in {"disaster"}:
            mult *= self.model_gauss_sample(3.0, relative_std=0.25, lo=2.0, hi=4.5)
        return mult

    def _goal_tag(self, agent: Citizen) -> str:
        goals = agent.conscious_core.get("self_model", {}).get("goals") or []
        if goals:
            return goals[0]
        reasoning = agent.latent.get("reasoning", 0.5)
        empathy = agent.latent.get("empathy", 0.5)
        reason_thr = self.model_gauss_sample(0.65, relative_std=0.2, lo=0.4, hi=0.85)
        emp_thr = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.3, hi=0.7)
        if reasoning > reason_thr and empathy < emp_thr:
            return "orden_formal"
        if empathy > self.model_gauss_sample(0.65, relative_std=0.2, lo=0.4, hi=0.85):
            return "cuidado_comunidad"
        if agent.latent.get("impulsivity", 0.5) > self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85):
            return "innovacion_caotica"
        return "neutro"

    def _latent_vector(self, agent: Citizen, keys: Tuple[str, ...]) -> np.ndarray:
        return np.array([agent.latent.get(k, 0.5) for k in keys], dtype=float)

    def _cosine_similarity(self, a_vec: np.ndarray, b_vec: np.ndarray) -> float:
        denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)) or 1e-6
        return float(np.dot(a_vec, b_vec) / denom)

    def _update_alliances(self):
        # GAUSSIANIZED: alliance formation weights and probabilities.
        alive = self.agents_alive()
        for a in alive:
            a.alliance_id = None
        alliances: Dict[str, Dict[str, object]] = {}
        if not alive:
            self.alliances = alliances
            return
        keys = ("reasoning", "empathy", "dominance", "impulsivity")
        aid_counter = 0
        for a in alive:
            w_emp, w_soc, w_lang = gauss_weights(self.rng, [0.4, 0.3, 0.3], relative_std=0.2, normalize=True)
            affinity = clamp01(
                w_emp * a.latent.get("empathy", 0.5)
                + w_soc * a.latent.get("sociality", 0.5)
                + w_lang * a.latent.get("language", 0.5)
            )
            if self.rng.random() > affinity:
                continue
            partners = [n for n in self.get_social_neighbors(a, radius=2, include_center=True) if isinstance(n, Citizen) and n.alive]
            members = [a]
            for n in partners:
                sim = self._cosine_similarity(self._latent_vector(a, keys), self._latent_vector(n, keys))
                w_sim, w_emp = gauss_weights(self.rng, [0.5, 0.5], relative_std=0.2, normalize=True)
                join_p = clamp01(w_sim * sim + w_emp * n.latent.get("empathy", 0.5))
                if self.rng.random() < join_p:
                    members.append(n)
            min_members = int(self.model_gauss_sample(2, std=0.5, lo=2, hi=4, integer=True))
            if len(members) < min_members:
                continue
            aid = f"ally_{aid_counter}"
            aid_counter += 1
            rule_threshold = self.model_gauss_sample(0.0, std=0.1, lo=-0.3, hi=0.3)
            alliances[aid] = {
                "goal": self._goal_tag(a),
                "members": members,
                "rule": "prosocial"
                if (np.mean([m.latent.get("empathy", 0.5) for m in members]) - np.mean([m.latent.get("dominance", 0.5) for m in members]))
                >= rule_threshold
                else "dominance",
            }
            for m in members:
                m.alliance_id = aid
                rep_nudge = self.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
                if alliances[aid]["rule"] == "prosocial":
                    m.reputation_coop = clamp01(m.reputation_coop + rep_nudge * affinity)
                else:
                    m.reputation_fear = clamp01(m.reputation_fear + rep_nudge * (1.0 - affinity))
                m.reward("alliance", intensity=1.0)
            # Gossip-based recruitment: boost reputation of aligned candidates.
            if alliances[aid]["rule"] == "prosocial":
                member_vecs = [self._latent_vector(m, keys) for m in members]
                mean_vec = np.mean(member_vecs, axis=0) if member_vecs else None
                if mean_vec is not None:
                    candidates = [c for c in alive if c not in members]
                    if candidates:
                        candidate = self.rng.choice(candidates)
                        sim = self._cosine_similarity(mean_vec, self._latent_vector(candidate, keys))
                        sim_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.85)
                        if sim > sim_thr:
                            boost = self.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
                            candidate.reputation_coop = clamp01(candidate.reputation_coop + boost)
        self.alliances = alliances
        self._resolve_alliance_conflict()

    def _resolve_alliance_conflict(self):
        # GAUSSIANIZED: conflict probabilities and penalties.
        if len(self.alliances) < int(self.model_gauss_sample(2, std=0.5, lo=2, hi=4, integer=True)):
            return
        ids = list(self.alliances.keys())
        for aid in ids:
            ally = self.alliances[aid]
            members: List[Citizen] = ally["members"]  # type: ignore
            if not members:
                continue
            agg = np.mean([m.latent.get("aggression", 0.5) for m in members])
            reason = np.mean([m.latent.get("reasoning", 0.5) for m in members])
            target_id = None
            for other_id in ids:
                if other_id == aid:
                    continue
                target_id = other_id
                break
            if not target_id:
                continue
            target = self.alliances[target_id]
            target_members: List[Citizen] = target["members"]  # type: ignore
            if not target_members:
                continue
            w_agg = self.model_gauss_sample(0.4, relative_std=0.25, lo=0.05, hi=0.8)
            w_reason = self.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.6)
            clash_p = clamp01(w_agg * agg + w_reason * (1.0 - reason))
            if self.rng.random() < clash_p:
                victim = self.rng.choice(target_members)
                wealth_hit = self.model_gauss_sample(0.2, relative_std=0.4, lo=0.05, hi=0.5)
                happy_hit = self.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.15)
                fear_boost = self.model_gauss_sample(0.05, relative_std=0.4, lo=0.01, hi=0.15)
                victim.wealth -= wealth_hit
                victim.happiness = clamp01(victim.happiness - happy_hit)
                victim.reputation_fear = clamp01(victim.reputation_fear + fear_boost)
            else:
                for m in members:
                    boost_rate = self.model_gauss_sample(0.02, relative_std=0.5, lo=0.005, hi=0.06)
                    boost = m.latent.get("language", 0.5) * m.latent.get("reasoning", 0.5) * boost_rate
                    m.reputation_coop = clamp01(m.reputation_coop + boost)
                # Prosocial coalitions reduce violence pressure proportionally
                formalism_nudge = self.model_gauss_sample(0.01, relative_std=0.6, lo=0.002, hi=0.04)
                self.legal_formalism = clamp01(self.legal_formalism + formalism_nudge * reason)

    def _compose_latent(self, jitter: float = 0.05):
        base_trait = lambda: self.model_gauss_sample(0.5, relative_std=0.1, lo=0.3, hi=0.7)
        traits = {
            "attn_selective": base_trait(),
            "attn_flex": base_trait(),
            "hyperfocus": base_trait(),
            "impulsivity": base_trait(),
            "risk_aversion": base_trait(),
            "sociality": base_trait(),
            "language": base_trait(),
            "reasoning": base_trait(),
            "trust": base_trait(),
            "emotional_impulsivity": base_trait(),
            "resilience": base_trait(),
        }
        if self.calibrate_traits and self.calibration_data and sample_calibrated_traits is not None:
            calibrated = sample_calibrated_traits(self.calibration_data, self.rng)
            for k, v in calibrated.items():
                traits[k] = clamp01(v)
        bias_min: Dict[str, float] = {}
        bias_max: Dict[str, float] = {}
        bias_w: Dict[str, float] = {}
        spec_min: Dict[str, float] = {}
        spec_max: Dict[str, float] = {}
        spec_w: Dict[str, float] = {}
        ids = [self.profile_weights["profile1"], self.profile_weights["profile2"], self.profile_weights["profile3"]]
        weights = [self.profile_weights["weight1"], self.profile_weights["weight2"], self.profile_weights["weight3"]]
        total_w = sum(w for w in weights if w > 0)
        chosen_id = None
        for pid, w in zip(ids, weights):
            profile_def = PROFILE_MAP.get(pid)
            if not pid or profile_def is None or w <= 0:
                continue
            if chosen_id is None:
                chosen_id = pid
            for k, v in profile_def.get("traits", {}).items():
                traits[k] = traits.get(k, 0.5) + v * w / max(total_w, 1e-6)
            bio_bias = profile_def.get("biological_bias", {}) or {}
            for trait, rng in bio_bias.items():
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    bias_min[trait] = bias_min.get(trait, 0.0) + float(rng[0]) * w
                    bias_max[trait] = bias_max.get(trait, 0.0) + float(rng[1]) * w
                    bias_w[trait] = bias_w.get(trait, 0.0) + w
            spec_ranges = profile_def.get("spectrum_ranges", {}) or {}
            for trait, rng in spec_ranges.items():
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    spec_min[trait] = spec_min.get(trait, 0.0) + float(rng[0]) * w
                    spec_max[trait] = spec_max.get(trait, 0.0) + float(rng[1]) * w
                    spec_w[trait] = spec_w.get(trait, 0.0) + w
        for k in traits:
            traits[k] = clamp01(traits[k] + float(self.rng.normal(0, jitter)))
        default_low = self.model_gauss_sample(0.4, relative_std=0.2, lo=0.2, hi=0.6)
        default_high = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        default_range = (min(default_low, default_high), max(default_low, default_high))
        moral_emotional_keys = {
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
            "emotional_impulsivity",
            "resilience",
            "guilt",
            "shame",
        }
        combined_bias: Dict[str, List[float]] = {}
        for trait in moral_emotional_keys | set(bias_min.keys()):
            weight = bias_w.get(trait, 0.0)
            if weight > 0:
                low = bias_min.get(trait, 0.0) / weight
                high = bias_max.get(trait, 0.0) / weight
            else:
                low, high = default_range
            combined_bias[trait] = [clamp01(low), clamp01(max(low, high))]

        if self.initial_moral_bias:
            dark_related = {"dark_narc", "dark_mach", "dark_psycho", "aggression", "moral_spite"}
            prosocial_related = {"moral_prosocial", "moral_common_good", "moral_honesty", "empathy"}

            for trait, rng in combined_bias.items():
                low, high = rng
                shift = self.model_gauss_sample(0.1, relative_std=0.4, lo=0.03, hi=0.2)
                if self.initial_moral_bias == "high_dark" and trait in dark_related:
                    low = clamp01(low + shift)
                    high = clamp01(high + shift)
                elif self.initial_moral_bias == "high_prosocial" and trait in prosocial_related:
                    low = clamp01(low + shift)
                    high = clamp01(high + shift)
                elif self.initial_moral_bias == "low_dark" and trait in dark_related:
                    low = clamp01(low - shift)
                    high = clamp01(high - shift)
                combined_bias[trait] = [min(low, high), max(low, high)]

        if self.resilience_bias:
            for trait in ["resilience", "affect_reg"]:
                if trait in combined_bias:
                    low, high = combined_bias[trait]
                    shift = self.model_gauss_sample(0.1, relative_std=0.4, lo=0.03, hi=0.2)
                    shift = shift if self.resilience_bias == "high" else -shift
                    combined_bias[trait] = [clamp01(low + shift), clamp01(high + shift)]

        if self.emotional_bias:
            for trait in ["emotional_impulsivity", "aggression"]:
                if trait in combined_bias:
                    low, high = combined_bias[trait]
                    shift = self.model_gauss_sample(0.1, relative_std=0.4, lo=0.03, hi=0.2)
                    shift = shift if self.emotional_bias == "high" else -shift
                    combined_bias[trait] = [clamp01(low + shift), clamp01(high + shift)]

        combined_spec: Dict[str, List[float]] = {}
        for trait, weight in spec_w.items():
            low = spec_min.get(trait, 0.0) / max(weight, 1e-6)
            high = spec_max.get(trait, 0.0) / max(weight, 1e-6)
            combined_spec[trait] = [clamp01(low), clamp01(max(low, high))]

        return traits, combined_bias, combined_spec, chosen_id or ""

    def agents_alive(self) -> List[Citizen]:
        return [a for a in self.agents if isinstance(a, Citizen) and a.alive]

    def _top5_power(self):
        alive = self.agents_alive()
        if not alive:
            return 0.0, 0.0, 0.0
        sorted_agents = sorted(alive, key=lambda a: a.wealth, reverse=True)
        top_k = max(1, int(self.model_gauss_sample(5, std=1.5, lo=3, hi=10, integer=True)))
        top5 = sorted_agents[:top_k]
        total_wealth = sum(a.wealth for a in alive) or 1e-6
        top5_share = sum(a.wealth for a in top5) / total_wealth
        fear_avg = np.mean([a.reputation_fear for a in top5])
        coop_avg = np.mean([a.reputation_coop for a in top5])
        return top5_share, fear_avg, coop_avg

    def normalized_wealth(self, wealth: float) -> float:
        alive = self.agents_alive()
        if not alive:
            return 0.5
        vals = np.array([a.wealth for a in alive], dtype=float)
        mean = vals.mean() if vals.size else 1.0
        std = vals.std() if vals.size else 1.0
        if std < 1e-6:
            return clamp01(wealth / (mean + 1e-6))
        z = (wealth - mean) / std
        base = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.2, hi=0.8)
        scale = self.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        return clamp01(base + scale * z)

    def _update_regime(self, avg_reasoning: float, avg_empathy: float, avg_dom: float):
        g = self.gini_wealth
        pop_scaled = len(self.agents_alive()) * self.scale_factor
        violence = self.last_metrics.get("violence_rate", 0.0)
        avg_language = self.latent_mean("language")
        avg_affect_reg = self.latent_mean("affect_reg")
        top_share, top_fear, top_coop = self._top5_power()

        collapse_thr = self.model_gauss_sample(0.2, relative_std=0.2, lo=0.1, hi=0.35)
        tyr_share = self.model_gauss_sample(0.65, relative_std=0.15, lo=0.45, hi=0.8)
        tyr_fear = self.model_gauss_sample(0.75, relative_std=0.12, lo=0.6, hi=0.9)
        olg_share = self.model_gauss_sample(0.60, relative_std=0.15, lo=0.4, hi=0.78)
        olg_coop = self.model_gauss_sample(0.80, relative_std=0.12, lo=0.6, hi=0.92)
        pred_g = self.model_gauss_sample(0.58, relative_std=0.15, lo=0.4, hi=0.8)
        pred_dom = self.model_gauss_sample(0.70, relative_std=0.12, lo=0.5, hi=0.9)
        tech_reason = self.model_gauss_sample(0.78, relative_std=0.12, lo=0.6, hi=0.9)
        tech_g = self.model_gauss_sample(0.55, relative_std=0.15, lo=0.35, hi=0.75)
        tech_emp = self.model_gauss_sample(0.55, relative_std=0.15, lo=0.35, hi=0.7)
        demo_emp = self.model_gauss_sample(0.75, relative_std=0.12, lo=0.6, hi=0.9)
        demo_g = self.model_gauss_sample(0.38, relative_std=0.2, lo=0.2, hi=0.6)
        demo_v = self.model_gauss_sample(0.08, relative_std=0.4, lo=0.02, hi=0.2)
        anar_v = self.model_gauss_sample(0.30, relative_std=0.2, lo=0.15, hi=0.5)
        anar_r = self.model_gauss_sample(0.4, relative_std=0.25, lo=0.2, hi=0.6)
        anar_v2 = self.model_gauss_sample(0.2, relative_std=0.25, lo=0.1, hi=0.4)
        olg_g = self.model_gauss_sample(0.55, relative_std=0.15, lo=0.35, hi=0.75)
        olg_dom = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.4, hi=0.8)
        olg_lang = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.4, hi=0.8)
        dict_form = self.model_gauss_sample(0.7, relative_std=0.12, lo=0.55, hi=0.9)
        dict_lib = self.model_gauss_sample(0.3, relative_std=0.2, lo=0.15, hi=0.5)
        theo_dom = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.4, hi=0.8)
        theo_aff = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.4, hi=0.8)
        plut_g = self.model_gauss_sample(0.6, relative_std=0.15, lo=0.4, hi=0.85)
        plut_emp = self.model_gauss_sample(0.4, relative_std=0.2, lo=0.2, hi=0.6)

        if pop_scaled < collapse_thr * self.initial_population_scaled:
            self.regime = "Colapso Social"
        elif top_share > tyr_share and top_fear > tyr_fear:
            self.regime = "Tirania Psicopatica"
        elif top_share > olg_share and top_coop > olg_coop:
            self.regime = "Oligarquia Carismatica / Liderazgo Empatico"
        elif g > pred_g and avg_dom > pred_dom:
            self.regime = "Oligarquia Predatoria"
        elif avg_reasoning > tech_reason and g > tech_g and avg_empathy < tech_emp:
            self.regime = "Tecnocracia Autoritaria"
        elif avg_empathy > demo_emp and g < demo_g and violence < demo_v:
            self.regime = "Democracia Ilustrada / Comunidad Empatica"
        elif violence > anar_v:
            self.regime = "Anarquia Violenta"
        elif avg_reasoning < anar_r and violence > anar_v2:
            self.regime = "Anarquia Tribal"
        elif g > olg_g and avg_dom > olg_dom and avg_language > olg_lang:
            self.regime = "Oligarquia Carismatica"
        elif self.legal_formalism > dict_form and self.liberty_index < dict_lib:
            self.regime = "Dictadura Burocratica"
        elif avg_dom > theo_dom and avg_affect_reg > theo_aff:
            self.regime = "Teocracia Moralista"
        elif g > plut_g and avg_empathy < plut_emp:
            self.regime = "Plutocracia"
        else:
            self.regime = "Regimen Transicional"

    def latent_mean(self, key: str) -> float:
        alive = self.agents_alive()
        if not alive:
            return 0.0
        return float(np.mean([a.latent.get(key, 0.5) for a in alive]))

    def _compute_network_metrics(self) -> Dict[str, float]:
        if self.social_graph is None or compute_network_metrics is None or nx is None:
            return {
                "network_clustering": 0.0,
                "network_avg_path_len": 0.0,
                "network_degree_mean": 0.0,
                "network_degree_std": 0.0,
                "network_degree_gini": 0.0,
            }
        alive_ids = [a.unique_id for a in self.agents_alive()]
        if not alive_ids:
            return {
                "network_clustering": 0.0,
                "network_avg_path_len": 0.0,
                "network_degree_mean": 0.0,
                "network_degree_std": 0.0,
                "network_degree_gini": 0.0,
            }
        sub = self.social_graph.subgraph(alive_ids)
        metrics = compute_network_metrics(sub, self.rng, sample_size=200)
        return {
            "network_clustering": metrics.clustering,
            "network_avg_path_len": metrics.avg_path_len,
            "network_degree_mean": metrics.degree_mean,
            "network_degree_std": metrics.degree_std,
            "network_degree_gini": metrics.degree_gini,
        }

    def _git_hash(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _fermi_update(self):
        if not self.enable_fermi_update:
            return
        alive = self.agents_alive()
        if len(alive) < 2:
            return
        beta = max(0.1, self.fermi_beta)
        for agent in alive:
            neighbors = self.get_social_neighbors(agent, radius=1, include_center=False)
            if not neighbors:
                continue
            other = self.rng.choice(neighbors)
            payoff_self = (
                agent.wealth
                + agent.reputation_total()
                + (1.0 - agent.reputation_fear)
                + (1.0 if getattr(agent, "alliance_id", None) else 0.0)
            )
            payoff_other = (
                other.wealth
                + other.reputation_total()
                + (1.0 - other.reputation_fear)
                + (1.0 if getattr(other, "alliance_id", None) else 0.0)
            )
            prob = 1.0 / (1.0 + math.exp(-beta * (payoff_other - payoff_self)))
            if self.rng.random() < prob:
                for key in ("moral_prosocial", "moral_common_good", "aggression", "empathy", "guilt"):
                    agent.latent[key] = clamp01(other.latent.get(key, agent.latent.get(key, 0.5)))
                agent.guilt = clamp01(agent.latent.get("guilt", agent.guilt))

    def _apply_policy_mode(self):
        mode = (self.policy_mode or "none").lower()
        if mode in {"top_down_coercive", "coercive"}:
            self.enforcement_multiplier = 1.5
            self.reward_multiplier = 0.8
            self.legal_system.enforcement_style = "universal"
        elif mode in {"bottom_up", "emergent"}:
            self.enforcement_multiplier = 1.0
            self.reward_multiplier = 1.0
        elif mode in {"incentives", "reward"}:
            self.enforcement_multiplier = 0.7
            self.reward_multiplier = 1.3
        elif mode in {"meritocratic", "merit"}:
            self.enforcement_multiplier = 0.9
            self.reward_multiplier = 1.2
        else:
            self.enforcement_multiplier = 1.0
            self.reward_multiplier = 1.0

    def _update_metrics(self):
        alive = self.agents_alive()
        pop = len(alive)
        if pop == 0:
            self.last_metrics = {
                "coop_rate": 0.0,
                "violence_rate": 0.0,
                "life_expectancy": 0.0,
                "avg_reasoning": 0.0,
                "avg_empathy": 0.0,
                "avg_dominance": 0.0,
                "male_violence_rate": 0.0,
                "female_violence_rate": 0.0,
                "births": 0.0,
                "sex_ratio": 0.0,
                "coalition_count": 0.0,
                "coalition_wins": 0.0,
                "sneaky_success_rate": 0.0,
            "male_male_conflicts": 0.0,
            "female_indirect_competition": 0.0,
            "mating_inequality": 0.0,
            "mean_harem_size": 0.0,
            "repro_gini_males": 0.0,
            "male_childless_share": 0.0,
            "mean_partners_male": 0.0,
            "mean_partners_female": 0.0,
            "gossip_rate": 0.0,
            "positive_gossip_rate": 0.0,
            "negative_gossip_rate": 0.0,
            "lethality_rate": 0.0,
        }
            self.running = False
            return

        coop = sum(1 for a in alive if a.last_action == "coop")
        viol = sum(1 for a in alive if a.last_action == "violence")
        avg_age = float(np.mean([a.age for a in alive])) if alive else 0.0
        avg_reasoning = self.latent_mean("reasoning")
        avg_empathy = self.latent_mean("empathy")
        avg_dom = self.latent_mean("dominance")
        male_count = sum(1 for a in alive if a.gender == "Male")
        female_count = sum(1 for a in alive if a.gender == "Female")
        social_weight = self.model_gauss_sample(1.0, relative_std=0.2, lo=0.6, hi=1.4)
        leadership_scores = sorted(
            [a.latent.get("reasoning", 0.5) * (1.0 - social_weight * a.latent.get("sociality", 0.5)) for a in alive], reverse=True
        )
        top_frac = self.model_gauss_sample(0.05, relative_std=0.4, lo=0.02, hi=0.12)
        top_leaders = leadership_scores[: max(1, int(top_frac * len(leadership_scores)))] if leadership_scores else []
        top_mean_leader = float(np.mean(top_leaders)) if top_leaders else 0.0
        conscious_mean = float(np.mean([a.conscious_core.get("awareness", 0.0) for a in alive])) if alive else 0.0
        allied_share = (sum(1 for a in alive if getattr(a, "alliance_id", None)) / pop) if pop else 0.0
        sneaky_attempts = max(1.0, self.step_events.get("sneaky_attempts", 0.0))
        sneaky_success_rate = self.step_events.get("sneaky_success", 0.0) / sneaky_attempts
        mating_success_vals = [a.mating_success for a in alive if a.gender == "Male"]
        mean_harem_size = (sum(mating_success_vals) / max(1, male_count)) if mating_success_vals else 0.0
        male_children = [len(a.children_ids) for a in alive if a.gender == "Male"]
        repro_gini_m = gini(male_children) if male_children else 0.0
        male_childless = sum(1 for c in male_children if c == 0)
        male_childless_share = male_childless / max(1, len(male_children)) if male_children else 0.0
        mean_partners_male = float(np.mean([len(a.mates_lifetime) for a in alive if a.gender == "Male"])) if male_count else 0.0
        mean_partners_female = float(np.mean([len(a.mates_lifetime) for a in alive if a.gender == "Female"])) if female_count else 0.0
        gossip_total = self.step_events.get("gossip_total", 0.0)
        positive_gossip = self.step_events.get("positive_gossip", 0.0)
        negative_gossip = self.step_events.get("negative_gossip", 0.0)
        violence_events = max(1.0, self.step_events.get("violence_events", 0.0))
        violence_deaths = self.step_events.get("violence_deaths", 0.0)
        lethality_raw = violence_deaths / violence_events
        lethality_rate = self.model_gauss_sample(lethality_raw, relative_std=0.2, lo=0.0, hi=0.2)
        net_metrics = self._compute_network_metrics()

        self.last_metrics = {
            "coop_rate": coop / pop,
            "violence_rate": viol / pop,
            "male_violence_rate": self.step_events.get("male_violence", 0.0) / max(pop, 1),
            "female_violence_rate": self.step_events.get("female_violence", 0.0) / max(pop, 1),
            "life_expectancy": avg_age,
            "avg_reasoning": avg_reasoning,
            "avg_empathy": avg_empathy,
            "avg_dominance": avg_dom,
            "avg_leadership": top_mean_leader,
            "conscious_awareness": conscious_mean,
            "alliances_count": len(self.alliances),
            "allied_share": allied_share,
            "births": self.step_events.get("births", 0.0),
            "sex_ratio": male_count / max(pop, 1),
            "coalition_count": self.step_events.get("coalition_count", 0.0),
            "coalition_wins": self.step_events.get("coalition_wins", 0.0),
            "sneaky_success_rate": sneaky_success_rate,
            "male_male_conflicts": self.step_events.get("male_male_conflicts", 0.0),
            "female_indirect_competition": self.step_events.get("female_indirect_competition", 0.0),
            "mating_inequality": gini(mating_success_vals),
            "mean_harem_size": mean_harem_size,
            "repro_gini_males": repro_gini_m,
            "male_childless_share": male_childless_share,
            "mean_partners_male": mean_partners_male,
            "mean_partners_female": mean_partners_female,
            "gossip_rate": gossip_total / max(pop, 1),
            "positive_gossip_rate": positive_gossip / max(pop, 1),
            "negative_gossip_rate": negative_gossip / max(pop, 1),
            "lethality_rate": lethality_rate,
            "network_clustering": net_metrics["network_clustering"],
            "network_avg_path_len": net_metrics["network_avg_path_len"],
            "network_degree_mean": net_metrics["network_degree_mean"],
            "network_degree_std": net_metrics["network_degree_std"],
            "network_degree_gini": net_metrics["network_degree_gini"],
            "cultural_convergence": self.cultural_convergence,
        }

        pressure_weight = self.model_gauss_sample(0.3, relative_std=0.3, lo=0.1, hi=0.6)
        if self.legal_system.norms:
            self.legal_formalism = self.legal_system.formalism_index
        else:
            self.legal_formalism = clamp01(top_mean_leader * (1 + pressure_weight * self.institution_pressure))
        w_emp = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.3, hi=0.8)
        w_dom = self.model_gauss_sample(0.2, relative_std=0.3, lo=0.05, hi=0.5)
        self.liberty_index = clamp01(w_emp * avg_empathy + w_dom * (1 - avg_dom))
        self.gini_wealth = gini([a.wealth for a in alive])
        self._update_regime(avg_reasoning, avg_empathy, avg_dom)
        if hasattr(self, "recent_violence_events") and self.recent_violence_events:
            decay_window = int(self.model_gauss_sample(10, std=3, lo=5, hi=20, integer=True))
            decay = max(0, len(self.recent_violence_events) - decay_window)
            if decay > 0:
                self.recent_violence_events = self.recent_violence_events[-decay_window:]

    def step(self):
        self._reset_step_events()
        self.agents.shuffle_do("step")
        self.step_count += 1
        if self.policy_schedule and self.step_count in self.policy_schedule:
            self.policy_mode = self.policy_schedule[self.step_count]
        self._apply_policy_mode()
        # economic growth from high reasoning prosocial agents
        prosocial_thr = self.model_gauss_sample(0.5, relative_std=0.2, lo=0.3, hi=0.7)
        reason_thr = self.model_gauss_sample(0.6, relative_std=0.2, lo=0.4, hi=0.8)
        for a in self.agents_alive():
            if a.latent.get("moral_prosocial", 0.5) > prosocial_thr and a.latent.get("reasoning", 0.5) > reason_thr:
                growth_base = self.model_gauss_sample(0.3, std=0.15, lo=0.05, hi=0.6)
                growth = float(growth_base * a.latent.get("reasoning", 0.5))
                a.wealth += growth
                self.total_wealth += growth
                a.nd_contribution += growth
        if hasattr(self, "recent_violence_events") and len(self.recent_violence_events) > 5:
            formalism_boost = self.model_gauss_sample(0.05, relative_std=0.5, lo=0.01, hi=0.12)
            self.legal_formalism = clamp01(self.legal_formalism + formalism_boost)
        self._update_alliances()
        interval = int(self.model_gauss_sample(5, std=1.2, lo=3, hi=9, integer=True))
        if self.step_count % interval == 0:
            self.update_legal_system()
            self.update_political_system()
            self.execute_governance_decision()
        if self.enable_cultural_transmission and cultural_transmission_step is not None:
            self.cultural_convergence = cultural_transmission_step(self)
        self._fermi_update()
        self._update_metrics()
        self.update_institutional_metrics()
        self.total_wealth = sum(agent.wealth for agent in self.agents_alive())
        self.datacollector.collect(self)
        if not self.agents_alive():
            self.running = False

    def redistribute_wealth_to_allies(self, amount: float, allies: List[Citizen]):
        if not allies:
            return
        share = amount / len(allies)
        for ally in allies:
            ally.wealth += share

    def log_event(self, tag: str, payload: object):
        if not hasattr(self, "event_log"):
            self.event_log = []
        self.event_log.append((tag, payload))

    def register_violence(self, attacker: Citizen, victim: Citizen):
        self.recent_violence_events.append((attacker.unique_id, victim.unique_id))
        self.step_events["violence_events"] += 1
        if attacker.gender == "Male":
            self.step_events["male_violence"] += 1
        else:
            self.step_events["female_violence"] += 1
