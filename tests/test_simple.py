#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_simple import PsychNeuroSociety


def test_initialization():
    model = PsychNeuroSociety(seed=42, population_scale="tiny")
    assert len(model.alive_agents()) == 10
    agent = model.alive_agents()[0]
    assert "empathy" in agent.traits
    assert "reasoning" in agent.traits
    assert "dominance" in agent.traits


def test_no_biological_lifecycle_fields():
    model = PsychNeuroSociety(seed=42, population_scale="tiny")
    agent = model.alive_agents()[0]
    forbidden = [
        "age",
        "gender",
        "sex",
        "gestation_timer",
        "fertility_cooldown",
        "offspring_count",
        "current_partner",
    ]
    for name in forbidden:
        assert not hasattr(agent, name), f"Unexpected biological field: {name}"


def test_run_collects_metrics():
    model = PsychNeuroSociety(seed=42, population_scale="tiny")
    history = model.run(5)
    assert len(history) == 5
    for column in ["coop_rate", "violence_rate", "defection_rate", "gini_wealth", "regime"]:
        assert column in history.columns


def test_reproducibility():
    model_a = PsychNeuroSociety(seed=123, population_scale="tiny")
    model_b = PsychNeuroSociety(seed=123, population_scale="tiny")
    hist_a = model_a.run(10)
    hist_b = model_b.run(10)
    assert hist_a["regime"].tolist() == hist_b["regime"].tolist()
    assert hist_a["coop_rate"].round(8).tolist() == hist_b["coop_rate"].round(8).tolist()


def main() -> int:
    tests = [
        test_initialization,
        test_no_biological_lifecycle_fields,
        test_run_collects_metrics,
        test_reproducibility,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"✗ {test.__name__}: {exc}")
    print(f"Passed {passed}/{len(tests)}")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
