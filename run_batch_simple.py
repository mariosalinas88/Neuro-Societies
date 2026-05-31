from __future__ import annotations

import csv
import os
from typing import Dict, List

from model_simple import PsychNeuroSociety


SCENARIOS: List[Dict[str, object]] = [
    {"name": "all_neutral", "profiles": [("1", 1.0)], "notes": "Control profile only"},
    {"name": "nt_adhd_autism", "profiles": [("1", 0.4), ("2", 0.3), ("3", 0.3)], "notes": "Mixed neutral, ADHD, autism-like profiles"},
    {"name": "high_reasoning_prosocial", "profiles": [("3", 0.5), ("11", 0.5)], "notes": "High reasoning and prosocial profile mix"},
    {"name": "high_reasoning_dark", "profiles": [("3", 0.4), ("10", 0.6)], "notes": "High reasoning under dark/dominance pressure"},
    {"name": "low_resilience_mix", "profiles": [("2", 0.3), ("12", 0.4), ("7", 0.3)], "notes": "Low resilience and competitive vulnerability"},
    {"name": "inclusive_nd_mix", "profiles": [("3", 0.3), ("2", 0.3), ("6", 0.4)], "notes": "Neurodiverse mix with social/cooperative potential"},
]


def run_scenario(cfg: Dict[str, object], seed: int, steps: int, population_scale: str) -> Dict[str, object]:
    profiles = list(cfg["profiles"])  # type: ignore[index]
    padded = profiles + [("", 0.0)] * (3 - len(profiles))

    model = PsychNeuroSociety(
        seed=seed,
        population_scale=population_scale,
        profile1=str(padded[0][0]),
        weight1=float(padded[0][1]),
        profile2=str(padded[1][0]),
        weight2=float(padded[1][1]),
        profile3=str(padded[2][0]),
        weight3=float(padded[2][1]),
    )
    history = model.run(steps)
    last = history.iloc[-1].to_dict() if not history.empty else {}
    per_profile = model.per_profile_stats()

    dominant_profile = ""
    vulnerable_profile = ""
    if not per_profile.empty:
        dominant_profile = str(per_profile.sort_values("wealth_mean", ascending=False).iloc[0]["profile"])
        vulnerable_profile = str(per_profile.sort_values("fear_rep_mean", ascending=False).iloc[0]["profile"])

    return {
        "scenario": cfg["name"],
        "notes": cfg.get("notes", ""),
        "seed": seed,
        "steps": steps,
        "population_scale": population_scale,
        "regime": last.get("regime", ""),
        "coop_rate": round(float(last.get("coop_rate", 0.0)), 4),
        "violence_rate": round(float(last.get("violence_rate", 0.0)), 4),
        "defection_rate": round(float(last.get("defection_rate", 0.0)), 4),
        "support_rate": round(float(last.get("support_rate", 0.0)), 4),
        "avoidance_rate": round(float(last.get("avoidance_rate", 0.0)), 4),
        "gini_wealth": round(float(last.get("gini_wealth", 0.0)), 4),
        "wealth_mean": round(float(last.get("wealth_mean", 0.0)), 4),
        "empathy_mean": round(float(last.get("empathy_mean", 0.0)), 4),
        "reasoning_mean": round(float(last.get("reasoning_mean", 0.0)), 4),
        "dominance_mean": round(float(last.get("dominance_mean", 0.0)), 4),
        "dark_core_mean": round(float(last.get("dark_core_mean", 0.0)), 4),
        "dominant_profile_by_wealth": dominant_profile,
        "vulnerable_profile_by_fear": vulnerable_profile,
        "input_profiles": str(cfg["profiles"]),
    }


def main() -> None:
    os.makedirs("results", exist_ok=True)
    rows = []
    seeds = [101, 102, 103]
    for cfg in SCENARIOS:
        for seed in seeds:
            rows.append(run_scenario(cfg, seed=seed, steps=150, population_scale="small"))

    out_path = "results/simple_batch_runs.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_path}")
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()
