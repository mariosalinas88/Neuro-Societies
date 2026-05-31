from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from model_asperger import PsychNeuroSociety


Scenario = Dict[str, object]


BASE_SCENARIOS: List[Scenario] = [
    {"name": "pure_neutral_1", "profiles": [("1", 1.0)], "kind": "pure"},
    {"name": "pure_adhd_2", "profiles": [("2", 1.0)], "kind": "pure"},
    {"name": "pure_asperger_3", "profiles": [("3", 1.0)], "kind": "pure"},
    {"name": "pure_ocd_5", "profiles": [("5", 1.0)], "kind": "pure"},
    {"name": "pure_empathy_9", "profiles": [("9", 1.0)], "kind": "pure"},
    {"name": "pure_dark_10", "profiles": [("10", 1.0)], "kind": "pure"},
    {"name": "pure_gifted_11", "profiles": [("11", 1.0)], "kind": "pure"},
    {"name": "pure_depression_12", "profiles": [("12", 1.0)], "kind": "pure"},
    {"name": "dark_vs_empathy", "profiles": [("10", 0.5), ("9", 0.5)], "kind": "extreme_mix"},
    {"name": "dark_vs_asperger", "profiles": [("10", 0.5), ("3", 0.5)], "kind": "extreme_mix"},
    {"name": "dark_vs_ocd", "profiles": [("10", 0.5), ("5", 0.5)], "kind": "extreme_mix"},
    {"name": "empathy_asperger_ocd", "profiles": [("9", 0.34), ("3", 0.33), ("5", 0.33)], "kind": "cooperative_mix"},
    {"name": "adhd_dark_depression", "profiles": [("2", 0.34), ("10", 0.33), ("12", 0.33)], "kind": "risk_mix"},
    {"name": "adhd_asperger_empathy", "profiles": [("2", 0.34), ("3", 0.33), ("9", 0.33)], "kind": "mixed"},
]


def random_scenario(rng: random.Random, scenario_index: int) -> Scenario:
    profile_pool = ["1", "2", "3", "5", "6", "7", "8", "9", "10", "11", "12"]
    n = rng.choice([1, 2, 3])
    chosen = rng.sample(profile_pool, n)
    raw_weights = [rng.random() for _ in chosen]
    total = sum(raw_weights) or 1.0
    weights = [w / total for w in raw_weights]
    return {
        "name": f"random_{scenario_index:03d}_" + "_".join(chosen),
        "profiles": list(zip(chosen, weights)),
        "kind": "random",
    }


def pad_profiles(profiles: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    return profiles + [("", 0.0)] * (3 - len(profiles))


def run_once(scenario: Scenario, seed: int, steps: int, population_scale: str) -> Dict[str, object]:
    profiles = pad_profiles(list(scenario["profiles"]))  # type: ignore[index]
    model = PsychNeuroSociety(
        seed=seed,
        population_scale=population_scale,
        profile1=str(profiles[0][0]),
        weight1=float(profiles[0][1]),
        profile2=str(profiles[1][0]),
        weight2=float(profiles[1][1]),
        profile3=str(profiles[2][0]),
        weight3=float(profiles[2][1]),
    )
    history = model.run(steps)
    last = history.iloc[-1].to_dict() if not history.empty else {}
    per_profile = model.per_profile_stats()

    dominant_profile = ""
    fear_profile = ""
    if not per_profile.empty:
        dominant_profile = str(per_profile.sort_values("wealth_mean", ascending=False).iloc[0]["profile"])
        fear_profile = str(per_profile.sort_values("fear_rep_mean", ascending=False).iloc[0]["profile"])

    return {
        "scenario": scenario["name"],
        "kind": scenario.get("kind", ""),
        "seed": seed,
        "steps": steps,
        "population_scale": population_scale,
        "profiles": str(scenario["profiles"]),
        "regime": last.get("regime", ""),
        "coop_rate": float(last.get("coop_rate", 0.0)),
        "violence_rate": float(last.get("violence_rate", 0.0)),
        "defection_rate": float(last.get("defection_rate", 0.0)),
        "support_rate": float(last.get("support_rate", 0.0)),
        "avoidance_rate": float(last.get("avoidance_rate", 0.0)),
        "gini_wealth": float(last.get("gini_wealth", 0.0)),
        "wealth_mean": float(last.get("wealth_mean", 0.0)),
        "empathy_mean": float(last.get("empathy_mean", 0.0)),
        "reasoning_mean": float(last.get("reasoning_mean", 0.0)),
        "dominance_mean": float(last.get("dominance_mean", 0.0)),
        "dark_core_mean": float(last.get("dark_core_mean", 0.0)),
        "dominant_profile_by_wealth": dominant_profile,
        "highest_fear_profile": fear_profile,
    }


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["scenario"])].append(row)

    out: List[Dict[str, object]] = []
    numeric = [
        "coop_rate",
        "violence_rate",
        "defection_rate",
        "support_rate",
        "avoidance_rate",
        "gini_wealth",
        "wealth_mean",
        "empathy_mean",
        "reasoning_mean",
        "dominance_mean",
        "dark_core_mean",
    ]
    for scenario, vals in buckets.items():
        item: Dict[str, object] = {
            "scenario": scenario,
            "kind": vals[0].get("kind", ""),
            "runs": len(vals),
            "profiles": vals[0].get("profiles", ""),
            "dominant_regime": Counter(str(v.get("regime", "")) for v in vals).most_common(1)[0][0],
            "dominant_profile_by_wealth_mode": Counter(str(v.get("dominant_profile_by_wealth", "")) for v in vals).most_common(1)[0][0],
            "highest_fear_profile_mode": Counter(str(v.get("highest_fear_profile", "")) for v in vals).most_common(1)[0][0],
        }
        for key in numeric:
            nums = [float(v[key]) for v in vals]
            item[f"{key}_mean"] = sum(nums) / len(nums)
            item[f"{key}_min"] = min(nums)
            item[f"{key}_max"] = max(nums)
        out.append(item)
    return sorted(out, key=lambda x: str(x["scenario"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded randomized stress tests for the simplified psych-neuro model.")
    parser.add_argument("--runs_per_scenario", type=int, default=10)
    parser.add_argument("--random_scenarios", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--population_scale", choices=["tiny", "small", "tribe", "city"], default="small")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_seconds", type=float, default=25.0)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    started = time.perf_counter()
    rng = random.Random(args.seed)
    scenarios = list(BASE_SCENARIOS)
    for idx in range(args.random_scenarios):
        scenarios.append(random_scenario(rng, idx))

    os.makedirs(args.output_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []
    stopped_by_time = False

    for sidx, scenario in enumerate(scenarios):
        for ridx in range(args.runs_per_scenario):
            if time.perf_counter() - started > args.max_seconds:
                stopped_by_time = True
                break
            seed = rng.randint(1, 2_000_000_000)
            rows.append(run_once(scenario, seed=seed, steps=args.steps, population_scale=args.population_scale))
        if stopped_by_time:
            break

    if not rows:
        raise RuntimeError("No stress-test rows were produced. Increase --max_seconds or reduce model size.")

    aggregate_rows = aggregate(rows)
    raw_path = os.path.join(args.output_dir, "stress_simple_raw.csv")
    agg_path = os.path.join(args.output_dir, "stress_simple_aggregate.csv")

    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print("STRESS TEST COMPLETE")
    print(f"rows={len(rows)} scenarios_completed={len(set(r['scenario'] for r in rows))} stopped_by_time={stopped_by_time}")
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
    print(f"raw={raw_path}")
    print(f"aggregate={agg_path}")

    # Print the most important aggregate diagnostics.
    by_violence = sorted(aggregate_rows, key=lambda x: float(x["violence_rate_mean"]), reverse=True)[:5]
    by_wealth = sorted(aggregate_rows, key=lambda x: float(x["wealth_mean_mean"]), reverse=True)[:5]
    by_gini = sorted(aggregate_rows, key=lambda x: float(x["gini_wealth_mean"]), reverse=True)[:5]

    print("\nTop violence scenarios:")
    for row in by_violence:
        print(f"  {row['scenario']}: violence={float(row['violence_rate_mean']):.3f}, wealth={float(row['wealth_mean_mean']):.3f}, regime={row['dominant_regime']}")

    print("\nTop wealth scenarios:")
    for row in by_wealth:
        print(f"  {row['scenario']}: wealth={float(row['wealth_mean_mean']):.3f}, violence={float(row['violence_rate_mean']):.3f}, regime={row['dominant_regime']}")

    print("\nTop inequality scenarios:")
    for row in by_gini:
        print(f"  {row['scenario']}: gini={float(row['gini_wealth_mean']):.3f}, violence={float(row['violence_rate_mean']):.3f}, regime={row['dominant_regime']}")


if __name__ == "__main__":
    main()
