#!/usr/bin/env python3
"""Run profile-specific society simulations and export aggregate results."""
import json
import os
from pathlib import Path
from statistics import mean

import pandas as pd

from model import Citizen, PROFILE_MAP, SocietyModel


OUTPUT_DIR = Path("simulation_results/profile_societies")
STEPS = 120
SEEDS = [301, 302, 303]
PROFILE_IDS = sorted(PROFILE_MAP.keys())

TRAITS = [
    "empathy",
    "moral_prosocial",
    "moral_common_good",
    "moral_honesty",
    "moral_spite",
    "aggression",
    "dark_narc",
    "dark_mach",
    "dark_psycho",
    "sociality",
    "reasoning",
    "resilience",
]

METRICS = [
    "coop_rate",
    "violence_rate",
    "gossip_rate",
    "positive_gossip_rate",
    "negative_gossip_rate",
    "lethality_rate",
    "gini_wealth",
    "legal_formalism",
    "liberty_index",
    "norm_count",
    "norm_density",
    "norm_consistency",
    "power_concentration",
    "political_legitimacy",
    "network_clustering",
    "cultural_convergence",
]


def profile_name(profile_id):
    profile = PROFILE_MAP.get(profile_id, {})
    return profile.get("name", f"profile_{profile_id}")


def alive_trait_mean(agents, trait):
    values = [float(a.latent.get(trait, 0.5)) for a in agents if isinstance(a, Citizen)]
    return mean(values) if values else 0.0


def run_one(profile_id, seed):
    model = SocietyModel(
        seed=seed,
        population_scale="tiny",
        profile1=str(profile_id),
        profile2=None,
        profile3=None,
        weight1=1.0,
        weight2=0.0,
        weight3=0.0,
        enable_reproduction=False,
        enable_sexual_selection=False,
        coalition_enabled=True,
        enable_guilt=True,
        enable_ostracism=True,
        enable_fermi_update=True,
        enable_cultural_transmission=True,
    )
    initial_agents = [a for a in model.agents if isinstance(a, Citizen)]
    initial_traits = {f"initial_{trait}": alive_trait_mean(initial_agents, trait) for trait in TRAITS}

    for _ in range(STEPS):
        model.step()

    alive = model.agents_alive()
    df = model.datacollector.get_model_vars_dataframe()
    final = df.iloc[-1].to_dict()
    row = {
        "profile_id": profile_id,
        "profile_name": profile_name(profile_id),
        "seed": seed,
        "steps": STEPS,
        "alive": len(alive),
        "regime": model.regime,
        "economic_mechanism": model.economic_mechanism,
        "conflict_resolution": model.conflict_resolution,
        "moral_framework": model.moral_framework,
        "participation_structure": model.political_system.participation_structure,
        "benefit_orientation": model.political_system.benefit_orientation,
    }
    row.update(initial_traits)
    for trait in TRAITS:
        row[f"final_{trait}"] = alive_trait_mean(alive, trait)
    for metric in METRICS:
        value = final.get(metric, None)
        row[metric] = float(value) if isinstance(value, (int, float)) else value
    return row, df


def summarize(rows):
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    excluded = {"profile_id", "seed", "steps"}
    metric_cols = [c for c in numeric_cols if c not in excluded]
    grouped = df.groupby(["profile_id", "profile_name"], as_index=False)[metric_cols].mean()
    return df, grouped


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    print("=" * 80)
    print("PROFILE SOCIETY SIMULATION RUNS")
    print("=" * 80)
    print(f"Profiles: {PROFILE_IDS}")
    print(f"Seeds per profile: {SEEDS}")
    print(f"Steps per run: {STEPS}")

    for profile_id in PROFILE_IDS:
        for seed in SEEDS:
            row, evolution = run_one(profile_id, seed)
            rows.append(row)
            evolution_path = OUTPUT_DIR / f"evolution_profile_{profile_id}_seed_{seed}.csv"
            evolution.to_csv(evolution_path, index=True)
            print(
                f"profile={profile_id:>2} seed={seed} alive={row['alive']:>3} "
                f"coop={row.get('coop_rate', 0):.3f} violence={row.get('violence_rate', 0):.3f} "
                f"dark={row.get('final_dark_mach', 0):.3f}/{row.get('final_dark_psycho', 0):.3f} "
                f"formalism={row.get('legal_formalism', 0):.3f} norms={row.get('norm_count', 0):.1f}"
            )

    raw_df, grouped_df = summarize(rows)
    raw_csv = OUTPUT_DIR / "profile_society_runs_raw.csv"
    grouped_csv = OUTPUT_DIR / "profile_society_runs_summary.csv"
    raw_json = OUTPUT_DIR / "profile_society_runs_raw.json"
    grouped_json = OUTPUT_DIR / "profile_society_runs_summary.json"
    raw_df.to_csv(raw_csv, index=False)
    grouped_df.to_csv(grouped_csv, index=False)
    raw_df.to_json(raw_json, orient="records", force_ascii=False, indent=2)
    grouped_df.to_json(grouped_json, orient="records", force_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("AVERAGE RESULTS BY PROFILE")
    print("=" * 80)
    display_cols = [
        "profile_id",
        "profile_name",
        "alive",
        "coop_rate",
        "violence_rate",
        "gossip_rate",
        "negative_gossip_rate",
        "lethality_rate",
        "gini_wealth",
        "legal_formalism",
        "liberty_index",
        "norm_count",
        "power_concentration",
        "political_legitimacy",
        "final_empathy",
        "final_moral_prosocial",
        "final_aggression",
        "final_dark_mach",
        "final_dark_psycho",
    ]
    print(grouped_df[display_cols].to_string(index=False))
    print("\nSaved outputs:")
    for path in [raw_csv, grouped_csv, raw_json, grouped_json]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
