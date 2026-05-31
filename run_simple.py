from __future__ import annotations

import argparse
import os

from model_asperger import PsychNeuroSociety


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simplified psychological/neurocognitive simulation.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population_scale", choices=["tiny", "small", "tribe", "city"], default="small")
    parser.add_argument("--profiles_path", default="profiles.json")
    parser.add_argument("--profile1", default=None)
    parser.add_argument("--profile2", default=None)
    parser.add_argument("--profile3", default=None)
    parser.add_argument("--weight1", type=float, default=0.6)
    parser.add_argument("--weight2", type=float, default=0.3)
    parser.add_argument("--weight3", type=float, default=0.1)
    parser.add_argument("--spectrum_level", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--jitter", type=float, default=0.05)
    args = parser.parse_args()

    model = PsychNeuroSociety(
        seed=args.seed,
        profiles_path=args.profiles_path,
        population_scale=args.population_scale,
        profile1=args.profile1,
        profile2=args.profile2,
        profile3=args.profile3,
        weight1=args.weight1,
        weight2=args.weight2,
        weight3=args.weight3,
        jitter=args.jitter,
        spectrum_level=args.spectrum_level,
    )

    print("Running simplified psych-neuro simulation...")
    history = model.run(args.steps)

    os.makedirs("results", exist_ok=True)
    history_path = "results/simple_summary_evolution.csv"
    profiles_path = "results/simple_per_profile_stats.csv"

    history.to_csv(history_path, index=False)
    model.per_profile_stats().to_csv(profiles_path, index=False)

    last = history.iloc[-1].to_dict() if not history.empty else {}
    print("Final metrics:")
    for key in [
        "regime",
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
    ]:
        print(f"  {key}: {last.get(key)}")

    print(f"\nSaved: {history_path}")
    print(f"Saved: {profiles_path}")


if __name__ == "__main__":
    main()
