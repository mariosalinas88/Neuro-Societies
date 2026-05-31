from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

from model_asperger import PsychNeuroSociety


def run_once(prosocial_count: int, dark_count: int, seed: int, steps: int) -> Dict[str, object]:
    total = prosocial_count + dark_count
    if total <= 0:
        raise ValueError("Total agents must be positive")

    model = PsychNeuroSociety(
        seed=seed,
        population_scale="tiny",
        profile1="9",  # high empathy / prosocial
        weight1=prosocial_count / total,
        profile2="10",  # dark triad / dominance
        weight2=dark_count / total,
        profile3="",
        weight3=0.0,
    )

    # Force the exact requested ratio because weighted random sampling can deviate in tiny populations.
    for idx, agent in enumerate(model.agents):
        if idx < prosocial_count:
            agent.profile_id = "9"
            traits = model.profiles["9"]["traits"]
            agent.traits = dict(traits)
        else:
            agent.profile_id = "10"
            traits = model.profiles["10"]["traits"]
            agent.traits = dict(traits)

    history = model.run(steps)
    last = history.iloc[-1].to_dict() if not history.empty else {}
    per_profile = model.per_profile_stats()

    p9_wealth = None
    p10_wealth = None
    p9_fear = None
    p10_fear = None
    if not per_profile.empty:
        for _, row in per_profile.iterrows():
            if str(row["profile"]) == "9":
                p9_wealth = float(row["wealth_mean"])
                p9_fear = float(row["fear_rep_mean"])
            elif str(row["profile"]) == "10":
                p10_wealth = float(row["wealth_mean"])
                p10_fear = float(row["fear_rep_mean"])

    return {
        "prosocial_count": prosocial_count,
        "dark_count": dark_count,
        "dark_share": dark_count / total,
        "seed": seed,
        "steps": steps,
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
        "p9_wealth_mean": p9_wealth,
        "p10_wealth_mean": p10_wealth,
        "p9_fear_mean": p9_fear,
        "p10_fear_mean": p10_fear,
    }


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["prosocial_count"], row["dark_count"])].append(row)

    numeric = [
        "coop_rate", "violence_rate", "defection_rate", "support_rate", "avoidance_rate",
        "gini_wealth", "wealth_mean", "empathy_mean", "reasoning_mean", "dominance_mean",
        "dark_core_mean", "p9_wealth_mean", "p10_wealth_mean", "p9_fear_mean", "p10_fear_mean",
    ]
    out = []
    for (prosocial_count, dark_count), vals in sorted(buckets.items(), reverse=True):
        item: Dict[str, object] = {
            "prosocial_count": prosocial_count,
            "dark_count": dark_count,
            "dark_share": dark_count / (prosocial_count + dark_count),
            "runs": len(vals),
            "dominant_regime": Counter(str(v["regime"]) for v in vals).most_common(1)[0][0],
        }
        for key in numeric:
            nums = [float(v[key]) for v in vals if v.get(key) is not None]
            item[f"{key}_mean"] = sum(nums) / len(nums) if nums else None
            item[f"{key}_min"] = min(nums) if nums else None
            item[f"{key}_max"] = max(nums) if nums else None
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ratio experiment: high-empathy/prosocial agents vs dark-triad agents.")
    parser.add_argument("--runs_per_ratio", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: List[Dict[str, object]] = []

    for dark_count in range(0, 11):
        prosocial_count = 10 - dark_count
        for _ in range(args.runs_per_ratio):
            rows.append(
                run_once(
                    prosocial_count=prosocial_count,
                    dark_count=dark_count,
                    seed=rng.randint(1, 2_000_000_000),
                    steps=args.steps,
                )
            )

    agg = aggregate(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "ratio_prosocial_dark_raw.csv")
    agg_path = os.path.join(args.output_dir, "ratio_prosocial_dark_aggregate.csv")

    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)

    print(f"Saved raw results: {raw_path}")
    print(f"Saved aggregate results: {agg_path}")
    print("\nSummary:")
    for row in agg:
        print(
            f"{row['prosocial_count']} prosocial / {row['dark_count']} dark | "
            f"dark_share={row['dark_share']:.1f} | "
            f"violence={row['violence_rate_mean']:.3f} | "
            f"defection={row['defection_rate_mean']:.3f} | "
            f"support={row['support_rate_mean']:.3f} | "
            f"gini={row['gini_wealth_mean']:.3f} | "
            f"wealth={row['wealth_mean_mean']:.3f} | "
            f"regime={row['dominant_regime']}"
        )


if __name__ == "__main__":
    main()
