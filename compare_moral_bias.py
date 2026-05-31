#!/usr/bin/env python3
"""Compare prosocial and dark-triad moral-bias scenarios."""
import json
from statistics import mean

from model import Citizen, SocietyModel


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
]


def run_scenario(label, moral_bias, seed, steps=100):
    model = SocietyModel(
        seed=seed,
        population_scale="tiny",
        initial_moral_bias=moral_bias,
        enable_reproduction=False,
        enable_sexual_selection=False,
        coalition_enabled=True,
        enable_guilt=True,
        enable_ostracism=True,
        enable_fermi_update=True,
    )
    initial_agents = list(model.agents)
    initial_trait_means = {
        trait: mean(float(a.latent.get(trait, 0.5)) for a in initial_agents if isinstance(a, Citizen))
        for trait in TRAITS
    }

    for _ in range(steps):
        model.step()

    agents = model.agents_alive()
    df = model.datacollector.get_model_vars_dataframe()
    final = df.iloc[-1].to_dict()
    final_trait_means = {
        trait: mean(float(a.latent.get(trait, 0.5)) for a in agents) if agents else 0.0
        for trait in TRAITS
    }
    result = {
        "label": label,
        "moral_bias": moral_bias,
        "seed": seed,
        "steps": steps,
        "alive": len(agents),
        "regime": model.regime,
        "economic_mechanism": model.economic_mechanism,
        "conflict_resolution": model.conflict_resolution,
        "moral_framework": model.moral_framework,
        "participation_structure": model.political_system.participation_structure,
        "benefit_orientation": model.political_system.benefit_orientation,
        "initial_traits": initial_trait_means,
        "final_traits": final_trait_means,
        "final_metrics": {metric: final.get(metric, None) for metric in METRICS},
    }
    return result


def fmt(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_comparison(results):
    print("=" * 80)
    print("PROSOCIAL VS DARK-TRIAD MORAL-BIAS COMPARISON")
    print("=" * 80)
    for result in results:
        print(f"\nScenario: {result['label']} | bias={result['moral_bias']} | seed={result['seed']}")
        print(f"Alive={result['alive']} | Regime={result['regime']} | Moral framework={result['moral_framework']}")
        print(f"Economic={result['economic_mechanism']} | Conflict={result['conflict_resolution']} | Politics={result['participation_structure']}")
        print("Final metrics:")
        for metric, value in result["final_metrics"].items():
            print(f"  {metric:24} {fmt(value)}")
        print("Trait means: initial -> final")
        for trait in TRAITS:
            before = result["initial_traits"][trait]
            after = result["final_traits"][trait]
            print(f"  {trait:24} {before:.3f} -> {after:.3f}")

    if len(results) == 2:
        a, b = results
        print("\n" + "=" * 80)
        print("DIFFERENCE: dark-triad minus prosocial")
        print("=" * 80)
        for metric in METRICS:
            av = a["final_metrics"].get(metric)
            bv = b["final_metrics"].get(metric)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                print(f"  {metric:24} {bv - av:+.3f}")
        print("Trait differences:")
        for trait in TRAITS:
            diff = b["final_traits"][trait] - a["final_traits"][trait]
            print(f"  {trait:24} {diff:+.3f}")


def main():
    results = [
        run_scenario("Prosocial", "high_prosocial", seed=201, steps=100),
        run_scenario("Dark triad", "high_dark", seed=202, steps=100),
    ]
    print_comparison(results)
    with open("results_moral_bias_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved JSON: results_moral_bias_comparison.json")


if __name__ == "__main__":
    main()
