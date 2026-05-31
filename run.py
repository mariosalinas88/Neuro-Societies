#!/usr/bin/env python3
"""Command-line runner for Neuro-Societies simulations."""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from model import Citizen, SocietyModel


MORAL_BIAS_ALIASES = {
    "highdark": "high_dark",
    "high_dark": "high_dark",
    "lowdark": "low_dark",
    "low_dark": "low_dark",
    "highprosocial": "high_prosocial",
    "high_prosocial": "high_prosocial",
}


def normalize_moral_bias(value):
    if value is None:
        return None
    return MORAL_BIAS_ALIASES.get(str(value).strip().lower(), value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Neuro-Societies simulation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--populationscale", "--population_scale", dest="population_scale", default="tribe",
                        choices=["tiny", "tribe", "city", "nation"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--profile1", type=str, default=None)
    parser.add_argument("--profile2", type=str, default=None)
    parser.add_argument("--profile3", type=str, default=None)
    parser.add_argument("--weight1", type=float, default=0.6)
    parser.add_argument("--weight2", type=float, default=0.3)
    parser.add_argument("--weight3", type=float, default=0.1)
    parser.add_argument("--jitter", type=float, default=0.05)

    parser.add_argument("--spectrumlevel", "--spectrum_level", dest="spectrum_level", type=int,
                        choices=[1, 2, 3], default=None)
    parser.add_argument("--initialmoralbias", "--initial_moral_bias", dest="initial_moral_bias",
                        choices=["highdark", "lowdark", "highprosocial", "high_dark", "low_dark", "high_prosocial"],
                        default=None)
    parser.add_argument("--resiliencebias", "--resilience_bias", dest="resilience_bias",
                        choices=["high", "low"], default=None)
    parser.add_argument("--emotionalbias", "--emotional_bias", dest="emotional_bias",
                        choices=["high", "low"], default=None)

    parser.add_argument("--enablereproduction", "--enable_reproduction", dest="enable_reproduction",
                        action="store_true", default=False)
    parser.add_argument("--enablesexualselection", "--enable_sexual_selection", dest="enable_sexual_selection",
                        action="store_true", default=False)
    parser.add_argument("--maleviolencemultiplier", "--male_violence_multiplier",
                        dest="male_violence_multiplier", type=float, default=1.2)
    parser.add_argument("--coalitionenabled", "--coalition_enabled", dest="coalition_enabled",
                        action="store_true", default=False)

    parser.add_argument("--interactiontopology", "--interaction_topology", dest="interaction_topology",
                        default="grid_local")
    parser.add_argument("--topologyp", "--topology_p", dest="topology_p", type=float, default=None)
    parser.add_argument("--topologyk", "--topology_k", dest="topology_k", type=int, default=None)
    parser.add_argument("--topologym", "--topology_m", dest="topology_m", type=int, default=None)

    parser.add_argument("--enableculturaltransmission", "--enable_cultural_transmission",
                        dest="enable_cultural_transmission", action="store_true", default=False)
    parser.add_argument("--culturallearningrate", "--cultural_learning_rate",
                        dest="cultural_learning_rate", type=float, default=0.05)
    parser.add_argument("--imitationbias", "--imitation_bias", dest="imitation_bias", default="prestige")
    parser.add_argument("--conformitybias", "--conformity_bias", dest="conformity_bias", type=float, default=0.2)
    parser.add_argument("--innovationrate", "--innovation_rate", dest="innovation_rate", type=float, default=0.02)

    parser.add_argument("--enableguilt", "--enable_guilt", dest="enable_guilt", action="store_true", default=False)
    parser.add_argument("--enableostracism", "--enable_ostracism", dest="enable_ostracism", action="store_true", default=False)
    parser.add_argument("--enablefermiupdate", "--enable_fermi_update", dest="enable_fermi_update", action="store_true", default=False)
    parser.add_argument("--fermibeta", "--fermi_beta", dest="fermi_beta", type=float, default=1.0)

    parser.add_argument("--policymode", "--policy_mode", dest="policy_mode", default="none")
    args = parser.parse_args()
    args.initial_moral_bias = normalize_moral_bias(args.initial_moral_bias)
    return args


def topology_params_from_args(args):
    params = {}
    if args.topology_p is not None:
        params["p"] = float(args.topology_p)
    if args.topology_k is not None:
        params["k"] = int(args.topology_k)
    if args.topology_m is not None:
        params["m"] = int(args.topology_m)
    return params


def build_model(args):
    return SocietyModel(
        seed=args.seed,
        climate="stable",
        population_scale=args.population_scale,
        profile1=args.profile1,
        profile2=args.profile2,
        profile3=args.profile3,
        weight1=args.weight1,
        weight2=args.weight2,
        weight3=args.weight3,
        jitter=args.jitter,
        spectrum_level=args.spectrum_level,
        initial_moral_bias=args.initial_moral_bias,
        resilience_bias=args.resilience_bias,
        emotional_bias=args.emotional_bias,
        enable_reproduction=args.enable_reproduction,
        enable_sexual_selection=args.enable_sexual_selection,
        male_violence_multiplier=args.male_violence_multiplier,
        coalition_enabled=args.coalition_enabled,
        interaction_topology=args.interaction_topology,
        topology_params=topology_params_from_args(args),
        enable_cultural_transmission=args.enable_cultural_transmission,
        cultural_learning_rate=args.cultural_learning_rate,
        imitation_bias=args.imitation_bias,
        conformity_bias=args.conformity_bias,
        innovation_rate=args.innovation_rate,
        enable_guilt=args.enable_guilt,
        enable_ostracism=args.enable_ostracism,
        enable_fermi_update=args.enable_fermi_update,
        fermi_beta=args.fermi_beta,
        policy_mode=args.policy_mode,
    )


def summarize_profiles(agents):
    profile_stats = {}
    for agent in agents:
        pid = getattr(agent, "profile_id", "unknown")
        entry = profile_stats.setdefault(
            pid,
            dict(wealth=[], reputation=[], violence_actions=0, nd_contribution=[], nd_cost=[]),
        )
        entry["wealth"].append(agent.wealth)
        entry["reputation"].append(agent.reputation_coop - agent.reputation_fear)
        entry["nd_contribution"].append(agent.nd_contribution)
        entry["nd_cost"].append(agent.nd_cost)
        if agent.last_action == "violence":
            entry["violence_actions"] += 1
    rows = []
    for pid, data in profile_stats.items():
        rows.append(
            {
                "profile": pid,
                "wealth_avg": float(np.mean(data["wealth"])) if data["wealth"] else 0.0,
                "reputation_avg": float(np.mean(data["reputation"])) if data["reputation"] else 0.0,
                "violence_actions": data["violence_actions"],
                "nd_contribution_avg": float(np.mean(data["nd_contribution"])) if data["nd_contribution"] else 0.0,
                "nd_cost_avg": float(np.mean(data["nd_cost"])) if data["nd_cost"] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(model, args):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    df = model.datacollector.get_model_vars_dataframe()
    metadata = getattr(model, "run_metadata", {})
    metadata.update({"requested_steps": args.steps, "seed": args.seed})

    for key, value in metadata.items():
        if isinstance(value, (dict, list, tuple)):
            df[f"meta_{key}"] = json.dumps(value, ensure_ascii=False)
        else:
            df[f"meta_{key}"] = value

    for col in [c for c in df.columns if c.endswith("rate")]:
        df[col] = df[col].clip(lower=0.0, upper=1.0)

    summary_path = f"results/summary_evolution_{timestamp}.csv"
    latest_summary_path = "results/summary_evolution.csv"
    df.to_csv(summary_path, index=True)
    df.to_csv(latest_summary_path, index=True)

    agents = [a for a in model.agents if isinstance(a, Citizen) and a.alive]
    profiles_df = summarize_profiles(agents)
    profiles_path = f"results/per_profile_stats_{timestamp}.csv"
    latest_profiles_path = "results/per_profile_stats.csv"
    profiles_df.to_csv(profiles_path, index=False)
    profiles_df.to_csv(latest_profiles_path, index=False)

    causal = {}
    if hasattr(model, "analyze_cognitive_institutional_causality"):
        causal = model.analyze_cognitive_institutional_causality()
    elif hasattr(model, "analyzecognitiveinstitutionalcausality"):
        causal = model.analyzecognitiveinstitutionalcausality()

    causal_path = f"results/causal_analysis_{timestamp}.json"
    latest_causal_path = "results/causal_analysis.json"
    payload = {"metadata": metadata, "analysis": causal}
    with open(causal_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(latest_causal_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "summary": summary_path,
        "summary_latest": latest_summary_path,
        "profiles": profiles_path,
        "profiles_latest": latest_profiles_path,
        "causal": causal_path,
        "causal_latest": latest_causal_path,
    }


def main():
    args = parse_args()
    model = build_model(args)

    print("Starting Neuro-Societies simulation...")
    if getattr(model, "weight_warning", False):
        print(f"Profile weights normalized; original sum={model.weight_sum_original:.3f}")

    for step in range(args.steps):
        model.step()
        if step == 0 or (step + 1) % 50 == 0 or step == args.steps - 1:
            print(
                f"Step {step + 1} | regime={model.regime} "
                f"formalism={model.legal_formalism:.3f} "
                f"liberty={model.liberty_index:.3f} "
                f"alive={len(model.agents_alive())}"
            )

    agents = [a for a in model.agents if isinstance(a, Citizen) and a.alive]
    print("\n" + "=" * 30 + " FINAL REPORT " + "=" * 30)
    if not agents:
        print("The simulated society has collapsed: no active agents remain.")
    else:
        print(f"Final regime: {model.regime}")
        print(f"Active agents: {len(agents)}")
        print(f"Mean happiness: {float(np.mean([a.happiness for a in agents])):.3f}")
        print(f"Mean wealth: {float(np.mean([a.wealth for a in agents])):.3f}")
        print(f"Mean dark core: {float(np.mean([a.dark_core for a in agents])):.3f}")
        print(f"Mean ND contribution: {float(np.mean([a.nd_contribution for a in agents])):.3f}")
        print(f"Mean ND cost: {float(np.mean([a.nd_cost for a in agents])):.3f}")
        print(f"Legal formalism: {model.legal_formalism:.3f}")
        print(f"Liberty index: {model.liberty_index:.3f}")
        print(f"Wealth Gini: {model.gini_wealth:.3f}")
        print(f"Active norms: {len(model.legal_system.norms)}")

    paths = write_outputs(model, args)
    print("\nOutput files:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
