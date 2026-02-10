import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

from model import Citizen, SocietyModel


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)

parser.add_argument("--spectrumlevel", type=int, choices=[1, 2, 3], default=None)
parser.add_argument("--initialmoralbias", type=str,
                    choices=["highdark", "lowdark", "highprosocial"],
                    default=None)
parser.add_argument("--resiliencebias", type=str,
                    choices=["high", "low"],
                    default=None)
parser.add_argument("--emotionalbias", type=str,
                    choices=["high", "low"],
                    default=None)

parser.add_argument("--enablereproduction", action="store_true", default=False)
parser.add_argument("--enablesexualselection", action="store_true", default=False)
parser.add_argument("--maleviolencemultiplier", type=float, default=1.2)
parser.add_argument("--coalitionenabled", action="store_true", default=False)

parser.add_argument("--interactiontopology", type=str, default="gridlocal")
parser.add_argument("--topologyp", type=float, default=None)
parser.add_argument("--topologyk", type=int, default=None)
parser.add_argument("--topologym", type=int, default=None)

parser.add_argument("--enableculturaltransmission", action="store_true", default=False)
parser.add_argument("--culturallearningrate", type=float, default=0.05)
parser.add_argument("--imitationbias", type=str, default="prestige")
parser.add_argument("--conformitybias", type=float, default=0.2)
parser.add_argument("--innovationrate", type=float, default=0.02)

parser.add_argument("--enableguilt", action="store_true", default=False)
parser.add_argument("--enableostracism", action="store_true", default=False)
parser.add_argument("--enablefermiupdate", action="store_true", default=False)
parser.add_argument("--fermibeta", type=float, default=1.0)

parser.add_argument("--policymode", type=str, default="none")

args, unknown = parser.parse_known_args()


def main() -> None:
    topologyparams = {}
    if args.topologyp is not None:
        topologyparams["p"] = float(args.topologyp)
    if args.topologyk is not None:
        topologyparams["k"] = int(args.topologyk)
    if args.topologym is not None:
        topologyparams["m"] = int(args.topologym)

    model = SocietyModel(
        width=30,
        height=30,
        seed=42,
        climate="stable",
        spectrumlevel=args.spectrumlevel,
        initialmoralbias=args.initialmoralbias,
        resiliencebias=args.resiliencebias,
        emotionalbias=args.emotionalbias,
        enablereproduction=args.enablereproduction,
        enablesexualselection=args.enablesexualselection,
        maleviolencemultiplier=args.maleviolencemultiplier,
        coalitionenabled=args.coalitionenabled,
        interactiontopology=args.interactiontopology,
        topologyparams=topologyparams,
        enableculturaltransmission=args.enableculturaltransmission,
        culturallearningrate=args.culturallearningrate,
        imitationbias=args.imitationbias,
        conformitybias=args.conformitybias,
        innovationrate=args.innovationrate,
        enableguilt=args.enableguilt,
        enableostracism=args.enableostracism,
        enablefermiupdate=args.enablefermiupdate,
        fermibeta=args.fermibeta,
        policymode=args.policymode,
    )

    print("Iniciando simulación evolutiva...")

    if getattr(model, "weightwarning", False):
        print(f"Pesos normalizados, suma original={model.weightsumoriginal:.3f}")

    for step in range(args.steps):
        model.step()
        if step % 50 == 0:
            print(
                f"Step {step} | Régimen={model.regime} "
                f"Formalismo={model.legalformalism:.2f} "
                f"Libertad={model.libertyindex:.2f}"
            )

    agents = [a for a in model.agents if isinstance(a, Citizen) and a.alive]

    print("\n" + "=" * 30 + " REPORTE DE NEUROPLASTICIDAD " + "=" * 30)

    neutralkeys = [
        "attnselective", "attnflex", "hyperfocus", "impulsivity",
        "riskaversion", "sociality", "language", "reasoning",
        "emotionalimpulsivity", "resilience", "sexualimpulsivity",
    ]
    moralkeys = [
        "empathy", "dominance", "affectreg", "aggression",
        "moralprosocial", "moralcommongood", "moralhonesty", "moralspite",
        "darknarc", "darkmach", "darkpsycho",
    ]

    def printblock(blockkeys, title):
        print(f"-- {title} --")
        print(f"{'Rasgo':20} {'Inicial':10} {'Final':10} {'Cambio %':10}")
        print("-" * 65)
        for trait in sorted(blockkeys):
            avginit = np.mean([a.originallatent.get(trait, 0.5) for a in agents])
            avgcurr = np.mean([a.latent.get(trait, 0.5) for a in agents])
            deltapct = ((avgcurr - avginit) / avginit * 100) if avginit != 0 else 0
            print(f"{trait:20} {avginit:10.3f} {avgcurr:10.3f} {deltapct:10.2f}")
        print()

    if agents:
        printblock(neutralkeys, "Cambios Neutrales/Biológicos")
        printblock(moralkeys, "Cambios Morales/Emocionales")

        avghappy = np.mean([a.happiness for a in agents])
        avgwealth = np.mean([a.wealth for a in agents])
        avgdarkcore = np.mean([a.darkcore for a in agents])
        ndcontrib = np.mean([a.ndcontribution for a in agents]) if agents else 0.0
        ndcosts = np.mean([a.ndcost for a in agents]) if agents else 0.0
        consciousness = np.mean(
            [a.consciouscore.get("awareness", 0.0) for a in agents]
        ) if agents else 0.0

        profilestats = {}
        for a in agents:
            pid = getattr(a, "profileid", "unknown")
            entry = profilestats.setdefault(
                pid,
                dict(wealth=[], reputation=[], victim=0, leader=0,
                     ndcontrib=[], ndcost=[])
            )
            entry["wealth"].append(a.wealth)
            entry["reputation"].append(a.reputationcoop - a.reputationfear)
            entry["ndcontrib"].append(a.ndcontribution)
            entry["ndcost"].append(a.ndcost)
            if a.lastaction == "violence":
                entry["leader"] += 1
                entry["victim"] += 1

        print("-- Métricas por perfil (wealth, reputación, liderazgo, víctimas, ND contrib/costo) --")
        for pid, data in profilestats.items():
            wmean = np.mean(data["wealth"]) if data["wealth"] else 0.0
            rmean = np.mean(data["reputation"]) if data["reputation"] else 0.0
            leaders = data["leader"]
            victims = data["victim"]
            ndc = np.mean(data["ndcontrib"]) if data["ndcontrib"] else 0.0
            ndcost = np.mean(data["ndcost"]) if data["ndcost"] else 0.0
            print(
                f"Perfil {pid:15} wealth={wmean:.2f}, rep={rmean:.2f}, "
                f"líder_violencia={leaders}, víctimas={victims}, "
                f"nd_contrib={ndc:.3f}, nd_cost={ndcost:.3f}"
            )

        print(f"\nPromedio Final felicidad={avghappy:.3f}")
        print(f"Wealth Promedio Final={avgwealth:.3f}")
        print(f"Dark core medio={avgdarkcore:.3f}")
        print(f"ND contribución media={ndcontrib:.3f} | ND costo={ndcosts:.3f}")
        print(f"Conciencia media={consciousness:.3f}")

        alliances = getattr(model, "alliances", {})
        print(f"Alianzas activas={len(alliances)}")
        for aid, data in alliances.items():
            members = data.get("members", [])
            goal = data.get("goal")
            rule = data.get("rule")
            print(f" - {aid} miembros={len(members)} objetivo={goal} norma={rule}")

        print(f"\nRégimen Final Establecido: {model.regime}")
        causal = model.analyzecognitiveinstitutionalcausality()
        print("\n" + "=" * 30 + " ARQUITECTURA INSTITUCIONAL " + "=" * 30)
        print("-- Perfil cognitivo dominante top 20 por poder --")
        for trait, value in causal.get("cognitive_profile", {}).items():
            print(f"{trait:22} {value:6.3f}")
        print("-- Sistema legal emergente --")
        print(f"Formalismo={model.legalsystem.formalismindex:.3f}")
        print(f"Normas activas={len(model.legalsystem.norms)}")
        print(f"Consistencia={model.normconsistency:.3f}")
        print(f"Centralización={model.normcentralization:.3f}")
        print("-- Sistema económico detectado --")
        print(f"Mecanismo dominante={model.economicmechanism}")
        print("-- Sistema político detectado --")
        print(f"Participación={model.politicalsystem.participationstructure}")
        print(f"Beneficio={model.politicalsystem.benefitorientation}")
        print(f"Legitimidad={model.politicalsystem.legitimacy:.3f}")
        print("-- Correlación causal cognición - arquitectura --")
        print(causal.get("causal_narrative", ""))

    else:
        print("La sociedad se ha extinguido!")

    df = model.datacollector.get_model_vars_dataframe()
    os.makedirs("results", exist_ok=True)
    meta = getattr(model, "runmetadata", {})

    for k, v in meta.items():
        df[k] = v

    categoricalcols = [
        "regime", "economicmechanism", "participationstructure",
        "benefitorientation", "conflictresolution", "moralframework",
    ]
    for col in categoricalcols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    ratecols = [c for c in df.columns if c.endswith("rate")]
    for col in ratecols:
        df[col] = df[col].clip(lower=0.0, upper=1.0)

    try:
        df.to_csv("results/summary_evolution.csv")
    except PermissionError:
        alt = f"results/summary_evolution_{int(np.random.randint(1e9))}.csv"
        df.to_csv(alt)

    if agents:
        try:
            rows = []
            for idx, row in df.iterrows():
                rows.append(
                    dict(
                        step=idx,
                        **{f"meta_{k}": v for k, v in meta.items()},
                        cognitiveprofilevector=json.dumps(
                            row.get("cognitiveprofiletop20", []),
                            ensure_ascii=False,
                        ),
                        institutionalarchitecturevector=json.dumps(
                            dict(
                                formalism=row.get("legalformalism", 0.0),
                                normcount=row.get("normcount", 0.0),
                                normconsistency=row.get("normconsistency", 0.0),
                                economicmechanism=row.get("economicmechanism", "mixed"),
                                participationstructure=row.get("participationstructure", ""),
                                benefitorientation=row.get("benefitorientation", ""),
                                powerconcentration=row.get("powerconcentration", 0.0),
                                conflictresolution=row.get("conflictresolution", "consensus"),
                                moralframework=row.get("moralframework", "mixed"),
                            ),
                            ensure_ascii=False,
                        ),
                    )
                )
            pd.DataFrame(rows).to_csv(
                "results/institutional_trajectory.csv", index=False
            )
        except Exception:
            pass

        try:
            with open("results/causal_analysis.json", "w", encoding="utf-8") as f:
                payload = {"metadata": meta, "analysis": causal}
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        calibreport = getattr(model, "calibrationreport", None)
        if calibreport:
            try:
                with open("results/calibration_report.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {"metadata": meta, "calibration": calibreport},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                pass

        with open("results/per_profile_stats.csv", "w", newline="", encoding="utf-8") as f:
            metafields = [f"meta_{k}" for k in meta.keys()]
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "profile",
                    "wealthavg",
                    "leadershipavg",
                    "repavg",
                    "victimsavg",
                    "ndcontribavg",
                    "ndcostavg",
                    *metafields,
                ],
            )
            writer.writeheader()
            for pid, data in profilestats.items():
                writer.writerow(
                    dict(
                        profile=pid,
                        wealthavg=np.mean(data["wealth"]) if data["wealth"] else 0.0,
                        leadershipavg=data["leader"],
                        repavg=np.mean(data["reputation"]) if data["reputation"] else 0.0,
                        victimsavg=data["victim"],
                        ndcontribavg=np.mean(data["ndcontrib"]) if data["ndcontrib"] else 0.0,
                        ndcostavg=np.mean(data["ndcost"]) if data["ndcost"] else 0.0,
                        **{f"meta_{k}": v for k, v in meta.items()},
                    )
                )

        print(
            "Datos guardados en results/summary_evolution.csv, "
            "results/institutional_trajectory.csv, "
            "results/causal_analysis.json y "
            "results/per_profile_stats.csv"
        )


if __name__ == "__main__":
    main()
