from __future__ import annotations

import threading
import time
from typing import Dict, List

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import solara

from model import Citizen, SocietyModel, loadprofiles

PROFILEMAP = loadprofiles()


def profilelabel(pid: str) -> str:
    meta = PROFILEMAP.get(pid, {})
    name = meta.get("name") or ""
    return f"{pid} - {name}" if name else pid


def profileoptions() -> List[str]:
    opts: List[str] = []
    for pid in PROFILEMAP.keys():
        opts.append(profilelabel(pid))
    return opts


def profileidfromvalue(value: str) -> str:
    if value in PROFILEMAP:
        return value
    if " " in value:
        return value.split(" ", 1)[0].strip()
    return value.strip()


def profiledescription(value: str) -> str:
    pid = profileidfromvalue(value)
    return PROFILEMAP.get(pid, {}).get("description", "") or ""


PROFILEOPTIONS = profileoptions()

DEFAULTPARAMS: Dict[str, object] = dict(
    seed=42,
    profile1=PROFILEOPTIONS[0] if len(PROFILEOPTIONS) > 0 else "",
    weight1=0.6,
    profile2=PROFILEOPTIONS[1] if len(PROFILEOPTIONS) > 1 else "",
    weight2=0.3,
    profile3=PROFILEOPTIONS[2] if len(PROFILEOPTIONS) > 2 else "",
    weight3=0.1,
    climate="stable",
    externalfactor="none",
    populationscale="tribe",
    enablereproduction=False,
    enablesexualselection=False,
    maleviolencemultiplier=1.2,
    coalitionenabled=False,
    mateweightwealth=0.4,
    mateweightdom=0.3,
    mateweighthealth=0.2,
    mateweightage=0.1,
    matechoicebeta=1.0,
    femalereprocooldown=10,
    malereprocooldown=2,
    reprobaseoffset=0.2,
    reprodesirescale=0.3,
    maleinitiationbase=0.05,
    maledesirescale=0.3,
    neurodecayk=0.1,
    bondingsteps=5,
    bondingdelta=0.02,
    enablecoercion=False,
)


def makegridfigure(model: SocietyModel) -> Figure:
    fig = Figure(figsize=(5.5, 5.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    xs, ys, colors, sizes = [], [], [], []
    for agent in model.agentsalive:
        if not isinstance(agent, Citizen) or agent.pos is None:
            continue
        x, y = agent.pos
        xs.append(x)
        ys.append(y)
        if getattr(agent, "allianceid", None):
            seed = abs(hash(agent.allianceid)) % 2**32
            rng = np.random.default_rng(seed)
            base = rng.random(3) * 0.5 + 0.4
            colors.append(tuple(base.tolist()))
        else:
            r = float(np.clip(agent.latent.get("dominance", 0.5), 0, 1))
            g = float(np.clip(agent.latent.get("empathy", 0.5), 0, 1))
            b = float(np.clip(agent.latent.get("language", 0.5), 0, 1))
            colors.append((r, g, b))
        sizes.append(40 * float(np.clip(agent.wealth, 0.2, 3.0)))
    if xs:
        ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.85, edgecolors="k", linewidths=0.4)
    ax.set_xlim(-0.5, model.grid.width - 0.5)
    ax.set_ylim(-0.5, model.grid.height - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Grid: color(dom/empa/idioma), tamaño=riqueza")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.invert_yaxis()
    return fig


def makelinefigure(history: pd.DataFrame, column: str, title: str, color: str) -> Figure:
    fig = Figure(figsize=(4.5, 3))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    if column in history:
        ax.plot(history.index, history[column], color=color, linewidth=2)
        ax.set_ylabel(column)
    else:
        ax.text(0.5, 0.5, "sin datos", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


@solara.component
def InfoPanel(model: SocietyModel):
    m = model.lastmetrics or {}
    return solara.Card(
        title="Métricas",
        children=[
            solara.Markdown(f"**Régimen:** {model.regime}"),
            solara.Markdown(
                f"Formalismo legal={model.legalformalism:.3f} | "
                f"Libertad={model.libertyindex:.3f}"
            ),
            solara.Markdown(
                f"Cooperación={m.get('cooprate', 0):.2f} | "
                f"Violencia={m.get('violencerate', 0):.2f}"
            ),
            solara.Markdown(
                f"Violencia M/F={m.get('maleviolencerate', 0):.2f} / "
                f"{m.get('femaleviolencerate', 0):.2f}"
            ),
            solara.Markdown(
                f"Población escala={len(model.agentsalive)*model.scalefactor:.0f}"
            ),
            solara.Markdown(
                f"Gini={getattr(model, 'giniwealth', 0.0):.3f} "
                f"Vida={m.get('lifeexpectancy', 0):.1f}"
            ),
        ],
    )


def summarytext(model: SocietyModel, history: pd.DataFrame, window: int = 30) -> str:
    if history is None or history.empty:
        return "Sin datos."
    tail = history.tail(window)
    last = tail.iloc[-1]

    def pctval(val):
        return f"{val:.2f}"

    coopmean = pctval(tail["cooprate"].mean()) if "cooprate" in tail else "nd"
    violmean = pctval(tail["violencerate"].mean()) if "violencerate" in tail else "nd"
    ginilast = f"{last.get('giniwealth', 0):.3f}"
    regime = last.get("regime", model.regime)
    alliances = last.get("alliancescount", 0)
    awareness = last.get("consciousawarenessmean", last.get("consciousawareness", 0))

    return " | ".join(
        [
            f"Régimen={regime}",
            f"Coop media ult.{len(tail)}={coopmean}",
            f"Violencia={violmean}",
            f"Gini={ginilast}",
            f"Alianzas={alliances}",
            f"Conciencia={awareness:.3f}",
        ]
    )


def profilemetrics(model: SocietyModel) -> pd.DataFrame:
    rows = []
    for a in model.agentsalive:
        pid = getattr(a, "profileid", "na")
        rows.append(
            dict(
                profile=pid,
                wealth=a.wealth,
                repnet=a.reputationcoop - a.reputationfear,
                fear=a.reputationfear,
                coop=a.reputationcoop,
                darkcore=a.darkcore,
                violence=1 if a.lastaction == "violence" else 0,
            )
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("profile").agg(
        wealthmean=("wealth", "mean"),
        wealthgini=("wealth", lambda s: float(
            np.abs(np.subtract.outer(s, s)).sum() / (2 * len(s) ** 2 * (s.mean() or 1e-6))
        )),
        repmean=("repnet", "mean"),
        fearmean=("fear", "mean"),
        coopmean=("coop", "mean"),
        darkcoremean=("darkcore", "mean"),
        violencerate=("violence", "mean"),
        count=("wealth", "count"),
    )
    return agg.reset_index()


@solara.component
def SummaryPanel(model: SocietyModel, history: pd.DataFrame):
    with solara.Card(title="Resumen"):
        solara.Markdown(summarytext(model, history))
        cols = [
            "population",
            "cooprate",
            "violencerate",
            "maleviolencerate",
            "femaleviolencerate",
            "giniwealth",
            "lifeexpectancy",
            "legalformalism",
            "libertyindex",
            "totalwealth",
            "ndcontributionmean",
            "alliancescount",
            "alliedshare",
            "consciousawarenessmean",
            "births",
            "sexratio",
        ]
        cols = [c for c in cols if c in history.columns]
        if cols:
            solara.DataFrame(history[cols].tail(8).reset_index(drop=True))
        profdf = profilemetrics(model)
        if not profdf.empty:
            solara.Markdown("**Perfiles** (wealth, rep, miedo, coop, dark, violencia)")
            solara.DataFrame(profdf)


@solara.component
def Controls(paramsstate, resetmodel):
    p = paramsstate.value

    def setparam(key, value):
        paramsstate.value = {**paramsstate.value, key: value}

    solara.Markdown("### Parámetros")
    solara.InputInt("Semilla", value=int(p["seed"]),
                    on_value=lambda v: setparam("seed", int(v)))

    solara.Select("Clima", value=p["climate"],
                  values=["scarce", "stable", "abundant"],
                  on_value=lambda v: setparam("climate", v))

    solara.Select("Factor externo", value=p["externalfactor"],
                  values=["none", "disaster", "epidemic", "technological"],
                  on_value=lambda v: setparam("externalfactor", v))

    solara.Select("Escala", value=p["populationscale"],
                  values=["tiny", "tribe", "city", "nation"],
                  on_value=lambda v: setparam("populationscale", v))

    solara.Checkbox(
        label="Reproducción",
        value=bool(p["enablereproduction"]),
        on_value=lambda v: setparam("enablereproduction", bool(v)),
    )
    solara.Checkbox(
        label="Selección sexual",
        value=bool(p["enablesexualselection"]),
        on_value=lambda v: setparam("enablesexualselection", bool(v)),
    )

    solara.SliderFloat(
        "Violencia masculina x",
        value=float(p["maleviolencemultiplier"]),
        min=0.5,
        max=2.0,
        step=0.05,
        on_value=lambda v: setparam("maleviolencemultiplier", float(v)),
    )
    solara.Checkbox(
        label="Coaliciones",
        value=bool(p["coalitionenabled"]),
        on_value=lambda v: setparam("coalitionenabled", bool(v)),
    )

    solara.Markdown("**Preferencia femenina (pesos softmax)**")
    solara.SliderFloat(
        "w_wealth",
        value=float(p["mateweightwealth"]),
        min=0.0,
        max=1.0,
        step=0.05,
        on_value=lambda v: setparam("mateweightwealth", float(v)),
    )
    solara.SliderFloat(
        "w_dom",
        value=float(p["mateweightdom"]),
        min=0.0,
        max=1.0,
        step=0.05,
        on_value=lambda v: setparam("mateweightdom", float(v)),
    )
    solara.SliderFloat(
        "w_health",
        value=float(p["mateweighthealth"]),
        min=0.0,
        max=1.0,
        step=0.05,
        on_value=lambda v: setparam("mateweighthealth", float(v)),
    )
    solara.SliderFloat(
        "w_age",
        value=float(p["mateweightage"]),
        min=0.0,
        max=1.0,
        step=0.05,
        on_value=lambda v: setparam("mateweightage", float(v)),
    )
    solara.SliderFloat(
        "beta",
        value=float(p["matechoicebeta"]),
        min=0.0,
        max=3.0,
        step=0.1,
        on_value=lambda v: setparam("matechoicebeta", float(v)),
    )

    solara.Markdown("**Reproducción / deseo**")
    solara.SliderInt(
        "Cooldown F", value=int(p["femalereprocooldown"]),
        min=1, max=30, step=1,
        on_value=lambda v: setparam("femalereprocooldown", int(v)),
    )
    solara.SliderInt(
        "Cooldown M", value=int(p["malereprocooldown"]),
        min=1, max=20, step=1,
        on_value=lambda v: setparam("malereprocooldown", int(v)),
    )
    solara.SliderFloat(
        "Base offset", value=float(p["reprobaseoffset"]),
        min=-0.5, max=0.5, step=0.05,
        on_value=lambda v: setparam("reprobaseoffset", float(v)),
    )
    solara.SliderFloat(
        "Desire scale", value=float(p["reprodesirescale"]),
        min=0.0, max=1.0, step=0.05,
        on_value=lambda v: setparam("reprodesirescale", float(v)),
    )

    solara.Markdown("**Iniciativa masculina**")
    solara.SliderFloat(
        "init_base", value=float(p["maleinitiationbase"]),
        min=0.0, max=0.5, step=0.02,
        on_value=lambda v: setparam("maleinitiationbase", float(v)),
    )
    solara.SliderFloat(
        "init_desire_scale", value=float(p["maledesirescale"]),
        min=0.0, max=1.0, step=0.05,
        on_value=lambda v: setparam("maledesirescale", float(v)),
    )

    solara.Markdown("**Neurobonding**")
    solara.SliderFloat(
        "neurodecayk",
        value=float(p["neurodecayk"]),
        min=0.01,
        max=0.5,
        step=0.01,
        on_value=lambda v: setparam("neurodecayk", float(v)),
    )
    solara.SliderInt(
        "bondingsteps",
        value=int(p["bondingsteps"]),
        min=0,
        max=20,
        step=1,
        on_value=lambda v: setparam("bondingsteps", int(v)),
    )
    solara.SliderFloat(
        "bondingdelta",
        value=float(p["bondingdelta"]),
        min=0.0,
        max=0.2,
        step=0.01,
        on_value=lambda v: setparam("bondingdelta", float(v)),
    )
    solara.Checkbox(
        label="Enable coercion (rare)",
        value=bool(p["enablecoercion"]),
        on_value=lambda v: setparam("enablecoercion", bool(v)),
    )

    solara.Markdown("**Perfiles**")
    solara.Select(
        "Perfil 1",
        value=p["profile1"],
        values=PROFILEOPTIONS,
        on_value=lambda v: setparam("profile1", v),
    )
    desc1 = profiledescription(p["profile1"])
    if desc1:
        solara.Markdown(desc1)
    solara.SliderFloat(
        "Peso 1",
        value=float(p["weight1"]),
        min=0.0,
        max=1.0,
        step=0.05,
        on_value=lambda v: setparam("weight1", float(v)),
    )

    solara.Select(
        "Perfil 2",
        value=p["profile2"],
        values=PROFILEOPTIONS,
        on_value=lambda v: setparam("profile2", v),
    )
    desc2 = profiledescription(p["profile2"])
    if desc2:
        solara.Markdown(desc2)
    solara.SliderFloat(
        "Peso 2",
        value=float(p["weight2"]),
        min=0.0,
        max=1.0,
        step=0.05,
        on_value=lambda v: setparam("weight2", float(v)),
    )

    solara.Select(
        "Perfil 3",
        value=p["profile3"],
        values=PROFILEOPTIONS,
        on_value=lambda v: setparam("profile3", v),
    )
    desc3 = profiledescription(p["profile3"])
    if desc3:
        solara.Markdown(desc3)
    solara.SliderFloat(
        "Peso 3",
        value=float(p["weight3"]),
        min=0.0,
        max=1.0,
        step=0.05,
        on_value=lambda v: setparam("weight3", float(v)),
    )

    solara.Button("Aplicar y reiniciar", icon_name="refresh",
                  on_click=resetmodel, color="primary", text=True)


@solara.component
def Page():
    paramsstate = solara.use_reactive(dict(DEFAULTPARAMS))
    simstate = solara.use_reactive(dict(history=None, steps=0))
    modelref = solara.use_ref(None)

    def buildmodel(params: Dict[str, object]):
        clean = dict(params)
        clean["profile1"] = profileidfromvalue(str(clean.get("profile1", "")))
        clean["profile2"] = profileidfromvalue(str(clean.get("profile2", "")))
        clean["profile3"] = profileidfromvalue(str(clean.get("profile3", "")))
        model = SocietyModel(**clean)
        history = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        modelref.current = model
        simstate.value = dict(history=history, steps=0)

    def ensuremodel():
        if modelref.current is None:
            buildmodel(paramsstate.value)

    solara.use_effect(ensuremodel, [])

    def resetmodel():
        buildmodel(paramsstate.value)

    def stepmodel(n: int = 1):
        model = modelref.current
        if model is None:
            return
        stepsdone = 0
        for _ in range(n):
            if not model.running:
                break
            model.step()
            stepsdone += 1
        history = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        simstate.value = dict(history=history,
                              steps=simstate.value["steps"] + stepsdone)

    model = modelref.current
    history = simstate.value["history"]

    if model is None or history is None:
        solara.Text("Inicializando modelo...")
        return

    with solara.Column(gap="1.25rem"):
        solara.Markdown("# Neuro Societies - Simulación Solara")
        with solara.Row(gap="1rem"):
            with solara.Column(gap="0.8rem", style={"minWidth": "320px"}):
                Controls(paramsstate=paramsstate, resetmodel=resetmodel)
                solara.Button("Step", on_click=lambda: stepmodel(1))
                solara.Button("Step x50", on_click=lambda: stepmodel(50),
                              text=True, color="primary")
                solara.Button("Run 1000 steps", on_click=lambda: stepmodel(1000),
                              text=True, color="primary")
                solara.Button("Reset modelo", on_click=resetmodel,
                              icon_name="refresh", color="warning", text=True)
                solara.Markdown(f"**Steps ejecutados:** {simstate.value['steps']}")

            with solara.Column(gap="1rem", style={"alignItems": "stretch"}):
                InfoPanel(model=model)
                solara.FigureMatplotlib(makegridfigure(model))
                SummaryPanel(model=model, history=history)

        with solara.Tabs():
            with solara.Tab("Desarrollo ND"):
                solara.FigureMatplotlib(
                    makelinefigure(history, "totalwealth", "Total Wealth Desarrollo", "#059669")
                )
                solara.FigureMatplotlib(
                    makelinefigure(history, "ndcontributionmean", "ND Contribución media", "#22c55e")
                )
            with solara.Tab("Dinámica Social"):
                solara.FigureMatplotlib(
                    makelinefigure(history, "cooprate", "Cooperación", "#16a34a")
                )
                solara.FigureMatplotlib(
                    makelinefigure(history, "violencerate", "Violencia", "#dc2626")
                )
            with solara.Tab("Economía / Poder"):
                solara.FigureMatplotlib(
                    makelinefigure(history, "giniwealth", "Gini riqueza", "#9333ea")
                )
                solara.FigureMatplotlib(
                    makelinefigure(history, "top5wealthshare", "Top5 share", "#f59e0b")
                )

        with solara.Row():
            solara.FigureMatplotlib(
                makelinefigure(history, "population", "Población", "#2563eb")
            )
            solara.FigureMatplotlib(
                makelinefigure(history, "cooprate", "Cooperación", "#16a34a")
            )


if __name__ == "__main__":
    print("Ejecuta: python -m solara run server:Page")
