# Neuro Societies – Agent-Based Model of Neurodiversity and Emergent Institutions

This repository contains an agent‑based model (ABM) that simulates how neurocognitive diversity, social interaction patterns, and institutional feedback co‑evolve into emergent social regimes (e.g., democratic, oligarchic, authoritarian, collapsed). The core model is implemented with Mesa (Python), with optional interactive visualization via Solara.

## Features

- Heterogeneous agents with:
  - Continuous latent traits (attention, sociality, language, reasoning, empathy, dominance, impulsivity, dark triad, etc.).
  - Neurodiversity parameters (spectrum level, bias ranges, rare variants).
  - Dual reputation system (cooperation vs. fear‑based status).
  - Neurochemical state (dopamine, oxytocin, serotonin, endorphins) and happiness.
- Life‑cycle and reproduction:
  - Ageing, mortality and health.
  - Sexual selection with female preference functions and male initiative.
  - Fertility cooldowns, gestation, reproduction costs.
- Social interaction and conflict:
  - Local or graph‑based topologies (grid, small‑world, scale‑free, etc., if networkx is available).
  - Cooperation, defection and violence.
  - Alliances and coalitions with prosocial vs. dominance goals.
- Institutions and regimes:
  - Emergent legal system (norms, formalism, consistency, centralization).
  - Political system (participation structure, benefit orientation, legitimacy).
  - Economic patterns (redistribution modes, inequality, top‑5 wealth share).
  - Endogenous regime classification (collapse, tyranny, oligarchy, technocracy, democracy, etc.).
- Cultural and policy mechanisms (optional):
  - Cultural transmission and imitation.
  - Policy modes and schedules (for “wind tunnel” evaluation of interventions).
- Data outputs:
  - Time‑series of macro variables (cooperation, violence, inequality, population, etc.).
  - Per‑profile statistics (wealth, reputation, dark core, violence rate).
  - Institutional trajectory and causal analysis JSON.

## Repository structure

```text
.
├── model.py                    # Core Mesa model and agent classes
├── run.py                      # CLI runner to execute batch simulations
├── server.py                   # Optional Solara dashboard for interactive exploration
├── governance_windtunnel.py    # Policy “wind tunnel” evaluation helpers
├── profiles.json               # Cognitive / neurodiversity profiles
├── requirements.txt
├── README.md
└── results/                    # Created automatically by run.py (CSV/JSON outputs)
