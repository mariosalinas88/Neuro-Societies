from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from model import SocietyModel


@dataclass
class PolicyConfig:
    name: str
    policymode: str
    schedule: Dict[int, str] | None = None


def evaluate_policies(
    policies: List[PolicyConfig],
    seeds: List[int],
    populationscale: str,
    steps: int,
    baseparams: Dict[str, object],
) -> pd.DataFrame:
    """Ejecuta varias políticas sobre distintas semillas y devuelve resultados agregados."""
    rows = []
    for policy in policies:
        for seed in seeds:
            params = dict(baseparams)
            params.update(
                seed=seed,
                populationscale=populationscale,
                policymode=policy.policymode,
                policyschedule=policy.schedule or {},
            )
            model = SocietyModel(**params)
            for _ in range(steps):
                model.step()
                if not model.running:
                    break
            last = model.lastmetrics or {}
            rows.append(
                dict(
                    policy=policy.name,
                    seed=seed,
                    populationscale=populationscale,
                    giniwealth=model.giniwealth,
                    violencerate=last.get("violencerate", 0.0),
                    libertyindex=model.libertyindex,
                    regime=model.regime,
                    formalism=model.legalformalism,
                    normcount=model.normcount,
                    powerconcentration=model.powerconcentration,
                )
            )
    return pd.DataFrame(rows)
