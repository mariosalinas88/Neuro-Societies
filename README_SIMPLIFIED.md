# Rama `psych-neuro-simple-base`

Esta rama conserva los archivos originales del repositorio, pero añade una vía paralela y simplificada para trabajar solo con perfiles psicológicos/neurocognitivos.

## Objetivo

Observar cómo perfiles con rasgos peculiares reaccionan frente a acontecimientos e interacciones repetidas, sin mezclar el análisis con variables biológicas de ciclo vital.

## Qué se eliminó en el modelo simplificado

El nuevo archivo `model_simple.py` no usa:

- edad;
- sexo/género;
- reproducción;
- gestación;
- fertilidad;
- selección sexual;
- parejas;
- descendencia;
- mortalidad por ciclo vital.

Los agentes quedan definidos por:

- rasgos psicológicos/neurocognitivos;
- memoria interpersonal;
- confianza;
- miedo;
- reputación cooperativa;
- reputación de amenaza;
- riqueza/status;
- conducta en interacciones.

## Archivos añadidos

- `model_simple.py`: modelo simplificado sin variables biológicas.
- `run_simple.py`: ejecución individual del modelo simplificado.
- `run_batch_simple.py`: escenarios comparativos por lotes.
- `tests/test_simple.py`: pruebas mínimas del modelo simplificado.

## Ejecución rápida

```bash
python run_simple.py --steps 100 --population_scale small
```

Ejemplo con mezcla de perfiles:

```bash
python run_simple.py --steps 150 --profile1 1 --weight1 0.4 --profile2 2 --weight2 0.3 --profile3 3 --weight3 0.3
```

Ejecución por lotes:

```bash
python run_batch_simple.py
```

Pruebas:

```bash
python tests/test_simple.py
```

## Salidas

El modelo simplificado guarda resultados en `results/`:

- `simple_summary_evolution.csv`
- `simple_per_profile_stats.csv`
- `simple_batch_runs.csv`

## Nota metodológica

Esta rama no pretende validar científicamente perfiles clínicos. Su utilidad es experimental y comparativa: revisar patrones de reacción, cooperación, evitación, agresión, apoyo, traición, reputación, desigualdad y régimen emergente bajo combinaciones de rasgos.
