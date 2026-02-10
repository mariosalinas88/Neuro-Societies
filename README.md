# Neuro Societies – Agent-Based Model of Neurodiversity and Emergent Institutions

This repository contains an agent‑based model (ABM) that simulates how neurocognitive diversity, social interaction patterns, and institutional feedback co‑evolve into emergent social regimes (e.g., democratic, oligarchic, authoritarian, collapsed). The core model is implemented with Mesa (Python), with optional interactive visualization via Solara.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mariosalinas88/Neuro-Societies.git
cd Neuro-Societies

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python verify.py

# 4. Run tests
python tests/test_basic.py

# 5. Run a quick simulation
python run.py --steps 50
```

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
  - Policy modes and schedules (for "wind tunnel" evaluation of interventions).
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
├── verify.py                   # Quick verification script
├── governance_windtunnel.py    # Policy "wind tunnel" evaluation helpers
├── profiles.json               # Cognitive / neurodiversity profiles
├── requirements.txt
├── tests/
│   ├── test_basic.py           # Basic test suite
│   └── README.md               # Testing documentation
├── .github/workflows/
│   └── tests.yml              # CI/CD pipeline
├── README.md
└── results/                    # Created automatically by run.py (CSV/JSON outputs)
```

## Testing

### Quick Verification
```bash
python verify.py
```

This will check:
- Python version (3.8+)
- Required files
- Dependencies
- Python syntax
- Quick smoke test (5 steps)

### Full Test Suite
```bash
# Run all basic tests
python tests/test_basic.py

# Or with pytest
pytest tests/ -v
```

Tests include:
1. Model initialization
2. Single step execution
3. Multiple steps
4. Agent traits validation
5. Metrics collection
6. Reproducibility
7. Population stability

### Simulation Levels

**Level 1: Quick Test (5 seconds)**
```bash
python run.py --steps 5
```

**Level 2: Basic Simulation (30 seconds)**
```bash
python run.py --steps 50
```

**Level 3: Standard Simulation (2-5 minutes)**
```bash
python run.py --steps 200
```

**Level 4: Full Features (5-10 minutes)**
```bash
python run.py --steps 200 \
  --enablereproduction \
  --coalitionenabled \
  --enableguilt \
  --enableculturaltransmission
```

## Usage Examples

### Basic Simulation
```bash
python run.py --steps 200
```

### With Reproduction and Coalitions
```bash
python run.py --steps 500 \
  --enablereproduction \
  --coalitionenabled
```

### Custom Topology
```bash
python run.py --steps 200 \
  --interactiontopology small_world \
  --topologyk 6 \
  --topologyp 0.1
```

### Interactive Visualization
```bash
python server.py
# Open http://localhost:8765 in browser
```

## Command Line Arguments

### Population & Scale
- `--populationscale`: Population size (`tiny`, `tribe`, `city`, `nation`)
- `--seed`: Random seed for reproducibility

### Agent Traits
- `--spectrumlevel {1,2,3}`: Neurodiversity spectrum level
- `--initialmoralbias {highdark,lowdark,highprosocial}`: Initial moral bias
- `--resiliencebias {high,low}`: Resilience trait bias
- `--emotionalbias {high,low}`: Emotional regulation bias

### Social Dynamics
- `--enablereproduction`: Enable reproduction mechanics
- `--enablesexualselection`: Enable sexual selection
- `--coalitionenabled`: Enable coalition formation
- `--maleviolencemultiplier FLOAT`: Male violence modifier (default: 1.2)

### Network Topology
- `--interactiontopology {gridlocal,small_world,scale_free,erdos_renyi,regular}`: Social network type
- `--topologyk INT`: Network parameter k (degree/neighbors)
- `--topologyp FLOAT`: Network parameter p (rewiring probability)
- `--topologym INT`: Network parameter m (edges to attach)

### Cultural Mechanisms
- `--enableculturaltransmission`: Enable cultural learning
- `--culturallearningrate FLOAT`: Learning rate (0-1)
- `--imitationbias {prestige,success,conformity}`: Imitation strategy
- `--conformitybias FLOAT`: Conformity weight
- `--innovationrate FLOAT`: Innovation probability

### Psychological Mechanisms
- `--enableguilt`: Enable guilt mechanism
- `--enableostracism`: Enable ostracism
- `--enablefermiupdate`: Enable Fermi strategy update
- `--fermibeta FLOAT`: Fermi temperature parameter

### Policy Interventions
- `--policymode {none,enforcement,reward,mixed}`: Policy intervention type

## Output Files

Simulations generate files in the `results/` directory:

- `simulation_YYYYMMDD_HHMMSS.csv`: Time-series metrics
- `profiles_YYYYMMDD_HHMMSS.csv`: Per-profile statistics
- `trajectory_YYYYMMDD_HHMMSS.json`: Institutional evolution data

### Key Metrics

- **population**: Scaled population count
- **coop_rate**: Cooperation rate (0-1)
- **violence_rate**: Violence rate (0-1)
- **gini_wealth**: Wealth inequality (Gini coefficient)
- **legal_formalism**: Legal system formalism index
- **liberty_index**: Individual liberty measure
- **regime**: Emergent regime type
- **life_expectancy**: Average agent lifespan

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass: `python tests/test_basic.py`
5. Submit a pull request

## CI/CD

GitHub Actions automatically runs tests on:
- Every push to main
- Every pull request
- Python versions: 3.9, 3.10, 3.11

See [.github/workflows/tests.yml](.github/workflows/tests.yml) for details.

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@software{neuro_societies,
  author = {Salinas, Mario},
  title = {Neuro-Societies: Agent-Based Model of Neurodiversity and Emergent Institutions},
  year = {2026},
  url = {https://github.com/mariosalinas88/Neuro-Societies}
}
```

## Support

For issues, questions, or contributions:
- Open an issue: [GitHub Issues](https://github.com/mariosalinas88/Neuro-Societies/issues)
- Check documentation: [tests/README.md](tests/README.md)
