#!/usr/bin/env python3
"""Basic tests for Neuro-Societies model."""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import numpy as np
    from model import Citizen, SocietyModel, gauss_sample, gauss_weights
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


def make_tiny_model(seed=42, **kwargs):
    """Create a small deterministic model for fast smoke tests."""
    defaults = dict(
        seed=seed,
        population_scale="tiny",
        enable_reproduction=False,
        enable_sexual_selection=False,
        coalition_enabled=False,
    )
    defaults.update(kwargs)
    return SocietyModel(**defaults)


def test_model_initialization():
    """Test that model can be created successfully."""
    print("Test 1: Model Initialization...")
    try:
        model = make_tiny_model()
        agent_count = len(list(model.agents))
        assert agent_count > 0, "No agents created"
        assert len(model.agents_alive()) == agent_count, "Initial agents should be alive"
        print(f"  ✓ Model created with {agent_count} agents")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_single_step():
    """Test that model can execute one step."""
    print("Test 2: Single Step Execution...")
    try:
        model = make_tiny_model()
        model.step()
        assert model.step_count == 1, "Step count not incremented"
        alive = len(model.agents_alive())
        assert alive > 0, "All agents died in first step"
        print(f"  ✓ Step executed, {alive} agents alive")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_multiple_steps():
    """Test that model can run multiple steps."""
    print("Test 3: Multiple Steps (10 steps)...")
    try:
        model = make_tiny_model()
        for _ in range(10):
            model.step()

        assert model.step_count == 10, "Step count incorrect"
        alive = len(model.agents_alive())
        assert alive > 0, "All agents died during 10-step run"
        print(f"  ✓ 10 steps completed, {alive} agents alive")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_agent_traits():
    """Test that agents have required neurocognitive traits."""
    print("Test 4: Agent Traits...")
    try:
        model = make_tiny_model()
        agent = list(model.agents)[0]

        required_traits = [
            'empathy', 'dominance', 'reasoning',
            'aggression', 'moral_prosocial', 'impulsivity',
            'moral_common_good', 'moral_honesty', 'dark_mach',
            'dark_narc', 'dark_psycho', 'affect_reg',
        ]

        for trait in required_traits:
            assert trait in agent.latent, f"Missing trait: {trait}"
            value = agent.latent[trait]
            assert 0 <= value <= 1, f"{trait} out of range: {value}"

        assert 0 <= agent.dark_core <= 1, f"dark_core out of range: {agent.dark_core}"
        assert 0 <= agent.reputation_coop <= 1, "cooperative reputation out of range"
        assert 0 <= agent.reputation_fear <= 1, "fear reputation out of range"
        print("  ✓ All required traits present and valid")
        print(f"    Sample: empathy={agent.latent['empathy']:.2f}, "
              f"dominance={agent.latent['dominance']:.2f}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_metrics_collection():
    """Test that model collects metrics correctly."""
    print("Test 5: Metrics Collection...")
    try:
        model = make_tiny_model()
        initial_rows = len(model.datacollector.get_model_vars_dataframe())
        for _ in range(5):
            model.step()

        df = model.datacollector.get_model_vars_dataframe()
        assert len(df) == initial_rows + 5, "Wrong number of data rows"

        required_metrics = ['population', 'coop_rate', 'violence_rate', 'gini_wealth']
        for metric in required_metrics:
            assert metric in df.columns, f"Missing metric: {metric}"

        # Verify metric ranges. Population can be scaled, so only rates are clamped.
        assert (df['coop_rate'] >= 0).all() and (df['coop_rate'] <= 1).all(), "coop_rate out of range"
        assert (df['violence_rate'] >= 0).all() and (df['violence_rate'] <= 1).all(), "violence_rate out of range"
        assert (df['gini_wealth'] >= 0).all(), "gini_wealth negative"

        print("  ✓ Metrics collected correctly")
        print(f"    Final: coop_rate={df['coop_rate'].iloc[-1]:.3f}, "
              f"violence_rate={df['violence_rate'].iloc[-1]:.3f}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_reproducibility():
    """Test that same seed produces same results."""
    print("Test 6: Reproducibility...")
    try:
        results = []

        for _ in range(2):
            model = make_tiny_model(seed=42)
            for _ in range(5):
                model.step()
            df = model.datacollector.get_model_vars_dataframe()
            results.append((
                float(df['coop_rate'].iloc[-1]),
                float(df['violence_rate'].iloc[-1]),
                float(df['gini_wealth'].iloc[-1]),
                len(model.agents_alive()),
            ))

        assert results[0] == results[1], f"Results differ: {results[0]} vs {results[1]}"
        print("  ✓ Same seed produces same results")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_population_stability():
    """Test that population doesn't collapse immediately."""
    print("Test 7: Population Stability...")
    try:
        model = make_tiny_model()
        initial_pop = len(model.agents_alive())

        for _ in range(20):
            model.step()

        final_pop = len(model.agents_alive())
        survival_rate = final_pop / initial_pop

        assert survival_rate > 0.3, f"Population collapsed: {survival_rate:.1%} survival"
        print(f"  ✓ Population stable: {initial_pop} → {final_pop} ({survival_rate:.1%} survival)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_gaussian_sampling_contract():
    """Protect the intended clipped Gaussian behavior."""
    print("Test 8: Gaussian Sampling Contract...")
    try:
        rng = np.random.default_rng(123)
        samples = [gauss_sample(rng, 0.5, relative_std=0.2, lo=0.0, hi=1.0) for _ in range(200)]
        assert all(0.0 <= s <= 1.0 for s in samples), "Gaussian samples out of [0, 1]"
        mean = float(np.mean(samples))
        assert 0.4 <= mean <= 0.6, f"Gaussian mean drifted too far: {mean}"

        weights = gauss_weights(rng, [0.5, 0.3, 0.2], relative_std=0.2, normalize=True)
        assert all(0.0 <= w <= 1.0 for w in weights), "Gaussian weights out of range"
        assert abs(sum(weights) - 1.0) < 1e-9, "Gaussian weights are not normalized"
        print(f"  ✓ Gaussian samples clipped and centered; sample mean={mean:.3f}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("NEURO-SOCIETIES TEST SUITE")
    print("="*60)
    print()

    tests = [
        test_model_initialization,
        test_single_step,
        test_multiple_steps,
        test_agent_traits,
        test_metrics_collection,
        test_reproducibility,
        test_population_stability,
        test_gaussian_sampling_contract,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append(False)
        print()

    print("="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)")

    if all(results):
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
