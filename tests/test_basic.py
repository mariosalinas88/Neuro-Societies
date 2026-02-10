#!/usr/bin/env python3
"""Basic tests for Neuro-Societies model."""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from model import SocietyModel, Citizen
    import numpy as np
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


def test_model_initialization():
    """Test that model can be created successfully."""
    print("Test 1: Model Initialization...")
    try:
        model = SocietyModel(
            seed=42,
            population_scale="tiny",  # 10 agents
            enable_reproduction=False
        )
        agent_count = len(list(model.agents))
        assert agent_count > 0, "No agents created"
        print(f"  ✓ Model created with {agent_count} agents")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_single_step():
    """Test that model can execute one step."""
    print("Test 2: Single Step Execution...")
    try:
        model = SocietyModel(
            seed=42,
            population_scale="tiny",
            enable_reproduction=False
        )
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
        model = SocietyModel(
            seed=42,
            population_scale="tiny",
            enable_reproduction=False
        )
        for i in range(10):
            model.step()
        
        assert model.step_count == 10, "Step count incorrect"
        alive = len(model.agents_alive())
        print(f"  ✓ 10 steps completed, {alive} agents alive")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_agent_traits():
    """Test that agents have required neurocognitive traits."""
    print("Test 4: Agent Traits...")
    try:
        model = SocietyModel(seed=42, population_scale="tiny")
        agent = list(model.agents)[0]
        
        required_traits = [
            'empathy', 'dominance', 'reasoning',
            'aggression', 'moral_prosocial', 'impulsivity'
        ]
        
        for trait in required_traits:
            assert trait in agent.latent, f"Missing trait: {trait}"
            value = agent.latent[trait]
            assert 0 <= value <= 1, f"{trait} out of range: {value}"
        
        print(f"  ✓ All required traits present and valid")
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
        model = SocietyModel(seed=42, population_scale="tiny")
        for _ in range(5):
            model.step()
        
        df = model.datacollector.get_model_vars_dataframe()
        assert len(df) == 5, "Wrong number of data rows"
        
        required_metrics = ['population', 'coop_rate', 'violence_rate', 'gini_wealth']
        for metric in required_metrics:
            assert metric in df.columns, f"Missing metric: {metric}"
        
        # Verify metric ranges
        assert (df['coop_rate'] >= 0).all() and (df['coop_rate'] <= 1).all(), "coop_rate out of range"
        assert (df['violence_rate'] >= 0).all() and (df['violence_rate'] <= 1).all(), "violence_rate out of range"
        
        print(f"  ✓ Metrics collected correctly")
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
        results1 = []
        results2 = []
        
        for run in range(2):
            model = SocietyModel(seed=42, population_scale="tiny")
            for _ in range(5):
                model.step()
            df = model.datacollector.get_model_vars_dataframe()
            results = results1 if run == 0 else results2
            results.append(df['coop_rate'].iloc[-1])
        
        assert results1 == results2, f"Results differ: {results1} vs {results2}"
        print(f"  ✓ Same seed produces same results")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_population_stability():
    """Test that population doesn't collapse immediately."""
    print("Test 7: Population Stability...")
    try:
        model = SocietyModel(seed=42, population_scale="tiny")
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
        test_population_stability
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
