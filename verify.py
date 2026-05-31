#!/usr/bin/env python3
"""Quick verification script for Neuro-Societies setup."""
import os
import py_compile
import subprocess
import sys
import traceback


def check_python_version():
    """Verify Python version is 3.10+."""
    print("Checking Python version...")
    version = sys.version_info
    if (version.major, version.minor) >= (3, 10):
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    print(f"  ✗ Python {version.major}.{version.minor} (need 3.10+)")
    return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    required = ["mesa", "numpy", "pandas", "networkx"]
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False
    return True


def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")
    required = ["model.py", "run.py", "requirements.txt", "profiles.json"]
    missing = []
    for filename in required:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024
            print(f"  ✓ {filename} ({size:.1f} KB)")
        else:
            print(f"  ✗ {filename}")
            missing.append(filename)
    if missing:
        print(f"\n  Missing files: {', '.join(missing)}")
        return False
    return True


def check_syntax():
    """Check Python syntax of main files."""
    print("\nChecking Python syntax...")
    files = ["model.py", "run.py", "tests/test_basic.py"]
    for filename in files:
        try:
            py_compile.compile(filename, doraise=True)
            print(f"  ✓ {filename}")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {filename}: {e}")
            return False
    return True


def run_quick_test():
    """Run a quick smoke test."""
    print("\nRunning quick smoke test...")
    try:
        from model import SocietyModel
        print("  Creating model...")
        model = SocietyModel(
            seed=42,
            population_scale="tiny",
            enable_reproduction=False,
            enable_sexual_selection=False,
            coalition_enabled=False,
        )
        print(f"  ✓ Model created with {len(list(model.agents))} agents")
        print("  Running 5 steps...")
        for _ in range(5):
            model.step()
        active = len(model.agents_alive())
        df = model.datacollector.get_model_vars_dataframe()
        if active <= 0:
            raise RuntimeError("No active agents remain during smoke test")
        if df.empty:
            raise RuntimeError("DataCollector returned no rows")
        print(f"  ✓ Completed, {active} agents active, {len(df)} metric rows")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return False


def install_dependencies():
    """Prompt to install dependencies when running interactively."""
    if not sys.stdin.isatty():
        return False
    response = input("\nInstall missing dependencies? (y/n): ")
    if response.lower() == "y":
        print("Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Dependencies installed successfully")
            return True
        print(f"✗ Installation failed: {result.stderr}")
    return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("NEURO-SOCIETIES VERIFICATION")
    print("=" * 60)
    checks = [
        ("Python Version", check_python_version),
        ("Required Files", check_files),
        ("Syntax Check", check_syntax),
    ]
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n  ✗ Unexpected error in {name}: {e}")
            traceback.print_exc()
            results[name] = False
    dep_check = check_dependencies()
    if not dep_check and install_dependencies():
        dep_check = check_dependencies()
    results["Dependencies"] = dep_check
    if all(results.values()):
        results["Quick Test"] = run_quick_test()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    if all(results.values()):
        print("\n✓ ALL CHECKS PASSED - Ready to run simulations!")
        print("\nNext steps:")
        print("  1. Run basic tests: python tests/test_basic.py")
        print("  2. Run simulation: python run.py --steps 50 --populationscale tiny")
        print("  3. View help: python run.py --help")
        return 0
    print("\n✗ SOME CHECKS FAILED - Please fix issues above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
