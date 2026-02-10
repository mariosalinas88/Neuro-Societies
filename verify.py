#!/usr/bin/env python3
"""Quick verification script for Neuro-Societies setup."""
import sys
import subprocess
import os


def check_python_version():
    """Verify Python version is 3.8+."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    required = ['mesa', 'numpy', 'pandas', 'networkx']
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
    required = ['model.py', 'run.py', 'requirements.txt', 'profiles.json']
    missing = []
    
    for file in required:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✓ {file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n  Missing files: {', '.join(missing)}")
        return False
    return True


def check_syntax():
    """Check Python syntax of main files."""
    print("\nChecking Python syntax...")
    import py_compile
    files = ['model.py', 'run.py']
    
    for file in files:
        try:
            py_compile.compile(file, doraise=True)
            print(f"  ✓ {file}")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {file}: {e}")
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
            enable_reproduction=False
        )
        print(f"  ✓ Model created with {len(list(model.agents))} agents")
        
        print("  Running 5 steps...")
        for i in range(5):
            model.step()
        alive = len(model.agents_alive())
        print(f"  ✓ Completed, {alive} agents alive")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def install_dependencies():
    """Prompt to install dependencies."""
    response = input("\nInstall missing dependencies? (y/n): ")
    if response.lower() == 'y':
        print("Installing dependencies...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ Dependencies installed successfully")
            return True
        else:
            print(f"✗ Installation failed: {result.stderr}")
            return False
    return False


def main():
    """Run all verification checks."""
    print("="*60)
    print("NEURO-SOCIETIES VERIFICATION")
    print("="*60)
    
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
            results[name] = False
    
    # Check dependencies separately with option to install
    dep_check = check_dependencies()
    if not dep_check:
        if install_dependencies():
            dep_check = check_dependencies()
    results["Dependencies"] = dep_check
    
    # Only run test if all other checks pass
    if all(results.values()):
        results["Quick Test"] = run_quick_test()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    if all(results.values()):
        print("\n✓ ALL CHECKS PASSED - Ready to run simulations!")
        print("\nNext steps:")
        print("  1. Run basic tests: python tests/test_basic.py")
        print("  2. Run simulation: python run.py --steps 50")
        print("  3. View help: python run.py --help")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - Please fix issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
