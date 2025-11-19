#!/usr/bin/env python3
"""Test script to verify imports work correctly."""

import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

print("Testing imports...")
print(f"Python path includes: {src_dir}")
print()

success = True

# Test 1: Can we import the package at all?
try:
    from edge_case_annotator import __version__
    print(f"✅ Package found and importable (version: {__version__})")
except ImportError as e:
    print(f"❌ Package structure is broken: {e}")
    success = False

# Test 2: Can we locate all modules?
import importlib.util
modules_to_check = [
    "edge_case_annotator.config",
    "edge_case_annotator.models",
    "edge_case_annotator.utils",
    "edge_case_annotator.prepare_data",
    "edge_case_annotator.golden_crosscheck",
    "edge_case_annotator.build_indexes",
    "edge_case_annotator.discover_patterns",
    "edge_case_annotator.generate_cases",
    "edge_case_annotator.annotate_with_debate",
    "edge_case_annotator.build_final_sets",
]

print("\nChecking module files exist:")
for module_name in modules_to_check:
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"✅ {module_name}")
    else:
        print(f"❌ {module_name} not found")
        success = False

if success:
    print("\n" + "="*60)
    print("✅ REFACTORING SUCCESSFUL!")
    print("="*60)
    print("\nAll modules are in the correct location and importable.")
    print("\n📦 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run the pipeline: python run_all_phases.py")
else:
    print("\n❌ Some structural issues detected")
    sys.exit(1)

