#!/usr/bin/env python3
"""
Master Script: Run All Phases (Refactored)
Executes the complete Edge-Case Annotator Agent pipeline.

Usage:
  python run_all_phases.py [--skip-phase N] [--only-phase N] [--config PATH]
"""

import sys
import subprocess
import argparse
from pathlib import Path

from edge_case_annotator.config import load_config
from edge_case_annotator.utils import setup_logging


PHASES = [
    {
        "number": 1,
        "name": "Data Preparation",
        "scripts": ["src/edge_case_annotator/prepare_data.py", "src/edge_case_annotator/golden_crosscheck.py"],
        "description": "Parse guidelines and crosscheck annotations"
    },
    {
        "number": 2,
        "name": "Build Embedding Indexes",
        "scripts": ["src/edge_case_annotator/build_indexes.py"],
        "description": "Create guideline, example, and annotation indexes"
    },
    {
        "number": 3,
        "name": "Pattern Discovery",
        "scripts": ["src/edge_case_annotator/discover_patterns.py"],
        "description": "Discover error patterns and allocate budget"
    },
    {
        "number": 4,
        "name": "Generate Synthetic Cases",
        "scripts": ["src/edge_case_annotator/generate_cases.py"],
        "description": "Generate new edge cases using LLM #1"
    },
    {
        "number": 5,
        "name": "Annotate with Debate",
        "scripts": ["src/edge_case_annotator/annotate_with_debate.py"],
        "description": "Annotate using three-LLM adversarial system"
    },
    {
        "number": 6,
        "name": "Build Final Datasets",
        "scripts": ["src/edge_case_annotator/build_final_sets.py"],
        "description": "Create HUMAN_SET, SYNTHETIC_SET, and HYBRID_SET"
    }
]


def run_script(script_path: Path) -> bool:
    """
    Run a Python script and return success status.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Script failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Error running script: {e}")
        return False


def run_phase(phase: dict, base_path: Path) -> bool:
    """
    Run all scripts in a phase.
    """
    print("\n" + "=" * 70)
    print(f"PHASE {phase['number']}: {phase['name']}")
    print("=" * 70)
    print(f"Description: {phase['description']}")
    print()
    
    for script_name in phase['scripts']:
        script_path = base_path / script_name
        
        if not script_path.exists():
            print(f"\n✗ Script not found: {script_name}")
            return False
        
        print(f"\nRunning {script_name}...")
        print("-" * 70)
        
        success = run_script(script_path)
        
        if not success:
            print(f"\n✗ Phase {phase['number']} failed at {script_name}")
            return False
    
    print(f"\n✅ Phase {phase['number']} completed successfully!")
    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run all phases of the Edge-Case Annotator Agent pipeline"
    )
    parser.add_argument(
        '--skip-phase',
        type=int,
        help='Skip a specific phase number (1-6)'
    )
    parser.add_argument(
        '--only-phase',
        type=int,
        help='Run only a specific phase number (1-6)'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        help='Start from a specific phase number (1-6)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent
    
    # Load and validate configuration
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    try:
        config = load_config(args.config)
        print(f"✓ Configuration loaded from {args.config}")
        print(f"  LLM Model: {config.llm.generation_model}")
        print(f"  Embedding Model: {config.llm.embedding_model}")
        print(f"  Synthetic Budget: {config.pattern_discovery.total_synthetic_budget}")
        print(f"  Target Quality: {config.pattern_discovery.target_quality}")
        print(f"  Logging Level: {config.logging.level}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("  Please ensure config.yaml exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Configuration error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("EDGE-CASE ANNOTATOR AGENT - Complete Pipeline")
    print("=" * 70)
    print("\nThis will run all phases in sequence:")
    for phase in PHASES:
        status = ""
        if args.skip_phase == phase['number']:
            status = " [SKIPPED]"
        elif args.only_phase and args.only_phase != phase['number']:
            status = " [SKIPPED]"
        elif args.start_from and phase['number'] < args.start_from:
            status = " [SKIPPED]"
        
        print(f"  Phase {phase['number']}: {phase['name']}{status}")
    
    print("\n" + "=" * 70)
    
    # Determine which phases to run
    phases_to_run = []
    for phase in PHASES:
        if args.skip_phase == phase['number']:
            continue
        if args.only_phase and args.only_phase != phase['number']:
            continue
        if args.start_from and phase['number'] < args.start_from:
            continue
        phases_to_run.append(phase)
    
    if not phases_to_run:
        print("\n⚠️  No phases to run based on arguments provided.")
        return
    
    # Run phases
    start_phase = phases_to_run[0]['number']
    end_phase = phases_to_run[-1]['number']
    
    print(f"\nStarting execution (Phase {start_phase} → Phase {end_phase})...")
    
    for i, phase in enumerate(phases_to_run, 1):
        success = run_phase(phase, base_path)
        
        if not success:
            print("\n" + "=" * 70)
            print("❌ PIPELINE FAILED")
            print("=" * 70)
            print(f"\nFailed at Phase {phase['number']}: {phase['name']}")
            print("\nTo resume from this phase, run:")
            print(f"  python run_all_phases.py --start-from {phase['number']}")
            sys.exit(1)
    
    # Success summary
    print("\n" + "=" * 70)
    print("✅ ALL PHASES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📊 Final Datasets Generated:")
    print("  - human_set.jsonl       (Correct human annotations)")
    print("  - synthetic_set.jsonl   (All accepted synthetic cases)")
    print("  - hybrid_set.jsonl      (RECOMMENDED: Combined high-quality set)")
    print("  - dataset_metrics.json  (Quality metrics and statistics)")
    print("\n💡 Next Steps:")
    print("  1. Review dataset_metrics.json for quality statistics")
    print("  2. Use hybrid_set.jsonl for evaluation (best coverage)")
    print("  3. Optional: Add human QA results and re-run Phase 6")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

