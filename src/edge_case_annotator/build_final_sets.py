#!/usr/bin/env python3
"""
Script 7: Build Final Datasets (Refactored)
Builds three final evaluation datasets with quality metrics.

Input:
  - crosschecked_annotations.jsonl
  - synthetic_annotations.jsonl
  - ambiguous_cases.jsonl
  - human_qa_results.jsonl (optional)

Output:
  - human_set.jsonl
  - synthetic_set.jsonl
  - hybrid_set.jsonl
  - dataset_metrics.json
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from edge_case_annotator.config import get_config
from edge_case_annotator.models import FinalAnnotation, DatasetMetrics, AllMetrics, HumanQAResult
from edge_case_annotator.utils import setup_logging, load_jsonl_raw, save_jsonl_raw, get_file_path, validate_input_files


# Setup logger and config
config = get_config()
logger = setup_logging(__name__, config)


def apply_human_qa_results(
    synthetic_annotations: List[dict],
    ambiguous_cases: List[dict],
    human_qa_results: List[dict]
) -> Tuple[List[dict], List[dict]]:
    """
    Apply human QA results to synthetic annotations.
    
    Human QA results format:
    {
      "case_id": "syn_00001",
      "human_review": "accept" | "reject",
      "human_label": 1 (optional, if different from final_label),
      "human_notes": "..." (optional)
    }
    
    Updates qa_status:
      - "judge_accepted" + human "accept" → "synthetic_vetted"
      - "needs_human_review" + human "accept" → "synthetic_vetted"
      - any + human "reject" → removed from dataset
    
    Args:
        synthetic_annotations: List of synthetic annotation records
        ambiguous_cases: List of ambiguous case records
        human_qa_results: List of human QA review records
        
    Returns:
        Tuple of (updated synthetic annotations, updated ambiguous cases)
    """
    if not human_qa_results:
        logger.info("No human QA results found, using judge decisions only")
        return synthetic_annotations, ambiguous_cases
    
    logger.info(f"Applying {len(human_qa_results)} human QA results...")
    
    # Build lookup by case_id
    qa_lookup = {r['case_id']: r for r in human_qa_results}
    
    # Process synthetic annotations
    updated_synthetic = []
    accepted_count = 0
    rejected_count = 0
    
    for ann in synthetic_annotations:
        case_id = ann.get('case_id')
        
        if case_id in qa_lookup:
            qa = qa_lookup[case_id]
            review = qa.get('human_review', 'accept')
            
            if review == 'accept':
                # Update to vetted
                ann['qa_status'] = 'synthetic_vetted'
                ann['human_reviewed'] = True
                
                # Update label if human provided different one
                if 'human_label' in qa:
                    ann['final_label'] = qa['human_label']
                    logger.debug(f"Updated label for {case_id} based on human review")
                
                updated_synthetic.append(ann)
                accepted_count += 1
            else:
                # Reject, don't add to updated list
                rejected_count += 1
                logger.debug(f"Rejected {case_id} based on human review")
        else:
            # No human review, keep as is
            updated_synthetic.append(ann)
    
    # Process ambiguous cases
    updated_ambiguous = []
    ambig_accepted = 0
    ambig_rejected = 0
    
    for amb in ambiguous_cases:
        case_id = amb.get('case_id')
        
        if case_id in qa_lookup:
            qa = qa_lookup[case_id]
            review = qa.get('human_review', 'reject')
            
            if review == 'accept':
                # Move to synthetic annotations with vetted status
                amb['qa_status'] = 'synthetic_vetted'
                amb['human_reviewed'] = True
                
                # Use human label if provided
                if 'human_label' in qa:
                    amb['final_label'] = qa['human_label']
                
                updated_synthetic.append(amb)
                ambig_accepted += 1
            else:
                # Reject, don't add anywhere
                ambig_rejected += 1
        else:
            # No human review, keep in ambiguous
            updated_ambiguous.append(amb)
    
    logger.info(f"QA results applied:")
    logger.info(f"  Synthetic: {accepted_count} accepted, {rejected_count} rejected")
    logger.info(f"  Ambiguous: {ambig_accepted} accepted, {ambig_rejected} rejected")
    
    return updated_synthetic, updated_ambiguous


def build_human_set(crosschecked: List[dict]) -> List[dict]:
    """
    Build HUMAN_SET from correct human annotations.
    
    Filter: is_golden==true AND is_correct==true
    
    Args:
        crosschecked: List of crosschecked annotation records
        
    Returns:
        List of clean human set records
    """
    logger.info("Building HUMAN_SET...")
    
    human_set = []
    
    for record in crosschecked:
        if record.get('is_golden') and record.get('is_correct'):
            # Create clean record
            clean_record = {
                'query': record['query'],
                'candidate': record['candidate'],
                'label': record['label'],
                'tags': record.get('tags', {}),
                'metadata': record.get('metadata', {}),
                'source': 'human',
                'is_golden': True
            }
            human_set.append(clean_record)
    
    logger.info(f"✓ HUMAN_SET: {len(human_set)} cases")
    return human_set


def build_synthetic_set(synthetic_annotations: List[dict]) -> List[dict]:
    """
    Build SYN_SET from synthetic annotations.
    
    Filter: qa_status in ["judge_accepted", "synthetic_vetted"]
    
    Args:
        synthetic_annotations: List of synthetic annotation records
        
    Returns:
        List of clean synthetic set records
    """
    logger.info("Building SYNTHETIC_SET...")
    
    synthetic_set = []
    
    for record in synthetic_annotations:
        qa_status = record.get('qa_status')
        
        if qa_status in ['judge_accepted', 'synthetic_vetted']:
            # Create clean record
            clean_record = {
                'case_id': record.get('case_id'),
                'query': record['query'],
                'candidate': record['candidate'],
                'label': record['final_label'],
                'tags': record.get('metadata', {}),
                'metadata': record.get('metadata', {}),
                'pattern_id': record.get('pattern_id'),
                'source': 'synthetic',
                'qa_status': qa_status,
                'final_rationale': record.get('final_rationale', '')
            }
            synthetic_set.append(clean_record)
    
    logger.info(f"✓ SYNTHETIC_SET: {len(synthetic_set)} cases")
    return synthetic_set


def build_hybrid_set(
    human_set: List[dict],
    synthetic_annotations: List[dict]
) -> List[dict]:
    """
    Build HYBRID_SET from vetted human + synthetic annotations.
    
    Components:
      - All of HUMAN_SET (already correct)
      - Synthetic cases with qa_status="synthetic_vetted" only
    
    Args:
        human_set: Human set records
        synthetic_annotations: Synthetic annotation records
        
    Returns:
        List of clean hybrid set records
    """
    logger.info("Building HYBRID_SET...")
    
    hybrid_set = []
    
    # Add all human set
    hybrid_set.extend(human_set)
    
    # Add vetted synthetics only
    vetted_count = 0
    for record in synthetic_annotations:
        if record.get('qa_status') == 'synthetic_vetted':
            clean_record = {
                'case_id': record.get('case_id'),
                'query': record['query'],
                'candidate': record['candidate'],
                'label': record['final_label'],
                'tags': record.get('metadata', {}),
                'metadata': record.get('metadata', {}),
                'pattern_id': record.get('pattern_id'),
                'source': 'synthetic',
                'qa_status': 'synthetic_vetted',
                'final_rationale': record.get('final_rationale', '')
            }
            hybrid_set.append(clean_record)
            vetted_count += 1
    
    logger.info(f"✓ HYBRID_SET: {len(hybrid_set)} cases ({len(human_set)} human + {vetted_count} synthetic)")
    return hybrid_set


def compute_metrics(
    human_set: List[dict],
    synthetic_set: List[dict],
    hybrid_set: List[dict],
    crosschecked: List[dict]
) -> dict:
    """
    Compute quality metrics for each dataset.
    
    Args:
        human_set: Human set records
        synthetic_set: Synthetic set records
        hybrid_set: Hybrid set records
        crosschecked: Original crosschecked annotations
        
    Returns:
        Complete metrics dictionary
    """
    logger.info("Computing metrics...")
    
    # Helper functions
    def count_by_tags(dataset):
        tag_counts = defaultdict(int)
        for record in dataset:
            tags = record.get('tags', {})
            intent = tags.get('intent', 'unknown')
            entity_type = tags.get('entity_type', 'unknown')
            locale = tags.get('locale', 'unknown')
            
            tag_counts[f"intent:{intent}"] += 1
            tag_counts[f"entity_type:{entity_type}"] += 1
            tag_counts[f"locale:{locale}"] += 1
        
        return dict(tag_counts)
    
    def label_distribution(dataset):
        dist = defaultdict(int)
        for record in dataset:
            label = record.get('label', 'unknown')
            dist[str(label)] += 1
        return dict(dist)
    
    # Calculate human accuracy from crosschecked
    golden_count = sum(1 for r in crosschecked if r.get('is_golden'))
    correct_count = sum(1 for r in crosschecked if r.get('is_correct'))
    human_accuracy = correct_count / golden_count if golden_count > 0 else 0.0
    
    # Count vetted vs judge_accepted in synthetic set
    vetted_count = sum(1 for r in synthetic_set if r.get('qa_status') == 'synthetic_vetted')
    judge_accepted_count = sum(1 for r in synthetic_set if r.get('qa_status') == 'judge_accepted')
    
    metrics = {
        "human_set": {
            "total_cases": len(human_set),
            "quality": 1.0,
            "label_distribution": label_distribution(human_set),
            "tag_distribution": count_by_tags(human_set)
        },
        "synthetic_set": {
            "total_cases": len(synthetic_set),
            "judge_accepted": judge_accepted_count,
            "human_vetted": vetted_count,
            "label_distribution": label_distribution(synthetic_set),
            "tag_distribution": count_by_tags(synthetic_set)
        },
        "hybrid_set": {
            "total_cases": len(hybrid_set),
            "human_cases": len(human_set),
            "synthetic_cases": len(hybrid_set) - len(human_set),
            "estimated_quality": "high (vetted only)",
            "label_distribution": label_distribution(hybrid_set),
            "tag_distribution": count_by_tags(hybrid_set)
        },
        "baseline": {
            "original_human_accuracy": round(human_accuracy, 4),
            "total_golden_annotations": golden_count,
            "correct_human_annotations": correct_count
        }
    }
    
    return metrics


def print_summary(metrics: dict) -> None:
    """Print summary of dataset metrics."""
    logger.info("=" * 60)
    logger.info("DATASET METRICS SUMMARY")
    logger.info("=" * 60)
    
    logger.info("\n📊 BASELINE (Original Human Annotations)")
    logger.info(f"  Total golden annotations: {metrics['baseline']['total_golden_annotations']}")
    logger.info(f"  Correct annotations: {metrics['baseline']['correct_human_annotations']}")
    logger.info(f"  Human accuracy: {metrics['baseline']['original_human_accuracy']:.2%}")
    
    logger.info("\n📁 HUMAN_SET")
    logger.info(f"  Total cases: {metrics['human_set']['total_cases']}")
    logger.info(f"  Quality: {metrics['human_set']['quality']:.2%}")
    logger.info(f"  Label distribution: {metrics['human_set']['label_distribution']}")
    
    logger.info("\n🤖 SYNTHETIC_SET")
    logger.info(f"  Total cases: {metrics['synthetic_set']['total_cases']}")
    logger.info(f"  Judge accepted: {metrics['synthetic_set']['judge_accepted']}")
    logger.info(f"  Human vetted: {metrics['synthetic_set']['human_vetted']}")
    logger.info(f"  Label distribution: {metrics['synthetic_set']['label_distribution']}")
    
    logger.info("\n🎯 HYBRID_SET (Recommended)")
    logger.info(f"  Total cases: {metrics['hybrid_set']['total_cases']}")
    logger.info(f"  Human cases: {metrics['hybrid_set']['human_cases']}")
    logger.info(f"  Synthetic cases: {metrics['hybrid_set']['synthetic_cases']}")
    logger.info(f"  Quality: {metrics['hybrid_set']['estimated_quality']}")
    logger.info(f"  Label distribution: {metrics['hybrid_set']['label_distribution']}")
    
    # Calculate improvement
    if metrics['baseline']['total_golden_annotations'] > 0:
        coverage_increase = (
            (metrics['hybrid_set']['total_cases'] - metrics['human_set']['total_cases']) /
            metrics['baseline']['total_golden_annotations'] * 100
        )
        logger.info("\n📈 IMPROVEMENT")
        logger.info(f"  Coverage increase: +{coverage_increase:.1f}% (synthetic cases added)")
        logger.info(f"  Edge-case coverage: Enhanced with {metrics['hybrid_set']['synthetic_cases']} targeted cases")


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("PHASE 6: Build Final Datasets")
    logger.info("=" * 60)
    
    # Get file paths from config
    crosschecked_path = get_file_path('crosschecked_annotations', config, 'intermediate')
    synthetic_path = get_file_path('synthetic_annotations', config, 'intermediate')
    ambiguous_path = get_file_path('ambiguous_cases', config, 'intermediate')
    human_qa_path = get_file_path('human_qa_results', config, 'input')
    
    human_set_path = get_file_path('human_set', config, 'output')
    synthetic_set_path = get_file_path('synthetic_set', config, 'output')
    hybrid_set_path = get_file_path('hybrid_set', config, 'output')
    metrics_path = get_file_path('dataset_metrics', config, 'output')
    
    # Validate required input files
    required = [str(crosschecked_path), str(synthetic_path), str(ambiguous_path)]
    if not validate_input_files(required, logger):
        logger.error("Missing required input files. Please run previous phase scripts first.")
        return
    
    try:
        # Load data
        logger.info("Loading data...")
        crosschecked = load_jsonl_raw(str(crosschecked_path))
        synthetic_annotations = load_jsonl_raw(str(synthetic_path))
        ambiguous_cases = load_jsonl_raw(str(ambiguous_path))
        
        # Try to load human QA results (optional)
        if human_qa_path.exists():
            human_qa_results = load_jsonl_raw(str(human_qa_path))
        else:
            human_qa_results = []
        
        logger.info(f"✓ Loaded {len(crosschecked)} crosschecked annotations")
        logger.info(f"✓ Loaded {len(synthetic_annotations)} synthetic annotations")
        logger.info(f"✓ Loaded {len(ambiguous_cases)} ambiguous cases")
        logger.info(f"✓ Loaded {len(human_qa_results)} human QA results")
        
        # Apply human QA results if available
        synthetic_annotations, ambiguous_cases = apply_human_qa_results(
            synthetic_annotations,
            ambiguous_cases,
            human_qa_results
        )
        
        # Build datasets
        human_set = build_human_set(crosschecked)
        synthetic_set = build_synthetic_set(synthetic_annotations)
        hybrid_set = build_hybrid_set(human_set, synthetic_annotations)
        
        # Compute metrics
        metrics = compute_metrics(human_set, synthetic_set, hybrid_set, crosschecked)
        
        # Write outputs
        logger.info("\nWriting outputs...")
        
        save_jsonl_raw(human_set, str(human_set_path))
        logger.info(f"✓ Wrote {human_set_path.name}")
        
        save_jsonl_raw(synthetic_set, str(synthetic_set_path))
        logger.info(f"✓ Wrote {synthetic_set_path.name}")
        
        save_jsonl_raw(hybrid_set, str(hybrid_set_path))
        logger.info(f"✓ Wrote {hybrid_set_path.name}")
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Wrote {metrics_path.name}")
        
        # Print summary
        print_summary(metrics)
        
        logger.info("=" * 60)
        logger.info("✓ All datasets built successfully!")
        logger.info("=" * 60)
        logger.info("\n💡 RECOMMENDATION: Use HYBRID_SET for evaluation")
        logger.info("   It combines high-quality human annotations with")
        logger.info("   vetted synthetic edge cases for better coverage.")
        
        # Suggest human QA if not done
        if not human_qa_results and synthetic_annotations:
            vetted = sum(1 for r in synthetic_annotations if r.get('qa_status') == 'synthetic_vetted')
            if vetted == 0:
                logger.info("\n⚠️  NOTE: No human QA results detected.")
                logger.info("   Consider reviewing a sample of synthetic annotations")
                logger.info("   and creating human_qa_results.jsonl for higher confidence.")
                sample_size = int(len(synthetic_annotations) * config.qa.judge_accepted_sample_rate)
                logger.info(f"\n   Suggested sample: ~{sample_size} cases")
        
    except Exception as e:
        logger.error(f"Fatal error during dataset building: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

