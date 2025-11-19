#!/usr/bin/env python3
"""
Script 2: Golden Crosscheck (Refactored)
Joins annotations with goldens to identify errors.

Input:
  - tagged_annotations.jsonl
  - tagged_goldens.jsonl

Output:
  - crosschecked_annotations.jsonl
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, List

from edge_case_annotator.config import get_config
from edge_case_annotator.models import Annotation
from edge_case_annotator.utils import setup_logging, load_jsonl_raw, save_jsonl_raw, get_file_path, validate_input_files


# Setup logger
config = get_config()
logger = setup_logging(__name__, config)


def build_golden_index(goldens: List[dict]) -> Dict[Tuple[str, str], dict]:
    """
    Build index of golden annotations keyed by (query, candidate).
    
    Args:
        goldens: List of golden annotation records
        
    Returns:
        Dict mapping (query, candidate) -> golden record
    """
    logger.info("Building golden index...")
    
    index: Dict[Tuple[str, str], dict] = {}
    
    for golden in goldens:
        query = golden.get('query', '').strip()
        candidate = golden.get('candidate', '').strip()
        key = (query, candidate)
        
        # If multiple goldens exist for same (query, candidate), keep first
        if key not in index:
            index[key] = golden
        else:
            logger.warning(f"Duplicate golden found for query='{query}', candidate='{candidate}'")
    
    logger.info(f"Indexed {len(index)} unique (query, candidate) pairs")
    return index


def crosscheck_annotations(
    annotations: List[dict],
    golden_index: Dict[Tuple[str, str], dict]
) -> List[dict]:
    """
    Join annotations with goldens and add crosscheck fields.
    
    Adds fields:
      - is_golden: bool (true if this annotation has a golden)
      - golden_label: int or None (the correct label from golden)
      - is_correct: bool or None (true if annotation matches golden)
      - is_error: bool (true only when is_golden==true AND is_correct==false)
    
    Args:
        annotations: List of annotation records
        golden_index: Index of golden annotations
        
    Returns:
        List of crosschecked annotations
    """
    logger.info(f"Crosschecking {len(annotations)} annotations...")
    
    crosschecked: List[dict] = []
    golden_count = 0
    correct_count = 0
    error_count = 0
    
    for annotation in annotations:
        query = annotation.get('query', '').strip()
        candidate = annotation.get('candidate', '').strip()
        key = (query, candidate)
        
        # Create crosschecked record (copy original)
        record = annotation.copy()
        
        # Check if golden exists
        if key in golden_index:
            golden = golden_index[key]
            golden_count += 1
            
            record['is_golden'] = True
            record['golden_label'] = golden.get('label')
            
            # Compare labels
            annotation_label = annotation.get('label')
            golden_label = golden.get('label')
            
            if annotation_label is not None and golden_label is not None:
                is_correct = (annotation_label == golden_label)
                record['is_correct'] = is_correct
                record['is_error'] = not is_correct
                
                if is_correct:
                    correct_count += 1
                else:
                    error_count += 1
                    logger.debug(f"Error found: query='{query[:30]}...', "
                               f"human={annotation_label}, golden={golden_label}")
            else:
                # If label missing, cannot determine correctness
                record['is_correct'] = None
                record['is_error'] = False
                logger.warning(f"Missing label for query='{query}'")
        else:
            # No golden for this annotation
            record['is_golden'] = False
            record['golden_label'] = None
            record['is_correct'] = None
            record['is_error'] = False
        
        crosschecked.append(record)
    
    # Log statistics
    human_accuracy = correct_count / golden_count if golden_count > 0 else 0.0
    
    logger.info(f"Crosscheck complete:")
    logger.info(f"  Total annotations: {len(crosschecked)}")
    logger.info(f"  Has golden: {golden_count} ({golden_count/len(crosschecked)*100:.1f}%)")
    logger.info(f"  Correct: {correct_count} ({correct_count/golden_count*100:.1f}% of goldens)")
    logger.info(f"  Errors: {error_count} ({error_count/golden_count*100:.1f}% of goldens)")
    logger.info(f"  Human accuracy: {human_accuracy:.2%}")
    
    return crosschecked


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Golden Crosscheck")
    logger.info("=" * 60)
    
    # Get file paths from config
    annotations_path = get_file_path('tagged_annotations', config, 'intermediate')
    goldens_path = get_file_path('tagged_goldens', config, 'intermediate')
    output_path = get_file_path('crosschecked_annotations', config, 'intermediate')
    
    # Validate input files
    if not validate_input_files([str(annotations_path), str(goldens_path)], logger):
        logger.error("Missing required input files. Please run prepare_data.py first.")
        return
    
    try:
        # Load data
        logger.info("Loading tagged data...")
        annotations = load_jsonl_raw(str(annotations_path))
        goldens = load_jsonl_raw(str(goldens_path))
        
        logger.info(f"✓ Loaded {len(annotations)} annotations")
        logger.info(f"✓ Loaded {len(goldens)} goldens")
        
        # Build golden index
        golden_index = build_golden_index(goldens)
        
        # Crosscheck annotations
        crosschecked = crosscheck_annotations(annotations, golden_index)
        
        # Write output
        save_jsonl_raw(crosschecked, str(output_path))
        logger.info(f"✓ Wrote {len(crosschecked)} crosschecked annotations → {output_path.name}")
        
        logger.info("=" * 60)
        logger.info("✓ Golden crosscheck complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error during crosscheck: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

