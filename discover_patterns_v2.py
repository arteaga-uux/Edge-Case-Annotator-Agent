#!/usr/bin/env python3
"""
Script 4: Pattern Discovery (Refactored)
Discovers edge-case patterns from annotation errors using two-level clustering.

Input:
  - crosschecked_annotations.jsonl
  - all_annotations_index.jsonl

Output:
  - edge_case_profiles.jsonl
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from config import get_config
from models import EdgeCaseProfile, PatternMetrics, SeedError, SeedGood, Tags
from utils import setup_logging, load_jsonl_raw, save_jsonl_raw, get_file_path, validate_input_files


# Setup logger and config
config = get_config()
logger = setup_logging(__name__, config)


def get_tag_key(tags: Dict[str, str]) -> Tuple[str, ...]:
    """
    Create hashable key from tags for grouping.
    
    Args:
        tags: Tags dictionary
        
    Returns:
        Tuple of tag values
    """
    return (
        tags.get('intent', 'unknown'),
        tags.get('entity_type', 'unknown'),
        tags.get('locale', 'unknown'),
        tags.get('domain', 'unknown')
    )


def cluster_by_embeddings(
    records: List[dict],
    embeddings: np.ndarray,
    eps: float,
    min_samples: int
) -> List[List[int]]:
    """
    Cluster records by embedding similarity using DBSCAN.
    
    Args:
        records: List of annotation records
        embeddings: numpy array of embeddings (n_samples, embedding_dim)
        eps: Maximum distance for clustering (lower = tighter clusters)
        min_samples: Minimum samples per cluster
    
    Returns:
        List of clusters, where each cluster is a list of record indices
    """
    if len(records) < min_samples:
        logger.debug(f"Too few records ({len(records)}) for clustering, returning single cluster")
        return [list(range(len(records)))]
    
    # Compute cosine similarity matrix
    similarity = cosine_similarity(embeddings)
    
    # Convert to distance (1 - similarity)
    distance = 1 - similarity
    
    # Cluster using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance)
    
    # Group by cluster label
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:  # -1 is noise in DBSCAN
            clusters[label].append(idx)
    
    logger.debug(f"DBSCAN found {len(clusters)} clusters from {len(records)} records")
    return list(clusters.values())


def calculate_pattern_metrics(
    error_records: List[dict],
    all_records: List[dict],
    tag_key: Tuple[str, ...]
) -> PatternMetrics:
    """
    Calculate metrics for a pattern.
    
    Args:
        error_records: List of error records in this pattern
        all_records: All annotation records
        tag_key: Tag key for this pattern
    
    Returns:
        PatternMetrics object
    """
    # Count total annotations with same tags
    total_with_tags = sum(
        1 for r in all_records
        if r.get('is_golden') and get_tag_key(r.get('tags', {})) == tag_key
    )
    
    # Count errors
    error_count = len(error_records)
    
    # Calculate metrics
    if total_with_tags > 0:
        error_rate = error_count / total_with_tags
        human_acc = 1 - error_rate
    else:
        error_rate = 0.0
        human_acc = 1.0
    
    quality_gap = max(0, config.pattern_discovery.target_quality - human_acc)
    
    return PatternMetrics(
        volume=error_count,
        total_with_tags=total_with_tags,
        error_rate=round(error_rate, 3),
        human_acc=round(human_acc, 3),
        quality_gap=round(quality_gap, 3)
    )


def allocate_budget(patterns: List[EdgeCaseProfile]) -> List[EdgeCaseProfile]:
    """
    Allocate synthetic budget across patterns based on quality gap and volume.
    
    Formula: score_i = quality_gap_i^power × log(base + volume_i)
             N_i = (total_budget × score_i) / Σ(score_j)
    
    Args:
        patterns: List of edge case profiles
        
    Returns:
        Updated list with synthetic_budget allocated
    """
    logger.info("Allocating synthetic budget...")
    
    total_budget = config.pattern_discovery.total_synthetic_budget
    gap_power = config.pattern_discovery.budget_allocation.quality_gap_power
    log_base = config.pattern_discovery.budget_allocation.volume_log_base
    
    # Calculate scores
    scores: List[float] = []
    for pattern in patterns:
        quality_gap = pattern.metrics.quality_gap
        volume = pattern.metrics.volume
        
        # Score formula
        score = (quality_gap ** gap_power) * np.log(log_base + volume)
        scores.append(score)
    
    # Total score
    total_score = sum(scores)
    
    if total_score == 0:
        # No quality gaps, distribute evenly
        budget_per_pattern = total_budget // len(patterns) if patterns else 0
        for pattern in patterns:
            pattern.synthetic_budget = budget_per_pattern
        logger.warning("Total score is 0, distributing budget evenly")
    else:
        # Allocate proportionally
        for pattern, score in zip(patterns, scores):
            budget = int((total_budget * score) / total_score)
            pattern.synthetic_budget = budget
            logger.debug(f"Pattern {pattern.pattern_id}: budget={budget}, score={score:.3f}")
    
    return patterns


def create_pattern_profile(
    pattern_id: str,
    tag_key: Tuple[str, ...],
    error_records: List[dict],
    all_records: List[dict]
) -> EdgeCaseProfile:
    """
    Create edge-case profile for a pattern.
    
    Args:
        pattern_id: Unique pattern identifier
        tag_key: Tag tuple for this pattern
        error_records: Error records in this pattern
        all_records: All annotation records
        
    Returns:
        EdgeCaseProfile object
    """
    # Build tags dict
    tags = Tags(
        intent=tag_key[0],
        entity_type=tag_key[1],
        locale=tag_key[2],
        domain=tag_key[3]
    )
    
    # Calculate metrics
    metrics = calculate_pattern_metrics(error_records, all_records, tag_key)
    
    # Sample seed errors (up to 5)
    seed_errors = [
        SeedError(
            query=r['query'],
            candidate=r['candidate'],
            label=r['label'],
            golden_label=r['golden_label']
        )
        for r in error_records[:5]
    ]
    
    # Find good examples with same tags (up to 5)
    good_examples_raw = [
        r for r in all_records
        if r.get('is_golden') and r.get('is_correct')
        and get_tag_key(r.get('tags', {})) == tag_key
    ][:5]
    
    seed_good = [
        SeedGood(
            query=r['query'],
            candidate=r['candidate'],
            label=r['label']
        )
        for r in good_examples_raw
    ]
    
    # Generate description
    description = generate_pattern_description(error_records, tags)
    
    return EdgeCaseProfile(
        pattern_id=pattern_id,
        description=description,
        tags=tags,
        metrics=metrics,
        seed_errors=seed_errors,
        seed_good=seed_good,
        synthetic_budget=0  # Will be set later by allocate_budget
    )


def generate_pattern_description(error_records: List[dict], tags: Tags) -> str:
    """
    Generate human-readable description of the pattern.
    
    Args:
        error_records: List of error records
        tags: Pattern tags
        
    Returns:
        Description string
    """
    entity_type = tags.entity_type
    intent = tags.intent
    locale = tags.locale
    
    # Analyze common themes in errors
    queries = [r['query'].lower() for r in error_records]
    
    # Detect common keywords
    keywords: List[str] = []
    if any('live' in q or 'en vivo' in q for q in queries):
        keywords.append('live/en vivo')
    if any('tour' in q or 'concert' in q for q in queries):
        keywords.append('concert/tour')
    if any('studio' in q for q in queries):
        keywords.append('studio')
    
    # Build description
    if keywords:
        theme = ', '.join(keywords)
        description = f"Pattern involving {theme} in {entity_type} queries. "
    else:
        description = f"Pattern in {entity_type} queries. "
    
    description += f"Human annotators show confusion in {locale} locale with {intent} intent."
    
    return description


def discover_patterns(
    errors: List[dict],
    all_annotations: List[dict],
    embeddings_index: List[dict]
) -> List[EdgeCaseProfile]:
    """
    Discover edge-case patterns using two-level clustering.
    
    Level 1: Group by exact tags
    Level 2: Cluster by embedding similarity within each tag group
    
    Args:
        errors: List of error records
        all_annotations: All annotation records
        embeddings_index: Embedding index
        
    Returns:
        List of EdgeCaseProfile objects
    """
    logger.info("Grouping errors by exact tags...")
    
    # Create embedding lookup
    embedding_lookup: Dict[str, List[float]] = {
        e['annotation_id']: e['embedding']
        for e in embeddings_index
    }
    
    # Group errors by tags
    tag_groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for error in errors:
        tag_key = get_tag_key(error.get('tags', {}))
        tag_groups[tag_key].append(error)
    
    logger.info(f"Found {len(tag_groups)} unique tag combinations")
    
    # Level 2: Cluster within each tag group
    logger.info("Clustering by embedding similarity...")
    
    patterns: List[EdgeCaseProfile] = []
    pattern_count = 0
    
    min_volume = config.pattern_discovery.min_pattern_volume
    eps = config.pattern_discovery.clustering.eps
    min_samples = config.pattern_discovery.clustering.min_samples
    
    for tag_key, tag_errors in tag_groups.items():
        if len(tag_errors) < min_volume:
            logger.debug(f"Skipping tag group with only {len(tag_errors)} errors (min: {min_volume})")
            continue
        
        # Get embeddings for these errors
        error_embeddings: List[List[float]] = []
        valid_errors: List[dict] = []
        
        for error in tag_errors:
            # Find corresponding embedding
            ann_id = None
            for e in embeddings_index:
                if (e['query'] == error['query'] and 
                    e['candidate'] == error['candidate'] and
                    e['label'] == error['label']):
                    ann_id = e['annotation_id']
                    break
            
            if ann_id and ann_id in embedding_lookup:
                error_embeddings.append(embedding_lookup[ann_id])
                valid_errors.append(error)
        
        if not error_embeddings:
            logger.warning(f"No embeddings found for tag group {tag_key}")
            continue
        
        # Convert to numpy array
        embeddings_array = np.array(error_embeddings)
        
        # Cluster by embedding similarity
        clusters = cluster_by_embeddings(valid_errors, embeddings_array, eps, min_samples)
        
        # Create pattern profile for each cluster
        for cluster_idx, cluster_indices in enumerate(clusters):
            if len(cluster_indices) < min_volume:
                logger.debug(f"Skipping small cluster with {len(cluster_indices)} errors")
                continue
            
            cluster_errors = [valid_errors[i] for i in cluster_indices]
            
            # Generate pattern ID
            pattern_id = f"pattern_{pattern_count:03d}"
            pattern_count += 1
            
            # Create profile
            profile = create_pattern_profile(
                pattern_id,
                tag_key,
                cluster_errors,
                all_annotations
            )
            
            patterns.append(profile)
            
            logger.info(f"Pattern {pattern_id}: {len(cluster_errors)} errors, "
                       f"human_acc={profile.metrics.human_acc:.2f}")
    
    logger.info(f"Discovered {len(patterns)} patterns")
    return patterns


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Pattern Discovery")
    logger.info("=" * 60)
    
    # Get file paths from config
    crosschecked_path = get_file_path('crosschecked_annotations', config, 'intermediate')
    embeddings_path = get_file_path('all_annotations_index', config, 'intermediate')
    output_path = get_file_path('edge_case_profiles', config, 'intermediate')
    
    # Validate input files
    if not validate_input_files([str(crosschecked_path), str(embeddings_path)], logger):
        logger.error("Missing required input files. Please run previous phase scripts first.")
        return
    
    try:
        # Load data
        logger.info("Loading data...")
        crosschecked = load_jsonl_raw(str(crosschecked_path))
        embeddings_index = load_jsonl_raw(str(embeddings_path))
        
        logger.info(f"✓ Loaded {len(crosschecked)} annotations")
        logger.info(f"✓ Loaded {len(embeddings_index)} embeddings")
        
        # Filter errors
        errors = [r for r in crosschecked if r.get('is_error')]
        logger.info(f"✓ Found {len(errors)} errors")
        
        if len(errors) == 0:
            logger.warning("No errors found. Cannot discover patterns.")
            logger.warning("This might happen if all annotations are correct.")
            return
        
        # Discover patterns
        patterns = discover_patterns(errors, crosschecked, embeddings_index)
        
        if len(patterns) == 0:
            logger.warning("No patterns discovered.")
            logger.warning("Try adjusting min_pattern_volume or clustering parameters in config.yaml")
            return
        
        # Allocate budget
        patterns = allocate_budget(patterns)
        
        # Convert to dicts for saving
        pattern_dicts = [p.model_dump() for p in patterns]
        
        # Write output
        save_jsonl_raw(pattern_dicts, str(output_path))
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Pattern Discovery Summary")
        logger.info("=" * 60)
        
        for pattern in sorted(patterns, key=lambda p: p.synthetic_budget, reverse=True):
            logger.info(f"\n{pattern.pattern_id}:")
            logger.info(f"  Description: {pattern.description[:80]}...")
            logger.info(f"  Tags: {pattern.tags.model_dump()}")
            logger.info(f"  Metrics: volume={pattern.metrics.volume}, "
                       f"human_acc={pattern.metrics.human_acc:.2%}, "
                       f"quality_gap={pattern.metrics.quality_gap:.2%}")
            logger.info(f"  Synthetic budget: {pattern.synthetic_budget}")
        
        logger.info("=" * 60)
        logger.info(f"✓ Saved {len(patterns)} patterns → {output_path.name}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error during pattern discovery: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

