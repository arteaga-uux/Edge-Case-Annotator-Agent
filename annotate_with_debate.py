#!/usr/bin/env python3
"""
Script 6: Annotate with Three-LLM Adversarial Debate (Refactored)
Annotates synthetic cases using a multi-LLM debate system.

Input:
  - new_cases.jsonl
  - guideline_index.jsonl
  - example_index.jsonl
  - edge_case_profiles.jsonl
  - resolved_synthetics_index.jsonl (optional, created if doesn't exist)

Output:
  - synthetic_annotations.jsonl
  - ambiguous_cases.jsonl
  - resolved_synthetics_index.jsonl (updated)
"""

import json
import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Literal
from openai import OpenAI

from config import get_config
from models import (
    EdgeCaseProfile, SyntheticCase, OptimisticAnnotation,
    StrictCritique, NeutralJudgment, AnnotatedCase, ResolvedSynthetic
)
from utils import setup_logging, load_jsonl_raw, save_jsonl_raw, get_file_path, validate_input_files


# Setup logger and config
config = get_config()
logger = setup_logging(__name__, config)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def cosine_similarity_search(
    query_embedding: List[float],
    index: List[dict],
    top_k: int
) -> List[dict]:
    """Search index by cosine similarity."""
    if not index:
        return []
    
    query_vec = np.array(query_embedding)
    index_vecs = np.array([item['embedding'] for item in index])
    
    similarities = np.dot(index_vecs, query_vec) / (
        np.linalg.norm(index_vecs, axis=1) * np.linalg.norm(query_vec)
    )
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [index[i] for i in top_indices]


def get_case_embedding(case: dict) -> List[float]:
    """Create embedding for a case."""
    text = f"Query: {case['query']}\nCandidate: {case['candidate']}"
    
    if 'metadata' in case:
        metadata = case['metadata']
        text += f"\nLocale: {metadata.get('locale', 'unknown')}"
        text += f"\nEntity Type: {metadata.get('entity_type', 'unknown')}"
    
    try:
        response = client.embeddings.create(
            model=config.llm.embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get case embedding: {e}")
        raise


def build_rag_context(
    case: dict,
    pattern: dict,
    guideline_index: List[dict],
    example_index: List[dict],
    resolved_index: List[dict]
) -> str:
    """
    Build shared RAG context (Stage 1 + Stage 2).
    
    Stage 1 (from original data):
      - Relevant guideline snippets
      - Similar good examples
      - Edge-case profile
    
    Stage 2 (from resolved synthetics):
      - Previously resolved synthetic cases
    """
    logger.debug("Building RAG context (Stage 1 + Stage 2)")
    
    # Get case embedding
    case_embedding = get_case_embedding(case)
    
    # Stage 1: Retrieve from original data
    top_k_guidelines = config.rag.annotation.top_k_guidelines
    top_k_examples = config.rag.annotation.top_k_examples
    
    relevant_guidelines = cosine_similarity_search(case_embedding, guideline_index, top_k_guidelines)
    similar_examples = cosine_similarity_search(case_embedding, example_index, top_k_examples)
    
    # Stage 2: Retrieve from resolved synthetics
    top_k_resolved = config.rag.annotation.top_k_resolved
    resolved_similar = cosine_similarity_search(case_embedding, resolved_index, top_k_resolved) if resolved_index else []
    
    # Build context string
    context = "# ANNOTATION TASK\n\n"
    context += f"Query: \"{case['query']}\"\n"
    context += f"Candidate: \"{case['candidate']}\"\n"
    context += f"Metadata: {json.dumps(case.get('metadata', {}), indent=2)}\n"
    
    context += "\n# EDGE-CASE PATTERN CONTEXT\n\n"
    context += f"This case belongs to pattern: {pattern['pattern_id']}\n"
    context += f"Description: {pattern['description']}\n"
    context += f"Common errors in this pattern:\n"
    for i, error in enumerate(pattern['seed_errors'][:2], 1):
        context += f"  {i}. Query: \"{error['query']}\" → Human incorrectly labeled as {error['label']} (should be {error['golden_label']})\n"
    
    context += "\n# ANNOTATION GUIDELINES\n\n"
    context += "Labels: 0 (Not Relevant), 1 (Somewhat Relevant), 2 (Highly Relevant)\n\n"
    
    for i, guide in enumerate(relevant_guidelines, 1):
        context += f"## Section {guide['section_id']}: {guide['title']}\n"
        context += f"{guide['text']}\n\n"
    
    context += "\n# GOOD EXAMPLES (Correct Annotations)\n\n"
    for i, example in enumerate(similar_examples, 1):
        context += f"{i}. Query: \"{example['query']}\"\n"
        context += f"   Candidate: \"{example['candidate']}\"\n"
        context += f"   Label: {example['label']}\n"
        context += f"   Tags: {example.get('tags', {})}\n\n"
    
    if resolved_similar:
        context += "\n# PREVIOUSLY RESOLVED SYNTHETIC CASES\n\n"
        context += "(These are similar synthetic cases that were successfully annotated)\n\n"
        for i, resolved in enumerate(resolved_similar, 1):
            context += f"{i}. Query: \"{resolved['query']}\"\n"
            context += f"   Candidate: \"{resolved['candidate']}\"\n"
            context += f"   Final Label: {resolved['final_label']}\n"
            rationale = resolved.get('final_rationale', '')[:200]
            context += f"   Rationale: {rationale}...\n\n"
    
    return context


def llm_2_optimistic_annotator(case: dict, rag_context: str) -> dict:
    """LLM #2: Optimistic Annotator."""
    logger.debug("Calling LLM #2 (Optimistic Annotator)")
    
    system_prompt = """You are an optimistic annotator evaluating search result relevance.

Your approach:
- If the candidate could REASONABLY satisfy the user's intent, favor higher labels
- Look for ways the candidate COULD be relevant
- Be generous but still follow the guidelines
- Cite specific guideline sections that support your annotation

Output valid JSON only."""

    user_prompt = f"""{rag_context}

---

Annotate the query-candidate pair above.

Provide:
1. label (0, 1, or 2)
2. rationale (explain why this label is appropriate, cite guidelines)
3. guideline_citations (list of section IDs you referenced)
4. confidence (0.0 to 1.0)

Output format:
{{
  "label": 2,
  "rationale": "...",
  "guideline_citations": ["4.1", "7.3"],
  "confidence": 0.85
}}"""

    try:
        response = client.chat.completions.create(
            model=config.llm.annotation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.llm.annotation_temperature,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.debug(f"LLM #2 result: label={result.get('label')}, confidence={result.get('confidence')}")
        return result
        
    except Exception as e:
        logger.error(f"LLM #2 call failed: {e}")
        raise


def llm_3_strict_critic(case: dict, rag_context: str, annotation: dict) -> dict:
    """LLM #3: Strict Critic."""
    logger.debug("Calling LLM #3 (Strict Critic)")
    
    system_prompt = """You are a strict quality reviewer analyzing annotations.

Your approach:
- Look for PROBLEMS with the proposed annotation
- What did the annotator MISS or get WRONG?
- Find guidelines that suggest a LOWER label
- Be critical and thorough
- Only agree if the annotation is clearly correct

Output valid JSON only."""

    user_prompt = f"""{rag_context}

---

PROPOSED ANNOTATION:
{json.dumps(annotation, indent=2)}

---

Review this annotation critically.

Provide:
1. agrees (true/false) - do you agree with this annotation?
2. alternative_label (if disagree, what should it be?)
3. counter_rationale (explain problems with original annotation)
4. guideline_citations (sections supporting your critique)
5. strength ("weak", "moderate", "strong") - how strong is your disagreement?

Output format:
{{
  "agrees": false,
  "alternative_label": 1,
  "counter_rationale": "...",
  "guideline_citations": ["4.2"],
  "strength": "strong"
}}"""

    try:
        response = client.chat.completions.create(
            model=config.llm.annotation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.llm.critique_temperature,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.debug(f"LLM #3 result: agrees={result.get('agrees')}, strength={result.get('strength')}")
        return result
        
    except Exception as e:
        logger.error(f"LLM #3 call failed: {e}")
        raise


def llm_4_neutral_judge(case: dict, rag_context: str, annotation: dict, critique: dict) -> dict:
    """LLM #4: Neutral Judge."""
    logger.debug("Calling LLM #4 (Neutral Judge)")
    
    system_prompt = """You are a neutral judge resolving annotation debates.

Your approach:
- Weigh BOTH arguments objectively
- Who cited guidelines more accurately?
- Whose reasoning is more complete and grounded?
- Consider confidence and strength of arguments
- Decide: accept original, accept alternative, or mark ambiguous

Output valid JSON only."""

    user_prompt = f"""{rag_context}

---

OPTIMISTIC ANNOTATOR'S ANNOTATION:
{json.dumps(annotation, indent=2)}

STRICT CRITIC'S REVIEW:
{json.dumps(critique, indent=2)}

---

As a neutral judge, make a final decision.

Provide:
1. decision ("accept_original", "accept_alternative", or "ambiguous")
2. final_label (the label you're accepting)
3. final_rationale (explain your decision)
4. reasoning (why did you choose this over the other?)

Use "ambiguous" if:
- Both sides have equally strong arguments
- Guidelines are unclear or contradictory
- You cannot confidently determine the correct label

Output format:
{{
  "decision": "accept_alternative",
  "final_label": 1,
  "final_rationale": "...",
  "reasoning": "..."
}}"""

    try:
        response = client.chat.completions.create(
            model=config.llm.annotation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.llm.judge_temperature,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.debug(f"LLM #4 result: decision={result.get('decision')}, final_label={result.get('final_label')}")
        return result
        
    except Exception as e:
        logger.error(f"LLM #4 call failed: {e}")
        raise


def annotate_case_with_debate(
    case: dict,
    pattern: dict,
    guideline_index: List[dict],
    example_index: List[dict],
    resolved_index: List[dict]
) -> dict:
    """Annotate a single case using the three-LLM debate system."""
    # Build shared RAG context
    rag_context = build_rag_context(case, pattern, guideline_index, example_index, resolved_index)
    
    # LLM #2: Optimistic Annotator
    annotation = llm_2_optimistic_annotator(case, rag_context)
    
    # LLM #3: Strict Critic
    critique = llm_3_strict_critic(case, rag_context, annotation)
    
    # LLM #4: Neutral Judge
    judgment = llm_4_neutral_judge(case, rag_context, annotation, critique)
    
    # Build result
    result = {
        **case,
        "annotation": annotation,
        "critique": critique,
        "judgment": judgment,
        "final_label": judgment.get("final_label"),
        "final_rationale": judgment.get("final_rationale"),
        "decision": judgment.get("decision")
    }
    
    # Set qa_status based on decision
    if judgment.get("decision") == "ambiguous":
        result["qa_status"] = "needs_human_review"
    else:
        result["qa_status"] = "judge_accepted"
    
    return result


def update_resolved_index(annotated_case: dict, resolved_index: List[dict]) -> List[dict]:
    """Add judge_accepted case to resolved synthetics index."""
    if annotated_case.get("qa_status") != "judge_accepted":
        return resolved_index
    
    # Get embedding for this case
    case_embedding = get_case_embedding(annotated_case)
    
    # Create index entry
    entry = {
        "case_id": annotated_case["case_id"],
        "query": annotated_case["query"],
        "candidate": annotated_case["candidate"],
        "final_label": annotated_case["final_label"],
        "final_rationale": annotated_case["final_rationale"],
        "pattern_id": annotated_case.get("pattern_id"),
        "qa_status": annotated_case["qa_status"],
        "embedding": case_embedding
    }
    
    resolved_index.append(entry)
    return resolved_index


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("PHASE 5: Annotate with Three-LLM Debate")
    logger.info("=" * 60)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Get file paths from config
    cases_path = get_file_path('new_cases', config, 'intermediate')
    guidelines_path = get_file_path('guideline_index', config, 'intermediate')
    examples_path = get_file_path('example_index', config, 'intermediate')
    patterns_path = get_file_path('edge_case_profiles', config, 'intermediate')
    resolved_path = get_file_path('resolved_synthetics_index', config, 'intermediate')
    
    output_synthetic_path = get_file_path('synthetic_annotations', config, 'intermediate')
    output_ambiguous_path = get_file_path('ambiguous_cases', config, 'intermediate')
    
    # Validate input files
    required_files = [str(cases_path), str(guidelines_path), str(examples_path), str(patterns_path)]
    if not validate_input_files(required_files, logger):
        logger.error("Missing required input files. Please run previous phase scripts first.")
        return
    
    try:
        # Load data
        logger.info("Loading data...")
        cases = load_jsonl_raw(str(cases_path))
        guideline_index = load_jsonl_raw(str(guidelines_path))
        example_index = load_jsonl_raw(str(examples_path))
        patterns = load_jsonl_raw(str(patterns_path))
        resolved_index = load_jsonl_raw(str(resolved_path))  # OK if empty
        
        logger.info(f"✓ Loaded {len(cases)} synthetic cases")
        logger.info(f"✓ Loaded {len(guideline_index)} guideline chunks")
        logger.info(f"✓ Loaded {len(example_index)} good examples")
        logger.info(f"✓ Loaded {len(patterns)} patterns")
        logger.info(f"✓ Loaded {len(resolved_index)} resolved synthetics")
        
        if not cases:
            logger.warning("No synthetic cases to annotate.")
            return
        
        # Build pattern lookup
        pattern_lookup = {p['pattern_id']: p for p in patterns}
        
        # Annotate each case
        synthetic_annotations = []
        ambiguous_cases = []
        
        for i, case in enumerate(cases, 1):
            case_id = case.get('case_id', f'case_{i}')
            pattern_id = case.get('pattern_id', 'unknown')
            
            logger.info(f"[{i}/{len(cases)}] Annotating {case_id}...")
            logger.info(f"  Query: \"{case['query'][:50]}...\"")
            logger.info(f"  Pattern: {pattern_id}")
            
            # Get pattern
            pattern = pattern_lookup.get(pattern_id)
            if not pattern:
                logger.warning(f"Pattern {pattern_id} not found, skipping...")
                continue
            
            try:
                annotated = annotate_case_with_debate(case, pattern, guideline_index, example_index, resolved_index)
                
                decision = annotated['decision']
                final_label = annotated['final_label']
                qa_status = annotated['qa_status']
                
                logger.info(f"  ✓ Decision: {decision}, Label: {final_label}, QA Status: {qa_status}")
                
                # Route based on qa_status
                if qa_status == "needs_human_review":
                    ambiguous_cases.append(annotated)
                else:
                    synthetic_annotations.append(annotated)
                    resolved_index = update_resolved_index(annotated, resolved_index)
                
            except Exception as e:
                logger.error(f"Error annotating case {case_id}: {e}", exc_info=True)
                continue
        
        # Write outputs
        logger.info("=" * 60)
        logger.info("Writing outputs...")
        
        save_jsonl_raw(synthetic_annotations, str(output_synthetic_path))
        logger.info(f"✓ Wrote {len(synthetic_annotations)} synthetic annotations → {output_synthetic_path.name}")
        
        save_jsonl_raw(ambiguous_cases, str(output_ambiguous_path))
        logger.info(f"✓ Wrote {len(ambiguous_cases)} ambiguous cases → {output_ambiguous_path.name}")
        
        save_jsonl_raw(resolved_index, str(resolved_path))
        logger.info(f"✓ Updated resolved index: {len(resolved_index)} entries → {resolved_path.name}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("Annotation Summary")
        logger.info("=" * 60)
        logger.info(f"Total cases processed: {len(cases)}")
        logger.info(f"Judge accepted: {len(synthetic_annotations)} ({len(synthetic_annotations)/len(cases)*100:.1f}%)")
        logger.info(f"Needs human review: {len(ambiguous_cases)} ({len(ambiguous_cases)/len(cases)*100:.1f}%)")
        
        # Label distribution
        if synthetic_annotations:
            label_dist = {}
            for ann in synthetic_annotations:
                label = ann.get('final_label', 'unknown')
                label_dist[label] = label_dist.get(label, 0) + 1
            
            logger.info("\nLabel distribution (judge accepted):")
            for label in sorted(label_dist.keys()):
                count = label_dist[label]
                logger.info(f"  Label {label}: {count} ({count/len(synthetic_annotations)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Fatal error during annotation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

