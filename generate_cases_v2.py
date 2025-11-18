#!/usr/bin/env python3
"""
Script 5: Generate Synthetic Cases (Refactored)
Generates new synthetic cases for each pattern using LLM #1 with RAG context.

Input:
  - edge_case_profiles.jsonl
  - guideline_index.jsonl
  - example_index.jsonl

Output:
  - new_cases.jsonl
"""

import json
import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

from config import get_config
from models import EdgeCaseProfile, SyntheticCase, Metadata
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
    """
    Search index by cosine similarity to query embedding.
    
    Args:
        query_embedding: Query embedding vector
        index: List of items with 'embedding' field
        top_k: Number of top results to return
        
    Returns:
        Top K most similar items
    """
    if not index:
        return []
    
    query_vec = np.array(query_embedding)
    index_vecs = np.array([item['embedding'] for item in index])
    
    similarities = np.dot(index_vecs, query_vec) / (
        np.linalg.norm(index_vecs, axis=1) * np.linalg.norm(query_vec)
    )
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [index[i] for i in top_indices]


def get_pattern_embedding(pattern: EdgeCaseProfile) -> List[float]:
    """
    Create embedding for pattern based on description and tags.
    
    Args:
        pattern: Edge case profile
        
    Returns:
        Embedding vector
    """
    text = f"{pattern.description}\n"
    text += f"Intent: {pattern.tags.intent}\n"
    text += f"Entity Type: {pattern.tags.entity_type}\n"
    text += f"Locale: {pattern.tags.locale}"
    
    try:
        response = client.embeddings.create(
            model=config.llm.embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get pattern embedding: {e}")
        raise


def build_rag_context(
    pattern: EdgeCaseProfile,
    guideline_index: List[dict],
    example_index: List[dict]
) -> str:
    """
    Build RAG context for generation.
    
    Retrieves:
      - Relevant guideline snippets
      - Similar good examples
      - Pattern profile
      
    Args:
        pattern: Edge case profile
        guideline_index: Guideline chunks index
        example_index: Good examples index
        
    Returns:
        RAG context string
    """
    logger.debug(f"Building RAG context for pattern {pattern.pattern_id}")
    
    # Get pattern embedding
    pattern_embedding = get_pattern_embedding(pattern)
    
    # Retrieve relevant guidelines
    top_k_guidelines = config.rag.generation.top_k_guidelines
    relevant_guidelines = cosine_similarity_search(
        pattern_embedding,
        guideline_index,
        top_k_guidelines
    )
    
    # Retrieve similar good examples with matching tags
    filtered_examples = [
        ex for ex in example_index
        if ex.get('tags', {}).get('intent') == pattern.tags.intent
        or ex.get('tags', {}).get('entity_type') == pattern.tags.entity_type
    ]
    
    if not filtered_examples:
        filtered_examples = example_index
    
    top_k_examples = config.rag.generation.top_k_examples
    similar_examples = cosine_similarity_search(
        pattern_embedding,
        filtered_examples,
        top_k_examples
    )
    
    # Build context string
    context = "# EDGE-CASE PATTERN\n\n"
    context += f"Pattern ID: {pattern.pattern_id}\n"
    context += f"Description: {pattern.description}\n"
    context += f"Tags: {json.dumps(pattern.tags.model_dump(), indent=2)}\n"
    context += f"\n## Error Examples\n"
    
    for i, error in enumerate(pattern.seed_errors[:3], 1):
        context += f"\n{i}. Query: \"{error.query}\"\n"
        context += f"   Candidate: \"{error.candidate}\"\n"
        context += f"   Human Label: {error.label} (Incorrect, should be {error.golden_label})\n"
    
    context += "\n\n# RELEVANT GUIDELINES\n\n"
    for i, guide in enumerate(relevant_guidelines, 1):
        context += f"## Guideline {i} (Section {guide['section_id']}): {guide['title']}\n"
        context += f"{guide['text']}\n\n"
    
    context += "\n# GOOD EXAMPLES (Correct Annotations)\n\n"
    for i, example in enumerate(similar_examples, 1):
        context += f"{i}. Query: \"{example['query']}\"\n"
        context += f"   Candidate: \"{example['candidate']}\"\n"
        context += f"   Label: {example['label']}\n\n"
    
    return context


def generate_synthetic_cases(
    pattern: EdgeCaseProfile,
    rag_context: str,
    n_cases: int
) -> List[dict]:
    """
    Generate N synthetic cases for pattern using LLM #1.
    
    Args:
        pattern: Edge case profile
        rag_context: RAG context string
        n_cases: Number of cases to generate
        
    Returns:
        List of generated cases (as dicts)
    """
    logger.debug(f"Calling LLM to generate {n_cases} cases")
    
    system_prompt = """You are an expert at generating edge-case test queries for annotation quality testing.

Your task is to generate NEW query-candidate pairs that fit the given error pattern.
These should be realistic, challenging cases that human annotators might struggle with.

Requirements:
- Generate queries that match the pattern's tags (intent, entity_type, locale)
- Queries should be similar to the error examples but NOT identical
- Follow the locale's language (e.g., es-ES queries should be in Spanish)
- Create realistic candidates that would be confusing
- Output ONLY valid JSON"""

    user_prompt = f"""{rag_context}

---

Generate {n_cases} new synthetic query-candidate pairs that fit this edge-case pattern.

For each pair:
1. Create a query similar to the error examples above
2. Create a plausible but potentially confusing candidate
3. Ensure the query matches the locale ({pattern.tags.locale})
4. Keep the entity type as {pattern.tags.entity_type}

Output format (JSON array):
[
  {{
    "query": "example query here",
    "candidate": "example candidate here",
    "metadata": {{
      "locale": "{pattern.tags.locale}",
      "device": "mobile",
      "entity_type": "{pattern.tags.entity_type}"
    }}
  }},
  ...
]

Generate exactly {n_cases} pairs."""

    try:
        response = client.chat.completions.create(
            model=config.llm.generation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=config.llm.generation_temperature,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        
        # Parse response
        parsed = json.loads(result)
        
        # Handle different response formats
        if isinstance(parsed, list):
            cases = parsed
        elif 'cases' in parsed:
            cases = parsed['cases']
        elif 'pairs' in parsed:
            cases = parsed['pairs']
        else:
            cases = list(parsed.values())[0] if parsed else []
        
        logger.debug(f"LLM returned {len(cases)} cases")
        return cases
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return []


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Generate Synthetic Cases")
    logger.info("=" * 60)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    # Get file paths from config
    patterns_path = get_file_path('edge_case_profiles', config, 'intermediate')
    guidelines_path = get_file_path('guideline_index', config, 'intermediate')
    examples_path = get_file_path('example_index', config, 'intermediate')
    output_path = get_file_path('new_cases', config, 'intermediate')
    
    # Validate input files
    if not validate_input_files([str(patterns_path), str(guidelines_path), str(examples_path)], logger):
        logger.error("Missing required input files. Please run previous phase scripts first.")
        return
    
    try:
        # Load data
        logger.info("Loading data...")
        patterns_data = load_jsonl_raw(str(patterns_path))
        guideline_index = load_jsonl_raw(str(guidelines_path))
        example_index = load_jsonl_raw(str(examples_path))
        
        # Parse patterns into Pydantic models
        patterns = [EdgeCaseProfile(**p) for p in patterns_data]
        
        logger.info(f"✓ Loaded {len(patterns)} patterns")
        logger.info(f"✓ Loaded {len(guideline_index)} guideline chunks")
        logger.info(f"✓ Loaded {len(example_index)} good examples")
        
        # Filter patterns with budget > 0
        patterns_to_generate = [p for p in patterns if p.synthetic_budget > 0]
        
        if not patterns_to_generate:
            logger.warning("No patterns with allocated budget.")
            return
        
        logger.info(f"✓ Found {len(patterns_to_generate)} patterns with budget > 0")
        
        # Generate cases for each pattern
        all_cases: List[dict] = []
        case_id_counter = 0
        
        for i, pattern in enumerate(patterns_to_generate, 1):
            pattern_id = pattern.pattern_id
            budget = pattern.synthetic_budget
            
            logger.info(f"[{i}/{len(patterns_to_generate)}] Generating cases for {pattern_id}...")
            logger.info(f"  Budget: {budget} cases")
            
            # Build RAG context
            logger.debug("Building RAG context...")
            rag_context = build_rag_context(pattern, guideline_index, example_index)
            
            # Generate cases
            logger.info(f"  Calling LLM #1...")
            generated_cases = generate_synthetic_cases(pattern, rag_context, budget)
            
            logger.info(f"  ✓ Generated {len(generated_cases)} cases")
            
            # Add pattern_id and case_id
            for case in generated_cases:
                case_id = f"syn_{case_id_counter:05d}"
                case_id_counter += 1
                
                case['case_id'] = case_id
                case['pattern_id'] = pattern_id
                
                all_cases.append(case)
        
        # Write output
        save_jsonl_raw(all_cases, str(output_path))
        
        logger.info("=" * 60)
        logger.info(f"✓ Generated {len(all_cases)} synthetic cases")
        logger.info("=" * 60)
        logger.info(f"Output file: {output_path.name}")
        
        # Print sample
        if all_cases:
            logger.info("\nSample cases:")
            for case in all_cases[:3]:
                logger.info(f"\n  {case['case_id']} (pattern: {case['pattern_id']}):")
                logger.info(f"    Query: {case['query']}")
                logger.info(f"    Candidate: {case['candidate']}")
        
    except Exception as e:
        logger.error(f"Fatal error during case generation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

