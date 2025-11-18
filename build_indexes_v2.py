#!/usr/bin/env python3
"""
Script 3: Build Embedding Indexes (Refactored)
Creates three embedding indexes from prepared data.

Input:
  - prepared_guidelines.json
  - crosschecked_annotations.jsonl
  - tagged_goldens.jsonl

Output:
  - guideline_index.jsonl
  - example_index.jsonl
  - all_annotations_index.jsonl
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict
import tiktoken
from openai import OpenAI

from config import get_config
from models import Guidelines, GuidelineChunk, ExampleIndex, AnnotationIndex, Tags
from utils import setup_logging, load_jsonl_raw, save_jsonl_raw, get_file_path, validate_input_files


# Setup logger and config
config = get_config()
logger = setup_logging(__name__, config)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def chunk_text(text: str, min_size: int, max_size: int) -> List[str]:
    """
    Split text into chunks of min_size to max_size tokens.
    Tries to break at sentence boundaries.
    
    Args:
        text: Text to chunk
        min_size: Minimum chunk size in tokens
        max_size: Maximum chunk size in tokens
        
    Returns:
        List of text chunks
    """
    if count_tokens(text) <= max_size:
        return [text]
    
    # Split by sentences (simple heuristic)
    sentences = text.replace('\n', ' ').split('. ')
    
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Add period back if it was removed
        if not sentence.endswith('.'):
            sentence += '.'
        
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence exceeds max_size, add it as its own chunk
        if sentence_tokens > max_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(sentence)
            logger.debug(f"Large sentence ({sentence_tokens} tokens) added as single chunk")
            continue
        
        # If adding this sentence would exceed max_size, start new chunk
        if current_tokens + sentence_tokens > max_size:
            if current_tokens >= min_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                # Current chunk too small, add sentence anyway
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logger.debug(f"Text chunked into {len(chunks)} pieces")
    return chunks


def get_embedding(text: str) -> List[float]:
    """
    Get embedding for text using OpenAI API.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
        
    Raises:
        Exception: If API call fails
    """
    try:
        response = client.embeddings.create(
            model=config.llm.embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise


def build_guideline_index(guidelines_path: Path, output_path: Path) -> int:
    """
    Build GUIDELINE_INDEX from prepared guidelines.
    Splits into chunks and embeds each chunk.
    
    Args:
        guidelines_path: Path to prepared_guidelines.json
        output_path: Path to output index file
        
    Returns:
        Number of chunks created
    """
    logger.info("Building GUIDELINE_INDEX...")
    
    with open(guidelines_path, 'r', encoding='utf-8') as f:
        guidelines_data = json.load(f)
    
    guidelines = Guidelines(**guidelines_data)
    
    index_entries: List[dict] = []
    chunk_id = 0
    
    min_tokens = config.chunking.guidelines.min_tokens
    max_tokens = config.chunking.guidelines.max_tokens
    
    for section in guidelines.sections:
        section_id = section.section_id
        title = section.title
        text = section.text
        
        # Skip empty sections
        if not text.strip():
            logger.debug(f"Skipping empty section {section_id}")
            continue
        
        # Create full text for chunking
        full_text = f"{title}\n\n{text}"
        
        # Split into chunks
        chunks = chunk_text(full_text, min_tokens, max_tokens)
        
        for chunk in chunks:
            # Get embedding
            embedding = get_embedding(chunk)
            
            entry = {
                "chunk_id": f"guide_{chunk_id:04d}",
                "section_id": section_id,
                "title": title,
                "text": chunk,
                "tokens": count_tokens(chunk),
                "embedding": embedding
            }
            
            index_entries.append(entry)
            chunk_id += 1
            
            if chunk_id % 5 == 0:
                logger.info(f"  Processed {chunk_id} chunks...")
    
    # Write to output
    save_jsonl_raw(index_entries, str(output_path))
    
    logger.info(f"✓ Built GUIDELINE_INDEX: {len(index_entries)} chunks → {output_path.name}")
    return len(index_entries)


def build_example_index(
    crosschecked_path: Path,
    output_path: Path
) -> int:
    """
    Build EXAMPLE_INDEX from good examples.
    Filter: is_golden==true AND is_correct==true
    
    Args:
        crosschecked_path: Path to crosschecked_annotations.jsonl
        output_path: Path to output index file
        
    Returns:
        Number of examples indexed
    """
    logger.info("Building EXAMPLE_INDEX...")
    
    # Load crosschecked annotations
    all_annotations = load_jsonl_raw(str(crosschecked_path))
    
    # Filter: good examples only
    examples = [
        r for r in all_annotations
        if r.get('is_golden') and r.get('is_correct')
    ]
    
    logger.info(f"  Found {len(examples)} good examples")
    
    # Embed each example
    index_entries: List[dict] = []
    
    for i, example in enumerate(examples):
        # Create embedding text: query + candidate + metadata
        embed_text = f"Query: {example['query']}\nCandidate: {example['candidate']}\nLabel: {example['label']}"
        
        # Add relevant metadata
        tags = example.get('tags', {})
        embed_text += f"\nIntent: {tags.get('intent', 'unknown')}"
        embed_text += f"\nEntity Type: {tags.get('entity_type', 'unknown')}"
        embed_text += f"\nLocale: {tags.get('locale', 'unknown')}"
        
        # Get embedding
        embedding = get_embedding(embed_text)
        
        entry = {
            "example_id": f"ex_{i:04d}",
            "query": example['query'],
            "candidate": example['candidate'],
            "label": example['label'],
            "tags": tags,
            "metadata": example.get('metadata', {}),
            "embedding": embedding
        }
        
        index_entries.append(entry)
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Processed {i + 1}/{len(examples)} examples...")
    
    # Write to output
    save_jsonl_raw(index_entries, str(output_path))
    
    logger.info(f"✓ Built EXAMPLE_INDEX: {len(index_entries)} examples → {output_path.name}")
    return len(index_entries)


def build_all_annotations_index(annotations_path: Path, output_path: Path) -> int:
    """
    Build ALL_ANNOTATIONS_INDEX from all annotations.
    Embeds all annotations (including errors).
    
    Args:
        annotations_path: Path to crosschecked_annotations.jsonl
        output_path: Path to output index file
        
    Returns:
        Number of annotations indexed
    """
    logger.info("Building ALL_ANNOTATIONS_INDEX...")
    
    # Load all annotations
    annotations = load_jsonl_raw(str(annotations_path))
    
    logger.info(f"  Found {len(annotations)} annotations")
    
    # Embed each annotation
    index_entries: List[dict] = []
    
    for i, annotation in enumerate(annotations):
        # Create embedding text
        embed_text = f"Query: {annotation['query']}\nCandidate: {annotation['candidate']}\nLabel: {annotation['label']}"
        
        # Add tags
        tags = annotation.get('tags', {})
        embed_text += f"\nIntent: {tags.get('intent', 'unknown')}"
        embed_text += f"\nEntity Type: {tags.get('entity_type', 'unknown')}"
        embed_text += f"\nLocale: {tags.get('locale', 'unknown')}"
        
        # Get embedding
        embedding = get_embedding(embed_text)
        
        entry = {
            "annotation_id": f"ann_{i:04d}",
            "query": annotation['query'],
            "candidate": annotation['candidate'],
            "label": annotation['label'],
            "tags": tags,
            "metadata": annotation.get('metadata', {}),
            "is_golden": annotation.get('is_golden', False),
            "is_correct": annotation.get('is_correct'),
            "is_error": annotation.get('is_error', False),
            "embedding": embedding
        }
        
        index_entries.append(entry)
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Processed {i + 1}/{len(annotations)} annotations...")
    
    # Write to output
    save_jsonl_raw(index_entries, str(output_path))
    
    logger.info(f"✓ Built ALL_ANNOTATIONS_INDEX: {len(index_entries)} annotations → {output_path.name}")
    return len(index_entries)


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Build Embedding Indexes")
    logger.info("=" * 60)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.error("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Get file paths from config
    guidelines_path = get_file_path('prepared_guidelines', config, 'intermediate')
    crosschecked_path = get_file_path('crosschecked_annotations', config, 'intermediate')
    
    guideline_index_path = get_file_path('guideline_index', config, 'intermediate')
    example_index_path = get_file_path('example_index', config, 'intermediate')
    all_annotations_index_path = get_file_path('all_annotations_index', config, 'intermediate')
    
    # Validate input files
    if not validate_input_files([str(guidelines_path), str(crosschecked_path)], logger):
        logger.error("Missing required input files. Please run previous phase scripts first.")
        return
    
    try:
        # Build indexes
        guideline_count = build_guideline_index(guidelines_path, guideline_index_path)
        example_count = build_example_index(crosschecked_path, example_index_path)
        annotations_count = build_all_annotations_index(crosschecked_path, all_annotations_index_path)
        
        # Summary
        logger.info("=" * 60)
        logger.info("✓ All indexes built successfully!")
        logger.info("=" * 60)
        logger.info("Output files:")
        logger.info(f"  - {guideline_index_path.name} ({guideline_count} chunks)")
        logger.info(f"  - {example_index_path.name} ({example_count} examples)")
        logger.info(f"  - {all_annotations_index_path.name} ({annotations_count} annotations)")
        
    except Exception as e:
        logger.error(f"Fatal error during index building: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

