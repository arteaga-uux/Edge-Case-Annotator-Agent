#!/usr/bin/env python3
"""
Synthetic Dataset Generator
Generates golden annotations and human annotations (with errors) for testing.

This script is independent from the Edge-Case Annotator Agent.
It creates synthetic training data by:
1. Extracting examples from guidelines PDF
2. Generating additional goldens as variations
3. Creating annotations with controlled error rates

Usage:
  python generate_synthetic_dataset.py --goldens 300 --annotations 2000 --golden-error-rate 0.30 --annotation-error-rate 0.15

Output:
  - generated_goldens.jsonl
  - generated_annotations.jsonl
"""

import json
import argparse
import os
import random
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pdfplumber
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# LLM Configuration
EXTRACTION_MODEL = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE_GENERATION = 0.7  # Lower for more consistency with examples
TEMPERATURE_ERROR = 0.7

# RAG Configuration
TOP_K_SIMILAR_EXAMPLES = 8  # Retrieve more similar examples for context


class GoldenExample:
    """Represents a golden annotation example."""
    def __init__(
        self,
        query: str,
        candidate: str,
        label: int,
        rationale: Optional[str] = None,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ):
        self.query = query
        self.candidate = candidate
        self.label = label
        self.rationale = rationale
        self.metadata = metadata or {}
        self.embedding = embedding
    
    def to_dict(self, include_embedding: bool = False) -> Dict:
        """Convert to dictionary."""
        result = {
            "query": self.query,
            "candidate": self.candidate,
            "label": self.label,
            "rationale": self.rationale,
            "metadata": self.metadata
        }
        if include_embedding and self.embedding:
            result["embedding"] = self.embedding
        return result
    
    def get_text_for_embedding(self) -> str:
        """Get text representation for embedding."""
        text = f"Query: {self.query}\nCandidate: {self.candidate}\nLabel: {self.label}"
        if self.rationale:
            text += f"\nRationale: {self.rationale}"
        if self.metadata:
            text += f"\nMetadata: {json.dumps(self.metadata)}"
        return text


def get_embedding(text: str) -> List[float]:
    """
    Get embedding for text using OpenAI API.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise


def embed_examples(examples: List[GoldenExample], batch_size: int = 50) -> List[GoldenExample]:
    """
    Add embeddings to examples in batches.
    
    Args:
        examples: List of GoldenExample objects
        batch_size: Number of examples to embed per batch
        
    Returns:
        Updated list with embeddings
    """
    logger.info(f"Generating embeddings for {len(examples)} examples...")
    
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        logger.info(f"  Embedding batch {i//batch_size + 1}/{(len(examples)-1)//batch_size + 1}...")
        
        for example in batch:
            if not example.embedding:
                text = example.get_text_for_embedding()
                example.embedding = get_embedding(text)
    
    logger.info(f"✓ All examples embedded")
    return examples


def find_similar_examples(
    query_text: str,
    examples: List[GoldenExample],
    top_k: int = TOP_K_SIMILAR_EXAMPLES
) -> List[GoldenExample]:
    """
    Find most similar examples using embedding similarity.
    
    Args:
        query_text: Text to search for
        examples: List of examples with embeddings
        top_k: Number of similar examples to return
        
    Returns:
        Top K most similar examples
    """
    # Get query embedding
    query_embedding = get_embedding(query_text)
    query_vec = np.array(query_embedding).reshape(1, -1)
    
    # Get all example embeddings
    example_embeddings = np.array([ex.embedding for ex in examples if ex.embedding])
    
    if len(example_embeddings) == 0:
        logger.warning("No examples with embeddings found")
        return examples[:top_k]
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_vec, example_embeddings)[0]
    
    # Get top K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return top K examples
    return [examples[i] for i in top_indices]


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    text_chunks = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"PDF has {total_pages} pages")
        
        for i, page in enumerate(pdf.pages, 1):
            if i % 10 == 0:
                logger.info(f"  Processing page {i}/{total_pages}...")
            
            text = page.extract_text()
            if text:
                text_chunks.append(text)
    
    full_text = "\n".join(text_chunks)
    logger.info(f"Extracted {len(full_text)} characters from PDF")
    
    return full_text


def extract_examples_from_guidelines_llm(
    guidelines_text: str,
    chunk_size: int = 15000
) -> List[GoldenExample]:
    """
    Extract golden examples from guidelines using LLM.
    Processes text in chunks to handle large documents.
    
    Args:
        guidelines_text: Full guidelines text
        chunk_size: Size of text chunks to process
        
    Returns:
        List of GoldenExample objects
    """
    logger.info("Extracting examples from guidelines using LLM...")
    
    # Split text into chunks
    words = guidelines_text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    logger.info(f"Split guidelines into {len(chunks)} chunks")
    
    all_examples = []
    
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Processing chunk {i}/{len(chunks)}...")
        
        system_prompt = """You are an expert at extracting annotation examples from guidelines.

Extract ALL examples where there is:
1. A query (user search/question)
2. A candidate (result/answer)
3. A label (relevance score: 0, 1, or 2)
4. Optionally a rationale/explanation

Output ONLY valid JSON array of examples."""

        user_prompt = f"""Extract ALL annotation examples from this guidelines text.

Guidelines text:
{chunk}

---

For each example found, extract:
- query: the user's query/search
- candidate: the result/answer being evaluated
- label: relevance score (0=Not Relevant, 1=Somewhat Relevant, 2=Highly Relevant)
- rationale: explanation (if provided)
- metadata: any additional context (locale, intent, etc.)

Output format (JSON array):
[
  {{
    "query": "...",
    "candidate": "...",
    "label": 2,
    "rationale": "...",
    "metadata": {{"locale": "en-US"}}
  }},
  ...
]

If no examples found in this chunk, return empty array: []"""

        try:
            response = client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            parsed = json.loads(result)
            
            # Handle different response formats
            if isinstance(parsed, list):
                examples_data = parsed
            elif 'examples' in parsed:
                examples_data = parsed['examples']
            elif 'results' in parsed:
                examples_data = parsed['results']
            else:
                examples_data = list(parsed.values())[0] if parsed else []
            
            # Convert to GoldenExample objects
            for ex_data in examples_data:
                if 'query' in ex_data and 'candidate' in ex_data and 'label' in ex_data:
                    example = GoldenExample(
                        query=ex_data['query'],
                        candidate=ex_data['candidate'],
                        label=int(ex_data['label']),
                        rationale=ex_data.get('rationale'),
                        metadata=ex_data.get('metadata', {})
                    )
                    all_examples.append(example)
            
            logger.info(f"  Extracted {len(examples_data)} examples from chunk {i}")
            
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            continue
    
    logger.info(f"Total examples extracted: {len(all_examples)}")
    return all_examples


def generate_golden_variations(
    seed_examples: List[GoldenExample],
    target_count: int,
    guidelines_sample: str
) -> List[GoldenExample]:
    """
    Generate additional goldens as variations of seed examples using RAG.
    
    Args:
        seed_examples: Original examples from guidelines (WITH EMBEDDINGS)
        target_count: Total number of goldens desired
        guidelines_sample: Sample of guidelines for context
        
    Returns:
        List of generated GoldenExample objects
    """
    needed = target_count - len(seed_examples)
    
    if needed <= 0:
        logger.info(f"Already have {len(seed_examples)} examples, no variations needed")
        return []
    
    logger.info(f"Generating {needed} golden variations using RAG...")
    
    generated = []
    
    # Generate in batches of 10
    batch_size = 10
    batches = (needed + batch_size - 1) // batch_size
    
    for batch_num in range(batches):
        batch_count = min(batch_size, needed - len(generated))
        
        # Select a random theme/topic from seed examples to guide this batch
        theme_example = random.choice(seed_examples)
        theme_query = f"Generate variations similar to: {theme_example.query}"
        
        # Use RAG to find most similar examples
        similar_examples = find_similar_examples(theme_query, seed_examples, top_k=TOP_K_SIMILAR_EXAMPLES)
        
        logger.info(f"Generating batch {batch_num + 1}/{batches} ({batch_count} examples)...")
        logger.debug(f"  Theme: {theme_example.query}")
        
        system_prompt = """You are an expert annotation specialist creating test cases for a music search system.

CRITICAL RULES:
1. Follow the EXACT style and format of the provided examples from the guidelines
2. Maintain the SAME quality standards shown in the examples
3. Create REALISTIC queries and candidates that users would encounter
4. Rationales must be 2-3 concise sentences explaining the relevance
5. Stay consistent with the annotation guidelines shown in examples

Output ONLY valid JSON."""

        # Format similar examples (RAG context)
        examples_text = "\n\n".join([
            f"EXAMPLE {i+1} (from guidelines):\n"
            f"Query: \"{ex.query}\"\n"
            f"Candidate: \"{ex.candidate}\"\n"
            f"Label: {ex.label}\n"
            f"Rationale: {ex.rationale or 'N/A'}\n"
            f"Metadata: {json.dumps(ex.metadata)}"
            for i, ex in enumerate(similar_examples[:8])
        ])
        
        user_prompt = f"""Generate {batch_count} NEW annotation examples that follow the EXACT style of these examples from the guidelines:

{examples_text}

Guidelines context:
{guidelines_sample[:1500]}

---

REQUIREMENTS:
1. Create query-candidate pairs SIMILAR to but NOT identical to examples above
2. Follow the SAME annotation logic shown in the examples
3. Label correctly based on relevance (0=Not Relevant, 1=Somewhat Relevant, 2=Highly Relevant)
4. Provide 2-3 sentence rationales like in the examples
5. Match metadata style (locale, intent, entity_type, device)
6. Keep queries and candidates realistic and natural

Output format (JSON array):
[
  {{
    "query": "workout playlist",
    "candidate": "Pure Workout Apple Music playlist",
    "label": 2,
    "rationale": "High quality playlists relevant to the activity (workout). User clearly looking for workout music.",
    "metadata": {{"locale": "en-US", "intent": "navigational", "entity_type": "playlist", "device": "mobile"}}
  }},
  ...
]

Generate EXACTLY {batch_count} examples. Follow the style of the examples above STRICTLY."""

        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE_GENERATION,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            parsed = json.loads(result)
            
            # Handle different response formats
            if isinstance(parsed, list):
                examples_data = parsed
            elif 'examples' in parsed:
                examples_data = parsed['examples']
            else:
                examples_data = list(parsed.values())[0] if parsed else []
            
            # Convert to GoldenExample objects
            for ex_data in examples_data:
                if 'query' in ex_data and 'candidate' in ex_data and 'label' in ex_data:
                    example = GoldenExample(
                        query=ex_data['query'],
                        candidate=ex_data['candidate'],
                        label=int(ex_data['label']),
                        rationale=ex_data.get('rationale'),
                        metadata=ex_data.get('metadata', {})
                    )
                    generated.append(example)
            
            logger.info(f"  Generated {len(examples_data)} examples (similar to: {theme_example.query[:40]}...)")
            
        except Exception as e:
            logger.error(f"Error generating batch {batch_num + 1}: {e}")
            continue
    
    logger.info(f"Total generated variations: {len(generated)}")
    return generated


def introduce_errors(
    golden: GoldenExample,
    error_type: str = "random"
) -> Tuple[int, str]:
    """
    Introduce realistic annotation errors.
    
    Args:
        golden: Original golden example
        error_type: Type of error to introduce
        
    Returns:
        Tuple of (incorrect_label, error_reason)
    """
    correct_label = golden.label
    
    # Define realistic error patterns
    if error_type == "overestimate":
        # Annotator too generous
        if correct_label < 2:
            return correct_label + 1, "overestimated_relevance"
        else:
            return correct_label, "already_max"
    
    elif error_type == "underestimate":
        # Annotator too strict
        if correct_label > 0:
            return correct_label - 1, "underestimated_relevance"
        else:
            return correct_label, "already_min"
    
    elif error_type == "confusion":
        # Common confusions (e.g., 0 vs 1, 1 vs 2)
        if correct_label == 1:
            return random.choice([0, 2]), "confused_boundary"
        elif correct_label == 0:
            return 1, "confused_irrelevant"
        else:  # correct_label == 2
            return 1, "confused_highly_relevant"
    
    else:  # random
        # Random incorrect label
        possible_labels = [0, 1, 2]
        possible_labels.remove(correct_label)
        return random.choice(possible_labels), "random_error"


def generate_annotations(
    goldens: List[GoldenExample],
    total_annotations: int,
    golden_error_rate: float,
    non_golden_error_rate: float
) -> List[Dict]:
    """
    Generate annotations including goldens (with errors) and additional non-golden annotations.
    
    Args:
        goldens: List of golden examples
        total_annotations: Total number of annotations to generate
        golden_error_rate: Proportion of goldens that should have errors
        non_golden_error_rate: Proportion of non-golden annotations that should have errors
        
    Returns:
        List of annotation dictionaries
    """
    logger.info(f"Generating {total_annotations} annotations...")
    logger.info(f"  Golden error rate: {golden_error_rate:.1%}")
    logger.info(f"  Non-golden error rate: {non_golden_error_rate:.1%}")
    
    annotations = []
    
    # 1. Add goldens (some with errors)
    logger.info(f"Adding {len(goldens)} goldens to annotations...")
    
    num_golden_errors = int(len(goldens) * golden_error_rate)
    golden_indices_with_errors = set(random.sample(range(len(goldens)), num_golden_errors))
    
    for i, golden in enumerate(goldens):
        if i in golden_indices_with_errors:
            # Introduce error
            error_types = ["overestimate", "underestimate", "confusion", "random"]
            error_type = random.choice(error_types)
            incorrect_label, error_reason = introduce_errors(golden, error_type)
            
            annotation = {
                "query": golden.query,
                "candidate": golden.candidate,
                "label": incorrect_label,
                "annotator": f"human_{random.randint(1, 10):03d}",
                "metadata": golden.metadata,
                "_is_golden_with_error": True,
                "_error_type": error_reason,
                "_correct_label": golden.label
            }
        else:
            # Correct annotation
            annotation = {
                "query": golden.query,
                "candidate": golden.candidate,
                "label": golden.label,
                "annotator": f"human_{random.randint(1, 10):03d}",
                "metadata": golden.metadata
            }
        
        annotations.append(annotation)
    
    # 2. Generate additional non-golden annotations using RAG
    num_additional = total_annotations - len(goldens)
    
    if num_additional > 0:
        logger.info(f"Generating {num_additional} additional non-golden annotations using RAG...")
        
        # Generate in batches
        batch_size = 20
        batches = (num_additional + batch_size - 1) // batch_size
        
        for batch_num in range(batches):
            batch_count = min(batch_size, num_additional - (len(annotations) - len(goldens)))
            
            logger.info(f"  Batch {batch_num + 1}/{batches} ({batch_count} annotations)...")
            
            # Select theme and find similar examples using RAG
            theme_example = random.choice(goldens)
            theme_query = f"Similar to: {theme_example.query}"
            similar_goldens = find_similar_examples(theme_query, goldens, top_k=6)
            
            system_prompt = """You are a human annotator evaluating search results for a music system.

IMPORTANT:
- Follow the style and logic shown in the examples from the guidelines
- Create realistic annotations that a human would make
- Rationales should be 2-3 sentences
- Be consistent with the annotation standards shown

Output ONLY valid JSON."""

            examples_text = "\n".join([
                f"Example: Query: \"{ex.query}\" → Candidate: \"{ex.candidate}\" → Label: {ex.label}"
                for ex in similar_goldens[:6]
            ])
            
            user_prompt = f"""Generate {batch_count} NEW realistic annotations following these examples:

{examples_text}

Create diverse query-candidate pairs with appropriate labels (0, 1, or 2).
Follow the same annotation logic and style as the examples above.

Output format (JSON array):
[
  {{
    "query": "...",
    "candidate": "...",
    "label": 2,
    "metadata": {{"locale": "en-US", "device": "mobile", "intent": "navigational", "entity_type": "album"}}
  }},
  ...
]

Generate exactly {batch_count} annotations. Stay consistent with the examples."""

            try:
                response = client.chat.completions.create(
                    model=GENERATION_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=TEMPERATURE_GENERATION,
                    response_format={"type": "json_object"}
                )
                
                result = response.choices[0].message.content
                parsed = json.loads(result)
                
                if isinstance(parsed, list):
                    new_annotations = parsed
                elif 'annotations' in parsed:
                    new_annotations = parsed['annotations']
                else:
                    new_annotations = list(parsed.values())[0] if parsed else []
                
                # Introduce errors to some annotations
                for ann in new_annotations:
                    ann['annotator'] = f"human_{random.randint(1, 10):03d}"
                    
                    # Randomly introduce errors
                    if random.random() < non_golden_error_rate:
                        # This annotation will have an error
                        original_label = ann['label']
                        possible_labels = [0, 1, 2]
                        possible_labels.remove(original_label)
                        ann['label'] = random.choice(possible_labels)
                        ann['_has_error'] = True
                        ann['_correct_label_unknown'] = True
                
                annotations.extend(new_annotations)
                
            except Exception as e:
                logger.error(f"Error generating annotation batch: {e}")
                continue
    
    logger.info(f"Total annotations generated: {len(annotations)}")
    
    # Count errors
    golden_errors = sum(1 for ann in annotations if ann.get('_is_golden_with_error'))
    non_golden_errors = sum(1 for ann in annotations if ann.get('_has_error'))
    
    logger.info(f"  Golden annotations with errors: {golden_errors}/{len(goldens)} ({golden_errors/len(goldens):.1%})")
    logger.info(f"  Non-golden annotations with errors: ~{non_golden_errors}")
    
    return annotations


def estimate_token_cost(
    num_goldens: int,
    num_annotations: int,
    goldens_from_pdf: int
) -> Dict[str, int]:
    """
    Estimate token costs for generation.
    
    Args:
        num_goldens: Total goldens desired
        num_annotations: Total annotations desired
        goldens_from_pdf: Number of goldens extracted from PDF
        
    Returns:
        Dict with token estimates
    """
    # Extraction from PDF (done once)
    # Large PDF processed in chunks
    extraction_input = 200000  # PDF text across multiple chunks
    extraction_output = goldens_from_pdf * 50  # ~50 tokens per extracted example
    extraction_tokens = extraction_input + extraction_output
    
    # Embeddings for extracted examples (for RAG)
    # Each example: ~100 tokens for embedding generation
    embedding_tokens = goldens_from_pdf * 100
    
    # Golden variations to generate
    goldens_to_generate = max(0, num_goldens - goldens_from_pdf)
    
    # Tokens per golden generation (batches of 10)
    # Input: ~2000 tokens per batch (context + examples)
    # Output: ~50 tokens per golden (short format: query, candidate, label, brief rationale)
    batches_golden = (goldens_to_generate + 9) // 10
    golden_input = batches_golden * 2000
    golden_output = goldens_to_generate * 50
    golden_generation_tokens = golden_input + golden_output
    
    # Non-golden annotations to generate
    non_golden_annotations = num_annotations - num_goldens
    
    # Tokens per annotation generation (batches of 20)
    # Input: ~500 tokens per batch (lighter context)
    # Output: ~50 tokens per annotation
    batches_annotation = (non_golden_annotations + 19) // 20
    annotation_input = batches_annotation * 500
    annotation_output = non_golden_annotations * 50
    annotation_generation_tokens = annotation_input + annotation_output
    
    # Calculate totals
    total_input = extraction_input + embedding_tokens + golden_input + annotation_input
    total_output = extraction_output + golden_output + annotation_output
    total_tokens = total_input + total_output
    
    # Estimate cost (gpt-4o-mini: $0.150 per 1M input, $0.600 per 1M output)
    # Embeddings use text-embedding-3-small: $0.020 per 1M tokens (much cheaper!)
    cost_input_llm = (extraction_input + golden_input + annotation_input) / 1_000_000 * 0.150
    cost_embeddings = (embedding_tokens / 1_000_000) * 0.020
    cost_output = (total_output / 1_000_000) * 0.600
    total_cost = cost_input_llm + cost_embeddings + cost_output
    
    return {
        "extraction_tokens": extraction_tokens,
        "embedding_tokens": embedding_tokens,
        "golden_generation_tokens": golden_generation_tokens,
        "annotation_generation_tokens": annotation_generation_tokens,
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 3)
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic annotation dataset from guidelines"
    )
    parser.add_argument(
        '--pdf',
        type=str,
        default='Guidelines_SR-Music V4.pdf',
        help='Path to guidelines PDF (default: Guidelines_SR-Music V4.pdf)'
    )
    parser.add_argument(
        '--goldens',
        type=int,
        default=300,
        help='Total number of golden annotations to generate (default: 300)'
    )
    parser.add_argument(
        '--annotations',
        type=int,
        default=2000,
        help='Total number of annotations to generate (default: 2000)'
    )
    parser.add_argument(
        '--golden-error-rate',
        type=float,
        default=0.30,
        help='Proportion of goldens with errors in annotations (default: 0.30)'
    )
    parser.add_argument(
        '--annotation-error-rate',
        type=float,
        default=0.15,
        help='Proportion of non-golden annotations with errors (default: 0.15)'
    )
    parser.add_argument(
        '--estimate-only',
        action='store_true',
        help='Only estimate token cost, do not generate'
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY") and not args.estimate_only:
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.error("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    logger.info("=" * 70)
    logger.info("SYNTHETIC DATASET GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  PDF: {args.pdf}")
    logger.info(f"  Target goldens: {args.goldens}")
    logger.info(f"  Total annotations: {args.annotations}")
    logger.info(f"  Golden error rate: {args.golden_error_rate:.1%}")
    logger.info(f"  Non-golden error rate: {args.annotation_error_rate:.1%}")
    
    # Estimate token cost
    logger.info("\n" + "=" * 70)
    logger.info("TOKEN COST ESTIMATION")
    logger.info("=" * 70)
    
    # Assume we'll extract ~100 examples from PDF
    estimated_pdf_examples = 100
    
    costs = estimate_token_cost(
        args.goldens,
        args.annotations,
        estimated_pdf_examples
    )
    
    logger.info(f"Extraction from PDF: ~{costs['extraction_tokens']:,} tokens")
    logger.info(f"Embeddings (for RAG): ~{costs['embedding_tokens']:,} tokens")
    logger.info(f"Golden generation: ~{costs['golden_generation_tokens']:,} tokens")
    logger.info(f"Annotation generation: ~{costs['annotation_generation_tokens']:,} tokens")
    logger.info(f"")
    logger.info(f"Total INPUT tokens:  ~{costs['total_input']:,}")
    logger.info(f"Total OUTPUT tokens: ~{costs['total_output']:,}")
    logger.info(f"TOTAL ESTIMATED: ~{costs['total_tokens']:,} tokens")
    logger.info(f"")
    logger.info(f"ESTIMATED COST: ${costs['estimated_cost_usd']:.3f} USD")
    logger.info(f"  (LLM I/O + Embeddings for RAG-based generation)")
    
    if args.estimate_only:
        logger.info("\n(Estimation only mode - exiting)")
        return
    
    # Extract examples from PDF
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Extract Examples from PDF")
    logger.info("=" * 70)
    
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    guidelines_text = extract_text_from_pdf(str(pdf_path))
    seed_examples = extract_examples_from_guidelines_llm(guidelines_text)
    
    if not seed_examples:
        logger.error("No examples extracted from PDF. Cannot proceed.")
        return
    
    logger.info(f"✓ Extracted {len(seed_examples)} examples from guidelines")
    
    # Generate embeddings for all extracted examples (CRITICAL FOR RAG)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1.5: Generate Embeddings for RAG")
    logger.info("=" * 70)
    
    seed_examples = embed_examples(seed_examples)
    logger.info(f"✓ All {len(seed_examples)} examples now have embeddings for semantic search")
    
    # Generate additional goldens if needed
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Generate Golden Variations")
    logger.info("=" * 70)
    
    if len(seed_examples) < args.goldens:
        guidelines_sample = guidelines_text[:5000]  # Sample for context
        additional_goldens = generate_golden_variations(
            seed_examples,
            args.goldens,
            guidelines_sample
        )
        all_goldens = seed_examples + additional_goldens
    else:
        # We have enough, just use first N
        all_goldens = seed_examples[:args.goldens]
    
    logger.info(f"✓ Total goldens: {len(all_goldens)}")
    
    # Generate annotations
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Generate Annotations")
    logger.info("=" * 70)
    
    annotations = generate_annotations(
        all_goldens,
        args.annotations,
        args.golden_error_rate,
        args.annotation_error_rate
    )
    
    # Save outputs
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Save Outputs")
    logger.info("=" * 70)
    
    goldens_path = Path("generated_goldens.jsonl")
    goldens_with_embeddings_path = Path("generated_goldens_with_embeddings.jsonl")
    annotations_path = Path("generated_annotations.jsonl")
    
    # Save goldens (without embeddings for cleaner format)
    with open(goldens_path, 'w', encoding='utf-8') as f:
        for golden in all_goldens:
            f.write(json.dumps(golden.to_dict(include_embedding=False), ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Saved {len(all_goldens)} goldens → {goldens_path}")
    
    # Save goldens WITH embeddings (for potential reuse/caching)
    with open(goldens_with_embeddings_path, 'w', encoding='utf-8') as f:
        for golden in all_goldens:
            f.write(json.dumps(golden.to_dict(include_embedding=True), ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Saved goldens with embeddings (for caching) → {goldens_with_embeddings_path}")
    
    # Save annotations (clean up internal fields)
    with open(annotations_path, 'w', encoding='utf-8') as f:
        for ann in annotations:
            # Remove internal fields
            clean_ann = {k: v for k, v in ann.items() if not k.startswith('_')}
            f.write(json.dumps(clean_ann, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Saved {len(annotations)} annotations → {annotations_path}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ DATASET GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nOutput files:")
    logger.info(f"  - {goldens_path.name} ({len(all_goldens)} goldens)")
    logger.info(f"  - {goldens_with_embeddings_path.name} (with embeddings for caching)")
    logger.info(f"  - {annotations_path.name} ({len(annotations)} annotations)")
    logger.info(f"\nQuality notes:")
    logger.info(f"  ✓ RAG-based generation using {TOP_K_SIMILAR_EXAMPLES} similar examples per batch")
    logger.info(f"  ✓ Generated cases follow style/logic of examples from guidelines PDF")
    logger.info(f"  ✓ Semantic search ensures consistency with original examples")
    logger.info(f"\nYou can now use these as inputs to the Edge-Case Annotator Agent:")
    logger.info(f"  cp {goldens_path} goldens.jsonl")
    logger.info(f"  cp {annotations_path} annotations.jsonl")


if __name__ == "__main__":
    main()

