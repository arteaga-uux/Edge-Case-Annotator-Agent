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
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pdfplumber
from openai import OpenAI

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
TEMPERATURE_GENERATION = 0.8
TEMPERATURE_ERROR = 0.7


class GoldenExample:
    """Represents a golden annotation example."""
    def __init__(
        self,
        query: str,
        candidate: str,
        label: int,
        rationale: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.query = query
        self.candidate = candidate
        self.label = label
        self.rationale = rationale
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "candidate": self.candidate,
            "label": self.label,
            "rationale": self.rationale,
            "metadata": self.metadata
        }


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
    Generate additional goldens as variations of seed examples.
    
    Args:
        seed_examples: Original examples from guidelines
        target_count: Total number of goldens desired
        guidelines_sample: Sample of guidelines for context
        
    Returns:
        List of generated GoldenExample objects
    """
    needed = target_count - len(seed_examples)
    
    if needed <= 0:
        logger.info(f"Already have {len(seed_examples)} examples, no variations needed")
        return []
    
    logger.info(f"Generating {needed} golden variations...")
    
    generated = []
    
    # Generate in batches of 10
    batch_size = 10
    batches = (needed + batch_size - 1) // batch_size
    
    for batch_num in range(batches):
        # Sample random seed examples for this batch
        samples = random.sample(seed_examples, min(5, len(seed_examples)))
        
        batch_count = min(batch_size, needed - len(generated))
        
        logger.info(f"Generating batch {batch_num + 1}/{batches} ({batch_count} examples)...")
        
        system_prompt = """You are an expert at creating realistic search annotation examples.

Generate NEW query-candidate pairs that are SIMILAR to the examples provided but NOT identical.
Maintain the same style, domain, and annotation quality.

Output ONLY valid JSON."""

        # Format seed examples
        examples_text = "\n\n".join([
            f"Example {i+1}:\n"
            f"Query: \"{ex.query}\"\n"
            f"Candidate: \"{ex.candidate}\"\n"
            f"Label: {ex.label}\n"
            f"Rationale: {ex.rationale or 'N/A'}"
            for i, ex in enumerate(samples)
        ])
        
        user_prompt = f"""Generate {batch_count} NEW annotation examples similar to these:

{examples_text}

Guidelines context:
{guidelines_sample[:2000]}

---

Requirements:
1. Create REALISTIC queries users would search
2. Create PLAUSIBLE candidates that could be returned
3. Assign CORRECT labels based on relevance
4. Provide brief rationales
5. Vary locales, intents, entity types

Output format (JSON array):
[
  {{
    "query": "...",
    "candidate": "...",
    "label": 2,
    "rationale": "...",
    "metadata": {{"locale": "en-US", "intent": "navigational", "entity_type": "album"}}
  }},
  ...
]

Generate exactly {batch_count} examples."""

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
            
            logger.info(f"  Generated {len(examples_data)} examples")
            
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
    
    # 2. Generate additional non-golden annotations
    num_additional = total_annotations - len(goldens)
    
    if num_additional > 0:
        logger.info(f"Generating {num_additional} additional non-golden annotations...")
        
        # Generate in batches
        batch_size = 20
        batches = (num_additional + batch_size - 1) // batch_size
        
        for batch_num in range(batches):
            batch_count = min(batch_size, num_additional - (len(annotations) - len(goldens)))
            
            logger.info(f"  Batch {batch_num + 1}/{batches} ({batch_count} annotations)...")
            
            # Sample golden examples as seeds
            samples = random.sample(goldens, min(5, len(goldens)))
            
            system_prompt = """You are a human annotator evaluating search results.

Generate NEW query-candidate pairs with annotations.
Be realistic - sometimes annotations might have subtle errors."""

            examples_text = "\n".join([
                f"Query: \"{ex.query}\" → Candidate: \"{ex.candidate}\" → Label: {ex.label}"
                for ex in samples[:3]
            ])
            
            user_prompt = f"""Generate {batch_count} NEW annotations for a music search system.

Example style:
{examples_text}

Create realistic query-candidate pairs with labels (0, 1, or 2).

Output format (JSON array):
[
  {{
    "query": "...",
    "candidate": "...",
    "label": 2,
    "metadata": {{"locale": "en-US", "device": "mobile"}}
  }},
  ...
]

Generate exactly {batch_count} annotations."""

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
    extraction_tokens = 200000  # Large PDF, multiple chunks
    
    # Golden variations to generate
    goldens_to_generate = max(0, num_goldens - goldens_from_pdf)
    
    # Tokens per golden generation
    # Input: ~2000 tokens (context + examples)
    # Output: ~200 tokens per golden
    tokens_per_golden = 2200
    
    # Non-golden annotations to generate
    non_golden_annotations = num_annotations - num_goldens
    
    # Tokens per annotation generation  
    # Input: ~1500 tokens (lighter context)
    # Output: ~150 tokens per annotation
    tokens_per_annotation = 1650
    
    # Calculate totals
    golden_generation_tokens = goldens_to_generate * tokens_per_golden
    annotation_generation_tokens = non_golden_annotations * tokens_per_annotation
    
    total_tokens = extraction_tokens + golden_generation_tokens + annotation_generation_tokens
    
    # Estimate cost (gpt-4o-mini: $0.150 per 1M input, $0.600 per 1M output)
    # Rough estimate: ~60% input, ~40% output
    input_tokens = int(total_tokens * 0.6)
    output_tokens = int(total_tokens * 0.4)
    
    cost_input = (input_tokens / 1_000_000) * 0.150
    cost_output = (output_tokens / 1_000_000) * 0.600
    total_cost = cost_input + cost_output
    
    return {
        "extraction_tokens": extraction_tokens,
        "golden_generation_tokens": golden_generation_tokens,
        "annotation_generation_tokens": annotation_generation_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 2)
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
    logger.info(f"Golden generation: ~{costs['golden_generation_tokens']:,} tokens")
    logger.info(f"Annotation generation: ~{costs['annotation_generation_tokens']:,} tokens")
    logger.info(f"TOTAL ESTIMATED: ~{costs['total_tokens']:,} tokens")
    logger.info(f"ESTIMATED COST: ${costs['estimated_cost_usd']:.2f} USD (gpt-4o-mini)")
    
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
    annotations_path = Path("generated_annotations.jsonl")
    
    # Save goldens
    with open(goldens_path, 'w', encoding='utf-8') as f:
        for golden in all_goldens:
            f.write(json.dumps(golden.to_dict(), ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Saved {len(all_goldens)} goldens → {goldens_path}")
    
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
    logger.info(f"  - {annotations_path.name} ({len(annotations)} annotations)")
    logger.info(f"\nYou can now use these as inputs to the Edge-Case Annotator Agent:")
    logger.info(f"  cp {goldens_path} goldens.jsonl")
    logger.info(f"  cp {annotations_path} annotations.jsonl")


if __name__ == "__main__":
    main()

