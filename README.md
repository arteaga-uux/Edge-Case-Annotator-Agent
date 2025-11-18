# Edge-Case Annotator Agent

An intelligent system that identifies systematic human annotation errors, generates synthetic edge cases, and produces high-quality evaluation datasets through multi-LLM adversarial debate.

## Project Status

✅ **Complete** - All 7 scripts implemented with production-ready code quality

**Code Quality:**
- ✅ Pydantic models for all data structures
- ✅ Centralized configuration (`config.yaml`)
- ✅ Type hints everywhere
- ✅ Proper logging (replaces print statements)
- ✅ Comprehensive docstrings
- ✅ Fail-fast error handling

## Quick Start

### Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY='your-key-here'
```

3. **Configure parameters** (optional):
   Edit `config.yaml` to adjust:
   - LLM models and temperatures
   - RAG top_k parameters
   - Chunking sizes (3 types)
   - Pattern discovery thresholds
   - QA sampling rates

4. **Prepare input files** (examples provided):
   - `guidelines.md` - Annotation guidelines
   - `goldens.jsonl` - Golden annotations
   - `annotations.jsonl` - Human annotations to evaluate

### Complete Workflow

**Option 1: Run all phases at once (recommended)**

```bash
python run_all_phases.py
```

**Option 2: Run phases individually**

```bash
# Phase 1: Data Preparation
python prepare_data.py
python golden_crosscheck.py

# Phase 2: Build Embedding Indexes
python build_indexes.py

# Phase 3: Pattern Discovery
python discover_patterns.py

# Phase 4: Generate Synthetic Cases
python generate_cases.py

# Phase 5: Annotate with Debate
python annotate_with_debate.py

# Phase 6: Build Final Datasets
python build_final_sets.py
```

**Advanced usage:**

```bash
# Run only a specific phase
python run_all_phases.py --only-phase 3

# Skip a specific phase
python run_all_phases.py --skip-phase 2

# Resume from a specific phase
python run_all_phases.py --start-from 4
```

### Optional: Human QA

After Phase 5, review synthetic annotations and create `human_qa_results.jsonl`:

```json
{"case_id": "syn_00001", "human_review": "accept"}
{"case_id": "syn_00002", "human_review": "reject"}
```

Then re-run Phase 6 to incorporate human feedback.

## System Overview

### Architecture

```
RAW INPUTS → PHASE 1: Tag + Crosscheck
           ↓
PHASE 2: Build Embedding Indexes
           ↓
PHASE 3: Error Mining + Pattern Discovery
           ↓
PHASE 4: Budget Allocation
           ↓
PHASE 5: Generate Synthetic Cases (LLM #1)
           ↓
PHASE 6: 3-LLM Adversarial Annotation
           ↓
PHASE 7: Human QA Sampling
           ↓
FINAL OUTPUTS: HUMAN_SET, SYN_SET, HYBRID_SET
```

### Phase 1: Data Preparation

**Script 1: `prepare_data.py`**
- Parses `guidelines.md` into structured sections
- Adds tags to annotations: `intent`, `entity_type`, `locale`, `device`
- Uses rule-based heuristics from metadata and content

**Script 2: `golden_crosscheck.py`**
- Joins annotations with goldens by `(query, candidate)`
- Adds fields: `is_golden`, `golden_label`, `is_correct`, `is_error`
- Calculates human accuracy statistics

### Phase 2: Embedding Indexes

**Script 3: `build_indexes.py`**
- Builds GUIDELINE_INDEX: guidelines split into 300-600 token chunks with embeddings
- Builds EXAMPLE_INDEX: good examples (is_golden==true AND is_correct==true) with embeddings
- Builds ALL_ANNOTATIONS_INDEX: all annotations with embeddings for pattern discovery
- Uses OpenAI `text-embedding-3-small`

### Phase 3: Pattern Discovery

**Script 4: `discover_patterns.py`**
- Filters errors (is_error==true)
- Two-level clustering:
  - Level 1: Groups by exact tags
  - Level 2: DBSCAN clustering by embedding similarity
- Generates edge-case profiles with metrics
- Allocates synthetic budget: `score = quality_gap × log(1 + volume)`

### Phase 4: Synthetic Case Generation

**Script 5: `generate_cases.py`**
- For each pattern with budget > 0:
  - Builds RAG context (guidelines + examples + pattern profile)
  - Calls LLM #1 (gpt-4o-mini) to generate synthetic queries
  - Stores new cases with pattern_id

### Phase 5: Three-LLM Adversarial Annotation

**Script 6: `annotate_with_debate.py`**
- For each synthetic case:
  - Builds shared RAG context (Stage 1: original data + Stage 2: resolved synthetics)
  - **LLM #2 (Optimistic Annotator)**: Annotates favorably with guideline citations
  - **LLM #3 (Strict Critic)**: Critiques annotation, suggests alternatives
  - **LLM #4 (Neutral Judge)**: Weighs arguments, makes final decision
- Routes based on decision:
  - `accept_original`/`accept_alternative` → `synthetic_annotations.jsonl`
  - `ambiguous` → `ambiguous_cases.jsonl` (needs human review)
- Updates `resolved_synthetics_index.jsonl` for future iterations

### Phase 6: Final Dataset Construction

**Script 7: `build_final_sets.py`**
- Applies optional human QA results
- Builds three datasets:
  - **HUMAN_SET**: Correct human annotations only
  - **SYNTHETIC_SET**: All accepted/vetted synthetic cases
  - **HYBRID_SET**: HUMAN_SET + vetted synthetics (recommended)
- Computes quality metrics and coverage statistics

## Configuration

All tunable parameters are centralized in `config.yaml`:

```yaml
llm:
  embedding_model: "text-embedding-3-small"
  generation_model: "gpt-4o-mini"
  generation_temperature: 0.8  # Higher for diverse cases

rag:
  generation:
    top_k_guidelines: 3
    top_k_examples: 5
  annotation:
    top_k_guidelines: 5
    top_k_examples: 5
    top_k_resolved: 3  # Stage 2 RAG

chunking:
  guidelines:
    min_tokens: 300
    max_tokens: 600
  annotations:
    max_tokens: 500
  resolved_synthetics:
    max_tokens: 400
    max_rationale_tokens: 200

pattern_discovery:
  total_synthetic_budget: 100
  target_quality: 0.95
  min_pattern_volume: 3
  clustering:
    eps: 0.3
    min_samples: 2

qa:
  judge_accepted_sample_rate: 0.30
  ambiguous_sample_rate: 1.0

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

See `config.yaml` for complete configuration options.

## Project Structure

```
Edge-Case Annotator Agent/
├── README.md
├── requirements.txt
├── config.yaml                        # ⭐ Central configuration
├── config.py                          # Configuration loader (Pydantic)
├── models.py                          # ⭐ Data models (Pydantic)
├── utils.py                           # Utility functions + logging setup
├── run_all_phases.py                  # Master script (runs all phases)
│
# Phase 1: Data Preparation
├── prepare_data.py                    # Script 1
├── golden_crosscheck.py               # Script 2
│
# Phase 2: Embedding Indexes
├── build_indexes.py                   # Script 3
│
# Phase 3: Pattern Discovery
├── discover_patterns.py               # Script 4
│
# Phase 4: Case Generation
├── generate_cases.py                  # Script 5
│
# Phase 5: Adversarial Annotation
├── annotate_with_debate.py            # Script 6
│
# Phase 6: Final Datasets
├── build_final_sets.py                # Script 7
│
# Input files (examples provided)
├── guidelines.md
├── goldens.jsonl
├── annotations.jsonl
├── human_qa_results.jsonl.example
│
# Output files (generated by scripts)
├── prepared_guidelines.json
├── tagged_annotations.jsonl
├── tagged_goldens.jsonl
├── crosschecked_annotations.jsonl
├── guideline_index.jsonl
├── example_index.jsonl
├── all_annotations_index.jsonl
├── edge_case_profiles.jsonl
├── new_cases.jsonl
├── synthetic_annotations.jsonl
├── ambiguous_cases.jsonl
├── resolved_synthetics_index.jsonl
├── human_set.jsonl                    # Final dataset 1
├── synthetic_set.jsonl                # Final dataset 2
├── hybrid_set.jsonl                   # Final dataset 3 (recommended)
├── dataset_metrics.json
└── logs/                               # Logging directory
    └── edge_case_annotator.log        # Execution logs
```

## Key Features

- **Automated Error Pattern Discovery**: Uses two-level clustering (tags + embeddings) to find systematic annotation errors
- **RAG-Enhanced Generation**: Two-stage retrieval (original data + resolved synthetics) for context-aware generation
- **Three-LLM Adversarial System**: Optimistic annotator, strict critic, and neutral judge debate each annotation
- **Quality Assurance**: Configurable human review with support for selective sampling
- **Comprehensive Metrics**: Track coverage, quality, and distribution across all datasets

## License

MIT

