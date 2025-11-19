# Edge-Case Annotator Agent - Architecture

## 🎯 Overview

System that identifies systematic human annotation errors, generates synthetic edge cases, and produces high-quality evaluation datasets through multi-LLM adversarial debate.

## 🔄 Complete Pipeline Flow

```mermaid
graph TD
    A[Raw Inputs] -->|guidelines.md<br/>goldens.jsonl<br/>annotations.jsonl| B[Phase 1: Data Prep]
    B -->|tagged data| C[Phase 2: Build Indexes]
    C -->|3 embedding indexes| D[Phase 3: Pattern Discovery]
    D -->|edge_case_profiles| E[Phase 4: Generate Cases]
    E -->|new_cases| F[Phase 5: Annotate with Debate]
    F -->|synthetic_annotations<br/>ambiguous_cases| G[Phase 6: Build Final Sets]
    G -->|HUMAN_SET<br/>SYNTHETIC_SET<br/>HYBRID_SET| H[Final Datasets]
    
    F -.->|accumulates| I[(resolved_synthetics_index)]
    I -.->|RAG Stage 2| F
    
    style H fill:#90EE90
    style I fill:#FFD700
```

---

## 📊 Phase 1: Data Preparation

### Script 1: `prepare_data.py`

```mermaid
graph LR
    A[guidelines.md] --> B[parse_guidelines]
    B --> C[prepared_guidelines.json]
    
    D[goldens.jsonl] --> E[extract_tags]
    F[annotations.jsonl] --> E
    
    E --> G[tagged_goldens.jsonl]
    E --> H[tagged_annotations.jsonl]
    
    I[config.yaml] -.->|tagging heuristics| E
```

**What it does:**
- Parses guidelines into structured sections
- Adds tags to annotations: `intent`, `entity_type`, `locale`, `device`
- Uses rule-based heuristics from config

### Script 2: `golden_crosscheck.py`

```mermaid
graph TD
    A[tagged_annotations.jsonl] --> B[Build Golden Index]
    C[tagged_goldens.jsonl] --> B
    
    B --> D[Crosscheck Annotations]
    D --> E{Match Found?}
    
    E -->|Yes| F[Add: is_golden=true<br/>golden_label<br/>is_correct<br/>is_error]
    E -->|No| G[Add: is_golden=false]
    
    F --> H[crosschecked_annotations.jsonl]
    G --> H
    
    H --> I[Calculate Stats:<br/>Human Accuracy<br/>Error Count]
```

**Key Logic:**
- Joins annotations with goldens by `(query, candidate)` key
- Marks errors: `is_error = true` only when `is_golden AND !is_correct`

---

## 🔍 Phase 2: Build Embedding Indexes

### Script 3: `build_indexes.py`

```mermaid
graph TD
    A[prepared_guidelines.json] -->|chunk 300-600 tokens| B[GUIDELINE_INDEX]
    
    C[crosschecked_annotations.jsonl] -->|filter: is_golden<br/>AND is_correct| D[EXAMPLE_INDEX]
    
    C -->|embed all| E[ALL_ANNOTATIONS_INDEX]
    
    B --> F[text-embedding-3-small]
    D --> F
    E --> F
    
    F --> G[guideline_index.jsonl]
    F --> H[example_index.jsonl]
    F --> I[all_annotations_index.jsonl]
    
    style G fill:#B0E0E6
    style H fill:#B0E0E6
    style I fill:#B0E0E6
```

**Three Indexes Created:**

1. **GUIDELINE_INDEX**: Guidelines split into chunks with embeddings
2. **EXAMPLE_INDEX**: Good examples (correct annotations only)
3. **ALL_ANNOTATIONS_INDEX**: All annotations (for clustering)

**Chunking Types (from config.yaml):**
- Guidelines: 300-600 tokens
- Annotations: max 500 tokens
- Resolved synthetics: max 400 tokens

---

## 🔬 Phase 3: Pattern Discovery

### Script 4: `discover_patterns.py`

```mermaid
graph TD
    A[crosschecked_annotations] -->|filter| B[Errors Only<br/>is_error=true]
    C[all_annotations_index] --> B
    
    B --> D[Level 1: Group by<br/>Exact Tags]
    D --> E[Tag Group 1:<br/>intent=nav, entity=album<br/>locale=es-ES]
    D --> F[Tag Group 2:<br/>intent=info, entity=song<br/>locale=en-US]
    D --> G[Tag Group N]
    
    E --> H[Level 2: DBSCAN<br/>Clustering by Embeddings]
    F --> H
    G --> H
    
    H --> I[Pattern 1:<br/>Cluster A from Group 1]
    H --> J[Pattern 2:<br/>Cluster B from Group 1]
    H --> K[Pattern 3:<br/>Cluster A from Group 2]
    
    I --> L[Calculate Metrics:<br/>volume, error_rate<br/>human_acc, quality_gap]
    J --> L
    K --> L
    
    L --> M[Allocate Budget:<br/>score = quality_gap × log1+volume]
    
    M --> N[edge_case_profiles.jsonl]
    
    style N fill:#FFB6C1
```

**Two-Level Clustering:**
1. **Level 1**: Exact tag matching (fast grouping)
2. **Level 2**: Embedding similarity (find nuanced patterns)

**Budget Allocation Formula:**
```
score_i = quality_gap_i^power × log(base + volume_i)
budget_i = (total_budget × score_i) / Σ(score_j)
```

---

## 🎨 Phase 4: Synthetic Case Generation

### Script 5: `generate_cases.py`

```mermaid
graph TD
    A[edge_case_profiles.jsonl] --> B{For each pattern<br/>with budget > 0}
    
    B --> C[Get Pattern Embedding]
    
    C --> D[RAG Stage 1:<br/>Retrieve from Original Data]
    
    D --> E[guideline_index]
    D --> F[example_index]
    
    E -->|top_k=3| G[Relevant Guidelines]
    F -->|top_k=5| H[Similar Examples]
    
    G --> I[Build RAG Context]
    H --> I
    A --> I
    
    I --> J[LLM #1:<br/>gpt-4o-mini<br/>temp=0.8]
    
    J --> K[Generate N Cases<br/>matching pattern]
    
    K --> L[new_cases.jsonl]
    
    style L fill:#98FB98
```

**RAG Context for Generation:**
```
1. Pattern description + seed errors
2. Top 3 most relevant guideline chunks
3. Top 5 similar good examples
```

**LLM #1 Task:**
- Generate NEW queries fitting the error pattern
- Follow locale/intent/entity_type from pattern
- Create realistic query-candidate pairs

---

## ⚔️ Phase 5: Three-LLM Adversarial Debate

### Script 6: `annotate_with_debate.py` (Most Complex)

```mermaid
graph TD
    A[new_cases.jsonl] --> B{For each case}
    
    B --> C[Get Case Embedding]
    
    C --> D[RAG Stage 1:<br/>Original Data]
    C --> E[RAG Stage 2:<br/>Historical Synthetics]
    
    D --> F[guideline_index<br/>top_k=5]
    D --> G[example_index<br/>top_k=5]
    D --> H[pattern profile]
    
    E --> I[resolved_synthetics_index<br/>top_k=3]
    
    F --> J[Build Shared<br/>RAG Context]
    G --> J
    H --> J
    I --> J
    
    J --> K[LLM #2:<br/>Optimistic Annotator<br/>temp=0.3]
    
    K --> L[Label + Rationale<br/>+ Citations<br/>+ Confidence]
    
    J --> M[LLM #3:<br/>Strict Critic<br/>temp=0.3]
    L --> M
    
    M --> N[Agrees?<br/>Alternative Label?<br/>Counter-Rationale<br/>+ Strength]
    
    J --> O[LLM #4:<br/>Neutral Judge<br/>temp=0.2]
    L --> O
    N --> O
    
    O --> P{Decision}
    
    P -->|accept_original<br/>or accept_alternative| Q[qa_status=<br/>judge_accepted]
    P -->|ambiguous| R[qa_status=<br/>needs_human_review]
    
    Q --> S[synthetic_annotations.jsonl]
    R --> T[ambiguous_cases.jsonl]
    
    Q --> U[Update resolved_synthetics_index]
    U --> V[(resolved_synthetics_index.jsonl)]
    V -.->|RAG Stage 2<br/>next iteration| E
    
    style V fill:#FFD700
    style S fill:#90EE90
    style T fill:#FFA07A
```

### RAG Stage 2 - Historical Context

```mermaid
graph LR
    A[Run 1: Dataset A] -->|80 cases accepted| B[(resolved_synthetics<br/>80 cases)]
    
    B -.->|RAG Stage 2| C[Run 2: Dataset B]
    
    C -->|75 new cases| D[(resolved_synthetics<br/>155 cases)]
    
    D -.->|RAG Stage 2| E[Run 3: Dataset C]
    
    E -->|90 new cases| F[(resolved_synthetics<br/>245 cases)]
    
    style B fill:#FFD700
    style D fill:#FFD700
    style F fill:#FFD700
```

**Key: File persists and grows across runs!**

### Three-LLM Debate Detail

```mermaid
sequenceDiagram
    participant Case
    participant RAG
    participant LLM2 as LLM #2<br/>Optimistic
    participant LLM3 as LLM #3<br/>Strict Critic
    participant LLM4 as LLM #4<br/>Neutral Judge
    participant Output
    
    Case->>RAG: Get embedding
    RAG->>RAG: Stage 1: Original data
    RAG->>RAG: Stage 2: Historical synthetics
    
    RAG->>LLM2: Context + Case
    LLM2->>LLM2: Favor higher labels<br/>Cite guidelines
    LLM2->>LLM3: Annotation
    
    RAG->>LLM3: Context + Case + Annotation
    LLM3->>LLM3: Find problems<br/>Suggest lower label
    LLM3->>LLM4: Critique
    
    RAG->>LLM4: Context + Case + Both arguments
    LLM4->>LLM4: Weigh arguments<br/>Check citations
    
    alt Decision: accept_original
        LLM4->>Output: synthetic_annotations.jsonl
    else Decision: accept_alternative
        LLM4->>Output: synthetic_annotations.jsonl
    else Decision: ambiguous
        LLM4->>Output: ambiguous_cases.jsonl
    end
    
    Output->>RAG: Update resolved index
```

---

## 📦 Phase 6: Build Final Datasets

### Script 7: `build_final_sets.py`

```mermaid
graph TD
    A[crosschecked_annotations] -->|filter: is_golden<br/>AND is_correct| B[HUMAN_SET]
    
    C[synthetic_annotations] -->|filter: qa_status in<br/>judge_accepted, synthetic_vetted| D[SYNTHETIC_SET]
    
    E[human_qa_results.jsonl] -->|optional| F[Apply Human QA]
    
    F --> G{Review Result}
    G -->|accept| H[Update: qa_status=<br/>synthetic_vetted]
    G -->|reject| I[Remove from dataset]
    
    H --> C
    
    B --> J[HYBRID_SET]
    C -->|only qa_status=<br/>synthetic_vetted| J
    
    J --> K[Compute Metrics:<br/>Coverage, Quality<br/>Label Distribution<br/>Tag Distribution]
    
    B --> L[human_set.jsonl]
    D --> M[synthetic_set.jsonl]
    J --> N[hybrid_set.jsonl]
    K --> O[dataset_metrics.json]
    
    style L fill:#90EE90
    style M fill:#87CEEB
    style N fill:#FFD700
    style O fill:#DDA0DD
```

### QA Flow

```mermaid
graph TD
    A[synthetic_annotations] --> B{qa_status?}
    
    B -->|judge_accepted| C[30% Random Sample]
    B -->|needs_human_review| D[100% Review]
    
    C --> E[Human Reviewer]
    D --> E
    
    E --> F{Accept/Reject?}
    
    F -->|accept| G[qa_status=<br/>synthetic_vetted]
    F -->|reject| H[Remove]
    
    G --> I[Include in<br/>HYBRID_SET]
    H --> J[Exclude]
```

---

## 🎯 RAG System Architecture

### Two-Stage RAG

```mermaid
graph TD
    A[New Synthetic Case] --> B[Generate Embedding]
    
    B --> C[RAG Stage 1:<br/>Original Data]
    B --> D[RAG Stage 2:<br/>Resolved Synthetics]
    
    C --> E[Search guideline_index<br/>cosine similarity]
    C --> F[Search example_index<br/>cosine similarity]
    C --> G[Get pattern profile]
    
    D --> H[Search resolved_synthetics_index<br/>cosine similarity]
    
    E -->|top 5| I[Relevant Guidelines]
    F -->|top 5| J[Similar Good Examples]
    G --> K[Pattern Context]
    H -->|top 3| L[Previously Resolved Cases]
    
    I --> M[Combined RAG Context]
    J --> M
    K --> M
    L --> M
    
    M --> N[LLM Prompts]
    
    style C fill:#B0E0E6
    style D fill:#FFD700
```

**Why Two Stages?**
- **Stage 1**: Learn from original human-annotated data
- **Stage 2**: Maintain consistency with previously resolved synthetics

---

## 📈 Data Flow Summary

```mermaid
graph LR
    A[Raw Data] -->|Phase 1-2| B[Indexed Data]
    B -->|Phase 3| C[Error Patterns]
    C -->|Phase 4| D[Synthetic Cases]
    D -->|Phase 5| E[Annotated Cases]
    E -->|Phase 6| F[Final Datasets]
    
    E -.->|accumulates| G[(Historical<br/>Synthetics)]
    G -.->|informs| E
    
    style F fill:#90EE90
    style G fill:#FFD700
```

### File Sizes (typical)

| File | Phase | Size | Purpose |
|------|-------|------|---------|
| `guidelines.md` | Input | ~50 KB | Annotation rules |
| `goldens.jsonl` | Input | ~100 KB | Golden annotations |
| `annotations.jsonl` | Input | ~500 KB | Human annotations |
| `guideline_index.jsonl` | 2 | ~5 MB | Embedded guidelines |
| `example_index.jsonl` | 2 | ~2 MB | Good examples |
| `all_annotations_index.jsonl` | 2 | ~10 MB | All annotations |
| `edge_case_profiles.jsonl` | 3 | ~50 KB | Discovered patterns |
| `new_cases.jsonl` | 4 | ~200 KB | Generated cases |
| `resolved_synthetics_index.jsonl` | 5 | ~3 MB (grows) | Historical resolved |
| `synthetic_annotations.jsonl` | 5 | ~500 KB | Annotated synthetics |
| `hybrid_set.jsonl` | 6 | ~800 KB | **Final dataset** |

---

## 🔧 Configuration Overview

### Key Parameters (config.yaml)

```yaml
llm:
  generation_temperature: 0.8    # Higher for diversity
  annotation_temperature: 0.3    # Lower for consistency
  judge_temperature: 0.2         # Lowest for stability

rag:
  generation:
    top_k_guidelines: 3
    top_k_examples: 5
  annotation:
    top_k_guidelines: 5
    top_k_examples: 5
    top_k_resolved: 3             # Stage 2

chunking:
  guidelines: 300-600 tokens
  annotations: max 500 tokens
  resolved_synthetics: max 400 tokens

pattern_discovery:
  total_synthetic_budget: 100
  target_quality: 0.95
  clustering_eps: 0.3              # DBSCAN distance threshold

qa:
  judge_accepted_sample_rate: 0.30  # 30% human review
  ambiguous_sample_rate: 1.0        # 100% human review
```

---

## 🚀 Execution Flow

```mermaid
graph TD
    A[python run_all_phases.py] --> B[Load & Validate config.yaml]
    
    B --> C[Phase 1: Data Prep]
    C --> D[Phase 2: Build Indexes]
    D --> E[Phase 3: Pattern Discovery]
    E --> F[Phase 4: Generate Cases]
    F --> G[Phase 5: Annotate with Debate]
    G --> H[Phase 6: Build Final Sets]
    
    H --> I{Success?}
    I -->|Yes| J[✓ HYBRID_SET ready<br/>for evaluation]
    I -->|No| K[Check logs/<br/>edge_case_annotator.log]
    
    G -.->|updates| L[(resolved_synthetics_index)]
    L -.->|next run| G
```

### Command Line Options

```bash
# Run all phases
python run_all_phases.py

# Run specific phase only
python run_all_phases.py --only-phase 3

# Skip a phase
python run_all_phases.py --skip-phase 2

# Start from specific phase
python run_all_phases.py --start-from 4

# Use different config
python run_all_phases.py --config custom_config.yaml
```

---

## 💾 Persistent State

**Files that persist across runs:**

1. **`resolved_synthetics_index.jsonl`** ⭐ Most important
   - Accumulates successfully resolved cases
   - Used in RAG Stage 2
   - Grows with each run
   - Never reset (unless manual deletion)

2. **Configuration:**
   - `config.yaml` (manually edited)

3. **Logs:**
   - `logs/edge_case_annotator.log` (appends)

**Files that are regenerated each run:**
- All Phase 1-4 outputs
- `synthetic_annotations.jsonl` (Phase 5)
- Final datasets (Phase 6)

---

## 🎓 Key Design Decisions

### 1. Why Three LLMs for Annotation?
```
Single LLM → Biased toward one style
Two LLMs → Tie-breaking problem
Three LLMs → Adversarial debate with neutral judge
```

### 2. Why Two-Level Clustering?
```
Embeddings only → Misses exact tag patterns
Tags only → Misses semantic similarity
Both → Finds nuanced patterns within tag groups
```

### 3. Why RAG Stage 2?
```
Without: Each run independent, no learning
With: System learns from previously resolved cases
     Maintains consistency across multiple runs
```

### 4. Why Persistent Index?
```
Transient: Loses knowledge between runs
Persistent: Accumulates expertise over time
           Better RAG context for future cases
```

---

## 📊 Metrics & Quality Control

### Quality Gates

```mermaid
graph TD
    A[Synthetic Case] --> B{LLM #4 Decision}
    
    B -->|accept_original<br/>or accept_alternative| C{Confidence Check}
    B -->|ambiguous| D[100% Human Review]
    
    C -->|high confidence| E[30% Sample Review]
    C -->|low confidence| D
    
    E --> F{Human Accept?}
    D --> F
    
    F -->|Yes| G[qa_status=<br/>synthetic_vetted]
    F -->|No| H[Remove from dataset]
    
    G --> I[Include in<br/>HYBRID_SET]
```

### Final Dataset Quality

```
HUMAN_SET:
  - Quality: 100% (by definition)
  - Coverage: Limited to correct human annotations

SYNTHETIC_SET:
  - Quality: ~85-95% (judge accepted + sample vetted)
  - Coverage: Targeted edge cases

HYBRID_SET: ⭐ Recommended
  - Quality: ~95%+ (only vetted synthetics)
  - Coverage: Human baseline + targeted edge cases
  - Best for evaluation
```

---

## 🐛 Common Issues & Solutions

### Issue 1: No patterns discovered
**Cause:** Not enough errors or too few per tag group  
**Solution:** Adjust `min_pattern_volume` in config.yaml

### Issue 2: All cases marked ambiguous
**Cause:** LLM #4 can't decide due to unclear guidelines  
**Solution:** Review `guideline_index` quality, add more examples

### Issue 3: RAG Stage 2 not helping
**Cause:** `resolved_synthetics_index.jsonl` too small  
**Solution:** Run multiple iterations to build up the index

### Issue 4: Generated cases unrealistic
**Cause:** Temperature too high or poor RAG context  
**Solution:** Lower `generation_temperature` or improve `example_index`

---

## 🔍 Debugging Tips

### Check Intermediate Files

```bash
# See discovered patterns
head edge_case_profiles.jsonl

# See generated cases
head new_cases.jsonl

# See debate results
head synthetic_annotations.jsonl

# Check logs
tail -f logs/edge_case_annotator.log
```

### Validate Data Flow

```bash
# Count records at each stage
wc -l *.jsonl

# Check for empty files
find . -name "*.jsonl" -size 0
```

---

## 📚 Related Documentation

- `README.md` - Quick start guide
- `config.yaml` - All tunable parameters
- `models.py` - Pydantic data models
- `utils.py` - Shared utilities

---

**Last Updated:** November 2024  
**Version:** 1.0

