"""
Data models for Edge-Case Annotator Agent using Pydantic.
All structured data uses these models for validation and type safety.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Core Data Models
# =============================================================================

class Tags(BaseModel):
    """Tags for categorizing annotations."""
    intent: str = Field(default="unknown", description="User intent: navigational, informational, transactional")
    entity_type: str = Field(default="unknown", description="Type of entity: album, artist, song, etc.")
    locale: str = Field(default="en-US", description="Locale code: en-US, es-ES, etc.")
    device: str = Field(default="desktop", description="Device type: mobile, desktop, tablet")
    domain: str = Field(default="music_search", description="Domain of the query")


class Metadata(BaseModel):
    """Metadata for annotations."""
    locale: str = Field(default="en-US")
    device: str = Field(default="desktop")
    entity_type: Optional[str] = None
    domain: Optional[str] = None


class Annotation(BaseModel):
    """Human or synthetic annotation of a query-candidate pair."""
    query: str = Field(..., min_length=1, description="User query")
    candidate: str = Field(..., min_length=1, description="Candidate result")
    label: int = Field(..., ge=0, le=2, description="Relevance label: 0, 1, or 2")
    tags: Optional[Tags] = None
    metadata: Optional[Metadata] = None
    annotator: Optional[str] = None
    
    # Crosscheck fields (added in Phase 1)
    is_golden: Optional[bool] = None
    golden_label: Optional[int] = None
    is_correct: Optional[bool] = None
    is_error: Optional[bool] = None


class GuidelineSection(BaseModel):
    """A section from the annotation guidelines."""
    section_id: str = Field(..., description="Section identifier (e.g., '4.2')")
    title: str = Field(..., description="Section title")
    level: int = Field(..., ge=1, le=6, description="Header level (1-6)")
    text: str = Field(..., description="Section content")


class Guidelines(BaseModel):
    """Complete structured guidelines."""
    sections: List[GuidelineSection]


# =============================================================================
# Index Models (Phase 2)
# =============================================================================

class GuidelineChunk(BaseModel):
    """A chunk from guidelines with embedding."""
    chunk_id: str
    section_id: str
    title: str
    text: str
    tokens: int
    embedding: List[float]


class ExampleIndex(BaseModel):
    """A good example in the index."""
    example_id: str
    query: str
    candidate: str
    label: int = Field(ge=0, le=2)
    tags: Tags
    metadata: Optional[Dict] = None
    embedding: List[float]


class AnnotationIndex(BaseModel):
    """An annotation in the all_annotations index."""
    annotation_id: str
    query: str
    candidate: str
    label: int = Field(ge=0, le=2)
    tags: Tags
    metadata: Optional[Dict] = None
    is_golden: bool
    is_correct: Optional[bool] = None
    is_error: bool
    embedding: List[float]


# =============================================================================
# Pattern Discovery Models (Phase 3)
# =============================================================================

class PatternMetrics(BaseModel):
    """Metrics for an edge-case pattern."""
    volume: int = Field(..., ge=0, description="Number of errors in this pattern")
    total_with_tags: int = Field(..., ge=0, description="Total annotations with these tags")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate for this pattern")
    human_acc: float = Field(..., ge=0.0, le=1.0, description="Human accuracy for this pattern")
    quality_gap: float = Field(..., ge=0.0, description="Gap between target and human accuracy")


class SeedError(BaseModel):
    """An error example in a pattern."""
    query: str
    candidate: str
    label: int
    golden_label: int


class SeedGood(BaseModel):
    """A good example in a pattern."""
    query: str
    candidate: str
    label: int


class EdgeCaseProfile(BaseModel):
    """Profile of an edge-case pattern."""
    pattern_id: str
    description: str
    tags: Tags
    metrics: PatternMetrics
    seed_errors: List[SeedError] = Field(default_factory=list)
    seed_good: List[SeedGood] = Field(default_factory=list)
    synthetic_budget: int = Field(default=0, ge=0)


# =============================================================================
# Synthetic Case Generation Models (Phase 4)
# =============================================================================

class SyntheticCase(BaseModel):
    """A generated synthetic case."""
    case_id: str
    query: str
    candidate: str
    metadata: Metadata
    pattern_id: str


# =============================================================================
# Three-LLM Debate Models (Phase 5)
# =============================================================================

class OptimisticAnnotation(BaseModel):
    """Annotation from LLM #2 (Optimistic Annotator)."""
    label: int = Field(..., ge=0, le=2)
    rationale: str
    guideline_citations: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class StrictCritique(BaseModel):
    """Critique from LLM #3 (Strict Critic)."""
    agrees: bool
    alternative_label: Optional[int] = Field(None, ge=0, le=2)
    counter_rationale: str
    guideline_citations: List[str] = Field(default_factory=list)
    strength: Literal["weak", "moderate", "strong"]


class NeutralJudgment(BaseModel):
    """Judgment from LLM #4 (Neutral Judge)."""
    decision: Literal["accept_original", "accept_alternative", "ambiguous"]
    final_label: int = Field(..., ge=0, le=2)
    final_rationale: str
    reasoning: str


class AnnotatedCase(BaseModel):
    """A synthetic case with full debate history."""
    case_id: str
    query: str
    candidate: str
    metadata: Metadata
    pattern_id: str
    
    # Debate outputs
    annotation: OptimisticAnnotation
    critique: StrictCritique
    judgment: NeutralJudgment
    
    # Final results
    final_label: int = Field(..., ge=0, le=2)
    final_rationale: str
    decision: Literal["accept_original", "accept_alternative", "ambiguous"]
    qa_status: Literal["judge_accepted", "needs_human_review", "synthetic_vetted"]


class ResolvedSynthetic(BaseModel):
    """A successfully resolved synthetic case for RAG Stage 2."""
    case_id: str
    query: str
    candidate: str
    final_label: int = Field(ge=0, le=2)
    final_rationale: str
    pattern_id: str
    qa_status: Literal["judge_accepted", "synthetic_vetted"]
    embedding: List[float]


# =============================================================================
# Human QA Models (Phase 6)
# =============================================================================

class HumanQAResult(BaseModel):
    """Human QA review result for a synthetic case."""
    case_id: str
    human_review: Literal["accept", "reject"]
    human_label: Optional[int] = Field(None, ge=0, le=2)
    human_notes: Optional[str] = None


# =============================================================================
# Final Dataset Models (Phase 6)
# =============================================================================

class FinalAnnotation(BaseModel):
    """An annotation in the final datasets."""
    query: str
    candidate: str
    label: int = Field(ge=0, le=2)
    tags: Tags
    metadata: Optional[Dict] = None
    source: Literal["human", "synthetic"]
    
    # Optional fields for synthetic cases
    case_id: Optional[str] = None
    pattern_id: Optional[str] = None
    qa_status: Optional[str] = None
    final_rationale: Optional[str] = None
    is_golden: Optional[bool] = None


class DatasetMetrics(BaseModel):
    """Metrics for a final dataset."""
    total_cases: int
    quality: Optional[float] = None
    label_distribution: Dict[str, int]
    tag_distribution: Dict[str, int]
    
    # Specific to different sets
    judge_accepted: Optional[int] = None
    human_vetted: Optional[int] = None
    human_cases: Optional[int] = None
    synthetic_cases: Optional[int] = None
    estimated_quality: Optional[str] = None


class AllMetrics(BaseModel):
    """Complete metrics for all datasets."""
    human_set: DatasetMetrics
    synthetic_set: DatasetMetrics
    hybrid_set: DatasetMetrics
    baseline: Dict[str, float]

