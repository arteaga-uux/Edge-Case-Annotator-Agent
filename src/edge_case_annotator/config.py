"""
Configuration loader for Edge-Case Annotator Agent.
Loads and validates config.yaml using Pydantic models.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """LLM configuration."""
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    annotation_model: str = "gpt-4o-mini"
    generation_temperature: float = Field(0.8, ge=0.0, le=2.0)
    annotation_temperature: float = Field(0.3, ge=0.0, le=2.0)
    critique_temperature: float = Field(0.3, ge=0.0, le=2.0)
    judge_temperature: float = Field(0.2, ge=0.0, le=2.0)


class RAGGenerationConfig(BaseModel):
    """RAG configuration for generation phase."""
    top_k_guidelines: int = Field(3, ge=1)
    top_k_examples: int = Field(5, ge=1)


class RAGAnnotationConfig(BaseModel):
    """RAG configuration for annotation phase."""
    top_k_guidelines: int = Field(5, ge=1)
    top_k_examples: int = Field(5, ge=1)
    top_k_resolved: int = Field(3, ge=1)


class RAGConfig(BaseModel):
    """RAG configuration."""
    generation: RAGGenerationConfig
    annotation: RAGAnnotationConfig


class GuidelinesChunkingConfig(BaseModel):
    """Guidelines chunking configuration."""
    min_tokens: int = Field(300, ge=50)
    max_tokens: int = Field(600, ge=100)
    split_by: str = "sentence"
    
    @field_validator('split_by')
    @classmethod
    def validate_split_by(cls, v: str) -> str:
        if v not in ['sentence', 'paragraph']:
            raise ValueError('split_by must be "sentence" or "paragraph"')
        return v


class AnnotationsChunkingConfig(BaseModel):
    """Annotations chunking configuration."""
    max_tokens: int = Field(500, ge=100)
    include_metadata: bool = True
    truncate_if_exceeds: bool = True


class ResolvedSyntheticsChunkingConfig(BaseModel):
    """Resolved synthetics chunking configuration."""
    max_tokens: int = Field(400, ge=100)
    include_rationale: bool = True
    max_rationale_tokens: int = Field(200, ge=50)


class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    guidelines: GuidelinesChunkingConfig
    annotations: AnnotationsChunkingConfig
    resolved_synthetics: ResolvedSyntheticsChunkingConfig


class ClusteringConfig(BaseModel):
    """DBSCAN clustering configuration."""
    eps: float = Field(0.3, ge=0.0, le=1.0)
    min_samples: int = Field(2, ge=1)


class BudgetAllocationConfig(BaseModel):
    """Budget allocation formula configuration."""
    quality_gap_power: float = Field(1.0, ge=0.1)
    volume_log_base: float = Field(1.0, ge=0.0)


class PatternDiscoveryConfig(BaseModel):
    """Pattern discovery configuration."""
    total_synthetic_budget: int = Field(100, ge=1)
    target_quality: float = Field(0.95, ge=0.5, le=1.0)
    min_pattern_volume: int = Field(3, ge=1)
    clustering: ClusteringConfig
    budget_allocation: BudgetAllocationConfig


class QAConfig(BaseModel):
    """Quality assurance configuration."""
    judge_accepted_sample_rate: float = Field(0.30, ge=0.0, le=1.0)
    ambiguous_sample_rate: float = Field(1.0, ge=0.0, le=1.0)


class InputPathsConfig(BaseModel):
    """Input file paths."""
    guidelines: str = "guidelines.md"
    goldens: str = "goldens.jsonl"
    annotations: str = "annotations.jsonl"
    human_qa_results: str = "human_qa_results.jsonl"


class IntermediatePathsConfig(BaseModel):
    """Intermediate file paths."""
    prepared_guidelines: str = "prepared_guidelines.json"
    tagged_annotations: str = "tagged_annotations.jsonl"
    tagged_goldens: str = "tagged_goldens.jsonl"
    crosschecked_annotations: str = "crosschecked_annotations.jsonl"
    guideline_index: str = "guideline_index.jsonl"
    example_index: str = "example_index.jsonl"
    all_annotations_index: str = "all_annotations_index.jsonl"
    edge_case_profiles: str = "edge_case_profiles.jsonl"
    new_cases: str = "new_cases.jsonl"
    synthetic_annotations: str = "synthetic_annotations.jsonl"
    ambiguous_cases: str = "ambiguous_cases.jsonl"
    resolved_synthetics_index: str = "resolved_synthetics_index.jsonl"


class OutputPathsConfig(BaseModel):
    """Output file paths."""
    human_set: str = "human_set.jsonl"
    synthetic_set: str = "synthetic_set.jsonl"
    hybrid_set: str = "hybrid_set.jsonl"
    dataset_metrics: str = "dataset_metrics.json"


class PathsConfig(BaseModel):
    """All file paths configuration."""
    input: InputPathsConfig
    intermediate: IntermediatePathsConfig
    output: OutputPathsConfig
    log_dir: str = "logs"
    log_file: str = "edge_case_annotator.log"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console: bool = True
    file: bool = True
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'level must be one of {valid_levels}')
        return v.upper()


class IntentSignalsConfig(BaseModel):
    """Intent detection signals."""
    navigational: List[str] = ["download", "go to", "open", "show me", "find", "navigate"]
    transactional: List[str] = ["buy", "purchase", "order", "subscribe", "download"]


class EntitySignalsConfig(BaseModel):
    """Entity type detection signals."""
    album: List[str] = ["album", "lp", "ep", "disc"]
    artist: List[str] = ["artist", "band", "singer", "musician"]
    song: List[str] = ["song", "track", "single"]
    playlist: List[str] = ["playlist", "mix", "compilation"]
    video: List[str] = ["video", "mv", "music video"]
    concert: List[str] = ["concert", "live", "tour", "show", "en vivo"]


class TaggingConfig(BaseModel):
    """Tagging heuristics configuration."""
    intent_signals: IntentSignalsConfig
    entity_signals: EntitySignalsConfig


class Config(BaseModel):
    """Complete application configuration."""
    llm: LLMConfig
    rag: RAGConfig
    chunking: ChunkingConfig
    pattern_discovery: PatternDiscoveryConfig
    qa: QAConfig
    paths: PathsConfig
    logging: LoggingConfig
    tagging: TaggingConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate and return
    return Config(**config_dict)


# Singleton config instance
_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get configuration singleton.
    Loads config on first call, returns cached instance on subsequent calls.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Config singleton instance
    """
    global _config
    
    if _config is None:
        _config = load_config(config_path)
    
    return _config


def reload_config(config_path: str = "config.yaml") -> Config:
    """
    Force reload of configuration from file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Freshly loaded Config instance
    """
    global _config
    _config = load_config(config_path)
    return _config


if __name__ == "__main__":
    # Test config loading
    try:
        config = load_config()
        print("✓ Configuration loaded successfully!")
        print(f"\nLLM Model: {config.llm.generation_model}")
        print(f"Total Synthetic Budget: {config.pattern_discovery.total_synthetic_budget}")
        print(f"Target Quality: {config.pattern_discovery.target_quality}")
        print(f"Logging Level: {config.logging.level}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")

