"""
Utility functions for Edge-Case Annotator Agent.
"""

import json
import logging
from pathlib import Path
from typing import List, TypeVar, Generic
from pydantic import BaseModel, ValidationError

from config import get_config, Config


T = TypeVar('T', bound=BaseModel)


def setup_logging(logger_name: str, config: Config) -> logging.Logger:
    """
    Setup logger with configuration from config.yaml.
    
    Args:
        logger_name: Name for the logger (usually __name__)
        config: Configuration object
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, config.logging.level))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(config.logging.format)
    
    # Console handler
    if config.logging.console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.logging.level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.logging.file:
        log_dir = Path(config.paths.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / config.paths.log_file,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.logging.level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_jsonl(file_path: str, model_class: type[T]) -> List[T]:
    """
    Load JSONL file and validate each record using Pydantic model.
    
    Args:
        file_path: Path to JSONL file
        model_class: Pydantic model class for validation
        
    Returns:
        List of validated model instances
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If any record fails validation
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    record = model_class(**data)
                    records.append(record)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
                except ValidationError as e:
                    raise ValueError(f"Validation error at line {line_num}: {e}")
    
    return records


def load_jsonl_raw(file_path: str) -> List[dict]:
    """
    Load JSONL file without validation (returns raw dicts).
    Use this when you need flexible loading or partial data.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    if not Path(file_path).exists():
        return []
    
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    return records


def save_jsonl(records: List[BaseModel], file_path: str) -> None:
    """
    Save list of Pydantic models to JSONL file.
    
    Args:
        records: List of Pydantic model instances
        file_path: Path to output file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            # Use model_dump() for Pydantic v2
            json_str = json.dumps(record.model_dump(), ensure_ascii=False)
            f.write(json_str + '\n')


def save_jsonl_raw(records: List[dict], file_path: str) -> None:
    """
    Save list of dictionaries to JSONL file.
    
    Args:
        records: List of dictionaries
        file_path: Path to output file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            json_str = json.dumps(record, ensure_ascii=False)
            f.write(json_str + '\n')


def validate_input_files(file_paths: List[str], logger: logging.Logger) -> bool:
    """
    Validate that all required input files exist.
    
    Args:
        file_paths: List of file paths to check
        logger: Logger instance
        
    Returns:
        True if all files exist, False otherwise
    """
    all_exist = True
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Missing input file: {file_path}")
            all_exist = False
        else:
            logger.debug(f"Found input file: {file_path}")
    
    return all_exist


def get_file_path(path_key: str, config: Config, path_type: str = "intermediate") -> Path:
    """
    Get absolute file path from config.
    
    Args:
        path_key: Key in the paths config (e.g., "tagged_annotations")
        config: Configuration object
        path_type: Type of path: "input", "intermediate", or "output"
        
    Returns:
        Absolute Path object
    """
    base_path = Path.cwd()
    
    if path_type == "input":
        relative_path = getattr(config.paths.input, path_key)
    elif path_type == "intermediate":
        relative_path = getattr(config.paths.intermediate, path_key)
    elif path_type == "output":
        relative_path = getattr(config.paths.output, path_key)
    else:
        raise ValueError(f"Invalid path_type: {path_type}")
    
    return base_path / relative_path

