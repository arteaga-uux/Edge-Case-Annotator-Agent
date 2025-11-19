#!/usr/bin/env python3
"""
Script 1: Data Preparation (Refactored)
Parses guidelines and adds tags to annotations and goldens.

Input:
  - guidelines.md
  - goldens.jsonl
  - annotations.jsonl

Output:
  - prepared_guidelines.json
  - tagged_annotations.jsonl
  - tagged_goldens.jsonl
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict

from edge_case_annotator.config import get_config
from edge_case_annotator.models import Guidelines, GuidelineSection, Annotation, Tags, Metadata
from edge_case_annotator.utils import setup_logging, save_jsonl_raw, get_file_path, validate_input_files


# Setup logger
config = get_config()
logger = setup_logging(__name__, config)


def parse_guidelines(guidelines_path: Path) -> Guidelines:
    """
    Parse guidelines.md into structured format with sections.
    
    Args:
        guidelines_path: Path to guidelines.md file
        
    Returns:
        Guidelines object with parsed sections
        
    Raises:
        FileNotFoundError: If guidelines file doesn't exist
    """
    logger.info(f"Parsing guidelines from {guidelines_path}")
    
    if not guidelines_path.exists():
        raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")
    
    with open(guidelines_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections: List[GuidelineSection] = []
    lines = content.split('\n')
    
    current_section: Dict = {}
    current_text: List[str] = []
    section_counter = 1
    subsection_counter = 1
    
    for line in lines:
        # Check if it's a header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            # Save previous section if exists
            if current_section:
                current_section['text'] = '\n'.join(current_text).strip()
                sections.append(GuidelineSection(**current_section))
                current_text = []
            
            # Create new section
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            
            if level == 1:
                section_id = f"{section_counter}.0"
                section_counter += 1
                subsection_counter = 1
            else:
                section_id = f"{section_counter - 1}.{subsection_counter}"
                subsection_counter += 1
            
            current_section = {
                "section_id": section_id,
                "title": title,
                "level": level,
                "text": ""
            }
            logger.debug(f"Found section: {section_id} - {title}")
        else:
            # Accumulate text for current section
            if line.strip():
                current_text.append(line)
    
    # Don't forget the last section
    if current_section:
        current_section['text'] = '\n'.join(current_text).strip()
        sections.append(GuidelineSection(**current_section))
    
    logger.info(f"Parsed {len(sections)} sections from guidelines")
    return Guidelines(sections=sections)


def extract_tags(record: dict) -> Tags:
    """
    Extract tags from annotation/golden record using metadata and heuristics.
    
    Args:
        record: Raw annotation record (dict)
        
    Returns:
        Tags object with extracted fields
    """
    metadata = record.get('metadata', {})
    query = record.get('query', '').lower()
    candidate = record.get('candidate', '').lower()
    
    # Extract locale and device from metadata
    locale = metadata.get('locale', 'en-US')
    device = metadata.get('device', 'desktop')
    
    # Heuristic intent detection using config
    intent_signals = config.tagging.intent_signals
    intent = 'informational'  # default
    
    for signal_type, keywords in [
        ('navigational', intent_signals.navigational),
        ('transactional', intent_signals.transactional)
    ]:
        if any(keyword in query for keyword in keywords):
            intent = signal_type
            break
    
    # Heuristic entity_type detection using config
    entity_signals = config.tagging.entity_signals
    entity_type = 'general'  # default
    
    for entity, keywords in [
        ('album', entity_signals.album),
        ('artist', entity_signals.artist),
        ('song', entity_signals.song),
        ('playlist', entity_signals.playlist),
        ('video', entity_signals.video),
        ('concert', entity_signals.concert)
    ]:
        if any(keyword in candidate or keyword in query for keyword in keywords):
            entity_type = entity
            break
    
    # Override with metadata if present
    entity_type = metadata.get('entity_type', entity_type)
    domain = metadata.get('domain', 'music_search')
    
    logger.debug(f"Extracted tags for query '{record.get('query', '')[:30]}...': "
                f"intent={intent}, entity_type={entity_type}")
    
    return Tags(
        intent=intent,
        entity_type=entity_type,
        locale=locale,
        device=device,
        domain=domain
    )


def tag_records(input_path: Path, output_path: Path) -> int:
    """
    Read JSONL file, add tags to each record, write to output.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        
    Returns:
        Number of records processed
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    logger.info(f"Tagging records from {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    tagged_records = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                    
                    # Extract and add tags
                    tags = extract_tags(record)
                    record['tags'] = tags.model_dump()
                    
                    tagged_records.append(record)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at line {line_num}: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error processing record at line {line_num}: {e}")
                    raise
    
    # Write tagged records
    save_jsonl_raw(tagged_records, str(output_path))
    
    logger.info(f"Tagged {len(tagged_records)} records → {output_path.name}")
    return len(tagged_records)


def main() -> None:
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Preparation")
    logger.info("=" * 60)
    
    # Get file paths from config
    guidelines_path = get_file_path('guidelines', config, 'input')
    goldens_path = get_file_path('goldens', config, 'input')
    annotations_path = get_file_path('annotations', config, 'input')
    
    prepared_guidelines_path = get_file_path('prepared_guidelines', config, 'intermediate')
    tagged_goldens_path = get_file_path('tagged_goldens', config, 'intermediate')
    tagged_annotations_path = get_file_path('tagged_annotations', config, 'intermediate')
    
    # Validate input files
    if not validate_input_files(
        [str(guidelines_path), str(goldens_path), str(annotations_path)],
        logger
    ):
        logger.error("Missing required input files. Exiting.")
        return
    
    try:
        # Parse guidelines
        logger.info("[1/3] Parsing guidelines...")
        guidelines = parse_guidelines(guidelines_path)
        
        with open(prepared_guidelines_path, 'w', encoding='utf-8') as f:
            json.dump(guidelines.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Parsed {len(guidelines.sections)} sections → {prepared_guidelines_path.name}")
        
        # Tag goldens
        logger.info("[2/3] Tagging goldens...")
        goldens_count = tag_records(goldens_path, tagged_goldens_path)
        
        # Tag annotations
        logger.info("[3/3] Tagging annotations...")
        annotations_count = tag_records(annotations_path, tagged_annotations_path)
        
        # Summary
        logger.info("=" * 60)
        logger.info("✓ Data preparation complete!")
        logger.info("=" * 60)
        logger.info("Output files:")
        logger.info(f"  - {prepared_guidelines_path.name} ({len(guidelines.sections)} sections)")
        logger.info(f"  - {tagged_goldens_path.name} ({goldens_count} records)")
        logger.info(f"  - {tagged_annotations_path.name} ({annotations_count} records)")
        
    except Exception as e:
        logger.error(f"Fatal error during data preparation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

