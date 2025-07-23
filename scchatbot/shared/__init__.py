"""
Shared utilities module.

This module contains common utilities and functions
used across the workflow system.
"""

from .cell_type_utils import (
    extract_cell_types_from_question,
    needs_cell_discovery,
    create_cell_discovery_steps
)

from .result_extraction import (
    extract_key_findings_from_execution,
    extract_dea_key_findings,
    extract_process_cells_findings,
    extract_comparison_findings,
    format_findings_for_synthesis
)

__all__ = [
    'extract_cell_types_from_question',
    'needs_cell_discovery', 
    'create_cell_discovery_steps',
    'extract_key_findings_from_execution',
    'extract_dea_key_findings',
    'extract_process_cells_findings',
    'extract_comparison_findings',
    'format_findings_for_synthesis'
]