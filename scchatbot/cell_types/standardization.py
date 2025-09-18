"""
Cell type name standardization and normalization functions.

This module contains functions for standardizing cell type names to ensure
consistent naming conventions across the system. These functions were moved
from annotation.py and utils.py to create a centralized standardization module.

Functions:
- unified_cell_type_handler(): Primary standardization function (used everywhere)
- standardize_cell_type(): Base form normalization helper
- get_possible_cell_types(): Generate name variations for matching
"""

import re
import os
import json


def unified_cell_type_handler(cell_type):
    """
    Standardizes cell type names to match the new database format:
    - Singular form (ends with "cell" not "cells")
    - CD markers use "-positive" format
    - First word capitalized (except special abbreviations)
    - Handles multi-word cell types with proper capitalization
    """
    if not cell_type or not isinstance(cell_type, str):
        return "Unknown cell"
    clean_type = cell_type.strip()
    clean_type = re.sub(r'\bCD(\d+)\+', r'CD\1-positive', clean_type, flags=re.IGNORECASE)
    clean_type = re.sub(r'\bCD(\d+)-(?!positive|negative)', r'CD\1-negative', clean_type, flags=re.IGNORECASE)
    clean_type = re.sub(r'\bCD(\d+) positive\b', r'CD\1-positive', clean_type, flags=re.IGNORECASE)
    clean_type = re.sub(r'\bCD(\d+) negative\b', r'CD\1-negative', clean_type, flags=re.IGNORECASE)
    flag = 0
    if clean_type.lower().endswith(' cells'):
        base_type = clean_type[:-6].strip()
        flag = 1
    elif clean_type.lower().endswith(' cell'):
        base_type = clean_type[:-5].strip()
        flag = 2
    else:
        base_type = clean_type
        flag = 3
    words = base_type.split()
    if not words:
        return "Unknown cell"
    processed_words = []
    for i, word in enumerate(words):
        punct = ''
        m = re.match(r'^(.*?)([,.\.;:])$', word)
        if m:
            core_word, punct = m.groups()
        else:
            core_word = word
        word_to_process = core_word
        word_lower = word_to_process.lower()
        cd_match = re.match(r'^(cd\d+)(?:[+\-]|(-positive|-negative))$', word_to_process, flags=re.IGNORECASE)
        if cd_match:
            cd_part = cd_match.group(1).upper()
            suffix = ''
            suff_match = re.search(r'(-positive|-negative)', word_to_process, flags=re.IGNORECASE)
            if suff_match:
                suffix = suff_match.group(1).lower()
            processed_core = cd_part + suffix
            processed_words.append(processed_core + punct)
            continue
        elif word_lower in ['t', 'b', 'nk', 'th1', 'th2', 'th17', 'treg', 'dc']:
            processed_core = word_to_process.upper()
            processed_words.append(processed_core + punct)
        elif word_lower in ['alpha', 'beta', 'gamma', 'delta']:
            processed_core = word_lower
            processed_words.append(processed_core + punct)
        elif word_lower in ['and', 'or', 'of', 'in']:
            processed_core = word_lower
            processed_words.append(processed_core + punct)
        elif word_to_process in ['+', '-', '/', ',']:
            processed_core = word_to_process
            processed_words.append(processed_core + punct)
        else:
            if i == 0 or words[i-1].lower() in ['and', 'or']:
                processed_core = word_to_process.capitalize()
            else:
                processed_core = word_to_process.lower()
            processed_words.append(processed_core + punct)
    result = ' '.join(processed_words)
    special = [
        'platelet', 'platelets',
        'erythrocyte', 'erythrocytes',
        'lymphocyte', 'lymphocytes',
        'monocyte', 'monocytes',
        'neutrophil', 'neutrophils',
        'eosinophil', 'eosinophils',
        'basophil', 'basophils'
    ]
    if result.lower() in special:
        singular = result.lower().rstrip('s')
        return singular.capitalize()
    if flag == 1 or flag == 2:
        result += ' cell'
    return result


def standardize_cell_type(cell_type):
    """
    Standardize cell type strings for flexible matching with the new database.
    Converts to lowercase and handles CD markers consistently.
    """
    clean_type = cell_type.lower().strip()
    clean_type = re.sub(r'\bcd(\d+)\+', r'cd\1-positive', clean_type)
    clean_type = re.sub(r'\bcd(\d+)-(?!positive|negative)', r'cd\1-negative', clean_type)
    clean_type = re.sub(r'\bcd(\d+) positive\b', r'cd\1-positive', clean_type)
    clean_type = re.sub(r'\bcd(\d+) negative\b', r'cd\1-negative', clean_type)
    if clean_type.endswith(' cells'):
        clean_type = clean_type[:-6].strip()
    elif clean_type.endswith(' cell'):
        clean_type = clean_type[:-5].strip()
    return clean_type


def get_possible_cell_types(cell_type):
    """
    Generate all possible forms of a cell type for flexible matching.
    """
    base_form = standardize_cell_type(cell_type)
    result = unified_cell_type_handler(cell_type)
    words = base_form.split()
    possible_types = [base_form]
    possible_types.append(result)
    if not result.lower().endswith("cells"):
        possible_types.append(f"{result} cells")
    if len(words) == 1:
        if words[0].endswith('s'):
            possible_types.append(words[0][:-1])
        else:
            possible_types.append(f"{words[0]}s")
    return list(dict.fromkeys(possible_types))