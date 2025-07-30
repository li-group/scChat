"""
Cell type name standardization and normalization functions.

This module contains functions for standardizing cell type names to ensure
consistent naming conventions across the system. These functions were moved
from annotation.py and utils.py to create a centralized standardization module.

Functions:
- unified_cell_type_handler(): Primary standardization function (used everywhere)
- standardize_cell_type(): Base form normalization helper
- get_possible_cell_types(): Generate name variations for matching
- get_subtypes(): Neo4j-based subtype lookup
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
    # Clean and normalize input
    clean_type = cell_type.strip()
    # Handle CD markers: convert all formats to -positive/-negative
    # Convert + to -positive
    clean_type = re.sub(r'\bCD(\d+)\+', r'CD\1-positive', clean_type, flags=re.IGNORECASE)
    # Convert - to -negative
    clean_type = re.sub(r'\bCD(\d+)-(?!positive|negative)', r'CD\1-negative', clean_type, flags=re.IGNORECASE)
    # Convert space-separated format
    clean_type = re.sub(r'\bCD(\d+) positive\b', r'CD\1-positive', clean_type, flags=re.IGNORECASE)
    clean_type = re.sub(r'\bCD(\d+) negative\b', r'CD\1-negative', clean_type, flags=re.IGNORECASE)
    # Remove trailing "cell" or "cells" for processing
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
    # Split into words for processing
    words = base_type.split()
    if not words:
        return "Unknown cell"
    # Process each word
    processed_words = []
    for i, word in enumerate(words):
        # Strip trailing punctuation for processing
        punct = ''
        m = re.match(r'^(.*?)([,.\.;:])$', word)
        if m:
            core_word, punct = m.groups()
        else:
            core_word = word
        word_to_process = core_word
        word_lower = word_to_process.lower()
        # Special handling for CD markers: uppercase CD and number, lowercase suffix
        cd_match = re.match(r'^(cd\d+)(?:[+\-]|(-positive|-negative))$', word_to_process, flags=re.IGNORECASE)
        if cd_match:
            cd_part = cd_match.group(1).upper()
            suffix = ''
            # handle explicit suffix (e.g., "-positive" or "-negative")
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
            # Capitalize first letter of regular words
            if i == 0 or words[i-1].lower() in ['and', 'or']:
                processed_core = word_to_process.capitalize()
            else:
                processed_core = word_to_process.lower()
            processed_words.append(processed_core + punct)
    # Join words and add "cell" at the end (singular)
    result = ' '.join(processed_words)
    # Special cases: immunocyte names should always be singular
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
        # singularize by stripping trailing 's'
        singular = result.lower().rstrip('s')
        return singular.capitalize()
    # Ensure it ends with "cell" (singular)
    if flag == 1 or flag == 2:
        result += ' cell'
    return result


def standardize_cell_type(cell_type):
    """
    Standardize cell type strings for flexible matching with the new database.
    Converts to lowercase and handles CD markers consistently.
    """
    clean_type = cell_type.lower().strip()
    # Normalize CD markers to -positive/-negative format
    clean_type = re.sub(r'\bcd(\d+)\+', r'cd\1-positive', clean_type)
    clean_type = re.sub(r'\bcd(\d+)-(?!positive|negative)', r'cd\1-negative', clean_type)
    clean_type = re.sub(r'\bcd(\d+) positive\b', r'cd\1-positive', clean_type)
    clean_type = re.sub(r'\bcd(\d+) negative\b', r'cd\1-negative', clean_type)
    # Remove "cells" or "cell" suffix for base form
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


def get_subtypes(cell_type):
    """Get subtypes of a given cell type using Neo4j."""
    from neo4j import GraphDatabase
    
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "37754262"))
    specification = None
    file_path = "media/specification_graph.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            specification = json.load(file)
    else:
        print("specification not found")
        return {}
    
    database = specification['database']
    organ = specification['organ']  # ✅ ADDED: Extract organ from specification
        
    subtypes_data = {}
    try:
        with driver.session(database=database) as session:
            query = """
            MATCH (parent:CellType {name: $parent_cell, organ: $organ})-[:DEVELOPS_TO {organ: $organ}]->(child:CellType)
            WHERE child.organ = $organ
            RETURN child.name as cell_name, child.markers as marker_list
            """
            # ✅ FIXED: Added missing organ parameter
            result = session.run(query, parent_cell=cell_type, organ=organ)
            
            for record in result:
                subtype_name = record["cell_name"]  # ✅ Now matches query alias
                marker_genes = record["marker_list"] or []  # ✅ Now matches query alias, handle null
                
                if subtype_name not in subtypes_data:
                    subtypes_data[subtype_name] = {"markers": []}
                
                # ✅ FIXED: Markers are already a list, don't iterate through them
                subtypes_data[subtype_name]["markers"] = marker_genes
                
    except Exception as e:
        print(f"Error accessing Neo4j: {e}")
        return {}
    finally:
        driver.close()
    return subtypes_data