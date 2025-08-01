"""
Cell type hierarchy management and extraction.

This module provides hierarchical cell type management, lineage tracking,
and intelligent cell type extraction from various sources.

Classes moved from cell_type_hierarchy.py:
- HierarchicalCellTypeManager: Neo4j-based hierarchy management
- CellTypeExtractor: Multi-strategy cell type extraction
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime

import pandas as pd

from ..cell_type_models import CellTypeRelation, CellTypeLineage


class HierarchicalCellTypeManager:
    """
    Manages hierarchical relationships between cell types using Neo4j database.
    
    Provides cell type validation, lineage tracking, processing path resolution,
    and intelligent aggregation of cell types based on their hierarchical relationships.
    """
    
    def __init__(self, adata, config_file="media/specification_graph.json"):
        self.adata = adata
        
        # Neo4j connection setup (absorbed from CellTypeChecker)
        self.driver = None
        self.valid_cell_types = []
        self.config = self._load_config(config_file)
        
        # Database configuration
        self.db_name = self.config.get("database")
        self.organ = self.config.get("organ")
        
        # Initialize Neo4j connection
        self._initialize_neo4j_connection()
        
        # Core hierarchical data structures
        self.cell_lineages: Dict[str, CellTypeLineage] = {}  # cell_id -> lineage
        self.type_hierarchy_cache: Dict[str, Dict] = {}  # cell_type -> hierarchy info
        self.ancestor_descendant_map: Dict[str, Set[str]] = {}  # type -> all descendants
        self.processing_snapshots: List[Dict] = []  # Snapshots after each process_cells
        self.path_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}  # Cache for paths
        
        # Initialize with current state
        self._initialize_lineages()
        self._build_hierarchy_cache()
        print(f"🧬 Initialized HierarchicalCellTypeManager with {len(self.cell_lineages)} cells and {len(self.valid_cell_types)} known cell types")
    
    def _load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                print(f"✅ Loaded configuration from {config_file}")
                return config
        except FileNotFoundError:
            print(f"⚠️ Configuration file {config_file} not found. Using default values.")
            return {}
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing configuration file {config_file}: {e}. Using default values.")
            return {}
        except Exception as e:
            print(f"⚠️ Error loading configuration: {e}. Using default values.")
            return {}
    
    def _initialize_neo4j_connection(self):
        """Initialize Neo4j connection and load valid cell types"""
        try:
            required_keys = ["url", "username", "password"]
            missing_keys = [key for key in required_keys if key not in self.config or not self.config[key]]
            
            if missing_keys:
                raise ValueError(f"Missing required configuration keys: {missing_keys}")
            
            uri = self.config["url"]
            username = self.config["username"]
            password = self.config["password"]
            
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.valid_cell_types = self._load_valid_cell_types()
            print(f"✅ Neo4j connection successful to '{self.db_name}'. Loaded {len(self.valid_cell_types)} cell types.")
        except Exception as e:
            print(f"⚠️ Could not connect to Neo4j: {e}. Proceeding without database features.")
            self.driver = None
            self.valid_cell_types = []
    
    def _load_valid_cell_types(self):
        """Load all valid cell types from Neo4j"""
        if not self.driver: 
            return []
        try:
            with self.driver.session(database=self.db_name) as session:
                cypher = """
                MATCH (o:Organ {name: $organ})-[:HAS_CELL]->(root:CellType)
                MATCH (root)-[:DEVELOPS_TO*0..]->(c:CellType)
                RETURN collect(DISTINCT c.name) AS cell_names
                """
                record = session.run(cypher, {"organ": self.organ}).single()
                return record["cell_names"] if record else []
        except Exception as e:
            print(f"⚠️ Error loading cell types from Neo4j: {e}")
            return []
    
    def _initialize_lineages(self):
        """Initialize lineages from current adata state"""
        for cell_id in self.adata.obs.index:
            current_type = self.adata.obs.loc[cell_id, "cell_type"]
            self.cell_lineages[cell_id] = CellTypeLineage(
                cell_id=cell_id,
                current_type=current_type,
                lineage_path=[current_type],
                processing_history=[{
                    "step": "initial",
                    "timestamp": pd.Timestamp.now(),
                    "method": "initial_annotation"
                }]
            )
    
    def _build_hierarchy_cache(self):
        """Build comprehensive hierarchy cache from Neo4j"""
        if not self.driver:
            print("⚠️ No Neo4j connection, hierarchy features limited")
            return
            
        try:
            with self.driver.session(database=self.db_name) as session:
                # Get full hierarchy for the organ
                query = """
                MATCH (o:Organ {name: $organ})-[:HAS_CELL]->(root:CellType)
                MATCH path = (root)-[:DEVELOPS_TO*0..]->(descendant:CellType)
                WITH root, descendant, [node in nodes(path) | node.name] as lineage_path
                RETURN root.name as root_type, 
                       collect(DISTINCT descendant.name) as all_descendants,
                       collect(DISTINCT lineage_path) as all_paths
                """
                
                result = session.run(query, organ=self.organ)
                
                for record in result:
                    root_type = record["root_type"]
                    descendants = set(record["all_descendants"])
                    
                    # Build ancestor-descendant mappings for each cell type
                    for desc in descendants:
                        if desc not in self.ancestor_descendant_map:
                            self.ancestor_descendant_map[desc] = set()
                        # Each cell type maps to all its descendants
                        descendant_query = """
                        MATCH path = (start:CellType {name: $cell_type})-[:DEVELOPS_TO*0..]->(desc:CellType)
                        RETURN collect(DISTINCT desc.name) as descendants
                        """
                        desc_result = session.run(descendant_query, cell_type=desc)
                        desc_record = desc_result.single()
                        if desc_record:
                            self.ancestor_descendant_map[desc] = set(desc_record["descendants"])
                
                # Cache hierarchy info for each type
                for cell_type in self.valid_cell_types:
                    if cell_type not in self.type_hierarchy_cache:
                        ancestor_path = self._get_ancestor_path(cell_type)
                        direct_children = self._get_direct_children(cell_type)
                        all_descendants = list(self.ancestor_descendant_map.get(cell_type, set()))
                        
                        self.type_hierarchy_cache[cell_type] = {
                            "ancestors": ancestor_path,
                            "all_descendants": all_descendants,
                            "direct_children": direct_children,
                            "level": len(ancestor_path)
                        }
                        
            print(f"✅ Built hierarchy cache for {len(self.type_hierarchy_cache)} cell types")
            
        except Exception as e:
            print(f"⚠️ Error building hierarchy cache: {e}")
    
    def is_valid_cell_type(self, cell_type: str) -> bool:
        """Check if a cell type exists in the ontology"""
        return cell_type in self.valid_cell_types
    
    def get_path_to_target(self, current_type: str, target_type: str) -> Optional[List[str]]:
        """Find the shortest path from current cell type to target cell type"""
        if not self.driver: 
            return None
        
        if self.is_valid_cell_type(current_type) and self.is_valid_cell_type(target_type):
            with self.driver.session(database=self.db_name) as session:
                query = """
                MATCH path = (start:CellType {name: $current})-[:DEVELOPS_TO*1..5]->(end:CellType {name: $target})
                RETURN [node in nodes(path) | node.name] as path_names, length(path) as len
                ORDER BY len ASC
                LIMIT 1
                """
                result = session.run(query, current=current_type, target=target_type)
                record = result.single()
                return record["path_names"] if record else None
        else:
            return None
    
    def find_parent_path(self, target_type: str, available_types: List[str]) -> Optional[Tuple[str, List[str]]]:
        """Find the best parent type and path to reach target type"""
        if not self.driver:
            return None
            
        best_path = None
        best_parent = None
        shortest_length = float('inf')
        
        for available_type in available_types:
            path = self.get_path_to_target(available_type, target_type)
            if path and len(path) < shortest_length:
                shortest_length = len(path)
                best_path = path
                best_parent = available_type
                
        return (best_parent, best_path) if best_path else None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver: 
            self.driver.close()
    
    def get_cell_type_relation(self, type1: str, type2: str) -> CellTypeRelation:
        """Determine relationship between two cell types"""
        if type1 == type2:
            return CellTypeRelation.SAME
        
        type1_info = self.type_hierarchy_cache.get(type1, {})
        type2_info = self.type_hierarchy_cache.get(type2, {})
        
        # Check if type1 is ancestor of type2
        if type2 in type1_info.get("all_descendants", []):
            return CellTypeRelation.ANCESTOR
        
        # Check if type2 is ancestor of type1  
        if type1 in type2_info.get("all_descendants", []):
            return CellTypeRelation.DESCENDANT
        
        # Check if they share a common ancestor (siblings)
        type1_ancestors = set(type1_info.get("ancestors", []))
        type2_ancestors = set(type2_info.get("ancestors", []))
        if type1_ancestors.intersection(type2_ancestors):
            return CellTypeRelation.SIBLING
            
        return CellTypeRelation.UNRELATED
    
    def resolve_cell_type_for_analysis(self, target_type: str) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
        current_types = set(self.adata.obs["cell_type"].unique())
        
        print(f"🔍 Resolving '{target_type}' from available types: {current_types}")
        
        # Case 1: Target type already exists in current data
        if target_type in current_types:
            print(f"✅ Direct match found for '{target_type}'")
            return self.adata.obs["cell_type"], {
                "resolution_method": "direct", 
                "target_type": target_type,
                "cell_count": (self.adata.obs["cell_type"] == target_type).sum()
            }
        
        # Case 2: Target is ancestor of current types (AGGREGATION)
        descendant_types = []
        for current_type in current_types:
            relation = self.get_cell_type_relation(target_type, current_type)
            if relation == CellTypeRelation.ANCESTOR:
                descendant_types.append(current_type)
                print(f"🔗 '{current_type}' is descendant of '{target_type}'")
        
        if descendant_types:
            # Create virtual cell type series that groups descendants under ancestor label
            virtual_series = self.adata.obs["cell_type"].copy()
            mask = virtual_series.isin(descendant_types)
            virtual_series.loc[mask] = target_type
            
            cell_count = mask.sum()
            print(f"🎯 Aggregated {len(descendant_types)} descendant types into '{target_type}' ({cell_count} cells)")
            
            return virtual_series, {
                "resolution_method": "ancestor_aggregation",
                "target_type": target_type,
                "aggregated_types": descendant_types,
                "cell_count": cell_count,
                "original_type_counts": self.adata.obs["cell_type"].value_counts().to_dict()
            }
        
        # Case 3: Target is descendant - need process_cells (PROCESSING REQUIRED)
        ancestor_types = []
        for current_type in current_types:
            relation = self.get_cell_type_relation(target_type, current_type)
            if relation == CellTypeRelation.DESCENDANT:
                ancestor_types.append(current_type)
                print(f"🔗 '{target_type}' is descendant of '{current_type}'")
        
        if ancestor_types:
            # Find the best processing path
            processing_paths = []
            for ancestor in ancestor_types:
                path = self._get_processing_path(ancestor, target_type)
                if path:
                    processing_paths.append((ancestor, path))
            
            if processing_paths:
                # Choose the shortest path
                best_ancestor, best_path = min(processing_paths, key=lambda x: len(x[1]))
                print(f"🛤️ Found processing path: {' → '.join(best_path)}")
                
                return None, {
                    "resolution_method": "needs_processing",
                    "target_type": target_type,
                    "processing_path": best_path,
                    "best_ancestor": best_ancestor,
                    "requires_process_cells": True
                }
        
        # Case 4: Not found or unrelated
        suggestions = self._suggest_similar_types(target_type, current_types)
        print(f"❌ '{target_type}' not found. Suggestions: {suggestions}")
        
        return None, {
            "resolution_method": "not_found",
            "target_type": target_type,
            "suggestions": suggestions,
            "available_types": list(current_types)
        }
    
    def update_after_process_cells(self, processed_type: str, new_types: List[str]):
        """Update lineages after process_cells operation"""
        print(f"🔄 Updating lineages after processing '{processed_type}' → {new_types}")
        
        # Create snapshot before modification
        snapshot = {
            "timestamp": pd.Timestamp.now(),
            "operation": "process_cells",
            "processed_type": processed_type,
            "cell_type_state": self.adata.obs["cell_type"].copy(),
            "discovered_types": new_types
        }
        self.processing_snapshots.append(snapshot)
        
        # Update cell lineages
        updated_count = 0
        for cell_id in self.adata.obs.index:
            current_type = self.adata.obs.loc[cell_id, "cell_type"]
            
            # If this cell was part of the processed type and now has a new type
            if cell_id in self.cell_lineages and current_type in new_types:
                lineage = self.cell_lineages[cell_id]
                
                # Check if this is a progression in the lineage
                if processed_type in lineage.lineage_path and current_type not in lineage.lineage_path:
                    lineage.lineage_path.append(current_type)
                    lineage.current_type = current_type
                    lineage.processing_history.append({
                        "step": "process_cells",
                        "timestamp": pd.Timestamp.now(),
                        "processed_from": processed_type,
                        "discovered": current_type
                    })
                    updated_count += 1
        
        print(f"✅ Updated {updated_count} cell lineages")
    
    def get_analysis_ready_adata(self, target_type: str):
        resolved_series, metadata = self.resolve_cell_type_for_analysis(target_type)
        
        if resolved_series is None:
            if metadata["resolution_method"] == "needs_processing":
                raise ValueError(
                    f"Cell type '{target_type}' requires process_cells operation. "
                    f"Processing path: {' → '.join(metadata['processing_path'])}"
                )
            else:
                available = ", ".join(metadata.get("available_types", []))
                suggestions = ", ".join(metadata.get("suggestions", []))
                raise ValueError(
                    f"Cell type '{target_type}' not found.\n"
                    f"Available types: {available}\n"
                    f"Suggestions: {suggestions}"
                )
        
        # Create temporary adata with resolved cell types
        temp_adata = self.adata.copy()
        temp_adata.obs["cell_type"] = resolved_series
        
        return temp_adata, metadata
    
    def _get_processing_path(self, from_type: str, to_type: str) -> Optional[List[str]]:
        """Get the processing path from ancestor to descendant with caching"""
        cache_key = (from_type, to_type)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        path = self.get_path_to_target(from_type, to_type)
        self.path_cache[cache_key] = path
        return path
    
    def _get_ancestor_path(self, cell_type: str) -> List[str]:
        """Get full ancestor path for a cell type"""
        if not self.driver:
            return [cell_type]
            
        try:
            with self.driver.session(database=self.db_name) as session:
                query = """
                MATCH path = (root)-[:DEVELOPS_TO*0..]->(target:CellType {name: $cell_type})
                WHERE NOT EXISTS(()-[:DEVELOPS_TO]->(root))
                RETURN [node in nodes(path) | node.name] as ancestor_path
                ORDER BY length(path) ASC
                LIMIT 1
                """
                result = session.run(query, cell_type=cell_type)
                record = result.single()
                return record["ancestor_path"] if record else [cell_type]
        except Exception as e:
            print(f"⚠️ Error getting ancestor path for {cell_type}: {e}")
            return [cell_type]
    
    def _get_direct_children(self, cell_type: str) -> List[str]:
        """Get direct children of a cell type"""
        if not self.driver:
            return []
            
        try:
            with self.driver.session(database=self.db_name) as session:
                query = """
                MATCH (parent:CellType {name: $cell_type})-[:DEVELOPS_TO]->(child:CellType)
                RETURN collect(child.name) as children
                """
                result = session.run(query, cell_type=cell_type)
                record = result.single()
                return record["children"] if record else []
        except Exception as e:
            print(f"⚠️ Error getting children for {cell_type}: {e}")
            return []
    
    def _suggest_similar_types(self, target_type: str, available_types: Set[str]) -> List[str]:
        """Suggest similar cell types using string similarity"""
        target_lower = target_type.lower()
        suggestions = []
        
        # Exact substring matches
        for ct in available_types:
            if target_lower in ct.lower() or ct.lower() in target_lower:
                suggestions.append(ct)
        
        # Add known valid types that contain similar words
        for ct in self.valid_cell_types:
            if target_lower in ct.lower() and ct not in suggestions:
                suggestions.append(ct)
        
        return suggestions[:5]
    
    def get_lineage_summary(self) -> Dict[str, Any]:
        """Get summary of current lineage state"""
        summary = {
            "total_cells": len(self.cell_lineages),
            "unique_current_types": len(set(l.current_type for l in self.cell_lineages.values())),
            "processing_snapshots": len(self.processing_snapshots),
            "hierarchy_cache_size": len(self.type_hierarchy_cache),
            "current_type_distribution": {}
        }
        
        # Count cells by current type
        for lineage in self.cell_lineages.values():
            ct = lineage.current_type
            summary["current_type_distribution"][ct] = summary["current_type_distribution"].get(ct, 0) + 1
        
        return summary


class CellTypeExtractor:
    """
    🧬 UNIFIED: Centralized cell type extraction with multiple strategies
    Consolidates all redundant cell type parsing logic into a single system
    """
    
    def __init__(self, hierarchy_manager=None, adata=None):
        self.hierarchy_manager = hierarchy_manager
        self.adata = adata
        
        # Common cell type patterns for validation
        self.valid_cell_type_patterns = [
            r'\b(T cell)s?\b',
            r'\b(B cell)s?\b', 
            r'\b(Natural killer cell)s?\b',
            r'\b(NK cell)s?\b',
            r'\b(Mononuclear phagocyte)s?\b',
            r'\b(Plasmacytoid dendritic cell)s?\b',
            r'\b(Dendritic cell)s?\b',
            r'\b(Macrophage)s?\b',
            r'\b(Monocyte)s?\b',
            r'\b([A-Z][a-z]+\s+cell)s?\b',  # Generic pattern for "Xyz cell"
        ]
        
        # Common separators for multi-cell-type strings
        self.cell_type_separators = [',', ' and ', ' & ', ';', ' vs ', ' versus ', ' or ']
        
        # Invalid patterns to filter out
        self.invalid_patterns = [
            r'^\d+$',        # Pure numbers
            r'^[a-z]+$',     # All lowercase
            r'^[A-Z]$',      # Single capital letter
            r'cluster',      # Contains 'cluster'
            r'sample',       # Contains 'sample'  
            r'condition',    # Contains 'condition'
            r'group',        # Contains 'group'
            r'^type$',       # Just 'type'
            r'analysis',     # Contains 'analysis'
        ]
    
    def extract_from_annotation_result(self, result: Any) -> List[str]:
        """
        🎯 STRATEGY 1: Extract cell types from annotation results
        Handles dictionary format, markdown headers, and text patterns
        """
        print(f"🔍 CellTypeExtractor.extract_from_annotation_result() called with result type: {type(result)}")
        
        if not isinstance(result, str):
            print(f"⚠️ CellTypeExtractor: Result is not string, returning empty list")
            return []
        
        cell_types = []
        text = str(result)
        
        # PRIORITY 1: Extract from dictionary format (most reliable)
        dict_pattern = r"group_to_cell_type\s*=\s*\{([^}]+)\}"
        dict_match = re.search(dict_pattern, text)
        if dict_match:
            dict_content = dict_match.group(1)
            # Extract values from the dictionary (cell types are the values)
            value_pattern = r"'[^']*':\s*'([^']+)'"
            values = re.findall(value_pattern, dict_content)
            cell_types.extend(values)
            print(f"✅ CellTypeExtractor: Extracted {len(values)} cell types from dictionary format: {values}")
            
            # If we found cell types in dictionary format, return them (most reliable)
            if values:
                cleaned_result = self._clean_and_deduplicate(values)
                print(f"🎯 CellTypeExtractor.extract_from_annotation_result() returning (dictionary): {cleaned_result}")
                return cleaned_result
        
        # PRIORITY 2: Extract from markdown headers
        header_matches = re.findall(r"###\s*Cluster\s*\d+:\s*([^:\n]+)", text)
        cell_types.extend([match.strip() for match in header_matches])
        
        # PRIORITY 3: Use specific known patterns if above methods fail
        if not cell_types:
            for pattern in self.valid_cell_type_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                cell_types.extend(matches)
        
        final_result = self._clean_and_deduplicate(cell_types)
        print(f"🎯 CellTypeExtractor.extract_from_annotation_result() returning (patterns): {final_result}")
        return final_result
    
    def extract_from_analysis_description(self, analysis_description: str) -> Optional[str]:
        """
        🎯 STRATEGY 2: Extract single cell type from analysis description
        Uses multiple regex patterns to find cell types in text
        """
        print(f"🔍 CellTypeExtractor.extract_from_analysis_description() called with: '{analysis_description}'")
        
        if not analysis_description:
            print(f"⚠️ CellTypeExtractor: Empty analysis description, returning None")
            return None
        
        # Common patterns for cell type extraction from analysis descriptions
        patterns = [
            r'analyze\s+([A-Z][a-z]*\s+cell)',
            r'for\s+([A-Z][a-z]*\s+cell)', 
            r'of\s+([A-Z][a-z]*\s+cell)',
            r'([A-Z][a-z]*\s+cell)\s+analysis',
            r'([A-Z][a-z]*\s+cell)\s+enrichment',
            r'([A-Z][a-z]*\s+cell)\s+comparison',
            r'([A-Z][a-z]*\s+cell)\s+dea',
            r'([A-Z][a-z]*\s+cell)\s+markers'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, analysis_description, re.IGNORECASE)
            if match:
                cell_type = match.group(1).strip()
                print(f"📋 CellTypeExtractor: Found potential cell type '{cell_type}' with pattern '{pattern}'")
                if self._is_valid_cell_type(cell_type):
                    print(f"🎯 CellTypeExtractor.extract_from_analysis_description() returning: '{cell_type}'")
                    return cell_type
                else:
                    print(f"❌ CellTypeExtractor: '{cell_type}' failed validation")
        
        print(f"🎯 CellTypeExtractor.extract_from_analysis_description() returning: None")
        return None
    
    def parse_multi_cell_type_string(self, cell_type_string: str) -> List[str]:
        """
        🎯 STRATEGY 3: Parse strings that might contain multiple cell types
        Handles various separators and formats
        """
        print(f"🔍 CellTypeExtractor.parse_multi_cell_type_string() called with: '{cell_type_string}'")
        
        if not cell_type_string:
            print(f"⚠️ CellTypeExtractor: Empty cell type string, returning empty list")
            return []
        
        # Start with the original string
        cell_types = [cell_type_string]
        
        # Apply each separator to split the string
        for separator in self.cell_type_separators:
            new_cell_types = []
            for ct in cell_types:
                if separator in ct:
                    new_cell_types.extend([part.strip() for part in ct.split(separator)])
                else:
                    new_cell_types.append(ct)
            cell_types = new_cell_types
        
        final_result = self._clean_and_deduplicate(cell_types)
        print(f"🎯 CellTypeExtractor.parse_multi_cell_type_string() returning: {final_result}")
        return final_result
    
    def extract_from_execution_state(self, state: dict) -> List[str]:
        """
        🎯 STRATEGY 4: Extract all cell types from execution state
        Combines execution plan, history, and critic feedback
        """
        print(f"🔍 CellTypeExtractor.extract_from_execution_state() called")
        cell_types = set()
        
        # From execution plan
        if state.get("execution_plan") and state["execution_plan"].get("steps"):
            for step in state["execution_plan"]["steps"]:
                # Check parameters
                cell_type = step.get("parameters", {}).get("cell_type")
                if cell_type and cell_type != "overall":
                    cell_types.update(self.parse_multi_cell_type_string(cell_type))
                
                # Check target_cell_type
                target_cell_type = step.get("target_cell_type")
                if target_cell_type and target_cell_type != "overall":
                    cell_types.update(self.parse_multi_cell_type_string(target_cell_type))
        
        # From execution history
        if state.get("execution_history"):
            for execution in state["execution_history"]:
                if execution.get("success") and execution.get("step"):
                    step = execution["step"]
                    cell_type = step.get("parameters", {}).get("cell_type")
                    if cell_type and cell_type != "overall":
                        cell_types.update(self.parse_multi_cell_type_string(cell_type))
        
        # From critic feedback
        for feedback in state.get("critic_feedback_history", []):
            for missing in feedback.get("missing_analyses", []):
                cell_type = self.extract_from_analysis_description(missing)
                if cell_type:
                    cell_types.add(cell_type)
        
        final_result = list(cell_types)
        print(f"🎯 CellTypeExtractor.extract_from_execution_state() returning: {final_result}")
        return final_result
    
    def extract_from_execution_context(self, state: dict, include_history: bool = True) -> List[str]:
        """
        🎯 STRATEGY 5: Smart context-based extraction for cache and analysis
        Prioritizes current execution context over historical data
        """
        print(f"🔍 CellTypeExtractor.extract_from_execution_context() called with include_history={include_history}")
        relevant_cell_types = set()
        
        print("🔍 CellTypeExtractor: Extracting cell types from execution context...")
        
        # 1. PRIORITY: Current execution plan (LLM already identified these!)
        if state.get("execution_plan") and state["execution_plan"].get("steps"):
            for step in state["execution_plan"]["steps"]:
                cell_type = step.get("parameters", {}).get("cell_type")
                if cell_type and cell_type != "overall":
                    parsed_types = self.parse_multi_cell_type_string(cell_type)
                    relevant_cell_types.update(parsed_types)
                    for ct in parsed_types:
                        print(f"   📋 From execution plan: {ct}")
                
                target_cell_type = step.get("target_cell_type")
                if target_cell_type and target_cell_type != "overall":
                    parsed_types = self.parse_multi_cell_type_string(target_cell_type)
                    relevant_cell_types.update(parsed_types)
                    for ct in parsed_types:
                        print(f"   🎯 From target cell type: {ct}")
        
        # 2. Current execution history (what was actually analyzed)
        if state.get("execution_history"):
            for execution in state["execution_history"]:
                if execution.get("success") and execution.get("step"):
                    step = execution["step"]
                    cell_type = step.get("parameters", {}).get("cell_type")
                    if cell_type and cell_type != "overall":
                        parsed_types = self.parse_multi_cell_type_string(cell_type)
                        relevant_cell_types.update(parsed_types)
                        for ct in parsed_types:
                            print(f"   ✅ From execution history: {ct}")
        
        # 3. Include historical data if requested
        if include_history and hasattr(self, '_get_historical_cell_types'):
            historical_types = self._get_historical_cell_types()
            relevant_cell_types.update(historical_types)
            for ct in historical_types:
                print(f"   📜 From function history: {ct}")
        
        # 4. Fallbacks
        if not relevant_cell_types:
            fallback_types = self._get_fallback_cell_types(state)
            relevant_cell_types.update(fallback_types)
            print(f"   🔄 Using fallback cell types: {fallback_types}")
        
        result = list(relevant_cell_types)
        print(f"🎯 CellTypeExtractor.extract_from_execution_context() returning: {result}")
        return result
    
    def _clean_and_deduplicate(self, cell_types: List[str]) -> List[str]:
        """Clean, validate, and deduplicate cell type list"""
        print(f"🧹 CellTypeExtractor._clean_and_deduplicate() called with: {cell_types}")
        cleaned_types = []
        
        for cell_type in cell_types:
            cleaned = cell_type.strip()
            
            # Basic validation
            if (len(cleaned) > 2 and 
                cleaned not in cleaned_types and
                self._is_valid_cell_type(cleaned)):
                cleaned_types.append(cleaned)
                print(f"✅ CellTypeExtractor: Accepted cell type: '{cleaned}'")
            else:
                print(f"❌ CellTypeExtractor: Rejected cell type: '{cleaned}' (len={len(cleaned)}, duplicate={cleaned in cleaned_types}, valid={self._is_valid_cell_type(cleaned)})")
        
        print(f"🧹 CellTypeExtractor._clean_and_deduplicate() returning: {cleaned_types}")
        return cleaned_types
    
    def _is_valid_cell_type(self, cell_type: str) -> bool:
        """Enhanced validation for cell types"""
        if not cell_type:
            return False
        
        cell_type_lower = cell_type.lower()
        
        # Check against invalid patterns
        for pattern in self.invalid_patterns:
            if re.search(pattern, cell_type_lower):
                print(f"❌ CellTypeExtractor: '{cell_type}' matches invalid pattern: {pattern}")
                return False
        
        # If we have a hierarchy manager, use it for validation
        if self.hierarchy_manager:
            is_valid = self.hierarchy_manager.is_valid_cell_type(cell_type)
            print(f"🔍 CellTypeExtractor: Hierarchy manager validation for '{cell_type}': {is_valid}")
            return is_valid
        
        # Basic validation: should contain "cell" or be a known cell type
        if "cell" in cell_type_lower:
            return True
        
        # Check against known patterns
        for pattern in self.valid_cell_type_patterns:
            if re.search(pattern, cell_type, re.IGNORECASE):
                return True
        
        return False
    
    def _get_fallback_cell_types(self, state: dict) -> List[str]:
        """Get fallback cell types when no extraction succeeds"""
        # Try available cell types from state
        if state.get("available_cell_types"):
            return state["available_cell_types"]
        
        # Try extracting from adata if available
        if self.adata and hasattr(self.adata, 'obs') and 'cell_type' in self.adata.obs.columns:
            return list(self.adata.obs["cell_type"].unique())
        
        return []
    
    def _get_historical_cell_types(self) -> List[str]:
        """Get cell types from historical function executions"""
        # This would be implemented to get historical data
        # For now, return empty list as placeholder
        return []