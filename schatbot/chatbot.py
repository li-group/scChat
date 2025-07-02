from .annotation import initial_cell_annotation, process_cells, handle_process_cells_result
from .visualizations import (
    display_dotplot,
    display_cell_type_composition,
    display_gsea_dotplot,
    display_umap,
    display_processed_umap,
    display_enrichment_barplot,
    display_enrichment_dotplot
)
from .enrichment import perform_enrichment_analyses
from .utils import (
    clear_directory,
    dea_split_by_condition,
    compare_cell_count,
)
import os
import json
import re
import openai
import pandas as pd
import glob  # Added for cache file pattern matching
from typing import Dict, Any, Tuple, Optional, List, TypedDict, Literal, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import time
from pathlib import Path
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class CellTypeRelation(Enum):
    """Enumeration of possible relationships between cell types"""
    ANCESTOR = "ancestor"
    DESCENDANT = "descendant" 
    SIBLING = "sibling"
    SAME = "same"
    UNRELATED = "unrelated"

@dataclass
class CellTypeLineage:
    cell_id: str
    current_type: str
    lineage_path: List[str]
    processing_history: List[Dict[str, Any]]

class HierarchicalCellTypeManager:
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


class AnalysisFunctionWrapper:
    """Wrapper that makes existing analysis functions hierarchy-aware"""
    
    def __init__(self, hierarchy_manager: HierarchicalCellTypeManager):
        self.hierarchy_manager = hierarchy_manager
    
    def perform_enrichment_analyses_hierarchical(self, cell_type: str, **kwargs):
        """Hierarchy-aware enrichment analysis"""
        try:
            # Get analysis-ready adata
            analysis_adata, metadata = self.hierarchy_manager.get_analysis_ready_adata(cell_type)
            
            print(f"🧬 Performing enrichment analysis on {metadata.get('cell_count', 'unknown')} cells")
            
            # Call original function with resolved adata
            result = perform_enrichment_analyses(analysis_adata, cell_type, **kwargs)
            
            # Add hierarchy metadata to result
            if isinstance(result, dict):
                result["hierarchy_metadata"] = metadata
                result["resolution_info"] = f"Resolved via {metadata['resolution_method']}"
            
            return result
            
        except ValueError as e:
            # Handle case where process_cells is needed
            if "requires process_cells" in str(e):
                return {
                    "status": "needs_processing", 
                    "message": str(e),
                    "required_steps": metadata.get("processing_path", []) if 'metadata' in locals() else []
                }
            raise e
    
    def dea_split_by_condition_hierarchical(self, cell_type: str, **kwargs):
        """Hierarchy-aware DEA"""
        try:
            analysis_adata, metadata = self.hierarchy_manager.get_analysis_ready_adata(cell_type)
            
            print(f"🧬 Performing DEA on {metadata.get('cell_count', 'unknown')} cells")
            
            # Call original function with resolved adata
            result = dea_split_by_condition(analysis_adata, cell_type, **kwargs)
            
            # Add metadata
            if isinstance(result, (list, dict)):
                return {
                    "dea_results": result,
                    "hierarchy_metadata": metadata,
                    "resolution_info": f"Resolved via {metadata['resolution_method']}"
                }
            
            return result
            
        except ValueError as e:
            if "requires process_cells" in str(e):
                return {
                    "status": "needs_processing",
                    "message": str(e),
                    "required_steps": metadata.get("processing_path", []) if 'metadata' in locals() else []
                }
            raise e
    
    def compare_cell_count_hierarchical(self, cell_type: str, **kwargs):
        """Hierarchy-aware cell count comparison"""
        try:
            analysis_adata, metadata = self.hierarchy_manager.get_analysis_ready_adata(cell_type)
            
            # Call original function
            result = compare_cell_count(analysis_adata, cell_type, **kwargs)
            
            # Add metadata
            if isinstance(result, list):
                return {
                    "count_results": result,
                    "hierarchy_metadata": metadata,
                    "resolution_info": f"Resolved via {metadata['resolution_method']}"
                }
            
            return result
            
        except ValueError as e:
            if "requires process_cells" in str(e):
                return {
                    "status": "needs_processing",
                    "message": str(e),
                    "required_steps": metadata.get("processing_path", []) if 'metadata' in locals() else []
                }
            raise e


# ==============================================================================
# Function History System
# ==============================================================================
class FunctionHistoryManager:
    """Manages function execution history using single global JSON file"""
    
    def __init__(self, history_dir: str = "function_history"):
        self.history_dir = history_dir
        self.history_file = os.path.join(history_dir, "execution_history.json")
        
        # Create directory if it doesn't exist
        os.makedirs(history_dir, exist_ok=True)
        
        # Load existing history
        self.history = self._load_history()
        
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load function history from JSON file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load function history: {e}")
        return []
    
    def _save_history(self):
        """Save function history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Could not save function history: {e}")
    
    def record_execution(self, function_name: str, parameters: Dict[str, Any], 
                        result: Any, success: bool, error: Optional[str] = None):
        """Record a function execution"""
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "function_name": function_name,
            "parameters": parameters,
            "success": success,
            "error": error,
            "result_summary": str(result)[:500] if result else None,  # Truncate long results
        }
        
        # Add to persistent history only
        self.history.append(execution_record)
        
        # Save to file
        self._save_history()
    
    def has_been_executed(self, function_name: str, parameters: Dict[str, Any]) -> bool:
        """Check if a function with specific parameters has been executed recently"""
        for record in reversed(self.history[-50:]):  # Check last 50 executions
            if (record["function_name"] == function_name and 
                record["parameters"] == parameters and 
                record["success"]):
                return True
        return False
    
    def get_recent_executions(self, function_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent executions of a specific function"""
        matching = [r for r in self.history if r["function_name"] == function_name and r["success"]]
        return matching[-limit:] if matching else []
    
    def get_available_results(self) -> Dict[str, Any]:
        """Get summary of available analysis results"""
        results_summary = {}
        
        for record in self.history:
            if record["success"]:
                func_name = record["function_name"]
                params = record["parameters"]
                
                if func_name == "process_cells":
                    cell_type = params.get("cell_type", "unknown")
                    if "processed_cell_types" not in results_summary:
                        results_summary["processed_cell_types"] = []
                    if cell_type not in results_summary["processed_cell_types"]:
                        results_summary["processed_cell_types"].append(cell_type)
                
                elif "enrichment" in func_name:
                    cell_type = params.get("cell_type", "unknown")
                    analysis_type = params.get("analysis", "unknown")
                    if "enrichment_analyses" not in results_summary:
                        results_summary["enrichment_analyses"] = {}
                    if cell_type not in results_summary["enrichment_analyses"]:
                        results_summary["enrichment_analyses"][cell_type] = []
                    if analysis_type not in results_summary["enrichment_analyses"][cell_type]:
                        results_summary["enrichment_analyses"][cell_type].append(analysis_type)
                
                elif func_name == "dea_split_by_condition":
                    cell_type = params.get("cell_type", "unknown")
                    if "dea_analyses" not in results_summary:
                        results_summary["dea_analyses"] = []
                    if cell_type not in results_summary["dea_analyses"]:
                        results_summary["dea_analyses"].append(cell_type)
        
        return results_summary

# ==============================================================================
# Enhanced Cache System with Smart Insights
# ==============================================================================
class SimpleIntelligentCache:
    """
    🧠 ENHANCED CACHE: Smart cache with analysis insights extraction
    """
    
    def __init__(self):
        # Map analysis types to their result directories (matching your current structure)
        self.cache_directories = {
            "enrichment": {
                "reactome": "schatbot/enrichment/reactome",
                "go_bp": "schatbot/enrichment/go_bp", 
                "go_mf": "schatbot/enrichment/go_mf",
                "go_cc": "schatbot/enrichment/go_cc",
                "kegg": "schatbot/enrichment/kegg",
                "gsea": "schatbot/enrichment/gsea"
            },
            "dea": "schatbot/deg_res",
            "visualizations": "umaps/annotated",
            "process_cells": "umaps/annotated"
        }
        
        # Default cache TTL (Time To Live) in hours
        self.default_ttl_hours = 24
        
        # Cache hit/miss statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0
        }
    
    def get_analysis_insights(self, cell_type: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        🎯 CORE: Extract key insights from cached analysis results for context building
        """
        insights = {
            "enrichment_insights": {},
            "dea_insights": {},
            "cell_counts": {},
            "summary": []
        }
        
        if not analysis_types:
            analysis_types = ["enrichment", "dea"]
        
        print(f"🔍 Extracting insights for {cell_type} from {analysis_types}")
        
        # Get enrichment insights
        if "enrichment" in analysis_types:
            enrichment_patterns = self._get_cache_file_patterns("enrichment", cell_type, 
                                                               {"analyses": ["reactome", "go", "kegg", "gsea"]})
            
            for pattern in enrichment_patterns:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if self._is_file_recent(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            analysis_name = self._extract_analysis_name_from_path(file_path)
                            
                            # Extract top 5 most significant terms
                            if not df.empty and 'p_value' in df.columns:
                                top_terms = df.nsmallest(5, 'p_value')
                                insights["enrichment_insights"][analysis_name] = {
                                    "top_terms": top_terms['Term'].tolist() if 'Term' in df.columns else [],
                                    "p_values": top_terms['p_value'].tolist(),
                                    "total_significant": len(df[df['p_value'] < 0.05]) if 'p_value' in df.columns else 0
                                }
                                insights["summary"].append(
                                    f"{analysis_name.upper()}: Found {len(df[df['p_value'] < 0.05])} significant pathways/terms"
                                )
                                print(f"✅ Extracted {analysis_name} insights: {len(df[df['p_value'] < 0.05])} significant terms")
                        except Exception as e:
                            print(f"⚠️ Error extracting insights from {file_path}: {e}")
        
        # Get DEA insights  
        if "dea" in analysis_types:
            dea_patterns = self._get_cache_file_patterns("dea", cell_type)
            for pattern in dea_patterns:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if self._is_file_recent(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            condition = self._extract_condition_from_path(file_path)
                            
                            if not df.empty:
                                # Extract key DEA statistics
                                significant_genes = len(df[df['pvals_adj'] < 0.05]) if 'pvals_adj' in df.columns else 0
                                upregulated = len(df[(df['pvals_adj'] < 0.05) & (df['logfoldchanges'] > 1)]) if 'logfoldchanges' in df.columns else 0
                                downregulated = len(df[(df['pvals_adj'] < 0.05) & (df['logfoldchanges'] < -1)]) if 'logfoldchanges' in df.columns else 0
                                
                                insights["dea_insights"][condition] = {
                                    "significant_genes": significant_genes,
                                    "upregulated": upregulated,
                                    "downregulated": downregulated,
                                    "top_genes": df.nlargest(5, 'logfoldchanges')['names'].tolist() if 'names' in df.columns else []
                                }
                                insights["summary"].append(
                                    f"DEA ({condition}): {significant_genes} significant genes ({upregulated} up, {downregulated} down)"
                                )
                                print(f"✅ Extracted DEA insights for {condition}: {significant_genes} significant genes")
                        except Exception as e:
                            print(f"⚠️ Error extracting DEA insights from {file_path}: {e}")
        
        print(f"📊 Total insights extracted: {len(insights['enrichment_insights'])} enrichment + {len(insights['dea_insights'])} DEA")
        return insights

    def _extract_analysis_name_from_path(self, file_path: str) -> str:
        """Extract analysis type from file path"""
        path_lower = file_path.lower()
        if "reactome" in path_lower:
            return "Reactome"
        elif "go_bp" in path_lower:
            return "GO_BP"
        elif "go_mf" in path_lower:
            return "GO_MF" 
        elif "go_cc" in path_lower:
            return "GO_CC"
        elif "kegg" in path_lower:
            return "KEGG"
        elif "gsea" in path_lower:
            return "GSEA"
        else:
            return "Unknown"

    def _extract_condition_from_path(self, file_path: str) -> str:
        """Extract condition from DEA file path"""
        filename = os.path.basename(file_path)
        # Pattern: {cell_type}_markers_{condition}.csv
        parts = filename.split('_markers_')
        if len(parts) == 2:
            return parts[1].replace('.csv', '')
        return "Unknown"
    
    def _generate_cache_key(self, function_name: str, parameters: dict) -> str:
        """Generate a simple cache key from function name and parameters"""
        # Create a deterministic string from parameters
        param_items = []
        for key, value in sorted(parameters.items()):
            param_items.append(f"{key}={value}")
        param_str = "_".join(param_items)
        
        # Create hash to handle long parameter strings
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return f"{function_name}_{param_hash}"
    
    def _get_cache_file_patterns(self, analysis_type: str, cell_type: str, 
                                 parameters: dict = None) -> List[str]:
        """Get file patterns to look for based on analysis type and parameters"""
        patterns = []
        
        if analysis_type == "enrichment":
            analyses = parameters.get("analyses", ["reactome", "go", "kegg", "gsea"]) if parameters else ["reactome", "go", "kegg", "gsea"]
            for analysis in analyses:
                if analysis == "go":
                    # GO has subdomains
                    for domain in ["bp", "mf", "cc"]:
                        dir_path = self.cache_directories["enrichment"][f"go_{domain}"]
                        patterns.append(f"{dir_path}/results_summary_{cell_type}.csv")
                else:
                    dir_path = self.cache_directories["enrichment"][analysis]
                    patterns.append(f"{dir_path}/results_summary_{cell_type}.csv")
        
        elif analysis_type == "dea":
            # DEA files have specific naming pattern
            dir_path = self.cache_directories["dea"]
            patterns.append(f"{dir_path}/{cell_type}_markers_*.csv")  # Multiple condition files
        
        elif analysis_type == "process_cells":
            # Process cells creates annotated UMAP files
            dir_path = self.cache_directories["visualizations"]
            patterns.append(f"{dir_path}/{cell_type}_umap_data.csv")
        
        elif analysis_type == "visualization":
            # Visualization files
            dir_path = self.cache_directories["visualizations"]
            patterns.append(f"{dir_path}/*{cell_type}*.csv")
        
        return patterns
    
    def _is_file_recent(self, file_path: str, ttl_hours: int = None) -> bool:
        """Check if file is recent enough to be considered valid cache"""
        if ttl_hours is None:
            ttl_hours = self.default_ttl_hours
        
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return False
            
            # Get file modification time
            file_mtime = datetime.fromtimestamp(file_path_obj.stat().st_mtime)
            
            # Check if file is within TTL
            age = datetime.now() - file_mtime
            is_recent = age.total_seconds() < (ttl_hours * 3600)
            
            if is_recent:
                print(f"✅ Cache file is recent: {file_path} (age: {age.total_seconds()/3600:.1f}h)")
            else:
                print(f"⏰ Cache file is stale: {file_path} (age: {age.total_seconds()/3600:.1f}h)")
            
            return is_recent
            
        except Exception as e:
            print(f"⚠️ Error checking file age: {e}")
            return False
    
    def _load_cached_result(self, file_patterns: List[str]) -> Tuple[Optional[Any], Optional[str]]:
        """Load cached result from file patterns"""
        for pattern in file_patterns:
            # Use glob to find matching files
            matching_files = glob.glob(pattern)
            
            for file_path in matching_files:
                if self._is_file_recent(file_path):
                    try:
                        # Load the CSV file
                        df = pd.read_csv(file_path)
                        
                        # Create a result summary
                        result_summary = {
                            "status": "cached_result",
                            "file_path": file_path,
                            "data": df,
                            "summary": f"Loaded cached result from {file_path}",
                            "row_count": len(df),
                            "cached_at": datetime.fromtimestamp(Path(file_path).stat().st_mtime).isoformat()
                        }
                        
                        return result_summary, file_path
                        
                    except Exception as e:
                        print(f"⚠️ Error loading cached file {file_path}: {e}")
                        continue
        
        return None, None
    
    def check_cache(self, function_name: str, parameters: dict, 
                   analysis_type: str = "general") -> Tuple[Optional[Any], bool]:
        """
        Check if we have a cached result for this function call
        
        Returns:
            (cached_result, is_cache_hit)
        """
        self.stats["total_requests"] += 1
        
        # Extract cell type from parameters
        cell_type = parameters.get("cell_type", "unknown")
        
        print(f"🔍 Checking cache for {function_name} with {cell_type}...")
        
        # Get file patterns to search
        file_patterns = self._get_cache_file_patterns(analysis_type, cell_type, parameters)
        
        # Try to load cached result
        cached_result, cache_file = self._load_cached_result(file_patterns)
        
        if cached_result:
            self.stats["cache_hits"] += 1
            print(f"🎯 CACHE HIT: {function_name} - using {cache_file}")
            return cached_result, True
        else:
            self.stats["cache_misses"] += 1
            print(f"❌ CACHE MISS: {function_name} - will compute new result")
            return None, False
    
    def ensure_cache_directories(self):
        """Ensure all cache directories exist (matching your current structure)"""
        all_dirs = []
        
        # Collect all directory paths
        for analysis_type, dirs in self.cache_directories.items():
            if isinstance(dirs, dict):
                all_dirs.extend(dirs.values())
            else:
                all_dirs.append(dirs)
        
        # Create directories
        for dir_path in all_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def invalidate_cache(self, cell_type: str = None, analysis_type: str = None):
        """Invalidate cache entries (delete old files)"""
        print(f"🔥 Invalidating cache - cell_type: {cell_type}, analysis_type: {analysis_type}")
        
        if analysis_type and analysis_type in self.cache_directories:
            dirs_to_clean = self.cache_directories[analysis_type]
            if isinstance(dirs_to_clean, dict):
                dirs_to_clean = dirs_to_clean.values()
            else:
                dirs_to_clean = [dirs_to_clean]
            
            for dir_path in dirs_to_clean:
                if cell_type:
                    # Remove files for specific cell type
                    pattern = f"{dir_path}/*{cell_type}*"
                    files_to_remove = glob.glob(pattern)
                    for file_path in files_to_remove:
                        try:
                            Path(file_path).unlink()
                            print(f"🗑️ Removed: {file_path}")
                        except Exception as e:
                            print(f"⚠️ Error removing {file_path}: {e}")
                else:
                    # Remove all files in directory
                    try:
                        import shutil
                        if Path(dir_path).exists():
                            shutil.rmtree(dir_path)
                            Path(dir_path).mkdir(parents=True, exist_ok=True)
                            print(f"🗑️ Cleared directory: {dir_path}")
                    except Exception as e:
                        print(f"⚠️ Error clearing {dir_path}: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        hit_rate = (self.stats["cache_hits"] / self.stats["total_requests"] * 100 
                   if self.stats["total_requests"] > 0 else 0)
        
        return {
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "total_requests": self.stats["total_requests"],
            "hit_rate_percent": round(hit_rate, 1),
            "cache_directories": self.cache_directories
        }
    
    def cache_aware_function_wrapper(self, function_name: str, analysis_type: str, 
                                   compute_function, **kwargs):
        """
        Universal wrapper that checks cache first, then computes if needed
        """
        # Check cache first
        cached_result, is_hit = self.check_cache(function_name, kwargs, analysis_type)
        
        if is_hit:
            return cached_result
        
        # Cache miss - compute the result
        print(f"💻 Computing {function_name}...")
        start_time = time.time()
        
        try:
            result = compute_function()
            compute_time = time.time() - start_time
            
            print(f"✅ Computed {function_name} in {compute_time:.2f} seconds")
            
            # The result should already be saved to CSV by your existing functions
            # No additional caching needed since your functions already save results
            
            return result
            
        except Exception as e:
            print(f"❌ Error computing {function_name}: {e}")
            raise e


# ==============================================================================
# State Management
# ==============================================================================
class ChatState(TypedDict):
    """State that flows through the LangGraph workflow"""
    # Input/Output
    messages: List[BaseMessage]
    current_message: str
    response: str
    
    # Analysis context
    available_cell_types: List[str]
    adata: Optional[Any]
    
    # Planning and execution
    initial_plan: Optional[Dict[str, Any]]  # Initial plan before validation
    execution_plan: Optional[Dict[str, Any]]  # Validated plan after status checking
    current_step_index: int
    execution_history: List[Dict[str, Any]]
    
    # Function execution
    function_result: Optional[Any]
    function_name: Optional[str]
    function_args: Optional[Dict[str, Any]]
    
    # Memory and awareness
    function_history_summary: Optional[Dict[str, Any]]
    missing_cell_types: List[str]
    required_preprocessing: List[Dict[str, Any]]
    
    # Conversation management
    conversation_complete: bool
    errors: List[str]


@dataclass
class ExecutionStep:
    """Represents a single step in the execution plan"""
    step_type: str
    function_name: str
    parameters: Dict[str, Any]
    description: str
    expected_outcome: Optional[str] = None
    target_cell_type: Optional[str] = None
    expected_children: Optional[List[str]] = None


@dataclass
class ExecutionPlan:
    """Represents the complete execution plan"""
    steps: List[ExecutionStep]
    original_question: str
    plan_summary: str
    estimated_steps: int


# ==============================================================================
# LangGraph Multi-Agent ChatBot with Enhanced Cache Integration
# ==============================================================================
class MultiAgentChatBot:
    def __init__(self):
        # Initialize core components
        self._initialize_directories()
        self.api_key = os.getenv("OPENAI_API_KEY", "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")
        openai.api_key = self.api_key
        self.adata = None
        
        # Initialize memory and awareness systems
        self.history_manager = FunctionHistoryManager()
        
        # 🧠 ENHANCED: Initialize intelligent cache with insights
        self.simple_cache = SimpleIntelligentCache()
        self.simple_cache.ensure_cache_directories()
        
        # 🧬 UNIFIED: Single hierarchical cell type manager
        self.hierarchy_manager = None
        self.analysis_wrapper = None
        
        # Setup functions and initialize data
        self.setup_functions()
        self._initialize_annotation()
        
        # 🧬 Initialize hierarchical management after adata is ready
        self._initialize_hierarchical_management()
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
        
    def _initialize_directories(self):
        """Clean all directories at initialization"""
        directories_to_clear = [
            'figures', 'process_cell_data', 'schatbot/annotated_adata',
            'schatbot/enrichment', 'umaps/annotated', 'schatbot/runtime_data/basic_data/',
            'schatbot/deg_res'
        ]
        for directory in directories_to_clear:
            clear_directory(directory)

    def _initialize_annotation(self):
        """Initialize or load annotation data"""
        gene_dict, marker_tree, adata, explanation, annotation_result = initial_cell_annotation()
        self.adata = adata
        self.initial_cell_types = self._extract_initial_cell_types(annotation_result)
        self.initial_annotation_content = (
            f"Initial annotation complete.\n"
            f"• Annotation Result: {annotation_result}\n" 
            f"• Top‐genes per cluster: {gene_dict}\n"
            f"• Marker‐tree: {marker_tree}\n"
            f"• Explanation: {explanation}"
        )

    def _initialize_hierarchical_management(self):
        """🧬 UNIFIED: Initialize hierarchical cell type management"""
        if self.adata is not None:
            self.hierarchy_manager = HierarchicalCellTypeManager(self.adata)
            self.analysis_wrapper = AnalysisFunctionWrapper(self.hierarchy_manager)
            print("✅ Unified hierarchical cell type management initialized")
        else:
            print("⚠️ Cannot initialize hierarchical management without adata")

    def _extract_initial_cell_types(self, annotation_result: str) -> List[str]:
        """Extract cell types from initial annotation"""
        cell_types = []
        if isinstance(annotation_result, dict):
            for cluster, cell_type in annotation_result.items():
                if cell_type not in cell_types:
                    cell_types.append(cell_type)
        else:
            import re
            matches = re.findall(r"'([^']+)':\s*'([^']+)'", str(annotation_result))
            for _, cell_type in matches:
                if cell_type not in cell_types:
                    cell_types.append(cell_type)
        return cell_types

    def setup_functions(self):
        """Setup function descriptions and mappings"""
        self.visualization_functions = {
            "display_dotplot", "display_cell_type_composition", "display_gsea_dotplot",
            "display_umap", "display_processed_umap", "display_enrichment_barplot", 
            "display_enrichment_dotplot"
        }

        self.function_descriptions = [
            {
                "name": "display_dotplot",
                "description": "Display dotplot for the annotated results. Use when user wants to see gene expression patterns across cell types.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_cell_type_composition", 
                "description": "Display cell type composition graph. Use when user wants to see the proportion of different cell types.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_gsea_dotplot",
                "description": "Display GSEA dot plot. Use when user wants to see gene set enrichment analysis results.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "display_umap",
                "description": "Display UMAP that is NOT annotated with cell types. Use when user wants basic dimensionality reduction visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {"cell_type": {"type": "string", "description": "The cell type to focus on, or 'overall' for all cells"}},
                    "required": ["cell_type"],
                },
            },
            {
                "name": "display_processed_umap",
                "description": "Display UMAP that IS annotated with cell types. Use when user wants to see cell type annotations on UMAP.",
                "parameters": {
                    "type": "object", 
                    "properties": {"cell_type": {"type": "string", "description": "The cell type to focus on, or 'overall' for all cells"}},
                    "required": ["cell_type"],
                },
            },
            {
                "name": "perform_enrichment_analyses",
                "description": "Run enrichment analyses (reactome, go, kegg, gsea) on DE genes for a cell type. Use for pathway analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell_type": {"type": "string", "description": "The cell type to analyze."},
                        "analyses": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["reactome", "go", "kegg", "gsea"]},
                            "description": "Which enrichment analyses to run. Default: all if omitted."
                        },
                        "analysis_type": {
                            "type": "string", "enum": ["GSEA", "GO", "KEGG", "reactome"],
                            "description": "Specific type of enrichment analysis."
                        },
                        "logfc_threshold": {"type": "number", "description": "Minimum absolute log2 fold change."},
                        "pval_threshold": {"type": "number", "description": "Adjusted p‑value cutoff."},
                        "top_n_terms": {"type": "integer", "description": "How many top enriched terms to return."}
                    },
                    "required": ["cell_type"]
                }
            },
            {
                "name": "process_cells",
                "description": "Recluster and further annotate cells based on cell type. Use when user wants to find subtypes within a cell type.",
                "parameters": {
                    "type": "object",
                    "properties": {"cell_type": {"type": "string", "description": "The cell type to process for subtype discovery."}},
                    "required": ["cell_type"]
                }
            },
            {
                "name": "display_enrichment_barplot",
                "description": "Show barplot of top enriched terms. Use for visualizing enrichment analysis results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string", "enum": ["reactome","go","kegg","gsea"]},
                        "cell_type": {"type": "string"},
                        "top_n": {"type": "integer", "default": 10},
                        "domain": {"type": "string", "enum": ["BP","MF","CC"]},
                        "condition": {"type": "string"}
                    },
                    "required": ["analysis","cell_type"]
                }
            },
            {
                "name": "display_enrichment_dotplot", 
                "description": "Show dotplot of top enriched terms. Use for detailed enrichment visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string", "enum": ["reactome","go","kegg","gsea"]},
                        "cell_type": {"type": "string"},
                        "top_n": {"type": "integer", "default": 10}, 
                        "domain": {"type": "string", "enum": ["BP","MF","CC"]},
                        "condition": {"type": "string"}
                    },
                    "required": ["analysis","cell_type"]
                }
            },
            {
                "name": "dea_split_by_condition",
                "description": "Perform differential expression analysis split by condition. Use when comparing conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {"cell_type": {"type": "string"}},
                    "required": ["cell_type"]
                }
            },
            {
                "name": "compare_cell_counts",
                "description": "Compare cell counts between conditions. Use when analyzing cell type abundance differences.", 
                "parameters": {
                    "type": "object",
                    "properties": {"cell_type": {"type": "string"}},
                    "required": ["cell_type"]
                }
            },
            {
                "name": "conversational_response",
                "description": "Provide a conversational response without function calls. Use for greetings, clarifications, explanations, or when no analysis is needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response_type": {"type": "string", "enum": ["greeting", "explanation", "clarification", "general"]}
                    },
                    "required": ["response_type"]
                }
            },
            {
                "name": "validate_processing_results",
                "description": "Validate that process_cells discovered expected cell types. Internal validation step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "processed_parent": {"type": "string", "description": "The parent cell type that was processed"},
                        "expected_children": {"type": "array", "items": {"type": "string"}, "description": "Expected child cell types"}
                    },
                    "required": ["processed_parent", "expected_children"]
                }
            }
        ]
        
        # Function mappings
        self.function_mapping = {
            "display_dotplot": self._wrap_visualization(display_dotplot),
            "display_cell_type_composition": self._wrap_visualization(display_cell_type_composition),
            "display_gsea_dotplot": self._wrap_visualization(display_gsea_dotplot),
            "display_umap": self._wrap_visualization(display_umap),
            "display_processed_umap": self._wrap_visualization(display_processed_umap),
            "display_enrichment_barplot": self._wrap_visualization(display_enrichment_barplot),
            "display_enrichment_dotplot": self._wrap_visualization(display_enrichment_dotplot),
            "perform_enrichment_analyses": self._wrap_enrichment_analysis,
            "process_cells": self._wrap_process_cells,
            "dea_split_by_condition": self._wrap_dea_analysis,
            "compare_cell_counts": self._wrap_compare_cells,
            "conversational_response": self._wrap_conversational_response,
            "validate_processing_results": self._wrap_validate_processing_results,
        }

    # ========== LangGraph Workflow Creation ==========
    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(ChatState)
        
        workflow.add_node("input_processor", self.input_processor_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("status_checker", self.status_checker_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("evaluator", self.evaluator_node)
        workflow.add_node("response_generator", self.response_generator_node)
        
        # Set entry point
        workflow.set_entry_point("input_processor")
        
        workflow.add_edge("input_processor", "planner")
        workflow.add_edge("planner", "status_checker")  # Validate plan before execution
        workflow.add_edge("status_checker", "executor")
        workflow.add_edge("executor", "evaluator")
        
        # Conditional routing from evaluator
        workflow.add_conditional_edges(
            "evaluator", 
            self.route_from_evaluator,
            {
                "continue": "executor",
                "complete": "response_generator"
            }
        )
        
        # End at response generator
        workflow.add_edge("response_generator", END)
        
        return workflow.compile()

    # ========== LangGraph Node Functions ==========
    def input_processor_node(self, state: ChatState) -> ChatState:
        """Process incoming user message and initialize state"""
        # Initialize state if this is a new conversation
        if not state.get("messages"):
            state["messages"] = [AIMessage(content=self.initial_annotation_content)]
        
        # Add user message
        state["messages"].append(HumanMessage(content=state["current_message"]))
        
        # Initialize state variables
        state["available_cell_types"] = self.initial_cell_types
        state["adata"] = self.adata
        state["initial_plan"] = None
        state["execution_plan"] = None
        state["current_step_index"] = 0
        state["execution_history"] = []
        state["function_result"] = None
        state["function_name"] = None
        state["function_args"] = None
        state["conversation_complete"] = False
        state["errors"] = []
        
        # Load function history and memory context
        state["function_history_summary"] = self.history_manager.get_available_results()
        state["missing_cell_types"] = []
        state["required_preprocessing"] = []
        
        return state

    def planner_node(self, state: ChatState) -> ChatState:
        """Create initial execution plan (before validation)"""
        message = state["current_message"]
        available_functions = self.function_descriptions
        available_cell_types = state["available_cell_types"]
        function_history = state["function_history_summary"]
        
        planning_prompt = f"""
        You are an intelligent planner for single-cell RNA-seq analysis. 
        
        Create a step-by-step execution plan for the user query.
        
        CONTEXT:
        - Available cell types: {', '.join(available_cell_types)}
        - Previous analyses: {json.dumps(function_history, indent=2)}
        
        Available functions:
        {self._summarize_functions(available_functions)}
        
        User question: "{message}"
        
        Create a plan in this JSON format:
        {{
            "plan_summary": "Brief description of how you'll answer this question",
            "steps": [
                {{
                    "step_type": "analysis|visualization|conversation",
                    "function_name": "exact_function_name", 
                    "parameters": {{"param1": "value1"}},
                    "description": "What this step accomplishes",
                    "expected_outcome": "What we expect to produce",
                    "target_cell_type": "If applicable, which cell type"
                }}
            ]
        }}
        
        IMPORTANT GUIDELINES: 
        - When analyzing multiple cell types, create separate steps for each cell type
        - For example, if comparing "T cells" and "B cells", create separate steps:
          Step 1: analyze T cell, Step 2: analyze B cell, Step 3: compare results
        - Never put multiple cell types in a single parameter (e.g., don't use "T cells, B cells")
        - Use exact cell type names (e.g., "T cell", "B cell", not "T cells, B cells")
        - The status checker will validate and modify your plan if needed
        - Focus on creating a logical flow to answer the user's question
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            plan_data = json.loads(response.choices[0].message.content)
            
            # Store as initial plan (will be validated by status checker)
            state["initial_plan"] = plan_data
            
        except Exception as e:
            print(f"Planning error: {e}")
            # Fallback: create a simple conversational response plan
            state["initial_plan"] = {
                "plan_summary": "Fallback conversational response",
                "steps": [{
                    "step_type": "conversation",
                    "function_name": "conversational_response",
                    "parameters": {"response_type": "general"},
                    "description": "Provide a helpful response",
                    "expected_outcome": "Address user query",
                    "target_cell_type": None
                }]
            }
            
        return state

    def status_checker_node(self, state: ChatState) -> ChatState:
        if not state["initial_plan"]:
            state["execution_plan"] = state["initial_plan"]
            return state
        
        initial_steps = state["initial_plan"]["steps"]
        validated_steps = []
        missing_cell_types = []
        
        processing_paths_needed = {}  # parent_type -> set(target_children)
        processing_steps_created = {}  # parent_type -> step_object
        
        print(f"🔍 Hierarchical Status Checker: Validating {len(initial_steps)} steps...")
        
        for step in initial_steps:
            step_valid = True
            target_cell_type = step.get("target_cell_type") or step.get("parameters", {}).get("cell_type")
            
            # Check if step requires a specific cell type
            if target_cell_type and target_cell_type != "overall":
                # Parse multiple cell types if they exist
                cell_types_to_check = self._parse_cell_types(target_cell_type)
                
                for single_cell_type in cell_types_to_check:
                    single_cell_type = single_cell_type.strip()
                    
                    # 🧬 Use hierarchical manager to resolve cell type
                    if self.hierarchy_manager:
                        try:
                            resolved_series, metadata = self.hierarchy_manager.resolve_cell_type_for_analysis(single_cell_type)
                            
                            if metadata["resolution_method"] == "needs_processing":
                                processing_path = metadata["processing_path"]
                                print(f"🛤️ Processing path needed for '{single_cell_type}': {' → '.join(processing_path)}")
                                
                                # Track each parent->child relationship in the path
                                for i in range(len(processing_path) - 1):
                                    parent_type = processing_path[i]
                                    child_type = processing_path[i + 1]
                                    
                                    # Track that this parent needs to produce this child
                                    if parent_type not in processing_paths_needed:
                                        processing_paths_needed[parent_type] = set()
                                    processing_paths_needed[parent_type].add(child_type)
                                    
                                    print(f"📋 Tracking: {parent_type} → {child_type}")
                                
                                step_valid = True
                            
                            elif metadata["resolution_method"] == "not_found":
                                print(f"❌ Cell type '{single_cell_type}' not found. Suggestions: {metadata.get('suggestions', [])}")
                                step_valid = False
                            
                            elif metadata["resolution_method"] in ["direct", "ancestor_aggregation"]:
                                print(f"✅ Cell type '{single_cell_type}' can be resolved via {metadata['resolution_method']}")
                                step_valid = True
                                
                        except Exception as e:
                            print(f"⚠️ Error resolving cell type '{single_cell_type}': {e}")
                            step_valid = False
                
                # Handle multiple cell types (split into separate steps)
                if len(cell_types_to_check) > 1:
                    valid_cell_types = [ct.strip() for ct in cell_types_to_check 
                                      if self.hierarchy_manager and self.hierarchy_manager.is_valid_cell_type(ct.strip())]
                    
                    if valid_cell_types:
                        # Create a step for each valid cell type
                        for i, valid_ct in enumerate(valid_cell_types):
                            if i == 0:
                                # Modify the current step for the first cell type
                                step["parameters"]["cell_type"] = valid_ct
                                step["target_cell_type"] = valid_ct
                                step["description"] = step["description"].replace(target_cell_type, valid_ct)
                            else:
                                # Create new steps for additional cell types
                                new_step = step.copy()
                                new_step["parameters"] = step["parameters"].copy()
                                new_step["parameters"]["cell_type"] = valid_ct
                                new_step["target_cell_type"] = valid_ct
                                new_step["description"] = step["description"].replace(target_cell_type, valid_ct)
                                validated_steps.append(new_step)
                        step_valid = True
                    else:
                        step_valid = False
            
            # Check for redundant function calls
            if step_valid and step["function_name"] != "conversational_response":
                if self.history_manager.has_been_executed(step["function_name"], step.get("parameters", {})):
                    print(f"🔄 Function already executed recently: {step['function_name']}")
            
            if step_valid:
                validated_steps.append(step)
        
        consolidated_preprocessing = []
        
        for parent_type, target_children in processing_paths_needed.items():
            target_children_list = sorted(list(target_children))
            
            # Create a single processing step that will discover multiple children
            consolidated_step = {
                "step_type": "analysis",
                "function_name": "process_cells",
                "parameters": {"cell_type": parent_type},
                "description": f"Process {parent_type} to discover {', '.join(target_children_list)}",
                "expected_outcome": f"Discover cell types: {', '.join(target_children_list)}",
                "target_cell_type": parent_type,
                "expected_children": target_children_list
            }
            
            consolidated_preprocessing.append(consolidated_step)
            
            # Update available cell types for subsequent validation
            state["available_cell_types"].extend(target_children_list)
            
            print(f"🎯 Consolidated processing: {parent_type} → {', '.join(target_children_list)}")
        
        # Combine consolidated preprocessing with validated steps
        final_steps = consolidated_preprocessing + validated_steps
        
        final_steps_with_validation = []
        for step in final_steps:
            final_steps_with_validation.append(step)
            
            # Add validation step after process_cells operations
            if step["function_name"] == "process_cells" and "expected_children" in step:
                validation_step = {
                    "step_type": "validation",
                    "function_name": "validate_processing_results",
                    "parameters": {
                        "processed_parent": step["parameters"]["cell_type"],
                        "expected_children": step["expected_children"]
                    },
                    "description": f"Validate that {step['parameters']['cell_type']} processing discovered expected cell types",
                    "expected_outcome": "Confirm all expected cell types are available",
                    "target_cell_type": None
                }
                final_steps_with_validation.append(validation_step)
        
        # Create validated execution plan
        validated_plan = {
            "steps": final_steps_with_validation,
            "original_question": state["current_message"],
            "plan_summary": state["initial_plan"]["plan_summary"],
            "estimated_steps": len(final_steps_with_validation),
            "consolidation_summary": f"Consolidated {len(processing_paths_needed)} unique processing operations",
            "validation_notes": f"Added {len(consolidated_preprocessing)} consolidated processing steps with validation"
        }
        
        state["execution_plan"] = validated_plan
        state["missing_cell_types"] = missing_cell_types
        state["required_preprocessing"] = consolidated_preprocessing
        
        print(f"Final plan has {len(final_steps_with_validation)} steps")
        print(f"   • {len(consolidated_preprocessing)} consolidated processing operations")
        print(f"   • {len(validated_steps)} analysis/visualization steps")
        print(f"   • {len([s for s in final_steps_with_validation if s.get('step_type') == 'validation'])} validation steps")
        
        return state

    def executor_node(self, state: ChatState) -> ChatState:
        """Execute the current step in the plan with hierarchy awareness and validation"""
        if not state["execution_plan"] or state["current_step_index"] >= len(state["execution_plan"]["steps"]):
            state["conversation_complete"] = True
            return state
            
        step_data = state["execution_plan"]["steps"][state["current_step_index"]]
        step = ExecutionStep(**step_data)
        
        print(f"🔄 Executing step {state['current_step_index'] + 1}: {step.description}")
        
        success = False
        result = None
        error_msg = None
        
        try:
            # Handle validation steps specially
            if step.step_type == "validation":
                print("🔍 Executing validation step...")
                result = self.validate_processing_results(
                    step.parameters.get("processed_parent"),
                    step.parameters.get("expected_children", [])
                )
                
                # Check validation result
                if result["status"] == "success":
                    success = True
                    print(f"✅ Validation passed: {result['message']}")
                elif result["status"] == "partial_success":
                    success = True  # Continue but with warnings
                    print(f"⚠️ Validation partial: {result['message']}")
                    # Update available cell types with what we actually found
                    state["available_cell_types"] = result["available_types"]
                else:
                    success = False
                    error_msg = result["message"]
                    print(f"❌ Validation failed: {error_msg}")
                
            # Handle final question step differently
            elif step.step_type == "final_question":
                print("🎯 Executing final comprehensive question...")
                result = self._execute_final_question(state)
                success = True
                
            else:
                # Handle regular analysis/visualization steps
                if step.function_name not in self.function_mapping:
                    raise Exception(f"Function '{step.function_name}' not found")
                
                func = self.function_mapping[step.function_name]
                result = func(**step.parameters)
                success = True
                
                # 🧬 Update hierarchical manager after process_cells
                if step.function_name == "process_cells" and self.hierarchy_manager:
                    new_cell_types = self._extract_cell_types_from_result(result)
                    if new_cell_types:
                        print(f"🧬 Updating hierarchy manager with new cell types: {new_cell_types}")
                        self.hierarchy_manager.update_after_process_cells(
                            step.parameters.get("cell_type", "unknown"),
                            new_cell_types
                        )
                        
                        # Update available cell types in state
                        state["available_cell_types"] = list(set(state["available_cell_types"] + new_cell_types))
                        for new_type in new_cell_types:
                            print(f"✅ Discovered new cell type: {new_type}")
                    else:
                        print("⚠️ No new cell types discovered from process_cells")
            
            # Store results
            state["function_result"] = result
            state["function_name"] = step.function_name
            state["function_args"] = step.parameters
            
            print(f"✅ Step {state['current_step_index'] + 1} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            success = False
            print(f"❌ Step {state['current_step_index'] + 1} failed: {error_msg}")
            state["errors"].append(f"Step {state['current_step_index'] + 1} failed: {error_msg}")
        
        # Record function execution in history (skip validation steps for history)
        if step.step_type != "validation":
            self.history_manager.record_execution(
                function_name=step.function_name,
                parameters=step.parameters,
                result=result,
                success=success,
                error=error_msg
            )
        
        # Record execution in state
        state["execution_history"].append({
            "step_index": state["current_step_index"],
            "step": step_data,
            "success": success,
            "result": str(result)[:500] if result else "Success",
            "error": error_msg
        })
            
        return state

    def evaluator_node(self, state: ChatState) -> ChatState:
        """Evaluate execution results and determine next steps"""
        if not state["execution_history"]:
            state["conversation_complete"] = True
            return state
            
        last_execution = state["execution_history"][-1]
        
        if not last_execution["success"]:
            # If step failed, complete with error
            state["conversation_complete"] = True
            return state
        
        # Move to next step
        state["current_step_index"] += 1
        
        # Check if original plan steps are complete
        original_steps_complete = state["current_step_index"] >= len(state["execution_plan"]["steps"])
        
        # If all original steps are done, check if we need to add final question step
        if original_steps_complete:
            # Check if the last step was already a final question
            if state["execution_plan"]["steps"] and state["execution_plan"]["steps"][-1].get("step_type") == "final_question":
                # Final question already executed, we're truly done
                state["conversation_complete"] = True
                print("🏁 All steps including final question completed!")
            else:
                # Add final question step to the plan
                print("📝 Adding final comprehensive question step...")
                final_question_step = {
                    "step_type": "final_question",
                    "function_name": "final_question",
                    "parameters": {"original_question": state["execution_plan"]["original_question"]},
                    "description": "Ask comprehensive final question based on all analysis",
                    "expected_outcome": "Comprehensive answer to original question",
                    "target_cell_type": None
                }
                
                # Add the final step to the plan
                state["execution_plan"]["steps"].append(final_question_step)
                print(f"✅ Added final question step. Total steps now: {len(state['execution_plan']['steps'])}")
        
        return state

    def response_generator_node(self, state: ChatState) -> ChatState:
        """Generate final response based on execution results"""
        if not state["execution_plan"] or not state["execution_history"]:
            state["response"] = json.dumps({"response": "I encountered an issue processing your request."})
        else:
            # Check if the last executed step was a final question
            last_execution = state["execution_history"][-1] if state["execution_history"] else None
            
            if last_execution and last_execution["step"].get("step_type") == "final_question":
                # This was a final comprehensive question - use the result directly
                if last_execution["success"]:
                    comprehensive_answer = state["function_result"]
                    state["response"] = json.dumps({
                        "response": comprehensive_answer,
                        "response_type": "comprehensive_final_answer"
                    })
                    print("🎯 Using comprehensive final answer as response")
                else:
                    # Final question failed
                    error_msg = last_execution["error"] if last_execution else "Unknown error"
                    state["response"] = json.dumps({
                        "response": f"I encountered an error generating the final answer: {error_msg}"
                    })
            else:
                # Handle regular single-step or multi-step responses (existing logic)
                if len(state["execution_plan"]["steps"]) == 1:
                    step = state["execution_plan"]["steps"][0]
                    execution = state["execution_history"][0] if state["execution_history"] else None
                    
                    if execution and execution["success"]:
                        # Handle single-step responses
                        if step["function_name"] in self.visualization_functions:
                            # Visualization response
                            viz_summary = f"Generated {step['function_name']}"
                            if step.get("parameters", {}).get("cell_type"):
                                viz_summary += f" for {step['parameters']['cell_type']}"
                            state["response"] = json.dumps({
                                "response": viz_summary, 
                                "graph_html": state["function_result"]
                            })
                        elif step["function_name"] == "conversational_response":
                            # Conversational response
                            state["response"] = json.dumps({
                                "response": state["function_result"]
                            })
                        else:
                            # Analysis response with AI interpretation
                            interpretation = self._get_ai_interpretation_for_result(
                                state["current_message"], 
                                state["function_result"],
                                state["messages"]
                            )
                            state["response"] = json.dumps({"response": interpretation})
                    else:
                        # Single step failed
                        error_msg = execution["error"] if execution else "Unknown error"
                        state["response"] = json.dumps({
                            "response": f"I encountered an error: {error_msg}"
                        })
                else:
                    # Multi-step plan - generate comprehensive summary
                    summary = self._generate_execution_summary(state)
                    state["response"] = json.dumps({"response": summary})
        
        # Add response to message history
        try:
            response_content = json.loads(state["response"])["response"]
            state["messages"].append(AIMessage(content=response_content))
        except:
            state["messages"].append(AIMessage(content="Analysis completed."))
        
        return state

    # ========== 🎯 SOLUTION 1: Smart Cache Integration Methods ==========
    
    def _get_relevant_cell_types_from_context(self, state: ChatState) -> List[str]:
        """
        🎯 SMART: Extract relevant cell types from execution context
        No NLP needed - use LLM planning + execution history + state tracking
        """
        relevant_cell_types = set()
        
        print("🔍 Extracting cell types from execution context...")
        
        # 1. Extract from execution plan (LLM already identified these!)
        if state.get("execution_plan") and state["execution_plan"].get("steps"):
            for step in state["execution_plan"]["steps"]:
                # Get cell type from step parameters
                cell_type = step.get("parameters", {}).get("cell_type")
                if cell_type and cell_type != "overall":
                    relevant_cell_types.add(cell_type)
                    print(f"   📋 From execution plan: {cell_type}")
                
                # Also check target_cell_type
                target_cell_type = step.get("target_cell_type")
                if target_cell_type and target_cell_type != "overall":
                    relevant_cell_types.add(target_cell_type)
                    print(f"   🎯 From target cell type: {target_cell_type}")
        
        # 2. Extract from execution history (what was actually analyzed)
        if state.get("execution_history"):
            for execution in state["execution_history"]:
                if execution.get("success") and execution.get("step"):
                    step = execution["step"]
                    cell_type = step.get("parameters", {}).get("cell_type")
                    if cell_type and cell_type != "overall":
                        relevant_cell_types.add(cell_type)
                        print(f"   ✅ From execution history: {cell_type}")
        
        # 3. Extract from function history (previous sessions)
        recent_analyses = self.history_manager.get_recent_executions("perform_enrichment_analyses", limit=10)
        recent_analyses.extend(self.history_manager.get_recent_executions("dea_split_by_condition", limit=10))
        recent_analyses.extend(self.history_manager.get_recent_executions("process_cells", limit=10))
        
        for execution in recent_analyses:
            if execution.get("success") and execution.get("parameters"):
                cell_type = execution["parameters"].get("cell_type")
                if cell_type and cell_type != "overall":
                    relevant_cell_types.add(cell_type)
                    print(f"   📜 From function history: {cell_type}")
        
        # 4. Fallback: Use available cell types from state
        if not relevant_cell_types and state.get("available_cell_types"):
            relevant_cell_types.update(state["available_cell_types"])
            print(f"   🔄 Fallback to available cell types: {state['available_cell_types']}")
        
        # 5. Final fallback: Use initial cell types
        if not relevant_cell_types and hasattr(self, 'initial_cell_types'):
            relevant_cell_types.update(self.initial_cell_types)
            print(f"   🆘 Final fallback to initial cell types: {self.initial_cell_types}")
        
        result = list(relevant_cell_types)
        print(f"🔍 Context-based cell type extraction: {result}")
        return result

    def _build_cached_analysis_context(self, cell_types: List[str]) -> str:
        """Build analysis context from cached results for relevant cell types"""
        analysis_context = ""
        
        for cell_type in cell_types:
            print(f"🔍 Retrieving cached insights for {cell_type}...")
            insights = self.simple_cache.get_analysis_insights(cell_type)
            
            if insights and insights.get("summary"):
                analysis_context += f"\n🧬 **CACHED ANALYSIS RESULTS FOR {cell_type.upper()}**:\n"
                
                # Add enrichment insights with specific pathway names
                for analysis_name, data in insights.get("enrichment_insights", {}).items():
                    if data.get("top_terms"):
                        top_terms = data["top_terms"][:3]  # Top 3 terms
                        p_values = data.get("p_values", [])[:3]
                        
                        analysis_context += f"• **{analysis_name}**: "
                        term_details = []
                        for i, term in enumerate(top_terms):
                            p_val = f" (p={p_values[i]:.2e})" if i < len(p_values) else ""
                            term_details.append(f"{term}{p_val}")
                        analysis_context += ", ".join(term_details)
                        analysis_context += f" [{data.get('total_significant', 0)} total significant]\n"
                
                # Add DEA insights with specific gene information
                for condition, data in insights.get("dea_insights", {}).items():
                    analysis_context += f"• **DEA ({condition})**: {data.get('significant_genes', 0)} significant genes "
                    analysis_context += f"({data.get('upregulated', 0)} ↑, {data.get('downregulated', 0)} ↓)\n"
                    
                    top_genes = data.get("top_genes", [])[:3]
                    if top_genes:
                        analysis_context += f"  Top upregulated: {', '.join(top_genes)}\n"
                
                analysis_context += "\n"
        
        return analysis_context if analysis_context else "No cached analysis results found.\n"

    def _execute_final_question(self, state: ChatState) -> str:
        """
        🎯 ENHANCED: Execute final question with smart cache integration - no NLP needed!
        """
        original_question = state["execution_plan"]["original_question"]
        
        # 🎯 Method 1: Use execution context (recommended)
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        cached_context = self._build_cached_analysis_context(relevant_cell_types)
        
        # Build conversation context
        conversation_context = ""
        if state["messages"]:
            for msg in state["messages"][-5:]:  # Last 5 messages
                role = "User" if msg.type == "human" else "Assistant"
                conversation_context += f"{role}: {msg.content[:200]}...\n"
        
        # Build analysis summary from execution history
        analysis_summary = ""
        successful_analyses = [h for h in state["execution_history"] if h["success"] and h["step"].get("step_type") != "final_question"]
        
        if successful_analyses:
            analysis_summary = "ANALYSES PERFORMED IN THIS SESSION:\n"
            for h in successful_analyses:
                step_desc = h["step"]["description"]
                analysis_summary += f"✅ {step_desc}\n"
            analysis_summary += "\n"
        
        # Add hierarchical context if available
        hierarchy_context = ""
        if self.hierarchy_manager:
            lineage_summary = self.hierarchy_manager.get_lineage_summary()
            hierarchy_context = f"HIERARCHICAL CONTEXT:\n"
            hierarchy_context += f"• Total cells analyzed: {lineage_summary['total_cells']}\n"
            hierarchy_context += f"• Current cell types: {lineage_summary['unique_current_types']}\n"
            hierarchy_context += f"• Processing operations: {lineage_summary['processing_snapshots']}\n\n"
        
        # 🎯 ENHANCED: Create comprehensive final prompt with cached data
        final_prompt = f"""Based on the specific analysis results shown below, provide a comprehensive answer to the user's question.

ORIGINAL QUESTION:
{original_question}

SPECIFIC ANALYSIS RESULTS FROM CACHE:
{cached_context}

{hierarchy_context}{analysis_summary}RECENT CONVERSATION:
{conversation_context}

INSTRUCTIONS:
1. Reference the SPECIFIC pathways, genes, and statistics shown above
2. Use exact names and numbers from the cached results
3. Explain the biological significance of these specific findings
4. Connect the results directly to the user's question
5. Be quantitative and specific, not generic

Your response should cite the actual analysis results, not general knowledge."""

        try:
            # Use OpenAI to generate comprehensive response
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,  # Lower temperature for more consistent responses
                max_tokens=1500
            )
            
            final_answer = response.choices[0].message.content
            
            # 📊 Log cache usage
            if cached_context and "No cached analysis results found" not in cached_context:
                cache_cell_types = [ct for ct in relevant_cell_types if ct in cached_context]
                print(f"✅ Used cached insights from {len(cache_cell_types)} cell types: {cache_cell_types}")
            else:
                print("⚠️ No cached insights found - using execution history only")
            
            return final_answer
            
        except Exception as e:
            error_msg = f"Error generating final comprehensive answer: {e}"
            print(f"❌ {error_msg}")
            return error_msg

    def validate_processing_results(self, processed_parent: str, expected_children: List[str]) -> Dict[str, Any]:
        """Validate that process_cells discovered the expected cell types"""
        if not self.adata:
            return {"status": "error", "message": "No adata available"}
        
        current_cell_types = set(self.adata.obs["cell_type"].unique())
        found_children = []
        missing_children = []
        
        for expected_child in expected_children:
            # Check exact match or fuzzy match
            if expected_child in current_cell_types:
                found_children.append(expected_child)
            else:
                # Try fuzzy matching
                fuzzy_matches = [ct for ct in current_cell_types 
                               if expected_child.lower() in ct.lower() or ct.lower() in expected_child.lower()]
                if fuzzy_matches:
                    found_children.extend(fuzzy_matches)
                    print(f"🔄 Fuzzy match: '{expected_child}' → {fuzzy_matches}")
                else:
                    missing_children.append(expected_child)
        
        if missing_children:
            print(f"⚠️ Validation Warning: Expected children not found: {missing_children}")
            print(f"   Available cell types: {sorted(current_cell_types)}")
            
            # Try to suggest alternatives
            suggestions = []
            for missing in missing_children:
                for available in current_cell_types:
                    if self.hierarchy_manager and self.hierarchy_manager.get_cell_type_relation(missing, available).name in ["ANCESTOR", "DESCENDANT", "SIBLING"]:
                        suggestions.append(f"'{missing}' → '{available}'")
            
            return {
                "status": "partial_success",
                "message": f"Processed {processed_parent}, found {len(found_children)}/{len(expected_children)} expected children",
                "found_children": found_children,
                "missing_children": missing_children,
                "suggestions": suggestions,
                "available_types": sorted(current_cell_types)
            }
        else:
            print(f"✅ Validation Success: All expected children found: {found_children}")
            return {
                "status": "success",
                "message": f"Successfully found all expected children from {processed_parent}",
                "found_children": found_children,
                "available_types": sorted(current_cell_types)
            }

    # ========== Routing Functions ==========
    def route_from_evaluator(self, state: ChatState) -> Literal["continue", "complete"]:
        """Route from evaluator based on plan completion"""
        return "complete" if state["conversation_complete"] else "continue"

    # ========== Public API ==========
    def send_message(self, message: str) -> str:
        initial_state: ChatState = {
            "messages": [],
            "current_message": message,
            "response": "",
            "available_cell_types": [],
            "adata": None,
            "initial_plan": None,
            "execution_plan": None,
            "current_step_index": 0,
            "execution_history": [],
            "function_result": None,
            "function_name": None,
            "function_args": None,
            "function_history_summary": {},
            "missing_cell_types": [],
            "required_preprocessing": [],
            "conversation_complete": False,
            "errors": []
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        return final_state["response"]
    
    def cleanup(self):
        """Enhanced cleanup with cache stats"""
        print("🧹 Cleanup starting...")
        
        # Show cache statistics
        cache_stats = self.simple_cache.get_cache_stats()
        print(f"📊 Simple Cache Statistics:")
        print(f"   • Cache hits: {cache_stats['cache_hits']}")
        print(f"   • Cache misses: {cache_stats['cache_misses']}")
        print(f"   • Hit rate: {cache_stats['hit_rate_percent']}%")
        print(f"   • Total requests: {cache_stats['total_requests']}")
        
        # Cleanup hierarchy manager
        if hasattr(self, 'hierarchy_manager') and self.hierarchy_manager:
            self.hierarchy_manager.close()
        
        print("🧹 Cleanup completed")

    # Add a new method for cache management
    def manage_cache(self, action: str, **kwargs):
        """Cache management interface"""
        if action == "stats":
            return self.simple_cache.get_cache_stats()
        elif action == "invalidate":
            self.simple_cache.invalidate_cache(
                cell_type=kwargs.get("cell_type"),
                analysis_type=kwargs.get("analysis_type")
            )
            return "Cache invalidated"
        elif action == "clear_all":
            self.simple_cache.invalidate_cache()
            return "All cache cleared"
        else:
            return "Invalid action. Use: stats, invalidate, clear_all"

    # ========== 🧠 Debug Methods for Cache Testing ==========
    
    def debug_cache_status(self, cell_types: List[str] = None) -> Dict[str, Any]:
        """Debug method to check cache status for specific cell types"""
        if not cell_types:
            cell_types = ["T cell", "B cell"]  # Default examples
        
        debug_info = {}
        
        for cell_type in cell_types:
            debug_info[cell_type] = {
                "enrichment_files": [],
                "dea_files": [],
                "insights_available": False
            }
            
            # Check enrichment files
            enrichment_patterns = self.simple_cache._get_cache_file_patterns("enrichment", cell_type, 
                                                                            {"analyses": ["reactome", "go", "kegg", "gsea"]})
            for pattern in enrichment_patterns:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if self.simple_cache._is_file_recent(file_path):
                        debug_info[cell_type]["enrichment_files"].append(file_path)
            
            # Check DEA files
            dea_patterns = self.simple_cache._get_cache_file_patterns("dea", cell_type)
            for pattern in dea_patterns:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if self.simple_cache._is_file_recent(file_path):
                        debug_info[cell_type]["dea_files"].append(file_path)
            
            # Test insights extraction
            try:
                insights = self.simple_cache.get_analysis_insights(cell_type)
                debug_info[cell_type]["insights_available"] = bool(insights["summary"])
                debug_info[cell_type]["insights_summary"] = insights["summary"]
            except Exception as e:
                debug_info[cell_type]["insights_error"] = str(e)
        
        return debug_info

    # ========== Helper Methods ==========
    def _parse_cell_types(self, cell_type_string: str) -> List[str]:
        """Parse a cell type string that might contain multiple cell types"""
        if not cell_type_string:
            return []
        
        # Handle common separators
        separators = [',', ' and ', ' & ', ';', ' vs ', ' versus ']
        cell_types = [cell_type_string]
        
        for separator in separators:
            new_cell_types = []
            for ct in cell_types:
                if separator in ct:
                    new_cell_types.extend([part.strip() for part in ct.split(separator)])
                else:
                    new_cell_types.append(ct)
            cell_types = new_cell_types
        
        # Clean and filter
        cleaned_types = []
        for ct in cell_types:
            ct = ct.strip()
            if ct and ct not in cleaned_types:
                cleaned_types.append(ct)
        
        return cleaned_types

    def _summarize_functions(self, functions: List[Dict]) -> str:
        """Create a summary of available functions"""
        summary = []
        for func in functions:
            summary.append(f"- {func['name']}: {func['description']}")
        return "\n".join(summary)

    def _generate_execution_summary(self, state: ChatState) -> str:
        """Generate summary of multi-step execution results"""
        if not state["execution_plan"]:
            return "No execution plan was created."
            
        successful_steps = [h for h in state["execution_history"] if h["success"]]
        failed_steps = [h for h in state["execution_history"] if not h["success"]]
        
        summary = f"""**Analysis Complete!**

                    **Question**: {state['execution_plan']['original_question']}

                    **Approach**: {state['execution_plan']['plan_summary']}

                    **Results**: Completed {len(successful_steps)}/{state['execution_plan']['estimated_steps']} steps successfully.
                    """
        
        if successful_steps:
            summary += "\n**Completed Analyses:**\n"
            for h in successful_steps:
                summary += f"✅ {h['step']['description']}\n"
        
        if failed_steps:
            summary += "\n**Failed Steps:**\n"
            for h in failed_steps:
                summary += f"❌ {h['step']['description']} (Error: {h['error']})\n"
        
        # Add hierarchical insights if available
        if self.hierarchy_manager:
            lineage_summary = self.hierarchy_manager.get_lineage_summary()
            summary += f"\n**Hierarchical Analysis Summary:**\n"
            summary += f"• Analyzed {lineage_summary['total_cells']} cells across {lineage_summary['unique_current_types']} cell types\n"
            if lineage_summary['processing_snapshots'] > 0:
                summary += f"• Performed {lineage_summary['processing_snapshots']} cell type refinement operations\n"
        
        summary += "\nYou can now ask follow-up questions or request additional analyses!"
        return summary

    def _get_ai_interpretation_for_result(self, original_question: str, result: Any, message_history: List[BaseMessage]) -> str:
        """Get AI interpretation for a single result"""
        try:
            conversation_history = [{"role": msg.type, "content": msg.content} for msg in message_history]
            conversation_history.extend([
                {"role": "user", "content": original_question},
                {"role": "assistant", "content": str(result)}
            ])
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
                temperature=0.1,
                top_p=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis completed. {str(result)[:200]}..."

    # ========== Enhanced Function Wrappers with Smart Caching ==========
    
    def _wrap_enrichment_analysis(self, **kwargs):
        """🧠 CACHED: Enrichment analysis with simple intelligent caching"""
        
        # Use cache-aware wrapper
        return self.simple_cache.cache_aware_function_wrapper(
            function_name="perform_enrichment_analyses",
            analysis_type="enrichment", 
            compute_function=lambda: self._compute_enrichment_analysis(**kwargs),
            **kwargs
        )
    
    def _compute_enrichment_analysis(self, **kwargs):
        """Internal method to compute enrichment analysis (extracted for caching)"""
        if self.adata is None:
            _, _, self.adata = initial_cell_annotation()
        
        # Use hierarchical wrapper if available
        if self.analysis_wrapper:
            try:
                return self.analysis_wrapper.perform_enrichment_analyses_hierarchical(**kwargs)
            except Exception as e:
                print(f"⚠️ Hierarchical enrichment failed, falling back to standard: {e}")
        
        # Fallback to standard enrichment analysis
        analyses = kwargs.get("analyses")
        if not analyses and "analysis_type" in kwargs:
            analysis_type = kwargs["analysis_type"].lower()
            analyses = [analysis_type] if analysis_type in ["gsea", "go", "kegg", "reactome"] else None
        
        return perform_enrichment_analyses(
            self.adata,
            cell_type=kwargs.get("cell_type"),
            analyses=analyses,
            logfc_threshold=kwargs.get("logfc_threshold", 1.0),
            pval_threshold=kwargs.get("pval_threshold", 0.05),
            top_n_terms=kwargs.get("top_n_terms", 10),
        )

    def _wrap_dea_analysis(self, **kwargs):
        """🧠 CACHED: DEA analysis with simple intelligent caching"""
        
        return self.simple_cache.cache_aware_function_wrapper(
            function_name="dea_split_by_condition",
            analysis_type="dea",
            compute_function=lambda: self._compute_dea_analysis(**kwargs),
            **kwargs
        )
    
    def _compute_dea_analysis(self, **kwargs):
        """Internal method to compute DEA analysis (extracted for caching)"""
        cell_type = kwargs.get("cell_type")
        
        # Use hierarchical wrapper if available
        if self.analysis_wrapper:
            try:
                return self.analysis_wrapper.dea_split_by_condition_hierarchical(cell_type)
            except Exception as e:
                print(f"⚠️ Hierarchical DEA failed, falling back to standard: {e}")
        
        # Fallback to standard DEA
        dea_split_by_condition(self.adata, cell_type)
        return {"summary": f"DEA complete for {cell_type}. Results saved to local files."}

    def _wrap_process_cells(self, **kwargs):
        """🧠 CACHED: Process cells with simple intelligent caching"""
        
        # Note: Process cells is more complex because it modifies the adata state
        # For now, we'll check cache but always run the function to ensure state consistency
        
        cell_type = kwargs.get("cell_type")
        
        # Check if we have cached results
        cached_result, is_hit = self.simple_cache.check_cache(
            "process_cells", kwargs, "process_cells"
        )
        
        if is_hit:
            print(f"🎯 Found cached process_cells result for {cell_type}")
            # Still run the actual process_cells to update adata state
            # but we know the result should be similar
        
        # Always run process_cells to ensure adata state is correct
        result = self._compute_process_cells(**kwargs)
        
        return result
    
    def _compute_process_cells(self, **kwargs):
        """Internal method to compute process_cells (extracted for caching)"""
        cell_type = kwargs.get("cell_type")
        resolution = kwargs.get("resolution")
        
        print(f"🔄 Processing cell type: {cell_type}")
        result = process_cells(self.adata, cell_type, resolution)
        
        if isinstance(result, dict) and "status" in result:
            if result["status"] == "leaf_node":
                return f"✅ '{cell_type}' is a leaf node with no subtypes."
            elif result["status"] == "no_cells_found":
                return f"❌ No cells found with type '{cell_type}'."
            elif result["status"] == "insufficient_markers":
                return f"⚠️ Insufficient markers for '{cell_type}' subtypes."
        
        return result

    # ========== Function Wrappers ==========
    def _wrap_visualization(self, func):
        def wrapper(**kwargs):
            return func(**kwargs)
        return wrapper

    def _wrap_compare_cells(self, **kwargs):
        cell_type = kwargs.get("cell_type")
        
        # Use hierarchical wrapper if available
        if self.analysis_wrapper:
            try:
                return self.analysis_wrapper.compare_cell_count_hierarchical(cell_type)
            except Exception as e:
                print(f"⚠️ Hierarchical comparison failed, falling back to standard: {e}")
        
        # Fallback to standard comparison
        return compare_cell_count(self.adata, cell_type)

    def _wrap_conversational_response(self, **kwargs):
        """Handle conversational responses"""
        response_type = kwargs.get("response_type", "general")
        
        if response_type == "greeting":
            return "Hello! I'm your single-cell RNA-seq analysis assistant with hierarchical cell type management. I can help you analyze your data, create visualizations, and interpret results with intelligent cell type resolution. What would you like to explore?"
        elif response_type == "explanation":
            return "I'm here to help you understand your single-cell data using advanced hierarchical cell type management. I can automatically resolve ancestor/descendant relationships and suggest optimal analysis paths."
        elif response_type == "clarification":
            return "Could you please clarify what specific analysis or visualization you'd like me to perform? I can work with both specific cell subtypes and broader cell type categories."
        else:
            return "I'm ready to help with your single-cell RNA-seq analysis using intelligent cell type hierarchy management. What would you like to explore?"

    def _wrap_validate_processing_results(self, **kwargs):
        """Wrapper for validation function"""
        return self.validate_processing_results(
            kwargs.get("processed_parent"),
            kwargs.get("expected_children", [])
        )

    def _extract_cell_types_from_result(self, result: Any) -> List[str]:
        """Extract cell types from annotation result with improved precision"""
        cell_types = []
        if not isinstance(result, str):
            return cell_types
        
        text = result
        
        # PRIORITY 1: Extract from dictionary format (most reliable)
        dict_pattern = r"group_to_cell_type\s*=\s*\{([^}]+)\}"
        dict_match = re.search(dict_pattern, text)
        if dict_match:
            dict_content = dict_match.group(1)
            # Extract values from the dictionary (cell types are the values)
            value_pattern = r"'[^']*':\s*'([^']+)'"
            values = re.findall(value_pattern, dict_content)
            cell_types.extend(values)
            print(f"✅ Extracted {len(values)} cell types from dictionary format")
            
            # If we found cell types in dictionary format, return them (most reliable)
            if values:
                return list(set(values))  # Remove duplicates
        
        # PRIORITY 2: Extract from markdown headers (secondary)
        header_matches = re.findall(r"###\s*Cluster\s*\d+:\s*([^:\n]+)", text)
        cell_types.extend([match.strip() for match in header_matches])
        
        # PRIORITY 3: Only look for very specific patterns if above methods fail
        if not cell_types:
            # More conservative patterns
            specific_patterns = [
                r"\b(T cell)s?\b",
                r"\b(B cell)s?\b", 
                r"\b(Natural killer cell)s?\b",
                r"\b(Mononuclear phagocyte)s?\b",
                r"\b(Plasmacytoid dendritic cell)s?\b",
                # Add other known cell types as needed
            ]
            
            for pattern in specific_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                cell_types.extend(matches)
        
        # Clean and deduplicate
        cleaned_types = []
        for cell_type in cell_types:
            cleaned = cell_type.strip()
            if (len(cleaned) > 2 and 
                cleaned not in cleaned_types and
                not self._is_invalid_cell_type(cleaned)):
                cleaned_types.append(cleaned)
        
        return cleaned_types
    
    def _is_invalid_cell_type(self, cell_type: str) -> bool:
        """Check if a cell type should be filtered out"""
        invalid_patterns = [
            r'^\d+',  # Pure numbers
            r'^[a-z]+',  # All lowercase (likely not a proper cell type name)
            r'^[A-Z]',  # Single capital letter
            r'cluster',  # Contains 'cluster'
            r'sample',   # Contains 'sample'  
            r'condition', # Contains 'condition'
            r'group',    # Contains 'group'
            r'type',     # Just 'type'
            r'analysis', # Contains 'analysis'
        ]
        
        cell_type_lower = cell_type.lower()
        for pattern in invalid_patterns:
            if re.search(pattern, cell_type_lower):
                return True
        
        return False


# ==============================================================================
# Compatibility Wrapper
# ==============================================================================
class ChatBot(MultiAgentChatBot):
    """Compatibility wrapper for Django integration"""
    
    def __del__(self):
        """Ensure cleanup when object is destroyed"""
        try:
            self.cleanup()
        except:
            pass