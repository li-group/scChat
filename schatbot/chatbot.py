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
    
    # 🆕 CRITIC AGENT FIELDS
    critic_iterations: int
    critic_feedback_history: List[Dict[str, Any]]
    plan_revision_history: List[Dict[str, Any]]
    original_execution_complete: bool
    cumulative_analysis_results: Dict[str, Any]
    impossible_request_detected: bool
    degradation_strategy: Optional[Dict[str, Any]]
    error_recovery_strategy: Optional[Dict[str, Any]]
    revision_applied: bool


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


@dataclass
class CriticEvaluation:
    """Structured critic evaluation result"""
    relevance_score: float
    completeness_score: float  
    missing_analyses: List[str]
    recommendations: List[str]
    needs_revision: bool
    reasoning: str
    evaluation_type: str


class CriticLoopManager:
    """Manages critic agent iteration cycles and prevents infinite loops"""
    MAX_ITERATIONS = 3
    
    @staticmethod
    def should_continue_iteration(state: ChatState) -> bool:
        return (
            state.get("critic_iterations", 0) < CriticLoopManager.MAX_ITERATIONS and
            not state.get("impossible_request_detected", False) and
            state.get("critic_feedback_history", []) and
            state["critic_feedback_history"][-1]["needs_revision"]
        )
    
    @staticmethod
    def initialize_critic_state(state: ChatState) -> ChatState:
        state.setdefault("critic_iterations", 0)
        state.setdefault("critic_feedback_history", [])
        state.setdefault("plan_revision_history", [])
        state.setdefault("original_execution_complete", False)
        state.setdefault("cumulative_analysis_results", {})
        state.setdefault("impossible_request_detected", False)
        state.setdefault("degradation_strategy", None)
        state.setdefault("error_recovery_strategy", None)
        state.setdefault("revision_applied", False)
        return state


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
        
        # 🧬 UNIFIED: Initialize cell type extractor
        self.cell_type_extractor = None
        self._initialize_cell_type_extractor()
        
        # 🆕 Initialize critic loop manager
        self.critic_loop_manager = CriticLoopManager()
        
        # Create LangGraph workflow with critic agent
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
    
    def _initialize_cell_type_extractor(self):
        """🧬 UNIFIED: Initialize centralized cell type extractor"""
        if self.adata is not None:
            self.cell_type_extractor = CellTypeExtractor(
                hierarchy_manager=self.hierarchy_manager,
                adata=self.adata
            )
            # Add historical function access
            self.cell_type_extractor._get_historical_cell_types = self._get_historical_cell_types_for_extractor
            print("✅ Unified cell type extractor initialized")
        else:
            print("⚠️ Cannot initialize cell type extractor without adata")
    
    def _get_historical_cell_types_for_extractor(self) -> List[str]:
        """Get cell types from historical function executions for the extractor"""
        cell_types = set()
        
        # Get recent analyses from history manager
        recent_analyses = self.history_manager.get_recent_executions("perform_enrichment_analyses", limit=10)
        recent_analyses.extend(self.history_manager.get_recent_executions("dea_split_by_condition", limit=10))
        recent_analyses.extend(self.history_manager.get_recent_executions("process_cells", limit=10))
        
        for execution in recent_analyses:
            if execution.get("success") and execution.get("parameters"):
                cell_type = execution["parameters"].get("cell_type")
                if cell_type and cell_type != "overall":
                    cell_types.add(cell_type)
        
        return list(cell_types)

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
                "description": "Run enrichment analyses (reactome, go, kegg, gsea) on DE genes for a cell type. Use for pathway analysis. This function enables the agent to answer questions related to pathway analysis.",
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
                "description": "Perform differential expression analysis (DEA) split by condition. Use when comparing conditions. This function enables the agent to answer questions related to differential expression genes.",
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
        """Create enhanced workflow with critic agent system"""
        workflow = StateGraph(ChatState)
        
        # Existing nodes
        workflow.add_node("input_processor", self.input_processor_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("status_checker", self.status_checker_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("evaluator", self.evaluator_node)
        workflow.add_node("response_generator", self.response_generator_node)
        workflow.add_node("critic_agent", self.critic_agent_node)
        workflow.add_node("planner_reviser", self.planner_reviser_node)
        workflow.add_node("impossible_handler", self.impossible_handler_node)
        
        # Set entry point
        workflow.set_entry_point("input_processor")
        
        # Original workflow connections
        workflow.add_edge("input_processor", "planner")
        workflow.add_edge("planner", "status_checker")
        workflow.add_edge("status_checker", "executor")
        workflow.add_edge("executor", "evaluator")
        
        # Routing from evaluator
        workflow.add_conditional_edges(
            "evaluator", 
            self.route_from_evaluator,
            {
                "continue": "executor",
                "to_critic": "critic_agent"  
            }
        )
        
        workflow.add_conditional_edges(
            "critic_agent",
            self.route_from_critic,
            {
                "revise": "planner_reviser",
                "complete": "response_generator",
                "impossible": "impossible_handler"
            }
        )
        
        workflow.add_edge("planner_reviser", "status_checker")
        workflow.add_edge("impossible_handler", "response_generator")
        
        # Final response generation
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

    # ========== Cache Integration Methods ==========
    
    def _get_relevant_cell_types_from_context(self, state: ChatState) -> List[str]:
        """🧬 UNIFIED: Extract relevant cell types from execution context"""
        if self.cell_type_extractor:
            return self.cell_type_extractor.extract_from_execution_context(state, include_history=True)
        else:
            # Fallback if extractor not initialized
            print("⚠️ Cell type extractor not initialized, using state fallback")
            return state.get("available_cell_types", [])

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
        original_question = state["execution_plan"]["original_question"]
        
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
    def route_from_evaluator(self, state: ChatState) -> Literal["continue", "to_critic"]:
        """Enhanced routing from evaluator - go to critic when execution is complete"""
        
        if state["conversation_complete"]:
            return "to_critic"  # Route to critic agent instead of direct response
        else:
            return "continue"

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
            "errors": [],
            
            # 🆕 Critic agent fields
            "critic_iterations": 0,
            "critic_feedback_history": [],
            "plan_revision_history": [],
            "original_execution_complete": False,
            "cumulative_analysis_results": {},
            "impossible_request_detected": False,
            "degradation_strategy": None,
            "error_recovery_strategy": None,
            "revision_applied": False
        }
        
        # Run the enhanced workflow with critic agent
        final_state = self.workflow.invoke(initial_state)
        
        # 🆕 Log critic agent statistics
        self._log_critic_statistics(final_state)
        
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
        
        # Cleanup cell type extractor
        if hasattr(self, 'cell_type_extractor') and self.cell_type_extractor:
            print("🧬 Cell type extractor cleaned up")
        
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

    # ========== Helper Methods ==========
    def _parse_cell_types(self, cell_type_string: str) -> List[str]:
        """🧬 UNIFIED: Parse a cell type string that might contain multiple cell types"""
        if self.cell_type_extractor:
            return self.cell_type_extractor.parse_multi_cell_type_string(cell_type_string)
        else:
            # Fallback if extractor not initialized
            print("⚠️ Cell type extractor not initialized, using simple fallback")
            return [cell_type_string] if cell_type_string else []

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
        """🧬 UNIFIED: Extract cell types from annotation result"""
        if self.cell_type_extractor:
            return self.cell_type_extractor.extract_from_annotation_result(result)
        else:
            # Fallback if extractor not initialized
            print("⚠️ Cell type extractor not initialized, using fallback")
            return []
    
    def _is_invalid_cell_type(self, cell_type: str) -> bool:
        """🧬 UNIFIED: Check if a cell type should be filtered out (delegated to extractor)"""
        if self.cell_type_extractor:
            return not self.cell_type_extractor._is_valid_cell_type(cell_type)
        else:
            # Fallback validation
            return not cell_type or len(cell_type.strip()) <= 2

    # ========== 🆕 NEW CRITIC AGENT NODE METHODS ==========

    def critic_agent_node(self, state: ChatState) -> ChatState:
        
        # Initialize critic state
        state = CriticLoopManager.initialize_critic_state(state)
        
        # Extract information for evaluation
        original_question = state["execution_plan"]["original_question"]
        final_response = state["response"]
        execution_history = state["execution_history"]
        
        # Extract response content
        try:
            response_data = json.loads(final_response)
            response_content = response_data.get("response", "")
        except:
            response_content = final_response
        
        print(f"🔍 Critic Agent - Iteration {state['critic_iterations'] + 1}: Evaluating response quality...")
        
        # Perform comprehensive evaluation
        evaluation = self._evaluate_response_quality(
            original_question, 
            response_content, 
            execution_history,
            state
        )
        
        # Update state
        state["critic_iterations"] += 1
        state["critic_feedback_history"].append(evaluation)
        state["original_execution_complete"] = True
        
        # Accumulate analysis results for context
        self._accumulate_analysis_results(state, evaluation)
        
        print(f"✅ Critic evaluation complete:")
        print(f"   • Relevance: {evaluation['relevance_score']:.2f}")
        print(f"   • Completeness: {evaluation['completeness_score']:.2f}")
        print(f"   • Needs revision: {evaluation['needs_revision']}")
        print(f"   • Missing analyses: {evaluation['missing_analyses']}")
        
        return state

    def planner_reviser_node(self, state: ChatState) -> ChatState:
        """🆕 REUSABLE plan revision node with comprehensive error handling"""
        
        iteration = state.get("critic_iterations", 0)
        latest_feedback = state["critic_feedback_history"][-1]
        
        print(f"🔄 Plan Revision - Iteration {iteration}")
        
        # Step 1: Detect impossible requests (CRITICAL for loop prevention)
        impossible_patterns = self._detect_impossible_requests(state)
        
        if self._has_impossible_patterns(impossible_patterns):
            print("🚫 Impossible requests detected - flagging for graceful degradation")
            state["impossible_request_detected"] = True
            state["degradation_strategy"] = self._handle_impossible_requests(impossible_patterns, state)
            return state
        
        # Step 2: Handle execution errors
        error_recovery = {}
        if state.get("execution_history"):
            error_recovery = self._handle_execution_errors(state["execution_history"], latest_feedback)
            state["error_recovery_strategy"] = error_recovery
        
        # Step 3: Handle content gaps
        content_revision = self._handle_content_gaps(latest_feedback, state["execution_plan"])
        
        # Step 4: Generate revised plan
        revised_plan = self._generate_revised_plan(
            state["execution_plan"],
            content_revision,
            error_recovery,
            iteration
        )
        
        # Step 5: Update state for status checker
        state["execution_plan"] = revised_plan
        state["current_step_index"] = 0  # Reset execution index
        state["revision_applied"] = True
        
        # Record revision history
        state["plan_revision_history"].append({
            "iteration": iteration,
            "reason": latest_feedback["reasoning"],
            "content_changes": content_revision,
            "error_recovery": error_recovery
        })
        
        print(f"✅ Plan revised - Added {len(content_revision.get('add_steps', []))} steps")
        
        return state

    def impossible_handler_node(self, state: ChatState) -> ChatState:
        """🆕 Handle impossible requests with graceful degradation"""
        
        degradation_strategy = state.get("degradation_strategy", {})
        
        print("🚫 Handling impossible request with graceful degradation...")
        
        # Generate explanation response
        explanation = self._generate_impossible_request_explanation(
            state["execution_plan"]["original_question"],
            degradation_strategy,
            state
        )
        
        # Update response
        state["response"] = json.dumps({
            "response": explanation,
            "response_type": "impossible_request_handled"
        })
        
        state["conversation_complete"] = True
        
        return state

    def route_from_critic(self, state: ChatState) -> Literal["revise", "complete", "impossible"]:
        """Enhanced routing from critic agent with impossible request detection"""
        
        # Check for impossible requests first
        if state.get("impossible_request_detected"):
            print("🚫 Routing to impossible request handler")
            return "impossible"
        
        # Check iteration limit
        if state["critic_iterations"] >= CriticLoopManager.MAX_ITERATIONS:
            print(f"🔄 Maximum iterations ({CriticLoopManager.MAX_ITERATIONS}) reached - completing")
            return "complete"
        
        # Check if revision is needed
        if CriticLoopManager.should_continue_iteration(state):
            print(f"🔄 Critic recommends revision - starting iteration {state['critic_iterations'] + 1}")
            return "revise"
        
        print("✅ Critic satisfied with response - completing")
        return "complete"

    # ========== CRITIC EVALUATION METHODS ==========
    def _evaluate_response_quality(self, question: str, response: str, 
                                execution_history: List, state: ChatState) -> Dict:
        """Enhanced critic evaluation that is cache-aware"""
        
        # Prepare execution summary
        execution_summary = self._summarize_execution_history(execution_history)
        
        # Get available cell types for context
        available_cell_types = ", ".join(state.get("available_cell_types", []))
        
        # Check for repeated failures
        repeated_failures = self._detect_repeated_failures(state)
        
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        cache_analysis_summary = self._get_cache_analysis_summary(relevant_cell_types)
        
        comprehensive_analysis_context = self._build_comprehensive_analysis_context(state, execution_history)
        
        critic_prompt = f"""
                        You are a scientific analysis critic evaluating single-cell RNA-seq analysis results.

                        ORIGINAL QUESTION:
                        {question}

                        GENERATED RESPONSE:
                        {response}

                        ANALYSES PERFORMED IN CURRENT SESSION:
                        {execution_summary}

                        🆕 CACHED ANALYSIS RESULTS AVAILABLE:
                        {cache_analysis_summary}

                        🆕 COMPREHENSIVE ANALYSIS CONTEXT (Current + Cache + History):
                        {comprehensive_analysis_context}

                        AVAILABLE CELL TYPES IN DATASET:
                        {available_cell_types}

                        Available functions:
                        {self._summarize_functions(self.function_descriptions)}

                        CONTEXT:
                        - Iteration: {state.get('critic_iterations', 0) + 1}/3
                        - Previous failures: {repeated_failures}

                        CACHE AWARENESS:
                        - If the question asks for a summary/results of analyses (like GSEA, DEA), check if those results are available in the CACHED ANALYSIS RESULTS section
                        - If cached results exist for the requested analysis and cell type, then the analysis HAS BEEN COMPLETED and should NOT be marked as missing
                        - Only mark analyses as missing if they are truly not available anywhere (current session, cache, or history)

                        Evaluate the response on these criteria:

                        1. RELEVANCE (0.0-1.0): Does the response directly address what was asked?
                        - Does it answer the specific question?
                        - Are the right cell types being analyzed?

                        2. COMPLETENESS (0.0-1.0): Are all parts of the question covered?
                        - If asking for analysis results/summary, are cached results being used appropriately?
                        - If comparing multiple cell types, were all compared?
                        - Are visualizations included when appropriate?

                        3. SCIENTIFIC RIGOR: Are conclusions supported by proper analysis?
                        - Are the analysis methods appropriate?
                        - Is statistical information provided when relevant?

                        4. MISSING ELEMENTS: What specific analyses are missing?
                        - Check cached results first before marking as missing
                        - Only suggest new analyses if they are truly unavailable
                        - Be specific about function names and parameters

                        5. ERROR DETECTION: Are there any execution errors that need fixing?

                        IMPORTANT CONSTRAINTS:
                        - If cached results exist for the requested analysis type and cell type, DO NOT mark it as missing
                        - Only suggest analyses for cell types that exist in the available cell types list
                        - Don't suggest impossible analyses (e.g., cell types not in dataset)
                        - Be specific about function names: use "perform_enrichment_analyses", "process_cells", "display_processed_umap", etc.

                        Respond in JSON format:
                        {{
                            "relevance_score": 0.0-1.0,
                            "completeness_score": 0.0-1.0,
                            "needs_revision": true/false,
                            "missing_analyses": ["specific analysis with function name and cell type (The function name should be exactly same as it is in the available function)"],
                            "recommendations": ["actionable advice for improvement"],
                            "reasoning": "Detailed explanation of evaluation including cache awareness",
                            "evaluation_type": "content_gap|execution_error|quality_issue|cache_satisfied",
                            "impossible_requests": ["requests that cannot be fulfilled with available data"],
                            "cache_utilization": "How well the response utilized available cached results"
                        }}
                        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=[{"role": "user", "content": critic_prompt}],
                response_format={"type": "json_object"}
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            evaluation = self._sanitize_critic_evaluation_with_cache_awareness(evaluation, state)
            
            return evaluation
            
        except Exception as e:
            print(f"❌ Critic evaluation failed: {e}")
            # Fallback evaluation
            return {
                "relevance_score": 0.8,
                "completeness_score": 0.8,
                "needs_revision": False,
                "missing_analyses": [],
                "recommendations": [],
                "reasoning": f"Critic evaluation failed: {e}",
                "evaluation_type": "error",
                "impossible_requests": [],
                "cache_utilization": "Error in evaluation"
            }

    def _get_cache_analysis_summary(self, relevant_cell_types: List[str]) -> str:
        """🆕 NEW: Get summary of what analyses are available in cache"""
        
        cache_summary = "CACHED ANALYSIS AVAILABILITY:\n"
        
        for cell_type in relevant_cell_types:
            cache_summary += f"\n🧬 {cell_type.upper()}:\n"
            
            # Check for enrichment analyses in cache
            enrichment_available = []
            enrichment_patterns = self.simple_cache._get_cache_file_patterns("enrichment", cell_type, 
                                                                            {"analyses": ["reactome", "go", "kegg", "gsea"]})
            for pattern in enrichment_patterns:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if self.simple_cache._is_file_recent(file_path):
                        analysis_name = self.simple_cache._extract_analysis_name_from_path(file_path)
                        enrichment_available.append(analysis_name)
            
            if enrichment_available:
                cache_summary += f"  ✅ ENRICHMENT ANALYSES: {', '.join(set(enrichment_available))}\n"
            else:
                cache_summary += f"  ❌ ENRICHMENT ANALYSES: None available\n"
            
            # Check for DEA analyses in cache
            dea_available = []
            dea_patterns = self.simple_cache._get_cache_file_patterns("dea", cell_type)
            for pattern in dea_patterns:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    if self.simple_cache._is_file_recent(file_path):
                        condition = self.simple_cache._extract_condition_from_path(file_path)
                        dea_available.append(condition)
            
            if dea_available:
                cache_summary += f"  ✅ DEA ANALYSES: {', '.join(dea_available)}\n"
            else:
                cache_summary += f"  ❌ DEA ANALYSES: None available\n"
        
        return cache_summary

    def _build_comprehensive_analysis_context(self, state: ChatState, execution_history: List) -> str:
        """🆕 NEW: Build comprehensive context from current session + cache + function history"""
        
        context = "COMPREHENSIVE ANALYSIS STATUS:\n\n"
        
        # 1. Current session analyses
        current_session_analyses = []
        for execution in execution_history:
            if execution["success"]:
                func_name = execution["step"]["function_name"]
                if func_name in ["perform_enrichment_analyses", "dea_split_by_condition", "process_cells"]:
                    cell_type = execution["step"]["parameters"].get("cell_type", "unknown")
                    current_session_analyses.append(f"{func_name}({cell_type})")
        
        if current_session_analyses:
            context += "📋 CURRENT SESSION:\n"
            for analysis in current_session_analyses:
                context += f"  ✅ {analysis}\n"
        else:
            context += "📋 CURRENT SESSION: No analyses performed\n"
        
        # 2. Function history (previous sessions)
        recent_analyses = self.history_manager.get_recent_executions("perform_enrichment_analyses", limit=5)
        recent_analyses.extend(self.history_manager.get_recent_executions("dea_split_by_condition", limit=5))
        recent_analyses.extend(self.history_manager.get_recent_executions("process_cells", limit=5))
        
        if recent_analyses:
            context += "\n📜 PREVIOUS SESSIONS (Last 5 of each type):\n"
            for execution in recent_analyses:
                if execution.get("success"):
                    func_name = execution["function_name"]
                    cell_type = execution["parameters"].get("cell_type", "unknown")
                    timestamp = execution.get("timestamp", "unknown")
                    context += f"  ✅ {func_name}({cell_type}) - {timestamp}\n"
        else:
            context += "\n📜 PREVIOUS SESSIONS: No recent analyses found\n"
        
        # 3. Cache status summary
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        cache_stats = {}
        for cell_type in relevant_cell_types:
            insights = self.simple_cache.get_analysis_insights(cell_type)
            cache_stats[cell_type] = {
                "enrichment_analyses": len(insights.get("enrichment_insights", {})),
                "dea_analyses": len(insights.get("dea_insights", {}))
            }
        
        context += "\n💾 CACHE STATUS:\n"
        for cell_type, stats in cache_stats.items():
            context += f"  🧬 {cell_type}: {stats['enrichment_analyses']} enrichment + {stats['dea_analyses']} DEA results cached\n"
        
        return context

    def _sanitize_critic_evaluation_with_cache_awareness(self, evaluation: Dict, state: ChatState) -> Dict:
        
        available_cell_types = set(state.get("available_cell_types", []))
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        
        # Filter out analyses that are already available in cache
        sanitized_missing = []
        cache_satisfied = []
        
        for missing in evaluation.get("missing_analyses", []):
            cell_type = self._extract_cell_type_from_analysis(missing)
            
            if cell_type and cell_type in relevant_cell_types:
                # Check if this analysis is already available in cache
                is_cached = self._is_analysis_cached(missing, cell_type)
                
                if is_cached:
                    cache_satisfied.append(missing)
                    print(f"🎯 Analysis already cached, removing from missing: {missing}")
                else:
                    # Not cached, keep as missing if cell type is available
                    if cell_type in available_cell_types:
                        sanitized_missing.append(missing)
                    else:
                        print(f"🚫 Cell type not available: {missing}")
            else:
                # Keep original logic for other cases
                if cell_type and cell_type not in available_cell_types:
                    if self.hierarchy_manager and self.hierarchy_manager.is_valid_cell_type(cell_type):
                        sanitized_missing.append(missing)
                    else:
                        print(f"🚫 Filtered impossible request: {missing}")
                else:
                    sanitized_missing.append(missing)
        
        evaluation["missing_analyses"] = sanitized_missing
        evaluation["cache_satisfied_analyses"] = cache_satisfied
        
        # If no valid missing analyses remain and cache satisfied some, don't revise
        if not sanitized_missing and cache_satisfied:
            evaluation["needs_revision"] = False
            evaluation["reasoning"] += f" (Requested analyses are available in cache: {cache_satisfied})"
            evaluation["evaluation_type"] = "cache_satisfied"
        
        return evaluation

    def _is_analysis_cached(self, analysis_description: str, cell_type: str) -> bool:
        """🆕 NEW: Check if a specific analysis is available in cache"""
        
        analysis_lower = analysis_description.lower()
        
        if "enrichment" in analysis_lower or "gsea" in analysis_lower or "pathway" in analysis_lower:
            # Check enrichment cache
            insights = self.simple_cache.get_analysis_insights(cell_type, ["enrichment"])
            return len(insights.get("enrichment_insights", {})) > 0
        
        elif "dea" in analysis_lower or "differential" in analysis_lower:
            # Check DEA cache
            insights = self.simple_cache.get_analysis_insights(cell_type, ["dea"])
            return len(insights.get("dea_insights", {})) > 0
        
        elif "process" in analysis_lower:
            # Check if processed results exist
            available_types = set(self.adata.obs["cell_type"].unique())
            return cell_type in available_types
        
        return False

    # ========== IMPOSSIBLE REQUEST DETECTION AND HANDLING ==========

    def _detect_impossible_requests(self, state: ChatState) -> Dict:
        """Enhanced impossible request detection with cache awareness"""
        
        impossible_patterns = {
            "repeated_cell_type_failures": [],
            "hierarchical_dead_ends": [],
            "cache_misunderstanding": []  
        }
        
        # Original logic...
        if state.get("critic_feedback_history"):
            cell_type_failures = {}
            cache_available_but_marked_missing = []
            
            for iteration in state["critic_feedback_history"]:
                for missing in iteration.get("missing_analyses", []):
                    cell_type = self._extract_cell_type_from_analysis(missing)
                    if cell_type:
                        # 🆕 NEW: Check if this is actually available in cache
                        if self._is_analysis_cached(missing, cell_type):
                            cache_available_but_marked_missing.append({
                                "analysis": missing,
                                "cell_type": cell_type,
                                "reason": "Available in cache but marked as missing"
                            })
                        else:
                            cell_type_failures[cell_type] = cell_type_failures.get(cell_type, 0) + 1
            
            # 🆕 NEW: If analyses are repeatedly marked missing but are in cache, it's impossible
            if cache_available_but_marked_missing:
                impossible_patterns["cache_misunderstanding"] = cache_available_but_marked_missing
            
            # Rest of original logic for repeated failures...
            for cell_type, failure_count in cell_type_failures.items():
                if failure_count >= 2:
                    alternatives = []
                    if self.hierarchy_manager and hasattr(self.adata, 'obs'):
                        alternatives = self.hierarchy_manager._suggest_similar_types(
                            cell_type, set(self.adata.obs["cell_type"].unique())
                        )
                    
                    impossible_patterns["repeated_cell_type_failures"].append({
                        "cell_type": cell_type,
                        "failure_count": failure_count,
                        "alternatives": alternatives
                    })
        
        # Rest of original logic...
        if self.hierarchy_manager:
            available_types = set(state.get("available_cell_types", []))
            requested_types = self._extract_all_requested_cell_types(state)
            
            for req_type in requested_types:
                if not self.hierarchy_manager.is_valid_cell_type(req_type):
                    impossible_patterns["hierarchical_dead_ends"].append({
                        "requested_type": req_type,
                        "reason": "Not in cell type ontology",
                        "suggestions": self.hierarchy_manager._suggest_similar_types(req_type, available_types)
                    })
        
        return impossible_patterns

    def _has_impossible_patterns(self, impossible_patterns: Dict) -> bool:
        """Check if any impossible patterns were detected"""
        return (
            bool(impossible_patterns["repeated_cell_type_failures"]) or
            bool(impossible_patterns["hierarchical_dead_ends"]) or
            bool(impossible_patterns["resource_limitations"]) or
            bool(impossible_patterns["cache_misunderstanding"])
        )

    def _handle_impossible_requests(self, impossible_patterns: Dict, state: ChatState) -> Dict:
        """Generate graceful degradation strategy for impossible requests"""
        
        degradation_strategy = {
            "acknowledge_limitations": [],
            "propose_alternatives": [],
            "modify_question_scope": []
        }
        
        # Handle repeated cell type failures
        for failure in impossible_patterns["repeated_cell_type_failures"]:
            cell_type = failure["cell_type"]
            alternatives = failure["alternatives"]
            
            if alternatives:
                degradation_strategy["propose_alternatives"].append({
                    "impossible_request": f"Analysis of {cell_type}",
                    "alternatives": alternatives,
                    "explanation": f"{cell_type} not available in dataset. Suggesting related cell types."
                })
            else:
                degradation_strategy["acknowledge_limitations"].append({
                    "limitation": f"{cell_type} analysis not possible",
                    "reason": "Cell type not present in dataset and no suitable alternatives found"
                })
        
        # Handle hierarchical dead ends
        for dead_end in impossible_patterns["hierarchical_dead_ends"]:
            degradation_strategy["modify_question_scope"].append({
                "original_scope": dead_end["requested_type"],
                "suggested_scope": dead_end["suggestions"],
                "modification_reason": dead_end["reason"]
            })
        
        return degradation_strategy

    def _generate_impossible_request_explanation(self, original_question: str, 
                                                degradation_strategy: Dict, 
                                                state: ChatState) -> str:
        """Generate explanation for impossible requests with alternatives"""
        
        explanation = f"I understand you're asking: **{original_question}**\n\n"
        explanation += "After analyzing your data and attempting multiple approaches, I've identified some limitations:\n\n"
        
        # Handle acknowledged limitations
        for limitation in degradation_strategy.get("acknowledge_limitations", []):
            explanation += f"❌ **{limitation['limitation']}**: {limitation['reason']}\n"
        
        # Handle proposed alternatives
        if degradation_strategy.get("propose_alternatives"):
            explanation += "\n**However, I can offer these alternatives:**\n"
            for alternative in degradation_strategy["propose_alternatives"]:
                explanation += f"✅ Instead of {alternative['impossible_request']}, "
                explanation += f"I can analyze: {', '.join(alternative['alternatives'])}\n"
                explanation += f"   Reason: {alternative['explanation']}\n"
        
        # Handle scope modifications
        if degradation_strategy.get("modify_question_scope"):
            explanation += "\n**Suggested modifications to your question:**\n"
            for modification in degradation_strategy["modify_question_scope"]:
                explanation += f"🔄 Instead of '{modification['original_scope']}', "
                explanation += f"consider: {', '.join(modification['suggested_scope'])}\n"
                explanation += f"   Reason: {modification['modification_reason']}\n"
        
        explanation += f"\n**Available cell types in your dataset:** {', '.join(state.get('available_cell_types', []))}"
        explanation += "\n\nWould you like me to proceed with any of the suggested alternatives?"
        
        return explanation

    # ========== 🆕 CONTENT GAP AND ERROR HANDLING ==========

    def _handle_content_gaps(self, critic_feedback: Dict, current_plan: Dict) -> Dict:
        """Handle missing analyses or redundant steps"""
        
        missing_analyses = critic_feedback.get("missing_analyses", [])
        
        revision_actions = {
            "add_steps": [],
            "remove_steps": [],
            "modify_steps": []
        }
        
        # Add missing analyses
        for missing in missing_analyses:
            missing_lower = missing.lower()
            
            # Extract cell type from analysis description
            cell_type = self._extract_cell_type_from_analysis(missing)
            
            if "compare" in missing_lower and "count" in missing_lower:
                revision_actions["add_steps"].append({
                    "step_type": "analysis",
                    "function_name": "compare_cell_counts",
                    "parameters": {"cell_type": cell_type} if cell_type else {},
                    "description": f"Address critic feedback: {missing}",
                    "target_cell_type": cell_type
                })
            
            elif "enrichment" in missing_lower:
                revision_actions["add_steps"].append({
                    "step_type": "analysis",
                    "function_name": "perform_enrichment_analyses",
                    "parameters": {"cell_type": cell_type} if cell_type else {},
                    "description": f"Address critic feedback: {missing}",
                    "target_cell_type": cell_type
                })
            
            elif "process" in missing_lower or "subtype" in missing_lower:
                revision_actions["add_steps"].append({
                    "step_type": "analysis",
                    "function_name": "process_cells",
                    "parameters": {"cell_type": cell_type} if cell_type else {},
                    "description": f"Address critic feedback: {missing}",
                    "target_cell_type": cell_type
                })
            
            elif "visualization" in missing_lower or "umap" in missing_lower:
                revision_actions["add_steps"].append({
                    "step_type": "visualization",
                    "function_name": "display_processed_umap",
                    "parameters": {"cell_type": cell_type} if cell_type else {},
                    "description": f"Address critic feedback: {missing}",
                    "target_cell_type": cell_type
                })
            
            elif "dea" in missing_lower or "differential" in missing_lower:
                revision_actions["add_steps"].append({
                    "step_type": "analysis", 
                    "function_name": "dea_split_by_condition",
                    "parameters": {"cell_type": cell_type} if cell_type else {},
                    "description": f"Address critic feedback: {missing}",
                    "target_cell_type": cell_type
                })
        
        return revision_actions

    def _handle_execution_errors(self, execution_history: List, critic_feedback: Dict) -> Dict:
        """Smart error handling with multiple strategies"""
        
        failed_steps = [h for h in execution_history if not h["success"]]
        
        recovery_strategy = {
            "error_analysis": [],
            "recovery_actions": [],
            "fallback_options": []
        }
        
        for failed_step in failed_steps:
            error_type = self._classify_error(failed_step["error"])
            
            if error_type == "CELL_TYPE_NOT_FOUND":
                # Use hierarchical manager to suggest alternatives
                cell_type = failed_step["step"]["parameters"].get("cell_type")
                alternatives = []
                if self.hierarchy_manager and cell_type and hasattr(self.adata, 'obs'):
                    alternatives = self.hierarchy_manager._suggest_similar_types(
                        cell_type, set(self.adata.obs["cell_type"].unique())
                    )
                
                recovery_strategy["recovery_actions"].append({
                    "action": "suggest_alternatives",
                    "original_step": failed_step["step"],
                    "alternatives": alternatives
                })
                
            elif error_type == "INSUFFICIENT_DATA":
                recovery_strategy["recovery_actions"].append({
                    "action": "remove_with_explanation",
                    "step_to_remove": failed_step["step"],
                    "explanation": "Insufficient data for this analysis"
                })
                
            elif error_type == "PARAMETER_ERROR":
                recovery_strategy["recovery_actions"].append({
                    "action": "fix_parameters",
                    "original_step": failed_step["step"],
                    "suggested_fix": self._suggest_parameter_fix(failed_step)
                })
                
            else:
                recovery_strategy["recovery_actions"].append({
                    "action": "remove_step",
                    "step_to_remove": failed_step["step"]
                })
        
        return recovery_strategy

    def _generate_revised_plan(self, original_plan: Dict, content_revision: Dict, 
                              error_recovery: Dict, iteration: int) -> Dict:
        """Generate comprehensive revised plan"""
        
        revised_steps = []
        
        # Keep successful steps from original plan
        successful_original_steps = []
        if original_plan.get("steps"):
            for step in original_plan["steps"]:
                successful_original_steps.append(step)
        
        # Apply error recovery - add corrected versions of failed steps
        for recovery_action in error_recovery.get("recovery_actions", []):
            if recovery_action["action"] == "fix_parameters":
                fixed_step = recovery_action["original_step"].copy()
                fixed_step["parameters"] = recovery_action["suggested_fix"]
                fixed_step["description"] += f" (Fixed - Iteration {iteration})"
                revised_steps.append(fixed_step)
            elif recovery_action["action"] == "suggest_alternatives":
                for alt in recovery_action["alternatives"]:
                    alt_step = recovery_action["original_step"].copy()
                    alt_step["parameters"]["cell_type"] = alt
                    alt_step["description"] = f"Alternative analysis: {alt} (Iteration {iteration})"
                    revised_steps.append(alt_step)
        
        # Add new content steps from critic feedback
        for new_step in content_revision.get("add_steps", []):
            new_step["description"] += f" (Added by critic - Iteration {iteration})"
            revised_steps.append(new_step)
        
        # Combine successful original steps with new/fixed steps
        all_steps = successful_original_steps + revised_steps
        
        return {
            "steps": all_steps,
            "original_question": original_plan["original_question"],
            "plan_summary": f"Revised plan (Iteration {iteration}) - {len(all_steps)} steps",
            "estimated_steps": len(all_steps),
            "revision_iteration": iteration,
            "revision_reason": f"Critic feedback and error recovery (Iteration {iteration})"
        }

    # ========== 🆕 UTILITY METHODS ==========

    def _extract_cell_type_from_analysis(self, analysis_description: str) -> Optional[str]:
        """🧬 UNIFIED: Extract cell type from analysis description"""
        if self.cell_type_extractor:
            return self.cell_type_extractor.extract_from_analysis_description(analysis_description)
        else:
            # Fallback if extractor not initialized
            print("⚠️ Cell type extractor not initialized, using simple fallback")
            return None

    def _extract_all_requested_cell_types(self, state: ChatState) -> Set[str]:
        """🧬 UNIFIED: Extract all requested cell types from state"""
        if self.cell_type_extractor:
            cell_types_list = self.cell_type_extractor.extract_from_execution_state(state)
            return set(cell_types_list)
        else:
            # Fallback if extractor not initialized
            print("⚠️ Cell type extractor not initialized, using simple fallback")
            return set()

    def _detect_repeated_failures(self, state: ChatState) -> List[str]:
        """Detect patterns of repeated failures"""
        
        failures = []
        
        for execution in state.get("execution_history", []):
            if not execution["success"]:
                failures.append(execution["error"])
        
        return failures

    def _classify_error(self, error_message: str) -> str:
        """Classify error types for appropriate handling"""
        
        error_lower = error_message.lower()
        
        if "cell type" in error_lower and ("not found" in error_lower or "not exist" in error_lower):
            return "CELL_TYPE_NOT_FOUND"
        elif "insufficient" in error_lower or "not enough" in error_lower:
            return "INSUFFICIENT_DATA"
        elif "parameter" in error_lower or "argument" in error_lower:
            return "PARAMETER_ERROR"
        else:
            return "UNKNOWN_ERROR"

    def _suggest_parameter_fix(self, failed_step: Dict) -> Dict:
        """Suggest parameter fixes for failed steps"""
        
        # Simple parameter fix suggestions
        original_params = failed_step["step"]["parameters"]
        suggested_params = original_params.copy()
        
        # Example fixes (extend based on common error patterns)
        if "cell_type" in original_params:
            # Try to standardize cell type name
            cell_type = original_params["cell_type"]
            if cell_type and not cell_type.endswith(" cell"):
                suggested_params["cell_type"] = f"{cell_type} cell"
        
        return suggested_params
    
    def _summarize_execution_history(self, execution_history: List) -> str:
        """Create summary of execution history for critic evaluation"""
        
        if not execution_history:
            return "No analyses performed yet."
        
        summary = []
        successful_count = 0
        failed_count = 0
        
        for execution in execution_history:
            if execution["success"]:
                successful_count += 1
                step_desc = execution["step"]["description"]
                summary.append(f"✅ {step_desc}")
            else:
                failed_count += 1
                step_desc = execution["step"]["description"]
                error = execution["error"]
                summary.append(f"❌ {step_desc} (Failed: {error})")
        
        header = f"EXECUTION SUMMARY ({successful_count} successful, {failed_count} failed):"
        return header + "\n" + "\n".join(summary)

    def _accumulate_analysis_results(self, state: ChatState, evaluation: Dict):
        """Accumulate analysis results across iterations for context building"""
        
        current_results = state.get("cumulative_analysis_results", {})
        
        # Store this iteration's results
        iteration = state["critic_iterations"]
        current_results[f"iteration_{iteration}"] = {
            "evaluation": evaluation,
            "execution_history": state["execution_history"][-10:],  # Last 10 executions
            "response": state["response"]
        }
        
        state["cumulative_analysis_results"] = current_results

    def _log_critic_statistics(self, final_state: ChatState):
        """Log critic agent performance statistics"""
        
        print(f"\n📊 CRITIC AGENT STATISTICS:")
        print(f"   • Total iterations: {final_state.get('critic_iterations', 0)}")
        print(f"   • Plan revisions: {len(final_state.get('plan_revision_history', []))}")
        print(f"   • Impossible request detected: {final_state.get('impossible_request_detected', False)}")
        
        if final_state.get("critic_feedback_history"):
            avg_relevance = sum(f["relevance_score"] for f in final_state["critic_feedback_history"]) / len(final_state["critic_feedback_history"])
            avg_completeness = sum(f["completeness_score"] for f in final_state["critic_feedback_history"]) / len(final_state["critic_feedback_history"])
            print(f"   • Average relevance score: {avg_relevance:.2f}")
            print(f"   • Average completeness score: {avg_completeness:.2f}")
        
        print(f"   • Final conversation status: {'Complete' if final_state['conversation_complete'] else 'Incomplete'}")

    # ========== Routing Functions ==========


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


# ==============================================================================
# Unified Cell Type Extraction System
# ==============================================================================

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
        if not isinstance(result, str):
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
            print(f"✅ Extracted {len(values)} cell types from dictionary format")
            
            # If we found cell types in dictionary format, return them (most reliable)
            if values:
                return self._clean_and_deduplicate(values)
        
        # PRIORITY 2: Extract from markdown headers
        header_matches = re.findall(r"###\s*Cluster\s*\d+:\s*([^:\n]+)", text)
        cell_types.extend([match.strip() for match in header_matches])
        
        # PRIORITY 3: Use specific known patterns if above methods fail
        if not cell_types:
            for pattern in self.valid_cell_type_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                cell_types.extend(matches)
        
        return self._clean_and_deduplicate(cell_types)
    
    def extract_from_analysis_description(self, analysis_description: str) -> Optional[str]:
        """
        🎯 STRATEGY 2: Extract single cell type from analysis description
        Uses multiple regex patterns to find cell types in text
        """
        if not analysis_description:
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
                if self._is_valid_cell_type(cell_type):
                    return cell_type
        
        return None
    
    def parse_multi_cell_type_string(self, cell_type_string: str) -> List[str]:
        """
        🎯 STRATEGY 3: Parse strings that might contain multiple cell types
        Handles various separators and formats
        """
        if not cell_type_string:
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
        
        return self._clean_and_deduplicate(cell_types)
    
    def extract_from_execution_state(self, state: dict) -> List[str]:
        """
        🎯 STRATEGY 4: Extract all cell types from execution state
        Combines execution plan, history, and critic feedback
        """
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
        
        return list(cell_types)
    
    def extract_from_execution_context(self, state: dict, include_history: bool = True) -> List[str]:
        """
        🎯 STRATEGY 5: Smart context-based extraction for cache and analysis
        Prioritizes current execution context over historical data
        """
        relevant_cell_types = set()
        
        print("🔍 Extracting cell types from execution context...")
        
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
        print(f"🔍 Context-based extraction result: {result}")
        return result
    
    def _clean_and_deduplicate(self, cell_types: List[str]) -> List[str]:
        """Clean, validate, and deduplicate cell type list"""
        cleaned_types = []
        
        for cell_type in cell_types:
            cleaned = cell_type.strip()
            
            # Basic validation
            if (len(cleaned) > 2 and 
                cleaned not in cleaned_types and
                self._is_valid_cell_type(cleaned)):
                cleaned_types.append(cleaned)
        
        return cleaned_types
    
    def _is_valid_cell_type(self, cell_type: str) -> bool:
        """Enhanced validation for cell types"""
        if not cell_type:
            return False
        
        cell_type_lower = cell_type.lower()
        
        # Check against invalid patterns
        for pattern in self.invalid_patterns:
            if re.search(pattern, cell_type_lower):
                return False
        
        # If we have a hierarchy manager, use it for validation
        if self.hierarchy_manager:
            return self.hierarchy_manager.is_valid_cell_type(cell_type)
        
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
        # For now, return empty list
        return []