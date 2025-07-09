"""
Intelligent cache management for analysis results.

This module provides caching functionality for analysis results,
enabling efficient retrieval of previously computed results and
building context from cached data.
"""

import os
import json
import glob
import hashlib
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


class SimpleIntelligentCache:
    """
    Intelligent cache manager for analysis results.
    
    Handles caching of enrichment analyses, DEA results, visualizations,
    and other analysis outputs with automatic cache invalidation based on TTL.
    
    Public API Methods:
    - get_analysis_insights(): Extract insights from cached analysis results
    - check_cache(): Check for cached function results  
    - ensure_cache_directories(): Create cache directory structure
    - invalidate_cache(): Clear cached data
    - get_cache_stats(): Get performance statistics
    - cache_aware_function_wrapper(): Universal caching wrapper
    
    Public Convenience Methods:
    - check_dea_cache_exists(): Check if DEA cache exists for cell type
    - check_enrichment_cache_exists(): Check if enrichment cache exists for cell type
    - has_cached_analysis(): Check if any analysis cache exists for cell type
    - get_available_cached_cell_types(): Get list of cell types with cached results
    - get_cache_summary(): Comprehensive cache status summary
    - is_file_recent(): Check if file is within TTL
    - get_cache_file_patterns(): Get file search patterns
    - generate_cache_key(): Generate cache keys
    - extract_analysis_name_from_path(): Extract analysis type from file path
    - extract_condition_from_path(): Extract condition from DEA file path
    - load_cached_result(): Load cached results from file patterns
    """
    
    def __init__(self):
        # Map analysis types to their result directories (matching your current structure)
        self.cache_directories = {
            "enrichment": {
                "reactome": "scchatbot/enrichment/reactome",
                "go_bp": "scchatbot/enrichment/go_bp", 
                "go_mf": "scchatbot/enrichment/go_mf",
                "go_cc": "scchatbot/enrichment/go_cc",
                "kegg": "scchatbot/enrichment/kegg",
                "gsea": "scchatbot/enrichment/gsea"
            },
            "dea": "scchatbot/deg_res",
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
    
    def _is_file_recent(self, file_path: str, ttl_hours: float = None) -> bool:
        """Check if a file is recent enough to be considered valid"""
        if not os.path.exists(file_path):
            return False
            
        if ttl_hours is None:
            ttl_hours = self.default_ttl_hours
            
        file_age_hours = (time.time() - os.path.getmtime(file_path)) / 3600
        is_recent = file_age_hours <= ttl_hours
        
        print(f"✅ Cache file is recent: {file_path} (age: {file_age_hours:.1f}h)") if is_recent else print(f"⏰ Cache file is stale: {file_path} (age: {file_age_hours:.1f}h)")
        
        return is_recent

    def _check_dea_cache_exists(self, cell_type: str) -> bool:
        """Check if DEA cache exists for a specific cell type"""
        dea_patterns = self._get_cache_file_patterns("dea", cell_type)
        for pattern in dea_patterns:
            matching_files = glob.glob(pattern)
            for file_path in matching_files:
                if self._is_file_recent(file_path):
                    return True
        return False
    
    def _check_enrichment_cache_exists(self, cell_type: str) -> bool:
        """Check if enrichment cache exists for a specific cell type"""
        enrichment_patterns = self._get_cache_file_patterns("enrichment", cell_type, 
                                                           {"analyses": ["reactome", "go", "kegg", "gsea"]})
        for pattern in enrichment_patterns:
            matching_files = glob.glob(pattern)
            for file_path in matching_files:
                if self._is_file_recent(file_path):
                    return True
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
    
    # Public wrapper methods for commonly used private functionality
    
    def check_dea_cache_exists(self, cell_type: str) -> bool:
        """
        Public wrapper: Check if DEA cache exists for a specific cell type.
        
        Args:
            cell_type: The cell type to check
            
        Returns:
            bool: True if recent DEA cache exists for the cell type
        """
        return self._check_dea_cache_exists(cell_type)
    
    def check_enrichment_cache_exists(self, cell_type: str) -> bool:
        """
        Public wrapper: Check if enrichment cache exists for a specific cell type.
        
        Args:
            cell_type: The cell type to check
            
        Returns:
            bool: True if recent enrichment cache exists for the cell type
        """
        return self._check_enrichment_cache_exists(cell_type)
    
    def is_file_recent(self, file_path: str, ttl_hours: float = None) -> bool:
        """
        Public wrapper: Check if a file is recent enough to be considered valid.
        
        Args:
            file_path: Path to the file to check
            ttl_hours: Custom TTL in hours (uses default if None)
            
        Returns:
            bool: True if file exists and is within TTL
        """
        return self._is_file_recent(file_path, ttl_hours)
    
    def get_cache_file_patterns(self, analysis_type: str, cell_type: str, 
                               parameters: dict = None) -> List[str]:
        """
        Public wrapper: Get file patterns to look for based on analysis type and parameters.
        
        Args:
            analysis_type: Type of analysis ('enrichment', 'dea', 'visualization', etc.)
            cell_type: The cell type for the analysis
            parameters: Additional parameters to customize the search
            
        Returns:
            List[str]: List of file patterns to search for
        """
        return self._get_cache_file_patterns(analysis_type, cell_type, parameters)
    
    def generate_cache_key(self, function_name: str, parameters: dict) -> str:
        """
        Public wrapper: Generate a cache key from function name and parameters.
        
        Args:
            function_name: Name of the function
            parameters: Dictionary of parameters
            
        Returns:
            str: Generated cache key
        """
        return self._generate_cache_key(function_name, parameters)
    
    def extract_analysis_name_from_path(self, file_path: str) -> str:
        """
        Public wrapper: Extract analysis type from file path.
        
        Args:
            file_path: Path to the analysis file
            
        Returns:
            str: Analysis name (Reactome, GO_BP, KEGG, etc.)
        """
        return self._extract_analysis_name_from_path(file_path)
    
    def extract_condition_from_path(self, file_path: str) -> str:
        """
        Public wrapper: Extract condition from DEA file path.
        
        Args:
            file_path: Path to the DEA file
            
        Returns:
            str: Condition name extracted from filename
        """
        return self._extract_condition_from_path(file_path)
    
    def load_cached_result(self, file_patterns: List[str]) -> Tuple[Optional[Any], Optional[str]]:
        """
        Public wrapper: Load cached result from file patterns.
        
        Args:
            file_patterns: List of file patterns to search
            
        Returns:
            Tuple[Optional[Any], Optional[str]]: (cached_result, file_path) or (None, None)
        """
        return self._load_cached_result(file_patterns)
    
    # Additional convenience methods
    
    def has_cached_analysis(self, cell_type: str, analysis_type: str = "any") -> bool:
        """
        Check if any cached analysis exists for a cell type.
        
        Args:
            cell_type: The cell type to check
            analysis_type: Type to check ('dea', 'enrichment', 'any')
            
        Returns:
            bool: True if cached analysis exists
        """
        if analysis_type == "any":
            return (self.check_dea_cache_exists(cell_type) or 
                   self.check_enrichment_cache_exists(cell_type))
        elif analysis_type == "dea":
            return self.check_dea_cache_exists(cell_type)
        elif analysis_type == "enrichment":
            return self.check_enrichment_cache_exists(cell_type)
        else:
            print(f"⚠️ Unknown analysis type: {analysis_type}")
            return False
    
    def get_available_cached_cell_types(self, analysis_type: str = "any") -> List[str]:
        """
        Get list of cell types that have cached analysis results.
        
        Args:
            analysis_type: Type to check ('dea', 'enrichment', 'any')
            
        Returns:
            List[str]: List of cell types with cached results
        """
        cached_cell_types = set()
        
        # Check DEA cache directories
        if analysis_type in ["dea", "any"]:
            dea_dir = self.cache_directories["dea"]
            if os.path.exists(dea_dir):
                for file_path in glob.glob(f"{dea_dir}/*_markers_*.csv"):
                    if self.is_file_recent(file_path):
                        filename = os.path.basename(file_path)
                        cell_type = filename.split('_markers_')[0]
                        cached_cell_types.add(cell_type)
        
        # Check enrichment cache directories
        if analysis_type in ["enrichment", "any"]:
            enrichment_dirs = self.cache_directories["enrichment"]
            for analysis_name, dir_path in enrichment_dirs.items():
                if os.path.exists(dir_path):
                    for file_path in glob.glob(f"{dir_path}/results_summary_*.csv"):
                        if self.is_file_recent(file_path):
                            filename = os.path.basename(file_path)
                            cell_type = filename.replace('results_summary_', '').replace('.csv', '')
                            cached_cell_types.add(cell_type)
        
        return sorted(list(cached_cell_types))
    
    def get_cache_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive cache summary including available analyses.
        
        Returns:
            Dict[str, Any]: Summary of cache status and available data
        """
        summary = {
            "stats": self.get_cache_stats(),
            "available_cell_types": {
                "dea": self.get_available_cached_cell_types("dea"),
                "enrichment": self.get_available_cached_cell_types("enrichment"),
                "any": self.get_available_cached_cell_types("any")
            },
            "cache_directories": self.cache_directories,
            "default_ttl_hours": self.default_ttl_hours
        }
        
        # Count total files in each directory
        summary["directory_file_counts"] = {}
        for analysis_type, dirs in self.cache_directories.items():
            if isinstance(dirs, dict):
                for subtype, dir_path in dirs.items():
                    if os.path.exists(dir_path):
                        file_count = len([f for f in glob.glob(f"{dir_path}/*.csv") 
                                        if self.is_file_recent(f)])
                        summary["directory_file_counts"][f"{analysis_type}_{subtype}"] = file_count
            else:
                if os.path.exists(dirs):
                    file_count = len([f for f in glob.glob(f"{dirs}/*.csv") 
                                    if self.is_file_recent(f)])
                    summary["directory_file_counts"][analysis_type] = file_count
        
        return summary