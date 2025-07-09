"""
Function execution history management.

This module manages the history of function executions, tracking
which functions have been called with what parameters, their results,
and success/failure status.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


class FunctionHistoryManager:
    """
    Manages function execution history using single global JSON file.
    
    Tracks all function executions with their parameters, results, and
    success/failure status to enable smart caching and context building.
    """
    
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