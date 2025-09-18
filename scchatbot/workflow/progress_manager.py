"""
Progress tracking and WebSocket notification manager.

This module provides functionality to track workflow progress and send
real-time updates via WebSocket to the frontend.
"""

from typing import Optional
import logging
logger = logging.getLogger(__name__)


try:
    from ..consumers import send_progress_update
    WEBSOCKET_AVAILABLE = True
except Exception as e:
    def send_progress_update(room_name, message, progress=None, stage=None):
        logger.info(f"Progress [{room_name}]: {message} - {progress}% ({stage})")
    WEBSOCKET_AVAILABLE = False


class ProgressManager:
    """Manages progress tracking and WebSocket notifications."""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.current_stage = ""
        self.total_steps = 0
        self.completed_steps = 0
        
    def set_total_steps(self, total: int):
        """Set the total number of steps for progress calculation."""
        self.total_steps = total
        
    def update_stage(self, stage: str, message: str = None):
        """Update the current processing stage."""
        self.current_stage = stage
        if message is None:
            message = f"Processing: {stage}"
        
        progress = self._calculate_progress()
        send_progress_update(
            self.session_id, 
            message, 
            progress=progress,
            stage=stage
        )
        
    def increment_step(self, message: str = None):
        """Increment completed steps and send progress update."""
        self.completed_steps += 1
        progress = self._calculate_progress()
        
        if message is None:
            message = f"Completed step {self.completed_steps} of {self.total_steps}"
            
        send_progress_update(
            self.session_id,
            message,
            progress=progress,
            stage=self.current_stage
        )
        
    def send_custom_update(self, message: str, progress: Optional[float] = None):
        """Send a custom progress update."""
        if progress is None:
            progress = self._calculate_progress()
            
        send_progress_update(
            self.session_id,
            message,
            progress=progress,
            stage=self.current_stage
        )
        
    def _calculate_progress(self) -> float:
        """Calculate progress percentage."""
        if self.total_steps == 0:
            return 0
        return (self.completed_steps / self.total_steps) * 100
        
    @staticmethod
    def get_stage_messages():
        """Get predefined messages for different processing stages."""
        return {
            "initialization": "Initializing analysis pipeline...",
            "preprocessing": "Preprocessing single-cell data...",
            "annotation": "Performing cell type annotation...",
            "quality_control": "Running quality control checks...",
            "dimensionality_reduction": "Computing dimensionality reduction...",
            "clustering": "Performing clustering analysis...",
            "differential_expression": "Finding differentially expressed genes...",
            "enrichment": "Running enrichment analysis...",
            "visualization": "Generating visualizations...",
            "finalizing": "Finalizing results..."
        }