"""
Workflow node implementations for the multi-agent chatbot.

This module now serves as a facade to the refactored workflow components.
The original WorkflowNodes class has been split into modular components
for better maintainability and is now imported from the workflow package.
"""

from .workflow import WorkflowNodes

__all__ = ['WorkflowNodes']