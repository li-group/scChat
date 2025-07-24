"""
Execution logic for workflow nodes.

This module contains execution-related methods extracted from workflow_nodes.py:
- _execute_final_question(): Executes comprehensive final question using all available context
"""

from typing import Dict, Any

from ..cell_type_models import ChatState


class ExecutionMixin:
    """Execution logic mixin for WorkflowNodes class."""
    
    def _execute_final_question(self, state: ChatState) -> str:
        """Execute a comprehensive final question using all available context"""
        original_question = state["execution_plan"]["original_question"]
        
        relevant_cell_types = self._get_relevant_cell_types_from_context(state)
        cached_context = self._build_cached_analysis_context(relevant_cell_types)
        
        # Build conversation context - only include current session, not previous unrelated conversations
        conversation_context = f"CURRENT QUESTION: {original_question}\n"
        
        # Only include the most recent user question for context, ignore previous assistant responses
        # to avoid polluting the context with irrelevant previous analyses
        
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

                            {hierarchy_context}{analysis_summary}

                            INSTRUCTIONS:
                            1. Reference the SPECIFIC pathways, genes, and statistics shown above
                            2. Use exact names and numbers from the cached results
                            3. Explain the biological significance of these specific findings
                            4. Connect the results directly to the user's question
                            5. Be quantitative and specific, not generic
                            6. Focus ONLY on the current question and relevant analysis results

                            Your response should cite the actual analysis results, not general knowledge or previous conversations."""

        try:
            # Use LangChain to generate comprehensive response
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Create messages in LangChain format
            messages = [
                SystemMessage(content="You are an expert in single-cell RNA-seq analysis. Provide comprehensive, scientifically accurate responses."),
                HumanMessage(content=final_prompt)
            ]
            
            # Initialize model
            model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.2,  # Lower temperature for more consistent responses
                max_tokens=1500
            )
            
            # Get response
            response = model.invoke(messages)
            final_answer = response.content
            
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

