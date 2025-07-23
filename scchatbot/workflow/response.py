"""
Unified response generation for workflow nodes.

This module contains:
- unified_response_generator_node implementation
- LLM synthesis-based response generation
- Plot integration after response generation
- Analysis result extraction utilities
"""

import json
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..cell_type_models import ChatState
from ..shared import extract_cell_types_from_question, extract_key_findings_from_execution, format_findings_for_synthesis


class ResponseMixin:
    """Response generation mixin for workflow nodes."""
    
    def unified_response_generator_node(self, state: ChatState) -> ChatState:
        """
        NEW UNIFIED RESPONSE GENERATOR
        Generate response using LLM synthesis of all analysis results.
        This method replaces the complex template-based response logic.
        """
        print("🎯 UNIFIED: Generating LLM-synthesized response with conversation awareness...")
        
        try:
            # 1. Extract relevant results using shared utilities
            execution_history = state.get("execution_history", [])
            if not isinstance(execution_history, list):
                execution_history = list(execution_history) if hasattr(execution_history, '__iter__') else []
            key_findings = extract_key_findings_from_execution(execution_history)
            print("✅ Key findings extracted successfully")
        except Exception as e:
            print(f"❌ Error in extract_key_findings_from_execution: {e}")
            raise
        
        
        # Skip user intent guidance (legacy code removed)
        
        try:
            # 3. Get failed analyses for transparency
            failed_analyses = self._get_failed_analyses(state)
            print("✅ Failed analyses retrieved")
        except Exception as e:
            print(f"❌ Error getting failed analyses: {e}")
            raise
        
        try:
            # 4. Check if we have conversation context from vector search
            conversation_context = None
            if state.get("has_conversation_context"):
                # Extract conversation context from messages
                for msg in state.get("messages", []):
                    if (isinstance(msg, AIMessage) and 
                        msg.content.startswith("CONVERSATION_CONTEXT:")):
                        conversation_context = msg.content[21:]  # Remove prefix
                        print(f"🎯 UNIFIED: Found conversation context ({len(conversation_context)} chars)")
                        break
            print("✅ Conversation context processed")
        except Exception as e:
            print(f"❌ Error processing conversation context: {e}")
            raise
        
        try:
            # 5. Generate synthesis prompt with conversation awareness
            synthesis_prompt = self._create_enhanced_synthesis_prompt(
                original_question=state.get("current_message", ""),
                key_findings=key_findings,
                failed_analyses=failed_analyses,
                conversation_context=conversation_context
            )
            print("✅ Synthesis prompt created")
        except Exception as e:
            print(f"❌ Error creating synthesis prompt: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        
        # 6. Get LLM response (text only, no plots yet)
        try:
            response_text = self._call_llm_for_synthesis(synthesis_prompt)
        except Exception as e:
            print(f"❌ LLM synthesis failed: {e}")
            response_text = "I encountered an error generating the response. Please try again."
        
        # 7. Store response WITHOUT plots (plots added separately)
        # Format as JSON for compatibility with views.py
        response_data = {
            "response": response_text,
            "response_type": "llm_synthesized_answer"
        }
        state["response"] = json.dumps(response_data)
        state["available_plots"] = self._collect_available_plots(state)  # Store plots separately
        
        print(f"🎯 UNIFIED: Generated response ({len(response_text)} chars)")
        return state
    


    # ========== NEW UNIFIED RESPONSE GENERATION METHODS ==========
    
    def _get_failed_analyses(self, state: ChatState) -> List[Dict[str, Any]]:
        """Collect information about failed analyses for transparent reporting."""
        failed_analyses = []
        
        for step in state.get("execution_history", []):
            # Skip if step is not a dictionary
            if not isinstance(step, dict):
                continue
                
            if not step.get("success", True):  # Failed step
                # Extract function name - handle both nested and flat structures
                step_data = step.get("step", {})
                if isinstance(step_data, dict) and "function_name" in step_data:
                    # New nested structure
                    function_name = step_data.get("function_name", "unknown")
                    parameters = step_data.get("parameters", {})
                elif "function_name" in step:
                    # Flat structure (fallback)
                    function_name = step.get("function_name", "unknown")
                    parameters = step.get("parameters", {})
                else:
                    function_name = "unknown"
                    parameters = {}
                
                failed_analyses.append({
                    "function": function_name,
                    "parameters": parameters,
                    "error": step.get("error", "Unknown error")
                })
        
        return failed_analyses
    
    def _create_enhanced_synthesis_prompt(self, original_question: str, key_findings: Dict[str, Any], 
                                         failed_analyses: List[Dict],
                                         conversation_context: str = None) -> str:
        """Create prompt for synthesizing analysis results with conversation awareness."""
        
        prompt = f"""You are a single-cell RNA-seq analysis expert. 

                    USER'S QUESTION: "{original_question}"
                    """

        # Add conversation context if available
        if conversation_context:
            prompt += f"""
                        {conversation_context}

                        Please consider the conversation history above when formulating your response.
                        """

        prompt += f"""
                    CURRENT ANALYSIS RESULTS:
                    {format_findings_for_synthesis(key_findings)}
                    """

        # Add failed analyses if any
        if failed_analyses:
            prompt += "\n\nFAILED ANALYSES:\n"
            for failure in failed_analyses:
                prompt += f"- {failure['function']}: {failure['error']}\n"

        # User intent guidance removed

        prompt += """
                INSTRUCTIONS:
                1. Answer the user's question directly using the available data
                2. Use specific gene names, pathways, and statistics from the results
                3. If analyses failed, acknowledge this but provide insights using available data and biological knowledge
                4. For comparisons, list concrete distinguishing features with specific molecular evidence
                5. Be comprehensive but concise
                6. Focus on answering the specific question asked, not providing general information"""
        
        # Add conversation-aware instructions if context exists
        if conversation_context:
            prompt += """
                    7. Consider the conversation history and maintain continuity
                    8. Reference specific previous discussions when relevant
                    9. If the user is referring to something from earlier, address it specifically"""

        prompt += "\n\nAnswer:"
        
        return prompt
    
    def _call_llm_for_synthesis(self, prompt: str) -> str:
        """Call LLM to synthesize analysis results into a focused answer."""
        
        try:
            messages = [
                SystemMessage(content="You are a single-cell RNA-seq analysis expert who provides direct, specific answers to user questions based on analysis results."),
                HumanMessage(content=prompt)
            ]
            
            model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,  # Lower temperature for more consistent scientific responses
                max_tokens=800    # Reasonable limit for focused answers
            )
            
            response = model.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return f"I encountered an error generating the response: {e}"
    
    def _collect_available_plots(self, state: ChatState) -> List[str]:
        """Collect information about plots that were generated during execution."""
        available_plots = []
        
        for step in state.get("execution_history", []):
            if step.get("success", False):
                # Handle both nested and flat execution history structures
                step_data = step.get("step", {})
                if isinstance(step_data, dict) and "function_name" in step_data:
                    # New nested structure
                    function_name = step_data.get("function_name", "")
                    parameters = step_data.get("parameters", {})
                elif "function_name" in step:
                    # Flat structure (fallback)
                    function_name = step.get("function_name", "")
                    parameters = step.get("parameters", {})
                else:
                    continue
                    
                if function_name.startswith("display_"):
                    cell_type = parameters.get("cell_type", "unknown")
                    
                    # Create description of the available plot
                    plot_description = f"📊 {function_name.replace('display_', '').replace('_', ' ').title()}"
                    if cell_type != "unknown":
                        plot_description += f" for {cell_type}"
                    
                    available_plots.append(plot_description)
        
        return available_plots
    
    def add_plots_to_final_response(self, state: ChatState) -> ChatState:
        """Add plots to response after generation."""
        
        # This method is only called when workflow routes to plot_integration,
        # which means the response has been approved (either normally or by iteration limit)
        print("🎨 PLOT INTEGRATION: Adding plots to final response...")
        
        # Parse the JSON response
        try:
            response_data = json.loads(state.get("response", "{}"))
            response_text = response_data.get("response", "")
        except json.JSONDecodeError:
            # Fallback if response is not JSON
            response_text = state.get("response", "")
            response_data = {"response": response_text}
        
        # Extract actual HTML plots from execution history
        html_plots = self._extract_html_plots_from_execution(state)
        
        if html_plots:
            # Store plots separately - do NOT add to response text
            response_data["graph_html"] = html_plots
            
            # Keep response text clean - only add a simple note about available plots
            # response_text remains unchanged to avoid HTML contamination
            
            print("🎨 PLOT INTEGRATION: Successfully stored plots separately from response")
        else:
            print("🎨 PLOT INTEGRATION: No plots found in execution history")
        
        # Store back as JSON with size checking
        response_json = json.dumps(response_data)
        response_size = len(response_json)
        
        # Check final response size
        MAX_RESPONSE_SIZE = 50 * 1024 * 1024  # 50MB maximum
        if response_size > MAX_RESPONSE_SIZE:
            print(f"🎨 PLOT INTEGRATION: WARNING - Response too large ({response_size:,} chars > {MAX_RESPONSE_SIZE:,})")
            # Remove plots if response is too large
            response_data_fallback = {
                "response": response_data.get("response", ""),
                "response_type": response_data.get("response_type", ""),
                "error": f"Plots removed due to size limit (original size: {response_size:,} chars)"
            }
            response_json = json.dumps(response_data_fallback)
            print(f"🎨 PLOT INTEGRATION: Fallback response size: {len(response_json):,} chars")
        
        state["response"] = response_json
        
        # Add response to message history for conversation continuity (text only, no HTML)
        try:
            state["messages"].append(AIMessage(content=response_text))
        except Exception as e:
            print(f"⚠️ Could not add response to message history: {e}")
        
        return state
    
    def _extract_html_plots_from_execution(self, state: ChatState) -> str:
        """Extract actual HTML plots from execution history with deduplication and size limits."""
        plots = []
        plot_descriptions = []
        seen_plots = set()  # Track unique plots to avoid duplicates
        
        execution_history = state.get("execution_history", [])
        print(f"🎨 PLOT EXTRACTION: Checking {len(execution_history)} execution steps")
        
        MAX_PLOTS = 3  # Limit to 3 plots maximum
        MAX_PLOT_SIZE = 10 * 1024 * 1024  # 10MB per plot maximum
        
        for i, execution in enumerate(execution_history):
            if len(plots) >= MAX_PLOTS:
                print(f"🎨 PLOT EXTRACTION: Reached max plots limit ({MAX_PLOTS}), stopping")
                break
                
            # Handle both nested and flat execution history structures
            step_data = execution.get("step", {})
            if isinstance(step_data, dict) and "function_name" in step_data:
                # New nested structure
                function_name = step_data.get("function_name", "")
                success = execution.get("success", False)
                result = execution.get("result") or execution.get("result_summary")
            elif "function_name" in execution:
                # Flat structure (fallback)
                function_name = execution.get("function_name", "")
                success = execution.get("success", False)
                result = execution.get("result") or execution.get("result_summary")
            else:
                # No recognizable function structure
                function_name = ""
                success = False
                result = None
            
            has_result = result is not None
            
            print(f"🎨 PLOT EXTRACTION: Step {i+1}: {function_name}, success={success}, has_result={has_result}")
            
            # Debug: Check result content for display_ functions
            if function_name.startswith("display_") and has_result:
                result_preview = str(result)[:200] if result else "None"
                result_type = type(result).__name__
                has_html = isinstance(result, str) and ("<div" in result or "<html" in result)
                print(f"🎨 PLOT DEBUG: Result type={result_type}, length={len(str(result)) if result else 0}, has_html={has_html}")
                print(f"🎨 PLOT DEBUG: Result preview: {result_preview}...")
            
            # Check for plots by function name OR by HTML content (fallback for storage bugs)
            is_plot_function = function_name.startswith("display_")
            is_html_content = (isinstance(result, str) and 
                             ("<div" in result or "<html" in result) and 
                             len(result) > 1000)  # Likely a plot if it's large HTML
            
            if (success and has_result and (is_plot_function or is_html_content)):
                result_length = len(result)
                
                # Check size limit
                if result_length > MAX_PLOT_SIZE:
                    print(f"🎨 PLOT EXTRACTION: Skipping {function_name} - too large ({result_length} chars > {MAX_PLOT_SIZE})")
                    continue
                
                # Create a unique identifier for this plot (first 100 chars as fingerprint)
                plot_fingerprint = result[:100] if len(result) > 100 else result
                
                # Check for duplicates
                if plot_fingerprint in seen_plots:
                    print(f"🎨 PLOT EXTRACTION: Skipping {function_name} - duplicate plot detected")
                    continue
                
                seen_plots.add(plot_fingerprint)
                
                # Get description using extracted function_name
                if is_plot_function:
                    description = self._generate_visualization_description(function_name, step_data if "function_name" in step_data else execution, result)
                else:
                    # Fallback description for HTML content without display_ function name
                    description = "Generated visualization plot"
                plot_descriptions.append(f"<h4>{description}</h4>")
                plots.append(result)
                
                # Determine plot type for description
                if is_plot_function:
                    plot_type = f"plot {function_name}"
                else:
                    plot_type = f"HTML content (likely visualization)"
                
                print(f"🎨 PLOT EXTRACTION: Found unique {plot_type} ({result_length} chars)")
        
        if plots:
            combined_plots = "".join([f"<div class='plot-container'>{desc}{plot}</div>" 
                                     for desc, plot in zip(plot_descriptions, plots)])
            total_size = len(combined_plots)
            print(f"🎨 PLOT EXTRACTION: Successfully extracted {len(plots)} unique plots (total: {total_size:,} chars)")
            
            # Final size check
            MAX_TOTAL_SIZE = 20 * 1024 * 1024  # 20MB total maximum
            if total_size > MAX_TOTAL_SIZE:
                print(f"🎨 PLOT EXTRACTION: WARNING - Total plot size ({total_size:,}) exceeds limit ({MAX_TOTAL_SIZE:,})")
                # Take only the first plot if still too large
                if plots:
                    first_plot = f"<div class='plot-container'>{plot_descriptions[0]}{plots[0]}</div>"
                    print(f"🎨 PLOT EXTRACTION: Using only first plot ({len(first_plot):,} chars)")
                    return first_plot
            
            return combined_plots
        else:
            print("🎨 PLOT EXTRACTION: No valid plots found")
            return ""
    
    def _generate_visualization_description(self, function_name: str, execution: Dict, result: str) -> str:
        """Generate a descriptive summary for visualization functions"""
        parameters = execution.get("parameters", {})
        
        # Create user-friendly descriptions for different visualization types
        descriptions = {
            "display_dotplot": "gene expression dotplot",
            "display_cell_type_composition": "cell type composition dendrogram", 
            "display_gsea_dotplot": "GSEA enrichment dotplot",
            "display_umap": "UMAP dimensionality reduction plot",
            "display_processed_umap": "annotated UMAP plot with cell types",
            "display_enrichment_barplot": "enrichment analysis barplot",
            "display_enrichment_dotplot": "enrichment analysis dotplot",
            "display_enrichment_visualization": "comprehensive enrichment visualization"
        }
        
        base_desc = descriptions.get(function_name, function_name.replace("_", " "))
        
        # Add context based on parameters
        context_parts = []
        if parameters.get("cell_type"):
            context_parts.append(f"for {parameters['cell_type']}")
        if parameters.get("analysis"):
            context_parts.append(f"using {parameters['analysis'].upper()} analysis")
        if parameters.get("plot_type") == "both":
            context_parts.append("(both bar and dot plots)")
        
        context = " " + " ".join(context_parts) if context_parts else ""
        
        # Check if the plot was generated successfully
        if result and isinstance(result, str) and ("<div" in result or "<html" in result):
            status = "✅ Successfully generated"
        else:
            status = "⚠️ Generated"
        
        return f"{status} {base_desc}{context}. The interactive plot is displayed below."