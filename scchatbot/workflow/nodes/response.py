"""
Response generation node implementation.

This module contains the ResponseGeneratorNode which generates final responses
by synthesizing analysis results and conversation context.
"""

from typing import Dict, Any, List, Optional
import json
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ...cell_types.models import ChatState
from ..node_base import BaseWorkflowNode
from ..unified_result_accessor import get_unified_results_for_synthesis
import logging
logger = logging.getLogger(__name__)

class ResponseGeneratorNode(BaseWorkflowNode):
    """
    Response generator node that creates final responses.
    
    Responsibilities:
    - Synthesize analysis results into coherent responses
    - Integrate conversation context and history
    - Generate LLM-based responses with proper formatting
    - Handle visualization integration
    """
    
    def execute(self, state: ChatState) -> ChatState:
        """Main execution method for response generation."""
        return self.unified_response_generator_node(state)
    
    def unified_response_generator_node(self, state: ChatState) -> ChatState:
        """
        NEW UNIFIED RESPONSE GENERATOR
        Generate response using LLM synthesis of all analysis results.
        This method replaces the complex template-based response logic.
        """
        logger.info("🎯 UNIFIED: Generating LLM-synthesized response with conversation awareness...")
        
        try:
            # 1. Extract relevant results using unified result accessor (NEW SYSTEM)
            execution_history = state.get("execution_history", [])
            if not isinstance(execution_history, list):
                execution_history = list(execution_history) if hasattr(execution_history, '__iter__') else []
            
            # Use new unified accessor that handles mixed storage patterns
            formatted_findings = get_unified_results_for_synthesis(execution_history)
            logger.info("✅ Unified results extracted and formatted successfully")
            
            # No legacy fallback - unified accessor is the only method
            if not formatted_findings or len(formatted_findings.strip()) < 50:
                logger.info("⚠️ Unified accessor returned minimal results")
                formatted_findings = "No analysis results available for synthesis"
            
        except Exception as e:
            logger.info(f"❌ Error in unified result accessor: {e}")
            formatted_findings = f"Error extracting analysis results: {e}"
        
        try:
            # 3. Get failed analyses for transparency
            failed_analyses = self._get_failed_analyses(state)
            logger.info("✅ Failed analyses retrieved")
        except Exception as e:
            logger.info(f"❌ Error getting failed analyses: {e}")
            failed_analyses = []
        
        try:
            # 4. Extract unified context (should now be clean with no accumulation)
            conversation_context = None
            messages = state.get("messages", [])
            
            # Since we now clean old context, there should be at most ONE context message
            for msg in messages:
                if isinstance(msg, AIMessage) and any(prefix in msg.content for prefix in 
                    ["CURRENT_SESSION_STATE:", "CONVERSATION_HISTORY:", "CONVERSATION_CONTEXT:"]):
                    conversation_context = msg.content
                    logger.info(f"🎯 UNIFIED: Found context message ({len(conversation_context)} chars)")
                    break
            
            logger.info("✅ Conversation context processed")
        except Exception as e:
            logger.info(f"❌ Error processing conversation context: {e}")
            conversation_context = None
        
        try:
            # 5. Search for cell type discoveries if this is a "find X cell" query
            discovery_context = self._search_for_cell_discoveries(state)
            logger.info(f"✅ Discovery search completed")
        except Exception as e:
            logger.info(f"❌ Error searching for discoveries: {e}")
            discovery_context = None

        try:
            # 6. Get post-execution evaluation results for question type and relevance hints
            post_eval = state.get("post_execution_evaluation", {})
            question_type = post_eval.get("question_type")
            analysis_relevance = post_eval.get("analysis_relevance", {})

            # 7. Generate synthesis prompt with conversation awareness and relevance hints
            synthesis_prompt = self._create_enhanced_synthesis_prompt_with_formatted_findings(
                original_question=state.get("current_message", ""),
                formatted_findings=formatted_findings,
                failed_analyses=failed_analyses,
                conversation_context=conversation_context,
                discovery_context=discovery_context,
                question_type=question_type,
                analysis_relevance=analysis_relevance
            )
            logger.info("✅ Synthesis prompt created with relevance hints")
        except Exception as e:
            logger.info(f"❌ Error creating synthesis prompt: {e}")
            synthesis_prompt = f"Please answer the user's question: {state.get('current_message', '')}"
        
        # 6. Get LLM response (text only, no plots yet)
        try:
            response_text = self._call_llm_for_synthesis(synthesis_prompt)
        except Exception as e:
            logger.info(f"❌ LLM synthesis failed: {e}")
            response_text = "I encountered an error generating the response. Please try again."
        
        # 7. Collect plots as individual objects
        try:
            plots = self._extract_html_plots_from_execution(state)  # Returns List[Dict]
            logger.info(f"🎯 ResponseGeneratorNode: Found {len(plots)} individual plots")
        except Exception as e:
            logger.info(f"❌ Error extracting plots: {e}")
            plots = []
        
        # 8. NEW: Create multiple messages structure for separate rendering
        if plots and len(plots) > 0:
            logger.info(f"🎨 Creating multiple messages structure: {len(plots)} plots + 1 text (plots first)")
            
            # Create comprehensive response with separate messages array
            multiple_messages = []
            
            # Add each plot as separate message entry FIRST
            for i, plot in enumerate(plots):
                multiple_messages.append({
                    "message_type": "plot",
                    "response_type": "individual_plot",
                    "plot_title": plot.get("title", f"Plot {i+1}"),
                    "plot_description": plot.get("description", ""),
                    "plots": [plot],  # Single plot in array
                    "graph_html": f"<div class='plot-container'><h4>{plot.get('description', '')}</h4>{plot.get('html', '')}</div>"
                })
            
            # Add text message as LAST entry
            multiple_messages.append({
                "message_type": "text",
                "response": response_text,
                "response_type": "llm_synthesized_answer"
            })
            
            # Create response with multiple messages structure
            response_data = {
                "response": response_text,  # Main text response for backward compatibility
                "response_type": "multiple_messages",
                "plots_version": "3.0",  # New version for multiple messages
                "messages": multiple_messages,
                "total_messages": len(multiple_messages),
                # Backward compatibility fields
                "plots": plots,
                "graph_html": self._combine_plots_for_legacy(plots)
            }
            
            logger.info(f"🎨 Created multiple messages structure with {len(multiple_messages)} total messages (plots first, then text)")
            
        else:
            # No plots - single text response
            response_data = {
                "response": response_text,
                "response_type": "llm_synthesized_answer",
                "plots_version": "3.0"
            }
        
        state["response"] = json.dumps(response_data)
        
        logger.info(f"🎯 UNIFIED: Generated response ({len(response_text)} chars)")
        return state
    
    def add_plots_to_final_response(self, state: ChatState) -> ChatState:
        """Add plots to response after generation."""
        
        # This method is only called when workflow routes to plot_integration,
        # which means the response has been approved (either normally or by iteration limit)
        logger.info("🎨 PLOT INTEGRATION: Adding plots to final response...")
        
        # Parse the JSON response
        try:
            response_data = json.loads(state.get("response", "{}"))
            response_text = response_data.get("response", "")
        except json.JSONDecodeError:
            # Fallback if response is not JSON
            response_text = state.get("response", "")
            response_data = {"response": response_text}
        
        # Extract plots as individual objects
        plots = self._extract_html_plots_from_execution(state)
        
        if plots:
            # Store individual plots and maintain backward compatibility
            response_data["plots"] = plots
            response_data["graph_html"] = self._combine_plots_for_legacy(plots)
            
            # Keep response text clean - only add a simple note about available plots
            # response_text remains unchanged to avoid HTML contamination
            
            logger.info(f"🎨 PLOT INTEGRATION: Successfully stored {len(plots)} individual plots separately from response")
        else:
            logger.info("🎨 PLOT INTEGRATION: No plots found in execution history")
            response_data["plots"] = []
        
        # Store back as JSON with size checking
        response_json = json.dumps(response_data)
        response_size = len(response_json)
        
        # Check final response size
        MAX_RESPONSE_SIZE = 50 * 1024 * 1024  # 50MB maximum
        if response_size > MAX_RESPONSE_SIZE:
            logger.info(f"🎨 PLOT INTEGRATION: WARNING - Response too large ({response_size:,} chars > {MAX_RESPONSE_SIZE:,})")
            # Remove plots if response is too large
            response_data_fallback = {
                "response": response_data.get("response", ""),
                "response_type": response_data.get("response_type", ""),
                "plots_version": "2.0",
                "plots": [],
                "error": f"Plots removed due to size limit (original size: {response_size:,} chars)"
            }
            response_json = json.dumps(response_data_fallback)
            logger.info(f"🎨 PLOT INTEGRATION: Fallback response size: {len(response_json):,} chars")
        
        state["response"] = response_json
        
        # Add response to message history for conversation continuity (text only, no HTML)
        try:
            from langchain_core.messages import AIMessage
            state["messages"].append(AIMessage(content=response_text))
        except Exception as e:
            logger.info(f"⚠️ Could not add response to message history: {e}")
        
        return state
    
    def _generate_execution_summary_with_plots(self, state: ChatState):
        """Generate a summary of multi-step execution with collected plots"""
        logger.info(f"🔍 Plot collection: Total execution history entries: {len(state.get('execution_history', []))}")
        
        successful_steps = [h for h in state["execution_history"] if h["success"]]
        
        # Debug: Show all function names in execution history
        all_function_names = [h["step"]["function_name"] for h in successful_steps]
        logger.info(f"🔍 Plot collection: Function names in execution history: {all_function_names}")
        logger.info(f"🔍 Plot collection: Visualization functions set: {self.visualization_functions}")
        
        visualization_steps = [h for h in successful_steps if h["step"]["function_name"] in self.visualization_functions]
        
        logger.info(f"🔍 Plot collection: {len(successful_steps)} successful steps, {len(visualization_steps)} visualization steps")
        
        if not successful_steps:
            return "I encountered issues executing your request.", ""
        
        summary = f"I completed {len(successful_steps)} analysis steps:\n\n"
        collected_plots = []
        plot_descriptions = []
        
        for i, step in enumerate(successful_steps, 1):
            step_desc = step["step"]["description"]
            function_name = step["step"]["function_name"]
            result = step.get("result", "")
            
            # Check if this was a visualization step
            if function_name in self.visualization_functions:
                # Add plot description to summary
                plot_info = f"📊 {step_desc}"
                if step["step"].get("parameters", {}).get("cell_type"):
                    plot_info += f" (for {step['step']['parameters']['cell_type']})"
                summary += f"{i}. {plot_info}\n"
                
                # Debug: Check what we have in result
                logger.info(f"🔍 Plot collection debug - Function: {function_name}")
                logger.info(f"🔍 Result type: {type(result)}, length: {len(str(result)) if result else 0}")
                logger.info(f"🔍 Has HTML markers: <div={bool('<div' in str(result))}, <html={bool('<html' in str(result))}")
                logger.info(f"🔍 Result preview: {str(result)[:200]}...")
                
                # Collect the HTML plot if it's valid HTML
                if result and isinstance(result, str) and ("<div" in result or "<html" in result):
                    # Check if this is a duplicate plot
                    if step_desc not in plot_descriptions:
                        collected_plots.append(f"<div class='plot-container'><h4>{step_desc}</h4>{result}</div>")
                        plot_descriptions.append(step_desc)
                        logger.info(f"✅ Plot collected: {step_desc}")
                    else:
                        logger.info(f"⚠️ Duplicate plot detected, skipping: {step_desc}")
                else:
                    logger.info(f"❌ Plot NOT collected for {step_desc} - invalid HTML or empty result")
            else:
                # Regular analysis step
                summary += f"{i}. {step_desc}\n"
        
        summary += "\nAll analyses have been completed successfully."
        
        # Add plot information to summary if we have plots
        if plot_descriptions:
            summary += f"\n\n📊 Generated {len(plot_descriptions)} visualization(s):"
            for desc in plot_descriptions:
                summary += f"\n• {desc}"
        
        # Combine all plots into a single HTML string
        combined_plots = "\n".join(collected_plots) if collected_plots else ""
        
        logger.info(f"🔍 Plot collection summary: Collected {len(collected_plots)} plots, total HTML length: {len(combined_plots)}")
        if collected_plots:
            logger.info(f"🔍 First plot preview: {collected_plots[0][:100]}...")
        
        return summary, combined_plots
    
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
    
    def _create_enhanced_synthesis_prompt_with_formatted_findings(self, original_question: str, formatted_findings: str,
                                     failed_analyses: List[Dict],
                                     conversation_context: str = None,
                                     discovery_context: str = None,
                                     question_type: str = None,
                                     analysis_relevance: Dict[str, Any] = None) -> str:
        """Create prompt for synthesizing analysis results with pre-formatted findings from unified accessor."""
        
        prompt = f"""You are a single-cell RNA-seq analysis expert. 

                    USER'S QUESTION: "{original_question}"
                    """
        
        # Add question type and guidance if available
        if question_type and analysis_relevance:
            guidance = analysis_relevance.get("guidance", "")
            prompt += f"""
                        QUESTION TYPE: {question_type}
                        RESPONSE GUIDANCE: {guidance}
                        """
            
            # Add analysis relevance hints
            relevance_categories = analysis_relevance.get("relevance_categories", {})
            if relevance_categories:
                prompt += """
                        ANALYSIS RELEVANCE:
                        - Primary analyses (focus on these): """ + str(relevance_categories.get("primary", [])) + """
                        - Secondary analyses (use for support): """ + str(relevance_categories.get("secondary", [])) + """
                        """

        # Add conversation context if available
        if conversation_context:
            prompt += f"""
                        CURRENT SESSION CONTEXT:
                        {conversation_context}

                        Please use the current session information above to provide accurate, context-aware responses.
                        """

        prompt += f"""
                    CURRENT ANALYSIS RESULTS:
                    {formatted_findings}
                    """

        # Add failed analyses if any
        if failed_analyses:
            prompt += "\n\nFAILED ANALYSES:\n"
            for failure in failed_analyses:
                prompt += f"- {failure['function']}: {failure['error']}\n"

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

        # Add discovery context if available
        if discovery_context:
            prompt += f"""

                    CELL TYPE DISCOVERY INFORMATION:
                    {discovery_context}

                    CRITICAL: If the user asks to "find" a cell type and you see it was discovered above,
                    lead with "✅ [Cell Type] successfully discovered!" and explain it's now available for analysis."""

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
            logger.info(f"❌ OpenAI API error: {e}")
            return f"I encountered an error generating the response: {e}"
    
    def _extract_html_plots_from_execution(self, state: ChatState) -> List[Dict[str, Any]]:
        """Extract HTML plots as individual objects with metadata and deduplication."""
        plots = []
        seen_plots = set()  # Track unique plots to avoid duplicates
        
        execution_history = state.get("execution_history", [])
        logger.info(f"🎨 PLOT EXTRACTION: Checking {len(execution_history)} execution steps")
        
        MAX_PLOTS = 6  # Limited to 6 plots for frontend display
        MAX_PLOT_SIZE = 8 * 1024 * 1024  # 8MB per plot maximum (reduced from 10MB)
        
        for i, execution in enumerate(execution_history):
            if len(plots) >= MAX_PLOTS:
                logger.info(f"🎨 PLOT EXTRACTION: Reached max plots limit ({MAX_PLOTS}), stopping")
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
            
            logger.info(f"🎨 PLOT EXTRACTION: Step {i+1}: {function_name}, success={success}, has_result={has_result}")
            
            # Check for plots by function name OR by HTML content OR by multiple plots structure
            is_plot_function = function_name.startswith("display_")
            is_html_content = (isinstance(result, str) and 
                             ("<div" in result or "<html" in result) and 
                             len(result) > 1000)  # Likely a plot if it's large HTML
            is_multiple_plots = isinstance(result, dict) and result.get("multiple_plots")
            
            if (success and has_result and (is_plot_function or is_html_content or is_multiple_plots)):
                # Check if result is a multiple plots structure
                if isinstance(result, dict) and result.get("multiple_plots"):
                    logger.info(f"🎨 PLOT EXTRACTION: Found multiple plots structure in {function_name}")
                    # Handle multiple plots from single function call
                    for plot_data in result.get("plots", []):
                        if len(plots) >= MAX_PLOTS:
                            logger.info(f"🎨 PLOT EXTRACTION: Reached max plots limit ({MAX_PLOTS}), stopping")
                            break
                            
                        plot_html = plot_data.get("html", "")
                        plot_size = len(plot_html)
                        
                        # Check size limit for individual plot
                        if plot_size > MAX_PLOT_SIZE:
                            logger.info(f"🎨 PLOT EXTRACTION: Skipping {plot_data.get('type', 'unknown')} plot - too large ({plot_size} chars > {MAX_PLOT_SIZE})")
                            continue
                        
                        plot_id = f"plot_{len(plots) + 1}"
                        plot_obj = {
                            "id": plot_id,
                            "title": plot_data.get("title", f"Plot {plot_id}"),
                            "description": f"✅ Successfully generated {plot_data.get('type', 'visualization')} plot for {step_data.get('parameters', {}).get('cell_type', 'analysis')}",
                            "html": plot_html,
                            "function_name": function_name,
                            "plot_type": plot_data.get("type", "unknown"),
                            "size": plot_size,
                            "parameters": step_data.get("parameters", {}) if "function_name" in step_data else execution.get("parameters", {})
                        }
                        
                        plots.append(plot_obj)
                        logger.info(f"🎨 PLOT EXTRACTION: Added {plot_data.get('type')} plot {plot_id}: {plot_obj['title']} ({plot_size} chars)")
                    
                    continue  # Skip single plot processing for this result
                
                # Handle single plot result (existing logic)
                if isinstance(result, str):
                    result_length = len(result)
                else:
                    result_length = 0
                
                # Check size limit
                if result_length > MAX_PLOT_SIZE:
                    logger.info(f"🎨 PLOT EXTRACTION: Skipping {function_name} - too large ({result_length} chars > {MAX_PLOT_SIZE})")
                    continue
                
                # Create a unique identifier for this plot (first 100 chars as fingerprint)
                plot_fingerprint = result[:100] if len(result) > 100 else result
                
                # Check for duplicates
                if plot_fingerprint in seen_plots:
                    logger.info(f"🎨 PLOT EXTRACTION: Skipping {function_name} - duplicate plot detected")
                    continue
                
                seen_plots.add(plot_fingerprint)
                
                # Generate plot metadata
                plot_id = f"plot_{len(plots) + 1}"
                title = self._extract_plot_title(function_name, step_data if "function_name" in step_data else execution)
                description = self._generate_visualization_description(function_name, step_data if "function_name" in step_data else execution, result)
                
                # Create plot object with metadata
                plot_obj = {
                    "id": plot_id,
                    "title": title,
                    "description": description,
                    "html": result,
                    "function_name": function_name,
                    "size": result_length,
                    "parameters": step_data.get("parameters", {}) if "function_name" in step_data else execution.get("parameters", {})
                }
                
                plots.append(plot_obj)
                logger.info(f"🎨 PLOT EXTRACTION: Added plot {plot_id}: {title} ({result_length} chars)")
        
        if plots:
            total_size = sum(plot["size"] for plot in plots)
            logger.info(f"🎨 PLOT EXTRACTION: Successfully extracted {len(plots)} individual plots (total: {total_size:,} chars)")
            
            # Apply size validation
            validated_plots = self._validate_plot_sizes(plots)
            return validated_plots
        else:
            logger.info("🎨 PLOT EXTRACTION: No valid plots found")
            return []
    
    def _generate_visualization_description(self, function_name: str, execution: Dict, result: str) -> str:
        """Generate a descriptive summary for visualization functions"""
        parameters = execution.get("parameters", {})
        
        # Create user-friendly descriptions for different visualization types
        descriptions = {
            "display_dotplot": "gene expression dotplot",
            "display_processed_umap": "annotated UMAP plot with cell types",
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
    
    def create_individual_plot_messages(self, state: ChatState) -> List[ChatState]:
        """Create separate ChatState messages for each individual plot."""
        plot_messages = []
        plots = state.get("individual_plots", [])
        
        if not plots:
            return plot_messages
            
        logger.info(f"🎨 Creating {len(plots)} individual plot messages")
        
        for i, plot in enumerate(plots):
            # Create a new state for this plot message
            plot_state = state.copy()
            
            # Create response data with just this plot
            plot_response_data = {
                "response": "",  # No text, just the plot
                "response_type": "individual_plot",
                "plots_version": "2.0",
                "plot_title": plot.get("title", f"Plot {i+1}"),
                "plot_description": plot.get("description", ""),
                "plots": [plot],  # Single plot in array
                "graph_html": f"<div class='plot-container'><h4>{plot.get('description', '')}</h4>{plot.get('html', '')}</div>"
            }
            
            plot_state["response"] = json.dumps(plot_response_data)
            plot_messages.append(plot_state)
            
            logger.info(f"🎨 Created plot message {i+1}: {plot.get('title', 'Untitled')}")
        
        return plot_messages
    
    def _extract_plot_title(self, function_name: str, execution: Dict) -> str:
        """Extract a concise title for the plot."""
        titles = {
            "display_dotplot": "Gene Expression Dotplot",
            "display_processed_umap": "Annotated UMAP",
            "display_enrichment_visualization": "Enrichment Visualization"
        }
        
        base_title = titles.get(function_name, function_name.replace("display_", "").replace("_", " ").title())
        
        # Add context from parameters
        parameters = execution.get("parameters", {})
        if parameters.get("cell_type"):
            base_title += f" - {parameters['cell_type']}"
        if parameters.get("analysis"):
            base_title += f" ({parameters['analysis'].upper()})"
        
        return base_title
    
    def _validate_plot_sizes(self, plots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and potentially reduce plot sizes with intelligent prioritization."""
        MAX_TOTAL_SIZE = 30 * 1024 * 1024  # 30MB total limit
        MAX_PLOTS = 6  # Limited to 6 plots for frontend display
        
        # Sort plots by importance (certain visualization functions get priority)
        importance_order = ["display_processed_umap", "display_dotplot", "display_enrichment_visualization"]
        
        def plot_priority(plot):
            func_name = plot.get("function_name", "")
            if func_name in importance_order:
                return importance_order.index(func_name)
            return len(importance_order)
        
        plots_sorted = sorted(plots, key=plot_priority)
        
        # Take top plots within limits
        validated_plots = []
        total_size = 0
        
        for plot in plots_sorted[:MAX_PLOTS]:
            if total_size + plot["size"] <= MAX_TOTAL_SIZE:
                validated_plots.append(plot)
                total_size += plot["size"]
            else:
                logger.info(f"🎨 SIZE VALIDATION: Skipping {plot['id']} - would exceed total size limit")
                break
        
        logger.info(f"🎨 SIZE VALIDATION: Validated {len(validated_plots)} plots, total size: {total_size:,} bytes")
        return validated_plots
    
    def _combine_plots_for_legacy(self, plots: List[Dict[str, Any]]) -> str:
        """Create combined HTML string for backward compatibility."""
        if not plots:
            return ""
        
        combined_parts = []
        for plot in plots:
            plot_html = f"<div class='plot-container'><h4>{plot['description']}</h4>{plot['html']}</div>"
            combined_parts.append(plot_html)
        
        return "\n".join(combined_parts)

    def _search_for_cell_discoveries(self, state: ChatState) -> Optional[str]:
        """Search vector database for cell type discoveries relevant to user query."""
        try:
            # Get user's question
            user_question = state.get("current_message", "").lower()

            # Check if this is a "find X cell" type query
            discovery_keywords = ["find", "discover", "identify", "locate", "search for"]
            cell_keywords = ["cell", "cells", "cell type", "subtype"]

            if not any(keyword in user_question for keyword in discovery_keywords):
                return None

            if not any(keyword in user_question for keyword in cell_keywords):
                return None

            logger.info(f"🔍 DISCOVERY SEARCH: Detected discovery query: '{user_question}'")

            # Use the history manager available from BaseWorkflowNode
            if not self.history_manager:
                logger.info("⚠️ No history manager available for discovery search")
                return None

            # Search for cell type discoveries in conversation history
            search_queries = [
                f"discovered new cell type {user_question}",
                f"✅ Discovered new cell type",
                f"cell type discovery {user_question}",
                user_question
            ]

            discovery_results = []
            for query in search_queries:
                try:
                    results = self.history_manager.search_conversations(query, k=3)
                    if results:
                        discovery_results.extend(results)
                except Exception as e:
                    logger.info(f"⚠️ Search query '{query}' failed: {e}")
                    continue

            if not discovery_results:
                logger.info("🔍 No discovery results found in conversation history")
                return None

            # Format discovery context
            discovery_context = self._format_discovery_results(discovery_results, user_question)
            logger.info(f"✅ Found {len(discovery_results)} discovery-related conversations")

            return discovery_context

        except Exception as e:
            logger.info(f"❌ Discovery search failed: {e}")
            return None

    def _format_discovery_results(self, results: List[Any], user_question: str) -> str:
        """Format discovery search results into context string."""
        if not results:
            return ""

        discovery_lines = ["CELL TYPE DISCOVERY HISTORY:"]
        seen_discoveries = set()

        for result in results:
            exchange = result.metadata.get("full_exchange", "")
            if not exchange:
                continue

            # Look for discovery announcements in the exchange
            if "✅ Discovered new cell type" in exchange or "discovered" in exchange.lower():
                # Extract the discovery information
                lines = exchange.split('\n')
                for line in lines:
                    if "✅ Discovered new cell type:" in line or "discovered new cell type" in line.lower():
                        if line not in seen_discoveries:
                            seen_discoveries.add(line)
                            discovery_lines.append(f"  {line.strip()}")

        if len(discovery_lines) > 1:  # More than just the header
            discovery_lines.append("")
            discovery_lines.append("IMPORTANT: Use this discovery information to provide accurate availability status.")
            return "\n".join(discovery_lines)
        else:
            return ""

    def generate_response(self, state: ChatState) -> ChatState:
        """
        Alternative entry point for response generation.
        Delegates to the main unified response generator.
        """
        self._log_node_start("ResponseGenerator", state)
        
        result = self.unified_response_generator_node(state)
        
        self._log_node_complete("ResponseGenerator", state)
        return result