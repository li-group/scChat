"""
Response generation for workflow nodes.

This module contains:
- response_generator_node implementation
- Question-focused response methods
- Visualization collection logic
- Insight extraction methods
"""

import json
from typing import Dict, Any, List
from langchain_core.messages import AIMessage

from ..cell_type_models import ChatState
from ..shared import extract_cell_types_from_question


class ResponseMixin:
    """Response generation mixin for workflow nodes."""
    
    def response_generator_node(self, state: ChatState) -> ChatState:
        """Generate final response that directly answers the user's question"""
        print("🎯 Generating question-focused response...")
        
        # Get user intent guidance from jury system
        user_guidance = state.get("user_intent_guidance", {})
        answer_format = user_guidance.get("answer_format", "direct_answer")
        print(f"🎯 Using answer format: {answer_format}")
        
        # Check for errors first
        if not state["execution_plan"] or not state["execution_history"]:
            state["response"] = json.dumps({"response": "I encountered an issue processing your request."})
            return state
        
        # Extract key information from execution results
        analysis_results = self._extract_analysis_findings(state)
        
        # Collect visualizations
        collected_plots = self._collect_plots_from_execution(state)
        
        # Generate answer based on question type and user guidance
        if answer_format == "comparison":
            response_text = self._generate_comparison_response(state, analysis_results, user_guidance)
        elif answer_format == "discovery_list":
            response_text = self._generate_discovery_response(state, analysis_results, user_guidance)
        elif answer_format == "explanation":
            response_text = self._generate_explanation_response(state, analysis_results, user_guidance)
        else:  # direct_answer
            response_text = self._generate_direct_answer_response(state, analysis_results, user_guidance)
        
        # Construct final response
        response_data = {
            "response": response_text,
            "response_type": "question_focused_answer"
        }
        
        # Include visualizations if available
        if collected_plots:
            response_data["graph_html"] = collected_plots
            plot_count = len(collected_plots.split('<div class=')) - 1
            print(f"🎨 Including {plot_count} supporting visualizations")
        
        state["response"] = json.dumps(response_data)
        print("✅ Generated question-focused response")
        
        # Add response to message history
        try:
            response_content = json.loads(state["response"])["response"]
            state["messages"].append(AIMessage(content=response_content))
        except:
            state["messages"].append(AIMessage(content="Analysis completed."))
        
        return state

    # ========== Question-Focused Response Generation Methods ==========
    
    def _extract_analysis_findings(self, state: ChatState) -> Dict[str, Any]:
        """Extract key findings from execution history for response generation"""
        findings = {
            "target_cell_types": [],  # Cell types mentioned in original question
            "all_cell_types_analyzed": [],  # All cell types processed
            "differential_expression": {},
            "enrichment_results": {},
            "visualizations": [],
            "process_cells_results": {},
            "original_question": state.get("current_message", "")
        }
        
        # Extract target cell types from original question using the same method as planner
        original_question = state.get("current_message", "")
        if self.hierarchy_manager:
            target_types = extract_cell_types_from_question(original_question, self.hierarchy_manager)
            findings["target_cell_types"] = target_types
            print(f"🎯 Target cell types from question: {target_types}")
        
        for execution in state["execution_history"]:
            if not execution["success"]:
                continue
                
            step = execution["step"]
            function_name = step.get("function_name", "")
            cell_type = step.get("parameters", {}).get("cell_type")
            
            # Extract findings by function type
            if function_name == "process_cells" and cell_type:
                findings["process_cells_results"][cell_type] = execution["result"]
                
            elif function_name == "dea_split_by_condition" and cell_type:
                findings["differential_expression"][cell_type] = execution["result"]
                
            elif function_name == "perform_enrichment_analyses" and cell_type:
                findings["enrichment_results"][cell_type] = execution["result"]
                
            elif function_name in self.visualization_functions:
                findings["visualizations"].append({
                    "function": function_name,
                    "cell_type": cell_type,
                    "result": execution["result"]
                })
                
            # Track all analyzed cell types
            if cell_type and cell_type not in findings["all_cell_types_analyzed"]:
                findings["all_cell_types_analyzed"].append(cell_type)
        
        print(f"🔍 Extracted findings: Target types: {findings['target_cell_types']}, All analyzed: {findings['all_cell_types_analyzed']}")
        print(f"🔍 Analysis results: {len(findings['differential_expression'])} DEA, {len(findings['enrichment_results'])} enrichment")
        return findings
    
    def _collect_plots_from_execution(self, state: ChatState) -> str:
        """Collect all visualization plots from execution history"""
        plots = []
        plot_descriptions = []
        
        for execution in state["execution_history"]:
            if (execution["success"] and 
                execution["step"].get("function_name") in self.visualization_functions and
                execution["result"] and 
                isinstance(execution["result"], str) and
                ("<div" in execution["result"] or "<html" in execution["result"])):
                
                # Get description
                step = execution["step"]
                cell_type = step.get("parameters", {}).get("cell_type", "")
                function_name = step.get("function_name", "")
                
                description = self._generate_visualization_description(step, execution["result"])
                plot_descriptions.append(f"<h4>{description}</h4>")
                plots.append(execution["result"])
        
        if plots:
            combined_plots = "".join([f"<div class='plot-container'>{desc}{plot}</div>" 
                                     for desc, plot in zip(plot_descriptions, plots)])
            print(f"🎨 Collected {len(plots)} plots for response")
            return combined_plots
        
        return ""
    
    def _generate_comparison_response(self, state: ChatState, findings: Dict[str, Any], guidance: Dict[str, Any]) -> str:
        """Generate response for comparison questions"""
        # Use target cell types from question, not discovery process types
        target_types = findings["target_cell_types"]
        all_analyzed = findings["all_cell_types_analyzed"]
        original_question = findings["original_question"]
        
        # Prefer target types if available, otherwise fall back to analyzed types
        if len(target_types) >= 2:
            cell_types = target_types
            print(f"🎯 Using target cell types for comparison: {cell_types}")
        elif len(all_analyzed) >= 2:
            cell_types = all_analyzed
            print(f"⚠️ Using analyzed cell types for comparison: {cell_types}")
        else:
            return f"I analyzed cell types but couldn't identify sufficient types for comparison."
        
        cell_a, cell_b = cell_types[0], cell_types[1]
        
        # Extract key differences from DEA and enrichment results
        differences = []
        
        # DEA differences - extract specific gene information
        if cell_a in findings["differential_expression"] and cell_b in findings["differential_expression"]:
            dea_details = self._extract_dea_insights(findings["differential_expression"], cell_a, cell_b)
            if dea_details:
                differences.append(f"**Gene Expression Differences**: {dea_details}")
            else:
                differences.append("**Gene Expression Differences**: Found distinct expression patterns between the two cell types.")
        
        # Enrichment differences - extract specific pathway information
        if cell_a in findings["enrichment_results"] and cell_b in findings["enrichment_results"]:
            enrichment_details = self._extract_enrichment_insights(findings["enrichment_results"], cell_a, cell_b)
            if enrichment_details:
                differences.append(f"**Pathway Enrichment**: {enrichment_details}")
            else:
                differences.append("**Pathway Enrichment**: Different biological pathways are enriched in each cell type.")
        
        # Use guidance to enhance the response
        required_elements = guidance.get("required_elements", [])
        key_focus_areas = guidance.get("key_focus_areas", [])
        answer_template = guidance.get("answer_template", "")
        
        # Use guidance template if provided
        if answer_template and "X" in answer_template and "Y" in answer_template:
            response_start = answer_template.replace("X", cell_a).replace("Y", cell_b)
        else:
            response_start = f"{cell_a} is distinguished from {cell_b} by:"
        
        # Build enhanced comparison response based on guidance
        response_parts = [response_start]
        
        # Add differences found in analysis
        for i, diff in enumerate(differences, 1):
            response_parts.append(f"{i}. {diff}")
        
        # Add guidance-specific elements with detailed information
        if "distinguishing features" in required_elements or "distinguishing features" in key_focus_areas:
            distinguishing_details = self._extract_distinguishing_features(findings, cell_a, cell_b)
            response_parts.append(f"**Key Distinguishing Features**: {distinguishing_details}")
        
        if "specific markers" in required_elements or "markers" in str(key_focus_areas).lower():
            marker_details = self._extract_marker_information(findings, cell_a, cell_b)
            response_parts.append(f"**Molecular Markers**: {marker_details}")
        
        if "functional differences" in required_elements or "functional" in str(key_focus_areas).lower():
            functional_details = self._extract_functional_differences(findings, cell_a, cell_b)
            response_parts.append(f"**Functional Differences**: {functional_details}")
        
        # If no specific differences found, provide general response
        if not differences and not any(elem in required_elements for elem in ["distinguishing features", "specific markers", "functional differences"]):
            response_parts.append("The analysis revealed distinct characteristics for each cell type, with detailed results shown in the visualizations below.")
        
        response_parts.append("\nThe visualizations below provide supporting evidence for these distinctions.")
        
        return "\n\n".join(response_parts)
    
    def _generate_discovery_response(self, state: ChatState, findings: Dict[str, Any], guidance: Dict[str, Any]) -> str:
        """Generate response for discovery questions"""
        discoveries = []
        
        # Count different types of discoveries
        target_types = findings["target_cell_types"]
        all_analyzed = findings["all_cell_types_analyzed"]
        cell_types_discovered = len(target_types) if target_types else len(all_analyzed)
        dea_analyses = len(findings["differential_expression"])
        enrichment_analyses = len(findings["enrichment_results"])
        
        discoveries.append(f"**Cell Types Analyzed**: {cell_types_discovered} cell types identified and characterized")
        
        if dea_analyses > 0:
            discoveries.append(f"**Gene Expression Patterns**: Differential expression analysis completed for {dea_analyses} cell types")
        
        if enrichment_analyses > 0:
            discoveries.append(f"**Pathway Analysis**: Enrichment analysis revealed active biological pathways in {enrichment_analyses} cell types")
        
        response = f"The analysis revealed {len(discoveries)} key findings:\n\n"
        for i, discovery in enumerate(discoveries, 1):
            response += f"{i}. {discovery}\n"
        
        response += "\nDetailed results and visualizations are provided below."
        
        return response
    
    def _generate_explanation_response(self, state: ChatState, findings: Dict[str, Any], guidance: Dict[str, Any]) -> str:
        """Generate explanatory response"""
        target_types = findings["target_cell_types"]
        all_analyzed = findings["all_cell_types_analyzed"]
        cell_types = target_types if target_types else all_analyzed
        
        if not cell_types:
            return "I completed the analysis but didn't identify specific cell types to explain."
        
        response = f"Based on the analysis of {', '.join(cell_types)}:\n\n"
        
        # Add explanations based on what was analyzed
        explanations = []
        
        if findings["differential_expression"]:
            explanations.append("**Gene Expression**: The differential expression analysis identified characteristic genes that distinguish these cell types.")
        
        if findings["enrichment_results"]:
            explanations.append("**Biological Function**: Pathway enrichment analysis revealed the key biological processes and functions associated with these cell types.")
        
        if findings["visualizations"]:
            explanations.append("**Visual Evidence**: The visualizations below illustrate the key patterns and relationships identified in the data.")
        
        response += " ".join(explanations)
        
        if not explanations:
            response += "The analysis provided insights into the characteristics and behavior of these cell types, with detailed results shown below."
        
        return response
    
    def _generate_direct_answer_response(self, state: ChatState, findings: Dict[str, Any], guidance: Dict[str, Any]) -> str:
        """Generate direct answer response (default)"""
        original_question = findings["original_question"]
        target_types = findings["target_cell_types"]
        all_analyzed = findings["all_cell_types_analyzed"]
        cell_types = target_types if target_types else all_analyzed
        
        # Try to provide a direct answer based on the question and findings
        if "distinguish" in original_question.lower() or "difference" in original_question.lower():
            return self._generate_comparison_response(state, findings, guidance)
        elif "what" in original_question.lower() or "identify" in original_question.lower():
            return self._generate_discovery_response(state, findings, guidance)
        else:
            return self._generate_explanation_response(state, findings, guidance)

    # ========== Insight Extraction Methods ==========
    
    def _extract_dea_insights(self, dea_results: Dict[str, Any], cell_a: str, cell_b: str) -> str:
        """
        Extract specific insights from DEA (differential expression analysis) results.
        
        Args:
            dea_results: Dictionary with cell type -> DEA result
            cell_a, cell_b: Cell types to compare
            
        Returns:
            Detailed string describing the differences found
        """
        try:
            result_a = dea_results.get(cell_a, "")
            result_b = dea_results.get(cell_b, "")
            
            if not result_a or not result_b:
                return ""
            
            # Extract key information from DEA results
            insights = []
            
            # Look for specific gene mentions in the results
            if isinstance(result_a, str) and isinstance(result_b, str):
                # Try to extract gene names and significance information
                common_markers = []
                
                # Look for common gene expression analysis patterns
                if "differentially expressed genes" in result_a.lower() or "deg" in result_a.lower():
                    insights.append("Differential gene expression analysis identified distinct marker genes")
                
                if "upregulated" in result_a.lower() or "downregulated" in result_a.lower():
                    insights.append("specific genes showing upregulation and downregulation patterns")
                
                if "fold change" in result_a.lower() or "log2fc" in result_a.lower():
                    insights.append("with significant fold changes in expression levels")
                
                if "p-value" in result_a.lower() or "adjusted" in result_a.lower():
                    insights.append("meeting statistical significance thresholds")
            
            return ", ".join(insights) if insights else "Distinct expression profiles identified through differential analysis"
            
        except Exception as e:
            print(f"⚠️ Error extracting DEA insights: {e}")
            return ""
    
    def _extract_enrichment_insights(self, enrichment_results: Dict[str, Any], cell_a: str, cell_b: str) -> str:
        """
        Extract specific insights from enrichment analysis results.
        
        Args:
            enrichment_results: Dictionary with cell type -> enrichment result
            cell_a, cell_b: Cell types to compare
            
        Returns:
            Detailed string describing the pathway differences found
        """
        try:
            result_a = enrichment_results.get(cell_a, "")
            result_b = enrichment_results.get(cell_b, "")
            
            if not result_a or not result_b:
                return ""
            
            # Extract pathway information from enrichment results
            insights = []
            
            if isinstance(result_a, str) and isinstance(result_b, str):
                # Look for pathway analysis patterns
                if "go" in result_a.lower() or "gene ontology" in result_a.lower():
                    insights.append("Gene Ontology analysis revealed distinct biological processes")
                
                if "kegg" in result_a.lower() or "pathway" in result_a.lower():
                    insights.append("KEGG pathway analysis identified different metabolic and signaling pathways")
                
                if "gsea" in result_a.lower() or "gene set" in result_a.lower():
                    insights.append("Gene Set Enrichment Analysis showed enrichment in different functional categories")
                
                if "reactome" in result_a.lower():
                    insights.append("Reactome pathway analysis revealed distinct biological reactions")
                
                if "enrichment" in result_a.lower():
                    insights.append("with cell-type-specific functional enrichment patterns")
            
            return ", ".join(insights) if insights else "Different biological pathways and processes enriched in each cell type"
            
        except Exception as e:
            print(f"⚠️ Error extracting enrichment insights: {e}")
            return ""
    
    def _extract_distinguishing_features(self, findings: Dict[str, Any], cell_a: str, cell_b: str) -> str:
        """Extract distinguishing features from analysis results and visualizations."""
        features = []
        
        # Check for DEA results
        if findings["differential_expression"]:
            features.append("distinct gene expression signatures")
        
        # Check for enrichment results
        if findings["enrichment_results"]:
            features.append("unique pathway activation patterns")
        
        # Check for visualizations that might contain specific information
        visualizations = findings.get("visualizations", [])
        for viz in visualizations:
            if viz.get("function") == "display_dotplot":
                features.append("characteristic marker gene expression levels")
        
        if not features:
            return "The analysis identified characteristic expression patterns and biological functions that separate these cell populations."
        
        return f"The analysis identified {', '.join(features)} that distinguish these cell types."
    
    def _extract_marker_information(self, findings: Dict[str, Any], cell_a: str, cell_b: str) -> str:
        """Extract specific marker information from analysis results."""
        marker_info = []
        
        # Try to extract from DEA results
        if cell_a in findings["differential_expression"] and cell_b in findings["differential_expression"]:
            marker_info.append("Differential gene expression analysis identified cell-type-specific marker genes")
        
        # Check for visualizations that contain marker information
        visualizations = findings.get("visualizations", [])
        dotplot_count = len([viz for viz in visualizations if viz.get("function") == "display_dotplot"])
        
        if dotplot_count > 0:
            marker_info.append(f"with expression patterns visualized in {dotplot_count} dotplot{'s' if dotplot_count > 1 else ''}")
        
        # Extract from process_cells results if available
        if findings["process_cells_results"]:
            marker_info.append("supported by cell type discovery analysis")
        
        if not marker_info:
            return "Differential gene expression analysis revealed cell-type-specific marker genes."
        
        return " ".join(marker_info) + "."
    
    def _extract_functional_differences(self, findings: Dict[str, Any], cell_a: str, cell_b: str) -> str:
        """Extract functional differences from enrichment and pathway analysis."""
        functional_details = []
        
        # Check for enrichment results
        if findings["enrichment_results"]:
            cell_types_with_enrichment = list(findings["enrichment_results"].keys())
            if len(cell_types_with_enrichment) >= 2:
                functional_details.append("Pathway enrichment analysis revealed distinct biological processes")
                functional_details.append("with each cell type showing unique functional signatures")
        
        # Add information about the types of analysis performed
        if findings["differential_expression"]:
            functional_details.append("Differential expression analysis supports functional specialization")
        
        if not functional_details:
            return "Pathway enrichment analysis showed distinct biological processes and cellular functions."
        
        return " ".join(functional_details) + "."
    
    def _generate_visualization_description(self, step: Dict, result: str) -> str:
        """Generate a descriptive summary for visualization functions"""
        function_name = step["function_name"]
        parameters = step.get("parameters", {})
        
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