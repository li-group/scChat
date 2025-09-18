"""
Planning node implementation.

This module contains the PlannerNode which creates execution plans for user queries
using intelligent LLM-based analysis and cell type discovery.
"""

import json
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ...cell_types.models import ChatState
from ..node_base import BaseWorkflowNode
import logging
logger = logging.getLogger(__name__)


try:
    from ...analysis.enrichment_checker import EnrichmentChecker
except ImportError:
    EnrichmentChecker = None
class PlannerNode(BaseWorkflowNode):
    """
    Planner node that creates intelligent execution plans.
    
    Responsibilities:
    - Analyze user queries for intent and requirements
    - Create step-by-step execution plans
    - Handle cell type discovery and validation
    - Apply enrichment enhancement for analysis steps
    - Optimize plans for efficiency
    """
    
    def execute(self, state: ChatState) -> ChatState:
        return self.planner_node(state)
    
    def planner_node(self, state: ChatState) -> ChatState:
        self._log_node_start("Planner", state)
        
        message = state["current_message"]
        available_functions = self.function_descriptions
        available_cell_types = state["available_cell_types"]
        function_history = state["function_history_summary"]
        unavailable_cell_types = state.get("unavailable_cell_types", [])
        
        initial_count = len(self.initial_cell_types)
        current_count = len(available_cell_types)
        discovered_count = current_count - initial_count
        
        logger.info(f"🧬 PLANNER: Cell type status for planning:")
        logger.info(f"   • Initial types: {initial_count}")
        logger.info(f"   • Currently available: {current_count}")
        logger.info(f"   • Discovered this session: {discovered_count}")
        if unavailable_cell_types:
            logger.info(f"   • Failed discoveries: {len(unavailable_cell_types)} - {', '.join(unavailable_cell_types)}")
        
        if discovered_count > 0:
            discovered_types = set(available_cell_types) - set(self.initial_cell_types)
            logger.info(f"   • New types discovered: {', '.join(sorted(discovered_types))}")
        
        logger.info(f"🧬 Planning for question: '{message}'")
        
        plan_result = self._create_enhanced_plan(state, message, available_functions, available_cell_types, function_history, unavailable_cell_types)
        
        self._log_node_complete("Planner", state)
        logger.info ("FINAL PLAN : ", plan_result)
        return plan_result
    
    def _create_enhanced_plan(self, state: ChatState, message: str, available_functions: List, available_cell_types: List[str], function_history: Dict, unavailable_cell_types: List[str]) -> ChatState:
        
        conversation_context = ""
        has_conversation_context = state.get("has_conversation_context", False)
        if has_conversation_context:
            for msg in state.get("messages", []):
                if hasattr(msg, 'content') and msg.content.startswith("CONVERSATION_CONTEXT:"):
                    conversation_context = msg.content[len("CONVERSATION_CONTEXT: "):]
                    break
        
        planning_prompt = self._build_planning_prompt(
            message, available_cell_types, unavailable_cell_types, 
            available_functions, function_history, conversation_context
        )
        
        try:
            messages = [
                SystemMessage(content="You are a bioinformatics analysis planner. Generate execution plans in JSON format."),
                HumanMessage(content=planning_prompt)
            ]
            
            model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            
            response = model.invoke(messages)
            plan_data = json.loads(response.content)
            
            enhanced_plan = self._process_plan(plan_data, message, available_cell_types, unavailable_cell_types, state)
            
            state["execution_plan"] = enhanced_plan
            state["execution_plan"]["original_question"] = message
            
            self._log_plan_statistics(enhanced_plan)
            
        except Exception as e:
            logger.info(f"Planning error: {e}")
            state["execution_plan"] = self._create_fallback_plan()
            
        return state
    
    def _build_planning_prompt(self, message: str, available_cell_types: List[str], 
                             unavailable_cell_types: List[str], available_functions: List,
                             function_history: Dict, conversation_context: str) -> str:
        return f"""
                You are an intelligent planner for single-cell RNA-seq analysis. 
                
                Create a step-by-step execution plan for the user query.
                
                CONTEXT:
                - Currently available cell types ({len(available_cell_types)}): {', '.join(sorted(available_cell_types))}
                {f"- Cell types that failed discovery ({len(unavailable_cell_types)}): {', '.join(sorted(unavailable_cell_types))}" if unavailable_cell_types else "- No failed cell type discoveries"}
                - Cell type status: {'Expanded from initial set' if len(available_cell_types) > len(self.initial_cell_types) else 'Using initial cell types only'}
                - Previous analyses: {json.dumps(function_history, indent=2)}
                {"- Conversation context: " + conversation_context if conversation_context else ""}
                
                Available functions:
                {self._summarize_functions(available_functions)}
                
                User question: "{message}"
                
                SEMANTIC DECISION FRAMEWORK:
                GENE/MARKER ANALYSIS - Use DEA functions when the user asks about:
                - Gene expression differences between conditions/cell types
                - Marker genes, differentially expressed genes
                - Gene-level comparisons and signatures
                - Examples: "What genes are upregulated in T cells?", "Find marker genes for B cells", "Which genes differ between conditions?"
                - → Use functions like: dea_split_by_condition, compare_cell_counts
                
                PATHWAY ANALYSIS - Use enrichment functions when the user asks about:
                - Biological pathways, gene sets, functional categories
                - Pathway enrichment, functional analysis
                - Systems-level biological processes
                - Examples: "What pathways are enriched?", "Find interferon pathways", "Analyze GO terms"
                - → Use functions like: perform_enrichment_analyses
                
                CELL TYPE ANALYSIS - Use processing/counting functions when the user asks about:
                - Cell type abundance, proportions, distributions
                - Cell type identification and characterization
                - Cell count comparisons and visualizations
                - Examples: "How many T cells are there?", "Compare cell type proportions", "Show cell count plots"
                - → Use functions like: process_cells, compare_cell_counts, display_cell_count_stacked_plot
                
                CONVERSATIONAL RESPONSE - For simple queries:
                - Direct greetings or simple questions
                - When no analysis is needed
                - Examples: "Hi", "Hello", "What can you do?"
                - → Response will be generated by LLM synthesis
                
                ANALYSIS SELECTION GUIDE:
                - Gene/marker questions → DEA analysis
                - Pathway/functional questions → Enrichment analysis
                - Cell abundance/counting questions → Cell processing/counting + display_cell_count_stacked_plot
                - Cell count visualization requests → display_cell_count_stacked_plot (direct visualization)
                - Explanation/interpretation questions → Conversational response
                - When unclear, consider what type of biological question is being asked
                
                Create a plan in this JSON format:
                {{
                    "plan_summary": "Brief description of how you'll answer this question",
                    "visualization_only": true/false,
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
                - SKIP steps for unavailable cell types: {', '.join(unavailable_cell_types)}
                - CELL TYPE STRATEGY: If a requested cell type is not in the available list, consider if it needs discovery
                - AVAILABLE TYPES PRIORITY: Prefer using currently available cell types when possible
                - CRITICAL STEP ORDER: For DEA heatmaps, ALWAYS put dea_split_by_condition BEFORE display_dea_heatmap for the same cell type
                - Example: Step 1: dea_split_by_condition(cell_type="T cell"), Step 2: display_dea_heatmap(cell_type="T cell")
                - Focus on creating a logical flow to answer the user's question
                
                VISUALIZATION-ONLY DETECTION:
                - Set "visualization_only": true if the user ONLY wants to see plots/visualizations
                - Examples of visualization-only requests:
                * "Show the GSEA barplot for T cell"
                * "Display the GO dotplot"
                * "Plot the enrichment results"
                * "Generate UMAP visualization"
                - Set "visualization_only": false for analysis requests or questions needing interpretation:
                * "What pathways are enriched in T cells?"
                * "Run GSEA analysis and explain results"
                * "Compare T cell vs B cell enrichment"
                
                ENRICHMENT ANALYSIS GUIDELINES:
                - For enrichment analysis steps, use MINIMAL parameters - only specify "cell_type"
                - Do NOT specify "analyses", "gene_set_library", or "pathway_include" parameters
                - The EnrichmentChecker will automatically determine optimal analysis methods and parameters
                - Examples:
                * "Run GSEA analysis on T cells" → {{"cell_type": "T cell"}} (EnrichmentChecker adds analyses)
                * "Find pathways enriched in B cells" → {{"cell_type": "B cell"}} (EnrichmentChecker determines methods)
                * "Analyze immune pathways" → {{"cell_type": "Immune cell"}} (EnrichmentChecker handles targeting)
                
                VISUALIZATION GUIDELINES:
                - For enrichment visualization, ALWAYS use "display_enrichment_visualization":
                * Use "display_enrichment_visualization" for ALL enrichment plots (shows both bar + dot by default)
                * ALWAYS specify the "analysis" parameter to match what was performed
                * Use "plot_type" parameter to control visualization: "both" (default), "bar", or "dot"
                * Examples:
                    - "show GO plots" → use "display_enrichment_visualization" with "analysis": "go"
                    - "visualize GSEA results" → use "display_enrichment_visualization" with "analysis": "gsea"  
                    - "show ONLY barplot" → use "display_enrichment_visualization" with "plot_type": "bar"
                    - "show ONLY dotplot" → use "display_enrichment_visualization" with "plot_type": "dot"
                    - "display both plots" → use "display_enrichment_visualization" (default plot_type="both")
                
                - For cell count comparison visualization, use "display_cell_count_comparison":
                * Use AFTER multiple "compare_cell_counts" steps have been executed to show aggregate results
                * Aggregates results from multiple cell types into a single comparative visualization
                * Examples:
                    - After comparing T cell, B cell, Macrophages counts → create visualization step to show all together
                    - "show cell count comparison" → use "display_cell_count_comparison" with results from previous steps
                    - "visualize cell abundance across conditions" → use "display_cell_count_comparison"
                * The "cell_types_data" parameter should contain aggregated results from executed compare_cell_counts steps

                - For cell count stacked bar plot visualization, use "display_cell_count_stacked_plot":
                * Creates stacked bar plots comparing cell counts across conditions for multiple cell types
                * Use when user wants to see patient-specific treatment comparisons or cell type abundance changes
                * Works directly with live data - NO need for prior compare_cell_counts steps
                * Examples:
                    - "create stacked bar plot for T cells, B cells, and Macrophages" → use "display_cell_count_stacked_plot" with cell_types=["T cell", "B cell", "Macrophage"]
                    - "show cell count comparison across patients" → use "display_cell_count_stacked_plot" with relevant cell types
                    - "plot cell abundance by treatment" → use "display_cell_count_stacked_plot" with cell types of interest
                    - "compare cell counts in a stacked plot" → use "display_cell_count_stacked_plot"
                * Parameters: cell_types (required list) - can accept plural forms like "T cells", "B cells"
                * Automatically extracts patient and treatment information from metadata
                * Shows cell types on x-axis with stacked bars by patient-treatment combinations
                
                - For DEA heatmap visualization, use "display_dea_heatmap":
                * CRITICAL: Use IMMEDIATELY AFTER "dea_split_by_condition" step for the same cell type
                * Creates heatmap showing gene expression patterns across conditions for one cell type
                * CORRECT ORDER Examples:
                    - Step N: dea_split_by_condition(cell_type="T cell"), Step N+1: display_dea_heatmap(cell_type="T cell") 
                    - Step N: dea_split_by_condition(cell_type="B cell"), Step N+1: display_dea_heatmap(cell_type="B cell")
                * NEVER put display_dea_heatmap BEFORE the corresponding dea_split_by_condition
                * For multiple cell types, create pairs: analysis → heatmap for each cell type
                * Parameters: cell_type (required), top_n_genes (default: 20), cluster_genes (default: true), cluster_samples (default: true)

                - For gene expression visualization on UMAP, use "display_feature_plot":
                * Creates interactive scatter plots showing gene expression overlaid on UMAP coordinates
                * Use when user wants to see WHERE genes are expressed across the cell landscape
                * Examples:
                    - "show CD3E expression on UMAP" → use "display_feature_plot" with genes=["CD3E"]
                    - "visualize T cell markers on the map" → use "display_feature_plot" with genes=["CD3E", "CD4", "CD8A"]
                    - "plot gene expression patterns" → use "display_feature_plot" with relevant genes
                * Parameters: genes (required list), cell_type_filter (optional to highlight specific cell types)
                * Can plot single or multiple genes in subplot layout

                - For gene expression distribution analysis, use "display_violin_plot":
                * Creates interactive violin plots showing expression across leiden clusters with treatment comparison
                * Use when user wants to compare gene expression LEVELS across clusters or conditions
                * Examples:
                    - "compare IL32 expression across T cell clusters" → use "display_violin_plot" with cell_type="T cell", genes=["IL32"]
                    - "show treatment differences in gene expression" → use "display_violin_plot" with relevant cell_type and genes
                    - "violin plot for marker genes" → use "display_violin_plot" with cell_type and marker gene list
                * Parameters: cell_type (required), genes (required list)
                * Automatically shows pre/post treatment comparison if available
                * Uses hierarchical cell type discovery (e.g., Treg → finds T cell data)

                - For UMAP visualization colored by leiden clusters, use "display_leiden_umap":
                * Creates interactive UMAP plots colored by leiden cluster assignments instead of cell types
                * Use when user wants to see CLUSTER organization rather than cell type annotation
                * Examples:
                    - "show leiden clusters on UMAP" → use "display_leiden_umap" with cell_type for file discovery
                    - "visualize cluster organization" → use "display_leiden_umap"
                    - "display UMAP colored by clusters" → use "display_leiden_umap"
                    - "show T cell leiden clusters" → use "display_leiden_umap" with cell_type="T cell"
                * Parameters: cell_type (required for hierarchical file discovery)
                * Uses same intelligent file discovery as other UMAP functions
                * Colors points by leiden cluster numbers with Seurat discrete palette

                - For comprehensive global UMAP visualization, use "display_overall_umap":
                * Creates UMAP plot with complete dataset view showing entire cellular landscape
                * Use when user wants to see the GLOBAL perspective or complete dataset overview
                * Two color modes: biological (cell types) or computational (consolidated leiden clusters)
                * Examples:
                    - "show overall UMAP" → use "display_overall_umap" with color_mode="cell_type"
                    - "display global cell landscape" → use "display_overall_umap" with color_mode="cell_type"
                    - "show consolidated leiden clusters" → use "display_overall_umap" with color_mode="accumulative_leiden"
                    - "global clustering view" → use "display_overall_umap" with color_mode="accumulative_leiden"
                    - "complete cellular overview" → use "display_overall_umap"
                * Parameters: color_mode (optional, defaults to "cell_type")
                * Always uses complete dataset for comprehensive view
                * Consolidates cell types into unified clusters when using accumulative_leiden mode

                SEMANTIC SEARCH GUIDELINES:
                - For questions seeking specific pathway/term information beyond the top-ranked results, consider using "search_enrichment_semantic"
                - Use semantic search when:
                * User asks about specific pathways that might not be in top results (e.g., "cell cycle regulation", "apoptosis pathways")
                * User references conversation context about previous analyses and wants to explore related pathways
                * User wants to find terms similar to those mentioned in conversation context
                - Parameters for search_enrichment_semantic:
                * "query": the pathway/term to search for (e.g., "cell cycle regulation")
                * "cell_type": target cell type (can be inferred from conversation context if not explicitly mentioned)
                * Optional: "analysis_type", "condition", "limit"
                - Examples:
                * "Show me cell cycle related terms from the T cell analysis" → search_enrichment_semantic with query="cell cycle" and cell_type="T cell"
                * "Are there any apoptosis pathways in our results?" → search_enrichment_semantic with query="apoptosis"
                * "Find pathways similar to what we discussed earlier" → use conversation context to determine relevant search terms
                """
    
    def _process_plan(self, plan_data: Dict[str, Any], message: str,
                     available_cell_types: List[str], unavailable_cell_types: List[str], state: ChatState) -> Dict[str, Any]:
        enhanced_plan = self._add_cell_discovery_to_plan(plan_data, message, available_cell_types)
        
        enrichment_steps = [s for s in enhanced_plan.get('steps', []) if s.get('function_name') == 'perform_enrichment_analyses']
        logger.info(f"🔍 ENRICHMENT DEBUG: Found {len(enrichment_steps)} enrichment steps")
        
        # Extract pathway keywords and enhance enrichment steps
        enhanced_plan = self._extract_pathway_keywords_from_enrichment_steps(enhanced_plan, message, state.get("execution_history", []))
        
        # Skip steps for unavailable cell types
        if unavailable_cell_types:
            enhanced_plan = self._skip_unavailable_cell_steps(enhanced_plan, unavailable_cell_types)
        
        # Plan processing will be handled by EvaluatorNode
        
        return enhanced_plan
    
    
    def _summarize_functions(self, functions: List[Dict]) -> str:
        if not functions:
            return "No functions available"
        
        summary = []
        for func in functions:
            name = func.get("name", "unknown")
            description = func.get("description", "").split(".")[0]  # First sentence only
            summary.append(f"- {name}: {description}")
        
        return "\n".join(summary)
    
    def _add_cell_discovery_to_plan(self, plan_data: Dict[str, Any], message: str, available_cell_types: List[str]) -> Dict[str, Any]:
        from ...cell_types.validation import extract_cell_types_from_question, needs_cell_discovery
        
        if not plan_data or not self.hierarchy_manager:
            logger.info("🔍 Cell discovery: No hierarchy manager available")
            return plan_data
        
        # Extract cell types mentioned in the user's question
        needed_cell_types = extract_cell_types_from_question(message, self.hierarchy_manager)
        
        if not needed_cell_types:
            logger.info("🔍 No specific cell types identified in question")
            return plan_data
        
        logger.info(f"🧬 Planner identified needed cell types: {needed_cell_types}")
        logger.info(f"🧬 Available cell types: {available_cell_types}")
        
        # Step 1: Extract and categorize original steps
        original_steps = plan_data.get("steps", [])
        llm_analysis_steps = []
        other_steps = []
        
        for step in original_steps:
            func_name = step.get("function_name", "")
            # Preserve cell-type-specific analysis steps
            if func_name in ["perform_enrichment_analyses", "dea_split_by_condition", 
                           "compare_cell_counts", "analyze_cell_interaction"]:
                llm_analysis_steps.append(step)
                logger.info(f"📋 Preserving LLM analysis step: {func_name}({step.get('parameters', {}).get('cell_type', 'unknown')})")
            else:
                other_steps.append(step)
        
        # Step 2: Fix cell type names in preserved steps
        llm_analysis_steps = self._fix_cell_type_names_in_steps(llm_analysis_steps, needed_cell_types, message)
        other_steps = self._fix_cell_type_names_in_steps(other_steps, needed_cell_types, message)
        
        # Step 3: Filter out steps for undiscoverable cell types BEFORE creating discovery steps
        discoverable_types = []
        undiscoverable_types = []
        
        for needed_type in needed_cell_types:
            if needed_type in available_cell_types:
                discoverable_types.append(needed_type)
                logger.info(f"✅ '{needed_type}' already available")
            else:
                # Check if this type can be discovered from available types
                can_discover = False
                for available_type in available_cell_types:
                    path_result = self.hierarchy_manager.find_parent_path(needed_type, [available_type])
                    if path_result:
                        can_discover = True
                        discoverable_types.append(needed_type)
                        break
                
                if not can_discover:
                    undiscoverable_types.append(needed_type)
                    logger.info(f"❌ '{needed_type}' cannot be discovered from available cell types")
        
        # Step 4: Remove analysis steps for undiscoverable cell types
        if undiscoverable_types:
            logger.info(f"🚫 Filtering out analysis steps for undiscoverable types: {undiscoverable_types}")
            filtered_analysis_steps = []
            for step in llm_analysis_steps:
                step_cell_type = step.get("parameters", {}).get("cell_type")
                if step_cell_type not in undiscoverable_types:
                    filtered_analysis_steps.append(step)
                else:
                    logger.info(f"🚫 Removed step: {step.get('function_name')}({step_cell_type})")
            llm_analysis_steps = filtered_analysis_steps
        
        # Step 5: Create discovery steps ONLY for discoverable types
        discovery_steps = []
        if needs_cell_discovery(discoverable_types, available_cell_types):
            logger.info("🧬 Creating discovery steps only...")
            discovery_steps = self._create_discovery_steps_only(discoverable_types, available_cell_types)
        
        # Step 6: Update analysis steps to use discovered cell types
        updated_analysis_steps = self._update_analysis_steps_for_discovered_types(
            llm_analysis_steps, discoverable_types, available_cell_types
        )
        
        # Step 7: Create interleaved discovery + validation sequence
        final_steps = []
        
        # Step 7a: Add discovery steps with immediate validation
        validation_steps = self._create_validation_steps(discovery_steps)
        
        # Create mapping of process_cells to their validation steps
        validation_map = {}
        for val_step in validation_steps:
            processed_parent = val_step.get("parameters", {}).get("processed_parent")
            if processed_parent:
                validation_map[processed_parent] = val_step
        
        # Step 7b: Interleave process_cells with validation steps
        for disc_step in discovery_steps:
            final_steps.append(disc_step)  # Add process_cells step
            
            # Immediately add corresponding validation step
            cell_type = disc_step.get("parameters", {}).get("cell_type")
            if cell_type in validation_map:
                final_steps.append(validation_map[cell_type])
                logger.info(f"🔄 Interleaved: process_cells({cell_type}) → validation({cell_type})")
        
        # Step 8: Add analysis and other steps
        final_steps.extend(updated_analysis_steps)
        final_steps.extend(other_steps)
        
        plan_data["steps"] = final_steps
        
        # Update plan summary
        if discovery_steps:
            original_summary = plan_data.get("plan_summary", "")
            plan_data["plan_summary"] = f"Discover needed cell types then {original_summary.lower()}"
        
        logger.info(f"📋 Plan merge complete:")
        logger.info(f"   - Discovery steps: {len(discovery_steps)}")
        logger.info(f"   - Validation steps: {len(validation_steps)}")
        logger.info(f"   - Analysis steps: {len(updated_analysis_steps)}")
        logger.info(f"   - Other steps: {len(other_steps)}")
        logger.info(f"   - Filtered out: {len(llm_analysis_steps) - len(updated_analysis_steps)} steps for undiscoverable types")
        
        return plan_data
    
    def _fix_cell_type_names_in_steps(self, steps: List[Dict], needed_cell_types: List[str], message: str) -> List[Dict]:
        if not steps or not needed_cell_types:
            return steps
        
        fixed_steps = []
        # Track which needed cell types have been used
        used_cell_types = set()
        
        for step in steps:
            fixed_step = step.copy()
            params = fixed_step.get("parameters", {})
            
            if "cell_type" in params:
                original_name = params["cell_type"]
                
                        if original_name in needed_cell_types:
                    corrected_name = original_name
                    used_cell_types.add(corrected_name)
                else:
                    corrected_name = None
                    for needed_type in needed_cell_types:
                        if needed_type not in used_cell_types:
                            corrected_name = needed_type
                            used_cell_types.add(corrected_name)
                            break
                    
                    if corrected_name is None:
                        corrected_name = original_name
                
                if corrected_name != original_name:
                    fixed_step["parameters"] = params.copy()
                    fixed_step["parameters"]["cell_type"] = corrected_name
                    logger.info(f"🔧 Fixed cell type name using RAG-validated type: '{original_name}' → '{corrected_name}'")
            
            fixed_steps.append(fixed_step)
        
        return fixed_steps
    
    def _create_discovery_steps_only(self, needed_cell_types: List[str], available_cell_types: List[str]) -> List[Dict[str, Any]]:
        discovery_steps = []
        parent_to_children = {}
        
        for needed_type in needed_cell_types:
            if needed_type in available_cell_types:
                logger.info(f"✅ '{needed_type}' already available, no discovery needed")
                continue
            
            processing_path = None
            best_parent = None
            
            for available_type in available_cell_types:
                path_result = self.hierarchy_manager.find_parent_path(needed_type, [available_type])
                if path_result:
                    best_parent, processing_path = path_result
                    logger.info(f"🔄 Found path from '{best_parent}' to '{needed_type}': {' → '.join(processing_path)}")
                    break
            
            if processing_path and len(processing_path) > 1:
                for i in range(len(processing_path) - 1):
                    parent_type = processing_path[i]
                    child_type = processing_path[i + 1]
                    
                    if parent_type not in parent_to_children:
                        parent_to_children[parent_type] = []
                    
                    if child_type not in parent_to_children[parent_type]:
                        parent_to_children[parent_type].append(child_type)
            else:
                logger.info(f"⚠️ No processing path found for '{needed_type}'")
        
        for parent_type, expected_children in parent_to_children.items():
            existing = any(
                s.get("function_name") == "process_cells" and 
                s.get("parameters", {}).get("cell_type") == parent_type
                for s in discovery_steps
            )
            
            if not existing:
                discovery_steps.append({
                    "step_type": "analysis",
                    "function_name": "process_cells",
                    "parameters": {"cell_type": parent_type},
                    "description": f"Process {parent_type} to discover {', '.join(expected_children)}",
                    "expected_outcome": f"Discover {', '.join(expected_children)} cell type(s)",
                    "expected_children": expected_children
                })
                logger.info(f"🧬 Added process_cells({parent_type}) to discover {expected_children}")
        
        return discovery_steps
    
    def _find_best_parent_for_discovery(self, target_type: str, available_types: List[str]) -> str:
        if not self.hierarchy_manager:
            return None
        
        for available_type in available_types:
            path_result = self.hierarchy_manager.find_parent_path(target_type, [available_type])
            if path_result:
                best_parent, processing_path = path_result
                if processing_path and len(processing_path) > 1:
                    return processing_path[0]
        
        return None
    
    def _update_analysis_steps_for_discovered_types(self, analysis_steps: List[Dict], needed_cell_types: List[str], available_cell_types: List[str]) -> List[Dict]:
        updated_steps = []
        
        for step in analysis_steps:
            step_cell_type = step.get("parameters", {}).get("cell_type")
            
            if step_cell_type in available_cell_types:
                types_to_discover_from_parent = [
                    needed_type for needed_type in needed_cell_types 
                    if needed_type not in available_cell_types
                ]
                
                if types_to_discover_from_parent:
                    for discovered_type in types_to_discover_from_parent:
                        updated_step = step.copy()
                        updated_step["parameters"] = step["parameters"].copy()
                        updated_step["parameters"]["cell_type"] = discovered_type
                        
                        if "step_type" not in updated_step:
                            updated_step["step_type"] = "analysis"
                        
                        original_desc = step.get("description", "")
                        updated_desc = original_desc.replace(step_cell_type, discovered_type)
                        updated_step["description"] = updated_desc
                        
                        updated_steps.append(updated_step)
                        logger.info(f"🔄 Updated analysis step: {step.get('function_name')}({step_cell_type}) → {step.get('function_name')}({discovered_type})")
                else:
                    if "step_type" not in step:
                        step["step_type"] = "analysis"
                    updated_steps.append(step)
            else:
                if "step_type" not in step:
                    step["step_type"] = "analysis"
                updated_steps.append(step)
        
        return updated_steps
    
    
    
    def _skip_unavailable_cell_steps(self, plan: Dict[str, Any], unavailable_cell_types: List[str]) -> Dict[str, Any]:
        if not unavailable_cell_types:
            return plan
        
        original_steps = plan.get("steps", [])
        filtered_steps = []
        
        for step in original_steps:
            target_cell_type = step.get("target_cell_type")
            cell_type_param = step.get("parameters", {}).get("cell_type")
            
            should_skip = False
            for unavailable_type in unavailable_cell_types:
                if (target_cell_type and unavailable_type.lower() in target_cell_type.lower()) or \
                   (cell_type_param and unavailable_type.lower() in cell_type_param.lower()):
                    should_skip = True
                    break
            
            if should_skip:
                logger.info(f"⏭️ Skipping step for unavailable cell type: {step.get('description', 'Unknown step')}")
            else:
                filtered_steps.append(step)
        
        plan["steps"] = filtered_steps
        return plan
    
    def _create_fallback_plan(self) -> Dict[str, Any]:
        return {
            "plan_summary": "Direct response generation",
            "visualization_only": False,
            "steps": []  # Empty steps trigger immediate response generation
        }
    
    def _create_validation_steps(self, discovery_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create validation steps for discovery steps.
        
        These validation steps will check if expected cell types were discovered
        and update remaining plan steps accordingly.
        """
        validation_steps = []
        
        for step in discovery_steps:
            if step.get("function_name") == "process_cells":
                cell_type = step.get("parameters", {}).get("cell_type")
                expected_children = step.get("expected_children", [])
                
                if cell_type and expected_children:
                    validation_step = {
                        "step_type": "validation",
                        "function_name": "validate_processing_results",
                        "parameters": {
                            "processed_parent": cell_type,
                            "expected_children": expected_children
                        },
                        "description": f"Validate that {cell_type} processing discovered expected cell types: {', '.join(expected_children)}",
                        "expected_outcome": f"Confirm {', '.join(expected_children)} are available and update remaining steps",
                        "target_cell_type": None
                    }
                    validation_steps.append(validation_step)
                    logger.info(f"🔍 Created validation step for process_cells({cell_type}) expecting: {expected_children}")
        
        return validation_steps
    
    def _extract_pathway_keywords_from_enrichment_steps(self, plan_data: Dict[str, Any], message: str, execution_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"🔍 PATHWAY EXTRACTOR: Processing {len(plan_data.get('steps', []))} steps")
        enhanced_steps = []
        
        for i, step in enumerate(plan_data.get("steps", [])):
            if step.get("function_name") == "perform_enrichment_analyses":
                logger.info(f"🔍 PATHWAY EXTRACTOR: Extracting keywords for step {i+1}: {step.get('parameters', {}).get('cell_type', 'unknown')}")
                # Extract pathway keywords for this enrichment step
                enhanced_step = self._extract_pathway_keywords(step, message, execution_history)
                enhanced_steps.append(enhanced_step)
            else:
                # Keep non-enrichment steps as-is
                enhanced_steps.append(step)
        
        plan_data["steps"] = enhanced_steps
        return plan_data
    
    def _extract_pathway_keywords(self, step: Dict[str, Any], message: str, execution_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract pathway keywords and intelligently decide whether to use enrichment_checker.

        This function uses LLM to classify user intent and routes appropriately:
        - Explicit methods: Direct execution (bypass vector search)
        - Pathway queries: Use enrichment_checker with vector search
        - General requests: Default GO analysis

        Args:
            step: Enrichment step from planner with minimal parameters
            execution_history: Previous execution steps to avoid redundant analyses
            message: Original user message for context

        Returns:
            Step with proper analyses and gene_set_library parameters
        """
        try:
            # Get LLM analysis (now includes intent classification)
            llm_result = self._call_llm_for_pathway_extraction(message)

            enhanced_step = step.copy()
            if "parameters" not in enhanced_step:
                enhanced_step["parameters"] = {}

            # Clean and normalize LLM result
            llm_result_clean = llm_result.strip().strip('"').strip("'")
            logger.info(f"🧠 LLM result after cleaning: '{llm_result_clean}'")

            if llm_result_clean.startswith("EXPLICIT_METHOD:"):
                # User explicitly requested specific methods - bypass vector search
                method_part = llm_result_clean[len("EXPLICIT_METHOD:"):]

                # Check execution history for redundant enrichment analysis
                current_cell_type = enhanced_step["parameters"].get("cell_type", "unknown")
                existing_enrichment = self._check_existing_enrichment_analysis(execution_history, current_cell_type)

                if existing_enrichment:
                    logger.info(f"📋 HISTORY CHECK: Found existing enrichment for {current_cell_type} - skipping enrichment checker for explicit method")
                    # Use the analyses and parameters from the existing enrichment
                    enhanced_step["parameters"]["analyses"] = existing_enrichment.get("analyses", ["go"])
                    if existing_enrichment.get("gene_set_library"):
                        enhanced_step["parameters"]["gene_set_library"] = existing_enrichment["gene_set_library"]
                    if existing_enrichment.get("gene_set_libraries"):
                        enhanced_step["parameters"]["gene_set_libraries"] = existing_enrichment["gene_set_libraries"]
                    enhanced_step["description"] = f"Reuse existing enrichment analysis results for {current_cell_type}"
                    logger.info(f"✅ Bypassed enrichment checker, using existing: {existing_enrichment.get('analyses', ['go'])}")
                    return enhanced_step

                if ":" in method_part and method_part.split(":")[0] == "gsea":
                    # GSEA with specific library: "gsea:MSigDB_Hallmark_2020" or "gsea:hallmark"
                    parts = method_part.split(":")
                    analyses = [parts[0]]
                    user_library_query = parts[1] if len(parts) > 1 else None

                    # Use vector search to find the correct gene set library name
                    gene_set_library = None
                    if user_library_query and self.enrichment_checker_available and self.enrichment_checker:
                        logger.info(f"🔍 Searching for gene set library matching '{user_library_query}'")
                        library_matches = self.enrichment_checker._search_gene_set_library(user_library_query)

                        if library_matches:
                            # For explicit method requests, use only the first/best match (user was specific)
                            gene_set_library = library_matches[0]
                            logger.info(f"✅ Selected best match: '{gene_set_library}'")
                        else:
                            logger.info(f"⚠️ Could not find gene set library for '{user_library_query}', using default")
                            gene_set_library = "MSigDB_Hallmark_2020"  # Fallback
                    elif user_library_query:
                        # If enrichment_checker not available, try direct use (might be exact name)
                        gene_set_library = user_library_query
                else:
                    # Regular methods: "go,kegg"
                    analyses = [m.strip() for m in method_part.split(",")]
                    gene_set_library = None

                enhanced_step["parameters"]["analyses"] = analyses
                if gene_set_library:
                    enhanced_step["parameters"]["gene_set_library"] = gene_set_library

                logger.info(f"🎯 Explicit method request detected: {analyses}")
                if gene_set_library:
                    logger.info(f"   Gene set library: {gene_set_library}")

                # Use enrichment_checker only for description enhancement (no vector search)
                if self.enrichment_checker_available and self.enrichment_checker:
                    enhanced_step = self.enrichment_checker._enhance_explicit_analysis_plan(
                        enhanced_step,
                        enhanced_step["parameters"].get("cell_type", "unknown"),
                        analyses
                    )

            elif llm_result_clean and not llm_result_clean.startswith("EXPLICIT_METHOD:"):
                # User mentioned specific pathways - use vector search & enrichment_checker
                pathway_keywords = llm_result_clean

                if self.enrichment_checker_available and self.enrichment_checker:
                    logger.info(f"🔍 Pathway-specific request: Using enrichment_checker for '{pathway_keywords}'")

                    # Check execution history for redundant enrichment analysis
                    current_cell_type = enhanced_step["parameters"].get("cell_type", "unknown")
                    existing_enrichment = self._check_existing_enrichment_analysis(execution_history, current_cell_type)

                    if existing_enrichment:
                        logger.info(f"📋 HISTORY CHECK: Found existing enrichment for {current_cell_type} - using previous results")
                        # Use the analyses and parameters from the existing enrichment
                        enhanced_step["parameters"]["analyses"] = existing_enrichment.get("analyses", ["go"])
                        if existing_enrichment.get("gene_set_library"):
                            enhanced_step["parameters"]["gene_set_library"] = existing_enrichment["gene_set_library"]
                        if existing_enrichment.get("gene_set_libraries"):
                            enhanced_step["parameters"]["gene_set_libraries"] = existing_enrichment["gene_set_libraries"]
                        enhanced_step["description"] = f"Use existing enrichment analysis results for {current_cell_type}"
                        logger.info(f"✅ Using existing enrichment: {existing_enrichment.get('analyses', ['go'])}")
                    else:
                        # Set pathway_include to trigger enrichment_checker
                        enhanced_step["parameters"]["pathway_include"] = pathway_keywords

                        # Call enrichment_checker to get proper analyses and gene_set_library
                        enhanced_step = self.enrichment_checker.enhance_enrichment_plan(enhanced_step)

                    # Log the enhancement
                    analyses = enhanced_step["parameters"].get("analyses", [])
                    gene_set_library = enhanced_step["parameters"].get("gene_set_library")
                    logger.info(f"✅ EnrichmentChecker enhanced step:")
                    logger.info(f"   • analyses: {analyses}")
                    if gene_set_library:
                        logger.info(f"   • gene_set_library: {gene_set_library}")
                else:
                    logger.info(f"⚠️ Pathway keywords extracted but enrichment_checker not available")
                    enhanced_step["parameters"]["analyses"] = ["go"]

            else:
                # General enrichment request or no keywords - check history before defaulting
                current_cell_type = enhanced_step["parameters"].get("cell_type", "unknown")
                existing_enrichment = self._check_existing_enrichment_analysis(execution_history, current_cell_type)

                if existing_enrichment:
                    logger.info(f"📋 HISTORY CHECK: Found existing enrichment for {current_cell_type} - using previous results")
                    # Use the analyses and parameters from the existing enrichment
                    enhanced_step["parameters"]["analyses"] = existing_enrichment.get("analyses", ["go"])
                    if existing_enrichment.get("gene_set_library"):
                        enhanced_step["parameters"]["gene_set_library"] = existing_enrichment["gene_set_library"]
                    if existing_enrichment.get("gene_set_libraries"):
                        enhanced_step["parameters"]["gene_set_libraries"] = existing_enrichment["gene_set_libraries"]
                    enhanced_step["description"] = f"Reuse existing enrichment analysis results for {current_cell_type}"
                    logger.info(f"✅ Using existing enrichment: {existing_enrichment.get('analyses', ['go'])}")
                else:
                    logger.info("🔧 General enrichment request, using default GO analysis")
                    enhanced_step["parameters"]["analyses"] = ["go"]

            return enhanced_step

        except Exception as e:
            logger.error(f"Error in pathway extraction: {e}")
            enhanced_step = step.copy()
            if "parameters" not in enhanced_step:
                enhanced_step["parameters"] = {}
            enhanced_step["parameters"]["analyses"] = ["go"]
            return enhanced_step

    def _check_existing_enrichment_analysis(self, execution_history: List[Dict[str, Any]], cell_type: str) -> Optional[Dict[str, Any]]:
        """
        Check if enrichment analysis has already been performed for this cell type.

        Args:
            execution_history: List of previously executed steps
            cell_type: Cell type to check for existing enrichment analysis

        Returns:
            Dict with existing enrichment parameters if found, None otherwise
        """
        if not execution_history:
            return None

        for step in reversed(execution_history):  # Check most recent first
            if (step.get("function_name") == "perform_enrichment_analyses" and
                step.get("success") and
                step.get("parameters", {}).get("cell_type") == cell_type):

                parameters = step.get("parameters", {})
                logger.info(f"📋 Found existing enrichment for {cell_type}: {parameters.get('analyses', ['unknown'])}")

                # Extract relevant parameters to reuse
                existing_params = {
                    "analyses": parameters.get("analyses"),
                    "gene_set_library": parameters.get("gene_set_library"),
                    "gene_set_libraries": parameters.get("gene_set_libraries"),
                    "execution_step": step.get("step_number", "unknown")
                }

                return existing_params

        return None
    
    def _call_llm_for_pathway_extraction(self, message: str) -> str:
        """
        Enhanced pathway extraction with intent classification.

        Args:
            message: User query to extract pathway terms from

        Returns:
            String indicating intent and content:
            - For explicit methods: "EXPLICIT_METHOD:go,kegg" or "EXPLICIT_METHOD:gsea:MSigDB_Hallmark_2020"
            - For pathway queries: pathway keywords string
            - For general requests: empty string
        """
        prompt = f"""
        Analyze this enrichment analysis query and determine intent:

        User Query: "{message}"

        First classify the user intent:

        1. EXPLICIT_METHOD: User explicitly requests specific analysis methods
           Examples: "run GO analysis", "perform KEGG", "do GSEA with MSigDB_Hallmark_2020"

        2. PATHWAY_SPECIFIC: User mentions specific biological pathways/processes
           Examples: "find interferon response pathways", "cell cycle regulation", "apoptosis pathways"

        3. GENERAL: General enrichment without specific methods or pathways
           Examples: "do enrichment analysis", "find enriched pathways"

        Response format:
        - For EXPLICIT_METHOD: Return "EXPLICIT_METHOD:method1,method2" or "EXPLICIT_METHOD:gsea:LibraryName"
        - For PATHWAY_SPECIFIC: Return only the pathway keywords (like before)
        - For GENERAL: Return empty string

        Examples:
        - "run GO analysis" → "EXPLICIT_METHOD:go"
        - "perform KEGG and Reactome" → "EXPLICIT_METHOD:kegg,reactome"
        - "GSEA with MSigDB_Hallmark_2020" → "EXPLICIT_METHOD:gsea:MSigDB_Hallmark_2020"
        - "find interferon response pathways" → "interferon response"
        - "do enrichment analysis" → ""

        Extract specific pathway terms, biological processes, or cellular functions mentioned.
        Return only the most relevant pathway keywords as a single string.

        If no specific pathways are mentioned, return an empty string.

        Result:
        """

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=50
            )

            messages = [
                SystemMessage(content="You are an enrichment analysis classifier. Follow the format exactly."),
                HumanMessage(content=prompt)
            ]

            response = model.invoke(messages)
            result = response.content.strip()

            logger.info(f"🧠 LLM enrichment analysis: '{result}'")
            return result if result and result.lower() != "none" else ""

        except Exception as e:
            logger.info(f"⚠️ LLM pathway extraction failed: {e}")
            return ""
    
    def _log_plan_statistics(self, plan: Dict[str, Any]):
        steps = plan.get("steps", [])
        logger.info(f"✅ Planner created execution plan with {len(steps)} steps")
        logger.info(f"   • {len([s for s in steps if s.get('function_name') == 'process_cells'])} process_cells steps")
        logger.info(f"   • {len([s for s in steps if s.get('step_type') == 'validation'])} validation steps")
        logger.info(f"   • {len([s for s in steps if s.get('step_type') == 'analysis'])} analysis steps")