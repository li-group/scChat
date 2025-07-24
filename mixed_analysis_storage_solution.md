# Mixed Analysis Storage Solution: Unified Result Access Strategy

## Problem Analysis

The current system has **three different analysis types with different storage patterns**:

### 1. **Enrichment Analysis** 
- **Storage**: Vector database (complex, semantic searchable)
- **Reason**: 1000+ pathway terms, needs semantic search capabilities
- **Access Pattern**: Query-based, contextual search
- **Data Structure**: Complex nested dictionaries with embeddings

### 2. **Differential Expression Analysis (DEA)**
- **Storage**: CSV files (simple, tabular)
- **Reason**: Simple gene lists with fold changes and p-values
- **Access Pattern**: Direct file reading, top N genes
- **Data Structure**: Flat table (gene_name, log_fc, p_value)

### 3. **Cell Count Comparison** 
- **Storage**: CSV files (simple, tabular)
- **Reason**: Simple count data by condition
- **Access Pattern**: Direct file reading, condition comparisons
- **Data Structure**: Flat table (condition, count, percentage)

## Current System Issues

**One-Size-Fits-All Problem:**
```python
# Current approach tries to force everything through the same pipeline
extract_key_findings_from_execution(execution_history)
  ↓
format_findings_for_synthesis(key_findings)  # Fails for mixed storage types
  ↓
LLM synthesis
```

**Storage Mismatch:**
- Enrichment: Expects vector database queries
- DEA: Expects CSV file reading  
- Cell counts: Expects CSV file reading
- **Result**: None of them work properly in the unified pipeline

## Solution: Unified Result Accessor Pattern

### Architecture Design

```python
class UnifiedResultAccessor:
    """
    Intelligent result accessor that knows how to retrieve and format 
    different analysis types based on their optimal storage patterns.
    """
    
    def get_analysis_results(self, execution_history: List[Dict]) -> Dict[str, Any]:
        """
        Route each analysis type to its appropriate accessor method
        """
        unified_results = {
            "enrichment_analyses": {},
            "dea_analyses": {},
            "cellcount_analyses": {},
            "process_cell_results": {}
        }
        
        for step in execution_history:
            function_name = self._get_function_name(step)
            cell_type = self._get_cell_type(step)
            
            if function_name == "perform_enrichment_analyses":
                unified_results["enrichment_analyses"][cell_type] = self._get_enrichment_results(cell_type)
            elif function_name == "dea_split_by_condition":
                unified_results["dea_analyses"][cell_type] = self._get_dea_results(cell_type)
            elif function_name == "compare_cell_counts":
                unified_results["cellcount_analyses"][cell_type] = self._get_cellcount_results(cell_type)
            elif function_name == "process_cells":
                unified_results["process_cell_results"][cell_type] = self._get_process_results(step)
        
        return unified_results
```

### Implementation Strategy

#### **1. Enrichment Results Accessor**
```python
def _get_enrichment_results(self, cell_type: str) -> Dict[str, Any]:
    """
    Access enrichment results from vector database + CSV summaries
    """
    results = {}
    
    # Get structured summaries from CSV files (fast, reliable)
    for analysis_type in ["reactome", "go_bp", "go_cc", "go_mf", "kegg", "gsea"]:
        csv_path = f"scchatbot/enrichment/{analysis_type}/results_summary_{cell_type}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results[analysis_type] = {
                "top_terms": df["term_name"].head(10).tolist(),
                "p_values": df["adj_p_value"].head(10).tolist(),
                "total_significant": len(df)
            }
    
    # Vector database available for semantic search if needed
    results["vector_search_available"] = True
    return results
```

#### **2. DEA Results Accessor**
```python
def _get_dea_results(self, cell_type: str) -> Dict[str, Any]:
    """
    Access DEA results from CSV files directly
    """
    results = {}
    
    # Look for condition-specific DEA results
    for condition in ["cataract", "basal_laminar_drusen", "age_related_macular_degeneration"]:
        csv_path = f"scchatbot/dea/{cell_type}_markers_{condition}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Get top upregulated and downregulated genes
            upregulated = df[df["log_fc"] > 0].nlargest(10, "log_fc")
            downregulated = df[df["log_fc"] < 0].nsmallest(10, "log_fc")
            
            results[condition] = {
                "top_upregulated": [
                    {"gene": row["gene"], "log_fc": row["log_fc"], "p_value": row["p_adj"]}
                    for _, row in upregulated.iterrows()
                ],
                "top_downregulated": [
                    {"gene": row["gene"], "log_fc": row["log_fc"], "p_value": row["p_adj"]}
                    for _, row in downregulated.iterrows()
                ],
                "total_significant": len(df[df["p_adj"] < 0.05])
            }
    
    return results
```

#### **3. Cell Count Results Accessor**
```python
def _get_cellcount_results(self, cell_type: str) -> Dict[str, Any]:
    """
    Access cell count comparison results from execution result
    """
    # This comes from execution history result directly since it's computed on-demand
    # We can access the structured result from the step
    return self._get_structured_result_from_execution(cell_type, "compare_cell_counts")
```

### **4. Unified Formatter**

```python
class UnifiedResultFormatter:
    """
    Format different analysis types appropriately for LLM synthesis
    """
    
    def format_for_synthesis(self, unified_results: Dict[str, Any]) -> str:
        """
        Format mixed analysis results into coherent text for LLM
        """
        sections = []
        
        # Format enrichment results
        if unified_results["enrichment_analyses"]:
            sections.append(self._format_enrichment_section(unified_results["enrichment_analyses"]))
        
        # Format DEA results  
        if unified_results["dea_analyses"]:
            sections.append(self._format_dea_section(unified_results["dea_analyses"]))
        
        # Format cell count results
        if unified_results["cellcount_analyses"]:
            sections.append(self._format_cellcount_section(unified_results["cellcount_analyses"]))
        
        return "\n\n".join(sections)
    
    def _format_enrichment_section(self, enrichment_data: Dict) -> str:
        """Format enrichment results with pathway names and p-values"""
        lines = ["ENRICHMENT ANALYSIS RESULTS:"]
        
        for cell_type, analyses in enrichment_data.items():
            lines.append(f"\n{cell_type.upper()} CELL ENRICHMENT:")
            
            for analysis_type, data in analyses.items():
                if analysis_type == "vector_search_available":
                    continue
                    
                top_terms = data.get("top_terms", [])
                p_values = data.get("p_values", [])
                
                if top_terms:
                    formatted_terms = []
                    for term, pval in zip(top_terms[:5], p_values[:5]):
                        formatted_terms.append(f"{term} (p={pval:.2e})")
                    
                    lines.append(f"  {analysis_type.upper()}: {'; '.join(formatted_terms)}")
        
        return "\n".join(lines)
    
    def _format_dea_section(self, dea_data: Dict) -> str:
        """Format DEA results with specific gene names and fold changes"""
        lines = ["DIFFERENTIAL EXPRESSION ANALYSIS:"]
        
        for cell_type, conditions in dea_data.items():
            lines.append(f"\n{cell_type.upper()} DIFFERENTIAL EXPRESSION:")
            
            for condition, results in conditions.items():
                lines.append(f"  {condition.replace('_', ' ').title()}:")
                
                # Top upregulated genes
                up_genes = results.get("top_upregulated", [])[:3]
                if up_genes:
                    gene_strs = [f"{g['gene']} (FC={g['log_fc']:.2f})" for g in up_genes]
                    lines.append(f"    Upregulated: {', '.join(gene_strs)}")
                
                # Top downregulated genes
                down_genes = results.get("top_downregulated", [])[:3]
                if down_genes:
                    gene_strs = [f"{g['gene']} (FC={g['log_fc']:.2f})" for g in down_genes]
                    lines.append(f"    Downregulated: {', '.join(gene_strs)}")
        
        return "\n".join(lines)
```

## Integration with Current System

### **Modified Response Generation Flow**

```python
def unified_response_generator_node(self, state: ChatState) -> ChatState:
    """
    Updated response generator using unified result accessor
    """
    print("🎯 UNIFIED: Generating response with mixed storage access...")
    
    try:
        # Use unified accessor instead of extract_key_findings_from_execution
        execution_history = state.get("execution_history", [])
        unified_accessor = UnifiedResultAccessor()
        unified_results = unified_accessor.get_analysis_results(execution_history)
        
        # Format using storage-aware formatter
        result_formatter = UnifiedResultFormatter()
        formatted_findings = result_formatter.format_for_synthesis(unified_results)
        
        print("✅ Unified results accessed and formatted successfully")
        
    except Exception as e:
        print(f"❌ Error in unified result access: {e}")
        # Fallback to current method
        key_findings = extract_key_findings_from_execution(execution_history)
        formatted_findings = format_findings_for_synthesis(key_findings)
    
    # Rest of synthesis process remains the same
    synthesis_prompt = self._create_enhanced_synthesis_prompt(
        original_question=state.get("current_message", ""),
        formatted_findings=formatted_findings,  # Now properly formatted mixed results
        failed_analyses=self._get_failed_analyses(state),
        conversation_context=conversation_context
    )
    
    response_text = self._call_llm_for_synthesis(synthesis_prompt)
    # ... rest of the method
```

## Benefits of This Approach

### **1. Storage Optimization**
- **Enrichment**: Uses vector database for complex semantic search
- **DEA/Counts**: Uses CSV files for simple, fast access
- **Best of both worlds**: Each data type uses its optimal storage

### **2. Unified Interface**
- Single entry point: `unified_accessor.get_analysis_results()`
- Consistent formatting: All results formatted appropriately for LLM
- Easy to extend: Add new analysis types with their own accessors

### **3. Reliability**
- **Fallback mechanisms**: If vector DB fails, fall back to CSV
- **Validation**: Check file existence before access
- **Error handling**: Graceful degradation per analysis type

### **4. Performance**
- **Fast access**: CSV reading for simple data
- **Semantic search**: Vector DB only when needed
- **Caching**: Can cache CSV results in memory

## Implementation Plan

### **Phase 1: Core Infrastructure**
1. Create `UnifiedResultAccessor` class
2. Implement storage-specific accessor methods
3. Create `UnifiedResultFormatter` class

### **Phase 2: Integration**
1. Replace `extract_key_findings_from_execution()` calls
2. Update `unified_response_generator_node()` 
3. Add fallback mechanisms

### **Phase 3: Testing & Optimization**
1. Test with AQ1/AQ2 scenarios
2. Verify all analysis types appear in responses
3. Optimize performance and add caching

This approach respects the natural storage patterns of your different analysis types while providing a unified interface for response generation. Each analysis type gets handled by the method that works best for its data structure and access patterns.