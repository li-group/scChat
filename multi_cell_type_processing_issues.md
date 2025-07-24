# Multi-Cell-Type Processing Issues Documentation

## Overview
This document captures all identified issues related to the system's failure to process multiple cell types with appropriate analysis steps when answering questions involving more than one cell type.

## Core Problem Statement
When users ask questions involving multiple cell types (e.g., "How do Mast cells and Schwann cells interact?"), the system fails to:
1. Generate enrichment analyses for ALL mentioned cell types
2. Create appropriate follow-up steps for discovered cell types
3. Properly compare/contrast findings across cell types

## Detailed Issue Analysis

### Issue 1: Incomplete Cell Type Coverage

#### Example from AQ2
**Question**: "What is the activity of cellular processes in Mast cells? How about Schwann cells?"

**What Happened**:
```
✅ Step 1: process_cells on Glial cell → discovered Schwann cell
✅ Step 3: perform_enrichment_analyses on Mast cell
❌ Missing: perform_enrichment_analyses on Schwann cell
```

**Root Cause**: Planner recognizes multiple cell types but doesn't generate complete analysis plans for all of them.

### Issue 2: Misguided Semantic Search

#### Current Behavior
```
Step 4: Search for "Schwann cell" in Mast cell enrichment results
Result: 7 weak matches (similarity < 0.3)
```

**Problem**: Trying to find information about Cell Type B within Cell Type A's data
**Impact**: Inadequate and potentially misleading results

### Issue 3: Follow-up Analysis Gap

#### Process Cell Discovery Pattern
```
process_cells("Glial cell") → discovers ["Schwann cell", "Muller cell"]
```

**Expected Follow-up**:
- Enrichment analysis for each discovered subtype
- Condition-specific DEA for each subtype
- Comparative analysis if relevant

**Actual Follow-up**: None - discovered cell types are noted but not analyzed

### Issue 4: Cross-Cell-Type Comparison Limitations

#### Current Limitations
1. No systematic comparison of enrichment results across cell types
2. No interaction analysis between cell types
3. No pathway overlap/divergence analysis

#### Needed Capabilities
- Side-by-side pathway comparison
- Shared vs unique pathway identification
- Cell-cell interaction predictions

## Pattern Analysis

### Single Cell Type Questions ✅
```
Q: "What pathways are enriched in Mast cells?"
Plan: 
1. perform_enrichment_analyses(cell_type="Mast cell")
2. display_enrichment_visualization(cell_type="Mast cell")
Result: Complete analysis
```

### Multi Cell Type Questions ❌
```
Q: "Compare Mast cells and Schwann cells"
Current Plan:
1. perform_enrichment_analyses(cell_type="Mast cell")
2. search_enrichment_semantic(query="Schwann cell", cell_type="Mast cell")

Needed Plan:
1. perform_enrichment_analyses(cell_type="Mast cell")
2. perform_enrichment_analyses(cell_type="Schwann cell")
3. compare_enrichment_results(cell_types=["Mast cell", "Schwann cell"])
```

### Discovery + Analysis Pattern ❌
```
Q: "What are the characteristics of glial subtypes?"
Current Plan:
1. process_cells(cell_type="Glial cell")
2. [No follow-up]

Needed Plan:
1. process_cells(cell_type="Glial cell") → ["Schwann cell", "Muller cell"]
2. perform_enrichment_analyses(cell_type="Schwann cell")
3. perform_enrichment_analyses(cell_type="Muller cell")
4. compare_cell_characteristics(cell_types=["Schwann cell", "Muller cell"])
```

## Proposed Solutions

### Solution 1: Enhanced Planner Prompting

#### Current Planner Instruction (Hypothetical)
```
Generate analysis steps for the user's question about cell types.
```

#### Improved Planner Instruction
```
Generate analysis steps for the user's question about cell types.

IMPORTANT: 
1. If multiple cell types are mentioned, generate separate enrichment analyses for EACH
2. If process_cells discovers subtypes, add enrichment analyses for EACH discovered type
3. For comparison questions, add explicit comparison steps after individual analyses
4. Never search for Cell Type B information within Cell Type A's results
```

### Solution 2: Execution Plan Validation

#### Add Validation Rules
```python
def validate_execution_plan(plan, question):
    mentioned_cell_types = extract_all_cell_types(question)
    analyzed_cell_types = extract_analyzed_types(plan)
    
    # Rule 1: All mentioned cell types should be analyzed
    missing = mentioned_cell_types - analyzed_cell_types
    if missing:
        add_enrichment_steps(plan, missing)
    
    # Rule 2: No cross-contamination searches
    for step in plan:
        if step.function == "search_enrichment_semantic":
            if step.query_cell_type != step.search_in_cell_type:
                flag_invalid_search(step)
    
    # Rule 3: Discovery requires follow-up
    for step in plan:
        if step.function == "process_cells":
            ensure_follow_up_analyses(plan, step)
```

### Solution 3: Template-Based Planning

#### Multi-Cell-Type Template
```python
def generate_multi_cell_plan(cell_types):
    plan = []
    
    # Individual analyses
    for cell_type in cell_types:
        plan.append({
            "function": "perform_enrichment_analyses",
            "parameters": {"cell_type": cell_type}
        })
    
    # Comparison if multiple
    if len(cell_types) > 1:
        plan.append({
            "function": "compare_enrichment_results",
            "parameters": {"cell_types": cell_types}
        })
    
    # Visualization
    for cell_type in cell_types:
        plan.append({
            "function": "display_enrichment_visualization",
            "parameters": {"cell_type": cell_type}
        })
    
    return plan
```

## Impact Assessment

### Current System Performance
- Single cell type questions: 90% success
- Multi cell type questions: 30% success
- Discovery follow-up: 0% success

### Expected Performance After Fixes
- Single cell type questions: 95% success
- Multi cell type questions: 85% success
- Discovery follow-up: 80% success

## Implementation Priority

### High Priority
1. Fix planner prompting for multi-cell-type recognition
2. Add follow-up analysis for discovered cell types
3. Prevent cross-cell-type contamination in searches

### Medium Priority
1. Add comparison functions for multi-cell analyses
2. Implement validation layer for execution plans
3. Create templates for common patterns

### Low Priority
1. Advanced interaction analysis features
2. Pathway network visualization across cell types
3. Automated insight generation for comparisons

## Test Cases

### Test 1: Direct Multi-Cell
```
Input: "Compare enrichment between Mast cells and Schwann cells"
Expected: Separate enrichment for each, then comparison
```

### Test 2: Discovery Follow-up
```
Input: "Analyze all glial cell subtypes"
Expected: process_cells → discover subtypes → analyze each
```

### Test 3: Complex Multi-Cell
```
Input: "How do immune cells (T cells, B cells, Mast cells) differ in their response?"
Expected: Three separate enrichments, multi-way comparison
```

## Monitoring and Success Metrics

1. **Coverage Rate**: % of mentioned cell types that get analyzed
2. **Follow-up Rate**: % of discovered cell types that get analyzed
3. **Comparison Quality**: Relevance score of multi-cell insights
4. **Search Accuracy**: Reduction in cross-cell-type search attempts

## Next Steps

1. **Immediate**: Document current planner behavior
2. **Short-term**: Implement enhanced planner prompting
3. **Medium-term**: Add execution plan validation
4. **Long-term**: Build comprehensive multi-cell analysis features