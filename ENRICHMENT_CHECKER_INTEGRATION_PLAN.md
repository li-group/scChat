# EnrichmentChecker Integration Plan

## Overview

This document outlines the implementation plan for integrating the `EnrichmentChecker` intelligent pathway analysis system into the SCChatbot planner workflow. The integration will enhance the planner's ability to automatically select optimal enrichment analysis methods based on user pathway queries.

## Current Architecture

### Planner Workflow (core_nodes.py)
```
User Query → Query Type Detection → LLM Planning → Execution Plan
```

### Enrichment Function Call Flow
```
planner_node → perform_enrichment_analyses step → _wrap_enrichment_analysis → perform_enrichment_analyses
```

### Current Planning Parameters
- `cell_type`: Target cell type for analysis
- `pathway_include`: User-specified pathway terms
- `analyses`: List of analysis methods (optional)
- `gene_set_library`: Specific gene set library (optional)

## Integration Strategy: Pre-Planning Enhancement (Recommended)

### Why Pre-Planning Enhancement?
1. **Early Intelligence**: Enhance pathway understanding before LLM planning
2. **Better Context**: Provide LLM with validated pathway recommendations
3. **Consistency**: Maintain LLM's planning logic while improving input quality
4. **Fallback Safety**: Easy to disable if EnrichmentChecker fails

### Integration Architecture
```
User Query → Pathway Enhancement → Query Type Detection → Enhanced LLM Planning → Execution Plan
           ↓
    EnrichmentChecker
    (if pathway query)
```

## Implementation Plan

### Phase 1: Core Integration Infrastructure

#### 1.1 Add EnrichmentChecker to CoreNodes
**File**: `scchatbot/workflow/core_nodes.py`

```python
# Add to imports
from ..enrichment_checker import EnrichmentChecker

# Add to __init__ method
def __init__(self, ...):
    # ... existing init code ...
    
    # Initialize EnrichmentChecker with error handling
    try:
        self.enrichment_checker = EnrichmentChecker()
        self.enrichment_checker_available = (
            self.enrichment_checker.connection_status == "connected"
        )
        print(f"✅ EnrichmentChecker initialized: {self.enrichment_checker.connection_status}")
    except Exception as e:
        print(f"⚠️ EnrichmentChecker initialization failed: {e}")
        self.enrichment_checker = None
        self.enrichment_checker_available = False
```

#### 1.2 Add Pathway Enhancement Method
**File**: `scchatbot/workflow/core_nodes.py`

```python
def _enhance_pathway_query(self, message: str, query_type: str) -> Dict[str, Any]:
    """
    Enhance pathway queries using EnrichmentChecker intelligence.
    
    Returns:
        - enhancement_data: Dict with recommended analyses and parameters
        - None: If no enhancement needed or EnrichmentChecker unavailable
    """
    if not self.enrichment_checker_available or query_type != "pathway_specific":
        return None
    
    try:
        # Extract pathway terms from user message
        pathway_terms = self._extract_pathway_terms_from_message(message)
        if not pathway_terms:
            return None
        
        print(f"🧬 Enhancing pathway query with terms: {pathway_terms}")
        
        # Create mock plan step for EnrichmentChecker
        mock_plan_step = {
            "function_name": "perform_enrichment_analyses",
            "parameters": {
                "cell_type": "placeholder",  # Will be filled by planner
                "pathway_include": pathway_terms
            }
        }
        
        # Get pathway enhancement
        enhanced_plan = self.enrichment_checker.enhance_enrichment_plan(mock_plan_step)
        
        # Extract enhancement data
        enhancement_data = {
            "recommended_analyses": enhanced_plan["parameters"].get("analyses", []),
            "gene_set_library": enhanced_plan["parameters"].get("gene_set_library"),
            "pathway_terms": pathway_terms,
            "description": enhanced_plan.get("description", ""),
            "validation_details": enhanced_plan.get("validation_details", {}),
            "confidence": enhanced_plan.get("validation_details", {}).get("confidence", 0.0)
        }
        
        print(f"✅ Pathway enhancement successful: {enhancement_data['recommended_analyses']}")
        return enhancement_data
        
    except Exception as e:
        print(f"⚠️ Pathway enhancement failed: {e}")
        return None

def _extract_pathway_terms_from_message(self, message: str) -> str:
    """Extract pathway terms from user message using simple keyword matching."""
    # Simple extraction - could be enhanced with LLM in future
    pathway_keywords = [
        "interferon", "ifn", "apoptosis", "cell cycle", "inflammation",
        "immune", "signaling", "metabolism", "proliferation", "differentiation"
    ]
    
    message_lower = message.lower()
    found_terms = []
    
    for keyword in pathway_keywords:
        if keyword in message_lower:
            found_terms.append(keyword)
    
    return " ".join(found_terms) if found_terms else None
```

#### 1.3 Integrate Enhancement into Planner Flow
**File**: `scchatbot/workflow/core_nodes.py`

Modify `_create_enhanced_plan` method:

```python
def _create_enhanced_plan(self, state: ChatState, message: str, available_functions: List, 
                         available_cell_types: List[str], function_history: Dict, 
                         unavailable_cell_types: List[str], query_type: str) -> ChatState:
    """Create enhanced plan using LLM with query type-specific guidance"""
    
    # NEW: Try pathway enhancement first
    pathway_enhancement = self._enhance_pathway_query(message, query_type)
    state["pathway_enhancement"] = pathway_enhancement  # Store for later use
    
    # Get query type-specific instructions
    query_guidance = self._get_query_type_guidance(query_type, pathway_enhancement)
    
    # ... rest of existing method remains the same ...
```

#### 1.4 Update Query Type Guidance
**File**: `scchatbot/workflow/core_nodes.py`

Modify `_get_query_type_guidance` method to accept pathway enhancement:

```python
def _get_query_type_guidance(self, query_type: str, pathway_enhancement: Dict = None) -> str:
    """Get query type-specific guidance for the planning prompt"""
    
    if query_type == "pathway_specific":
        base_guidance = """
        🎯 PATHWAY-SPECIFIC QUERY DETECTED:
        - This query asks about specific pathway analysis (GSEA, GO, KEGG, REACTOME)
        - FOCUS: Create a streamlined plan targeting the specific pathway analysis mentioned
        - PRIORITY: Use the exact analysis type mentioned in the query
        - EFFICIENCY: Avoid unnecessary broader analyses unless specifically requested
        """
        
        # Add pathway enhancement guidance if available
        if pathway_enhancement:
            enhancement_guidance = f"""
        
        🧠 PATHWAY INTELLIGENCE ENHANCEMENT:
        - Recommended analyses: {pathway_enhancement['recommended_analyses']}
        - Gene set library: {pathway_enhancement.get('gene_set_library', 'Not specified')}
        - Pathway terms: {pathway_enhancement['pathway_terms']}
        - Confidence: {pathway_enhancement.get('confidence', 0.0):.2f}
        - PRIORITY: Use the recommended analyses for optimal pathway analysis
        - EXAMPLE: For enrichment steps, use "analyses": {pathway_enhancement['recommended_analyses']}
            """
            return base_guidance + enhancement_guidance
        
        return base_guidance
    
    # ... rest of existing method remains the same ...
```

### Phase 2: Enhanced Planning Integration

#### 2.1 Update ChatState Model
**File**: `scchatbot/cell_type_models.py`

```python
class ChatState(TypedDict, total=False):
    # ... existing fields ...
    
    # New fields for pathway enhancement
    pathway_enhancement: Optional[Dict[str, Any]]  # EnrichmentChecker results
    enrichment_confidence: Optional[float]         # Pathway matching confidence
```

#### 2.2 Enhance Enrichment Step Creation
**File**: `scchatbot/workflow/core_nodes.py`

Add post-processing to apply pathway enhancement to enrichment steps:

```python
def _apply_pathway_enhancement_to_plan(self, plan_data: Dict[str, Any], 
                                     pathway_enhancement: Dict[str, Any]) -> Dict[str, Any]:
    """Apply pathway enhancement recommendations to enrichment analysis steps."""
    if not pathway_enhancement:
        return plan_data
    
    enhanced_steps = []
    
    for step in plan_data.get("steps", []):
        if step.get("function_name") == "perform_enrichment_analyses":
            # Apply pathway enhancement
            if "parameters" not in step:
                step["parameters"] = {}
            
            # Add recommended analyses if not explicitly specified
            if not step["parameters"].get("analyses"):
                step["parameters"]["analyses"] = pathway_enhancement["recommended_analyses"]
            
            # Add gene set library if available
            if pathway_enhancement.get("gene_set_library"):
                step["parameters"]["gene_set_library"] = pathway_enhancement["gene_set_library"]
            
            # Ensure pathway_include is set
            if not step["parameters"].get("pathway_include") and pathway_enhancement.get("pathway_terms"):
                step["parameters"]["pathway_include"] = pathway_enhancement["pathway_terms"]
            
            # Update description with enhancement info
            validation_details = pathway_enhancement.get("validation_details", {})
            if validation_details:
                step["description"] += f" (Enhanced with pathway intelligence: {validation_details.get('total_recommendations', 0)} recommendations)"
            
            print(f"🧬 Enhanced enrichment step with: {step['parameters']}")
        
        enhanced_steps.append(step)
    
    plan_data["steps"] = enhanced_steps
    return plan_data
```

Add to `_create_enhanced_plan` method:

```python
# After plan creation and before cell discovery enhancement
if pathway_enhancement:
    enhanced_plan = self._apply_pathway_enhancement_to_plan(enhanced_plan, pathway_enhancement)
```

### Phase 3: Error Handling and Fallbacks

#### 3.1 Graceful Degradation
**File**: `scchatbot/workflow/core_nodes.py`

```python
def _safe_pathway_enhancement(self, message: str, query_type: str) -> Dict[str, Any]:
    """Safely attempt pathway enhancement with comprehensive error handling."""
    try:
        return self._enhance_pathway_query(message, query_type)
    except Exception as e:
        print(f"⚠️ Pathway enhancement failed, continuing with standard planning: {e}")
        return None

def __del__(self):
    """Cleanup EnrichmentChecker connection on destruction."""
    if hasattr(self, 'enrichment_checker') and self.enrichment_checker:
        try:
            self.enrichment_checker.close()
        except:
            pass  # Ignore cleanup errors
```

#### 3.2 Fallback Mechanisms
**File**: `scchatbot/workflow/core_nodes.py`

```python
def _get_fallback_enrichment_config(self, query_type: str) -> Dict[str, Any]:
    """Provide fallback enrichment configuration when EnrichmentChecker fails."""
    fallback_configs = {
        "pathway_specific": {
            "recommended_analyses": ["gsea"],
            "gene_set_library": "MSigDB_Hallmark_2020",
            "confidence": 0.5
        },
        "general": {
            "recommended_analyses": ["gsea", "go"],
            "gene_set_library": "MSigDB_Hallmark_2020", 
            "confidence": 0.3
        }
    }
    
    return fallback_configs.get(query_type, fallback_configs["general"])
```

### Phase 4: Monitoring and Logging

#### 4.1 Enhancement Tracking
**File**: `scchatbot/workflow/core_nodes.py`

```python
def _log_pathway_enhancement_stats(self, enhancement_data: Dict[str, Any]) -> None:
    """Log pathway enhancement statistics for monitoring."""
    if not enhancement_data:
        return
    
    validation_details = enhancement_data.get("validation_details", {})
    
    print("📊 PATHWAY ENHANCEMENT STATS:")
    print(f"   • Recommended analyses: {enhancement_data['recommended_analyses']}")
    print(f"   • Confidence: {enhancement_data.get('confidence', 0.0):.2f}")
    print(f"   • Total recommendations: {validation_details.get('total_recommendations', 0)}")
    print(f"   • Pathway matches: {len(validation_details.get('pathway_matches', []))}")
```


## Integration Points Summary

### 1. Initialization
- Add `EnrichmentChecker` to `CoreNodes.__init__`
- Handle connection failures gracefully
- Set availability flag for runtime checks

### 2. Planning Enhancement
- Call `_enhance_pathway_query` for pathway-specific queries
- Store enhancement data in `ChatState`
- Apply recommendations to enrichment steps

### 3. Error Handling
- Graceful degradation when EnrichmentChecker fails
- Fallback configurations for different query types
- Resource cleanup on destruction

### 4. Monitoring
- Log enhancement statistics
- Track performance metrics
- Monitor success/failure rates

## Benefits

1. **Intelligent Pathway Analysis**: Automatic selection of optimal enrichment methods
2. **Improved Accuracy**: LLM + Neo4j validation ensures relevant pathway targeting
3. **Better User Experience**: More precise and comprehensive enrichment results
4. **Backward Compatibility**: Existing functionality preserved with graceful fallbacks
5. **Extensible Design**: Easy to add new pathway databases and analysis methods

## Implementation Timeline

- **Phase 1** (Core Infrastructure): 2-3 hours
- **Phase 2** (Enhanced Integration): 2-3 hours  
- **Phase 3** (Error Handling): 1-2 hours
- **Phase 4** (Monitoring): 30 minutes
- **Testing & Validation**: 2-3 hours

**Total Estimated Time**: 7-10 hours

## Testing Strategy

1. **Unit Tests**: Test pathway enhancement methods independently
2. **Integration Tests**: Test full planning workflow with various pathway queries
3. **Error Scenarios**: Test behavior when Neo4j/OpenAI unavailable
4. **Functionality Tests**: Verify enhancement recommendations work correctly
5. **User Scenarios**: Test real pathway queries from domain experts

## Risk Mitigation

1. **Connection Failures**: Comprehensive fallback mechanisms
2. **Integration Bugs**: Feature flags to disable enhancement
3. **Data Quality**: Validation of EnrichmentChecker outputs

This integration plan provides a robust, well-tested pathway to enhance the SCChatbot planner with intelligent pathway analysis capabilities while maintaining system reliability and performance.