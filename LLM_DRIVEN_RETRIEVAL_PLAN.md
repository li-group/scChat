# LLM-Driven Conversation Retrieval Implementation Plan

## Overview
This plan outlines the transition from hard-coded keyword matching to a fully LLM-driven conversation retrieval system. The new system will let the LLM decide when context is needed and what to search for, eliminating all hard-coded logic.

## Functions to Remove

### 1. **`requires_conversation_context()`** in `enhanced_function_history.py`
```python
# REMOVE: Lines 267-279
def requires_conversation_context(self, message: str) -> bool:
    """Determine if message requires conversation context"""
    context_indicators = [
        "earlier", "before", "previous", "you said", "you mentioned", 
        "clarify", "explain that", "what did you mean", "that analysis",
        "those results", "the pathways you found", "show me that",
        "from last time", "remind me", "what was", "which one",
        "the genes", "those cells", "that enrichment"
    ]
    
    return any(indicator in message.lower() for indicator in context_indicators)
```
**Reason**: Hard-coded keyword matching is inflexible and brittle

### 2. **`_extract_scientific_entities()`** in `enhanced_function_history.py` 
```python
# REMOVE: Lines 181-202
def _extract_scientific_entities(self, text: str) -> List[str]:
    """Extract scRNA-seq specific entities for better searchability"""
    # Hard-coded regex patterns for genes, cell types, etc.
```
**Reason**: LLM can identify entities more intelligently without regex

### 3. **Complex Tool Definition Structure**
```python
# REMOVE: The complex return structure with detailed descriptions
return {
    "name": "search_conversation_history",
    "description": """[Long detailed description with specific cases]""",
    "function": search_conversation_history
}
```
**Reason**: Over-engineered for simple search functionality

## Functions to Modify

### 1. **`input_processor_node()`** in `core_nodes.py`
**Current**: Checks `requires_conversation_context()` with hard-coded logic
**New**: Direct LLM analysis for context needs

### 2. **`retrieve_relevant_conversation()`** in `enhanced_function_history.py`
**Current**: Uses current message directly as search query
**New**: Can accept LLM-generated search queries

## New Implementations

### 1. **Simplified Context Retrieval in `input_processor_node()`**
```python
def input_processor_node(self, state: ChatState) -> ChatState:
    """Process input with LLM-driven context retrieval"""
    
    # Initialize basic state
    if not state.get("messages"):
        state["messages"] = [AIMessage(content=self.initial_annotation_content)]
    
    current_message = state["current_message"]
    
    # Let LLM decide if context is needed and what to search for
    context_analysis_prompt = f"""
    User asked: "{current_message}"
    
    If this question seems to reference or build upon previous conversations, 
    generate 1-3 search queries to find relevant context.
    
    Return a JSON list of search queries, or an empty list if no context is needed.
    Only return the JSON list, nothing else.
    """
    
    try:
        # LLM decides everything
        search_queries = json.loads(self._call_llm(context_analysis_prompt))
        
        if search_queries:
            print(f"🧠 LLM generated {len(search_queries)} search queries")
            
            # Retrieve context using LLM-generated queries
            all_results = []
            for query in search_queries[:3]:
                results = self.history_manager.search_conversations(query, k=2)
                all_results.extend(results)
            
            if all_results:
                context = self.history_manager.format_search_results(all_results)
                state["messages"].append(
                    AIMessage(content=f"CONVERSATION_CONTEXT: {context}")
                )
                state["has_conversation_context"] = True
                print(f"✅ Retrieved context ({len(context)} chars)")
        
    except Exception as e:
        # Silent fail - continue without context
        print(f"⚠️ Context retrieval skipped: {e}")
    
    # Continue with rest of processing
    state["messages"].append(HumanMessage(content=current_message))
    # ... rest of initialization
    
    return state
```

### 2. **New Simple Search Method in `enhanced_function_history.py`**
```python
def search_conversations(self, query: str, k: int = 3) -> List[Any]:
    """Simple wrapper for vector database search"""
    if not self.conversation_vector_db:
        return []
    
    try:
        return self.conversation_vector_db.similarity_search(query, k=k)
    except:
        return []

def format_search_results(self, results: List[Any]) -> str:
    """Format search results as conversation history"""
    if not results:
        return ""
    
    formatted_parts = ["RELEVANT CONVERSATION HISTORY:"]
    seen_exchanges = set()
    
    for result in results:
        exchange = result.metadata.get("full_exchange", "")
        if exchange and exchange not in seen_exchanges:
            seen_exchanges.add(exchange)
            timestamp = result.metadata.get("timestamp", "")[:19]
            formatted_parts.append(f"\n[{timestamp}] {exchange}")
    
    return "\n".join(formatted_parts)
```

### 3. **Add LLM Call Method to `core_nodes.py`**
```python
def _call_llm(self, prompt: str) -> str:
    """Simple LLM call for analysis tasks"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temperature for consistency
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return "[]"  # Safe default for JSON parsing
```

## Implementation Steps

1. **Remove hard-coded functions** (listed above)
2. **Add simple search wrapper** to enhanced_function_history.py
3. **Update input_processor_node** with LLM-driven logic
4. **Test with various query types** to ensure LLM handles all cases
5. **Remove unused imports** and clean up code

## Benefits

1. **No Hard-Coding**: Zero keyword lists or rules
2. **Adaptive**: Works with any phrasing or language
3. **Maintainable**: Less code, easier to understand
4. **Flexible**: LLM adapts to new patterns automatically
5. **Fail-Safe**: Graceful degradation if retrieval fails

## Testing Strategy

### Test Cases:
1. **Direct Reference**: "What were those T cell pathways?"
2. **Ambiguous Reference**: "Show me the same analysis"  
3. **No Context Needed**: "What is UMAP?"
4. **Complex Reference**: "Compare with what we found earlier"
5. **Error Handling**: Test with vector DB offline

### Expected Behavior:
- Cases 1, 2, 4: LLM generates search queries, retrieves context
- Case 3: LLM returns empty list, no retrieval
- Case 5: System continues without context

## Migration Path

1. **Phase 1**: Implement new LLM-driven logic alongside existing
2. **Phase 2**: A/B test to ensure quality matches or exceeds current
3. **Phase 3**: Remove old hard-coded functions
4. **Phase 4**: Clean up and optimize

## Rollback Plan

If issues arise:
1. Keep old `requires_conversation_context()` commented out initially
2. Can quickly revert to keyword-based approach
3. Gradual transition allows testing without breaking existing functionality