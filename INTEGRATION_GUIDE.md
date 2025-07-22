# Vector Database Integration Guide for scChat

## Overview

This guide explains how to integrate the vector database-based conversation history system into your scChat application. The system provides semantic search capabilities for conversation history, reducing token consumption while maintaining context awareness.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_vector_db.txt
```

### 2. Update Your ChatBot Initialization

In `scchatbot/multi_agent_base.py`, make one simple change:

```python
# Line 54 - Change from:
self.history_manager = FunctionHistoryManager()

# To:
from .enhanced_function_history import EnhancedFunctionHistoryManager
self.history_manager = EnhancedFunctionHistoryManager()
```

That's it! The system will now automatically:
- Record all conversations in a vector database
- Detect when users reference previous conversations
- Retrieve only relevant context (saving ~60% tokens)
- Maintain conversation continuity

## How It Works

### Automatic Context Detection

The system detects when context is needed based on keywords like:
- "earlier", "before", "previous"
- "you said", "you mentioned"
- "that analysis", "those results"
- "remind me", "what was"

### Vector Search Process

1. **User asks**: "What were those T cell pathways from earlier?"
2. **System detects**: Context needed (keyword: "earlier")
3. **Vector search**: Finds semantically similar past conversations
4. **Context injection**: Adds only relevant history to the prompt
5. **Response generation**: LLM uses context to provide accurate answer

### Token Efficiency

Instead of including entire conversation history:
- **Traditional**: 800+ tokens for full history
- **Vector DB**: ~200-300 tokens for relevant context only
- **Result**: 60%+ reduction in context tokens

## Features

### 1. BioBERT Embeddings
- Optimized for biological/scientific terms
- Better understanding of gene names, cell types, pathways
- More accurate retrieval for domain-specific queries

### 2. Session Management
- Track conversations by session ID
- Filter context by session when needed
- Support for multi-user scenarios

### 3. Rich Metadata
- Store analysis context with conversations
- Track successful/failed analyses
- Enable more intelligent retrieval

### 4. Fallback Support
- Gracefully handles vector DB failures
- Falls back to recent conversation history
- Ensures system reliability

## Testing

Run the test script to verify the integration:

```bash
python test_enhanced_history.py
```

This will:
1. Test the enhanced history manager
2. Verify context detection
3. Test vector search retrieval
4. Show integration examples

## Advanced Configuration

### Using Faster Embeddings

If BioBERT is too slow, switch to a general-purpose model:

```python
# In enhanced_function_history.py, line 52
self.embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster
    # model_name="dmis-lab/biobert-v1.1",  # More accurate for biology
)
```

### Adjusting Context Retrieval

Modify the number of past conversations retrieved:

```python
# In workflow/core_nodes.py, line 82
conversation_context = self.history_manager.retrieve_relevant_conversation(
    current_query=current_message,
    top_k=5,  # Retrieve top 5 instead of 3
    session_filter=state.get("session_id", "default")
)
```

### Custom Context Indicators

Add domain-specific indicators in `enhanced_function_history.py`:

```python
def requires_conversation_context(self, message: str) -> bool:
    context_indicators = [
        # ... existing indicators ...
        "the genes", "those cells", "that enrichment",  # Added
        "previous GSEA", "earlier clustering"  # Domain-specific
    ]
```

## Troubleshooting

### Vector DB Connection Issues

If you see "Vector database not available":
1. Check ChromaDB installation: `pip show chromadb`
2. Verify write permissions in `function_history/conversation_vectors/`
3. Check for conflicting ChromaDB instances

### High Memory Usage

BioBERT can use significant memory. Solutions:
1. Switch to lighter embeddings model
2. Reduce batch size for embedding generation
3. Use CPU instead of GPU for embeddings

### Slow Response Times

If vector search is slow:
1. Reduce `top_k` parameter
2. Use faster embedding model
3. Implement caching for frequent queries

## Benefits for scRNA-seq Analysis

1. **Continuity**: "What genes were enriched in those T cells?" automatically finds relevant past analysis
2. **Comparison**: "How does this compare to the earlier macrophage analysis?" retrieves specific context
3. **Efficiency**: Only relevant scientific context included, not entire conversation
4. **Accuracy**: BioBERT understands gene names, cell types, and biological terms

## Next Steps

1. **Production Deployment**: Consider using persistent ChromaDB storage
2. **Multi-user Support**: Implement proper session management
3. **Analytics**: Track which contexts are most frequently retrieved
4. **Fine-tuning**: Consider fine-tuning embeddings on your specific scRNA-seq corpus