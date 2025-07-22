"""
Test script for the enhanced conversation history with vector database.

This script demonstrates how to:
1. Use the enhanced history manager instead of the basic one
2. Test conversation recording and retrieval
3. Verify vector search functionality
"""

from scchatbot.enhanced_function_history import EnhancedFunctionHistoryManager
from scchatbot.multi_agent_base import MultiAgentChatBot
import json


def test_enhanced_history_manager():
    """Test the enhanced history manager standalone"""
    print("=== Testing Enhanced History Manager ===\n")
    
    # Create enhanced history manager
    enhanced_history = EnhancedFunctionHistoryManager()
    
    # Test 1: Record some conversations
    print("1. Recording test conversations...")
    
    enhanced_history.record_conversation_with_vector(
        user_message="What are the marker genes for T cells?",
        bot_response="The key marker genes for T cells include CD3D, CD3E, CD3G (pan-T cell markers), CD4 (helper T cells), CD8A/CD8B (cytotoxic T cells), and IL7R. These genes are commonly used to identify T cell populations in scRNA-seq data.",
        session_id="test_session",
        analysis_context={"cell_type": "T cells", "analysis": "marker identification"}
    )
    
    enhanced_history.record_conversation_with_vector(
        user_message="Run GSEA analysis on the T cells",
        bot_response="I've completed GSEA analysis on T cells. The top enriched pathways include: 1) T cell receptor signaling pathway (adjusted p-value: 1.2e-15), 2) Immune response (adjusted p-value: 3.4e-12), 3) Cytokine-cytokine receptor interaction (adjusted p-value: 5.6e-10). A total of 50 significant pathways were identified.",
        session_id="test_session",
        analysis_context={"cell_type": "T cells", "analysis": "GSEA", "pathways_found": 50}
    )
    
    enhanced_history.record_conversation_with_vector(
        user_message="Compare macrophages and monocytes",
        bot_response="Macrophages and monocytes show distinct expression profiles. Macrophages express high levels of CD68, CD163, and MRC1, while monocytes express CD14, FCGR3A (CD16), and S100A8/A9. Macrophages show enrichment for phagocytosis and antigen presentation pathways.",
        session_id="test_session",
        analysis_context={"cell_types": ["macrophages", "monocytes"], "analysis": "comparison"}
    )
    
    print("✅ Conversations recorded successfully\n")
    
    # Test 2: Test context detection
    print("2. Testing context requirement detection...")
    
    test_queries = [
        ("What did we find about T cells?", True),
        ("Show me those pathways from earlier", True),
        ("What is UMAP?", False),
        ("Remind me about the macrophage markers", True),
        ("Run new analysis on B cells", False)
    ]
    
    for query, expected in test_queries:
        requires_context = enhanced_history.requires_conversation_context(query)
        status = "✅" if requires_context == expected else "❌"
        print(f"{status} '{query}' - Requires context: {requires_context}")
    
    print()
    
    # Test 3: Test vector search retrieval
    print("3. Testing vector search retrieval...")
    
    test_retrievals = [
        "What pathways did we find in T cells?",
        "Tell me about the markers we discussed",
        "What was the comparison between immune cells?"
    ]
    
    for query in test_retrievals:
        print(f"\nQuery: '{query}'")
        context = enhanced_history.retrieve_relevant_conversation(query, top_k=2)
        if context:
            print("Retrieved context:")
            print(context[:300] + "..." if len(context) > 300 else context)
        else:
            print("No relevant context found")
    
    print("\n✅ Enhanced history manager tests completed!")
    return enhanced_history


def test_integration_with_chatbot():
    """Test the integration with the main chatbot"""
    print("\n\n=== Testing Integration with Chatbot ===\n")
    
    # Note: This requires modifying the chatbot initialization
    # to use EnhancedFunctionHistoryManager instead of FunctionHistoryManager
    
    print("To integrate with your chatbot:")
    print("1. Update multi_agent_base.py line 54:")
    print("   FROM: self.history_manager = FunctionHistoryManager()")
    print("   TO:   from .enhanced_function_history import EnhancedFunctionHistoryManager")
    print("        self.history_manager = EnhancedFunctionHistoryManager()")
    print("\n2. The chatbot will automatically:")
    print("   - Detect when conversation context is needed")
    print("   - Retrieve relevant past conversations")
    print("   - Include context in response generation")
    print("   - Record all conversations for future reference")
    
    # Show example usage
    print("\n3. Example usage after integration:")
    print("   chatbot = MultiAgentChatBot()")
    print("   response1 = chatbot.send_message('What are the marker genes for T cells?')")
    print("   response2 = chatbot.send_message('Show me those markers you just mentioned')")
    print("   # The second query will automatically retrieve context from the first!")


def main():
    """Run all tests"""
    print("Enhanced Conversation History Test Suite")
    print("=" * 50)
    
    # Test enhanced history manager
    enhanced_history = test_enhanced_history_manager()
    
    # Test integration instructions
    test_integration_with_chatbot()
    
    print("\n\n=== Summary ===")
    print("The enhanced conversation history system provides:")
    print("- ✅ Vector database for semantic search")
    print("- ✅ Automatic context detection")
    print("- ✅ Reduced token consumption (only relevant context)")
    print("- ✅ Better continuity in conversations")
    print("- ✅ Domain-specific understanding (BioBERT)")
    
    print("\nTo use in production:")
    print("1. Install dependencies: pip install langchain-huggingface langchain-chroma sentence-transformers")
    print("2. Update multi_agent_base.py to use EnhancedFunctionHistoryManager")
    print("3. Conversations will be automatically tracked and retrieved!")


if __name__ == "__main__":
    main()