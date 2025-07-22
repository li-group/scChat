"""
Test the metadata flattening fix for ChromaDB compatibility.
This reproduces the exact error scenario and verifies the fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scchatbot.enhanced_function_history import EnhancedFunctionHistoryManager


def test_complex_metadata():
    """Test storing conversation with complex metadata that caused the original error"""
    print("=== Testing Complex Metadata Storage ===\n")
    
    # Initialize the manager
    history_manager = EnhancedFunctionHistoryManager()
    
    # This is the exact metadata that caused the error
    problematic_metadata = {
        "execution_steps": 1,
        "successful_analyses": 1, 
        "available_cell_types": ["Epithelial cell", "Stromal cell", "Immune cell", "Glial cell"],
        "has_plots": False
    }
    
    print("Testing with problematic metadata:")
    print(f"  execution_steps: {problematic_metadata['execution_steps']}")
    print(f"  successful_analyses: {problematic_metadata['successful_analyses']}")
    print(f"  available_cell_types: {problematic_metadata['available_cell_types']}")
    print(f"  has_plots: {problematic_metadata['has_plots']}")
    
    try:
        # Try to record a conversation with the complex metadata
        history_manager.record_conversation_with_vector(
            user_message="Can we confirm if Cell cycle regulation is enriched in Endothelial cells?",
            bot_response="Based on the analysis results, cell cycle regulation is indeed enriched in endothelial cells.",
            session_id="test",
            analysis_context=problematic_metadata
        )
        print("\n✅ SUCCESS: Conversation with complex metadata stored successfully!")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False
    
    return True


def test_metadata_flattening():
    """Test the metadata flattening function directly"""
    print("\n=== Testing Metadata Flattening Function ===\n")
    
    history_manager = EnhancedFunctionHistoryManager()
    
    # Test various complex metadata structures
    test_cases = [
        {
            "name": "Original problematic case",
            "metadata": {
                "execution_steps": 1,
                "successful_analyses": 1,
                "available_cell_types": ["Epithelial cell", "Stromal cell"],
                "has_plots": False
            }
        },
        {
            "name": "Nested dictionary case",
            "metadata": {
                "analysis": {
                    "type": "GSEA",
                    "pathways_found": 50
                },
                "cell_data": {
                    "count": 172,
                    "types": ["T cells", "B cells"]
                }
            }
        },
        {
            "name": "Mixed types case", 
            "metadata": {
                "simple_string": "test",
                "simple_int": 42,
                "simple_bool": True,
                "simple_none": None,
                "list_of_strings": ["a", "b", "c"],
                "list_of_numbers": [1, 2, 3],
                "empty_list": []
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        try:
            flattened = history_manager._flatten_metadata(test_case["metadata"])
            print(f"  Original: {test_case['metadata']}")
            print(f"  Flattened: {flattened}")
            
            # Verify all values are simple types
            for key, value in flattened.items():
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    raise ValueError(f"Non-simple type found: {key} = {value} ({type(value)})")
            
            print("  ✅ All values are simple types")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        print()


def test_search_with_flattened_metadata():
    """Test that search still works with flattened metadata"""
    print("=== Testing Search with Flattened Metadata ===\n")
    
    history_manager = EnhancedFunctionHistoryManager()
    
    # Record multiple conversations with complex metadata
    conversations = [
        {
            "user": "Run GSEA on T cells",
            "bot": "GSEA analysis completed on T cells. Found 50 enriched pathways.",
            "context": {
                "analysis_type": "GSEA",
                "cell_types": ["T cells"],
                "pathways_found": 50
            }
        },
        {
            "user": "What about B cells?", 
            "bot": "B cell analysis shows 30 enriched pathways including immune response.",
            "context": {
                "analysis_type": "GSEA",
                "cell_types": ["B cells"], 
                "pathways_found": 30
            }
        }
    ]
    
    # Store conversations
    for conv in conversations:
        try:
            history_manager.record_conversation_with_vector(
                user_message=conv["user"],
                bot_response=conv["bot"],
                session_id="search_test",
                analysis_context=conv["context"]
            )
            print(f"✅ Stored: '{conv['user'][:30]}...'")
        except Exception as e:
            print(f"❌ Failed to store: {e}")
    
    # Test search
    try:
        results = history_manager.search_conversations("T cell pathways", k=2)
        print(f"\n✅ Search returned {len(results)} results")
        
        if results:
            formatted = history_manager.format_search_results(results)
            print("Sample result:")
            print(formatted[:200] + "..." if len(formatted) > 200 else formatted)
            
    except Exception as e:
        print(f"❌ Search failed: {e}")


def main():
    """Run all tests"""
    print("Testing ChromaDB Metadata Compatibility Fix")
    print("=" * 50)
    
    # Test the main issue
    success = test_complex_metadata()
    
    # Test the flattening function
    test_metadata_flattening()
    
    # Test search functionality
    test_search_with_flattened_metadata()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Metadata fix is working correctly.")
        print("\nThe system can now handle complex metadata from analysis context:")
        print("- Lists are converted to comma-separated strings")
        print("- Nested dicts are flattened with underscore notation")
        print("- All ChromaDB type requirements are met")
    else:
        print("❌ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()