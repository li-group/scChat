#!/usr/bin/env python3
"""Test Phase 2 implementation for cell type discovery and missing cell type handling."""

import os
import sys
import time

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ["OPENAI_API_KEY"] = "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"

from scchatbot.chatbot import ChatBot

def test_missing_cell_types():
    """Test the handling of missing cell types in the system."""
    print("=" * 80)
    print("Testing Phase 2: Missing Cell Type Handling")
    print("=" * 80)
    
    try:
        # Initialize chatbot
        print("Initializing chatbot...")
        chatbot = ChatBot()
        
        # Test question with cell types that cannot be discovered
        test_question = "How is the known cell type 'Regulatory T cells' distinguished from the 'Conventional memory CD4 T cells'?"
        
        print(f"\nTest Question: {test_question}")
        print("-" * 80)
        
        # Process the question
        start_time = time.time()
        response = chatbot.chat_with_ai(test_question)
        elapsed_time = time.time() - start_time
        
        print(f"\n{'=' * 80}")
        print("RESPONSE:")
        print(f"{'=' * 80}")
        
        if isinstance(response, dict):
            print(f"Response Type: {response.get('response_type', 'unknown')}")
            print(f"\nAnswer: {response.get('response', 'No response')}")
            
            if 'graph_html' in response:
                print(f"\nVisualization: {'Generated' if response['graph_html'] else 'None'}")
        else:
            print(f"Response: {response}")
        
        print(f"\n{'=' * 80}")
        print(f"Execution Time: {elapsed_time:.2f} seconds")
        print(f"{'=' * 80}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_missing_cell_types()