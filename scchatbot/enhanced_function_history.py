"""
Enhanced function execution and conversation history management with vector database.

This module extends the existing FunctionHistoryManager with vector database
support for semantic conversation retrieval, reducing token consumption while
maintaining conversation context awareness.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage

from .function_history import FunctionHistoryManager


class EnhancedFunctionHistoryManager(FunctionHistoryManager):
    """
    Extends FunctionHistoryManager with vector database for conversation history.
    
    Features:
    - Semantic search for relevant conversation context
    - BioBERT embeddings for biological domain understanding
    - Reduced token consumption through selective context retrieval
    - Maintains all existing function history functionality
    """
    
    def __init__(self, history_dir: str = "function_history"):
        # Initialize parent class
        super().__init__(history_dir)
        
        # Additional files for conversation history
        self.conversation_file = os.path.join(history_dir, "conversation_history.json")
        
        # Vector database for conversation history
        self.vector_db_dir = os.path.join(history_dir, "conversation_vectors")
        self._setup_vector_db()
        
        # Load conversation history
        self.conversation_history = self._load_conversation_history()
    
    def _setup_vector_db(self):
        """Initialize vector database with fallback options"""
        self.conversation_vector_db = None
        self.embedder = None
        
        try:
            # Use a reliable sentence transformer model
            self.embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast and reliable
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize Chroma vector database
            self.conversation_vector_db = Chroma(
                collection_name="scrna_conversation_history",
                embedding_function=self.embedder,
                persist_directory=self.vector_db_dir
            )
            print(f"✅ Vector database initialized at {self.vector_db_dir}")
            
        except ImportError as e:
            print(f"⚠️ Vector DB dependencies not available: {e}")
            print("   Install with: pip install langchain-huggingface langchain-chroma sentence-transformers")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize vector database: {e}")
            print("   Falling back to simple conversation storage")
            
        # Always ensure these are set
        if self.conversation_vector_db is None:
            print("📝 Using fallback: conversations will be stored in JSON only")
            self.vector_db_available = False
        else:
            self.vector_db_available = True
    
    def _load_conversation_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from JSON file"""
        try:
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load conversation history: {e}")
        return []
    
    def _save_conversation_history(self):
        """Save conversation history to JSON file"""
        try:
            with open(self.conversation_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Could not save conversation history: {e}")
    
    def record_conversation_with_vector(self, user_message: str, bot_response: str, 
                                       session_id: str = "default", 
                                       analysis_context: Dict = None):
        """Record conversation in both JSON and vector database"""
        
        # 1. Traditional JSON storage
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "session_id": session_id,
            "analysis_context": analysis_context or {}
        }
        
        # 2. Vector database storage (if available)
        if self.conversation_vector_db and self.embedder:
            try:
                self._store_conversation_vectors(user_message, bot_response, conversation_entry)
            except Exception as e:
                print(f"⚠️ Failed to store in vector database: {e}")
        
        # 3. Traditional storage
        self.conversation_history.append(conversation_entry)
        self._save_conversation_history()
    
    def _store_conversation_vectors(self, user_message: str, bot_response: str, 
                                   metadata: Dict):
        """Store conversation as vectors with rich metadata"""
        
        # Create different embeddings for different query types
        conversation_chunks = [
            {
                "content": f"User Question: {user_message}",
                "type": "user_query",
                "full_exchange": f"User: {user_message}\nAssistant: {bot_response[:300]}..."
            },
            {
                "content": f"Analysis Response: {bot_response}",
                "type": "assistant_response", 
                "full_exchange": f"User: {user_message}\nAssistant: {bot_response[:300]}..."
            }
        ]
        
        for chunk in conversation_chunks:
            # Flatten metadata to only include simple types (ChromaDB requirement)
            flattened_metadata = self._flatten_metadata(metadata)
            enhanced_metadata = {
                **flattened_metadata,
                "chunk_type": chunk["type"],
                "full_exchange": chunk["full_exchange"]
            }
            
            # Generate unique ID
            doc_id = f"conv_{uuid.uuid4().hex[:8]}_{chunk['type']}"
            
            # Store in vector DB
            self.conversation_vector_db.add_texts(
                texts=[chunk["content"]],
                metadatas=[enhanced_metadata],
                ids=[doc_id]
            )
    
    def retrieve_relevant_conversation(self, current_query: str, top_k: int = 3, 
                                     session_filter: str = None) -> str:
        """Retrieve semantically relevant conversation history"""
        
        if not self.conversation_vector_db or not self.embedder:
            print("⚠️ Vector database not available, falling back to recent history")
            return self._get_recent_conversation_fallback(top_k)
        
        # Build search filters
        filter_dict = {}
        if session_filter:
            filter_dict["session_id"] = session_filter
        
        # Semantic search in vector database
        try:
            results = self.conversation_vector_db.similarity_search(
                current_query, 
                k=top_k * 2,  # Get more results to filter
                filter=filter_dict if filter_dict else None
            )
            
            # Post-process results for relevance and deduplication
            relevant_context = self._process_search_results(results, current_query)
            
            return relevant_context
            
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return self._get_recent_conversation_fallback(top_k)
    
    
    def _process_search_results(self, results, query: str) -> str:
        """Process and deduplicate search results into context string"""
        if not results:
            return ""
        
        # Group by conversation exchange to avoid fragmented context
        exchanges = {}
        for result in results:
            exchange = result.metadata.get("full_exchange", "")
            timestamp = result.metadata.get("timestamp", "")
            if exchange and exchange not in exchanges:
                exchanges[exchange] = timestamp
        
        # Sort by recency and build context
        sorted_exchanges = sorted(
            exchanges.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        context_parts = ["RELEVANT CONVERSATION HISTORY:"]
        for exchange, timestamp in sorted_exchanges:
            context_parts.append(f"[{timestamp[:19]}] {exchange}")
            context_parts.append("")  # Empty line for readability
        
        return "\n".join(context_parts)
    
    def _get_recent_conversation_fallback(self, limit: int = 3) -> str:
        """Fallback to recent conversation history when vector DB unavailable"""
        if not self.conversation_history:
            return ""
        
        recent = self.conversation_history[-limit:]
        context_parts = ["RECENT CONVERSATION HISTORY:"]
        
        for conv in recent:
            timestamp = conv.get("timestamp", "")[:19]
            user_msg = conv.get("user_message", "")
            bot_resp = conv.get("bot_response", "")[:300] + "..."
            
            context_parts.append(
                f"[{timestamp}] User: {user_msg}\nAssistant: {bot_resp}"
            )
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def search_conversations(self, query: str, k: int = 3) -> List[Any]:
        """Search conversations with fallback to recent history"""
        if self.conversation_vector_db:
            try:
                return self.conversation_vector_db.similarity_search(query, k=k)
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")
        
        # Fallback: return recent conversations as simple objects
        print("🔄 Using fallback: searching recent conversations")
        recent_conversations = self.conversation_history[-min(k*2, len(self.conversation_history)):]
        
        # Create simple result objects that mimic vector search results
        fallback_results = []
        for conv in recent_conversations[-k:]:
            # Create a simple object with metadata attribute
            class FallbackResult:
                def __init__(self, conversation):
                    self.metadata = {
                        "full_exchange": f"User: {conversation.get('user_message', '')}\nAssistant: {conversation.get('bot_response', '')}",
                        "timestamp": conversation.get('timestamp', ''),
                        "session_id": conversation.get('session_id', 'default')
                    }
            
            fallback_results.append(FallbackResult(conv))
        
        return fallback_results
    
    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten metadata to only include simple types for ChromaDB"""
        flattened = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                # Simple types - keep as is
                flattened[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                if value and all(isinstance(item, (str, int, float)) for item in value):
                    flattened[f"{key}_list"] = ", ".join(str(item) for item in value)
                    flattened[f"{key}_count"] = len(value)
                else:
                    flattened[f"{key}_count"] = len(value)
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                        flattened[f"{key}_{nested_key}"] = nested_value
                    elif isinstance(nested_value, list):
                        flattened[f"{key}_{nested_key}_count"] = len(nested_value)
            else:
                # Convert other types to string
                flattened[f"{key}_str"] = str(value)
        
        return flattened
    
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