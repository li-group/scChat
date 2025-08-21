"""
Function execution and conversation history management with vector database.

This module manages both function execution history and conversation history,
with optional vector database support for semantic conversation retrieval,
reducing token consumption while maintaining conversation context awareness.
"""

import os
import json
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
import logging
logger = logging.getLogger(__name__)


class FunctionHistoryManager:
    """
    Manages function execution history and conversation history with optional vector database.
    
    Features:
    - Function execution tracking with parameters, results, and success/failure status
    - Conversation history with semantic search capability  
    - Optional vector database for intelligent context retrieval
    - Graceful fallback when vector database is unavailable
    - LLM-driven conversation context analysis
    """
    
    def __init__(self, history_dir: str = "function_history"):
        # Initialize basic function history
        self.history_dir = history_dir
        self.history_file = os.path.join(history_dir, "execution_history.json")
        
        # Create directory if it doesn't exist
        os.makedirs(history_dir, exist_ok=True)
        
        # Load existing function execution history
        self.history = self._load_history()
        
        # Additional files for conversation history
        self.conversation_file = os.path.join(history_dir, "conversation_history.json")
        
        # Vector database for conversation history and enrichment results
        self.vector_db_dir = os.path.join(history_dir, "conversation_vectors")
        self._setup_vector_db()
        self._setup_enrichment_vector_db()
        
        # Load conversation history
        self.conversation_history = self._load_conversation_history()
    
    # ========== BASIC FUNCTION HISTORY METHODS ==========
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load function history from JSON file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.info(f"⚠️ Could not load function history: {e}")
        return []
    
    def _save_history(self):
        """Save function history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.info(f"⚠️ Could not save function history: {e}")
    
    def record_execution(self, function_name: str, parameters: Dict[str, Any], 
                        result: Any, success: bool, error: Optional[str] = None):
        """Record a function execution"""
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "function_name": function_name,
            "parameters": parameters,
            "success": success,
            "error": error,
            "result_summary": str(result)[:500] if result else None,  # Truncate long results
        }
        
        # Add to persistent history only
        self.history.append(execution_record)
        
        # Save to file
        self._save_history()
    
    def has_been_executed(self, function_name: str, parameters: Dict[str, Any]) -> bool:
        """Check if a function with specific parameters has been executed recently"""
        for record in reversed(self.history[-50:]):  # Check last 50 executions
            if (record["function_name"] == function_name and 
                record["parameters"] == parameters and 
                record["success"]):
                return True
        return False
    
    def get_recent_executions(self, function_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent executions of a specific function"""
        matching = [r for r in self.history if r["function_name"] == function_name and r["success"]]
        return matching[-limit:] if matching else []
    
    def get_available_results(self) -> Dict[str, Any]:
        """Get summary of available analysis results"""
        results_summary = {}
        
        for record in self.history:
            if record["success"]:
                func_name = record["function_name"]
                params = record["parameters"]
                
                if func_name == "process_cells":
                    cell_type = params.get("cell_type", "unknown")
                    if "processed_cell_types" not in results_summary:
                        results_summary["processed_cell_types"] = []
                    if cell_type not in results_summary["processed_cell_types"]:
                        results_summary["processed_cell_types"].append(cell_type)
                
                elif "enrichment" in func_name:
                    cell_type = params.get("cell_type", "unknown")
                    analysis_type = params.get("analysis", "unknown")
                    if "enrichment_analyses" not in results_summary:
                        results_summary["enrichment_analyses"] = {}
                    if cell_type not in results_summary["enrichment_analyses"]:
                        results_summary["enrichment_analyses"][cell_type] = []
                    if analysis_type not in results_summary["enrichment_analyses"][cell_type]:
                        results_summary["enrichment_analyses"][cell_type].append(analysis_type)
                
                elif func_name == "dea_split_by_condition":
                    cell_type = params.get("cell_type", "unknown")
                    if "dea_analyses" not in results_summary:
                        results_summary["dea_analyses"] = []
                    if cell_type not in results_summary["dea_analyses"]:
                        results_summary["dea_analyses"].append(cell_type)
        
        return results_summary
    
    # ========== CONVERSATION HISTORY METHODS ==========
    
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
            logger.info(f"✅ Vector database initialized at {self.vector_db_dir}")
            
        except ImportError as e:
            logger.info(f"⚠️ Vector DB dependencies not available: {e}")
            logger.info("   Install with: pip install langchain-huggingface langchain-chroma sentence-transformers")
            
        except Exception as e:
            logger.info(f"⚠️ Failed to initialize vector database: {e}")
            logger.info("   Falling back to simple conversation storage")
            
        # Always ensure these are set
        if self.conversation_vector_db is None:
            logger.info("📝 Using fallback: conversations will be stored in JSON only")
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
            logger.info(f"⚠️ Could not load conversation history: {e}")
        return []
    
    def _save_conversation_history(self):
        """Save conversation history to JSON file"""
        try:
            with open(self.conversation_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2, default=str)
        except Exception as e:
            logger.info(f"⚠️ Could not save conversation history: {e}")
    
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
                logger.info(f"⚠️ Failed to store in vector database: {e}")
        
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
            logger.info("⚠️ Vector database not available, falling back to recent history")
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
            logger.info(f"⚠️ Vector search failed: {e}")
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
                logger.info(f"⚠️ Vector search failed: {e}")
        
        # Fallback: return recent conversations as simple objects
        logger.info("🔄 Using fallback: searching recent conversations")
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
    
    # ========== ENRICHMENT VECTOR DATABASE METHODS ==========
    
    def _setup_enrichment_vector_db(self):
        """Initialize enrichment results vector database"""
        self.enrichment_vector_db = None
        self.enrichment_db_available = False
        
        try:
            # Only initialize if conversation vector DB was set up successfully
            if self.embedder is not None:
                # Create enrichment-specific directory
                enrichment_db_dir = os.path.join(self.vector_db_dir, "enrichment")
                
                # Initialize Chroma vector database for enrichment results
                self.enrichment_vector_db = Chroma(
                    collection_name="enrichment_analysis_results",
                    embedding_function=self.embedder,
                    persist_directory=enrichment_db_dir
                )
                self.enrichment_db_available = True
                logger.info(f"✅ Enrichment vector database initialized at {enrichment_db_dir}")
            else:
                logger.info("⚠️ Cannot initialize enrichment vector DB without embedder")
                
        except Exception as e:
            logger.info(f"⚠️ Failed to initialize enrichment vector database: {e}")
            logger.info("   Enrichment semantic search will not be available")
            self.enrichment_db_available = False
    
    def _create_standard_embedding(self, summary_row: Dict, raw_row: Optional[Dict], 
                                  analysis_type: str, cell_type: str, condition: str, rank: int) -> Dict[str, Any]:
        """Create embedding for GO/KEGG/Reactome with description from raw CSV"""
        
        # Handle actual CSV column names
        term_name = summary_row.get("Term", "") or summary_row.get("term_name", "")
        
        # Get description from raw data if available
        description = ""
        if raw_row:
            description = raw_row.get("description", "") or raw_row.get("Description", "") or raw_row.get("Term_description", "")
        
        # Get genes from raw data if available
        intersecting_genes = ""
        if raw_row:
            # Raw CSV has 'intersections' column with gene list
            intersections = raw_row.get("intersections", "")
            if intersections:
                try:
                    # Parse the list format: "['ESAM', 'JAM2', 'SELP', ...]"
                    import ast
                    gene_list = ast.literal_eval(intersections)
                    if isinstance(gene_list, list):
                        intersecting_genes = ",".join(gene_list)
                except:
                    # If parsing fails, use the raw string
                    intersecting_genes = str(intersections)
        
        # Get p-value (actual column: p_value)
        p_value = float(summary_row.get("p_value", 1.0))
        
        # Get fold change (actual column: avg_log2fc)
        avg_log2fc = float(summary_row.get("avg_log2fc", 0.0))
        
        # Get intersection size
        intersection_size = int(summary_row.get("intersection_size", 0))
        
        # Create searchable text combining term name, description, genes, and term ID
        term_id = summary_row.get("Term_ID", "")
        searchable_text = f"{term_name} {description} {intersecting_genes} {term_id}".strip()
        
        # Create metadata with all essential fields
        metadata = {
            "term_name": term_name,
            "term_id": term_id,
            "analysis_type": analysis_type,
            "cell_type": cell_type,
            "condition": condition,
            "adj_p_value": p_value,
            "avg_log2fc": avg_log2fc,
            "intersecting_genes": intersecting_genes,
            "description": description,
            "intersection_size": intersection_size,
            "rank": rank
        }
        
        return {
            "text": searchable_text,
            "metadata": metadata
        }
    
    def _create_gsea_embedding(self, summary_row: Dict, raw_row: Optional[Dict], 
                              cell_type: str, condition: str, rank: int) -> Dict[str, Any]:
        """Create embedding for GSEA with both term_name and gene_set"""
        
        # Handle actual CSV column names
        term_name = summary_row.get("Term", "") or summary_row.get("term_name", "")
        
        # For GSEA, gene_set might be in native column or similar
        gene_set = ""
        if raw_row:
            gene_set = raw_row.get("native", "") or raw_row.get("gene_set", "")
        
        # Get genes from raw data
        intersecting_genes = ""
        if raw_row:
            intersections = raw_row.get("intersections", "")
            if intersections:
                try:
                    import ast
                    gene_list = ast.literal_eval(intersections)
                    if isinstance(gene_list, list):
                        intersecting_genes = ",".join(gene_list)
                except:
                    intersecting_genes = str(intersections)
        
        # Get p-value and other metrics
        p_value = float(summary_row.get("p_value", 1.0))
        avg_log2fc = float(summary_row.get("avg_log2fc", 0.0))
        intersection_size = int(summary_row.get("intersection_size", 0))
        
        # Create searchable text combining term name, gene set, and genes
        searchable_text = f"{term_name} {gene_set} {intersecting_genes}".strip()
        
        # Create metadata with all essential fields (GSEA structure)
        metadata = {
            "term_name": term_name,
            "gene_set": gene_set,
            "analysis_type": "gsea",
            "cell_type": cell_type,
            "condition": condition,
            "adj_p_value": p_value,
            "avg_log2fc": avg_log2fc,
            "intersecting_genes": intersecting_genes,
            "description": "",  # Empty for GSEA
            "intersection_size": intersection_size,
            "rank": rank
        }
        
        return {
            "text": searchable_text,
            "metadata": metadata
        }
    
    def _read_dual_csvs(self, cell_type: str, analysis_type: str, condition: Optional[str] = None) -> tuple:
        """Read both summary and raw CSV files for an analysis type"""
        
        # Base enrichment directory
        base_dir = "scchatbot/enrichment"
        
        # Determine folder structure based on actual file organization
        if analysis_type == "go":
            # GO has subdirectories for bp, cc, mf
            # Try to find which domain folder exists
            for domain in ["bp", "cc", "mf"]:
                if condition and condition != "all_combined":
                    # Condition-specific GO folder
                    folder = f"{base_dir}/go_{condition}_{domain}"
                    summary_file = f"{folder}/results_summary_{cell_type}_{condition}.csv"
                    raw_file = f"{folder}/results_raw_{cell_type}_{condition}.csv"
                else:
                    # Full dataset GO folder
                    folder = f"{base_dir}/go_{domain}"
                    summary_file = f"{folder}/results_summary_{cell_type}.csv"
                    raw_file = f"{folder}/results_raw_{cell_type}.csv"
                
                # Check if this domain folder exists
                if os.path.exists(folder):
                    break
            else:
                # No GO domain folder found, try generic path
                if condition and condition != "all_combined":
                    summary_file = f"{base_dir}/results_summary_{cell_type}_{condition}.csv"
                    raw_file = f"{base_dir}/results_raw_{cell_type}_{condition}.csv"
                else:
                    summary_file = f"{base_dir}/results_summary_{cell_type}.csv"
                    raw_file = f"{base_dir}/results_raw_{cell_type}.csv"
        
        else:
            # For KEGG, Reactome, GSEA - simpler structure
            if condition and condition != "all_combined":
                # Condition-specific folder
                folder = f"{base_dir}/{analysis_type}_{condition}"
                summary_file = f"{folder}/results_summary_{cell_type}_{condition}.csv"
                raw_file = f"{folder}/results_raw_{cell_type}_{condition}.csv"
            else:
                # Full dataset folder
                folder = f"{base_dir}/{analysis_type}"
                summary_file = f"{folder}/results_summary_{cell_type}.csv"
                raw_file = f"{folder}/results_raw_{cell_type}.csv"
                
                # Special case for raw files that might have analysis type suffix
                if not os.path.exists(raw_file):
                    raw_file_alt = f"{folder}/results_raw_{cell_type}_{analysis_type}.csv"
                    if os.path.exists(raw_file_alt):
                        raw_file = raw_file_alt
        
        summary_df = None
        raw_df = None
        
        # Read summary CSV (required)
        try:
            if os.path.exists(summary_file):
                summary_df = pd.read_csv(summary_file)
                logger.info(f"📊 Read summary CSV: {summary_file} ({len(summary_df)} terms)")
            else:
                logger.info(f"⚠️ Summary CSV not found: {summary_file}")
                # Try to find any matching files in the expected folder
                if 'folder' in locals() and os.path.exists(folder):
                    files = os.listdir(folder)
                    matching_files = [f for f in files if 'summary' in f and cell_type in f]
                    if matching_files:
                        logger.info(f"   💡 Found alternative files: {matching_files}")
                return None, None
        except Exception as e:
            logger.info(f"❌ Error reading summary CSV {summary_file}: {e}")
            return None, None
        
        # Read raw CSV (optional for descriptions)
        try:
            if os.path.exists(raw_file):
                raw_df = pd.read_csv(raw_file)
                logger.info(f"📋 Read raw CSV: {raw_file} ({len(raw_df)} terms)")
            else:
                logger.info(f"⚠️ Raw CSV not found: {raw_file} (descriptions will be empty)")
        except Exception as e:
            logger.info(f"⚠️ Error reading raw CSV {raw_file}: {e}")
        
        return summary_df, raw_df
    
    def index_enrichment_results_from_dual_csvs(self, enrichment_results: Dict[str, Any], 
                                               cell_type: str, session_id: Optional[str] = None):
        """Index all enrichment terms from summary and raw CSV files"""
        
        if not self.enrichment_db_available or not self.enrichment_vector_db:
            logger.info("⚠️ Enrichment vector database not available, skipping indexing")
            return
        
        logger.info(f"🔍 Starting enrichment indexing for {cell_type}...")
        
        # Auto-detect performed analyses from enrichment_results structure
        performed_analyses = []
        for analysis_type in ["reactome", "go", "kegg", "gsea"]:
            if analysis_type in enrichment_results:
                performed_analyses.append(analysis_type)
        
        if not performed_analyses:
            logger.info("⚠️ No enrichment analyses found in results structure")
            return
        
        logger.info(f"🧬 Detected analyses: {performed_analyses}")
        
        # Determine conditions to index
        conditions_to_index = ["all_combined"]  # Always index full dataset
        
        # Check if condition-specific results exist
        conditions = enrichment_results.get("conditions", {})
        if conditions and isinstance(conditions, dict):
            available_conditions = conditions.get("available_conditions", [])
            if available_conditions:
                conditions_to_index.extend(available_conditions)
        
        total_indexed = 0
        
        # Index each analysis type and condition combination
        for analysis_type in performed_analyses:
            if analysis_type == "go":
                # GO needs to be indexed separately for each domain (bp, cc, mf)
                for domain in ["bp", "cc", "mf"]:
                    for condition in conditions_to_index:
                        try:
                            indexed_count = self._index_single_analysis(
                                cell_type, f"go_{domain}", condition, session_id
                            )
                            total_indexed += indexed_count
                            
                        except Exception as e:
                            logger.info(f"❌ Error indexing go_{domain} for condition {condition}: {e}")
                            continue
            else:
                # Other analysis types (kegg, reactome, gsea)
                for condition in conditions_to_index:
                    try:
                        indexed_count = self._index_single_analysis(
                            cell_type, analysis_type, condition, session_id
                        )
                        total_indexed += indexed_count
                        
                    except Exception as e:
                        logger.info(f"❌ Error indexing {analysis_type} for condition {condition}: {e}")
                        continue
        
        logger.info(f"✅ Enrichment indexing complete: {total_indexed} terms indexed for {cell_type}")
    
    def _index_single_analysis(self, cell_type: str, analysis_type: str, 
                              condition: str, session_id: Optional[str] = None) -> int:
        """Index a single analysis type and condition combination"""
        
        # Extract base analysis type for GO domains
        base_analysis_type = analysis_type
        if analysis_type.startswith("go_"):
            base_analysis_type = "go"
        
        # Read CSV files
        summary_df, raw_df = self._read_dual_csvs(cell_type, base_analysis_type, condition)
        if summary_df is None:
            return 0
        
        # Filter summary data for this analysis type
        # The actual CSVs don't have analysis_type column - they are separated by folder
        analysis_data = summary_df.copy()
        
        if len(analysis_data) == 0:
            logger.info(f"⚠️ No data found for {analysis_type} in summary CSV")
            return 0
        
        # Create index for raw data lookup (if available)
        raw_lookup = {}
        if raw_df is not None:
            # Match by Term column (which exists in both summary and raw)
            if 'Term' in raw_df.columns:
                for _, row in raw_df.iterrows():
                    term_name = row.get('Term', '')
                    if term_name:
                        raw_lookup[term_name] = row.to_dict()
            elif 'term_name' in raw_df.columns:
                for _, row in raw_df.iterrows():
                    term_name = row.get('term_name', '')
                    if term_name:
                        raw_lookup[term_name] = row.to_dict()
        
        # Prepare batch data for insertion
        texts = []
        metadatas = []
        ids = []
        
        for rank, (_, summary_row) in enumerate(analysis_data.iterrows(), 1):
            # Get term name from the actual column name in CSV
            term_name = summary_row.get('Term', '') or summary_row.get('term_name', '')
            if not term_name:
                continue
            
            # Get corresponding raw data
            raw_row = raw_lookup.get(term_name)
            
            # Create embedding based on analysis type
            if base_analysis_type == "gsea":
                embedding_data = self._create_gsea_embedding(
                    summary_row.to_dict(), raw_row, cell_type, condition, rank
                )
            else:
                # Use base_analysis_type for metadata (go, not go_bp)
                embedding_data = self._create_standard_embedding(
                    summary_row.to_dict(), raw_row, base_analysis_type, cell_type, condition, rank
                )
            
            # Flatten metadata for ChromaDB compatibility
            flattened_metadata = self._flatten_metadata(embedding_data["metadata"])
            
            # Add session info if available
            if session_id:
                flattened_metadata["session_id"] = session_id
            
            texts.append(embedding_data["text"])
            metadatas.append(flattened_metadata)
            ids.append(f"enrich_{uuid.uuid4().hex[:8]}_{analysis_type}_{cell_type}_{condition}_{rank}")
        
        if texts:
            # Batch insert into vector database
            try:
                self.enrichment_vector_db.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"📝 Indexed {len(texts)} terms for {analysis_type}/{condition}")
                return len(texts)
            except Exception as e:
                logger.info(f"❌ Error inserting {analysis_type} data into vector DB: {e}")
                return 0
        
        return 0
    
    def search_enrichment_semantic(self, query: str, cell_type: str, 
                                  condition_filter: Optional[str] = None,
                                  analysis_type_filter: Optional[str] = None,
                                  k: int = 10) -> Dict[str, Any]:
        """Semantic search with exact cell type matching"""
        
        if not self.enrichment_db_available or not self.enrichment_vector_db:
            return {
                "error": "Enrichment vector database not available",
                "query": query,
                "cell_type": cell_type,
                "total_matches": 0,
                "results": []
            }
        
        try:
            # Build search filters for exact matching
            filter_dict = {"cell_type": cell_type}  # EXACT cell type matching
            
            if condition_filter:
                filter_dict["condition"] = condition_filter
            
            if analysis_type_filter:
                filter_dict["analysis_type"] = analysis_type_filter
            
            logger.info(f"🔍 Searching enrichment: '{query}' for {cell_type} with filters: {filter_dict}")
            
            # Perform semantic search with relevance scores
            results = self.enrichment_vector_db.similarity_search_with_relevance_scores(
                query, 
                k=k * 2,  # Get more results for filtering and deduplication
                filter=filter_dict
            )
            
            # Process and deduplicate results
            processed_results = self._process_enrichment_search_results(results, query)
            
            return {
                "query": query,
                "cell_type": cell_type,
                "condition_filter": condition_filter,
                "analysis_type_filter": analysis_type_filter,
                "total_matches": len(processed_results),
                "results": processed_results[:k]  # Return top k results
            }
            
        except Exception as e:
            logger.info(f"❌ Enrichment search failed: {e}")
            return {
                "error": f"Search failed: {e}",
                "query": query,
                "cell_type": cell_type,
                "total_matches": 0,
                "results": []
            }
    
    def _process_enrichment_search_results(self, results: List[tuple], query: str) -> List[Dict[str, Any]]:
        """Process and deduplicate search results"""
        
        if not results:
            return []
        
        # Deduplicate by term name and collect best matches
        seen_terms = {}
        
        for doc, similarity_score in results:
            term_name = doc.metadata.get("term_name", "")
            
            if term_name and (term_name not in seen_terms or 
                             similarity_score > seen_terms[term_name]["similarity_score"]):
                
                # Convert flattened metadata back to structured format
                result_data = {
                    "term_name": term_name,
                    "analysis_type": doc.metadata.get("analysis_type", ""),
                    "adj_p_value": doc.metadata.get("adj_p_value", 1.0),
                    "avg_log2fc": doc.metadata.get("avg_log2fc", 0.0),
                    "intersecting_genes": doc.metadata.get("intersecting_genes", ""),
                    "description": doc.metadata.get("description", ""),
                    "rank": doc.metadata.get("rank", 0),
                    "condition": doc.metadata.get("condition", ""),
                    "similarity_score": float(similarity_score)
                }
                
                # Add gene_set for GSEA results
                if doc.metadata.get("analysis_type") == "gsea":
                    result_data["gene_set"] = doc.metadata.get("gene_set", "")
                
                seen_terms[term_name] = result_data
        
        # Sort by similarity score (descending)
        processed_results = sorted(
            seen_terms.values(), 
            key=lambda x: x["similarity_score"], 
            reverse=True
        )
        
        logger.info(f"🎯 Found {len(processed_results)} unique enrichment terms for query: '{query}'")
        
        # DEBUG: Show what was actually found
        if processed_results:
            logger.info("📋 SEMANTIC SEARCH RESULTS:")
            for i, result in enumerate(processed_results[:10], 1):  # Show top 10
                term_name = result.get("term_name", "Unknown")
                analysis_type = result.get("analysis_type", "unknown").upper()
                p_value = result.get("adj_p_value", "N/A")
                similarity = result.get("similarity_score", 0)
                rank = result.get("rank", "N/A")
                logger.info(f"  {i}. [{analysis_type}] {term_name}")
                logger.info(f"     p-value: {p_value}, similarity: {similarity:.3f}, rank: {rank}")
        else:
            logger.info("📋 No results found in processed_results")
        
        return processed_results