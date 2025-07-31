import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# OpenAI functionality now handled through LangChain
OPENAI_AVAILABLE = True

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

# Vector search dependencies
try:
    import pickle
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    pickle = faiss = np = SentenceTransformer = None


@dataclass 
class PathwayMatch:
    """Represents a pathway match from the database"""
    pathway_name: str
    database: str
    method: str
    gene_set_library: Optional[str]
    confidence: float
    description: str


@dataclass
class EnrichmentRecommendation:
    """Represents enrichment analysis recommendations"""
    analyses: List[str]
    gene_set_library: Optional[str]
    description: str
    expected_outcome: str
    confidence: float
    reasoning: str


class EnrichmentChecker:
    """
    Intelligent pathway analysis recommendation system.
    
    Uses Neo4j RAG database and GPT-4 to provide smart enrichment analysis
    method selection based on user pathway queries.
    """
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", 
                 neo4j_password="37754262", neo4j_database="pathways", 
                 confidence_threshold=0.8):
        """Initialize enrichment checker with Neo4j connection and vector search."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.confidence_threshold = confidence_threshold
        self.driver = None
        self.connection_status = "disconnected"
        
        # Vector search initialization
        self.vector_model = None
        self.vector_index = None
        self.pathway_data = None
        self.vector_search_status = "not_loaded"
        
        # Initialize Neo4j connection if available
        if NEO4J_AVAILABLE:
            try:
                self.driver = GraphDatabase.driver(
                    neo4j_uri, 
                    auth=(neo4j_user, neo4j_password)
                )
                # Test connection
                with self.driver.session(database=neo4j_database) as session:
                    result = session.run("MATCH (p:Pathway) RETURN count(p) as count LIMIT 1")
                    count = result.single()
                    print(f"   Neo4j test: Found {count['count'] if count else 0} pathways in '{neo4j_database}' database")
                self.connection_status = "connected"
                print("✅ EnrichmentChecker: Neo4j connection established")
            except Exception as e:
                print(f"⚠️ EnrichmentChecker: Neo4j connection failed: {e}")
                self.driver = None
                self.connection_status = f"failed: {str(e)}"
        else:
            print("⚠️ EnrichmentChecker: Neo4j module not available")
            self.connection_status = "neo4j_unavailable"
        
        # Initialize vector search if available
        if VECTOR_SEARCH_AVAILABLE:
            self._load_vector_search_model()
        else:
            print("⚠️ EnrichmentChecker: Vector search dependencies not available")
    
    def _load_vector_search_model(self):
        """Load vector search model and pathway data from pre-built files."""
        try:
            # Path to vector search files - relative to project root
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/scChat_v2/scchatbot
            project_root = os.path.dirname(current_dir)  # /path/to/scChat_v2  
            media_path = os.path.join(project_root, "media")
            pathway_data_path = os.path.join(media_path, "pathway_data.pkl")
            pathway_index_path = os.path.join(media_path, "pathway_index.faiss")
            
            # Check if files exist
            if not os.path.exists(pathway_data_path):
                print(f"⚠️ EnrichmentChecker: Pathway data not found at {pathway_data_path}")
                self.vector_search_status = "data_not_found"
                return
            
            if not os.path.exists(pathway_index_path):
                print(f"⚠️ EnrichmentChecker: Pathway index not found at {pathway_index_path}")
                self.vector_search_status = "index_not_found"
                return
            
            # Load pathway data
            with open(pathway_data_path, 'rb') as f:
                self.pathway_data = pickle.load(f)
            
            # Load FAISS index
            self.vector_index = faiss.read_index(pathway_index_path)
            
            # Load sentence transformer model (same as used for creating embeddings)
            self.vector_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.vector_search_status = "loaded"
            print(f"✅ EnrichmentChecker: Vector search loaded ({len(self.pathway_data)} pathways)")
            
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: Vector search loading failed: {e}")
            self.vector_search_status = f"failed: {str(e)}"
            self.vector_model = None
            self.vector_index = None
            self.pathway_data = None
    
    def _vector_search_pathways(self, pathway_query: str, k: int = 5) -> List[Dict]:
        """
        Use pre-built vector model to find k most similar pathways.
        
        Args:
            pathway_query: User pathway query (e.g., "interferon response")
            k: Number of top pathways to return
            
        Returns:
            List of pathway matches with similarity scores
        """
        if self.vector_search_status != "loaded":
            print(f"⚠️ EnrichmentChecker: Vector search not available (status: {self.vector_search_status})")
            return []
        
        try:
            # Encode the query
            query_embedding = self.vector_model.encode([pathway_query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search in FAISS index
            similarities, indices = self.vector_index.search(query_embedding, k)
            
            # Convert results to pathway matches
            pathway_matches = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.pathway_data):
                    pathway_info = self.pathway_data[idx]
                    pathway_name = pathway_info.get('name', f'Pathway_{idx}')
                    pathway_matches.append({
                        'pathway_name': pathway_name,
                        'database': pathway_info.get('database', 'unknown'),
                        'similarity_score': float(similarity),
                        'description': pathway_info.get('description', ''),
                        'gene_set_library': pathway_info.get('gene_set_library'),
                        'rank': i + 1
                    })
            
            print(f"🔍 EnrichmentChecker: Vector search found {len(pathway_matches)} matches for '{pathway_query}'")
            # Debug: Show exact names from vector search
            for match in pathway_matches[:3]:
                print(f"   📌 Vector result {match['rank']}: '{match['pathway_name']}' (score: {match['similarity_score']:.3f})")
            return pathway_matches
            
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: Vector search failed: {e}")
            return []
    
    def _validate_vector_matches_in_neo4j(self, vector_matches: List[Dict]) -> List[EnrichmentRecommendation]:
        """
        Validate vector search results against Neo4j database and build recommendations.
        
        Args:
            vector_matches: Results from vector search
            
        Returns:
            List of validated enrichment recommendations
        """
        if not self.driver:
            print("⚠️ EnrichmentChecker: No Neo4j connection for validation")
            return []
        
        validated_recommendations = []
        
        try:
            for match in vector_matches:
                pathway_name = match['pathway_name']
                database = match['database']
                similarity_score = match['similarity_score']
                
                # Query Neo4j to validate pathway exists and get metadata
                with self.driver.session() as session:
                    # Search for pathway in Neo4j
                    query = """
                    MATCH (p:Pathway) 
                    WHERE toLower(p.name) CONTAINS toLower($pathway_name)
                       OR toLower(p.displayName) CONTAINS toLower($pathway_name)
                    RETURN p.name as name, p.displayName as displayName, 
                           p.database as database, p.description as description
                    LIMIT 1
                    """
                    result = session.run(query, pathway_name=pathway_name)
                    record = result.single()
                    
                    if record:
                        # Build recommendation based on database type
                        if database.lower() in ['go', 'gene_ontology']:
                            recommendation = EnrichmentRecommendation(
                                analyses=['go'],
                                gene_set_library=None,
                                description=f"Gene Ontology analysis focusing on {record['name']}",
                                expected_outcome=f"GO enrichment results for {record['name']}",
                                confidence=similarity_score,
                                reasoning=f"Vector search identified relevant GO term (similarity: {similarity_score:.3f})"
                            )
                        elif database.lower() in ['kegg']:
                            recommendation = EnrichmentRecommendation(
                                analyses=['kegg'],
                                gene_set_library=None,
                                description=f"KEGG pathway analysis focusing on {record['name']}",
                                expected_outcome=f"KEGG enrichment results for {record['name']}",
                                confidence=similarity_score,
                                reasoning=f"Vector search identified relevant KEGG pathway (similarity: {similarity_score:.3f})"
                            )
                        elif database.lower() in ['reactome']:
                            recommendation = EnrichmentRecommendation(
                                analyses=['reactome'],
                                gene_set_library=None,
                                description=f"Reactome pathway analysis focusing on {record['name']}",
                                expected_outcome=f"Reactome enrichment results for {record['name']}",
                                confidence=similarity_score,
                                reasoning=f"Vector search identified relevant Reactome pathway (similarity: {similarity_score:.3f})"
                            )
                        else:
                            # Default to GSEA for other databases or unknown
                            gene_set_lib = match.get('gene_set_library', 'MSigDB_Hallmark_2020')
                            recommendation = EnrichmentRecommendation(
                                analyses=['gsea'],
                                gene_set_library=gene_set_lib,
                                description=f"GSEA analysis using {gene_set_lib} focusing on {record['name']}",
                                expected_outcome=f"GSEA enrichment results for {record['name']}",
                                confidence=similarity_score,
                                reasoning=f"Vector search identified relevant pathway for GSEA (similarity: {similarity_score:.3f})"
                            )
                        
                        validated_recommendations.append(recommendation)
                        print(f"✅ EnrichmentChecker: Validated '{pathway_name}' in Neo4j (confidence: {similarity_score:.3f})")
                        
                        # Only return the first high-confidence match to avoid too many recommendations
                        if similarity_score > 0.7:
                            break
                    else:
                        print(f"⚠️ EnrichmentChecker: '{pathway_name}' not found in Neo4j database")
            
            return validated_recommendations
            
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: Neo4j validation failed: {e}")
            return []
    
    def enhance_enrichment_plan(self, plan_step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance enrichment plan step with pathway intelligence.
        
        Args:
            plan_step: Original plan step with pathway_include field
            
        Returns:
            Enhanced plan step with analyses, description, expected_outcome
        """
        parameters = plan_step.get("parameters", {})
        cell_type = parameters.get("cell_type", "unknown")
        pathway_include = parameters.get("pathway_include")
        explicit_analyses = parameters.get("analyses")
        
        # If explicit analyses provided, use them directly
        if explicit_analyses and not pathway_include:
            return self._enhance_explicit_analysis_plan(plan_step, cell_type, explicit_analyses)
        
        # If pathway_include provided, use pathway intelligence
        if pathway_include:
            return self._enhance_pathway_semantic_plan(plan_step, cell_type, pathway_include)
        
        # Default enrichment plan
        return self._enhance_default_plan(plan_step, cell_type)
    
    def _enhance_explicit_analysis_plan(self, plan_step: Dict[str, Any], 
                                      cell_type: str, analyses: List[str]) -> Dict[str, Any]:
        """Enhance plan for explicit analysis requests (e.g., 'run GO analysis')"""
        enhanced_step = plan_step.copy()
        
        # Generate description and expected outcome for explicit analyses
        analysis_names = {
            'go': 'Gene Ontology (GO)',
            'kegg': 'KEGG pathway', 
            'reactome': 'Reactome pathway',
            'gsea': 'Gene Set Enrichment Analysis (GSEA)'
        }
        
        analysis_descriptions = [analysis_names.get(a.lower(), a.upper()) for a in analyses]
        description = f"Perform {', '.join(analysis_descriptions)} enrichment analysis on {cell_type}."
        expected_outcome = f"{', '.join(analysis_descriptions)} enrichment results for {cell_type}."
        
        enhanced_step.update({
            "description": description,
            "expected_outcome": expected_outcome
        })
        
        print(f"✅ EnrichmentChecker: Enhanced explicit analysis plan for {cell_type}")
        return enhanced_step
    
    def _enhance_pathway_semantic_plan(self, plan_step: Dict[str, Any], 
                                     cell_type: str, pathway_query: str) -> Dict[str, Any]:
        """Enhance plan for semantic pathway queries (e.g., 'find IFN pathways')"""
        try:
            # Get pathway recommendations from Neo4j
            recommendations = self._get_pathway_recommendations(pathway_query)
            
            if recommendations:
                # Select multiple high-confidence recommendations
                selected_recommendations = self._select_high_confidence_recommendations(
                    recommendations, confidence_threshold=self.confidence_threshold
                )
                enhanced_step = self._build_enhanced_step_from_recommendations(
                    plan_step, cell_type, pathway_query, selected_recommendations
                )
                print(f"✅ EnrichmentChecker: Enhanced pathway semantic plan for '{pathway_query}' in {cell_type}")
                return enhanced_step
            else:
                print(f"⚠️ EnrichmentChecker: No pathway matches for '{pathway_query}', using default GSEA")
                return self._enhance_default_gsea_plan(plan_step, cell_type, pathway_query)
                
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: Pathway intelligence failed: {e}")
            return self._enhance_default_gsea_plan(plan_step, cell_type, pathway_query)
    
    def _enhance_default_plan(self, plan_step: Dict[str, Any], cell_type: str) -> Dict[str, Any]:
        """Enhance plan with default GO enrichment analysis"""
        enhanced_step = plan_step.copy()
        enhanced_step["parameters"].update({
            "analyses": ["go"]
        })
        enhanced_step.update({
            "description": f"Perform Gene Ontology (GO) enrichment analysis on {cell_type}.",
            "expected_outcome": f"GO enrichment results for {cell_type} covering biological processes, molecular functions, and cellular components."
        })
        return enhanced_step
    
    def _create_standalone_go_recommendation(self) -> List[EnrichmentRecommendation]:
        """Create a standalone GO analysis recommendation for empty queries"""
        go_recommendation = EnrichmentRecommendation(
            analyses=["go"],
            gene_set_library=None,
            description="Perform comprehensive Gene Ontology (GO) enrichment analysis.",
            expected_outcome="GO enrichment results covering biological processes, molecular functions, and cellular components.",
            confidence=1.0,
            reasoning="Default GO analysis for general pathway exploration"
        )
        
        print(f"✅ EnrichmentChecker: Created standalone GO recommendation for empty query")
        return [go_recommendation]
    
    def _enhance_default_gsea_plan(self, plan_step: Dict[str, Any], 
                                 cell_type: str, pathway_query: str) -> Dict[str, Any]:
        """Enhance plan with default GSEA analysis when pathway intelligence fails"""
        enhanced_step = plan_step.copy()
        enhanced_step["parameters"].update({
            "analyses": ["gsea"],
            "gene_set_library": "MSigDB_Hallmark_2020"
        })
        enhanced_step.update({
            "description": f"Perform GSEA enrichment analysis on {cell_type} using MSigDB Hallmark gene sets (default for '{pathway_query}').",
            "expected_outcome": f"GSEA enrichment results for {cell_type} using MSigDB Hallmark pathways."
        })
        return enhanced_step
    
    def _get_pathway_recommendations(self, pathway_query: str, top_k: int = 3) -> List[EnrichmentRecommendation]:
        """Simplified pipeline: Query Check → Vector Search → Neo4j Lookup → Done"""
        
        # STEP 1: Empty query check - go directly to standalone GO
        if not pathway_query or pathway_query.strip() == "" or pathway_query.strip() == '""':
            print(f"🔄 EnrichmentChecker: Empty pathway query, using standalone GO analysis")
            return self._create_standalone_go_recommendation()
        
        # STEP 2: Vector search for top pathways
        print(f"🔍 EnrichmentChecker: Vector search for '{pathway_query}'")
        vector_matches = self._vector_search_pathways(pathway_query, k=top_k)
        
        if not vector_matches:
            print(f"⚠️ EnrichmentChecker: No vector matches found, using standalone GO")
            return self._create_standalone_go_recommendation()
        
        # STEP 3: For each pathway, lookup method + gene_set_library in Neo4j
        print(f"🔍 EnrichmentChecker: Looking up methods for {len(vector_matches)} pathways")
        recommendations = []
        
        for match in vector_matches:
            pathway_name = match['pathway_name']
            similarity_score = match['similarity_score']
            
            # Direct lookup: pathway → method + library
            method_info = self._lookup_pathway_method_in_neo4j(pathway_name)
            
            if method_info:
                recommendation = EnrichmentRecommendation(
                    analyses=[method_info['method']],
                    gene_set_library=method_info.get('gene_set_library'),
                    description=f"Perform {method_info['method'].upper()} analysis targeting '{pathway_name}' pathway",
                    expected_outcome=f"{method_info['method'].upper()} enrichment results for {pathway_name}",
                    confidence=similarity_score,
                    reasoning=f"Vector search found relevant pathway: {pathway_name} (similarity: {similarity_score:.3f})"
                )
                recommendations.append(recommendation)
                print(f"✅ EnrichmentChecker: Found {method_info['method'].upper()} for pathway '{pathway_name}'")
            else:
                print(f"⚠️ EnrichmentChecker: No method found for pathway '{pathway_name}'")
        
        if not recommendations:
            print(f"⚠️ EnrichmentChecker: No valid method mappings found, using standalone GO")
            return self._create_standalone_go_recommendation()
        
        print(f"✅ EnrichmentChecker: Simplified pipeline successful ({len(recommendations)} recommendations)")
        return recommendations
    
    def _lookup_pathway_method_in_neo4j(self, pathway_name: str) -> Optional[Dict[str, str]]:
        """
        Look up method and gene_set_library for a specific pathway in Neo4j.
        
        Correct schema:
        - Pathway -[:FOUND_IN]-> Database
        - Database -[:CONTAIN_LIBRARY]-> GeneSetLibrary (if applicable)
        
        Args:
            pathway_name: Name of the pathway to look up
            
        Returns:
            Dict with 'method' and optional 'gene_set_library', or None if not found
        """
        if not self.driver:
            print(f"⚠️ EnrichmentChecker: No Neo4j connection for pathway lookup")
            return None
        
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                print(f"🔍 Neo4j lookup: Searching for pathway '{pathway_name}' in database '{self.neo4j_database}'")
                
                # First, let's debug - check if we can query anything
                debug_query = "MATCH (p:Pathway) RETURN count(p) as total"
                debug_result = session.run(debug_query)
                debug_count = debug_result.single()
                print(f"   DEBUG: Total pathways in database: {debug_count['total'] if debug_count else 0}")
                
                # Look up: Pathway -> Database (simplified without GeneSetLibrary)
                query = """
                MATCH (p:Pathway)-[:FOUND_IN]->(d:Database)
                WHERE toLower(p.name) = toLower($pathway_name)
                RETURN d.name as database, p.name as exact_name
                LIMIT 1
                """
                result = session.run(query, pathway_name=pathway_name)
                record = result.single()
                
                if record:
                    database_name = record['database']
                    exact_name = record['exact_name']
                    gene_set_library = None  # Will be set for GSEA below
                    print(f"   ✅ Found in Neo4j: '{exact_name}' in database '{database_name}'")
                    
                    # Map database name to analysis method
                    # Actual databases in Neo4j: Gsea, GO, KEGG, Reactome
                    method_mapping = {
                        'GO': 'go',
                        'KEGG': 'kegg', 
                        'Reactome': 'reactome',
                        'Gsea': 'gsea'
                    }
                    
                    method = method_mapping.get(database_name, 'go')  # Default to GO
                    
                    result_info = {'method': method}
                    
                    # For GSEA database, we need to find the gene_set_library
                    if database_name == 'Gsea':
                        # Query for gene_set_library
                        lib_query = """
                        MATCH (p:Pathway)<-[:CONTAINS_PATHWAY]-(lib:GeneSetLibrary)<-[:CONTAINS_LIBRARY]-(d:Database {name: "Gsea"})
                        WHERE toLower(p.name) = toLower($pathway_name)
                        RETURN lib.name as gene_set_library
                        LIMIT 1
                        """
                        lib_result = session.run(lib_query, pathway_name=pathway_name)
                        lib_record = lib_result.single()
                        
                        if lib_record:
                            gene_set_library = lib_record['gene_set_library']
                            result_info['gene_set_library'] = gene_set_library
                            print(f"✅ Found pathway mapping: '{exact_name}' → {method.upper()} (library: {gene_set_library})")
                        else:
                            # Default GSEA library
                            result_info['gene_set_library'] = 'MSigDB_Hallmark_2020'
                            print(f"✅ Found pathway mapping: '{exact_name}' → {method.upper()} (database: {database_name}, default library)")
                    else:
                        print(f"✅ Found pathway mapping: '{exact_name}' → {method.upper()} (database: {database_name})")
                    
                    # Debug name mismatch
                    if exact_name.lower() != pathway_name.lower():
                        print(f"   ⚠️ Name mismatch: Vector gave '{pathway_name}', Neo4j has '{exact_name}'")
                    
                    return result_info
                else:
                    # Let's check if the pathway exists at all
                    check_query = """
                    MATCH (p:Pathway)
                    WHERE toLower(p.name) = toLower($pathway_name)
                    RETURN p.name as name, labels(p) as labels
                    LIMIT 5
                    """
                    check_result = session.run(check_query, pathway_name=pathway_name)
                    check_records = list(check_result)
                    
                    if check_records:
                        print(f"   ⚠️ Pathway exists but no FOUND_IN relationship:")
                        for r in check_records:
                            print(f"      - '{r['name']}' with labels: {r['labels']}")
                    else:
                        # Try partial match
                        fuzzy_query = """
                        MATCH (p:Pathway)
                        WHERE toLower(p.name) CONTAINS toLower($pathway_name)
                        RETURN p.name as name
                        LIMIT 5
                        """
                        fuzzy_result = session.run(fuzzy_query, pathway_name=pathway_name)
                        fuzzy_records = list(fuzzy_result)
                        
                        if fuzzy_records:
                            print(f"   ⚠️ No exact match, but found similar pathways:")
                            for r in fuzzy_records:
                                print(f"      - '{r['name']}'")
                
                print(f"⚠️ No pathway found for: {pathway_name}")
                return None
                
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: Neo4j pathway lookup failed: {e}")
            return None
    
    def _get_llm_method_recommendations(self, pathway_query: str) -> List[Dict[str, str]]:
        """Use LLM to get specific method recommendations for pathway query"""
        if not OPENAI_AVAILABLE:
            print("⚠️ EnrichmentChecker: OpenAI not available for LLM recommendations")
            return []
            
        try:
            prompt = f"""
            You are an expert in pathway enrichment analysis. I need recommendations for "{pathway_query}".
            
            Available methods:
            - GO (Gene Ontology) 
            - KEGG pathways
            - Reactome pathways
            - GSEA with gene libraries (MSigDB_Hallmark_2020, WikiPathways_2024_Human, KEGG_2021, etc.)
            
            IMPORTANT: Return ONLY valid JSON, no other text or explanations.
            
            For "{pathway_query}", recommend the most appropriate methods and specific pathways.
            
            JSON format:
            {{
              "recommended_methods": [
                {{"Method": "GO", "pathway": "pathway_name"}},
                {{"Method": "KEGG", "pathway": "pathway_name"}},
                {{"Method": "Reactome", "pathway": "pathway_name"}},
                {{"Method": "GSEA", "Gene library": "library_name", "pathway": "pathway_name"}}
              ]
            }}
            
            Examples for guidance:
            - "interferon response" → Reactome "Interferon Signaling", GSEA "MSigDB_Hallmark_2020" "HALLMARK_INTERFERON_ALPHA_RESPONSE"
            - "cell cycle" → GO "cell cycle", KEGG "Cell cycle", GSEA "MSigDB_Hallmark_2020" "HALLMARK_G2M_CHECKPOINT"
            
            Query: "{pathway_query}"
            
            Return only the JSON:"""
            
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Create messages in LangChain format
            messages = [
                SystemMessage(content="You are an expert in pathway analysis. Generate responses in JSON format."),
                HumanMessage(content=prompt)
            ]
            
            # Initialize model
            model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=500,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            
            # Get response
            response = model.invoke(messages)
            content = response.content.strip()
            
            # Clean up the response to extract JSON
            if not content:
                print("⚠️ EnrichmentChecker: Empty LLM response")
                return []
            
            print(f"🧠 LLM raw response: {content[:200]}...")
            
            # Try to extract JSON from the response
            json_content = self._extract_json_from_response(content)
            if not json_content:
                print("⚠️ EnrichmentChecker: No JSON found in LLM response")
                return []
            
            # Parse JSON response
            try:
                json_response = json.loads(json_content)
                recommendations = json_response.get("recommended_methods", [])
                print(f"🧠 LLM recommended {len(recommendations)} methods for '{pathway_query}':")
                for i, rec in enumerate(recommendations, 1):
                    method = rec.get("Method", "Unknown")
                    pathway = rec.get("pathway", "Unknown")
                    gene_lib = rec.get("Gene library", "")
                    if gene_lib:
                        print(f"   {i}. {method} - {pathway} (Library: {gene_lib})")
                    else:
                        print(f"   {i}. {method} - {pathway}")
                        
                return recommendations
                
            except json.JSONDecodeError as e:
                print(f"⚠️ EnrichmentChecker: Failed to parse LLM JSON response: {e}")
                print(f"Cleaned JSON content: {json_content}")
                return []
                
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: LLM method recommendation failed: {e}")
            return []
    
    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response that might contain extra text"""
        # First, try to extract from code blocks
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        code_matches = re.findall(code_block_pattern, content, re.DOTALL)
        
        if code_matches:
            candidate = code_matches[0].strip()
            if "recommended_methods" in candidate:
                return candidate
        
        # Try to find JSON using bracket counting for proper nesting
        start_idx = content.find('{')
        if start_idx != -1:
            bracket_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_candidate = content[start_idx:i+1]
                            if "recommended_methods" in json_candidate:
                                return json_candidate
                            break
        
        # Last resort - check if the entire content is JSON
        content_cleaned = content.strip()
        if content_cleaned.startswith('{') and content_cleaned.endswith('}'):
            if "recommended_methods" in content_cleaned:
                return content_cleaned
        
        return ""
    
    def _validate_recommendations_in_neo4j(self, llm_recommendations: List[Dict[str, str]], 
                                         original_query: str) -> List[EnrichmentRecommendation]:
        """Validate LLM recommendations against Neo4j database"""
        validated_recommendations = []
        
        if not self.driver:
            print(f"⚠️ EnrichmentChecker: Cannot validate recommendations - no database connection")
            return []
        
        with self.driver.session(database=self.neo4j_database) as session:
            for rec in llm_recommendations:
                method = rec.get("Method", "").lower()
                pathway = rec.get("pathway", "")
                gene_library = rec.get("Gene library", "")
                
                print(f"🔍 EnrichmentChecker: Validating {method} - {pathway} (library: {gene_library})")
                
                if not pathway:
                    print(f"⚠️ EnrichmentChecker: Skipping {method} - no pathway specified")
                    continue
                
                try:
                    if method in ["go", "kegg", "reactome"]:
                        # Validate gprofiler databases
                        validation_result = self._validate_gprofiler_pathway(session, method, pathway)
                        if validation_result:
                            validated_recommendations.append(validation_result)
                            print(f"✅ EnrichmentChecker: Validated {method} - {pathway}")
                        else:
                            print(f"❌ EnrichmentChecker: Failed to validate {method} - {pathway}")
                    
                    elif method == "gsea":
                        # Validate GSEA pathway in specific gene library
                        print(f"🔍 EnrichmentChecker: Checking GSEA pathway '{pathway}' in library '{gene_library}'")
                        validation_result = self._validate_gsea_pathway(session, pathway, gene_library)
                        if validation_result:
                            validated_recommendations.append(validation_result)
                            print(f"✅ EnrichmentChecker: Validated GSEA - {pathway} in {gene_library}")
                        else:
                            print(f"❌ EnrichmentChecker: Failed to validate GSEA - {pathway} in {gene_library}")
                    
                except Exception as e:
                    print(f"⚠️ EnrichmentChecker: Validation failed for {method} - {pathway}: {e}")
                    continue
        
        print(f"✅ EnrichmentChecker: Validated {len(validated_recommendations)} out of {len(llm_recommendations)} LLM recommendations")
        return validated_recommendations
    
    def _validate_gprofiler_pathway(self, session, method: str, pathway: str) -> Optional[EnrichmentRecommendation]:
        """Validate pathway exists in GO/KEGG/Reactome database using fuzzy matching"""
        database_map = {"go": "GO", "kegg": "KEGG", "reactome": "Reactome"}
        database = database_map.get(method)
        
        if not database:
            return None
        
        print(f"🔍 {database} Validation: Searching for pathway '{pathway}'")
        
        # Use fuzzy matching to find the best pathway match
        best_match = self._find_best_gprofiler_match(session, pathway, database)
        
        if best_match:
            pathway_name, similarity_score = best_match
            print(f"✅ {database} Validation: Found pathway '{pathway_name}' (similarity: {similarity_score:.3f})")
            
            # Set confidence based on similarity score
            confidence = min(similarity_score, 1.0)
            description = f"Perform {database} enrichment analysis targeting '{pathway_name}' pathway."
            expected_outcome = f"{database} enrichment results for {pathway_name} pathway."
            reasoning = f"LLM recommended and validated '{pathway_name}' in {database} database (similarity: {similarity_score:.3f})"
            
            return EnrichmentRecommendation(
                analyses=[method],
                gene_set_library=None,
                description=description,
                expected_outcome=expected_outcome,
                confidence=confidence,
                reasoning=reasoning
            )
        else:
            print(f"❌ {database} Validation: No similar pathway found for '{pathway}'")
            return None
    
    def _find_best_gprofiler_match(self, session, target_pathway: str, database: str) -> Optional[Tuple[str, float]]:
        """Find the best matching pathway in GO/KEGG/Reactome using fuzzy matching"""
        
        # Get all pathways in the database
        cypher_query = """
        MATCH (p:Pathway)-[:FOUND_IN]->(d:Database {name: $database})
        RETURN p.name as pathway_name
        """
        
        try:
            result = session.run(cypher_query, {"database": database})
            all_pathways = [record["pathway_name"] for record in result]
            
            if not all_pathways:
                print(f"🔍 No pathways found in {database} database")
                return None
            
            print(f"🔍 Fuzzy matching against {len(all_pathways)} pathways in {database}")
            
            # Find the best match using multiple similarity metrics
            best_match = None
            best_score = 0.0
            
            for pathway_name in all_pathways:
                similarity_score = self._calculate_pathway_similarity(target_pathway, pathway_name)
                
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = pathway_name
            
            # Only return matches above a threshold
            if best_score >= 0.6:  # Require at least 60% similarity
                return (best_match, best_score)
            else:
                print(f"🔍 Best match '{best_match}' has low similarity ({best_score:.3f}), rejecting")
                return None
                
        except Exception as e:
            print(f"⚠️ Fuzzy matching in {database} failed: {e}")
            return None
    
    def _validate_gsea_pathway(self, session, pathway: str, gene_library: str) -> Optional[EnrichmentRecommendation]:
        """Validate pathway exists in specific GSEA gene library using fuzzy matching"""
        print(f"🔍 GSEA Validation: Searching for pathway '{pathway}' in library '{gene_library}'")
        
        # Use fuzzy matching to find the best pathway match
        best_match = self._find_best_pathway_match(session, pathway, gene_library)
        
        if best_match:
            pathway_name, similarity_score = best_match
            print(f"✅ GSEA Validation: Found pathway '{pathway_name}' (similarity: {similarity_score:.3f})")
            
            # Set confidence based on similarity score
            confidence = min(similarity_score, 1.0)
            description = f"Perform GSEA enrichment analysis targeting '{pathway_name}' pathway."
            expected_outcome = f"GSEA enrichment results for {pathway_name} pathway."
            reasoning = f"LLM recommended and validated '{pathway_name}' in {gene_library} library (similarity: {similarity_score:.3f})"
            
            return EnrichmentRecommendation(
                analyses=["gsea"],
                gene_set_library=gene_library,
                description=description,
                expected_outcome=expected_outcome,
                confidence=confidence,
                reasoning=reasoning
            )
        else:
            print(f"❌ GSEA Validation: No similar pathway found for '{pathway}' in library '{gene_library}'")
            return None
    
    def _create_pathway_variations(self, pathway: str) -> List[str]:
        """Create variations of pathway names to handle different formats"""
        variations = [pathway]  # Original pathway name
        
        # Convert underscores to spaces and vice versa
        if "_" in pathway:
            variations.append(pathway.replace("_", " "))
        if " " in pathway:
            variations.append(pathway.replace(" ", "_"))
        
        # Handle HALLMARK prefix variations
        if pathway.startswith("HALLMARK_"):
            # Remove HALLMARK_ prefix
            variations.append(pathway[9:])
            # Remove HALLMARK_ and convert to spaces
            variations.append(pathway[9:].replace("_", " "))
        elif pathway.startswith("HALLMARK "):
            # Remove HALLMARK prefix
            variations.append(pathway[9:])
            # Remove HALLMARK and convert to underscores
            variations.append(pathway[9:].replace(" ", "_"))
        else:
            # Add HALLMARK prefix variations
            variations.append(f"HALLMARK_{pathway}")
            variations.append(f"HALLMARK {pathway}")
        
        # Case variations
        original_variations = variations.copy()
        for var in original_variations:
            variations.append(var.upper())
            variations.append(var.lower())
            variations.append(var.title())
        
        # Remove duplicates while preserving order
        unique_variations = []
        seen = set()
        for var in variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations
    
    def _find_best_pathway_match(self, session, target_pathway: str, gene_library: str) -> Optional[Tuple[str, float]]:
        """Find the best matching pathway using fuzzy string matching"""
        
        # First, get all pathways in the library
        cypher_query = """
        MATCH (d:Database {name: "Gsea"})-[:CONTAINS_LIBRARY]->(lib:GeneSetLibrary {name: $gene_library})
        -[:CONTAINS_PATHWAY]->(p:Pathway)
        RETURN p.name as pathway_name
        """
        
        try:
            result = session.run(cypher_query, {"gene_library": gene_library})
            all_pathways = [record["pathway_name"] for record in result]
            
            if not all_pathways:
                print(f"🔍 No pathways found in library '{gene_library}'")
                return None
            
            print(f"🔍 Fuzzy matching against {len(all_pathways)} pathways in {gene_library}")
            
            # Find the best match using multiple similarity metrics
            best_match = None
            best_score = 0.0
            
            for pathway_name in all_pathways:
                similarity_score = self._calculate_pathway_similarity(target_pathway, pathway_name)
                
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match = pathway_name
            
            # Only return matches above a threshold
            if best_score >= 0.6:  # Require at least 60% similarity
                return (best_match, best_score)
            else:
                print(f"🔍 Best match '{best_match}' has low similarity ({best_score:.3f}), rejecting")
                return None
                
        except Exception as e:
            print(f"⚠️ Fuzzy matching failed: {e}")
            return None
    
    def _calculate_pathway_similarity(self, target: str, candidate: str) -> float:
        """Calculate similarity between two pathway names using multiple metrics"""
        
        # Normalize both strings
        target_norm = self._normalize_pathway_name(target)
        candidate_norm = self._normalize_pathway_name(candidate)
        
        # Calculate multiple similarity metrics
        scores = []
        
        # 1. Exact match (highest weight)
        if target_norm == candidate_norm:
            return 1.0
        
        # 2. Containment score
        if target_norm in candidate_norm or candidate_norm in target_norm:
            scores.append(0.9)
        
        # 3. Token overlap score
        target_tokens = set(target_norm.split())
        candidate_tokens = set(candidate_norm.split())
        
        if target_tokens and candidate_tokens:
            overlap = len(target_tokens & candidate_tokens)
            union = len(target_tokens | candidate_tokens)
            jaccard_score = overlap / union if union > 0 else 0.0
            scores.append(jaccard_score * 0.8)
        
        # 4. Levenshtein distance (if strings are reasonably similar length)
        if abs(len(target_norm) - len(candidate_norm)) < max(len(target_norm), len(candidate_norm)) * 0.5:
            levenshtein_score = self._levenshtein_similarity(target_norm, candidate_norm)
            scores.append(levenshtein_score * 0.7)
        
        # 5. Key biological term matching
        bio_score = self._calculate_biological_term_similarity(target_norm, candidate_norm)
        if bio_score > 0:
            scores.append(bio_score * 0.8)
        
        # Return the best score
        return max(scores) if scores else 0.0
    
    def _normalize_pathway_name(self, pathway: str) -> str:
        """Normalize pathway name for comparison"""
        # Convert to lowercase
        normalized = pathway.lower()
        
        # Replace common separators with spaces
        normalized = re.sub(r'[_\-\.]', ' ', normalized)
        
        # Remove common prefixes
        prefixes = ['hallmark', 'kegg', 'go', 'reactome', 'wikipathways']
        for prefix in prefixes:
            if normalized.startswith(prefix + ' '):
                normalized = normalized[len(prefix) + 1:]
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein similarity"""
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        # Create matrix
        matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        
        # Initialize first row and column
        for i in range(len(s1) + 1):
            matrix[i][0] = i
        for j in range(len(s2) + 1):
            matrix[0][j] = j
        
        # Fill matrix
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i-1] == s2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = 1 + min(
                        matrix[i-1][j],    # deletion
                        matrix[i][j-1],    # insertion
                        matrix[i-1][j-1]   # substitution
                    )
        
        # Calculate similarity (1 - normalized distance)
        distance = matrix[len(s1)][len(s2)]
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    def _calculate_biological_term_similarity(self, target: str, candidate: str) -> float:
        """Calculate similarity based on biological terms"""
        # Common biological terms that indicate similar pathways
        biological_terms = {
            'interferon': ['interferon', 'ifn', 'antiviral', 'immune'],
            'apoptosis': ['apoptosis', 'cell death', 'programmed death'],
            'cell cycle': ['cell cycle', 'mitosis', 'g1', 'g2', 'm phase', 's phase'],
            'immune': ['immune', 'immunity', 'inflammatory', 'cytokine'],
            'metabolism': ['metabolism', 'metabolic', 'biosynthesis'],
            'signaling': ['signaling', 'pathway', 'cascade', 'transduction'],
            'response': ['response', 'activation', 'regulation']
        }
        
        target_terms = set(target.split())
        candidate_terms = set(candidate.split())
        
        # Check for biological term matches
        for term_group in biological_terms.values():
            target_matches = sum(1 for term in term_group if any(t in target for t in term_group))
            candidate_matches = sum(1 for term in term_group if any(t in candidate for t in term_group))
            
            if target_matches > 0 and candidate_matches > 0:
                return min(target_matches, candidate_matches) / max(target_matches, candidate_matches)
        
        return 0.0
    
    def _select_best_recommendation(self, recommendations: List[EnrichmentRecommendation]) -> EnrichmentRecommendation:
        """Select the best recommendation, prioritizing GSEA when available"""
        if not recommendations:
            return None
        
        # Separate GSEA and non-GSEA recommendations
        gsea_recommendations = [rec for rec in recommendations if "gsea" in rec.analyses]
        other_recommendations = [rec for rec in recommendations if "gsea" not in rec.analyses]
        
        print(f"🎯 Recommendation selection: {len(gsea_recommendations)} GSEA, {len(other_recommendations)} other")
        
        # Prioritize GSEA recommendations (especially MSigDB_Hallmark_2020)
        if gsea_recommendations:
            # Further prioritize MSigDB_Hallmark_2020
            hallmark_recommendations = [rec for rec in gsea_recommendations 
                                      if rec.gene_set_library == "MSigDB_Hallmark_2020"]
            
            if hallmark_recommendations:
                best_rec = max(hallmark_recommendations, key=lambda x: x.confidence)
                print(f"✅ Selected MSigDB_Hallmark_2020 recommendation: {best_rec.reasoning}")
                return best_rec
            else:
                best_rec = max(gsea_recommendations, key=lambda x: x.confidence)
                print(f"✅ Selected GSEA recommendation: {best_rec.reasoning}")
                return best_rec
        
        # If no GSEA recommendations, use highest confidence from others
        if other_recommendations:
            best_rec = max(other_recommendations, key=lambda x: x.confidence)
            print(f"✅ Selected non-GSEA recommendation: {best_rec.reasoning}")
            return best_rec
        
        # Fallback to first recommendation
        return recommendations[0]
    
    def _select_high_confidence_recommendations(self, recommendations: List[EnrichmentRecommendation], 
                                               confidence_threshold: float = 0.8) -> List[EnrichmentRecommendation]:
        """Select all recommendations above confidence threshold, with intelligent prioritization"""
        if not recommendations:
            return []
        
        # Filter by confidence threshold
        high_confidence_recs = [rec for rec in recommendations if rec.confidence >= confidence_threshold]
        
        if not high_confidence_recs:
            # If no high-confidence recommendations, fall back to single best
            best_rec = self._select_best_recommendation(recommendations)
            return [best_rec] if best_rec else []
        
        print(f"🎯 Multi-analysis selection: {len(high_confidence_recs)} high-confidence recommendations (≥{confidence_threshold})")
        
        # Group by analysis type for intelligent selection
        gprofiler_recs = [rec for rec in high_confidence_recs if rec.analyses[0] in ["go", "kegg", "reactome"]]
        gsea_recs = [rec for rec in high_confidence_recs if "gsea" in rec.analyses]
        
        selected_recs = []
        
        # Add the best from each gprofiler database
        for analysis_type in ["go", "kegg", "reactome"]:
            type_recs = [rec for rec in gprofiler_recs if rec.analyses[0] == analysis_type]
            if type_recs:
                best_of_type = max(type_recs, key=lambda x: x.confidence)
                selected_recs.append(best_of_type)
                print(f"✅ Selected {analysis_type.upper()}: {best_of_type.reasoning}")
        
        # Add the best GSEA recommendation (prioritize MSigDB_Hallmark_2020)
        if gsea_recs:
            hallmark_recs = [rec for rec in gsea_recs if rec.gene_set_library == "MSigDB_Hallmark_2020"]
            if hallmark_recs:
                best_gsea = max(hallmark_recs, key=lambda x: x.confidence)
            else:
                best_gsea = max(gsea_recs, key=lambda x: x.confidence)
            selected_recs.append(best_gsea)
            print(f"✅ Selected GSEA: {best_gsea.reasoning}")
        
        # Sort by confidence for consistent ordering
        selected_recs.sort(key=lambda x: x.confidence, reverse=True)
        
        print(f"🚀 Final selection: {len(selected_recs)} analyses will be performed")
        return selected_recs
    
    def _build_enhanced_step_from_recommendations(self, plan_step: Dict[str, Any], 
                                                cell_type: str, pathway_query: str,
                                                recommendations: List[EnrichmentRecommendation]) -> Dict[str, Any]:
        """Build enhanced plan step from multiple pathway recommendations"""
        if not recommendations:
            return plan_step
        
        enhanced_step = plan_step.copy()
        
        # Collect all analyses and determine gene_set_library
        all_analyses = []
        gene_set_library = None
        pathway_descriptions = []
        
        for rec in recommendations:
            all_analyses.extend(rec.analyses)
            if rec.gene_set_library:
                gene_set_library = rec.gene_set_library
            
            # Extract pathway name from reasoning for description
            if "validated" in rec.reasoning:
                pathway_name = rec.reasoning.split("'")[1] if "'" in rec.reasoning else "pathway"
                pathway_descriptions.append(f"{rec.analyses[0].upper()}: {pathway_name}")
        
        # Update parameters
        enhanced_step["parameters"].update({
            "analyses": all_analyses
        })
        
        if gene_set_library:
            enhanced_step["parameters"]["gene_set_library"] = gene_set_library
        
        # Remove pathway_include after processing to avoid filtering errors
        enhanced_step["parameters"].pop("pathway_include", None)
        
        # Create comprehensive description
        if len(recommendations) == 1:
            # Single analysis
            rec = recommendations[0]
            description = rec.description.replace("pathways.", f"pathways in {cell_type}.")
            expected_outcome = rec.expected_outcome.replace("pathways.", f"pathways for {cell_type}.")
        else:
            # Multiple analyses
            analysis_list = ", ".join([a.upper() for a in all_analyses])
            description = f"Perform comprehensive enrichment analysis ({analysis_list}) targeting '{pathway_query}' pathways in {cell_type}."
            expected_outcome = f"Comprehensive enrichment results from {analysis_list} for '{pathway_query}' pathways in {cell_type}."
        
        enhanced_step.update({
            "description": description,
            "expected_outcome": expected_outcome
        })
        
        # Log detailed reasoning for debugging (don't add to step to avoid ExecutionStep errors)
        reasoning_details = []
        for rec in recommendations:
            reasoning_details.append(f"- {rec.reasoning}")
        
        print(f"🔍 EnrichmentChecker: Enhanced step with {len(recommendations)} recommendations:")
        print(f"   • Analyses selected: {all_analyses}")
        print(f"   • Pathway matches: {pathway_descriptions}")
        for detail in reasoning_details:
            print(f"   {detail}")
        
        return enhanced_step
    
    def _query_neo4j_database_safe(self, pathway_query: str, top_k: int) -> List[PathwayMatch]:
        """Safely query Neo4j database with connection error handling"""
        try:
            return self._query_neo4j_database(pathway_query, top_k)
        except Exception as e:
            # Handle specific Neo4j errors
            error_type = type(e).__name__
            if "ServiceUnavailable" in error_type:
                print(f"⚠️ EnrichmentChecker: Neo4j service unavailable - {e}")
            elif "AuthError" in error_type:
                print(f"⚠️ EnrichmentChecker: Authentication failed - {e}")
            elif "DatabaseError" in error_type:
                print(f"⚠️ EnrichmentChecker: Database error - {e}")
            else:
                print(f"⚠️ EnrichmentChecker: Unexpected database error ({error_type}) - {e}")
            
            # Update connection status
            self.connection_status = f"error: {error_type}"
            return []
    
    def _clean_pathway_query(self, pathway_query: str) -> str:
        """Use GPT-4 to extract clean pathway terms from user query"""
        if not OPENAI_AVAILABLE:
            print("⚠️ EnrichmentChecker: OpenAI not available, using original query")
            return pathway_query
            
        try:
            prompt = f"""
            Extract the core biological pathway or process name from this user query: "{pathway_query}"
            
            Rules:
            - Extract only the biological pathway/process terms
            - Remove cell type mentions
            - Remove action words (find, analyze, etc.)
            - Keep pathway-specific terms (e.g., "IFN-stimulated", "interferon response", "cell cycle")
            - Return only the clean pathway term(s)
            
            Examples:
            - "find IFN-stimulated pathway from T cell" → "IFN-stimulated"
            - "interferon response in B cells" → "interferon response"
            - "cell cycle pathways analysis" → "cell cycle"
            - "apoptosis signaling" → "apoptosis signaling"
            
            Query: "{pathway_query}"
            Clean pathway term:
            """
            
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Create messages in LangChain format
            messages = [
                SystemMessage(content="You are an expert at cleaning pathway query terms."),
                HumanMessage(content=prompt)
            ]
            
            # Initialize model
            model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=200
            )
            
            # Get response
            response = model.invoke(messages)
            cleaned = response.content.strip()
            print(f"✅ EnrichmentChecker: Cleaned '{pathway_query}' → '{cleaned}'")
            return cleaned
            
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: Query cleaning failed: {e}")
            return pathway_query
    
    def _query_neo4j_database(self, pathway_query: str, top_k: int) -> List[PathwayMatch]:
        """Query Neo4j database for pathway matches using enhanced fuzzy matching"""
        with self.driver.session(database=self.neo4j_database) as session:
            
            # First try exact and direct matches
            exact_matches = self._query_exact_matches(session, pathway_query, top_k)
            if exact_matches:
                print(f"✅ EnrichmentChecker: Found {len(exact_matches)} exact matches for '{pathway_query}'")
                return exact_matches
            
            # If no exact matches, try fuzzy matching with multiple strategies
            fuzzy_matches = self._query_fuzzy_matches(session, pathway_query, top_k)
            if fuzzy_matches:
                print(f"✅ EnrichmentChecker: Found {len(fuzzy_matches)} fuzzy matches for '{pathway_query}'")
                return fuzzy_matches
            
            print(f"⚠️ EnrichmentChecker: No matches found for '{pathway_query}'")
            return []
    
    def _query_exact_matches(self, session, pathway_query: str, top_k: int) -> List[PathwayMatch]:
        """Query for exact and high-confidence matches"""
        cypher_query = """
        MATCH (p:Pathway)
        WHERE toLower(p.name) = toLower($query)
           OR toLower(p.name) CONTAINS toLower($query)
           OR any(keyword IN p.keywords WHERE toLower(keyword) = toLower($query))
        
        MATCH (p)-[:FOUND_IN]->(d:Database)
        OPTIONAL MATCH (lib:GeneSetLibrary)-[:CONTAINS_PATHWAY]->(p)
        MATCH (m:Method)-[:USES]->(d)
        
        WITH p, d, lib, m,
             CASE 
               WHEN toLower(p.name) = toLower($query) THEN 1.0
               WHEN toLower(p.name) CONTAINS toLower($query) THEN 0.9
               WHEN any(keyword IN p.keywords WHERE toLower(keyword) = toLower($query)) THEN 0.8
               ELSE 0.0
             END as confidence
        
        WHERE confidence >= 0.8
        ORDER BY confidence DESC, p.name
        LIMIT $top_k
        
        RETURN p.name as pathway_name,
               p.description as pathway_description,
               d.name as database,
               lib.name as gene_set_library,
               m.name as method,
               confidence
        """
        
        result = session.run(cypher_query, {"query": pathway_query, "top_k": top_k})
        return self._build_pathway_matches(result)
    
    def _query_fuzzy_matches(self, session, pathway_query: str, top_k: int) -> List[PathwayMatch]:
        """Query for fuzzy matches using multiple strategies"""
        
        # Strategy 1: Keyword-based fuzzy matching
        keywords = self._extract_query_keywords(pathway_query)
        print(f"🔍 Fuzzy matching keywords: {keywords}")
        keyword_matches = self._query_keyword_matches(session, keywords, top_k)
        print(f"  - Keyword matches: {len(keyword_matches)} found")
        
        # Strategy 2: Partial string matching
        partial_matches = self._query_partial_matches(session, pathway_query, top_k)
        print(f"  - Partial matches: {len(partial_matches)} found")
        
        # Strategy 3: LLM-based synonym matching (if OpenAI available)
        synonym_matches = self._query_llm_synonym_matches(session, pathway_query, top_k) if OPENAI_AVAILABLE else []
        print(f"  - LLM synonym matches: {len(synonym_matches)} found")
        
        # Combine and deduplicate matches
        all_matches = keyword_matches + partial_matches + synonym_matches
        unique_matches = self._deduplicate_matches(all_matches)
        
        # Sort by confidence and return top results
        unique_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        # Debug: Show top matches
        print(f"🎯 Top {min(3, len(unique_matches))} fuzzy matches:")
        for i, match in enumerate(unique_matches[:3]):
            print(f"   {i+1}. {match.pathway_name} ({match.database}) - confidence: {match.confidence:.2f}")
        
        return unique_matches[:top_k]
    
    def _extract_query_keywords(self, pathway_query: str) -> List[str]:
        """Extract meaningful keywords from pathway query"""
        # Remove common words and extract biological terms
        stop_words = {'pathway', 'pathways', 'analysis', 'enrichment', 'genes', 'response', 'from', 'the', 'and', 'for', 'with'}
        words = re.findall(r'\b[a-zA-Z]+\b', pathway_query.lower())
        
        # Keep hyphenated terms together (e.g., "ifn-stimulated")
        hyphenated = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+\b', pathway_query.lower())
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        keywords.extend(hyphenated)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)
        
        return unique_keywords
    
    def _query_keyword_matches(self, session, keywords: List[str], top_k: int) -> List[PathwayMatch]:
        """Query pathways matching individual keywords"""
        if not keywords:
            return []
        
        cypher_query = """
        MATCH (p:Pathway)
        WHERE any(keyword IN $keywords WHERE 
            toLower(p.name) CONTAINS keyword
            OR any(pk IN p.keywords WHERE toLower(pk) CONTAINS keyword)
            OR toLower(p.description) CONTAINS keyword
        )
        
        MATCH (p)-[:FOUND_IN]->(d:Database)
        OPTIONAL MATCH (lib:GeneSetLibrary)-[:CONTAINS_PATHWAY]->(p)
        MATCH (m:Method)-[:USES]->(d)
        
        WITH p, d, lib, m, $keywords as query_keywords,
             [keyword IN $keywords WHERE 
                toLower(p.name) CONTAINS keyword
                OR any(pk IN p.keywords WHERE toLower(pk) CONTAINS keyword)
                OR toLower(p.description) CONTAINS keyword
             ] as matched_keywords
        
        WITH p, d, lib, m,
             (size(matched_keywords) * 1.0 / size(query_keywords)) * 0.6 as confidence
        
        WHERE confidence > 0.2
        ORDER BY confidence DESC, p.name
        LIMIT $top_k
        
        RETURN p.name as pathway_name,
               p.description as pathway_description,
               d.name as database,
               lib.name as gene_set_library,
               m.name as method,
               confidence
        """
        
        result = session.run(cypher_query, {"keywords": keywords, "top_k": top_k})
        return self._build_pathway_matches(result)
    
    def _query_partial_matches(self, session, pathway_query: str, top_k: int) -> List[PathwayMatch]:
        """Query pathways with partial string matching"""
        # Break query into parts for partial matching
        query_parts = pathway_query.split()
        if len(query_parts) < 2:
            return []
        
        cypher_query = """
        MATCH (p:Pathway)
        WHERE any(part IN $query_parts WHERE 
            toLower(p.name) CONTAINS toLower(part)
            OR any(keyword IN p.keywords WHERE toLower(keyword) CONTAINS toLower(part))
        )
        
        MATCH (p)-[:FOUND_IN]->(d:Database)
        OPTIONAL MATCH (lib:GeneSetLibrary)-[:CONTAINS_PATHWAY]->(p)
        MATCH (m:Method)-[:USES]->(d)
        
        WITH p, d, lib, m, $query_parts as parts,
             [part IN $query_parts WHERE 
                toLower(p.name) CONTAINS toLower(part)
                OR any(keyword IN p.keywords WHERE toLower(keyword) CONTAINS toLower(part))
             ] as matched_parts
        
        WITH p, d, lib, m,
             (size(matched_parts) * 1.0 / size(parts)) * 0.5 as confidence
        
        WHERE confidence > 0.3
        ORDER BY confidence DESC, p.name
        LIMIT $top_k
        
        RETURN p.name as pathway_name,
               p.description as pathway_description,
               d.name as database,
               lib.name as gene_set_library,
               m.name as method,
               confidence
        """
        
        result = session.run(cypher_query, {"query_parts": query_parts, "top_k": top_k})
        return self._build_pathway_matches(result)
    
    def _query_llm_synonym_matches(self, session, pathway_query: str, top_k: int) -> List[PathwayMatch]:
        """Query pathways using LLM-generated biological synonyms"""
        try:
            # Use GPT-4 to generate biological synonyms
            synonyms = self._generate_pathway_synonyms(pathway_query)
            
            if not synonyms:
                return []
            
            cypher_query = """
            MATCH (p:Pathway)
            WHERE any(synonym IN $synonyms WHERE 
                toLower(p.name) CONTAINS synonym
                OR any(keyword IN p.keywords WHERE toLower(keyword) CONTAINS synonym)
            )
            
            MATCH (p)-[:FOUND_IN]->(d:Database)
            OPTIONAL MATCH (lib:GeneSetLibrary)-[:CONTAINS_PATHWAY]->(p)
            MATCH (m:Method)-[:USES]->(d)
            
            WITH p, d, lib, m, 0.4 as confidence
            
            ORDER BY p.name
            LIMIT $top_k
            
            RETURN p.name as pathway_name,
                   p.description as pathway_description,
                   d.name as database,
                   lib.name as gene_set_library,
                   m.name as method,
                   confidence
            """
            
            result = session.run(cypher_query, {"synonyms": synonyms, "top_k": top_k})
            return self._build_pathway_matches(result)
            
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: LLM synonym matching failed: {e}")
            return []
    
    def _generate_pathway_synonyms(self, pathway_query: str) -> List[str]:
        """Generate biological synonyms using GPT-4"""
        try:
            prompt = f"""
            Generate biological synonyms and related terms for this pathway query: "{pathway_query}"
            
            Instructions:
            - Focus on biological pathway terminology
            - Include abbreviations and full forms
            - Include related biological processes
            - Return only the synonyms, one per line
            - Maximum 5 synonyms
            
            Examples:
            Query: "interferon response"
            Synonyms:
            - IFN response
            - interferon signaling
            - type I interferon
            - antiviral response
            
            Query: "{pathway_query}"
            Synonyms:
            """
            
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Create messages in LangChain format
            messages = [
                SystemMessage(content="You are an expert in biological pathway terminology. Generate biological synonyms."),
                HumanMessage(content=prompt)
            ]
            
            # Initialize model
            model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                max_tokens=100
            )
            
            # Get response
            response = model.invoke(messages)
            content = response.content.strip()
            synonyms = [line.strip().lstrip('- ') for line in content.split('\n') if line.strip() and not line.strip().startswith('Query:')]
            
            print(f"🔍 Generated synonyms for '{pathway_query}': {synonyms}")
            return synonyms
            
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: Synonym generation failed: {e}")
            return []
    
    def _build_pathway_matches(self, result) -> List[PathwayMatch]:
        """Build PathwayMatch objects from Neo4j result"""
        matches = []
        for record in result:
            match = PathwayMatch(
                pathway_name=record["pathway_name"],
                database=record["database"],
                method=record["method"],
                gene_set_library=record["gene_set_library"],
                confidence=record["confidence"],
                description=record["pathway_description"] or ""
            )
            matches.append(match)
        return matches
    
    def _deduplicate_matches(self, matches: List[PathwayMatch]) -> List[PathwayMatch]:
        """Remove duplicate pathway matches, keeping highest confidence"""
        seen = {}
        for match in matches:
            key = (match.pathway_name, match.database, match.gene_set_library)
            if key not in seen or match.confidence > seen[key].confidence:
                seen[key] = match
        return list(seen.values())
    
    def _convert_matches_to_recommendations(self, matches: List[PathwayMatch], 
                                          original_query: str) -> List[EnrichmentRecommendation]:
        """Convert pathway matches to enrichment recommendations with support for multiple analyses"""
        if not matches:
            return []
        
        # Group matches by database/method for multi-analysis support
        database_groups = {}
        gsea_libraries = []
        
        for match in matches:
            if match.method == "gprofiler":
                database_groups[match.database.lower()] = match
            elif match.method == "gseapy":
                gsea_libraries.append(match)
        
        # Create recommendations based on available databases
        recommendations = []
        
        # If we have multiple databases, create a combined recommendation
        if len(database_groups) > 1:
            # Multi-database analysis
            analyses = list(database_groups.keys())
            best_matches = list(database_groups.values())
            avg_confidence = sum(m.confidence for m in best_matches) / len(best_matches)
            
            pathway_names = [m.pathway_name for m in best_matches]
            reasoning = f"Found pathways in multiple databases: {', '.join(database_groups.keys()).upper()}"
            
            description = f"Perform multi-database enrichment analysis ({', '.join(analyses).upper()}) targeting '{original_query}' pathways."
            expected_outcome = f"Comprehensive enrichment results from {', '.join(analyses).upper()} databases for {original_query} pathways."
            
            recommendation = EnrichmentRecommendation(
                analyses=analyses,
                gene_set_library=None,
                description=description,
                expected_outcome=expected_outcome,
                confidence=avg_confidence,
                reasoning=reasoning
            )
            recommendations.append(recommendation)
        
        # Single database recommendation
        elif len(database_groups) == 1:
            database, match = next(iter(database_groups.items()))
            analyses = [database]
            
            description = f"Perform {database.upper()} enrichment analysis targeting '{match.pathway_name}' pathways."
            expected_outcome = f"{database.upper()} results focusing on {original_query} pathways."
            reasoning = f"Found '{match.pathway_name}' in {database.upper()} database"
            
            recommendation = EnrichmentRecommendation(
                analyses=analyses,
                gene_set_library=None,
                description=description,
                expected_outcome=expected_outcome,
                confidence=match.confidence,
                reasoning=reasoning
            )
            recommendations.append(recommendation)
        
        # GSEA recommendations (can be in addition to or instead of database recommendations)
        if gsea_libraries:
            # Use the best GSEA library match
            best_gsea = max(gsea_libraries, key=lambda x: x.confidence)
            analyses = ["gsea"]
            
            description = f"Perform GSEA enrichment analysis targeting '{best_gsea.pathway_name}' pathways."
            expected_outcome = f"GSEA results focusing on {original_query} pathways."
            reasoning = f"Found '{best_gsea.pathway_name}' in {best_gsea.gene_set_library} library"
            
            recommendation = EnrichmentRecommendation(
                analyses=analyses,
                gene_set_library=best_gsea.gene_set_library,
                description=description,
                expected_outcome=expected_outcome,
                confidence=best_gsea.confidence,
                reasoning=reasoning
            )
            recommendations.append(recommendation)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations
    
    def _build_enhanced_step_from_recommendation(self, plan_step: Dict[str, Any], 
                                               cell_type: str, pathway_query: str,
                                               recommendation: EnrichmentRecommendation) -> Dict[str, Any]:
        """Build enhanced plan step from pathway recommendation"""
        enhanced_step = plan_step.copy()
        
        # Update parameters
        enhanced_step["parameters"].update({
            "analyses": recommendation.analyses
        })
        
        if recommendation.gene_set_library:
            enhanced_step["parameters"]["gene_set_library"] = recommendation.gene_set_library
        
        # Update description and expected outcome
        enhanced_step.update({
            "description": recommendation.description.replace("pathways.", f"pathways in {cell_type}."),
            "expected_outcome": recommendation.expected_outcome.replace("pathways.", f"pathways for {cell_type}.")
        })
        
        print(f"✅ EnrichmentChecker: {recommendation.reasoning} (confidence: {recommendation.confidence:.2f})")
        return enhanced_step
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("✅ EnrichmentChecker: Neo4j connection closed")


# Convenience function for testing
def test_enrichment_checker():
    """Test function for enrichment checker functionality"""
    print("🧪 Testing EnrichmentChecker...")
    checker = EnrichmentChecker()
    
    print(f"Connection Status: {checker.connection_status}")
    
    test_cases = [
        # Pathway semantic queries
        {
            "name": "Pathway Semantic - IFN",
            "parameters": {
                "cell_type": "T cell",
                "pathway_include": "IFN-stimulated pathway"
            }
        },
        {
            "name": "Pathway Semantic - Interferon",
            "parameters": {
                "cell_type": "B cell", 
                "pathway_include": "interferon response"
            }
        },
        # Explicit analysis queries
        {
            "name": "Explicit Analysis - GO",
            "parameters": {
                "cell_type": "T cell",
                "analyses": ["go"],
                "pathway_include": None
            }
        },
        # Default query
        {
            "name": "Default Analysis",
            "parameters": {
                "cell_type": "T cell"
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"Parameters: {test_case['parameters']}")
        print('='*60)
        
        try:
            enhanced = checker.enhance_enrichment_plan(test_case)
            print("✅ Enhanced plan:")
            print(json.dumps(enhanced, indent=2))
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    checker.close()
    print("🧪 Testing completed!")


def test_pathway_intelligence_integration():
    """Test complete pathway intelligence integration"""
    print("🧪 Testing Complete Pathway Intelligence Integration...")
    
    # Test the enrichment wrapper directly
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from enrichment import enrichment_wrapper
        
        print("✅ enrichment_wrapper imported successfully")
        
        # Mock adata for testing (in real usage, this would be actual data)
        print("⚠️ Note: This test requires actual AnnData object for full testing")
        print("✅ Integration test setup complete")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")


def run_all_tests():
    """Run all pathway intelligence tests"""
    print("🚀 Running All Pathway Intelligence Tests")
    print("="*70)
    
    # Test 1: EnrichmentChecker
    test_enrichment_checker()
    
    print("\n" + "="*70)
    
    # Test 2: Integration
    test_pathway_intelligence_integration()
    
    print("\n" + "="*70)
    print("🎉 All tests completed!")


if __name__ == "__main__":
    run_all_tests()