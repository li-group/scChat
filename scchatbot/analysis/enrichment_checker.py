import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

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
    
    def __init__(self, confidence_threshold=0.8, config_path=None):
        """Initialize enrichment checker with Neo4j connection and vector search."""
        # Load configuration from specification_graph.json
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels: analysis/ -> scchatbot/ -> project_root/
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, "media", "specification_graph.json")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required configuration fields
            required_fields = ["url", "username", "password", "pathway_rag"]
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                raise ValueError(f"Missing required configuration fields: {missing_fields}")
            
            self.neo4j_uri = config["url"]
            self.neo4j_user = config["username"]
            self.neo4j_password = config["password"]
            self.neo4j_database = config["pathway_rag"]
            print(f"✅ EnrichmentChecker: Loaded config from {config_path}")
        except Exception as e:
            print(f"❌ EnrichmentChecker: Configuration error: {e}")
            raise Exception(f"Failed to load Neo4j configuration from {config_path}: {e}")
        
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
                    self.neo4j_uri, 
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                # Test connection
                with self.driver.session(database=self.neo4j_database) as session:
                    result = session.run("MATCH (p:Pathway) RETURN count(p) as count LIMIT 1")
                    count = result.single()
                    print(f"   Neo4j test: Found {count['count'] if count else 0} pathways in '{self.neo4j_database}' database")
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
            current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/scChat_v2/scchatbot/analysis
            # Go up two levels: analysis/ -> scchatbot/ -> project_root/
            project_root = os.path.dirname(os.path.dirname(current_dir))  # /path/to/scChat_v2  
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
                print(f"⚠️ EnrichmentChecker: No pathway matches for '{pathway_query}', using default GO")
                return self._enhance_default_plan(plan_step, cell_type)
                
        except Exception as e:
            print(f"⚠️ EnrichmentChecker: Pathway intelligence failed: {e}")
            return self._enhance_default_plan(plan_step, cell_type)
    
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
    
    def _select_high_confidence_recommendations(self, recommendations: List[EnrichmentRecommendation], 
                                               confidence_threshold: float = 0.8) -> List[EnrichmentRecommendation]:
        """Select all recommendations above confidence threshold, with intelligent prioritization"""
        if not recommendations:
            return []
        
        # Filter by confidence threshold
        high_confidence_recs = [rec for rec in recommendations if rec.confidence >= confidence_threshold]
        
        if not high_confidence_recs:
            # If no high-confidence recommendations, fall back to highest confidence single recommendation
            best_rec = max(recommendations, key=lambda x: x.confidence) if recommendations else None
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
    
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("✅ EnrichmentChecker: Neo4j connection closed")