#!/usr/bin/env python3
"""
Create pathway embeddings that match Neo4j database exactly.
This version ensures the embedded pathway names match what's in Neo4j.
"""

import os
import time
import pickle
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict

def fetch_pathways_from_neo4j(driver) -> List[Dict]:
    """
    Fetch all pathways from Neo4j with their exact names and metadata.
    This ensures embeddings match Neo4j exactly.
    """
    query = """
    MATCH (p:Pathway)-[:FOUND_IN]->(d:Database)
    OPTIONAL MATCH (lib:GeneSetLibrary)-[:CONTAINS_PATHWAY]->(p)
    RETURN DISTINCT
        p.name as name,
        p.description as description,
        p.keywords as keywords,
        d.name as database,
        lib.name as gene_set_library,
        p.id as pathway_id
    ORDER BY p.name
    """
    
    pathways = []
    with driver.session(database="pathways") as session:
        result = session.run(query)
        for record in result:
            pathways.append({
                'name': record['name'],  # Exact name from Neo4j
                'description': record['description'] or '',
                'keywords': record['keywords'] or [],
                'database': record['database'],
                'gene_set_library': record['gene_set_library'],
                'id': record['pathway_id']
            })
    
    print(f"✅ Fetched {len(pathways)} pathways from Neo4j")
    
    # Show sample to verify names
    print("\nSample pathway names from Neo4j:")
    for i, p in enumerate(pathways[:5]):
        print(f"  {i+1}. '{p['name']}' (database: {p['database']})")
    
    # Check for "cell cycle" specifically
    cell_cycle_pathways = [p for p in pathways if 'cell cycle' in p['name'].lower()]
    print(f"\nFound {len(cell_cycle_pathways)} pathways containing 'cell cycle':")
    for p in cell_cycle_pathways[:5]:
        print(f"  - '{p['name']}' (database: {p['database']})")
    
    return pathways

def create_embeddings_from_neo4j():
    """Create embeddings that exactly match Neo4j pathway names."""
    
    # Neo4j connection
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "37754262")
    )
    
    try:
        # Fetch pathways
        print("Fetching pathways from Neo4j...")
        pathways = fetch_pathways_from_neo4j(driver)
        
        if not pathways:
            print("❌ No pathways found in Neo4j!")
            return
        
        # Initialize embedding model
        print("\nInitializing sentence transformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings using exact pathway names
        print(f"\nCreating embeddings for {len(pathways)} pathways...")
        texts = [p['name'] for p in pathways]  # Use exact names from Neo4j
        
        embeddings = model.encode(
            texts, 
            batch_size=128,
            show_progress_bar=True
        )
        
        # Build FAISS index
        print("\nBuilding FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        
        # Save to media directory
        media_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media")
        os.makedirs(media_dir, exist_ok=True)
        
        index_path = os.path.join(media_dir, "pathway_index.faiss")
        data_path = os.path.join(media_dir, "pathway_data.pkl")
        
        # Save index
        faiss.write_index(index, index_path)
        
        # Save pathway data
        with open(data_path, 'wb') as f:
            pickle.dump(pathways, f)
        
        print(f"\n✅ Successfully created embeddings for {len(pathways)} pathways")
        print(f"   - Index saved to: {index_path}")
        print(f"   - Data saved to: {data_path}")
        
        # Test the embeddings
        print("\n" + "="*60)
        print("Testing embeddings with sample queries...")
        print("="*60)
        
        test_queries = ["cell cycle", "Cell Cycle", "immune response", "apoptosis"]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            query_embedding = model.encode([query])
            distances, indices = index.search(query_embedding.astype(np.float32), 3)
            
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                pathway = pathways[idx]
                similarity = 1 / (1 + distance)
                print(f"  {i+1}. '{pathway['name']}' (database: {pathway['database']}, similarity: {similarity:.3f})")
        
    finally:
        driver.close()

if __name__ == "__main__":
    print("="*80)
    print("Creating pathway embeddings from Neo4j")
    print("="*80)
    
    response = input("\n⚠️  This will overwrite existing embeddings. Continue? (y/n): ")
    if response.lower() == 'y':
        start_time = time.time()
        create_embeddings_from_neo4j()
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed/60:.1f} minutes")
    else:
        print("Aborted.")