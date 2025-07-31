# Pathway Semantic Search User Manual

## Overview
This system enables semantic search across 178,742 biological pathways from your Neo4j database using state-of-the-art embedding technology.

## Quick Start

### 1. First-Time Setup (One-Time Process)
```bash
# Create embeddings for all pathways (~1 hour)
python3 run_full_embedding.py
```

This creates two files:
- `pathway_index.faiss` - Vector embeddings
- `pathway_data.pkl` - Pathway metadata

### 2. Search Pathways
```bash
# Interactive search interface
python3 load_and_search.py
```

## How It Works

The system uses **Sentence Transformers** to convert pathway names into high-dimensional vectors, enabling similarity-based search rather than exact keyword matching.

### Example Searches

| Query | Finds Pathways Related To |
|-------|--------------------------|
| "cell division" | Cell cycle, mitosis, chromosome segregation |
| "immune response" | Inflammation, cytokine signaling, T cell activation |
| "metabolism" | Glycolysis, TCA cycle, lipid biosynthesis |
| "cancer" | Oncogenic pathways, tumor suppression, apoptosis |

## Python API Usage

### Basic Search
```python
from neo4j_pathway_embedder import Neo4jPathwayEmbedder

# Load existing embeddings
embedder = Neo4jPathwayEmbedder(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="37754262"
)
embedder.load_index()

# Search
results = embedder.search("insulin signaling", k=5)
for r in results:
    print(f"{r['name']} - Score: {r['similarity_score']:.3f}")
```

### Integration Example
```python
def find_related_pathways(user_query, threshold=0.5):
    """Find pathways with similarity above threshold"""
    embedder.load_index()
    results = embedder.search(user_query, k=20)
    
    # Filter by similarity threshold
    relevant = [r for r in results if r['similarity_score'] > threshold]
    return relevant
```

## Search Tips

### 1. **Use Biological Terms**
✅ Good: "apoptosis", "cell proliferation", "glucose metabolism"  
❌ Poor: "death", "growth", "sugar"

### 2. **Be Specific**
✅ Good: "T cell activation"  
❌ Poor: "immune"

### 3. **Try Synonyms**
- "programmed cell death" → "apoptosis"
- "blood sugar" → "glucose homeostasis"
- "fat metabolism" → "lipid metabolism"

### 4. **Combine Concepts**
- "cancer AND metabolism" → "cancer metabolism"
- "heart AND development" → "cardiac development"

## Understanding Results

### Similarity Scores
- **0.8-1.0**: Very high similarity (often exact or near-exact matches)
- **0.6-0.8**: High similarity (closely related pathways)
- **0.4-0.6**: Moderate similarity (related concepts)
- **<0.4**: Low similarity (distant relationships)

### Example Output
```
Query: 'glucose metabolism'

1. Glucose Metabolic Process
   Score: 0.8234
   Database: GO

2. Regulation Of Glucose Metabolic Process  
   Score: 0.7891
   Database: GO

3. Glycolysis And Gluconeogenesis
   Score: 0.7123
   Database: KEGG
```

## Advanced Usage

### Batch Processing
```python
# Search multiple queries
queries = ["apoptosis", "cell cycle", "DNA repair"]
all_results = {}

for query in queries:
    all_results[query] = embedder.search(query, k=10)
```

### Export Results
```python
import pandas as pd

# Convert to DataFrame
results = embedder.search("cancer pathways", k=50)
df = pd.DataFrame(results)
df.to_csv("cancer_pathway_search_results.csv", index=False)
```

### Custom Similarity Threshold
```python
def get_highly_similar(query, min_score=0.7):
    results = embedder.search(query, k=100)
    return [r for r in results if r['similarity_score'] >= min_score]
```

## Troubleshooting

### "No embeddings found" Error
Run `python3 run_full_embedding.py` first to create embeddings.

### Slow Search Performance
- Ensure you're loading from disk, not re-embedding
- Consider using GPU if available
- Reduce `k` parameter for faster results

### Memory Issues
The system requires ~300MB RAM for 178k pathways. Ensure sufficient memory.

### Update Embeddings
Re-run `run_full_embedding.py` when pathways are added/modified in Neo4j.

## Performance

- **Search Speed**: <100ms per query
- **Accuracy**: Depends on query specificity
- **Index Size**: ~260MB for 178k pathways
- **Memory Usage**: ~300MB when loaded

## Best Practices

1. **Regular Updates**: Re-embed monthly or when significant changes occur
2. **Query Logging**: Track searches to improve the system
3. **Feedback Loop**: Note poor results to refine queries
4. **Combine Methods**: Use with traditional keyword search for best results

## Support

For issues or questions:
1. Check pathway names exist in Neo4j
2. Verify embedding files are present
3. Ensure Python packages are installed
4. Test with known pathway names first