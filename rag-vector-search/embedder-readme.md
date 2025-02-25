# NLP Text Embedder with GloVe and Milvus

This guide explains how to implement the Text Embedder component, which is the third and final step in the NLP processing pipeline. This component generates vector embeddings for structured text data and stores them in a Milvus vector database for efficient similarity search.

## Overview

The Text Embedder takes structured JSON output from the Text Structurer and:
1. Generates GloVe vector embeddings for entities, statements, and relationships
2. Stores these embeddings in a Milvus vector database with their metadata
3. Provides search functionality for finding similar entities and relationships

## Features

- **GloVe Embeddings**: Uses pre-trained GloVe word embeddings to vectorize text
- **Milvus Integration**: Stores embeddings in a high-performance vector database
- **Optimized Schema Design**: Separate collections for entities, statements, and relationships
- **Metadata Preservation**: Stores context and source information with embeddings
- **Similarity Search**: Search for semantically similar entities and statements
- **Relationship Queries**: Find relationships by subject, relation, or object

## Installation Requirements

### 1. Install Milvus Server

The easiest way to run Milvus is using Docker:

```bash
# Pull and run Milvus using Docker
docker run -d --name milvus_standalone \
    -p 19530:19530 \
    -p 19121:19121 \
    -v /path/to/milvus/data:/var/lib/milvus/data \
    -v /path/to/milvus/conf:/var/lib/milvus/conf \
    -v /path/to/milvus/logs:/var/lib/milvus/logs \
    milvusdb/milvus:latest
```

For production environments, refer to the [Milvus documentation](https://milvus.io/docs) for cluster setup.

### 2. Install Required Python Packages

```bash
pip install pymilvus pandas numpy tqdm gensim
```

### 3. Create directories for Milvus data (if using Docker)

```bash
mkdir -p /path/to/milvus/data
mkdir -p /path/to/milvus/conf
mkdir -p /path/to/milvus/logs
```

## Implementation

1. Save the Text Embedder code to a file named `text_embedder.py`
2. Ensure you have processed your text files with both the Text Cleaning Utility and Text Structurer, which produces files with the prefix `struct_` and a `.json` extension

## Usage

### Basic Usage

```bash
python text_embedder.py struct_mybook.txt.json
```

This will process the structured JSON file, generate embeddings, and store them in Milvus.

### Advanced Options

```bash
# Specify a different GloVe model
python text_embedder.py struct_mybook.txt.json --glove-model glove-wiki-gigaword-100

# Connect to a remote Milvus server
python text_embedder.py struct_mybook.txt.json --milvus-host 192.168.1.100 --milvus-port 19530

# Process all structured files in a directory
python text_embedder.py structured_data_directory/
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input JSON file or directory (output from TextStructurer) |
| `--glove-model` | GloVe model to use (default: glove-wiki-gigaword-300) |
| `--milvus-host` | Milvus server host (default: localhost) |
| `--milvus-port` | Milvus server port (default: 19530) |

## How It Works

### 1. GloVe Embedding Generation

The embedder uses pre-trained GloVe word embeddings from gensim to convert text to vector representations:

```python
# Load GloVe model
self.glove_model = api.load(glove_model)

# Get embedding for text (average of word vectors)
def get_embedding(self, text):
    words = text.lower().split()
    word_vectors = [self.glove_model[word] for word in words if word in self.glove_model]
    return np.mean(word_vectors, axis=0)
```

### 2. Milvus Collection Design

The embedder creates three separate collections in Milvus:

1. **Entities Collection**: Stores embeddings for named entities
   ```
   - id (primary key)
   - entity_id (string)
   - text (string)
   - type (string)
   - source_file (string)
   - embedding (vector)
   ```

2. **Statements Collection**: Stores embeddings for relationship statements
   ```
   - id (primary key)
   - statement (string)
   - type (string)
   - entities (JSON)
   - source_file (string)
   - embedding (vector)
   ```

3. **Relationships Collection**: Stores embeddings for relationship components
   ```
   - id (primary key)
   - subject (string)
   - relation (string)
   - object (string)
   - sentence (string)
   - source_file (string)
   - subject_embedding (vector)
   - relation_embedding (vector)
   - object_embedding (vector)
   ```

### 3. Data Processing and Insertion

The embedder processes structured data and prepares it for insertion into Milvus:

```python
def process_structured_data(self, structured_data, source_file):
    processed_data = {"entities": [], "statements": [], "relationships": []}
    
    # Process entities
    for entity_id, entity_data in structured_data.get("entities", {}).items():
        entity_text = entity_data.get("text", "")
        entity_embedding = self.get_embedding(entity_text)
        processed_data["entities"].append({
            "entity_id": entity_id,
            "text": entity_text,
            "type": entity_data.get("type", "unknown"),
            "source_file": source_file,
            "embedding": entity_embedding.tolist()
        })
    
    # Process statements and relationships similarly...
    
    return processed_data
```

### 4. Vector Similarity Search

The embedder provides methods to search for similar entities and statements:

```python
def search_similar_entities(self, query_text, limit=10):
    # Generate embedding for query text
    query_embedding = self.get_embedding(query_text)
    
    # Get entity collection
    collection = Collection("nlp_entities")
    collection.load()
    
    # Search for similar entities
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["entity_id", "text", "type", "source_file"]
    )
    
    # Format and return results...
```

## Optimizing Performance

### 1. GloVe Model Selection

Choose an appropriate GloVe model based on your needs:
- `glove-wiki-gigaword-50`: Smallest, fastest, less accurate
- `glove-wiki-gigaword-100`: Good balance for many applications
- `glove-wiki-gigaword-300`: Largest, most accurate, requires more memory

### 2. Milvus Index Parameters

Adjust the index parameters for different trade-offs:
```python
# For smaller datasets (faster build, slower search)
index_params = {
    "metric_type": "L2",
    "index_type": "FLAT",
    "params": {}
}

# For larger datasets (slower build, faster search)
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

# For very large datasets
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 500}
}
```

### 3. Batch Processing

For large datasets, process files in batches to manage memory usage:
```python
# Process files in batches of 10
for i in range(0, len(json_files), 10):
    batch_files = json_files[i:i+10]
    # Process batch...
```

## Using the Vector Database

After embedding and storing your data, you can use Milvus for various search operations:

### 1. Find Similar Entities

```python
similar_entities = embedder.search_similar_entities("artificial intelligence", limit=5)
```

### 2. Find Similar Statements

```python
similar_statements = embedder.search_similar_statements(
    "The company released a new product", limit=5
)
```

### 3. Search Relationships

```python
# Find relationships with "John" as the subject
john_relationships = embedder.search_relationships(subject="John", limit=5)

# Find relationships involving "acquisition"
acquisition_relationships = embedder.search_relationships(relation="acquired", limit=5)
```

## Integration with Full Pipeline

1. **First step**: Clean text with the Text Cleaning Utility
   ```bash
   python text_cleaner.py my_book.txt --output cleaned
   ```

2. **Second step**: Extract structured information
   ```bash
   python text_structurer.py cleaned/clean_my_book.txt --output structured
   ```

3. **Third step**: Generate embeddings and store in Milvus
   ```bash
   python text_embedder.py structured/struct_my_book.txt.json
   ```

## Extending the Embedder

### 1. Using Different Embedding Models

You can modify the embedder to use different embedding models:

```python
# Using Sentence Transformers instead of GloVe
from sentence_transformers import SentenceTransformer

# In __init__:
self.model = SentenceTransformer('all-MiniLM-L6-v2')

# In get_embedding:
def get_embedding(self, text):
    return self.model.encode(text)
```

### 2. Adding Custom Metadata Fields

Add additional metadata fields to the Milvus collections:

```python
# Add a confidence score field
FieldSchema(name="confidence", dtype=DataType.FLOAT)
```

### 3. Implementing Custom Search Functions

Create domain-specific search functions:

```python
def search_company_relationships(self, company_name, limit=10):
    # Custom search implementation...
```

## Troubleshooting

### Milvus Connection Issues

If you encounter connection issues:

1. Check if Milvus is running:
   ```bash
   docker ps -a
   ```

2. Restart Milvus if needed:
   ```bash
   docker restart milvus_standalone
   ```

3. Verify connectivity:
   ```python
   from pymilvus import connections
   connections.connect("default", host="localhost", port="19530")
   ```

### Memory Issues

If you encounter memory errors with large files:

1. Use a smaller GloVe model (e.g., `glove-wiki-gigaword-50`)
2. Process data in smaller batches
3. Implement a memory-efficient streaming approach

### Missing Words in GloVe Vocabulary

If important domain-specific terms are missing from GloVe:

1. Consider using a domain-specific embedding model
2. Implement a custom vocabulary handling strategy:
   ```python
   if word not in self.glove_model and '-' in word:
       parts = word.split('-')
       vectors = [self.glove_model[part] for part in parts if part in self.glove_model]
       if vectors:
           return np.mean(vectors, axis=0)
   ```

## Next Steps

After implementing the Text Embedder, you can:

1. **Build Search Applications**: Create applications that leverage the vector database for semantic search
2. **Implement Analytics**: Analyze relationships and entities in your corpus
3. **Explore Graph Visualization**: Visualize the knowledge graph using tools like NetworkX or D3.js
4. **Fine-tune Embeddings**: Train or fine-tune domain-specific embeddings for better performance

## References

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Milvus Documentation](https://milvus.io/docs)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [PyMilvus API Reference](https://pymilvus.readthedocs.io/)
