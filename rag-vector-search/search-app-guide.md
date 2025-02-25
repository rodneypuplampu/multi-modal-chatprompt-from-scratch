# Building a Vector Search Application - Step by Step Guide

This guide will walk you through creating a search application that allows users to input natural language queries and retrieve semantically similar information from your Milvus vector database.

## Overview

We'll build a simple web application with:
1. A text input interface for user queries
2. A backend that converts queries to vectors and searches the Milvus database
3. A results display that shows relevant entities, statements, and relationships

## Prerequisites

- A Milvus database populated with embeddings (using the Text Embedder component)
- Basic knowledge of Python and web development

## Step 1: Set Up Your Project Structure

Create a new directory for your search application:

```bash
mkdir vector-search-app
cd vector-search-app
```

Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required packages:

```bash
pip install flask pymilvus numpy gensim python-dotenv
```

Create the basic project structure:

```
vector-search-app/
├── app.py              # Flask application
├── search_engine.py    # Vector search functionality
├── templates/          # HTML templates
│   ├── index.html      # Search interface
│   └── results.html    # Results display
├── static/             # Static assets
│   ├── css/            # CSS files
│   │   └── style.css   # Styles
│   └── js/             # JavaScript files
│       └── script.js   # Client-side scripts
└── .env                # Environment variables
```

## Step 2: Create the Search Engine Module

Create a file called `search_engine.py` with the following code:

```python
import numpy as np
import gensim.downloader as api
from pymilvus import connections, Collection
from typing import List, Dict, Any

class VectorSearchEngine:
    def __init__(self, milvus_host="localhost", milvus_port="19530", 
                 glove_model="glove-wiki-gigaword-300"):
        """
        Initialize the search engine with connections to Milvus and GloVe
        
        Args:
            milvus_host (str): Milvus server host
            milvus_port (str): Milvus server port
            glove_model (str): GloVe model name
        """
        # Connect to Milvus
        try:
            connections.connect(
                alias="default", 
                host=milvus_host,
                port=milvus_port
            )
            print(f"Connected to Milvus at {milvus_host}:{milvus_port}")
        except Exception as e:
            print(f"Failed to connect to Milvus: {str(e)}")
            raise
        
        # Load GloVe model
        print(f"Loading GloVe model: {glove_model}")
        try:
            self.glove_model = api.load(glove_model)
            print(f"GloVe model loaded with {len(self.glove_model.key_to_index)} words")
        except Exception as e:
            print(f"Error loading GloVe model: {str(e)}")
            raise
        
        # Store vector dimension
        self.vector_dim = 300  # GloVe vectors typically have 300 dimensions
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get GloVe embedding for text (average of word vectors)
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Preprocess the text
        words = text.lower().split()
        
        # Filter out words not in vocabulary
        word_vectors = [self.glove_model[word] for word in words if word in self.glove_model]
        
        if not word_vectors:
            # Return zero vector if no words found
            return np.zeros(self.vector_dim)
        
        # Return average of word vectors
        return np.mean(word_vectors, axis=0)
    
    def search_entities(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities similar to the query text
        
        Args:
            query_text (str): Query text
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: Similar entities with metadata
        """
        # Generate embedding for query text
        query_embedding = self.get_embedding(query_text)
        
        # Get entity collection
        try:
            collection = Collection("nlp_entities")
            collection.load()
        except Exception as e:
            print(f"Error accessing entity collection: {str(e)}")
            return []
        
        # Search for similar entities
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["entity_id", "text", "type", "source_file"]
        )
        
        # Format results
        similar_entities = []
        for hits in results:
            for hit in hits:
                similar_entities.append({
                    "id": hit.id,
                    "entity_id": hit.entity_id,
                    "text": hit.text,
                    "type": hit.type,
                    "source_file": hit.source_file,
                    "similarity_score": 1.0 - (hit.distance / 2.0)  # Convert distance to similarity
                })
        
        return similar_entities
    
    def search_statements(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for statements similar to the query text
        
        Args:
            query_text (str): Query text
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: Similar statements with metadata
        """
        # Generate embedding for query text
        query_embedding = self.get_embedding(query_text)
        
        # Get statement collection
        try:
            collection = Collection("nlp_statements")
            collection.load()
        except Exception as e:
            print(f"Error accessing statement collection: {str(e)}")
            return []
        
        # Search for similar statements
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["statement", "type", "entities", "source_file"]
        )
        
        # Format results
        similar_statements = []
        for hits in results:
            for hit in hits:
                similar_statements.append({
                    "id": hit.id,
                    "statement": hit.statement,
                    "type": hit.type,
                    "entities": hit.entities,
                    "source_file": hit.source_file,
                    "similarity_score": 1.0 - (hit.distance / 2.0)  # Convert distance to similarity
                })
        
        return similar_statements
    
    def search_relationships(self, query_text: str, search_type: str = "all", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relationships related to the query text
        
        Args:
            query_text (str): Query text
            search_type (str): Type of search ("subject", "relation", "object", or "all")
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: Matching relationships with metadata
        """
        # Generate embedding for query text
        query_embedding = self.get_embedding(query_text)
        
        # Get relationship collection
        try:
            collection = Collection("nlp_relationships")
            collection.load()
        except Exception as e:
            print(f"Error accessing relationship collection: {str(e)}")
            return []
        
        # Determine which field to search
        if search_type == "subject":
            search_field = "subject_embedding"
        elif search_type == "relation":
            search_field = "relation_embedding"
        elif search_type == "object":
            search_field = "object_embedding"
        else:
            # For "all", we'll search all three fields and combine results
            results_subject = self._search_relationship_field(
                collection, query_embedding, "subject_embedding", limit)
            results_relation = self._search_relationship_field(
                collection, query_embedding, "relation_embedding", limit)
            results_object = self._search_relationship_field(
                collection, query_embedding, "object_embedding", limit)
            
            # Combine results (removing duplicates by ID)
            all_results = {}
            for result in results_subject + results_relation + results_object:
                if result["id"] not in all_results:
                    all_results[result["id"]] = result
            
            # Sort by similarity score and limit
            return sorted(all_results.values(), 
                          key=lambda x: x["similarity_score"], 
                          reverse=True)[:limit]
        
        # Search a single field
        return self._search_relationship_field(collection, query_embedding, search_field, limit)
    
    def _search_relationship_field(self, collection, query_embedding, field_name, limit):
        """Helper method to search a specific field in the relationship collection"""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field=field_name,
            param=search_params,
            limit=limit,
            output_fields=["subject", "relation", "object", "sentence", "source_file"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "subject": hit.subject,
                    "relation": hit.relation,
                    "object": hit.object,
                    "sentence": hit.sentence,
                    "source_file": hit.source_file,
                    "similarity_score": 1.0 - (hit.distance / 2.0)  # Convert distance to similarity
                })
        
        return formatted_results
    
    def comprehensive_search(self, query_text: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Perform a comprehensive search across all collections
        
        Args:
            query_text (str): Query text
            limit (int): Maximum number of results per category
            
        Returns:
            Dict: Results from all collections
        """
        return {
            "entities": self.search_entities(query_text, limit),
            "statements": self.search_statements(query_text, limit),
            "relationships": self.search_relationships(query_text, "all", limit)
        }
```

## Step 3: Create the Flask Application

Create a file called `app.py` with the following code:

```python
from flask import Flask, render_template, request, jsonify
from search_engine import VectorSearchEngine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the search engine
search_engine = VectorSearchEngine(
    milvus_host=os.getenv("MILVUS_HOST", "localhost"),
    milvus_port=os.getenv("MILVUS_PORT", "19530"),
    glove_model=os.getenv("GLOVE_MODEL", "glove-wiki-gigaword-300")
)

@app.route('/')
def index():
    """Render the search interface"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    # Get query text from form
    query_text = request.form.get('query', '')
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    # Perform comprehensive search
    results = search_engine.comprehensive_search(query_text)
    
    # Format results for display
    return render_template('results.html', 
                          query=query_text,
                          entities=results["entities"],
                          statements=results["statements"],
                          relationships=results["relationships"])

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for search (returns JSON)"""
    # Get data from JSON request
    data = request.get_json()
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    # Perform comprehensive search
    results = search_engine.comprehensive_search(query_text)
    
    # Return results as JSON
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
```

## Step 4: Create the HTML Templates

### Create Templates Directory

```bash
mkdir -p templates static/css static/js
```

### Create `templates/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vector Search Application</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <h1>Vector Search Application</h1>
            <p>Search for entities, relationships, and statements in your text corpus</p>
        </header>
        
        <main>
            <section class="search-section">
                <form action="/search" method="post" id="search-form">
                    <div class="search-container">
                        <input type="text" name="query" id="query" placeholder="Enter your search query..." required>
                        <button type="submit">Search</button>
                    </div>
                </form>
            </section>
            
            <section class="results-section" id="results-container">
                <!-- Results will be loaded here -->
            </section>
        </main>
        
        <footer>
            <p>Powered by Milvus Vector Database and GloVe Embeddings</p>
        </footer>
    </div>
    
    <script src="{{ url_for('static', filename='js/script.js') }}"></script>
</body>
</html>
```

### Create `templates/results.html`:

```html
{% if entities or statements or relationships %}
    <div class="results-summary">
        <h2>Search Results for "{{ query }}"</h2>
        <p>Found {{ entities|length }} entities, {{ statements|length }} statements, and {{ relationships|length }} relationships</p>
    </div>
    
    {% if entities %}
    <div class="result-category">
        <h3>Entities</h3>
        <div class="result-items">
            {% for entity in entities %}
            <div class="result-item">
                <div class="result-header">
                    <span class="result-title">{{ entity.text }}</span>
                    <span class="result-type">{{ entity.type }}</span>
                    <span class="result-score">{{ "%.2f"|format(entity.similarity_score * 100) }}% match</span>
                </div>
                <div class="result-details">
                    <p>Source: {{ entity.source_file }}</p>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
    
    {% if statements %}
    <div class="result-category">
        <h3>Statements</h3>
        <div class="result-items">
            {% for statement in statements %}
            <div class="result-item">
                <div class="result-header">
                    <span class="result-title">{{ statement.statement }}</span>
                    <span class="result-type">{{ statement.type }}</span>
                    <span class="result-score">{{ "%.2f"|format(statement.similarity_score * 100) }}% match</span>
                </div>
                <div class="result-details">
                    <p>Entities: {{ statement.entities }}</p>
                    <p>Source: {{ statement.source_file }}</p>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
    
    {% if relationships %}
    <div class="result-category">
        <h3>Relationships</h3>
        <div class="result-items">
            {% for rel in relationships %}
            <div class="result-item">
                <div class="result-header">
                    <span class="result-title">{{ rel.subject }} {{ rel.relation }} {{ rel.object }}</span>
                    <span class="result-score">{{ "%.2f"|format(rel.similarity_score * 100) }}% match</span>
                </div>
                <div class="result-details">
                    <p>Context: "{{ rel.sentence }}"</p>
                    <p>Source: {{ rel.source_file }}</p>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
{% else %}
    <div class="no-results">
        <h2>No results found for "{{ query }}"</h2>
        <p>Try a different search query or check if your vector database is properly populated.</p>
    </div>
{% endif %}
```

## Step 5: Add CSS and JavaScript

### Create `static/css/style.css`:

```css
/* Basic Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f7fa;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
header {
    text-align: center;
    margin-bottom: 40px;
    padding: 20px 0;
    border-bottom: 1px solid #eaeaea;
}

header h1 {
    font-size: 2.5rem;
    color: #2c3e50;
    margin-bottom: 10px;
}

header p {
    font-size: 1.1rem;
    color: #7f8c8d;
}

/* Search Section */
.search-section {
    margin-bottom: 30px;
}

.search-container {
    display: flex;
    max-width: 800px;
    margin: 0 auto;
}

input[type="text"] {
    flex: 1;
    padding: 15px;
    font-size: 1rem;
    border: 2px solid #ddd;
    border-radius: 4px 0 0 4px;
    outline: none;
    transition: border-color 0.3s;
}

input[type="text"]:focus {
    border-color: #3498db;
}

button {
    padding: 0 20px;
    background-color: #3498db;
    color: white;
    font-size: 1rem;
    border: none;
    border-radius: 0 4px 4px 0;
    cursor: pointer;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2980b9;
}

/* Results Section */
.results-summary {
    margin-bottom: 30px;
    padding-bottom: 15px;
    border-bottom: 1px solid #eaeaea;
}

.results-summary h2 {
    font-size: 1.8rem;
    color: #2c3e50;
    margin-bottom: 10px;
}

.result-category {
    margin-bottom: 40px;
}

.result-category h3 {
    font-size: 1.5rem;
    color: #34495e;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eaeaea;
}

.result-items {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 20px;
}

.result-item {
    background-color: #fff;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 15px;
    transition: transform 0.2s, box-shadow 0.2s;
}

.result-item:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.result-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 10px;
}

.result-title {
    font-weight: bold;
    flex: 1;
    margin-right: 10px;
}

.result-type {
    font-size: 0.8rem;
    text-transform: uppercase;
    background-color: #e0f7fa;
    color: #00838f;
    padding: 3px 8px;
    border-radius: 3px;
    margin-right: 10px;
}

.result-score {
    font-size: 0.9rem;
    color: #27ae60;
}

.result-details {
    font-size: 0.9rem;
    color: #7f8c8d;
}

.result-details p {
    margin-bottom: 5px;
}

.no-results {
    text-align: center;
    padding: 40px 0;
}

.no-results h2 {
    font-size: 1.8rem;
    color: #2c3e50;
    margin-bottom: 15px;
}

.no-results p {
    color: #7f8c8d;
}

/* Footer */
footer {
    text-align: center;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #eaeaea;
    color: #95a5a6;
}
```

### Create `static/js/script.js`:

```javascript
document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('search-form');
    const resultsContainer = document.getElementById('results-container');
    
    // Handle form submission with AJAX
    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading indicator
        resultsContainer.innerHTML = '<div class="loading">Searching...</div>';
        
        // Get form data
        const formData = new FormData(searchForm);
        
        // Send AJAX request
        fetch('/search', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(html => {
            // Update results container with response HTML
            resultsContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error:', error);
            resultsContainer.innerHTML = `
                <div class="error">
                    <h2>Error</h2>
                    <p>An error occurred while processing your request. Please try again.</p>
                </div>
            `;
        });
    });
});
```

## Step 6: Create Environment Variables File

Create a `.env` file for configuration:

```
MILVUS_HOST=localhost
MILVUS_PORT=19530
GLOVE_MODEL=glove-wiki-gigaword-300
PORT=5000
```

## Step 7: Run the Application

Start your application with:

```bash
python app.py
```

Your search application should now be running at http://localhost:5000.

## Step 8: Building a Command-Line Search Tool (Optional)

If you want a simple command-line tool for vector search, create a file called `cli_search.py`:

```python
import argparse
from search_engine import VectorSearchEngine
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Vector Search CLI Tool')
    parser.add_argument('query', help='Search query text')
    parser.add_argument('--type', choices=['all', 'entities', 'statements', 'relationships'], 
                        default='all', help='Type of search to perform')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of results to return')
    parser.add_argument('--output', choices=['pretty', 'json'], default='pretty', 
                        help='Output format')
    
    args = parser.parse_args()
    
    # Initialize search engine
    search_engine = VectorSearchEngine(
        milvus_host=os.getenv("MILVUS_HOST", "localhost"),
        milvus_port=os.getenv("MILVUS_PORT", "19530"),
        glove_model=os.getenv("GLOVE_MODEL", "glove-wiki-gigaword-300")
    )
    
    # Perform search based on type
    if args.type == 'all':
        results = search_engine.comprehensive_search(args.query, args.limit)
    elif args.type == 'entities':
        results = {"entities": search_engine.search_entities(args.query, args.limit)}
    elif args.type == 'statements':
        results = {"statements": search_engine.search_statements(args.query, args.limit)}
    elif args.type == 'relationships':
        results = {"relationships": search_engine.search_relationships(args.query, "all", args.limit)}
    
    # Display results
    if args.output == 'json':
        print(json.dumps(results, indent=2))
    else:
        print(f"\nSearch results for: '{args.query}'\n")
        
        if 'entities' in results and results['entities']:
            print("\n=== ENTITIES ===")
            for i, entity in enumerate(results['entities'], 1):
                print(f"{i}. {entity['text']} ({entity['type']}) - {entity['similarity_score']:.2f}")
                print(f"   Source: {entity['source_file']}")
                print()
        
        if 'statements' in results and results['statements']:
            print("\n=== STATEMENTS ===")
            for i, stmt in enumerate(results['statements'], 1):
                print(f"{i}. {stmt['statement']} - {stmt['similarity_score']:.2f}")
                print(f"   Type: {stmt['type']}")
                print(f"   Source: {stmt['source_file']}")
                print()
        
        if 'relationships' in results and results['relationships']:
            print("\n=== RELATIONSHIPS ===")
            for i, rel in enumerate(results['relationships'], 1):
                print(f"{i}. {rel['subject']} {rel['relation']} {rel['object']} - {rel['similarity_score']:.2f}")
                print(f"   Context: \"{rel['sentence']}\"")
                print(f"   Source: {rel['source_file']}")
                print()
        
        if not any(results.get(k) for k in ['entities', 'statements', 'relationships']):
            print("No results found.")

if __name__ == '__main__':
    main()
```

Run the CLI tool:

```bash
python cli_search.py "artificial intelligence" --limit 3
```

## Step 9: Deploying Your Application

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Create a `requirements.txt` file:

```
flask==2.0.1
pymilvus==2.0.2
numpy==1.21.2
gensim==4.1.2
python-dotenv==0.19.1
```

Build and run the Docker container:

```bash
docker build -t vector-search-app .
docker run -p 5000:5000 --env-file .env vector-search-app
```

## Step 10: Add Advanced Features (Optional)

### 1. Implement a Chat Interface

Enhance your application with a chat-like interface by modifying `index.html` and adding appropriate JavaScript.

### 2. Add Filtering Options

Add filters for entity types, relationship types, or source files.

### 3. Implement User Authentication

Add user authentication to restrict access to your search application.

### 4. Create a Result Visualization

Add a visualization for the knowledge graph using D3.js or a similar library.

## Troubleshooting

### Milvus Connection Issues

If you encounter connection issues with Milvus:

1. Check if Milvus is running:
   ```bash
   docker ps -a
   ```

2. Restart Milvus if needed:
   ```bash
   docker restart milvus_standalone
   ```

### GloVe Model Issues

If you encounter issues with the GloVe model:

1. Try a smaller model like `glove-twitter-25` for testing
2. Ensure you have sufficient RAM for the model

### Performance Optimization

If search performance is slow:

1. Increase the `nprobe` parameter in the search parameters
2. Decrease the vector dimension by using a smaller GloVe model
3. Add caching for frequent queries
4. Use a more efficient index type in Milvus

## Conclusion

You now have a complete vector search application that allows users to query your text corpus using natural language. The application supports searching for entities, statements, and relationships, providing a comprehensive view of the knowledge contained in your documents.

For further enhancements, consider:
- Implementing more advanced NLP techniques
- Adding document preview capabilities
- Creating a more sophisticated UI with filtering and visualization options
- Extending the search to include hybrid search (combining vector and keyword search)
