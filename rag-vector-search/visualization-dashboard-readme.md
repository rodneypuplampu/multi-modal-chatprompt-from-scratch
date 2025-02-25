# Vector Database Visualization Dashboard

This README guides you through implementing a comprehensive data visualization dashboard for exploring and analyzing the contents of your Milvus vector database. The dashboard will provide interactive visualizations of entities, relationships, and semantic clusters from your processed text data.

## Overview

The visualization dashboard provides:

1. **Entity Explorer** - Visualize and filter entities by type and source
2. **Relationship Network** - Interactive graph visualization of entities and their relationships
3. **Semantic Clustering** - Dimension reduction and clustering of vector embeddings
4. **Search Interface** - Visual interface for semantic search results
5. **Document Statistics** - Metrics and trends from your document corpus

## Technical Stack

- **Backend**: Flask/FastAPI for API endpoints
- **Vector Database**: Milvus for storing and querying embeddings
- **Visualization**: D3.js, Plotly, and ECharts for interactive visualizations
- **Frontend**: React or Vue.js for UI components
- **Authentication**: JWT for securing the dashboard (optional)

## Prerequisites

- Python 3.8+
- Node.js 14+
- Running Milvus instance with populated collections
- Processed data from the NLP pipeline (Text Cleaner, Structurer, and Embedder)

## Implementation Steps

### 1. Set Up Project Structure

```
vector-dashboard/
├── backend/                # Flask/FastAPI backend
│   ├── app.py              # Main application
│   ├── config.py           # Configuration
│   ├── routes/             # API routes
│   ├── services/           # Business logic
│   └── utils/              # Utility functions
├── frontend/               # React/Vue frontend
│   ├── public/             # Static assets
│   ├── src/                # Source code
│   │   ├── components/     # UI components
│   │   ├── pages/          # Dashboard pages
│   │   ├── services/       # API services
│   │   └── utils/          # Utility functions
│   ├── package.json        # Dependencies
│   └── vite.config.js      # Build configuration
└── README.md               # Documentation
```

### 2. Implement Backend API

Create a new file `backend/app.py`:

```python
from flask import Flask, jsonify, request
from pymilvus import connections, Collection
from flask_cors import CORS
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import gensim.downloader as api

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Load GloVe model for query processing
glove_model = api.load("glove-wiki-gigaword-300")

@app.route('/api/entity-types', methods=['GET'])
def get_entity_types():
    """Get all unique entity types and their counts"""
    collection = Collection("nlp_entities")
    collection.load()
    
    # Execute a query to get entity types
    results = collection.query(
        expr="type != ''",
        output_fields=["type"]
    )
    
    # Count occurrences of each type
    type_counts = {}
    for result in results:
        entity_type = result['type']
        if entity_type in type_counts:
            type_counts[entity_type] += 1
        else:
            type_counts[entity_type] = 1
    
    # Format response
    formatted_types = [{"name": key, "count": value} for key, value in type_counts.items()]
    
    return jsonify(formatted_types)

@app.route('/api/entities', methods=['GET'])
def get_entities():
    """Get entities with optional filtering"""
    entity_type = request.args.get('type', '')
    limit = int(request.args.get('limit', 100))
    
    collection = Collection("nlp_entities")
    collection.load()
    
    # Build query expression
    if entity_type:
        expr = f"type == '{entity_type}'"
    else:
        expr = "type != ''" 
    
    # Execute query
    results = collection.query(
        expr=expr,
        output_fields=["entity_id", "text", "type", "source_file"],
        limit=limit
    )
    
    return jsonify(results)

@app.route('/api/entity-network', methods=['GET'])
def get_entity_network():
    """Get entity relationship network data"""
    # Get relationships collection
    collection = Collection("nlp_relationships")
    collection.load()
    
    # Get relationships
    results = collection.query(
        expr="subject != ''",
        output_fields=["subject", "relation", "object", "source_file"],
        limit=1000
    )
    
    # Process into network format
    nodes = {}
    links = []
    
    for result in results:
        subject = result['subject']
        obj = result['object']
        relation = result['relation']
        
        # Add nodes if they don't exist
        if subject not in nodes:
            nodes[subject] = {"id": subject, "group": 1}
            
        if obj not in nodes:
            nodes[obj] = {"id": obj, "group": 2}
            
        # Add link
        links.append({
            "source": subject,
            "target": obj,
            "value": 1,
            "label": relation
        })
    
    # Format for D3 force-directed graph
    network_data = {
        "nodes": list(nodes.values()),
        "links": links
    }
    
    return jsonify(network_data)

@app.route('/api/semantic-clusters', methods=['GET'])
def get_semantic_clusters():
    """Get semantic clusters using t-SNE dimensionality reduction"""
    num_clusters = int(request.args.get('clusters', 5))
    entity_type = request.args.get('type', '')
    
    # Get entities collection
    collection = Collection("nlp_entities")
    collection.load()
    
    # Build query expression
    if entity_type:
        expr = f"type == '{entity_type}'"
    else:
        expr = "type != ''"
    
    # Get entities and embeddings
    results = collection.query(
        expr=expr,
        output_fields=["entity_id", "text", "type", "embedding"],
        limit=1000
    )
    
    if not results:
        return jsonify({"error": "No entities found"})
    
    # Extract embeddings and metadata
    embeddings = []
    metadata = []
    
    for result in results:
        embeddings.append(result['embedding'])
        metadata.append({
            "entity_id": result['entity_id'],
            "text": result['text'],
            "type": result['type']
        })
    
    # Apply t-SNE dimensionality reduction
    embeddings_array = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    reduced_vectors = tsne.fit_transform(embeddings_array)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)), random_state=42)
    clusters = kmeans.fit_predict(embeddings_array)
    
    # Prepare results
    cluster_data = []
    for i, (point, meta, cluster) in enumerate(zip(reduced_vectors, metadata, clusters)):
        cluster_data.append({
            "x": float(point[0]),
            "y": float(point[1]),
            "cluster": int(cluster),
            "text": meta["text"],
            "type": meta["type"],
            "entity_id": meta["entity_id"]
        })
    
    return jsonify(cluster_data)

@app.route('/api/document-stats', methods=['GET'])
def get_document_stats():
    """Get document statistics"""
    # Get entity collection
    entity_collection = Collection("nlp_entities")
    entity_collection.load()
    
    # Get relationships collection
    rel_collection = Collection("nlp_relationships")
    rel_collection.load()
    
    # Get statements collection
    stmt_collection = Collection("nlp_statements")
    stmt_collection.load()
    
    # Get document sources
    sources = entity_collection.query(
        expr="source_file != ''",
        output_fields=["source_file"],
        limit=10000
    )
    
    # Count entities and relationships per document
    doc_stats = {}
    
    for source in sources:
        doc = source['source_file']
        if doc not in doc_stats:
            doc_stats[doc] = {"entities": 0, "relationships": 0, "statements": 0}
        doc_stats[doc]["entities"] += 1
    
    # Count relationships
    rel_sources = rel_collection.query(
        expr="source_file != ''", 
        output_fields=["source_file"],
        limit=10000
    )
    
    for source in rel_sources:
        doc = source['source_file']
        if doc in doc_stats:
            doc_stats[doc]["relationships"] += 1
    
    # Count statements
    stmt_sources = stmt_collection.query(
        expr="source_file != ''",
        output_fields=["source_file"],
        limit=10000
    )
    
    for source in stmt_sources:
        doc = source['source_file']
        if doc in doc_stats:
            doc_stats[doc]["statements"] += 1
    
    # Format the results
    result = []
    for doc, stats in doc_stats.items():
        result.append({
            "document": doc,
            "entities": stats["entities"],
            "relationships": stats["relationships"],
            "statements": stats["statements"]
        })
    
    return jsonify(result)

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint with visualization data"""
    data = request.get_json()
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    # Get embedding for query
    words = query_text.lower().split()
    word_vectors = [glove_model[word] for word in words if word in glove_model]
    
    if not word_vectors:
        return jsonify({"error": "Query contains no known words"}), 400
    
    query_embedding = np.mean(word_vectors, axis=0).tolist()
    
    # Search in entities collection
    entity_collection = Collection("nlp_entities")
    entity_collection.load()
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    entity_results = entity_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=10,
        output_fields=["entity_id", "text", "type", "source_file"]
    )
    
    # Format entity results
    formatted_entities = []
    for hits in entity_results:
        for hit in hits:
            formatted_entities.append({
                "id": hit.id,
                "entity_id": hit.entity_id,
                "text": hit.text,
                "type": hit.type,
                "source_file": hit.source_file,
                "score": 1.0 - (hit.distance / 2.0)  # Convert distance to similarity score
            })
    
    # Search in statements collection
    stmt_collection = Collection("nlp_statements")
    stmt_collection.load()
    
    stmt_results = stmt_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=10,
        output_fields=["statement", "type", "entities", "source_file"]
    )
    
    # Format statement results
    formatted_statements = []
    for hits in stmt_results:
        for hit in hits:
            formatted_statements.append({
                "id": hit.id,
                "statement": hit.statement,
                "type": hit.type,
                "entities": hit.entities,
                "source_file": hit.source_file,
                "score": 1.0 - (hit.distance / 2.0)
            })
    
    # Combine results
    search_results = {
        "entities": formatted_entities,
        "statements": formatted_statements
    }
    
    return jsonify(search_results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
```

### 3. Create Frontend Application

Initialize a new React application using Vite:

```bash
# Create new React project with Vite
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install

# Install dependencies
npm install d3 plotly.js-dist-min react-plotly.js @types/plotly.js
npm install axios react-router-dom @mui/material @emotion/react @emotion/styled
npm install react-force-graph recharts
```

Create the main dashboard components:

#### App.tsx

```tsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import EntityExplorer from './pages/EntityExplorer';
import RelationshipNetwork from './pages/RelationshipNetwork';
import SemanticClusters from './pages/SemanticClusters';
import DocumentStats from './pages/DocumentStats';
import SearchVisualizer from './pages/SearchVisualizer';
import Navbar from './components/Navbar';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <div className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/entities" element={<EntityExplorer />} />
            <Route path="/network" element={<RelationshipNetwork />} />
            <Route path="/clusters" element={<SemanticClusters />} />
            <Route path="/documents" element={<DocumentStats />} />
            <Route path="/search" element={<SearchVisualizer />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
```

#### RelationshipNetwork.tsx

```tsx
import { useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import axios from 'axios';
import './RelationshipNetwork.css';

interface Node {
  id: string;
  group: number;
}

interface Link {
  source: string;
  target: string;
  value: number;
  label: string;
}

interface NetworkData {
  nodes: Node[];
  links: Link[];
}

function RelationshipNetwork() {
  const [networkData, setNetworkData] = useState<NetworkData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:5001/api/entity-network');
        setNetworkData(response.data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load network data');
        setLoading(false);
        console.error(err);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div className="loading">Loading network data...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="network-container">
      <h1>Entity Relationship Network</h1>
      <div className="graph-container">
        <ForceGraph2D
          graphData={networkData}
          nodeLabel="id"
          nodeColor={(node) => (node.group === 1 ? '#3498db' : '#e74c3c')}
          linkLabel={(link) => link.label}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          linkCurvature={0.25}
          width={800}
          height={600}
        />
      </div>
    </div>
  );
}

export default RelationshipNetwork;
```

#### SemanticClusters.tsx

```tsx
import { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';
import './SemanticClusters.css';

interface ClusterPoint {
  x: number;
  y: number;
  cluster: number;
  text: string;
  type: string;
  entity_id: string;
}

function SemanticClusters() {
  const [clusterData, setClusterData] = useState<ClusterPoint[]>([]);
  const [numClusters, setNumClusters] = useState<number>(5);
  const [entityType, setEntityType] = useState<string>('');
  const [entityTypes, setEntityTypes] = useState<{name: string, count: number}[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // Fetch entity types
    const fetchEntityTypes = async () => {
      try {
        const response = await axios.get('http://localhost:5001/api/entity-types');
        setEntityTypes(response.data);
      } catch (err) {
        console.error('Failed to fetch entity types:', err);
      }
    };

    fetchEntityTypes();
  }, []);

  useEffect(() => {
    // Fetch cluster data
    const fetchClusterData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(
          `http://localhost:5001/api/semantic-clusters?clusters=${numClusters}&type=${entityType}`
        );
        setClusterData(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch cluster data:', err);
        setLoading(false);
      }
    };

    fetchClusterData();
  }, [numClusters, entityType]);

  // Prepare data for Plotly
  const getPlotData = () => {
    // Group by cluster
    const clusters = new Map<number, ClusterPoint[]>();
    
    clusterData.forEach((point) => {
      if (!clusters.has(point.cluster)) {
        clusters.set(point.cluster, []);
      }
      clusters.get(point.cluster)?.push(point);
    });
    
    // Create traces for each cluster
    return Array.from(clusters.entries()).map(([cluster, points]) => ({
      x: points.map(p => p.x),
      y: points.map(p => p.y),
      text: points.map(p => `${p.text} (${p.type})`),
      mode: 'markers',
      type: 'scatter',
      name: `Cluster ${cluster}`,
      marker: {
        size: 10
      }
    }));
  };

  if (loading) return <div className="loading">Loading cluster data...</div>;

  return (
    <div className="clusters-container">
      <h1>Semantic Clusters Visualization</h1>
      
      <div className="controls">
        <div className="control-group">
          <label htmlFor="numClusters">Number of Clusters:</label>
          <input
            type="range"
            id="numClusters"
            min="2"
            max="10"
            value={numClusters}
            onChange={(e) => setNumClusters(parseInt(e.target.value))}
          />
          <span>{numClusters}</span>
        </div>
        
        <div className="control-group">
          <label htmlFor="entityType">Entity Type:</label>
          <select
            id="entityType"
            value={entityType}
            onChange={(e) => setEntityType(e.target.value)}
          >
            <option value="">All Types</option>
            {entityTypes.map((type) => (
              <option key={type.name} value={type.name}>
                {type.name} ({type.count})
              </option>
            ))}
          </select>
        </div>
      </div>
      
      <div className="plot-container">
        <Plot
          data={getPlotData()}
          layout={{
            title: 'Entity Clusters (t-SNE)',
            width: 800,
            height: 600,
            hovermode: 'closest',
            xaxis: { title: 't-SNE Component 1' },
            yaxis: { title: 't-SNE Component 2' }
          }}
        />
      </div>
    </div>
  );
}

export default SemanticClusters;
```

#### DocumentStats.tsx

```tsx
import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import './DocumentStats.css';

interface DocumentStat {
  document: string;
  entities: number;
  relationships: number;
  statements: number;
}

function DocumentStats() {
  const [stats, setStats] = useState<DocumentStat[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:5001/api/document-stats');
        setStats(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch document stats:', err);
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) return <div className="loading">Loading document statistics...</div>;

  return (
    <div className="stats-container">
      <h1>Document Statistics</h1>
      
      <div className="chart-container">
        <h2>Entities, Relationships, and Statements by Document</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={stats}
            margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="document" 
              angle={-45} 
              textAnchor="end" 
              height={70} 
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="entities" stackId="a" fill="#3498db" name="Entities" />
            <Bar dataKey="relationships" stackId="a" fill="#2ecc71" name="Relationships" />
            <Bar dataKey="statements" stackId="a" fill="#e74c3c" name="Statements" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="stats-table">
        <h2>Document Details</h2>
        <table>
          <thead>
            <tr>
              <th>Document</th>
              <th>Entities</th>
              <th>Relationships</th>
              <th>Statements</th>
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {stats.map((doc) => (
              <tr key={doc.document}>
                <td>{doc.document}</td>
                <td>{doc.entities}</td>
                <td>{doc.relationships}</td>
                <td>{doc.statements}</td>
                <td>{doc.entities + doc.relationships + doc.statements}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default DocumentStats;
```

#### SearchVisualizer.tsx

```tsx
import { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './SearchVisualizer.css';

interface Entity {
  id: number;
  entity_id: string;
  text: string;
  type: string;
  source_file: string;
  score: number;
}

interface Statement {
  id: number;
  statement: string;
  type: string;
  entities: string;
  source_file: string;
  score: number;
}

interface SearchResults {
  entities: Entity[];
  statements: Statement[];
}

function SearchVisualizer() {
  const [query, setQuery] = useState<string>('');
  const [results, setResults] = useState<SearchResults | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5001/api/search', { query });
      setResults(response.data);
      setLoading(false);
    } catch (err) {
      console.error('Search failed:', err);
      setLoading(false);
    }
  };

  // Prepare entity bar chart data
  const getEntityChartData = () => {
    if (!results || results.entities.length === 0) return [];
    
    return [{
      x: results.entities.map(e => e.text),
      y: results.entities.map(e => e.score),
      type: 'bar',
      marker: {
        color: results.entities.map(e => scoreToColor(e.score))
      }
    }];
  };

  // Prepare statement bar chart data
  const getStatementChartData = () => {
    if (!results || results.statements.length === 0) return [];
    
    return [{
      x: results.statements.map(s => s.statement.substring(0, 30) + '...'),
      y: results.statements.map(s => s.score),
      type: 'bar',
      marker: {
        color: results.statements.map(s => scoreToColor(s.score))
      }
    }];
  };

  // Convert score to color
  const scoreToColor = (score: number) => {
    // Generate color from red to green based on score
    const r = Math.round(255 * (1 - score));
    const g = Math.round(255 * score);
    return `rgb(${r}, ${g}, 100)`;
  };

  return (
    <div className="search-container">
      <h1>Search Visualization</h1>
      
      <div className="search-box">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your search query..."
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
        />
        <button onClick={handleSearch} disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>
      
      {results && (
        <div className="results-container">
          <div className="results-summary">
            <h2>Search Results for: "{query}"</h2>
            <p>Found {results.entities.length} entities and {results.statements.length} statements</p>
          </div>
          
          {results.entities.length > 0 && (
            <div className="chart-section">
              <h3>Entity Relevance</h3>
              <Plot
                data={getEntityChartData()}
                layout={{
                  height: 400,
                  width: 800,
                  title: 'Entity Similarity Scores',
                  xaxis: { title: 'Entity' },
                  yaxis: { title: 'Similarity Score', range: [0, 1] }
                }}
              />
              
              <div className="results-table">
                <h3>Entity Details</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Entity</th>
                      <th>Type</th>
                      <th>Source</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.entities.map((entity) => (
                      <tr key={entity.id}>
                        <td>{entity.text}</td>
                        <td>{entity.type}</td>
                        <td>{entity.source_file}</td>
                        <td>{entity.score.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          
          {results.statements.length > 0 && (
            <div className="chart-section">
              <h3>Statement Relevance</h3>
              <Plot
                data={getStatementChartData()}
                layout={{
                  height: 400,
                  width: 800,
                  title: 'Statement Similarity Scores',
                  xaxis: { title: 'Statement' },
                  yaxis: { title: 'Similarity Score', range: [0, 1] }
                }}
              />
              
              <div className="results-table">
                <h3>Statement Details</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Statement</th>
                      <th>Type</th>
                      <th>Source</th>
                      <th>Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.statements.map((statement) => (
                      <tr key={statement.id}>
                        <td>{statement.statement}</td>
                        <td>{statement.type}</td>
                        <td>{statement.source_file}</td>
                        <td>{statement.score.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default SearchVisualizer;
```

### 4. Start and Test the Dashboard

1. Start the backend server:
   ```bash
   cd backend
   python app.py
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

3. Access the dashboard at http://localhost:3000

## Key Dashboard Components

### 1. Entity Explorer

- **Purpose**: Visualize and filter entity data by type and source
- **Features**:
  - Interactive table of entities
  - Filtering by entity type
  - Sorting by relevance or alphabetical order
  - Source document information display

### 2. Relationship Network Graph

- **Purpose**: Interactively explore relationships between entities
- **Features**:
  - Force-directed graph visualization
  - Node clusters by entity type
  - Edge labels showing relationship types
  - Zooming and panning for exploration
  - Node selection to highlight connections

### 3. Semantic Clustering Visualization

- **Purpose**: Visualize how entities and concepts cluster semantically
- **Features**:
  - t-SNE dimensionality reduction for vector visualization
  - K-means clustering to identify semantic groups
  - Interactive controls for cluster count
  - Filtering by entity type
  - Tooltips with entity information

### 4. Document Statistics

- **Purpose**: Analyze the distribution of entities and relationships across documents
- **Features**:
  - Bar charts showing entity counts by document
  - Relationship density visualizations
  