# RAG Vector Search Web GUI Implementation Plan

This document outlines the step-by-step implementation plan for developing the web-based GUI for the RAG Vector Search system. The plan is divided into phases to ensure a structured and methodical approach.

## Phase 1: Environment Setup and Code Preparation

### Step 1: Set up the project structure
```bash
# Create project directory
mkdir -p rag-vector-search-web
cd rag-vector-search-web

# Create subdirectories
mkdir -p static/css static/js templates uploads processed structured
```

### Step 2: Create a Python virtual environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Convert the existing scripts into importable modules
1. Copy the original python files (`text-cleaner.py`, `text-structurer.py`, `text-embedder.py`) to your project directory
2. Run the conversion script to transform them into proper modules:
```bash
python convert_to_modules.py
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
```

## Phase 2: Milvus Setup

### Step 1: Install Milvus using Docker (standalone version)
```bash
# Pull and start Milvus containers
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

### Step 2: Verify Milvus installation
```bash
# Check if Milvus containers are running
docker-compose ps
```

### Step 3: Create a simple test script to verify connection
```python
# Create a file named test_milvus.py
from pymilvus import connections

connections.connect("default", host="localhost", port="19530")
print("Successfully connected to Milvus!")
```

## Phase 3: Web UI Implementation

### Step 1: Create the main Flask application file (app.py)
- Copy the content from the `app.py` artifact

### Step 2: Create the frontend templates
- Create `templates/index.html` with the content from the index-template artifact
- Create `static/css/style.css` with the content from the css-styles artifact 
- Create `static/js/main.js` with the content from the js-main artifact

### Step 3: Implement the JavaScript for result display
Add the missing parts to `static/js/main.js`:

```javascript
// Function to display document details (continued)
let relationshipsHtml = `
    <div class="mb-3">
        <h6>Relationships (${(data.relationships || []).length})</h6>
    </div>
    <div class="result-list">
`;

(data.relationships || []).forEach(rel => {
    relationshipsHtml += `
        <div class="result-item">
            <div class="relationship">
                <span class="relationship-subject">${rel.subject || 'Unknown'}</span>
                <span class="relationship-relation">${rel.relation || 'related to'}</span>
                <span class="relationship-object">${rel.object || 'Unknown'}</span>
            </div>
            <div class="result-text mt-2">
                "${rel.sentence || 'No context available'}"
            </div>
        </div>
    `;
});

relationshipsHtml += '</div>';

// Statements tab content
let statementsHtml = `
    <div class="mb-3">
        <h6>Statements (${(data.statements || []).length})</h6>
    </div>
    <div class="result-list">
`;

(data.statements || []).forEach(statement => {
    statementsHtml += `
        <div class="result-item">
            <div class="result-text">"${statement.statement || 'Unknown statement'}"</div>
            <div class="result-meta mt-2">
                <span class="badge bg-primary">${statement.type || 'unknown'}</span>
                <span class="ms-2">Entities: ${(statement.entities || []).join(', ')}</span>
            </div>
        </div>
    `;
});

statementsHtml += '</div>';

// Combine all tab content
let tabContentHtml = `
    <div class="tab-content" id="documentTabsContent">
        <div class="tab-pane fade show active" id="overview" role="tabpanel">
            ${overviewHtml}
        