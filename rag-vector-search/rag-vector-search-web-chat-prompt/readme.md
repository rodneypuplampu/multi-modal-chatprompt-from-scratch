# RAG Vector Search Web GUI

This document provides a step-by-step guide to implement a web-based Graphical User Interface (GUI) for the RAG Vector Search system, which processes text documents through multiple stages of cleaning, structuring, and embedding for semantic search capabilities.

## System Overview

The RAG Vector Search system consists of three main components:
1. **Text Cleaner** (`nlp_text_cleaner.py`) - Normalizes and cleans raw text
2. **Text Structurer** (`text_structurer.py`) - Extracts entities and relationships from cleaned text
3. **Text Embedder** (`text_embedder.py`) - Generates embeddings and stores them in Milvus for vector search

The web GUI will allow users to:
- Upload text documents for processing
- View processed documents and their structured information
- Perform semantic searches using various query types
- Visualize the extracted entities and relationships

## Directory Structure

```
rag-vector-search-web/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css           # Custom CSS styles
│   └── js/
│       └── main.js             # Frontend JavaScript
├── templates/                  # HTML templates
│   └── index.html              # Main interface template
├── uploads/                    # Directory for uploaded files
├── processed/                  # Directory for cleaned text files
├── structured/                 # Directory for structured JSON files
├── nlp_text_cleaner.py         # Renamed/adapted version of text-cleaner.py
├── text_structurer.py          # Renamed/adapted version of text-structurer.py
├── text_embedder.py            # Renamed/adapted version of text-embedder.py
└── README.md                   # This file
```

## Implementation Steps

### 1. Environment Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   Create a `requirements.txt` file with the following content:
   ```
   flask==2.3.3
   werkzeug==2.3.7
   spacy==3.7.2
   gensim==4.3.2
   pymilvus==2.3.1
   nltk==3.8.1
   pandas==2.1.1
   networkx==3.2.1
   tqdm==4.66.1
   numpy==1.26.0
   ```

   Then install the dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

### 2. Prepare the NLP Components

1. **Adapt the existing Python files**:
   - Copy `text-cleaner.py`, `text-structurer.py`, and `text-embedder.py` to the project directory
   - Rename them to `nlp_text_cleaner.py`, `text_structurer.py`, and `text_embedder.py` for better Python imports
   - Modify imports if necessary to ensure they work together

2. **Create the required directories**:
   ```bash
   mkdir -p uploads processed structured static/css static/js templates
   ```

### 3. Implement the Flask Application

1. **Create the main app.py file**:
   The main application file handles:
   - Flask application setup
   - File uploads and processing
   - API endpoints for search functionality
   - Document retrieval

   ```python
   import os
   import json
   import uuid
   from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
   from werkzeug.utils import secure_filename
   from nlp_text_cleaner import TextCleaner
   from text_structurer import TextStructurer
   from text_embedder import TextEmbedder

   # Initialize Flask app
   app = Flask(__name__)
   app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-for-session')
   app.config['UPLOAD_FOLDER'] = 'uploads'
   app.config['PROCESSED_FOLDER'] = 'processed'
   app.config['STRUCTURED_FOLDER'] = 'structured'
   app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}
   app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

   # Create necessary directories
   os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
   os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
   os.makedirs(app.config['STRUCTURED_FOLDER'], exist_ok=True)

   # Initialize NLP components
   text_cleaner = TextCleaner(remove_stopwords=True, lowercase=True, remove_punctuation=True, lemmatize=True)
   text_structurer = TextStructurer(spacy_model="en_core_web_sm", context_window=2)
   text_embedder = TextEmbedder(glove_model="glove-wiki-gigaword-300", milvus_host="localhost", milvus_port="19530")

   def allowed_file(filename):
       """Check if the file extension is allowed"""
       return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

   @app.route('/')
   def index():
       """Main page with upload form and search interface"""
       # Get list of processed files
       processed_files = []
       if os.path.exists(app.config['STRUCTURED_FOLDER']):
           processed_files = [f for f in os.listdir(app.config['STRUCTURED_FOLDER']) if f.endswith('.json')]
       
       return render_template('index.html', processed_files=processed_files)

   @app.route('/upload', methods=['POST'])
   def upload_file():
       """Handle file upload and processing through the NLP pipeline"""
       # Implement file upload functionality
       # Process through TextCleaner, TextStructurer, and TextEmbedder
       # Save results to appropriate directories
       # ...

   @app.route('/search', methods=['POST'])
   def search():
       """Perform vector search based on query"""
       # Implement search functionality based on query type
       # ...

   @app.route('/document/<filename>')
   def get_document(filename):
       """Get structured document details"""
       # Retrieve and return document data as JSON
       # ...

   if __name__ == '__main__':
       app.run(debug=True, host='0.0.0.0', port=5000)
   ```

### 4. Create the Frontend Templates

1. **Create the index.html template**:
   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>RAG Vector Search</title>
       <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
       <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
   </head>
   <body>
       <!-- Navbar -->
       <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
           <div class="container">
               <a class="navbar-brand" href="#">RAG Vector Search</a>
           </div>
       </nav>

       <div class="container mt-4">
           <!-- Flash messages -->
           <!-- Main content with upload form and search interface -->
           <!-- Document modal -->
       </div>

       <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
       <script src="{{ url_for('static', filename='js/main.js') }}"></script>
   </body>
   </html>
   ```

2. **Add CSS for styling**:
   Create `static/css/style.css` with custom styling for the application.

3. **Add JavaScript functionality**:
   Create `static/js/main.js` to handle:
   - Form submissions via AJAX
   - Search result display
   - Document detail view
   - UI interactions

### 5. Implement the Milvus Integration

1. **Set up Milvus**:
   - [Install Milvus](https://milvus.io/docs/install_standalone-docker.md) (standalone or distributed)
   - Configure the TextEmbedder to connect to your Milvus instance

2. **Test the connection**:
   - Ensure the TextEmbedder can connect to Milvus
   - Verify collection creation and vector storage

### 6. Implement File Processing Pipeline

1. **Complete the `upload_file` function**:
   ```python
   @app.route('/upload', methods=['POST'])
   def upload_file():
       if 'file' not in request.files:
           flash('No file part')
           return redirect(request.url)
       
       file = request.files['file']
       
       if file.filename == '':
           flash('No selected file')
           return redirect(request.url)
       
       if file and allowed_file(file.filename):
           # Generate a unique filename
           unique_id = str(uuid.uuid4())[:8]
           original_filename = secure_filename(file.filename)
           filename_base, filename_ext = os.path.splitext(original_filename)
           unique_filename = f"{filename_base}_{unique_id}{filename_ext}"
           
           # Save uploaded file
           upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
           file.save(upload_path)
           
           try:
               # Step 1: Clean the text
               with open(upload_path, 'r', encoding='utf-8', errors='replace') as f:
                   text = f.read()
               
               cleaned_text = text_cleaner.clean_text(text)
               
               # Save cleaned text
               cleaned_path = os.path.join(app.config['PROCESSED_FOLDER'], f"clean_{unique_filename}")
               with open(cleaned_path, 'w', encoding='utf-8') as f:
                   f.write(cleaned_text)
               
               # Step 2: Structure the text
               structured_data = text_structurer.process_text(cleaned_text)
               
               # Save structured data
               structured_path = os.path.join(app.config['STRUCTURED_FOLDER'], f"struct_{unique_filename}.json")
               with open(structured_path, 'w', encoding='utf-8') as f:
                   json.dump(structured_data, f, indent=2)
               
               # Step 3: Process with TextEmbedder and store in Milvus
               text_embedder.process_file(structured_path)
               
               flash(f'Successfully processed {original_filename}')
           except Exception as e:
               flash(f'Error processing file: {str(e)}')
               return redirect(url_for('index'))
           
           return redirect(url_for('index'))
       
       flash('File type not allowed')
       return redirect(url_for('index'))
   ```

### 7. Implement Search Functionality

1. **Complete the `search` function**:
   ```python
   @app.route('/search', methods=['POST'])
   def search():
       query = request.form.get('query', '')
       search_type = request.form.get('search_type', 'entity')
       limit = int(request.form.get('limit', 10))
       
       try:
           results = {}
           
           if search_type == 'entity':
               # Search for similar entities
               results = text_embedder.search_similar_entities(query, limit)
           elif search_type == 'statement':
               # Search for similar statements
               results = text_embedder.search_similar_statements(query, limit)
           elif search_type == 'relationship':
               # Parse the relationship components
               subject = request.form.get('subject', '')
               relation = request.form.get('relation', '')
               object_entity = request.form.get('object', '')
               
               if subject or relation or object_entity:
                   results = text_embedder.search_relationships(
                       subject=subject if subject else None,
                       relation=relation if relation else None,
                       obj=object_entity if object_entity else None,
                       limit=limit
                   )
               else:
                   # If no specific components provided, use the general query
                   results = text_embedder.search_relationships(
                       subject=query,
                       limit=limit
                   )
           
           return jsonify({"results": results})
       
       except Exception as e:
           return jsonify({"error": str(e)}), 500
   ```

### 8. Implement Document Retrieval

1. **Complete the `get_document` function**:
   ```python
   @app.route('/document/<filename>')
   def get_document(filename):
       try:
           file_path = os.path.join(app.config['STRUCTURED_FOLDER'], filename)
           if os.path.exists(file_path):
               with open(file_path, 'r', encoding='utf-8') as f:
                   data = json.load(f)
               return jsonify(data)
           else:
               return jsonify({"error": "File not found"}), 404
       except Exception as e:
           return jsonify({"error": str(e)}), 500
   ```

### 9. Testing and Debugging

1. **Run the application**:
   ```bash
   flask run --host=0.0.0.0 --port=5000
   ```

2. **Test each component**:
   - Upload a text file
   - Verify the cleaned and structured files are created
   - Check that entities are stored in Milvus
   - Test various search queries
   - View document details

### 10. Enhancements and Additional Features

1. **Visualizations**:
   - Add a network graph visualization for relationships using a JavaScript library like vis.js or D3.js
   - Create charts for entity distributions

2. **User Authentication**:
   - Add login/logout functionality
   - User-specific document collections

3. **Batch Processing**:
   - Allow uploading multiple files at once
   - Show processing progress

4. **Export Functionality**:
   - Export search results as CSV or JSON
   - Generate reports of document analysis

5. **Advanced Search Options**:
   - Filters for entity types
   - Time-based search
   - Combined queries

## Troubleshooting

### Common Issues

1. **Milvus Connection Problems**:
   - Ensure Milvus server is running
   - Check host and port configuration
   - Verify network connectivity

2. **File Processing Errors**:
   - Check file encoding (use UTF-8 when possible)
   - Look for memory issues with large files
   - Check permissions for writing to output directories

3. **Search Result Quality**:
   - Adjust embedding parameters
   - Try different GloVe models
   - Consider custom entity types for your domain

## Conclusion

This implementation provides a complete web-based interface for the RAG Vector Search system. By following these steps, you'll have a functional application that allows users to upload documents, process them through the NLP pipeline, and perform semantic searches using vector embeddings.

The modular architecture makes it easy to extend with additional features or adapt to specific domains by modifying the underlying NLP components.
