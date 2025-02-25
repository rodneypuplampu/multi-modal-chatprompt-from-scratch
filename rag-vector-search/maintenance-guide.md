# Maintenance and Optimization Guide
## Prompt Engineering, Fine-tuning, Document Updates, and Version Control

This guide outlines the processes for maintaining and optimizing your NLP Vector Search RAG system over time. It covers prompt engineering, embedding fine-tuning, document updates, and version control best practices.

## Table of Contents
1. [Prompt Engineering Updates](#1-prompt-engineering-updates)
2. [Hyperparameter and Embedding Fine-tuning](#2-hyperparameter-and-embedding-fine-tuning)
3. [Document Updates and Management](#3-document-updates-and-management)
4. [Version Control and System Maintenance](#4-version-control-and-system-maintenance)
5. [Monitoring and Evaluation](#5-monitoring-and-evaluation)

---

## 1. Prompt Engineering Updates

Effective prompt engineering is crucial for maximizing the performance of your RAG system. Here's how to systematically improve your prompts over time:

### 1.1 Prompt Optimization Workflow

1. **Create a Prompt Library**
   - Establish a version-controlled repository of prompts
   - Organize by use case (e.g., entity extraction, relationship queries)
   - Document each prompt's purpose and performance metrics

2. **Implement A/B Testing for Prompts**
   ```python
   def test_prompt_variants(query, prompt_variants, test_cases, metrics_fn):
       """
       Test multiple prompt variants against test cases.
       
       Args:
           query (str): Base query
           prompt_variants (list): List of prompt templates to test
           test_cases (list): List of test inputs
           metrics_fn (callable): Function to evaluate results
           
       Returns:
           dict: Performance metrics for each variant
       """
       results = {}
       
       for variant_name, prompt_template in prompt_variants.items():
           variant_results = []
           
           for test_case in test_cases:
               # Format prompt with test case
               formatted_prompt = prompt_template.format(query=query, **test_case)
               
               # Get response from system
               response = search_engine.query(formatted_prompt)
               
               # Calculate metrics
               metrics = metrics_fn(response, test_case["expected"])
               variant_results.append(metrics)
           
           # Calculate average metrics
           avg_metrics = {
               metric: sum(r[metric] for r in variant_results) / len(variant_results)
               for metric in variant_results[0].keys()
           }
           
           results[variant_name] = avg_metrics
       
       return results
   ```

3. **Regular Prompt Review Cycle**
   - Schedule monthly or quarterly prompt reviews
   - Analyze user interactions and failure cases
   - Update prompts based on new data or domain changes

### 1.2 Prompt Engineering Techniques

#### Chain-of-Thought Prompting
Improve reasoning by breaking complex tasks into steps:

```
# Before
"Find entities related to financial transactions in this text."

# After
"Analyze this text in steps:
1. Identify all organizations mentioned.
2. For each organization, find associated monetary values.
3. Determine if there's an action (purchased, invested, etc.) connecting them.
4. Return all financial transaction entities with their relationships."
```

#### Few-Shot Learning Examples
Include examples directly in your prompts:

```
"Extract entity relationships from the text. Here are examples:

Input: 'Microsoft acquired GitHub in 2018.'
Output: {subject: 'Microsoft', relationship: 'acquired', object: 'GitHub', time: '2018'}

Input: 'Tim Cook is the CEO of Apple Inc.'
Output: {subject: 'Tim Cook', relationship: 'is CEO of', object: 'Apple Inc.'}

Now extract from: '{input_text}'"
```

#### Prompt Templates
Create standardized templates for different query types:

```python
ENTITY_EXTRACTION_TEMPLATE = """
Identify and extract all {entity_type} entities from the following text.
For each entity, provide:
1. The entity name
2. The entity type
3. Any attributes mentioned
4. The context surrounding the entity (3-5 words before and after)

Text: {input_text}
"""

RELATIONSHIP_QUERY_TEMPLATE = """
Find all relationships where {entity} is the {relationship_position}.
For each relationship, provide:
1. The complete relationship triple (subject-relation-object)
2. Confidence score
3. Source document reference
4. The exact text where this relationship was found

Entity: {entity}
Relationship position: {relationship_position} (subject/object)
"""
```

### 1.3 Prompt Version Control

Create a `prompts.json` file to track prompt versions:

```json
{
  "version": "1.3.0",
  "updated_at": "2025-02-15",
  "prompts": {
    "entity_extraction": {
      "v1": {
        "template": "Extract entities from: {text}",
        "created_at": "2024-10-10",
        "performance": {
          "precision": 0.78,
          "recall": 0.65
        },
        "deprecated": true
      },
      "v2": {
        "template": "Identify and extract all entities from the following text...",
        "created_at": "2025-01-15",
        "performance": {
          "precision": 0.85,
          "recall": 0.72
        },
        "active": true
      }
    }
  }
}
```

---

## 2. Hyperparameter and Embedding Fine-tuning

### 2.1 Vector Search Hyperparameter Optimization

#### Milvus Index Optimization
Test different index configurations:

```python
def optimize_milvus_index(collection_name, embedding_field, test_queries):
    """Test different index configurations for optimal performance."""
    index_configs = [
        {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
            "metric_type": "L2"
        },
        {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 4096},
            "metric_type": "L2"
        },
        {
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 500},
            "metric_type": "L2"
        },
        {
            "index_type": "HNSW",
            "params": {"M": 32, "efConstruction": 200},
            "metric_type": "L2"
        }
    ]
    
    results = {}
    
    for config in index_configs:
        # Create index with this config
        collection = Collection(collection_name)
        collection.drop_index()
        collection.create_index(embedding_field, config)
        collection.load()
        
        # Measure query time and recall
        query_times = []
        recall_rates = []
        
        for query in test_queries:
            start_time = time.time()
            results = collection.search(
                data=[query["vector"]],
                anns_field=embedding_field,
                param={"metric_type": config["metric_type"], "params": {"ef": 64}},
                limit=10
            )
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # Calculate recall compared to ground truth
            recall = calculate_recall(results[0], query["ground_truth"])
            recall_rates.append(recall)
        
        config_name = f"{config['index_type']}_{config['params']}"
        results[config_name] = {
            "avg_query_time": sum(query_times) / len(query_times),
            "avg_recall": sum(recall_rates) / len(recall_rates)
        }
    
    return results
```

#### Search Parameter Tuning
Create a script to optimize search parameters:

```python
def tune_search_parameters(collection_name, test_queries):
    """Find optimal search parameters for a collection."""
    collection = Collection(collection_name)
    collection.load()
    
    # Parameters to test
    nprobe_values = [8, 16, 32, 64, 128]  # For IVF indexes
    ef_values = [32, 64, 128, 256, 512]   # For HNSW indexes
    
    results = {}
    
    # Test for IVF indexes
    for nprobe in nprobe_values:
        search_params = {"metric_type": "L2", "params": {"nprobe": nprobe}}
        
        query_times = []
        recall_rates = []
        
        for query in test_queries:
            start_time = time.time()
            search_results = collection.search(
                data=[query["vector"]],
                anns_field="embedding",
                param=search_params,
                limit=10
            )
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            recall = calculate_recall(search_results[0], query["ground_truth"])
            recall_rates.append(recall)
        
        results[f"nprobe_{nprobe}"] = {
            "avg_query_time": sum(query_times) / len(query_times),
            "avg_recall": sum(recall_rates) / len(recall_rates)
        }
    
    # Test for HNSW indexes (if applicable)
    for ef in ef_values:
        search_params = {"metric_type": "L2", "params": {"ef": ef}}
        
        # Similar testing as above
        # ...
    
    return results
```

### 2.2 Embedding Model Fine-tuning

#### Switch to Domain-Specific Embeddings
When your general GloVe embeddings aren't capturing domain-specific nuances:

```python
# Install SentenceTransformers
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
import torch

def create_training_pairs(documents, entities, relationships):
    """Create training pairs from your structured data."""
    pairs = []
    
    # Create positive pairs (entities that are related)
    for relationship in relationships:
        pairs.append({
            "texts": [relationship["subject"], relationship["object"]],
            "label": 1.0  # Related
        })
    
    # Create some negative pairs (random unrelated entities)
    import random
    all_entities = [e["text"] for e in entities]
    for _ in range(len(relationships)):
        entity1 = random.choice(all_entities)
        entity2 = random.choice(all_entities)
        # Ensure they're not actually related
        while any(r["subject"] == entity1 and r["object"] == entity2 for r in relationships):
            entity2 = random.choice(all_entities)
        
        pairs.append({
            "texts": [entity1, entity2],
            "label": 0.0  # Unrelated
        })
    
    return pairs

def fine_tune_embedding_model(base_model="all-MiniLM-L6-v2", training_pairs=None, output_path="./domain-embeddings"):
    """Fine-tune a sentence transformer model on domain data."""
    # Load base model
    model = SentenceTransformer(base_model)
    
    # Prepare training data
    train_examples = [
        InputExample(texts=pair["texts"], label=pair["label"])
        for pair in training_pairs
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Use cosine similarity loss
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=4,
        warmup_steps=100,
        output_path=output_path
    )
    
    return model

# Example usage
training_pairs = create_training_pairs(documents, entities, relationships)
fine_tuned_model = fine_tune_embedding_model(training_pairs=training_pairs)
```

#### Update Embedding Generation
Replace GloVe with fine-tuned embeddings:

```python
# Previous GloVe embedding function
def get_glove_embedding(text):
    words = text.lower().split()
    word_vectors = [glove_model[word] for word in words if word in glove_model]
    if not word_vectors:
        return np.zeros(300)
    return np.mean(word_vectors, axis=0)

# New fine-tuned embedding function
def get_domain_embedding(text):
    model = SentenceTransformer('./domain-embeddings')
    return model.encode(text)

# Update the TextEmbedder class
class TextEmbedder:
    def __init__(self, embedding_model="domain"):
        # ...
        if embedding_model == "glove":
            self.get_embedding = get_glove_embedding
        elif embedding_model == "domain":
            self.get_embedding = get_domain_embedding
        # ...
```

### 2.3 Hyperparameter Tracking and Versioning

Use MLflow to track hyperparameter experiments:

```python
import mlflow
import mlflow.sklearn

def run_hyperparameter_experiment(param_grid, test_queries):
    """Run and track hyperparameter experiments."""
    mlflow.set_experiment("vector_search_optimization")
    
    for params in param_grid:
        with mlflow.start_run():
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Configure system with these parameters
            configure_system(params)
            
            # Run evaluation
            metrics = evaluate_system(test_queries)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Optional: log model or configuration
            mlflow.log_artifact("config.json")
```

---

## 3. Document Updates and Management

### 3.1 Incremental Document Processing

Create a system to process only new or changed documents:

```python
def process_new_documents(documents_dir, processed_log_file="processed_documents.json"):
    """
    Process only new or changed documents.
    
    Args:
        documents_dir: Directory containing documents
        processed_log_file: JSON file tracking processed documents
    """
    # Load log of processed documents
    if os.path.exists(processed_log_file):
        with open(processed_log_file, 'r') as f:
            processed_docs = json.load(f)
    else:
        processed_docs = {}
    
    # Scan directory for documents
    documents = []
    for filename in os.listdir(documents_dir):
        if not filename.endswith('.txt'):
            continue
            
        file_path = os.path.join(documents_dir, filename)
        file_stat = os.stat(file_path)
        last_modified = file_stat.st_mtime
        
        # Check if file is new or modified
        if filename not in processed_docs or processed_docs[filename]["last_modified"] < last_modified:
            documents.append({
                "filename": filename,
                "path": file_path,
                "last_modified": last_modified
            })
    
    if not documents:
        print("No new documents to process")
        return
    
    print(f"Processing {len(documents)} new or modified documents")
    
    # Process each document through the pipeline
    for doc in documents:
        # 1. Clean text
        cleaned_file = clean_text(doc["path"])
        
        # 2. Extract structure
        structured_file = extract_structure(cleaned_file)
        
        # 3. Generate embeddings
        embeddings_success = generate_embeddings(structured_file)
        
        if embeddings_success:
            # Update processed documents log
            processed_docs[doc["filename"]] = {
                "last_modified": doc["last_modified"],
                "last_processed": time.time(),
                "cleaned_file": cleaned_file,
                "structured_file": structured_file
            }
    
    # Save updated log
    with open(processed_log_file, 'w') as f:
        json.dump(processed_docs, f, indent=2)
    
    print(f"Successfully processed {len(documents)} documents")
```

### 3.2 Document Versioning

Implement document versioning in Milvus:

```python
def add_versioned_document(collection, doc_id, document, embeddings):
    """
    Add a versioned document to Milvus.
    
    Args:
        collection: Milvus collection
        doc_id: Document ID
        document: Document content and metadata
        embeddings: Document embeddings
    """
    # Get current timestamp
    timestamp = int(time.time())
    
    # Add version tag to document
    document["version"] = {
        "timestamp": timestamp,
        "version_id": f"{doc_id}_v{timestamp}"
    }
    
    # Insert into Milvus
    collection.insert([
        doc_id,
        document["text"],
        document["source"],
        json.dumps(document["metadata"]),
        json.dumps(document["version"]),
        embeddings
    ])
```

### 3.3 Document Update Pipeline

Create a script for the complete document update pipeline:

```python
def document_update_pipeline(config_file="pipeline_config.json"):
    """
    Run the complete document update pipeline.
    
    Args:
        config_file: Configuration file path
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 1. Find new/modified documents
    new_docs = process_new_documents(
        config["documents_dir"],
        config["processed_log_file"]
    )
    
    if not new_docs:
        return
    
    # 2. Run cleaning pipeline
    cleaner = TextCleaner(
        remove_stopwords=config["cleaning"]["remove_stopwords"],
        lowercase=config["cleaning"]["lowercase"],
        remove_punctuation=config["cleaning"]["remove_punctuation"],
        stem=config["cleaning"]["stem"],
        lemmatize=config["cleaning"]["lemmatize"]
    )
    
    cleaned_docs = []
    for doc in new_docs:
        cleaned_file = cleaner.process_file(
            doc["path"],
            config["output_dirs"]["cleaned"]
        )
        if cleaned_file:
            cleaned_docs.append({
                "original": doc,
                "cleaned": cleaned_file
            })
    
    # 3. Run structuring pipeline
    structurer = TextStructurer(
        spacy_model=config["structuring"]["spacy_model"],
        context_window=config["structuring"]["context_window"]
    )
    
    structured_docs = []
    for doc in cleaned_docs:
        structured_file = structurer.process_file(
            doc["cleaned"],
            config["output_dirs"]["structured"]
        )
        if structured_file:
            structured_docs.append({
                "original": doc["original"],
                "cleaned": doc["cleaned"],
                "structured": structured_file
            })
    
    # 4. Run embedding pipeline
    embedder = TextEmbedder(
        embedding_model=config["embedding"]["model"],
        milvus_host=config["milvus"]["host"],
        milvus_port=config["milvus"]["port"]
    )
    
    for doc in structured_docs:
        embedder.process_file(doc["structured"])
    
    # 5. Update document log
    update_document_log(
        config["processed_log_file"],
        structured_docs
    )
    
    # 6. Rebuild search indices if needed
    if config["rebuild_indices"]:
        rebuild_search_indices()
    
    print(f"Successfully updated {len(structured_docs)} documents")
```

---

## 4. Version Control and System Maintenance

### 4.1 Configuration Management

Create a centralized configuration system:

```python
import yaml
import os
from pathlib import Path

class ConfigManager:
    def __init__(self, config_dir="./config"):
        self.config_dir = Path(config_dir)
        self.config_files = {
            "system": self.config_dir / "system.yaml",
            "pipeline": self.config_dir / "pipeline.yaml",
            "embedding": self.config_dir / "embedding.yaml",
            "search": self.config_dir / "search.yaml",
            "milvus": self.config_dir / "milvus.yaml",
            "prompts": self.config_dir / "prompts.yaml"
        }
        self.config = self._load_all_configs()
    
    def _load_config_file(self, file_path):
        """Load a single config file."""
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_all_configs(self):
        """Load all configuration files."""
        config = {}
        
        for section, file_path in self.config_files.items():
            config[section] = self._load_config_file(file_path)
        
        return config
    
    def get_config(self, section=None, key=None):
        """Get configuration value."""
        if section is None:
            return self.config
        
        if key is None:
            return self.config.get(section, {})
        
        return self.config.get(section, {}).get(key)
    
    def save_config(self, section, key=None, value=None, config_dict=None):
        """Update configuration."""
        if section not in self.config:
            self.config[section] = {}
        
        if config_dict:
            self.config[section].update(config_dict)
        elif key and value is not None:
            self.config[section][key] = value
        
        # Save to file
        file_path = self.config_files.get(section)
        if file_path:
            with open(file_path, 'w') as f:
                yaml.dump(self.config[section], f, default_flow_style=False)
```

Example configuration files:

`pipeline.yaml`:
```yaml
version: 1.3.0
input_dir: ./documents
output_dirs:
  cleaned: ./output/cleaned
  structured: ./output/structured
processors:
  cleaner:
    remove_stopwords: true
    lowercase: true
    remove_punctuation: true
    stem: false
    lemmatize: true
  structurer:
    spacy_model: en_core_web_sm
    context_window: 2
  embedder:
    model: domain
    vector_dim: 384
```

### 4.2 Code Versioning

Set up proper Git versioning with semantic versioning:

```
# .gitignore
__pycache__/
*.py[cod]
*$py.class
venv/
.env
*.log
data/
output/
mlruns/
.ipynb_checkpoints/
node_modules/

# .git-commit-template
# [type](scope): subject
#
# [optional body]
#
# [optional footer]
#
# Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore
# Example: feat(embeddings): add support for domain-specific model fine-tuning
```

Set up your Git commit template:
```bash
git config --global commit.template .git-commit-template
```

### 4.3 Database Schema Versioning

Track Milvus schema changes:

```python
def update_milvus_schema(version_file="milvus_schema_version.json"):
    """
    Update Milvus schema with versioning.
    
    Args:
        version_file: File tracking schema versions
    """
    # Load current schema version
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            version_info = json.load(f)
            current_version = version_info["version"]
    else:
        current_version = "0.0.0"
    
    # Define schema upgrades
    schema_versions = [
        {
            "version": "1.0.0",
            "changes": [
                # Initial schema
                create_initial_schema
            ]
        },
        {
            "version": "1.1.0",
            "changes": [
                # Add confidence score field
                add_confidence_field,
                # Add version tracking fields
                add_version_fields
            ]
        },
        {
            "version": "1.2.0",
            "changes": [
                # Add additional metadata fields
                add_domain_fields,
                # Update index types
                update_entity_index
            ]
        }
    ]
    
    # Find versions to apply
    versions_to_apply = []
    found_current = False
    
    for version in schema_versions:
        if found_current:
            versions_to_apply.append(version)
        elif version["version"] == current_version:
            found_current = True
    
    # Apply upgrades
    for version in versions_to_apply:
        print(f"Upgrading Milvus schema to version {version['version']}")
        
        for change_fn in version["changes"]:
            change_fn()
        
        # Update version file
        with open(version_file, 'w') as f:
            json.dump({
                "version": version["version"],
                "updated_at": time.time()
            }, f, indent=2)
        
        print(f"Successfully upgraded to version {version['version']}")
```

### 4.4 System Backup and Restore

Create scripts for system backup and restore:

```python
def backup_system(backup_dir="./backups"):
    """
    Create a complete system backup.
    
    Args:
        backup_dir: Directory to store backups
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
    os.makedirs(backup_path, exist_ok=True)
    
    # 1. Backup configuration
    config_backup_path = os.path.join(backup_path, "config")
    os.makedirs(config_backup_path, exist_ok=True)
    shutil.copytree("./config", config_backup_path, dirs_exist_ok=True)
    
    # 2. Backup processed documents log
    shutil.copy2("./processed_documents.json", backup_path)
    
    # 3. Backup Milvus data (if self-hosted)
    # This varies depending on your Milvus setup
    # For Docker-based Milvus, you might need to create a volume snapshot
    
    # 4. Backup any fine-tuned models
    if os.path.exists("./domain-embeddings"):
        shutil.copytree("./domain-embeddings", 
                       os.path.join(backup_path, "domain-embeddings"),
                       dirs_exist_ok=True)
    
    # 5. Create backup metadata
    with open(os.path.join(backup_path, "backup_info.json"), 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "version": get_system_version(),
            "components": {
                "config": True,
                "documents_log": True,
                "milvus": False,  # Manual backup required
                "models": os.path.exists("./domain-embeddings")
            }
        }, f, indent=2)
    
    print(f"Backup created at {backup_path}")
    return backup_path
```

---

## 5. Monitoring and Evaluation

### 5.1 Performance Metrics Tracking

Create a metrics tracking system:

```python
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime

class MetricsTracker:
    def __init__(self, metrics_file="metrics.json"):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics."""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "search_performance": [],
                "embedding_quality": [],
                "system_usage": []
            }
    
    def _save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def add_search_metrics(self, query_count, avg_query_time, avg_recall, precision=None):
        """Add search performance metrics."""
        self.metrics["search_performance"].append({
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query_count": query_count,
            "avg_query_time_ms": avg_query_time,
            "avg_recall": avg_recall,
            "precision": precision
        })
        self._save_metrics()
    
    def add_embedding_metrics(self, model_name, avg_cosine_similarity, cluster_separation):
        """Add embedding quality metrics."""
        self.metrics["embedding_quality"].append({
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "avg_cosine_similarity": avg_cosine_similarity,
            "cluster_separation": cluster_separation
        })
        self._save_metrics()
    
    def add_usage_metrics(self, daily_queries, unique_users, resource_usage):
        """Add system usage metrics."""
        self.metrics["system_usage"].append({
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "daily_queries": daily_queries,
            "unique_users": unique_users,
            "resource_usage": resource_usage
        })
        self._save_metrics()
    
    def plot_search_performance(self, metric="avg_query_time_ms"):
        """Plot search performance trend."""
        dates = [m["date"] for m in self.metrics["search_performance"]]
        values = [m[metric] for m in self.metrics["search_performance"]]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, marker='o')
        plt.title(f"Search Performance: {metric}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_embedding_quality(self):
        """Plot embedding quality metrics."""
        dates = [m["date"] for m in self.metrics["embedding_quality"]]
        similarities = [m["avg_cosine_similarity"] for m in self.metrics["embedding_quality"]]
        separations = [m["cluster_separation"] for m in self.metrics["embedding_quality"]]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, similarities, marker='o', label="Avg Cosine Similarity")
        plt.plot(dates, separations, marker='s', label="Cluster Separation")
        plt.title("Embedding Quality Metrics")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
```

### 5.2 Automated Evaluation System

Create a comprehensive evaluation script:

```python
def evaluate_system(test_set_file="test_queries.json", output_file="evaluation_results.json"):
    