# Lab 1: Offline RAG Standalone Machine

## Overview

In this lab, you will learn how to build a complete, offline-first Retrieval-Augmented Generation (RAG) pipeline. This solution runs entirely on your local machine, ensuring data privacy and removing reliance on external APIs.

We will use the following technologies:
*   **LanceDB:** A high-performance, local vector database.
*   **spaCy:** Robust document processing and chunking.
*   **BERTTopic:** Data exploration and theme discovery.
*   **Sentence-Transformers:** Generating vector embeddings.
*   **Gemma (via Hugging Face Transformers):** The local, offline Large Language Model (LLM).

## Objectives

*   Set up the necessary Python environment.
*   Use spaCy for intelligent, sentence-aware document chunking.
*   Generate embeddings and store them efficiently in LanceDB.
*   Analyze the document corpus to identify key themes using BERTTopic.
*   Build a RAG pipeline to retrieve context and generate answers using a local Gemma model.
*   Debug and refine the RAG pipeline.

## Prerequisites

*   Python 3.10+ environment.
*   Familiarity with Python.
*   Sufficient local compute (A GPU is recommended for Gemma, but not strictly required).
*   Downloaded Gemma model weights (e.g., `gemma-2b-it`).
*   A collection of unstructured documents (e.g., `.txt` files) placed in a `./data/` directory.

## Tutorial: Step-by-Step Guide

This tutorial guides you through the implementation. The code snippets can be run sequentially in a Python script or Jupyter Notebook.

### Task 1: Setup and Environment Configuration

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv rag_lab_env
    source rag_lab_env/bin/activate  # On Windows use `rag_lab_env\Scripts\activate`
    ```

2.  **Install Required Libraries:**
    ```bash
    pip install lancedb spacy bertopic sentence-transformers transformers accelerate bitsandbytes torch numpy pandas
    ```

3.  **Download spaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

4.  **Prepare Data and Model Paths:**
    Ensure your documents are in `./data/` and your Gemma model is accessible locally (e.g., `./gemma-2b-it/`).

### Task 2: Data Ingestion and Chunking (spaCy)

We use spaCy's sentence segmenter to create semantically meaningful chunks with overlap.

*Implementation Note: Sentence-based chunking preserves context better than fixed-size character splits. Overlap ensures context isn't lost at chunk boundaries.*

```python
import spacy
import os

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()

def load_and_chunk_docs(directory):
    chunks = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}. Please create it and add documents.")
        return chunks

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                text = f.read()
                doc = nlp(text)
                sents = list(doc.sents)

                current_chunk = ""
                for i, sent in enumerate(sents):
                    # Target chunk size: ~150 words
                    if len(current_chunk.split()) > 150 and i > 0:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "source": filename
                        })
                        # Create overlap (last 2 sentences)
                        overlap_sents = " ".join([s.text for s in sents[max(0, i-2):i]])
                        current_chunk = overlap_sents + " " + sent.text
                    else:
                        current_chunk += " " + sent.text

                # Add the last chunk
                if current_chunk.strip():
                     chunks.append({
                        "text": current_chunk.strip(),
                        "source": filename
                    })
    return chunks

# Process documents
document_chunks = load_and_chunk_docs("./data/")
print(f"Processed {len(document_chunks)} chunks.")
