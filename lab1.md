Lab 1: Offline RAG Standalone MachineFilename: README_Lab1.mdMarkdown# Lab 1: Offline RAG Standalone Machine

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
Task 3: Vectorization and Storage (LanceDB)Vectorize the chunks using Sentence Transformers and store them in LanceDB for fast retrieval.Pythonimport lancedb
from sentence_transformers import SentenceTransformer
import numpy as np

if not document_chunks:
    print("No documents processed. Exiting.")
    # In a notebook/interactive session, you might not exit, but proceed if possible
else:
    # 1. Initialize Embedding Model
    # 'all-MiniLM-L6-v2' is chosen for its balance of speed and accuracy locally.
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Generate Embeddings
    texts = [chunk['text'] for chunk in document_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # 3. Create and Populate LanceDB Table
    db = lancedb.connect("./.lancedb") # Connects to a local directory

    data_to_add = []
    for i, chunk in enumerate(document_chunks):
        # Ensure vector is a list for LanceDB compatibility
        vector = embeddings[i]
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        data_to_add.append({
            "vector": vector,
            "text": chunk['text'],
            "source": chunk['source']
        })

    # Create table (mode="overwrite" is useful for labs to ensure a clean start)
    tbl = db.create_table("memos", data=data_to_add, mode="overwrite")

    print(f"LanceDB table 'memos' now contains {len(tbl)} records.")
Task 4: Exploring Data Themes (BERTTopic)Use BERTTopic to discover the main topics in the unstructured data, similar to reviewing a database schema.Pythonfrom bertopic import BERTopic
import numpy as np

if document_chunks:
    # Reuse existing data
    texts = [d['text'] for d in data_to_add]
    # BERTTopic expects embeddings as a numpy array
    embeddings_for_topic = np.array([d['vector'] for d in data_to_add])

    # 1. Initialize and Fit Model
    # Setting calculate_probabilities=False can speed up processing
    topic_model = BERTopic(verbose=True, calculate_probabilities=False)
    topics, _ = topic_model.fit_transform(texts, embeddings_for_topic)

    # 2. Review Topics
    print("Top 10 most frequent topics:")
    print(topic_model.get_topic_info().head(10))
Task 5: Building the RAG Pipeline (Retrieval & Generation)Load the local Gemma model and define the RAG function.1. Load Local Gemma Model:Implementation Note: We configure quantization (load_in_4bit) using BitsAndBytesConfig if a GPU is available to significantly reduce memory footprint.Pythonimport torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# !!! UPDATE THIS PATH to your local Gemma directory !!!
model_id = "./gemma-2b-it"

# Configure quantization if CUDA is available
use_quantization = torch.cuda.is_available()
bnb_config = None

if use_quantization:
    print("CUDA detected. Loading model with 4-bit quantization.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
else:
    print("CUDA not detected. Loading model in full precision (requires significant RAM).")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # device_map="auto" helps utilize the GPU efficiently when quantized
    model_gemma = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if use_quantization else None,
        torch_dtype=torch.bfloat16 if use_quantization else torch.float32
    )
     # If not using quantization but GPU is available, move manually if device_map wasn't used
    if not use_quantization and torch.cuda.is_available():
         model_gemma.to("cuda")
            
    print("Gemma model loaded successfully.")
except OSError:
    print(f"Error loading Gemma model. Ensure the path '{model_id}' is correct.")
    model_gemma = None
except ImportError:
    print("Error: Missing required libraries for quantization. Please install 'accelerate' and 'bitsandbytes'.")
    model_gemma = None
2. Create the RAG Function:Implementation Note: The prompt template explicitly instructs the model to rely ONLY on the context, which is crucial for RAG accuracy and preventing hallucinations.Pythondef ask_rag_pipeline(question: str, table, embedding_model):
    if model_gemma is None:
        return "Error: Gemma model is not loaded.", ""
```
    # 1. Retrieve Context (LanceDB)
    query_vector = embedding_model.encode(question)

    # Search the table (Top 5 results) and convert to a list of dictionaries
    results = table.search(query_vector).limit(5).to_list()
    
    context = ""
    for res in results:
        # Use .get() for safe access
        source = res.get('source', 'N/A')
        text_content = res.get('text', '')
        context += f"Source: {source}\nContent: {text_content}\n\n"

    # 2. Build the Prompt for Gemma (Using Gemma's specific chat template)
    prompt_template = f"""<start_of_turn>user
CONTEXT:
{context}

QUESTION:
{question}

Based *only* on the provided CONTEXT, answer the QUESTION. If the context does not contain the answer, state that the information is not available.<end_of_turn>
<start_of_turn>model
"""
    # 3. Generate Answer (Gemma)
    inputs = tokenizer(prompt_template, return_tensors="pt").to(model_gemma.device)
    # do_sample=False for deterministic results in RAG
    outputs = model_gemma.generate(**inputs, max_new_tokens=250, do_sample=False)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean the output to only show the model's response
    final_answer = answer.split("<start_of_turn>model")[-1].strip()

    return final_answer, context

# 3. Run the Pipeline (Using 'tbl' and 'model' from Task 3)
if document_chunks and model_gemma:
    question = "What are the risks associated with Project Alpha's deadline?"
    # Ensure you have data related to "Project Alpha" in your ./data/ directory for a meaningful answer.
    answer, retrieved_context = ask_rag_pipeline(question, tbl, model)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
Task 6: Debugging and Refining the PipelineIf the RAG pipeline provides a poor answer (e.g., "information is not available" when you know it exists), analyze the retrieved context.Analyze the Context: Inspect the retrieved_context variable.Pythonif 'retrieved_context' in locals():
    print("\n--- RETRIEVED CONTEXT FOR DEBUGGING ---")
    print(retrieved_context)
Identify the Problem and Refine:If the context lacks the answer (Retrieval Issue): The vector search didn't find the right chunks.Refinement: Increase the retrieval limit in Task 5 (e.g., change .limit(5) to .limit(10)).Refinement: Increase the chunk overlap or adjust the chunk size in Task 2.If the context has the answer but the model missed it (Generation Issue): The LLM failed to extract the information.Refinement: Improve the prompt template in Task 5 to be more explicit or clearer.Data Issue: The information genuinely does not exist in the ./data/ directory.Appendix: Top 10 RAG Techniques and Mappings#Natural Language GoalKey TechniqueConcept/Implementation1"Find me all information about Project Alpha."Vector Search (LanceDB)tbl.search(query_vector).limit(k)2"Find info on 'Project Alpha' from the 'engineering' memo only."Vector Search with Pre-filteringtbl.search(query_vector).where("source = 'engineering.txt'").limit(k)3My document chunks are too small and miss context.Sentence Window / Overlapping Chunks (spaCy)Chunking strategy to include surrounding sentences (Task 2).4How do I know what's in my documents?Topic Modeling (BERTTopic)Unsupervised clustering to identify key themes (Task 4).5The answer is in the context, but Gemma ignores it.Prompt EngineeringModifying the prompt template (e.g., "Answer only based on the context...") (Task 5).6I need to find exact keywords like a 'SKU-123'.Hybrid SearchCombining vector search (semantic) with keyword search (e.g., BM25).7My answers are good but very slow.Model Quantization (Gemma)Loading the model in 4-bit or 8-bit (e.g., using BitsAndBytesConfig) (Task 5).8The retrieved chunks are too long.Text Chunking (spaCy)Reducing the chunk size in Task 2.9How do I make sure the answer is from a specific source?Metadata RetrievalEnsuring the source (filename) is stored in LanceDB and returned with the text.10How do I run this without an internet connection?Local Models (Gemma, MiniLM)Using fully downloaded models.
