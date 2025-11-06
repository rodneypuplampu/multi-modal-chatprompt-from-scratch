# Lab 2: Traditional Offline RAG Pipeline

## Overview

This lab focuses on consolidating the components of an offline RAG system (as explored in Lab 1) into a structured, traditional Python script (`traditional_rag.py`). We will organize the ingestion, vectorization, exploration, and generation steps into reusable functions within a monolithic application structure.

## Architecture Overview

The traditional RAG pipeline follows this sequential flow:
1.  **INGESTION:** Documents → spaCy → Chunks (with overlap)
2.  **VECTORIZATION:** Chunks → SentenceTransformer → Embeddings
3.  **STORAGE:** Embeddings → LanceDB → Persistent vector store
4.  **EXPLORATION:** LanceDB data → BERTTopic → Theme discovery
5.  **RETRIEVAL:** Question → Embed → Search LanceDB → Top-K chunks
6.  **GENERATION:** Context + Question → Gemma → Answer

### Key Design Decisions

*   **spaCy:** For sentence-aware chunking.
*   **all-MiniLM-L6-v2:** For fast, local embeddings.
*   **LanceDB:** For serverless, local vector storage.
*   **BERTTopic:** For theme discovery.
*   **Gemma 2B:** For local, offline generation.

## Prerequisites

(Assumes the environment setup from Lab 1 is completed)

*   Python 3.10+ environment with required libraries installed (`lancedb`, `spacy`, `bertopic`, `sentence-transformers`, `transformers`, `accelerate`, `bitsandbytes`, `torch`, `numpy`, `pandas`).
*   spaCy model downloaded (`en_core_web_sm`).
*   Gemma model downloaded locally (e.g., `./gemma-2b-it`).
*   Documents located in `./data/`.

## Project Structure

```
enhanced_rag_lab/
├── .venv/
├── .lancedb/              # LanceDB storage
├── data/                  # Your documents
├── gemma-2b-it/          # Local Gemma model
└── traditional_rag.py    # The main pipeline script (this lab)
```

## Tutorial: Implementing `traditional_rag.py`

The following script implements the entire pipeline in a structured manner. Create a file named `traditional_rag.py` and add the code below.

```python
import spacy
import os
from typing import List, Dict
import lancedb
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np

# --- Configuration ---
DATA_DIR = "./data/"
LANCEDB_DIR = "./.lancedb"
TABLE_NAME = "memos"
GEMMA_MODEL_ID = "./gemma-2b-it" # !!! Update this path !!!
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 150
CHUNK_OVERLAP_SENTENCES = 2

# --- Global Initialization ---
print("Initializing models...")

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    exit()

# Initialize Embeddings and DB
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
db = lancedb.connect(LANCEDB_DIR)

# Initialize Gemma
use_quantization = torch.cuda.is_available()
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if use_quantization else None

try:
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
    model_gemma = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto" if use_quantization else None,
        torch_dtype=torch.bfloat16 if use_quantization else torch.float32
    )
    # If not using quantization but GPU is available, move manually if device_map wasn't used
    if not use_quantization and torch.cuda.is_available():
         model_gemma.to("cuda")
            
    print("Gemma model loaded.")
except OSError:
    print(f"Error: Gemma model not found at {GEMMA_MODEL_ID}.")
    model_gemma, tokenizer = None, None

# --- Task 2: Data Ingestion and Chunking Function ---

def load_and_chunk_docs(directory: str) -> List[Dict]:
    chunks = []
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return chunks

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                text = f.read()
                doc = nlp(text)
                sents = list(doc.sents)
                current_chunk = ""
                for i, sent in enumerate(sents):
                    if len(current_chunk.split()) > CHUNK_SIZE and i > 0:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "source": filename
                        })
                        # Create overlap
                        overlap = " ".join([s.text for s in sents[max(0,i-CHUNK_OVERLAP_SENTENCES):i]])
                        current_chunk = overlap + " " + sent.text
                    else:
                        current_chunk += " " + sent.text
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": filename
                    })
    print(f"Generated {len(chunks)} chunks.")
    return chunks

# Implementation Note: This approach ensures semantic coherence and context preservation via overlap.

# --- Task 3: Vectorization and Storage Function ---

def create_vector_store(chunks: List[Dict], table_name: str) -> lancedb.db.Table:
    if not chunks:
        print("No chunks to process.")
        return None

    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    data = []
    for i, chunk in enumerate(chunks):
        vector = embeddings[i]
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        data.append({
            "vector": vector,
            "text": chunk['text'],
            "source": chunk['source']
        })

    table = db.create_table(table_name, data=data, mode="overwrite")
    print(f"Vector store '{table_name}' created/updated.")
    return table

# --- Task 4: Data Exploration Function ---

def explore_themes(table: lancedb.db.Table):
    if table is None or len(table) == 0:
        print("Cannot explore themes. Table is empty or does not exist.")
        return

    # Retrieve all data
    all_data = table.to_pandas()
    texts = all_data["text"].tolist()
    # BERTTopic needs embeddings as a numpy array
    embeddings = np.array(all_data["vector"].tolist())

    topic_model = BERTTopic(verbose=False, calculate_probabilities=False)
    topic_model.fit_transform(texts, embeddings)
    topic_info = topic_model.get_topic_info()

    print("Top 10 Themes Discovered:")
    print(topic_info.head(10))

# Implementation Note: This provides a 'schema' for unstructured data.

# --- Task 5: RAG Pipeline Function ---

def ask_rag_pipeline(question: str, table: lancedb.db.Table):
    if model_gemma is None or tokenizer is None:
        return "Error: LLM (Gemma) is not loaded.", ""
    if table is None or len(table) == 0:
         return "Error: Knowledge base is empty.", ""

    # 1. RETRIEVAL
    query_vector = embedding_model.encode(question)
    results = table.search(query_vector).limit(5).to_pandas()

    context = ""
    for _, row in results.iterrows():
        context += f"Source: {row['source']}\nContent: {row['text']}\n\n"

    # 2. GENERATION
    prompt = f"<start_of_turn>user\nCONTEXT:\n{context}\n"
    prompt += f"QUESTION:\n{question}\n"
    prompt += "Based *only* on the CONTEXT, answer the QUESTION. If the answer is not in the context, state that.<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model_gemma.device)
    outputs = model_gemma.generate(**inputs, max_new_tokens=250, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    final_answer = answer.split("<start_of_turn>model")[-1].strip()
    return final_answer, context

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n--- Starting Traditional RAG Pipeline Execution ---")
    
    # 1. Ingestion
    chunks = load_and_chunk_docs(DATA_DIR)

    # 2. Vectorization and Storage
    if chunks:
        table = create_vector_store(chunks, TABLE_NAME)
        
        # 3. Exploration
        print("\n--- Exploring Themes ---")
        explore_themes(table)
        
        # 4. Querying
        print("\n--- Querying the Pipeline ---")
        question = "What are the risks associated with Project Alpha?"
        answer, context = ask_rag_pipeline(question, table)
        print(f"Q: {question}")
        print(f"A: {answer}")

        # Task 6: Debugging Output
        print("\n--- Debugging Context (Retrieved Chunks) ---")
        print(context)

    else:
        print("Pipeline execution halted due to lack of input data.")
```

## Implementation Notes and Debugging (Task 6)

### Running the Script

To run the lab, ensure the `traditional_rag.py` file is complete and execute it from your terminal:

```bash
python traditional_rag.py
```

### Debugging the Monolithic Pipeline

In this traditional structure, debugging requires inspecting the intermediate results, primarily the retrieved context.

**Scenario:** The answer provided by the LLM is incorrect or states the information is missing.

**Debug Process:**

1. **Examine the Retrieved Context:** Look at the output under `--- Debugging Context (Retrieved Chunks) ---`.

2. **Identify Root Cause:**
   - **Retrieval Issue:** If the context does not contain the relevant information, the vector search failed to find the correct chunks.
   - **Generation Issue:** If the context contains the correct information, but the LLM's answer is wrong, the LLM misinterpreted the prompt or the context.

3. **Fixes:**
   - **Retrieval:** Adjust `CHUNK_SIZE` or `CHUNK_OVERLAP_SENTENCES` in the configuration section, or increase the `.limit(5)` in `ask_rag_pipeline`.
   - **Generation:** Refine the prompt template in `ask_rag_pipeline` to be more explicit.
