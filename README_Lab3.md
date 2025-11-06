# Lab 3: Agentic Enhancement with ADK

## Overview

This lab transforms the traditional RAG pipeline (Lab 2) into an intelligent, agent-controlled service using Google's Agent Development Kit (ADK). We shift from a monolithic, sequential script to an event-driven system where an AI agent orchestrates tools via natural language commands.

**Transformation Goal:**
*   **Before (Traditional):** Monolithic script, manual debugging, rigid flow.
*   **After (Agentic):** Persistent agent, automatic visibility (ADK UI), natural language control, dynamic knowledge management.

## Prerequisites

*   Python 3.10+ environment with RAG dependencies (see Lab 1/2).
*   **Ollama:** Required to serve the Gemma model locally.
*   **ADK and LiteLLM:** Required for the agent framework and the offline bridge.

### Setup

1.  **Install ADK and LiteLLM:**
    (Activate your existing virtual environment)
    ```bash
    pip install google-adk litellm
    ```

2.  **Install and Configure Ollama:**
    Ollama serves the Gemma model locally, enabling offline operation for the agent.
    *   Download Ollama from [https://ollama.com/](https://ollama.com/).
    *   Pull the Gemma model:
      ```bash
      ollama pull gemma:2b-it
      ```

## Project Structure

We refactor the project to separate the RAG logic (tools) from the agent definition.

```
enhanced_rag_lab/
├── .venv/
├── .lancedb/
├── data/
├── rag_tools.py      # RAG functions refactored as tools (New)
└── agent.py          # ADK agent definition (New)
```

## Tutorial: Step-by-Step Guide

### Task 7: Creating the RAG Tools Library (`rag_tools.py`)

We refactor the pipeline functions into discrete tools. The agent will use these functions to interact with the knowledge base.

**(Create `rag_tools.py`)**

```python
import spacy
import os
import lancedb
from sentence_transformers import SentenceTransformer
from bertopic import BERTTopic
from typing import Dict, List
import numpy as np
import pandas as pd

# --- Configuration ---
LANCEDB_DIR = "./.lancedb"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 150
CHUNK_OVERLAP_SENTENCES = 2

# --- Global Initialization (Tools share these resources) ---
try:
    nlp = spacy.load("en_core_web_sm")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    db = lancedb.connect(LANCEDB_DIR)
except Exception as e:
    print(f"Error during tool initialization: {e}")

# --- Helper Function (Internal use, not exposed as a tool) ---
# We include the full implementation here so the tools file is standalone

def _load_and_chunk_docs(directory: str) -> List[Dict]:
    chunks = []
    if not os.path.exists(directory):
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
                        chunks.append({"text": current_chunk.strip(), "source": filename})
                        overlap = " ".join([s.text for s in sents[max(0, i-CHUNK_OVERLAP_SENTENCES):i]])
                        current_chunk = overlap + " " + sent.text
                    else:
                        current_chunk += " " + sent.text
                        
                if current_chunk.strip():
                    chunks.append({"text": current_chunk.strip(), "source": filename})
    return chunks

# --- ADK TOOLS (Exposed to the Agent) ---

def initialize_rag_datastore(directory: str, table_name: str = "memos") -> Dict:
    """Ingests documents from the specified directory, vectorizes them, 
    and stores them in the specified LanceDB table (overwrites existing table)."""
    try:
        chunks = _load_and_chunk_docs(directory)
        if not chunks:
            return {"status": "failure", "reason": f"No documents found in {directory}"}

        texts = [c['text'] for c in chunks]
        embeddings = embedding_model.encode(texts)
        
        data = []
        for i, c in enumerate(chunks):
            vector = embeddings[i]
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            data.append({"vector": vector, "text": c['text'], "source": c['source']})

        db.create_table(table_name, data=data, mode="overwrite")
        return {"status": "success", "chunks_processed": len(chunks), "table": table_name}
    except Exception as e:
        return {"status": "error", "exception": str(e)}

def retrieve_context_for_question(question: str, table_name: str = "memos") -> Dict:
    """Retrieves the top 5 relevant text chunks from the specified LanceDB table 
    based on a semantic search of the question."""
    try:
        tbl = db.open_table(table_name)
    except (FileNotFoundError, Exception) as e:
        return {"status": "error", "reason": f"Table '{table_name}' not found. Please ingest data first. Details: {e}"}

    query_vector = embedding_model.encode(question)
    results = tbl.search(query_vector).limit(5).to_pandas()
    
    context = ""
    for _, r in results.iterrows():
         text_content = str(r['text']) if 'text' in r else ""
         source_info = str(r['source']) if 'source' in r else "Unknown"
         context += f"Source: {source_info}\n{text_content}\n\n"
             
    return {"status": "success", "context_string": context}

def explore_datastore_themes(table_name: str = "memos") -> Dict:
    """Analyzes the documents in the specified LanceDB table using BERTTopic 
    to identify and list the main themes/topics."""
    try:
        tbl = db.open_table(table_name)
        if len(tbl) == 0:
            return {"status": "failure", "reason": "Table is empty."}

        all_data = tbl.to_pandas()
        texts = all_data["text"].tolist()
        # BERTTopic expects numpy array for embeddings
        embeddings = np.array(all_data["vector"].tolist())

        topic_model = BERTTopic(verbose=False, calculate_probabilities=False)
        topic_model.fit_transform(texts, embeddings)
        topic_info = topic_model.get_topic_info()

        # Format output for the agent
        themes = topic_info[['Topic', 'Count', 'Name']].head(10).to_dict(orient='records')
        return {"status": "success", "themes": themes}
        
    except (FileNotFoundError, Exception) as e:
        return {"status": "error", "reason": f"Table '{table_name}' not found or error occurred. Details: {e}"}
```

### Task 8 & 9: Configuring the Offline Bridge and Building the Agent (`agent.py`)

We use LiteLLM to bridge ADK to Ollama, which serves the local Gemma model.

**Architecture:** ADK → LiteLLM → Ollama → Gemma

We define the agent, connect it to the local LLM via the offline bridge, and register the tools.

**(Create `agent.py`)**

```python
from google.adk.agents.llm_agent import Agent
# Import LiteLlm for the offline bridge
from google.adk.models.lite_llm import LiteLlm
# Import the tools library
import rag_tools

# --- Offline Bridge Configuration (Task 8) ---
# Connect ADK to Ollama (running by default on port 11434) via LiteLLM
try:
    local_gemma_model = LiteLlm(
        model="ollama_chat/gemma:2b-it",
        # Ensure Ollama is running at this address
        api_base="http://localhost:11434" 
    )
except Exception as e:
    print(f"Error initializing LiteLLM bridge: {e}")
    print("Ensure 'litellm' is installed.")
    exit()

# --- Agent Instruction (Meta-Prompt) (Task 9) ---
# This governs how the agent behaves and utilizes its tools.
RAG_INSTRUCTION = """
You are an intelligent RAG assistant managing a corporate knowledge base. 
Your goal is to manage the data and answer user questions accurately based ONLY on the stored knowledge.

RULES OF OPERATION:
1. **Knowledge Management:** If the user asks to ingest, load, or update data, use the `initialize_rag_datastore` tool.
2. **Exploration:** If the user asks about the contents or themes of the knowledge base, use the `explore_datastore_themes` tool.
3. **Question Answering (CRITICAL):** For ALL factual questions about the corporate data:
    a. FIRST, you MUST call `retrieve_context_for_question` to get the relevant information.
    b. SECOND, use the `context_string` returned by the tool to formulate your answer.
    c. You MUST answer *ONLY* based on the information present in the `context_string`.
    d. If the `context_string` does not contain the necessary information, you MUST state clearly: "I cannot answer this question based on the provided context." Do not hallucinate.
4. **Error Handling:** If a tool returns an error (e.g., table not found), inform the user and suggest the appropriate action (e.g., ingesting data).
"""

# --- Agent Definition ---
root_agent = Agent(
    model=local_gemma_model,
    name="rag_agent",
    description="Offline Agentic RAG assistant for corporate memos.",
    instruction=RAG_INSTRUCTION,
    # Register the tools from the library
    tools=[
        rag_tools.initialize_rag_datastore,
        rag_tools.explore_datastore_themes,
        rag_tools.retrieve_context_for_question
    ]
)

if __name__ == "__main__":
    print("RAG Agent configured. Use 'adk web agent.py' to start the interface.")
```

*Implementation Note: The RAG_INSTRUCTION (Meta-Prompt) is crucial. It enforces a specific chain-of-thought (Retrieve then Generate), ensuring answers are grounded in the knowledge base and reducing hallucinations.*

### Task 10: Running the Agent

To run the agent, you need two terminal sessions.

#### Terminal 1: Start Ollama

Ensure Ollama is running and serving the Gemma model.

```bash
ollama serve
```

#### Terminal 2: Start ADK Web UI

Navigate to your project directory, activate the environment, and start the ADK web interface.

```bash
source .venv/bin/activate  # Or your environment activation command
adk web agent.py --port 8000
```

#### Interact:

Open your web browser and navigate to `http://localhost:8000`.

### Example Interactions

**Ingestion:**
- **USER:** "Ingest all documents from ./data/ into the 'memos' table."
- **AGENT:** [Calls `initialize_rag_datastore`] "Success! Processed X chunks and stored them in the 'memos' table."

**Exploration:**
- **USER:** "What topics are covered in the memos?"
- **AGENT:** [Calls `explore_datastore_themes`] "The main themes are: Project Alpha risks, Q4 revenue projections..."

**Question Answering:**
- **USER:** "What are the main risks for Project Alpha?"
- **AGENT:** [Calls `retrieve_context_for_question`] "Based on the memos, the main risks are..."

### Task 11: Advanced Debugging (Agentic Visibility)

The ADK Web UI provides automatic visibility into the agent's thought process, tool calls, and results, making debugging significantly easier than the traditional pipeline.

**Debugging using the ADK UI Log:**

If the agent provides an incorrect answer, you can inspect the log in the Web UI to see:

1. **Agent Decision:** Which tool did the agent choose?
2. **Tool Result:** What `context_string` was returned by the retrieval tool?
3. **Agent Conclusion:** How did the agent interpret that context?

This allows you to immediately diagnose whether the issue was a failure in **Retrieval** (wrong context found) or **Generation** (agent misinterpreted the context).

## Comparison: Traditional vs Agentic

| Aspect | Traditional (Lab 2) | Agentic (Lab 3) |
|--------|---------------------|-----------------|
| **Control** | Code-driven, sequential | Natural language commands |
| **Debugging** | Manual print statements | Automatic UI visibility |
| **Flexibility** | Rigid flow | Dynamic tool selection |
| **User Interface** | Terminal/CLI | Web-based chat |
| **Error Handling** | Manual | Agent-assisted |
| **Extensibility** | Requires code changes | Add new tools easily |
