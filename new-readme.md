The system consists of four main components that work together:

1. **Text Cleaning Utility**: Normalizes and preprocesses raw text
2. **Text Structuring Tool**: Extracts entities and relationships with context
3. **Text Embedding Service**: Vectorizes structured data and stores it in Milvus
4. **Search Application**: Enables natural language queries against the vector database

## How the Components Work Together

### Data Processing Pipeline
```mermaid
flowchart TD
%% Query Initialization & Analysis Phase
A[User Natural Language Query Direct/Multi-Step/Complex] --> B[spaCy NLP Processing Parse entities, syntax, intent]
B --> C{Analysis Complete?}
C -->|No| D[spaCy: Expand semantic analysis Enhanced entities & context]
D --> E[TensorFlow: Build reasoning graph Identify logic gaps]
E --> F[Request additional context Missing entities/relationships]
F --> D
C -->|Yes| G[Send structured analysis Entities, relationships, intent]

%% Embedding & Retrieval Phase
G --> H[TensorFlow: Process query bundle Structured + context]
H --> I[Gemma: Generate embeddings Dense vectors + validation]
I --> J{Embedding Quality OK?}
J -->|No| K[TensorFlow: Refinement request Optimize for similarity search]
K --> I
J -->|Yes| L[LanceDB: Receive embeddings 768/1024/1536 dimensions]

L --> M[LanceDB: ANN Search Rank by cosine similarity]
M --> N{Search Results Satisfactory?}
N -->|No| O[Gemma: Adjust parameters Modify k, threshold, filters]
O --> M
N -->|Yes| P[Send ranked documents Metadata + embeddings + scores]

%% Template Generation & Response Phase
P --> Q[BERTopic: Extract topics Build context hierarchy]
Q --> R{Context Relevance OK?}
R -->|No| S[Validate document relevance Topic coherence check]
S --> M
R -->|Yes| T[Generate prompt template Context + instructions + examples]

T --> U[LLM: Generate response Assess initial quality]
U --> V{Prompt Optimization Needed?}
V -->|Yes| W[BERTopic: Enhance template Improve context ordering]
W --> U
V -->|No| X{Reasoning Validation Needed?}

X -->|Yes| Y[TensorFlow: Logic check Consistency validation]
Y --> Z{Logic Valid?}
Z -->|No| AA[Provide reasoning suggestions Framework adjustments]
AA --> U
Z -->|Yes| BB[Generate final response Answer + sources + confidence]
X -->|No| BB

%% Output & Feedback Phase
BB --> CC[User receives response Comprehensive output]
CC --> DD{User provides feedback?}
DD -->|No| EE[Process Complete]
DD -->|Yes| FF[Collect user feedback Rating, corrections, follow-up]

%% Learning Propagation
FF --> GG[LLM: Update effectiveness Prompt performance metrics]
GG --> HH[BERTopic: Update patterns Template optimization data]
HH --> II[LanceDB: Update retrieval Successful query patterns]
II --> JJ[Gemma: Update preferences Vector pattern optimization]
JJ --> KK[TensorFlow: Update weights Logic pattern improvements]
KK --> LL[spaCy: Update models Entity/intent recognition]
LL --> MM[System Learning Complete Models updated]

%% Styling
classDef userStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
classDef spacyStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
classDef tensorStyle fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
classDef gemmaStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
classDef lanceStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
classDef bertStyle fill:#e0f2f1,stroke:#00695c,stroke-width:2px
classDef llmStyle fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
classDef decisionStyle fill:#ffecb3,stroke:#ffa000,stroke-width:2px
classDef completeStyle fill:#ffcdd2,stroke:#d32f2f,stroke-width:2px

class A,CC,FF userStyle
class B,D,LL spacyStyle
class E,H,Y,KK tensorStyle
class I,O,JJ gemmaStyle
class L,M,S,II lanceStyle
class Q,W,HH bertStyle
class U,GG llmStyle
class C,J,N,R,V,X,Z,DD decisionStyle
class EE,MM completeStyle
```
```
Raw Text → Clean Text → Structured Data → Vector Embeddings → Search Interface
```

1. **Raw Text Intake**: The system begins with raw, unstructured text data (books, articles, documents)
2. **Text Cleaning**: The first component removes noise and normalizes the text
3. **Information Extraction**: The second component identifies entities and relationships
4. **Vector Embedding**: The third component converts structured data to vector representations
5. **Indexed Storage**: Vectors are stored in Milvus with their metadata and relationships
6. **Search Interface**: The final component enables semantic search via text queries

```mermaid
sequenceDiagram
    participant User as User Query
    participant spaCy as Query Analyzer<br/>(NLP Processing)
    participant TensorFlow as Reasoning Chain<br/>(Logic Framework)
    participant Gemma as Embedding Model<br/>(Vector Generation)
    participant LanceDB as Vector Database<br/>(Similarity Search)
    participant BERTopic as Prompt Template<br/>(Context Builder)
    participant LLM as Large Language Model<br/>(Response Generation)
    
    Note over User,LLM: Query Initialization & Analysis Phase
    User->>spaCy: Natural Language Query<br/>(Direct/Multi-Step/Complex)
    spaCy->>spaCy: Parse entities, syntax,<br/>intent classification
    spaCy->>TensorFlow: Structured query analysis<br/>(entities, relationships, intent)
    TensorFlow->>TensorFlow: Build reasoning graph<br/>& identify logic gaps
    TensorFlow-->>spaCy: Request additional context<br/>(missing entities/relationships)
    spaCy-->>TensorFlow: Enhanced semantic analysis<br/>(expanded entities, context)
    
    Note over TensorFlow,LanceDB: Embedding & Retrieval Phase
    TensorFlow->>Gemma: Processed query bundle<br/>(structured + context)
    Gemma->>Gemma: Generate dense embeddings<br/>& validate dimensionality
    Gemma-->>TensorFlow: Embedding quality metrics<br/>(confidence scores)
    TensorFlow-->>Gemma: Embedding refinement request<br/>(optimize for similarity search)
    Gemma->>LanceDB: High-dimensional embeddings<br/>(768/1024/1536 vectors)
    
    LanceDB->>LanceDB: Perform ANN search<br/>& rank by cosine similarity
    LanceDB-->>Gemma: Initial search results<br/>(top-k candidates + scores)
    Gemma-->>LanceDB: Search refinement parameters<br/>(adjust k, threshold, filters)
    LanceDB->>BERTopic: Ranked relevant documents<br/>(metadata + embeddings + scores)
    
    Note over BERTopic,User: Template Generation & Response Phase
    BERTopic->>BERTopic: Extract topics/themes<br/>& build context hierarchy
    BERTopic-->>LanceDB: Validate document relevance<br/>(topic coherence check)
    LanceDB-->>BERTopic: Confirmed context data<br/>(validated documents + metadata)
    BERTopic->>LLM: Structured prompt template<br/>(context + instructions + examples)
    
    LLM->>LLM: Generate initial response<br/>& assess quality
    LLM-->>BERTopic: Request prompt optimization<br/>(clarity, context balance)
    BERTopic-->>LLM: Enhanced prompt template<br/>(improved context ordering)
    LLM-->>TensorFlow: Reasoning validation request<br/>(logical consistency check)
    TensorFlow-->>LLM: Validated reasoning framework<br/>(logic approval + suggestions)
    
    LLM->>User: Comprehensive response<br/>(answer + sources + confidence)
    
    Note over User,spaCy: Feedback & Learning Phase
    User-->>LLM: User feedback<br/>(rating, corrections, follow-up)
    LLM-->>BERTopic: Update template effectiveness<br/>(prompt performance metrics)
    BERTopic-->>LanceDB: Update retrieval patterns<br/>(successful query patterns)
    LanceDB-->>Gemma: Update embedding preferences<br/>(successful vector patterns)
    Gemma-->>TensorFlow: Update reasoning weights<br/>(successful logic patterns)
    TensorFlow-->>spaCy: Update analysis models<br/>(improved entity/intent recognition)
```
## Component 1: Text Cleaning Utility

The Text Cleaning Utility prepares raw text for further processing by:

- Removing HTML/XML tags and irrelevant characters
- Normalizing text (lowercase conversion)
- Handling punctuation intelligently
- Removing stopwords and applying stemming/lemmatization
- Performing basic tokenization

**Input**: Raw text files (books, articles, documents)  
**Output**: Cleaned text files with prefix `clean_`

## Component 2: Text Structuring and Relationship Extraction

The Text Structuring component performs advanced NLP to extract meaningful information:

- Named Entity Recognition (NER) to identify people, organizations, locations, etc.
- Relationship extraction through dependency parsing
- Event detection to identify important occurrences
- Context preservation to maintain semantic understanding
- Knowledge graph construction to represent connections

**Input**: Cleaned text files with prefix `clean_`  
**Output**: Structured JSON files with prefix `struct_` containing entities, relationships, and context

## Component 3: Text Embedding with GloVe and Milvus

The Text Embedding component vectorizes the structured data:

- Generates GloVe embeddings for entities, statements, and relationships
- Creates specialized collections in Milvus for different data types
- Stores vectors with their metadata for retrieval
- Configures indices for efficient similarity search
- Preserves relationship information in the vector database

**Input**: Structured JSON files with prefix `struct_`  
**Output**: Populated Milvus vector database with entities, statements, and relationships

## Component 4: Vector Search Application

The Search Application provides the interface for querying the system:

- Converts natural language queries to vector embeddings
- Searches for similar entities, statements, and relationships
- Retrieves and ranks results by semantic similarity
- Presents information with context and source attribution
- Supports both web interface and API access

**Input**: User queries in natural language  
**Output**: Semantically relevant results from the document corpus

## Complete System Integration

### Data Flow Examples

#### Example 1: Book Processing

```
1. Raw Book → Text Cleaning Utility
   Input: "The Adventures of Sherlock Holmes.txt"
   Output: "clean_The Adventures of Sherlock Holmes.txt"

2. Cleaned Text → Text Structuring Tool
   Input: "clean_The Adventures of Sherlock Holmes.txt"
   Output: "struct_The Adventures of Sherlock Holmes.txt.json"

3. Structured Data → Text Embedding Service
   Input: "struct_The Adventures of Sherlock Holmes.txt.json"
   Output: Populated Milvus collections (entities, statements, relationships)

4. Vector Database → Search Application
   User query: "Who was Sherlock Holmes' nemesis?"
   Results: References to Professor Moriarty with contextual information
```

#### Example 2: Multi-Document Analysis

```
1. Process multiple documents through the pipeline
2. Build a comprehensive knowledge base in the vector database
3. Query across documents: "What companies were founded in California?"
4. Retrieve relevant information from multiple sources with attribution
```
