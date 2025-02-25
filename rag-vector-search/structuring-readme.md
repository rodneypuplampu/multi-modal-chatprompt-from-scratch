# NLP Text Structuring and Relationship Extraction

This guide explains how to implement the NLP Text Structuring and Relationship Extraction component, which forms the second step in the complete text processing pipeline.

## Overview

The Text Structuring component takes cleaned text (from the Text Cleaning Utility) and performs advanced entity recognition and relationship extraction. It uses natural language processing techniques to identify entities, their relationships, and preserve contextual information.

## Features

- **Named Entity Recognition (NER)**: Identifies people, organizations, locations, dates, and events
- **Relationship Extraction**: Discovers connections between entities using dependency parsing
- **Event Detection**: Identifies events with associated entities, times, and locations
- **Context Preservation**: Maintains surrounding sentences for better understanding
- **Knowledge Graph Construction**: Builds a graph representation of entities and their relationships

## Installation Requirements

### 1. Install Required Packages

```bash
pip install spacy networkx nltk
```

### 2. Download spaCy Model

```bash
# Download English language model (small version)
python -m spacy download en_core_web_sm

# For better accuracy with slightly higher resource usage, use the medium model
python -m spacy download en_core_web_md
```

### 3. Download NLTK Resources

```python
import nltk
nltk.download('punkt')
```

## Implementation

1. Save the Text Structurer code to a file named `text_structurer.py`
2. Ensure you have already processed your text files with the Text Cleaning Utility, which produces files with the prefix `clean_`

## Usage

### Basic Usage

```bash
python text_structurer.py clean_mybook.txt
```

This will process the cleaned text file and generate a corresponding structured output file with the prefix `struct_` and a `.json` extension.

### Advanced Options

```bash
# Specify custom output directory
python text_structurer.py clean_mybook.txt --output structured_output

# Use a more accurate (but slower) spaCy model
python text_structurer.py clean_mybook.txt --model en_core_web_md

# Adjust the context window size (default is 2 sentences before and after)
python text_structurer.py clean_mybook.txt --context 3

# Process all cleaned files in a directory
python text_structurer.py cleaned_books_directory/
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input text file or directory (cleaned text from the Text Cleaner) |
| `--output`, `-o` | Output directory (default: ./structured) |
| `--model`, `-m` | spaCy model to use (default: en_core_web_sm) |
| `--context`, `-c` | Context window size (sentences before/after) |

## Output Format

The Text Structurer outputs a JSON file with the following structure:

```json
{
  "entities": {
    "entity_id_1": {
      "text": "John Smith",
      "type": "person",
      "mentions": [{"start": 0, "end": 10}]
    },
    "entity_id_2": {
      "text": "Acme Corp",
      "type": "organization",
      "mentions": [{"start": 15, "end": 24}]
    }
  },
  "relationships": [
    {
      "subject": "John Smith",
      "relation": "works at",
      "object": "Acme Corp",
      "sentence_idx": 0,
      "sentence": "John Smith works at Acme Corp."
    }
  ],
  "events": [
    {
      "sentence": "The meeting occurred on January 5th in New York.",
      "entities": {
        "event": ["meeting"],
        "date": ["January 5th"],
        "location": ["New York"]
      },
      "context": ["Previous context sentence.", "The meeting occurred on January 5th in New York.", "Next context sentence."],
      "central_idx": 1
    }
  ],
  "statements": [
    {
      "statement": "John Smith works at Acme Corp.",
      "type": "relationship",
      "context": ["Full sentence with context."],
      "entities": ["John Smith", "Acme Corp"]
    }
  ],
  "knowledge_graph": {
    "nodes": [...],
    "edges": [...]
  }
}
```

## How It Works

### Entity Recognition

The Text Structurer uses spaCy's named entity recognition (NER) to identify entities like people, organizations, and locations in the text. Each entity is assigned a unique ID and its occurrences (mentions) are tracked.

```python
doc = nlp(text)
for ent in doc.ents:
    if ent.label_ in entity_types:
        # Process entity...
```

### Relationship Extraction

Relationships between entities are extracted using dependency parsing, which analyzes the grammatical structure of sentences.

1. **Subject-Verb-Object Patterns**: Identifies who did what to whom
   ```python
   # Find subjects, verbs, and objects
   for token in sent:
       if token.dep_ in ("nsubj", "nsubjpass"):
           subjects.append(token)
       elif token.pos_ == "VERB":
           verbs.append(token)
       elif token.dep_ in ("dobj", "pobj"):
           objects.append(token)
   ```

2. **Possessive Relationships**: Identifies ownership or association
   ```python
   if token.dep_ == "poss" and token.head.pos_ in ("NOUN", "PROPN"):
       # Process possessive relationship...
   ```

### Event Detection

Events are identified by looking for sentences that contain:
- Date/time expressions
- Location mentions
- Event indicator words (e.g., "meeting", "conference", "occurred")

```python
has_date = any(ent.label_ in ("DATE", "TIME") for ent in doc.ents)
has_location = any(ent.label_ in ("GPE", "LOC") for ent in doc.ents)
has_event_indicators = re.search(r'\b(meet|meeting|event|...)\b', sentence.lower())
```

### Context Preservation

For each relationship or event, the structurer preserves a window of surrounding sentences to maintain context:

```python
context_start = max(0, i - context_window)
context_end = min(len(sentences), i + context_window + 1)
context = sentences[context_start:context_end]
```

### Knowledge Graph Construction

The extracted entities and relationships are used to build a knowledge graph, which represents the connections between different entities:

```python
G = nx.Graph()
# Add entities as nodes
for entity_id, entity_data in entities.items():
    G.add_node(entity_id, **entity_data)
# Add relationships as edges
for rel in relationships:
    G.add_edge(subj_id, obj_id, relation=rel["relation"])
```

## Tips for Optimization

1. **Model Selection**: Balance accuracy vs. speed
   - `en_core_web_sm`: Faster but less accurate
   - `en_core_web_md`: More accurate but slower
   - `en_core_web_lg`: Most accurate but requires more memory

2. **Batch Processing**: For large document collections, process files in batches

3. **Custom Relationship Patterns**: Extend the `relationship_patterns` and `relationship_verbs` dictionaries to capture domain-specific relationships

## Integration with Full Pipeline

1. **First step**: Clean text with the Text Cleaning Utility
   ```bash
   python text_cleaner.py my_book.txt --output cleaned
   ```

2. **Second step**: Extract structured information
   ```bash
   python text_structurer.py cleaned/clean_my_book.txt --output structured
   ```

3. **Third step**: Generate embeddings and store in a vector database
   ```bash
   python text_embedder.py structured/struct_my_book.txt.json
   ```

## Troubleshooting

- **Memory Issues**: For very large texts, process them in smaller chunks
- **Missing Entities**: Adjust the entity types dictionary to include additional entity types
- **Incorrect Relationships**: Refine the relationship patterns or use a larger spaCy model

## Example

### Input (Cleaned Text):
```
john smith joined acme corp in 2020 as ceo the company headquartered in new york specializes in software development john previously worked at tech innovations where he led the research team
```

### Output (Structured JSON):
The structured output will contain entities (John Smith, Acme Corp, New York, Tech Innovations), relationships (John Smith works at Acme Corp, John Smith previously worked at Tech Innovations), and a knowledge graph connecting these entities.

## Extending the Structurer

To extend the Text Structurer for specific domains:

1. **Add Domain-Specific Entity Types**:
   ```python
   entity_types = {
       'PERSON': 'person',
       'ORG': 'organization',
       # Add custom types
       'CHEMICAL': 'chemical_compound',
       'GENE': 'gene'
   }
   ```

2. **Add Custom Relationship Patterns**:
   ```python
   relationship_verbs = {
       # Add domain-specific verbs
       "inhibit": "inhibits",
       "catalyze": "catalyzes",
       "bind": "binds to"
   }
   ```

3. **Implement Custom Event Detection**:
   ```python
   has_chemical_reaction = re.search(r'\b(reacts|synthesized|bonded)\b', sentence.lower())
   ```

## Advanced Use Cases

- **Scientific Literature Analysis**: Extract relationships between chemicals, genes, or proteins
- **Legal Document Processing**: Identify parties, dates, and legal relationships
- **News Article Analysis**: Extract events, people, organizations, and locations

## Next Steps

After structuring your text, proceed to the Text Embedding component to generate vector representations and store them in a vector database for efficient similarity search.
