# NLP Text Processing Pipeline

This repository contains a comprehensive pipeline for processing unstructured text data into structured, searchable vector embeddings. The pipeline consists of three main components:

1. **Text Cleaning Utility**: Cleans and normalizes raw text
2. **Text Structuring Tool**: Extracts entities and relationships with context
3. **Text Embedding Service**: Vectorizes structured data and stores it in Milvus

## 1. Text Cleaning Utility

The first step in the pipeline is to clean and normalize raw text input.

### Features

- Removes HTML/XML tags and markup
- Normalizes text (lowercase conversion)
- Handles punctuation intelligently
- Removes stopwords
- Performs stemming and/or lemmatization
- Tokenizes text properly
- Extracts basic relationships and patterns

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nlp-text-pipeline.git
cd nlp-text-pipeline

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install nltk
```

### Usage

```bash
# Basic usage
python text_cleaner.py my_book.txt

# Process with custom options
python text_cleaner.py my_book.txt --output cleaned_books --keep-stopwords --stem

# Process a directory of text files
python text_cleaner.py my_books_directory/ --output cleaned_books
```

#### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input text file or directory |
| `--output`, `-o` | Output directory (default: ./output) |
| `--keep-stopwords` | Keep stopwords in the text |
| `--keep-case` | Keep original case (don't convert to lowercase) |
| `--keep-punctuation` | Keep punctuation in the text |
| `--stem` | Apply stemming to words |
| `--no-lemmatize` | Disable lemmatization |

### Output

For each processed file, the utility generates two output files:
1. `clean_[filename]` - The cleaned text
2. `struct_[filename].txt` - Basic structured data with identified relationships

### Example

Input text:
```
<p>John Smith, CEO of Acme Corp., announced the merger on Jan. 15, 2023. The deal, worth $2.5M, was completed in New York.</p>
```

Cleaned output:
```
john smith ceo acme corp announced merger jan 15 2023 deal worth 2.5m completed new york
```

Basic structured output:
```
RELATIONSHIPS:
- john is the ceo of acme

DATES:
- Jan. 15, 2023

LOCATIONS:
- New York
```

## 2. Text Structuring

After cleaning, the next step is to perform more advanced entity and relationship extraction. This component builds on the cleaned output from step 1.

### Features and Implementation

See the full README for instructions on implementing the Text Structuring component.

## 3. Text Embedding

The final step is to vectorize the structured data and store it in a vector database. This component builds on the structured output from step 2.

### Features and Implementation

See the full README for instructions on implementing the Text Embedding component.

## Complete Pipeline

To process a text file through the complete pipeline:

```bash
# 1. Clean the text
python text_cleaner.py my_book.txt --output cleaned

# 2. Extract structured information
python text_structurer.py cleaned/clean_my_book.txt --output structured

# 3. Generate embeddings and store in Milvus
python text_embedder.py structured/struct_my_book.txt.json
```

## Advanced Usage

See the full documentation for advanced usage scenarios, including:
- Customizing the pipeline for specific domains
- Adding custom relationship extraction patterns
- Integrating with other NLP libraries
- Scaling to large document collections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
