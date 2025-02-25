import re
import os
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import html
import string

class TextCleaner:
    def __init__(self, remove_stopwords=True, lowercase=True, 
                 remove_punctuation=True, stem=False, lemmatize=True):
        """
        Initialize the TextCleaner with configurable options
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lowercase (bool): Whether to convert text to lowercase
            remove_punctuation (bool): Whether to remove punctuation
            stem (bool): Whether to apply stemming
            lemmatize (bool): Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.stem = stem
        self.lemmatize = lemmatize
        
        # Download necessary NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def remove_markup(self, text):
        """Remove HTML/XML tags and escape sequences"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = html.unescape(text)
        # Remove escape sequences
        text = re.sub(r'\\[ntrfv]', ' ', text)
        return text
    
    def normalize_text(self, text):
        """Apply text normalization"""
        if self.lowercase:
            text = text.lower()
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def handle_punctuation(self, text):
        """Handle punctuation based on configuration"""
        if self.remove_punctuation:
            # Remove all punctuation except apostrophes for contractions
            # and hyphens for compound words
            translator = str.maketrans('', '', string.punctuation.replace("'", "").replace("-", ""))
            text = text.translate(translator)
            
            # Clean up any remaining unnecessary apostrophes or hyphens
            text = re.sub(r'\s+\'\s+', ' ', text)  # Remove isolated apostrophes
            text = re.sub(r'\s+\-\s+', ' ', text)  # Remove isolated hyphens
        
        return text
    
    def process_tokens(self, tokens):
        """Process tokens with stemming/lemmatization and stopword removal"""
        processed_tokens = []
        
        for token in tokens:
            # Skip stopwords if configured
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Apply stemming if configured
            if self.stem:
                token = self.stemmer.stem(token)
                
            # Apply lemmatization if configured
            if self.lemmatize:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
            
        return processed_tokens
    
    def clean_text(self, text):
        """Apply the full cleaning pipeline to the text"""
        # Remove markup and irrelevant characters
        text = self.remove_markup(text)
        
        # Normalize text (lowercase, etc.)
        text = self.normalize_text(text)
        
        # Handle punctuation
        text = self.handle_punctuation(text)
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Process each sentence
        cleaned_sentences = []
        for sentence in sentences:
            # Tokenize sentence into words
            tokens = word_tokenize(sentence)
            
            # Process tokens (stopwords, stemming/lemmatization)
            processed_tokens = self.process_tokens(tokens)
            
            # Reconstruct the sentence if it has tokens left
            if processed_tokens:
                cleaned_sentence = ' '.join(processed_tokens)
                cleaned_sentences.append(cleaned_sentence)
        
        # Join sentences back together with appropriate spacing
        cleaned_text = ' '.join(cleaned_sentences)
        
        return cleaned_text
    
    def structure_relationships(self, text):
        """
        Identify and structure potential relationships in the text.
        This is a simplified version - more advanced NER and relation extraction
        would require additional NLP libraries like spaCy.
        """
        # A very simplistic approach to identify potential relationships
        # based on common patterns
        
        # Find potential "A is B" relationships
        is_relationships = re.findall(r'(\w+)\s+is\s+the\s+(\w+)\s+of\s+(\w+)', text)
        
        # Find potential date and location mentions
        date_mentions = re.findall(r'(\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},\s+\d{4})', text)
        location_mentions = re.findall(r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
        
        # Structure these findings
        structured_data = {
            "relationships": [f"{a} is the {b} of {c}" for a, b, c in is_relationships],
            "dates": date_mentions,
            "locations": location_mentions
        }
        
        return structured_data

def process_file(file_path, output_dir, cleaner):
    """Process a single text file"""
    try:
        # Get the base filename without path
        base_name = os.path.basename(file_path)
        
        # Create output file path
        clean_path = os.path.join(output_dir, f"clean_{base_name}")
        struct_path = os.path.join(output_dir, f"struct_{base_name}.txt")
        
        # Read the input file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Clean the text
        cleaned_text = cleaner.clean_text(text)
        
        # Extract and structure relationships
        structured_data = cleaner.structure_relationships(text)  # Use original text for relation extraction
        
        # Write cleaned text to file
        with open(clean_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Write structured data to file
        with open(struct_path, 'w', encoding='utf-8') as f:
            f.write("RELATIONSHIPS:\n")
            for rel in structured_data["relationships"]:
                f.write(f"- {rel}\n")
            
            f.write("\nDATES:\n")
            for date in structured_data["dates"]:
                f.write(f"- {date}\n")
            
            f.write("\nLOCATIONS:\n")
            for loc in structured_data["locations"]:
                f.write(f"- {loc}\n")
        
        print(f"Processed {file_path} -> {clean_path}, {struct_path}")
        return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Clean and structure text data from book files')
    parser.add_argument('input', help='Input text file or directory')
    parser.add_argument('--output', '-o', default='output', help='Output directory (default: ./output)')
    parser.add_argument('--keep-stopwords', action='store_true', help='Keep stopwords')
    parser.add_argument('--keep-case', action='store_true', help='Keep original case')
    parser.add_argument('--keep-punctuation', action='store_true', help='Keep punctuation')
    parser.add_argument('--stem', action='store_true', help='Apply stemming')
    parser.add_argument('--no-lemmatize', action='store_true', help='Disable lemmatization')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize the cleaner with command-line options
    cleaner = TextCleaner(
        remove_stopwords=not args.keep_stopwords,
        lowercase=not args.keep_case,
        remove_punctuation=not args.keep_punctuation,
        stem=args.stem,
        lemmatize=not args.no_lemmatize
    )
    
    # Process files
    if os.path.isdir(args.input):
        # Process all text files in directory
        success_count = 0
        total_count = 0
        
        for filename in os.listdir(args.input):
            if filename.endswith('.txt'):
                file_path = os.path.join(args.input, filename)
                if process_file(file_path, args.output, cleaner):
                    success_count += 1
                total_count += 1
        
        print(f"Processed {success_count} of {total_count} files successfully")
    else:
        # Process single file
        if process_file(args.input, args.output, cleaner):
            print("Processing completed successfully")
        else:
            print("Processing failed")

if __name__ == '__main__':
    main()
