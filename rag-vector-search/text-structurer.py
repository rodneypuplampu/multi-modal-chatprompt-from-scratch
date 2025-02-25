import os
import re
import json
import argparse
import spacy
import networkx as nx
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize

class TextStructurer:
    def __init__(self, spacy_model="en_core_web_sm", context_window=2):
        """
        Initialize the TextStructurer
        
        Args:
            spacy_model (str): The spaCy model to use for NER and dependency parsing
            context_window (int): Number of sentences before/after to include for context
        """
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model {spacy_model}...")
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)
        
        self.context_window = context_window
        
        # Entity types we're interested in
        self.entity_types = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',  # Countries, cities, etc.
            'LOC': 'location',  # Non-GPE locations
            'DATE': 'date',
            'TIME': 'time',
            'EVENT': 'event',
            'FAC': 'facility', 
            'WORK_OF_ART': 'work',
            'PRODUCT': 'product'
        }
        
        # Relationship patterns based on dependency parsing
        self.relationship_patterns = [
            # Subject-verb-object patterns
            {"SUBJECT": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}, 
             "VERB": {"POS": "VERB"}, 
             "OBJECT": {"DEP": {"IN": ["dobj", "pobj"]}}},
            
            # Possessive patterns
            {"ENTITY1": {"DEP": "poss"}, 
             "ENTITY2": {"POS": {"IN": ["NOUN", "PROPN"]}}}
        ]
        
        # Common relationship verbs and their meanings
        self.relationship_verbs = {
            "be": "is",
            "have": "has",
            "own": "owns",
            "create": "created",
            "write": "wrote",
            "direct": "directed",
            "produce": "produced",
            "found": "founded",
            "establish": "established",
            "marry": "married to",
            "lead": "leads",
            "manage": "manages",
            "parent": "is parent of",
            "birth": "gave birth to",
            "work": "works at",
            "employ": "employs",
            "live": "lives in",
            "locate": "located in",
            "occur": "occurred on",
            "happen": "happened at",
            "visit": "visited",
            "travel": "traveled to",
            "speak": "speaks",
            "contain": "contains"
        }
    
    def extract_entities_and_relationships(self, text):
        """
        Extract entities and their relationships from text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            tuple: (entities, relationships, contexts)
        """
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = {}
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entity_id = f"{ent.text.lower().replace(' ', '_')}_{len(entities)}"
                entities[entity_id] = {
                    "text": ent.text,
                    "type": self.entity_types[ent.label_],
                    "mentions": [{"start": ent.start_char, "end": ent.end_char}]
                }
        
        # Extract relationships through dependency parsing
        relationships = []
        
        # Process each sentence
        sentences = list(doc.sents)
        for sent_idx, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            
            # Skip very short sentences
            if len(sent_text.split()) < 3:
                continue
                
            # Find subject-verb-object relationships
            subjects = []
            verbs = []
            objects = []
            
            for token in sent:
                # Find subjects
                if token.dep_ in ("nsubj", "nsubjpass") and (token.pos_ == "PROPN" or token.pos_ == "NOUN"):
                    subjects.append(token)
                
                # Find main verbs
                if token.pos_ == "VERB":
                    verbs.append(token)
                
                # Find objects
                if token.dep_ in ("dobj", "pobj") and (token.pos_ == "PROPN" or token.pos_ == "NOUN"):
                    objects.append(token)
            
            # Create relationship triplets
            for subj in subjects:
                for verb in verbs:
                    # Check if the verb is related to this subject
                    if verb.is_ancestor(subj):
                        for obj in objects:
                            # Check if the object is related to this verb
                            if verb.is_ancestor(obj):
                                # Get relationship type from verb
                                rel_type = verb.lemma_
                                if rel_type in self.relationship_verbs:
                                    rel_type = self.relationship_verbs[rel_type]
                                    
                                # Create the relationship
                                relationship = {
                                    "subject": subj.text,
                                    "relation": rel_type,
                                    "object": obj.text,
                                    "sentence_idx": sent_idx,
                                    "sentence": sent_text
                                }
                                relationships.append(relationship)
            
            # Find possessive relationships
            for token in sent:
                if token.dep_ == "poss" and token.head.pos_ in ("NOUN", "PROPN"):
                    relationship = {
                        "subject": token.text,
                        "relation": "possesses",
                        "object": token.head.text,
                        "sentence_idx": sent_idx,
                        "sentence": sent_text
                    }
                    relationships.append(relationship)
        
        # Create context windows for relationships
        contexts = {}
        for idx, rel in enumerate(relationships):
            sent_idx = rel["sentence_idx"]
            context_start = max(0, sent_idx - self.context_window)
            context_end = min(len(sentences), sent_idx + self.context_window + 1)
            
            context_sentences = [s.text.strip() for s in sentences[context_start:context_end]]
            contexts[idx] = {
                "sentences": context_sentences,
                "central_idx": sent_idx - context_start
            }
        
        return entities, relationships, contexts
    
    def extract_events(self, text):
        """
        Extract events with related entities from text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            list: Events with associated entities and context
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        events = []
        
        # Identify sentences that might contain events
        for i, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            
            # Check if sentence contains date/time and location
            has_date = any(ent.label_ in ("DATE", "TIME") for ent in doc.ents)
            has_location = any(ent.label_ in ("GPE", "LOC") for ent in doc.ents)
            has_event_indicators = re.search(r'\b(meet|meeting|event|conference|ceremony|gathering|happened|occurred|took place)\b', 
                                             sentence.lower())
            
            # If it has event indicators or both date and location, it's likely an event
            if has_event_indicators or (has_date and has_location):
                # Collect context
                context_start = max(0, i - self.context_window)
                context_end = min(len(sentences), i + self.context_window + 1)
                context = sentences[context_start:context_end]
                
                # Extract entities
                entities = defaultdict(list)
                for ent in doc.ents:
                    if ent.label_ in self.entity_types:
                        entities[self.entity_types[ent.label_]].append(ent.text)
                
                # Create event record
                event = {
                    "sentence": sentence,
                    "entities": dict(entities),
                    "context": context,
                    "central_idx": i - context_start
                }
                events.append(event)
        
        return events
    
    def create_knowledge_graph(self, entities, relationships):
        """
        Create a knowledge graph from entities and relationships
        
        Args:
            entities (dict): Extracted entities
            relationships (list): Extracted relationships
            
        Returns:
            networkx.Graph: Knowledge graph
        """
        G = nx.Graph()
        
        # Add entities as nodes
        for entity_id, entity_data in entities.items():
            G.add_node(entity_id, **entity_data)
        
        # Add relationships as edges
        for rel in relationships:
            # Try to find matching entity IDs
            subj_id = None
            obj_id = None
            
            for entity_id, entity_data in entities.items():
                if entity_data["text"].lower() == rel["subject"].lower():
                    subj_id = entity_id
                if entity_data["text"].lower() == rel["object"].lower():
                    obj_id = entity_id
            
            # If both subject and object were found in entities, add an edge
            if subj_id and obj_id:
                G.add_edge(subj_id, obj_id, relation=rel["relation"], sentence=rel["sentence"])
        
        return G
    
    def generate_structured_data(self, entities, relationships, contexts, events):
        """
        Generate structured data representation from extracted information
        
        Args:
            entities (dict): Extracted entities
            relationships (list): Extracted relationships
            contexts (dict): Context windows for relationships
            events (list): Extracted events
            
        Returns:
            dict: Structured data representation
        """
        # Create a knowledge graph
        graph = self.create_knowledge_graph(entities, relationships)
        
        # Create structured statements
        statements = []
        
        # Relationship statements
        for rel in relationships:
            statement = f"{rel['subject']} {rel['relation']} {rel['object']}."
            context_idx = next((i for i, c in contexts.items() if c['central_idx'] == rel['sentence_idx']), None)
            
            if context_idx is not None:
                statements.append({
                    "statement": statement,
                    "type": "relationship",
                    "context": contexts[context_idx]['sentences'],
                    "entities": [rel['subject'], rel['object']]
                })
        
        # Event statements
        for event in events:
            # Create a statement about the event
            event_entities = []
            statement_parts = []
            
            # Add people
            if "person" in event["entities"]:
                people = ", ".join(event["entities"]["person"])
                event_entities.extend(event["entities"]["person"])
                statement_parts.append(f"{people}")
            
            # Add verb based on event indicators
            if re.search(r'\b(meet|meeting|conference|gathering)\b', event["sentence"].lower()):
                statement_parts.append("met")
            elif re.search(r'\b(happen|happened|occurred|took place)\b', event["sentence"].lower()):
                statement_parts.append("participated in an event")
            
            # Add location
            if "location" in event["entities"]:
                locations = ", ".join(event["entities"]["location"])
                event_entities.extend(event["entities"]["location"])
                statement_parts.append(f"in {locations}")
            
            # Add date/time
            if "date" in event["entities"]:
                dates = ", ".join(event["entities"]["date"])
                event_entities.extend(event["entities"]["date"])
                statement_parts.append(f"on {dates}")
            
            if statement_parts:
                event_statement = " ".join(statement_parts) + "."
                statements.append({
                    "statement": event_statement,
                    "type": "event",
                    "context": event["context"],
                    "entities": event_entities
                })
        
        # Entity statements (for entities with multiple relationships)
        entity_relationships = defaultdict(list)
        for rel in relationships:
            entity_relationships[rel["subject"]].append((rel["relation"], rel["object"]))
            entity_relationships[rel["object"]].append((f"is {rel['relation']} by" if not rel["relation"].startswith("is ") else rel["relation"].replace("is ", "of "), rel["subject"]))
        
        for entity, rels in entity_relationships.items():
            if len(rels) > 1:
                rel_statements = [f"{rel[0]} {rel[1]}" for rel in rels]
                statement = f"{entity} {'; '.join(rel_statements)}."
                
                statements.append({
                    "statement": statement,
                    "type": "entity_summary",
                    "entities": [entity] + [rel[1] for rel in rels]
                })
        
        # Return structured data representation
        return {
            "entities": entities,
            "relationships": relationships,
            "events": events,
            "statements": statements,
            "knowledge_graph": {
                "nodes": list(graph.nodes(data=True)),
                "edges": [(u, v, data) for u, v, data in graph.edges(data=True)]
            }
        }
    
    def process_text(self, text):
        """
        Process text and generate structured data
        
        Args:
            text (str): Text to process
            
        Returns:
            dict: Structured data
        """
        # Extract entities and relationships
        entities, relationships, contexts = self.extract_entities_and_relationships(text)
        
        # Extract events
        events = self.extract_events(text)
        
        # Generate structured data
        structured_data = self.generate_structured_data(entities, relationships, contexts, events)
        
        return structured_data

def process_file(input_file, output_file, structurer):
    """Process a single text file"""
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process the text
        structured_data = structurer.process_text(text)
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2)
        
        print(f"Processed {input_file} -> {output_file}")
        return True
    
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Structure text data and extract relationships')
    parser.add_argument('input', help='Input text file or directory (cleaned text from the Text Cleaner)')
    parser.add_argument('--output', '-o', default='structured', help='Output directory (default: ./structured)')
    parser.add_argument('--model', '-m', default='en_core_web_sm', help='spaCy model to use (default: en_core_web_sm)')
    parser.add_argument('--context', '-c', type=int, default=2, help='Context window size (sentences before/after)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize the structurer
    structurer = TextStructurer(spacy_model=args.model, context_window=args.context)
    
    # Process files
    if os.path.isdir(args.input):
        # Process all text files in directory
        success_count = 0
        total_count = 0
        
        for filename in os.listdir(args.input):
            if filename.startswith('clean_') and filename.endswith('.txt'):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, filename.replace('clean_', 'struct_') + '.json')
                
                if process_file(input_path, output_path, structurer):
                    success_count += 1
                total_count += 1
        
        print(f"Processed {success_count} of {total_count} files successfully")
    else:
        # Process single file
        output_path = os.path.join(args.output, os.path.basename(args.input).replace('clean_', 'struct_') + '.json')
        if process_file(args.input, output_path, structurer):
            print("Processing completed successfully")
        else:
            print("Processing failed")

if __name__ == '__main__':
    main()
