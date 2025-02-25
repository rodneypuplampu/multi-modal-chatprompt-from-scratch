import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymilvus import (
    connections, 
    utility,
    FieldSchema, 
    CollectionSchema, 
    DataType,
    Collection
)
import gensim.downloader as api
from typing import List, Dict, Any, Tuple, Optional, Union

class TextEmbedder:
    def __init__(self, glove_model="glove-wiki-gigaword-300", milvus_host="localhost", milvus_port="19530"):
        """
        Initialize the TextEmbedder
        
        Args:
            glove_model (str): The GloVe model to use for word embeddings
            milvus_host (str): Milvus server host
            milvus_port (str): Milvus server port
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.vector_dim = 300  # GloVe vectors are typically 300-dimensional
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        print(f"Loading GloVe model: {glove_model}")
        try:
            # Use gensim to download and load the GloVe model
            self.glove_model = api.load(glove_model)
            print(f"GloVe model loaded with {len(self.glove_model.key_to_index)} words")
        except Exception as e:
            print(f"Error loading GloVe model: {str(e)}")
            raise
    
    def _connect_to_milvus(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default", 
                host=self.milvus_host,
                port=self.milvus_port
            )
            print(f"Connected to Milvus server at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            print(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def _create_collections(self):
        """Create Milvus collections for different data types"""
        # Define collection schemas
        collections = {
            "entities": self._create_entity_collection(),
            "statements": self._create_statement_collection(),
            "relationships": self._create_relationship_collection()
        }
        return collections
    
    def _create_entity_collection(self):
        """Create and return the entity collection"""
        collection_name = "nlp_entities"
        
        # Drop collection if it already exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # Define fields for entity collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="entity_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        # Create collection schema
        schema = CollectionSchema(fields=fields, description="NLP Entity Embeddings")
        
        # Create collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create an IVF_FLAT index for fast vector similarity search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        return collection
    
    def _create_statement_collection(self):
        """Create and return the statement collection"""
        collection_name = "nlp_statements"
        
        # Drop collection if it already exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # Define fields for statement collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="statement", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="entities", dtype=DataType.JSON),  # Store entity IDs as JSON
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        # Create collection schema
        schema = CollectionSchema(fields=fields, description="NLP Statement Embeddings")
        
        # Create collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create an IVF_FLAT index for fast vector similarity search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        return collection
    
    def _create_relationship_collection(self):
        """Create and return the relationship collection"""
        collection_name = "nlp_relationships"
        
        # Drop collection if it already exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # Define fields for relationship collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="relation", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="object", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="subject_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="relation_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            FieldSchema(name="object_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
        ]
        
        # Create collection schema
        schema = CollectionSchema(fields=fields, description="NLP Relationship Embeddings")
        
        # Create collection
        collection = Collection(name=collection_name, schema=schema)
        
        # Create indices for fast vector similarity search
        for field_name in ["subject_embedding", "relation_embedding", "object_embedding"]:
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name=field_name, index_params=index_params)
        
        return collection
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get GloVe embedding for a text string (average of word vectors)
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Preprocess the text
        words = text.lower().split()
        
        # Filter out words not in the model's vocabulary
        word_vectors = [self.glove_model[word] for word in words if word in self.glove_model]
        
        if not word_vectors:
            # If no words are in vocabulary, return zero vector
            return np.zeros(self.vector_dim)
        
        # Return average of word vectors
        return np.mean(word_vectors, axis=0)
    
    def process_structured_data(self, structured_data: Dict[str, Any], source_file: str) -> Dict[str, List[Dict]]:
        """
        Process structured data and prepare for insertion into Milvus
        
        Args:
            structured_data (dict): Structured data from TextStructurer
            source_file (str): Source file name
            
        Returns:
            dict: Processed data ready for Milvus insertion
        """
        processed_data = {
            "entities": [],
            "statements": [],
            "relationships": []
        }
        
        # Process entities
        for entity_id, entity_data in structured_data.get("entities", {}).items():
            entity_text = entity_data.get("text", "")
            entity_embedding = self.get_embedding(entity_text)
            
            processed_data["entities"].append({
                "entity_id": entity_id,
                "text": entity_text,
                "type": entity_data.get("type", "unknown"),
                "source_file": source_file,
                "embedding": entity_embedding.tolist()
            })
        
        # Process statements
        for statement_data in structured_data.get("statements", []):
            statement_text = statement_data.get("statement", "")
            statement_embedding = self.get_embedding(statement_text)
            
            processed_data["statements"].append({
                "statement": statement_text,
                "type": statement_data.get("type", "unknown"),
                "entities": json.dumps(statement_data.get("entities", [])),
                "source_file": source_file,
                "embedding": statement_embedding.tolist()
            })
        
        # Process relationships
        for relationship in structured_data.get("relationships", []):
            subject = relationship.get("subject", "")
            relation = relationship.get("relation", "")
            obj = relationship.get("object", "")
            sentence = relationship.get("sentence", "")
            
            subject_embedding = self.get_embedding(subject)
            relation_embedding = self.get_embedding(relation)
            object_embedding = self.get_embedding(obj)
            
            processed_data["relationships"].append({
                "subject": subject,
                "relation": relation,
                "object": obj,
                "sentence": sentence,
                "source_file": source_file,
                "subject_embedding": subject_embedding.tolist(),
                "relation_embedding": relation_embedding.tolist(),
                "object_embedding": object_embedding.tolist(),
            })
        
        return processed_data
    
    def insert_to_milvus(self, collections: Dict[str, Collection], processed_data: Dict[str, List[Dict]]):
        """
        Insert processed data into Milvus collections
        
        Args:
            collections (dict): Dictionary of Milvus collections
            processed_data (dict): Processed data ready for insertion
        """
        # Insert entities
        if processed_data["entities"]:
            entities_df = pd.DataFrame(processed_data["entities"])
            collections["entities"].insert(entities_df)
            print(f"Inserted {len(processed_data['entities'])} entities into Milvus")
        
        # Insert statements
        if processed_data["statements"]:
            statements_df = pd.DataFrame(processed_data["statements"])
            collections["statements"].insert(statements_df)
            print(f"Inserted {len(processed_data['statements'])} statements into Milvus")
        
        # Insert relationships
        if processed_data["relationships"]:
            relationships_df = pd.DataFrame(processed_data["relationships"])
            collections["relationships"].insert(relationships_df)
            print(f"Inserted {len(processed_data['relationships'])} relationships into Milvus")
    
    def search_similar_entities(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        Search for entities similar to the query text
        
        Args:
            query_text (str): Query text
            limit (int): Maximum number of results
            
        Returns:
            list: Similar entities with metadata
        """
        # Generate embedding for query text
        query_embedding = self.get_embedding(query_text)
        
        # Get entity collection
        collection = Collection("nlp_entities")
        collection.load()
        
        # Search for similar entities
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["entity_id", "text", "type", "source_file"]
        )
        
        # Format results
        similar_entities = []
        for hits in results:
            for hit in hits:
                similar_entities.append({
                    "id": hit.id,
                    "entity_id": hit.entity,
                    "text": hit.text,
                    "type": hit.type,
                    "source_file": hit.source_file,
                    "distance": hit.distance
                })
        
        return similar_entities
    
    def search_similar_statements(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        Search for statements similar to the query text
        
        Args:
            query_text (str): Query text
            limit (int): Maximum number of results
            
        Returns:
            list: Similar statements with metadata
        """
        # Generate embedding for query text
        query_embedding = self.get_embedding(query_text)
        
        # Get statement collection
        collection = Collection("nlp_statements")
        collection.load()
        
        # Search for similar statements
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["statement", "type", "entities", "source_file"]
        )
        
        # Format results
        similar_statements = []
        for hits in results:
            for hit in hits:
                similar_statements.append({
                    "id": hit.id,
                    "statement": hit.statement,
                    "type": hit.type,
                    "entities": json.loads(hit.entities),
                    "source_file": hit.source_file,
                    "distance": hit.distance
                })
        
        return similar_statements
    
    def search_relationships(self, subject: Optional[str] = None, 
                            relation: Optional[str] = None, 
                            obj: Optional[str] = None, 
                            limit: int = 10) -> List[Dict]:
        """
        Search for relationships matching the given criteria
        
        Args:
            subject (str, optional): Subject entity
            relation (str, optional): Relation type
            obj (str, optional): Object entity
            limit (int): Maximum number of results
            
        Returns:
            list: Matching relationships with metadata
        """
        # Get relationship collection
        collection = Collection("nlp_relationships")
        collection.load()
        
        # Determine which field to search based on provided parameters
        if subject:
            query_embedding = self.get_embedding(subject)
            search_field = "subject_embedding"
        elif relation:
            query_embedding = self.get_embedding(relation)
            search_field = "relation_embedding"
        elif obj:
            query_embedding = self.get_embedding(obj)
            search_field = "object_embedding"
        else:
            return []  # No search criteria provided
        
        # Search for matching relationships
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field=search_field,
            param=search_params,
            limit=limit,
            output_fields=["subject", "relation", "object", "sentence", "source_file"]
        )
        
        # Format results
        matching_relationships = []
        for hits in results:
            for hit in hits:
                matching_relationships.append({
                    "id": hit.id,
                    "subject": hit.subject,
                    "relation": hit.relation,
                    "object": hit.object,
                    "sentence": hit.sentence,
                    "source_file": hit.source_file,
                    "distance": hit.distance
                })
        
        return matching_relationships
    
    def process_file(self, input_file: str) -> bool:
        """
        Process a single JSON file containing structured data
        
        Args:
            input_file (str): Path to input JSON file
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                structured_data = json.load(f)
            
            # Get source file name
            source_file = os.path.basename(input_file)
            
            # Process structured data
            processed_data = self.process_structured_data(structured_data, source_file)
            
            # Create Milvus collections
            collections = self._create_collections()
            
            # Insert processed data into Milvus
            self.insert_to_milvus(collections, processed_data)
            
            print(f"Successfully processed {input_file}")
            return True
        
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            return False
    
    def process_directory(self, input_dir: str) -> Tuple[int, int]:
        """
        Process all JSON files in a directory
        
        Args:
            input_dir (str): Path to input directory
            
        Returns:
            tuple: (success_count, total_count)
        """
        # Create Milvus collections (do this once for all files)
        collections = self._create_collections()
        
        success_count = 0
        total_count = 0
        
        # Get list of JSON files in directory
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        for filename in tqdm(json_files, desc="Processing files"):
            input_path = os.path.join(input_dir, filename)
            
            try:
                # Read input file
                with open(input_path, 'r', encoding='utf-8') as f:
                    structured_data = json.load(f)
                
                # Process structured data
                processed_data = self.process_structured_data(structured_data, filename)
                
                # Insert processed data into Milvus
                self.insert_to_milvus(collections, processed_data)
                
                success_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
            
            total_count += 1
        
        return success_count, total_count

def main():
    parser = argparse.ArgumentParser(description='Generate GloVe embeddings and store in Milvus')
    parser.add_argument('input', help='Input JSON file or directory (output from TextStructurer)')
    parser.add_argument('--glove-model', default='glove-wiki-gigaword-300', 
                        help='GloVe model to use (default: glove-wiki-gigaword-300)')
    parser.add_argument('--milvus-host', default='localhost', 
                        help='Milvus server host (default: localhost)')
    parser.add_argument('--milvus-port', default='19530', 
                        help='Milvus server port (default: 19530)')
    
    args = parser.parse_args()
    
    # Initialize the embedder
    embedder = TextEmbedder(
        glove_model=args.glove_model,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port
    )
    
    # Process files
    if os.path.isdir(args.input):
        # Process all JSON files in directory
        success_count, total_count = embedder.process_directory(args.input)
        print(f"Processed {success_count} of {total_count} files successfully")
    else:
        # Process single file
        if embedder.process_file(args.input):
            print("Processing completed successfully")
        else:
            print("Processing failed")

if __name__ == '__main__':
    main()
