#!/usr/bin/env python3
"""
CLI Search Tool for Test CLIP

This script allows searching for images using text queries via the command line.
It uses the implemented CLIP model, TextEncoder, and SearchEngine.
"""

import argparse
import sys
import os

# Add the project root to the python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clip_model import CLIPModel
from src.text_encoder import TextEncoder
from src.vector_store import VectorStore
from src.search import SearchEngine

def main():
    parser = argparse.ArgumentParser(description='Search for images using text queries.')
    parser.add_argument('query', type=str, help='The text query to search for.')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top results to return.')
    parser.add_argument('--vector-dir', type=str, default='data/vectors', help='Directory containing vector data.')
    parser.add_argument('--metadata-dir', type=str, default='data/metadata', help='Directory containing metadata.')
    parser.add_argument('--model-name', type=str, default='rinna/japanese-clip-vit-b-16', help='Model name to use.')
    
    args = parser.parse_args()
    
    try:
        # Initialize models and stores
        print(f"Initializing CLIP model ({args.model_name})...")
        clip_model = CLIPModel(model_name=args.model_name)
        
        print("Initializing Text Encoder...")
        text_encoder = TextEncoder(clip_model)
        
        print(f"Loading Vector Store from {args.vector_dir} and {args.metadata_dir}...")
        vector_store = VectorStore(args.vector_dir, args.metadata_dir)
        
        print("Initializing Search Engine...")
        search_engine = SearchEngine(vector_store)
        
        # Encode text
        print(f"Encoding query: '{args.query}'...")
        query_vector = text_encoder.encode_text(args.query)
        
        # Search
        print(f"Searching for top {args.top_k} results...")
        results = search_engine.search(query_vector, top_k=args.top_k)
        
        # Display results
        print("\nSearch Results:")
        print("-" * 80)
        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results):
                path = result['image_path']
                score = result['score']
                print(f"{i+1:2d}. {path} (Score: {score:.4f})")
        print("-" * 80)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
