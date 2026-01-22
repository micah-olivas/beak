#!/usr/bin/env python3
"""
ESM embedding generation script to run inside Docker container
"""

import argparse
import torch
from pathlib import Path
from Bio import SeqIO
import pickle
import sys


def generate_embeddings(
    input_fasta: str,
    output_dir: str,
    model_name: str,
    repr_layers: list,
    include_mean: bool,
    include_per_tok: bool,
    gpu_id: int
):
    """Generate ESM embeddings for sequences in FASTA file"""
    
    print(f"Loading model: {model_name}")
    
    # Import ESM
    try:
        import esm
    except ImportError:
        print("ERROR: ESM not installed")
        sys.exit(1)
    
    # Load model
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    
    # Move to GPU
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    # Read sequences
    print(f"Reading sequences from {input_fasta}")
    sequences = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(input_fasta, "fasta")]
    print(f"Found {len(sequences)} sequences")
    
    # Prepare batch converter
    batch_converter = alphabet.get_batch_converter()
    
    # Process sequences
    embeddings = {}
    per_tok_embeddings = {} if include_per_tok else None
    
    # Process in batches to handle memory
    batch_size = 1  # Process one at a time for safety with large proteins
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        
        print(f"Processing {i+1}/{len(sequences)}: {batch_labels[0]}")
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=repr_layers, return_contacts=False)
        
        # Extract embeddings for each layer
        for j, label in enumerate(batch_labels):
            # Mean pooled embeddings (across sequence length)
            if include_mean:
                layer_embeddings = {}
                for layer in repr_layers:
                    # results["representations"][layer] shape: (batch, seq_len, embed_dim)
                    # Take mean over sequence length, excluding special tokens
                    seq_len = len(batch_strs[j])
                    layer_emb = results["representations"][layer][j, 1:seq_len+1].mean(0)
                    layer_embeddings[f"layer_{layer}"] = layer_emb.cpu().numpy()
                
                embeddings[label] = layer_embeddings
            
            # Per-token embeddings
            if include_per_tok:
                layer_tok_embeddings = {}
                for layer in repr_layers:
                    seq_len = len(batch_strs[j])
                    layer_tok_emb = results["representations"][layer][j, 1:seq_len+1]
                    layer_tok_embeddings[f"layer_{layer}"] = layer_tok_emb.cpu().numpy()
                
                per_tok_embeddings[label] = layer_tok_embeddings
    
    # Save embeddings
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if include_mean:
        mean_file = output_path / "mean_embeddings.pkl"
        with open(mean_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Saved mean embeddings to {mean_file}")
    
    if include_per_tok:
        tok_file = output_path / "per_token_embeddings.pkl"
        with open(tok_file, 'wb') as f:
            pickle.dump(per_tok_embeddings, f)
        print(f"Saved per-token embeddings to {tok_file}")
    
    print("✓ Embedding generation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESM embeddings")
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", required=True, help="ESM model name")
    parser.add_argument("--repr-layers", nargs="+", type=int, default=[-1], 
                        help="Representation layers to extract")
    parser.add_argument("--include-mean", action="store_true", 
                        help="Include mean-pooled embeddings")
    parser.add_argument("--include-per-tok", action="store_true",
                        help="Include per-token embeddings")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    args = parser.parse_args()
    
    generate_embeddings(
        input_fasta=args.input,
        output_dir=args.output,
        model_name=args.model,
        repr_layers=args.repr_layers,
        include_mean=args.include_mean,
        include_per_tok=args.include_per_tok,
        gpu_id=args.gpu
    )