import torch
import requests
import pandas as pd
# from transformers import EsmModel, EsmTokenizer
from transformers import AutoTokenizer, EsmForProteinFolding
import time
from tqdm import tqdm
import os

def get_protein_sequence(ensp_id, max_retries=3, backoff_factor=2):
    """
    Fetch protein sequence from Ensembl using ENSP ID with retry logic and rate limiting.
    """
    base_url = "https://rest.ensembl.org/sequence/id"
    url = f"{base_url}/{ensp_id}?content-type=text/plain"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={"User-Agent": "ProteinEmbeddingPipeline"})
            
            if response.status_code == 429:  # Too Many Requests
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
                continue
                
            if response.ok:
                return response.text.strip()
            elif response.status_code == 404:
                print(f"Protein {ensp_id} not found in Ensembl")
                return None
            else:
                print(f"Error {response.status_code} for {ensp_id}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {ensp_id}: {e}")
            time.sleep(backoff_factor ** attempt)
    
    return None

def process_batch(batch_ids, tokenizer, model, device, batch_size=8):
    """Process a batch of protein IDs and generate embeddings"""
    results = []
    
    for i in range(0, len(batch_ids), batch_size):
        current_batch = batch_ids[i:i+batch_size]
        batch_data = []
        
        # Get sequences for the current batch
        for ensp_id in current_batch:
            try:
                sequence = get_protein_sequence(ensp_id)
                if sequence:
                    batch_data.append((ensp_id, sequence))
            except Exception as e:
                print(f"Error getting sequence for {ensp_id}: {e}")
        
        if not batch_data:
            continue
            
        # Process valid sequences
        for ensp_id, sequence in batch_data:
            try:
                # Process the full sequence without truncation
                inputs = tokenizer(sequence, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                # Get token embeddings and exclude [CLS] and [EOS]
                token_embeddings = outputs.last_hidden_state[0, 1:-1]

                # Mean pooling to get a fixed-size vector
                mean_embedding = token_embeddings.mean(dim=0).cpu().numpy()

                # Save as row: [ENSP_ID, dim1, dim2, ..., dimN]
                row = {"string_id": '9606.' + ensp_id}
                row.update({f"feature_{i}": val for i, val in enumerate(mean_embedding)})
                results.append(row)

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory for {ensp_id} with sequence length {len(sequence)}. Skipping.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Error processing {ensp_id}: {e}")
                
        # Free up GPU memory
        torch.cuda.empty_cache()
        
    return results

def main():
    output_dir = "/itf-fi-ml/shared/users/ziyuzh/svm/data/esmfold"
    output_file = os.path.join(output_dir, "protein_embeddings.csv")
    checkpoint_file = os.path.join(output_dir, "protein_embeddings_checkpoint.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load protein IDs
    gene_id_df = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/ppi_full_emb.csv')
    gene_list = gene_id_df['string_id'].str.replace(r'^9606\.', '', regex=True).tolist()
    
    # Check if checkpoint exists and load already processed proteins
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_ids = set(checkpoint_df['string_id'].tolist())
        print(f"Loaded {len(processed_ids)} already processed proteins from checkpoint")
        
        # Filter gene list to only include unprocessed proteins
        gene_list = [g for g in gene_list if g not in processed_ids]
    
    if not gene_list:
        print("All proteins have been processed already!")
        return
        
    # Setup GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available. Please run on a machine with CUDA support.")
    device = torch.device("cuda:1")
    
    # Load model and tokenizer
    # model_name = "facebook/esm2_t33_650M_UR50D"
    model_name = 'facebook/esmfold_v1'
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name)
    model.eval()
    torch.cuda.empty_cache()
    model.to(device)
    
    # Process in smaller batches
    batch_size = 100  # Process 100 proteins at a time
    all_results = []
    
    for i in tqdm(range(0, len(gene_list), batch_size)):
        batch_ids = gene_list[i:i+batch_size]
        batch_results = process_batch(batch_ids, tokenizer, model, device)
        all_results.extend(batch_results)
        
        # Save checkpoint after each batch
        if all_results:
            batch_df = pd.DataFrame(all_results)
            
            # If checkpoint exists, append to it
            if os.path.exists(checkpoint_file):
                existing_df = pd.read_csv(checkpoint_file)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                combined_df.to_csv(checkpoint_file, index=False)
            else:
                batch_df.to_csv(checkpoint_file, index=False)
                
            # Clear batch results to save memory
            all_results = []
    
    # Combine checkpoint with any new results for final output
    if os.path.exists(checkpoint_file):
        final_df = pd.read_csv(checkpoint_file)
        final_df.to_csv(output_file, index=False)
        print(f"Saved {len(final_df)} protein embeddings to {output_file}")

if __name__ == "__main__":
    main()