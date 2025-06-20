import torch
import requests
import pandas as pd
from transformers import EsmModel, EsmTokenizer
import time
from tqdm import tqdm
import os
import pickle

def get_protein_sequence(uniprot_id, max_retries=3, backoff_factor=2):
    """
    Fetch protein sequence from UniProt using UniProt ID with retry logic.
    """
    base_url = "https://rest.uniprot.org/uniprotkb"
    url = f"{base_url}/{uniprot_id}.fasta"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={"User-Agent": "ProteinEmbeddingPipeline"})

            if response.status_code == 429:  # Too Many Requests
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
                continue

            if response.ok:
                fasta = response.text
                lines = fasta.split('\n')
                sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
                return sequence
            elif response.status_code == 404:
                print(f"Protein {uniprot_id} not found in UniProt.")
                return None
            else:
                print(f"Error {response.status_code} for {uniprot_id}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed for {uniprot_id}: {e}")
            time.sleep(backoff_factor ** attempt)

    return None

def process_batch(batch_ids, tokenizer, model, device, batch_size=8):
    """Process a batch of UniProt IDs and generate embeddings"""
    results = []

    for i in range(0, len(batch_ids), batch_size):
        current_batch = batch_ids[i:i+batch_size]
        batch_data = []

        # Get sequences for the current batch
        for uniprot_id in current_batch:
            try:
                sequence = get_protein_sequence(uniprot_id)
                if sequence:
                    batch_data.append((uniprot_id, sequence))
            except Exception as e:
                print(f"Error getting sequence for {uniprot_id}: {e}")

        if not batch_data:
            continue

        # Process valid sequences
        for uniprot_id, sequence in batch_data:
            try:
                inputs = tokenizer(sequence, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                token_embeddings = outputs.last_hidden_state[0, 1:-1]
                mean_embedding = token_embeddings.mean(dim=0).cpu().numpy()

                row = {"string_id": uniprot_id}
                row.update({f"feature_{i}": val for i, val in enumerate(mean_embedding)})
                results.append(row)

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory for {uniprot_id} (sequence length {len(sequence)}). Skipping.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Error processing {uniprot_id}: {e}")

        torch.cuda.empty_cache()

    return results

def main():
    output_dir = "/itf-fi-ml/shared/users/ziyuzh/svm/data/esmfold"
    output_file = os.path.join(output_dir, "protein_embeddings.csv")
    checkpoint_file = os.path.join(output_dir, "protein_embeddings_checkpoint.csv")

    os.makedirs(output_dir, exist_ok=True)

    # 🔽 Load UniProt IDs from Pickle set
    with open('/itf-fi-ml/shared/users/ziyuzh/svm/data/id_maps/inter_2019.pkl', 'rb') as f:
        gene_set = pickle.load(f)
    
    gene_list = list(gene_set)  # Convert set to list for batching

    # Check for already processed IDs
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_ids = set(checkpoint_df['string_id'].tolist())
        print(f"Loaded {len(processed_ids)} already processed proteins from checkpoint")

        gene_list = [g for g in gene_list if g not in processed_ids]

    if not gene_list:
        print("All proteins have been processed already!")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available. Please run on a machine with CUDA support.")

    device = torch.device("cuda")

    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading {model_name}...")

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    batch_size = 100
    all_results = []

    for i in tqdm(range(0, len(gene_list), batch_size)):
        batch_ids = gene_list[i:i+batch_size]
        batch_results = process_batch(batch_ids, tokenizer, model, device)
        all_results.extend(batch_results)

        # Save checkpoint
        if all_results:
            batch_df = pd.DataFrame(all_results)
            if os.path.exists(checkpoint_file):
                existing_df = pd.read_csv(checkpoint_file)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                combined_df.to_csv(checkpoint_file, index=False)
            else:
                batch_df.to_csv(checkpoint_file, index=False)

            all_results = []

    if os.path.exists(checkpoint_file):
        final_df = pd.read_csv(checkpoint_file)
        final_df.to_csv(output_file, index=False)
        print(f"Saved {len(final_df)} protein embeddings to {output_file}")

if __name__ == "__main__":
    main()
