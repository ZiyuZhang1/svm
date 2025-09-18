import os
import time
import math
import pickle
import numpy as np
import gseapy as gp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

# -------- Paths / constants --------
UNI2NAME_PATH = '/itf-fi-ml/shared/users/ziyuzh/svm/data/uniport_id/uni2name.pkl'
# IN_ROOT = '/itf-fi-ml/shared/users/ziyuzh/svm/results/2019_lf_bag_cv_all_save_pred'
# OUT_ROOT = '/itf-fi-ml/shared/users/ziyuzh/svm/results/2019_lf_bag_cv_pred_sim'

IN_ROOT = '/itf-fi-ml/shared/users/ziyuzh/svm/results/2019_df_pred'
OUT_ROOT = '/itf-fi-ml/shared/users/ziyuzh/svm/results/2019_df_pred_sim'

TIME_FLAG = 2019                 # change to 2017 if needed
PVALUE_CUTOFF = 0.01
RATIOS = np.linspace(0, 0.5, 10) # inclusive of 0 and 0.5
MAX_WORKERS = 5  # leave headroom

# Ensure output dir exists (once in parent)
os.makedirs(OUT_ROOT, exist_ok=True)

# -------- Globals loaded per worker --------
_uni2name_dict = None

def _ensure_uni2name_loaded():
    """Load the UniProt->gene-name dict once per process."""
    global _uni2name_dict
    if _uni2name_dict is None:
        with open(UNI2NAME_PATH, 'rb') as fh:
            _uni2name_dict = pickle.load(fh)

def _get_enrich_db(time_flag: int):
    if time_flag == 2019:
        return [
            'GO_Biological_Process_2021',
            'GO_Cellular_Component_2021',
            'GO_Molecular_Function_2021',
            'KEGG_2019_Human',
            'Reactome_2022'
        ]
    elif time_flag == 2017:
        return [
            'GO_Biological_Process_2021',
            'GO_Cellular_Component_2021',
            'GO_Molecular_Function_2021',
            'KEGG_2016'
        ]
    # default
    return [
        'GO_Biological_Process_2021',
        'GO_Cellular_Component_2021',
        'GO_Molecular_Function_2021',
        'KEGG_2019_Human',
        'Reactome_2022'
    ]

def _to_list(iterable_like):
    """Normalize pandas Index/Series/ndarray/list/tuple/scalar -> list without truthiness."""
    if iterable_like is None:
        return []
    try:
        return list(iterable_like)
    except TypeError:
        return [iterable_like]

def _ids_to_gene_names(input_ids):
    """Map UniProt IDs -> unique gene symbols using the loaded dict."""
    _ensure_uni2name_loaded()
    ids = _to_list(input_ids)
    gene_names = set()
    for unid in ids:
        gene_list = _uni2name_dict.get(unid, [])
        gene_names.update(gene_list)
    return list(gene_names)

@lru_cache(maxsize=2048)
def _enriched_set_cached(ids_key: tuple, time_flag: int):
    """
    Cached enrichment call.
    ids_key: tuple of sorted unique UniProt IDs (hashable).
    Returns a frozenset of (Gene_set, Term) tuples.
    """
    gene_names = _ids_to_gene_names(ids_key)
    if len(gene_names) == 0:
        return frozenset()

    enrich_db = _get_enrich_db(time_flag)

    # retry/backoff (handles transient network/rate limits)
    for attempt in range(4):
        try:
            enr = gp.enrichr(
                gene_list=gene_names,
                gene_sets=enrich_db,
                organism='human',
                outdir=None
            )
            enr_df = getattr(enr, 'results', None)
            if enr_df is None or len(enr_df) == 0:
                return frozenset()

            filtered = enr_df.loc[enr_df['Adjusted P-value'] < PVALUE_CUTOFF, ['Gene_set', 'Term']]
            return frozenset(map(tuple, filtered.values))
        except Exception:
            time.sleep(1.5 ** attempt)

    return frozenset()

def enriched_set(input_ids, time_flag: int):
    """Wrapper that normalizes inputs and uses the cached core."""
    ids = _to_list(input_ids)
    if len(ids) == 0:
        return set()
    ids_key = tuple(sorted(set(ids)))
    return set(_enriched_set_cached(ids_key, time_flag))

def calculate_jac_sim(enrich_1: set, enrich_2: set) -> float:
    if len(enrich_1) == 0 and len(enrich_2) == 0:
        return 0.0
    inter = enrich_1 & enrich_2
    union = enrich_1 | enrich_2
    return (len(inter) / len(union)) if len(union) > 0 else 0.0

def _process_single_file(file_path: str, out_root: str, time_flag: int):
    """Worker: load one .pkl, compute similarities, write output .pkl."""
    _ensure_uni2name_loaded()

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Train set enrichment (once)
    train_genes = data.get('train_pos_genes', [])
    enrich_train_set = enriched_set(train_genes, time_flag)

    # Normalize test indices to numpy array
    test_indices = np.asarray(_to_list(data.get('test_genes', [])))

    disease_record = {}

    # Iterate features except reserved keys
    for feature, payload in data.items():
        if feature in ['test_genes', 'train_pos_genes', 'true_label']:
            continue

        final_y_score = np.asarray(payload)
        if final_y_score.size == 0 or test_indices.size == 0:
            disease_record[feature] = [0.0 for _ in RATIOS]
            continue

        # Descending order by score
        order = np.argsort(final_y_score)[::-1]
        sorted_test = test_indices[order]

        sim_scores = []
        n = len(final_y_score)
        for ratio in RATIOS:
            k = int(math.floor(ratio * n))
            if k <= 0:
                sim_scores.append(calculate_jac_sim(enrich_train_set, set()))
                continue
            top_pred_ids = sorted_test[:k]
            enrich_predict_set = enriched_set(top_pred_ids, time_flag)
            jac = calculate_jac_sim(enrich_train_set, enrich_predict_set)
            sim_scores.append(jac)

        disease_record[feature] = sim_scores

    # Save result
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, os.path.basename(file_path))
    with open(out_path, 'wb') as f:
        pickle.dump(disease_record, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path

def _collect_pkl_files(root_dir: str):
    """
    Collect all .pkl files but skip the first 9 entries
    in the directory listing (sorted for stability).
    """
    files = sorted(fn for fn in os.listdir(root_dir) if fn.endswith('.pkl'))
    return [os.path.join(root_dir, fn) for fn in files[9:]]

def main():
    files = _collect_pkl_files(IN_ROOT)
    if len(files) == 0:
        print(f'No .pkl files found in {IN_ROOT}')
        return

    # Avoid oversubscribing workers relative to files
    workers = min(MAX_WORKERS, len(files))
    print(f'Found {len(files)} files (skipped first 9). Starting parallel processing with {workers} workers...')

    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_file = {
            ex.submit(_process_single_file, fp, OUT_ROOT, TIME_FLAG): fp
            for fp in files
        }
        for fut in as_completed(future_to_file):
            src = future_to_file[fut]
            try:
                outp = fut.result()
                completed += 1
                print(f'[OK] {src} -> {outp}')
            except Exception as e:
                failed += 1
                print(f'[FAIL] {src}: {e}')

    print(f'Done. Completed: {completed}, Failed: {failed}')

if __name__ == '__main__':
    main()
