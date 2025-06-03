import pandas as pd
import os

def read_data(disease, dga, features):
    pos_genes_list = dga[dga['disease_id']==disease]['string_id']
    df = features
    df['label'] = df['string_id'].isin(pos_genes_list).astype(int)

    X = df.loc[:, df.columns.str.startswith("feature_")].to_numpy()
    y = df['label'].to_numpy()
    return df, X, y


def get_feature(root, feature_name):
    if feature_name == 'ppi':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_emb.csv'))
    elif feature_name == 'ppi_400':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_400_emb.csv'))
    elif feature_name == 'ppi_700':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_700_emb.csv'))
    elif feature_name == 'biograd':
        feature_df = pd.read_csv(os.path.join(root,'data/biograd/biograd_full_emb.csv'))
    elif feature_name == 'seq':
        feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/seq_emb/human_uniport_seqemb.csv'))
    elif feature_name == 'exp':
        # feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/expression_emb/SCGPT-HUMAN/scgpt.csv'))
        feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/expression_emb/exp_emb.csv'))
    elif feature_name == 'txt':
        # feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/text_minning_2015/text_features.csv'))
        feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/txt/BIOCONCEPTVEC-FASTTEXT/text.csv'))

    elif feature_name == 'go_svd':
        feature_df = pd.read_csv(os.path.join(root,'data/GO/GO_2023_features.csv'))
    elif feature_name == 'go_all':
        feature_df = pd.read_csv(os.path.join(root,'data/GO/GO_2023_all_features.csv'))
    elif feature_name == 'go_exp':
        feature_df = pd.read_csv(os.path.join(root,'data/GO/GO_exp_all_features.csv'))
    
    return feature_df