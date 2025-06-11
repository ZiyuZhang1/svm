import pandas as pd
import os

def read_data(disease, dga, features):
    columns_to_keep = [col for col in features.columns if col.startswith('feature') or col.startswith('string')]
    df = features[columns_to_keep]

    pos_genes_list = dga[dga['disease_id']==disease]['string_id']
    df['label'] = df['string_id'].isin(pos_genes_list).astype(int)

    # X = df.loc[:, df.columns.str.startswith("feature_")].to_numpy()
    y = df['label'].to_numpy()
    df.set_index('string_id', inplace=True)
    df.drop(columns='label', inplace=True)
    return df, y

def read_data_timecut(disease, dga, features,time):
    pos_genes_list = dga[dga['disease_id']==disease]['string_id']
    columns_to_keep = [col for col in features.columns if 'feature' in col or col.startswith('string')]
    df = features[columns_to_keep]
    df['label'] = df['string_id'].isin(pos_genes_list).astype(int)
    df['test'] = df['string_id'].isin(dga[(dga['disease_id'] == disease) & (dga['first_pub_year'] > time)]['string_id']).astype(int)

    # X = df.loc[:, df.columns.str.startswith("feature_")].to_numpy()
    y = df['label'].to_numpy()
    df.set_index('string_id', inplace=True)
    df.drop(columns='label', inplace=True)
    return df, y


def get_feature(root, feature_name):
    if feature_name == 'ppi_align':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_emb_aligned.csv'))
    elif feature_name == 'ppi':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_emb.csv'))
    elif feature_name == 'biograd':
        feature_df = pd.read_csv(os.path.join(root,'data/biograd/biograd_full_emb.csv'))
    elif feature_name == 'prose':
        feature_df = pd.read_csv(os.path.join(root, 'data/prose/data/prose_emb_full.csv'))
    # elif feature_name == 'ppi_400':
    #     feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_400_emb.csv'))
    elif feature_name == 'ppi_2016':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_2016_emb.csv'))
    elif feature_name == 'ppi_2019':
        feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_2019_emb.csv'))
    # elif feature_name == 'ppi_2013':
    #     feature_df = pd.read_csv(os.path.join(root,'data/ppi_full_2013_emb.csv'))
    elif feature_name == 'uniport':
        feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/seq_emb/human_uniport_seqemb.csv'))
    # elif feature_name == 't5_align':
    #     feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/T5/T5_align.csv'))
    elif feature_name == 'gene2vec':
        feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/expression_emb/exp_emb.csv'))
        # feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/GENE2VEC/GENE2VEC_align.csv'))
    
    elif feature_name == 'scgpt':
        feature_df = pd.read_csv(os.path.join(root,'data/scgpt/scgpt_full.csv'))
        # feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/expression_emb/SCGPT-HUMAN/scgpt.csv'))
    elif feature_name == 'bioconcept':
        # feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/text_minning_2015/text_features.csv'))
        # feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/txt/BIOCONCEPTVEC-FASTTEXT/text.csv'))
        feature_df = pd.read_csv(os.path.join(root,'data/bioconcept/bioconcept_full.csv'))
    # elif feature_name == 'go_svd':
    #     feature_df = pd.read_csv(os.path.join(root,'data/GO/GO_2023_features.csv'))
    # elif feature_name == 'go_all':
    #     feature_df = pd.read_csv(os.path.join(root,'data/GO/GO_2023_all_features_aligned.csv'))
    elif feature_name == 'esm2':
        feature_df = pd.read_csv(os.path.join(root,'data/esmfold/esm2.csv'))
        # feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/ESM2/ESM2_align.csv'))
    # elif feature_name == 'MASHUP':
    #     feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/MASHUP/MASHUP_align.csv'))
    # elif feature_name == 'GENEPT_MODEL3':
    #     feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/GENEPT_MODEL3/GENEPT_MODEL3_align.csv'))    
    # elif feature_name == 'GF_12L95M':
    #     feature_df = pd.read_csv(os.path.join(root,'data/pre_processed_features/GF_12L95M/GF_12L95M_align.csv')) 

    return feature_df