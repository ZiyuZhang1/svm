import pandas as pd
import os
# import mygene
filename = 'GF_12L95M'
emb_df = pd.read_csv(os.path.join('/itf-fi-ml/shared/users/ziyuzh/svm/data/pre_processed_features/',filename,filename+'_emb.csv'),header = None)
gene_list = pd.read_csv(os.path.join('/itf-fi-ml/shared/users/ziyuzh/svm/data/pre_processed_features/',filename,filename+'_genelist.txt'),header = None)

# mg = mygene.MyGeneInfo()
# map_df = mg.getgenes(gene_list[0], fields='name,symbol,entrezgene,taxid',as_dataframe=True)

# import os
# local_stringdb = '/itf-fi-ml/shared/users/ziyuzh/svm/data/stringdb/2023'

# ppidf = pd.read_csv(os.path.join(local_stringdb,'9606.protein.info.v12.0.txt'), sep='\t', header=0, usecols=['#string_protein_id', 'preferred_name'])
# ppidf['preferred_name'] = ppidf['preferred_name'].str.upper()
# stringId2name = ppidf.set_index('#string_protein_id')['preferred_name'].to_dict()
# name2stringId = ppidf.set_index('preferred_name')['#string_protein_id'].to_dict()
# ppidf = pd.read_csv(os.path.join(local_stringdb,'9606.protein.aliases.v12.0.txt'), sep='\t', header=0, usecols=['#string_protein_id', 'alias']).drop_duplicates(['alias'], keep='first')
# ppidf['alias'] = ppidf['alias'].str.upper()
# aliases2stringId = ppidf.set_index('alias')['#string_protein_id'].to_dict()

# def string_convert(gene):
#     if gene in name2stringId.keys():
#         return name2stringId[gene]
#     elif gene in aliases2stringId.keys():
#         return aliases2stringId[gene]
#     else:
#         return None
    
# entrz2string = dict()
# for gene in map_df['symbol']:
#     entrz2string[gene] = string_convert(gene)

# map_id = map_df[['_id','symbol']]
# map_id['string_id'] = map_id['symbol'].map(entrz2string)
# map_id = map_id.reset_index()
map_id = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/data/pre_processed_features/txt/BIOCONCEPTVEC-FASTTEXT/map_id.csv')
df_combined = emb_df.loc[map_id[~map_id['string_id'].isna()].index]
df_combined['string_id'] = map_id[~map_id['string_id'].isna()]['string_id']
new_columns = ['string_id'] + [f'feature_{i}' for i, col in enumerate(df_combined.columns) if col != 'string_id']

# Reorder the DataFrame so that 'string_id' is the first column
df_combined = df_combined[['string_id'] + [col for col in df_combined.columns if col != 'string_id']]
df_combined.columns = new_columns
outpath = os.path.join('/itf-fi-ml/shared/users/ziyuzh/svm/data/pre_processed_features/',filename,filename+'_align.csv')
df_combined = df_combined.drop_duplicates(subset='string_id')
df_combined.to_csv(outpath,index=False)