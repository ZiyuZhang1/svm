import pandas as pd
import os
from IPython.display import display, HTML

def rewrite_all_results(root, ban_list, fnum, fused):
    top = False
    if top == True:
        num = 12
    all_results = []
    counter = 0
    if fused:
        for disease in os.listdir(root):
            if disease.startswith('ICD') and disease.split('_')[-1].split('.')[0] not in ban_list:
                result_df = pd.read_csv(os.path.join(root,disease))
                if len(result_df) == fnum:
                    counter += 1
                    mean_df = result_df.groupby(['method','para'])[['top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30']].mean().reset_index()
                    # Add disease information
                    mean_df['disease'] = disease.split('.')[0]
                    # Append to all_results list
                    all_results.append(mean_df)
                else:
                    print(disease, 'not enough feature')
            if top == True:
                if counter == num:
                    break
        # Concatenate all results into a single DataFrame
        final_result = pd.concat(all_results, ignore_index=True)
        if top == True:
            final_result.to_csv(os.path.join(root,f'all_disease_{num}.csv'),index=False)
        else:
            final_result.to_csv(os.path.join(root,'all_disease.csv'),index=False)
        return final_result
    else:
        for disease in os.listdir(root):
            if disease.startswith('ICD'):
                counter += 1
                result_df = pd.read_csv(os.path.join(root,disease))
                mean_df = result_df.groupby(['method'])[['top_recall_25','top_recall_300','top_recall_10%', 'top_precision_10%', 'max_precision_10%','top_recall_30%', 'top_precision_30%', 'max_precision_30%','pm_0.5%','pm_1%','pm_5%','pm_10%','pm_15%','pm_20%','pm_25%','pm_30%','auroc',"rank_ratio",'bedroc_1','bedroc_5','bedroc_10','bedroc_30']].mean().reset_index()
                # Add disease information
                mean_df['disease'] = disease.split('.')[0]
                # Append to all_results list
                all_results.append(mean_df)
            if top == True:
                if counter == num:
                    break
        # Concatenate all results into a single DataFrame
        final_result = pd.concat(all_results, ignore_index=True)
        if top == True:
            final_result.to_csv(os.path.join(root,f'all_disease_{num}.csv'),index=False)
        else:
            final_result.to_csv(os.path.join(root,'all_disease.csv'),index=False)
        return final_result

def create_summary(results,col_name, fused):
    # Create an empty list to store results
    summary_list = []
    if fused == False:
        # Grouping by 'method' and calculating mean and std for selected metrics
        for method, subdf in results.groupby(col_name):
            num_cols = subdf.select_dtypes(include='number').columns
            mean_values = subdf[num_cols].mean()
            std_values  = subdf[num_cols].std()
            summary_list.append({
                'method': method,
                'top_recall_25_mean': mean_values['top_recall_25'], 'top_recall_25_std': std_values['top_recall_25'],
                'top_recall_300_mean': mean_values['top_recall_300'], 'top_recall_300_std': std_values['top_recall_300'],
                'top_recall_10%_mean': mean_values['top_recall_10%'], 'top_recall_10%_std': std_values['top_recall_10%'],
                'top_precision_10_mean': mean_values['top_precision_10%'], 'top_precision_10_std': std_values['top_precision_10%'],
                'max_precision_10_mean': mean_values['max_precision_10%'], 'max_precision_10_std': std_values['max_precision_10%'],
                'top_recall_30%_mean': mean_values['top_recall_30%'], 'top_recall_30%_std': std_values['top_recall_30%'],
                'top_precision_30_mean': mean_values['top_precision_30%'], 'top_precision_10_std': std_values['top_precision_30%'],
                'max_precision_30_mean': mean_values['max_precision_30%'], 'max_precision_10_std': std_values['max_precision_30%'],
                'pm_0.5%': mean_values['pm_0.5%'], 'pm_0.5%_std': std_values['pm_0.5%'],
                'pm_1%': mean_values['pm_1%'], 'pm_1%_std': std_values['pm_1%'],
                'pm_5%': mean_values['pm_5%'], 'pm_5%_std': std_values['pm_5%'],
                'pm_10%': mean_values['pm_10%'], 'pm_10%_std': std_values['pm_10%'],
                'pm_15%': mean_values['pm_15%'], 'pm_15%_std': std_values['pm_15%'],            
                'pm_20%': mean_values['pm_20%'], 'pm_20%_std': std_values['pm_20%'],
                'pm_25%': mean_values['pm_25%'], 'pm_1%_std': std_values['pm_25%'],
                'pm_30%': mean_values['pm_30%'], 'pm_30%_std': std_values['pm_30%'],
                'auroc_mean': mean_values['auroc'], 'auroc_std': std_values['auroc'],
                'rank_ratio_mean': mean_values['rank_ratio'], 'rank_ratio_std': std_values['rank_ratio'],
                'bedroc_1_mean': mean_values['bedroc_1'], 'bedroc_1_std': std_values['bedroc_1'],
                'bedroc_5_mean': mean_values['bedroc_5'], 'bedroc_5_std': std_values['bedroc_5'],
                'bedroc_10_mean': mean_values['bedroc_10'], 'bedroc_10_std': std_values['bedroc_10'],
                'bedroc_30_mean': mean_values['bedroc_30'], 'bedroc_30_std': std_values['bedroc_30'],
                'weights_1_mean': mean_values['weights_1'], 'weights_1_std': std_values['weights_1'],
                'weights_2_mean': mean_values['weights_2'], 'weights_2_std': std_values['weights_2'],
                'weights_3_mean': mean_values['weights_3'], 'weights_3_std': std_values['weights_3']
            })

        # Convert the list of dictionaries into a DataFrame
        summary_df = pd.DataFrame(summary_list)
    else:
                # Grouping by 'method' and calculating mean and std for selected metrics
        for paras, subdf in results.groupby(col_name):
            num_cols = subdf.select_dtypes(include='number').columns
            mean_values = subdf[num_cols].mean()
            std_values  = subdf[num_cols].std()
            summary_list.append({
                'method': paras[0],
                'para': paras[1],
                'top_recall_25_mean': mean_values['top_recall_25'], 'top_recall_25_std': std_values['top_recall_25'],
                'top_recall_300_mean': mean_values['top_recall_300'], 'top_recall_300_std': std_values['top_recall_300'],
                'top_recall_10%_mean': mean_values['top_recall_10%'], 'top_recall_10%_std': std_values['top_recall_10%'],
                'top_precision_10_mean': mean_values['top_precision_10%'], 'top_precision_10_std': std_values['top_precision_10%'],
                'max_precision_10_mean': mean_values['max_precision_10%'], 'max_precision_10_std': std_values['max_precision_10%'],
                'top_recall_30%_mean': mean_values['top_recall_30%'], 'top_recall_30%_std': std_values['top_recall_30%'],
                'top_precision_30_mean': mean_values['top_precision_30%'], 'top_precision_10_std': std_values['top_precision_30%'],
                'max_precision_30_mean': mean_values['max_precision_30%'], 'max_precision_10_std': std_values['max_precision_30%'],
                'pm_0.5%': mean_values['pm_0.5%'], 'pm_0.5%_std': std_values['pm_0.5%'],
                'pm_1%': mean_values['pm_1%'], 'pm_1%_std': std_values['pm_1%'],
                'pm_5%': mean_values['pm_5%'], 'pm_5%_std': std_values['pm_5%'],
                'pm_10%': mean_values['pm_10%'], 'pm_10%_std': std_values['pm_10%'],
                'pm_15%': mean_values['pm_15%'], 'pm_15%_std': std_values['pm_15%'],            
                'pm_20%': mean_values['pm_20%'], 'pm_20%_std': std_values['pm_20%'],
                'pm_25%': mean_values['pm_25%'], 'pm_1%_std': std_values['pm_25%'],
                'pm_30%': mean_values['pm_30%'], 'pm_30%_std': std_values['pm_30%'],
                'auroc_mean': mean_values['auroc'], 'auroc_std': std_values['auroc'],
                'rank_ratio_mean': mean_values['rank_ratio'], 'rank_ratio_std': std_values['rank_ratio'],
                'bedroc_1_mean': mean_values['bedroc_1'], 'bedroc_1_std': std_values['bedroc_1'],
                'bedroc_5_mean': mean_values['bedroc_5'], 'bedroc_5_std': std_values['bedroc_5'],
                'bedroc_10_mean': mean_values['bedroc_10'], 'bedroc_10_std': std_values['bedroc_10'],
                'bedroc_30_mean': mean_values['bedroc_30'], 'bedroc_30_std': std_values['bedroc_30'],
                'weights_1_mean': mean_values['weights_1'], 'weights_1_std': std_values['weights_1'],
                'weights_2_mean': mean_values['weights_2'], 'weights_2_std': std_values['weights_2'],
                'weights_3_mean': mean_values['weights_3'], 'weights_3_std': std_values['weights_3']
            })

        # Convert the list of dictionaries into a DataFrame
        summary_df = pd.DataFrame(summary_list)
    return summary_df

def show_table(root, ban_list, fnum, fused, input_weights):
    if fused:
        final_result = rewrite_all_results(root, ban_list, fnum, fused=True)
        if input_weights:
            weight_df = final_result.copy()
            
            weight_df[['para', 'weights_1', 'weights_2', 'weights_3']] = (
                weight_df['para'].str.split('-', expand=True)
            )

            # convert the weight columns to float
            weight_df[['weights_1', 'weights_2', 'weights_3']] = weight_df[['weights_1', 'weights_2', 'weights_3']].astype(float)

            all_sum = create_summary(weight_df, ['method', 'para'], fused=True)
            show_df = all_sum.sort_values(by='method', ascending=True) \
                .loc[:, all_sum.columns.str.contains(
                    'method|para|top_recall_25_mean|top_recall_300_mean|top_recall_10%_mean|top_recall_30%_mean|auroc_mean|rank_ratio_mean|bedroc_1_mean|bedroc_5_mean|bedroc_10_mean|bedroc_30_mean|weights_1_mean|weights_2_mean|weights_3_mean', case=False)] \
                .round(3).rename(columns=lambda x: x.replace('_mean', ''))
            display(HTML(show_df.to_html(index=False).replace('<table', '<table style="font-size:13px; white-space:nowrap;"')))
            return weight_df, show_df
        else:
            all_sum = create_summary(final_result, ['method', 'para'], fused=True)
            show_df = all_sum.sort_values(by='method', ascending=True) \
                .loc[:, all_sum.columns.str.contains(
                    'method|para|top_recall_25_mean|top_recall_300_mean|top_recall_10%_mean|top_recall_30%_mean|auroc_mean|bedroc_1_mean|bedroc_5_mean|bedroc_10_mean|bedroc_30_mean|rank_ratio', case=False)] \
                .round(3).rename(columns=lambda x: x.replace('_mean', ''))
            display(HTML(show_df.to_html(index=False).replace('<table', '<table style="font-size:13px; white-space:nowrap;"')))
            return final_result, show_df
    else:
        final_result = rewrite_all_results(root, fused=False)
        all_sum = create_summary(final_result, 'method', fused=False)
        if 'random_pos_negative_bagging' in all_sum['method'].unique():
            method_order = ['random_negativeauroc', 'random_negative_bagging', 'random_pos_negative_bagging']
            all_sum['method'] = pd.Categorical(all_sum['method'], categories=method_order, ordered=True)
        
        show_df = all_sum.sort_values(by='method', ascending=True) \
            .loc[:, all_sum.columns.str.contains(
                'method||top_recall_25_mean|top_recall_300_mean|top_recall_10%_mean|top_recall_30%_mean|auroc_mean|bedroc_1_mean|bedroc_5_mean|bedroc_10_mean|bedroc_30_mean|rank_ratio', case=False)] \
            .round(3).rename(columns=lambda x: x.replace('_mean', ''))
        
        display(HTML(show_df.to_html(index=False).replace('<table', '<table style="font-size:13px; white-space:nowrap;"')))
        return final_result, show_df

    

icd_dict = {
    'Certain infectious and parasitic diseases': ['A00','B99'],
    'Neoplasms': ['C00','D48'],
    'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism': ['D50','D89'],
    'Endocrine, nutritional and metabolic diseases': ['E00','E90'],
    'Mental and behavioural disorders': ['F00','F99'],
    'Diseases of the nervous system': ['G00','G99'],
    'Diseases of the eye and adnexa': ['H00','H59'],
    'Diseases of the ear and mastoid process': ['H60','H95'],
    'Diseases of the circulatory system': ['I00','I99'],
    'Diseases of the respiratory system': ['J00','J99'],
    'Diseases of the digestive system': ['K00','K93'],
    'Diseases of the skin and subcutaneous tissue': ['L00','L99'],
    'Diseases of the musculoskeletal system and connective tissue': ['M00','M99'],
    'Diseases of the genitourinary system': ['N00','N99']
}

def find_disease_category(icd_code):
    icd_num = icd_code.split('_')[1]  # Extract ICD-10 code
    # print(icd_num)
    icd_letter = icd_num[0]  # Extract first letter (C, D, etc.)
    icd_number = int(icd_num[1:])  # Extract numeric part

    for category, (start, end) in icd_dict.items():
        start_letter, start_num = start[0], int(start[1:])
        end_letter, end_num = end[0], int(end[1:])

        if start_letter <= icd_letter <= end_letter:  # Ensure it's within the letter range
            if start_letter == icd_letter and start_num <= icd_number:
                return category
            if end_letter == icd_letter and icd_number <= end_num:
                return category
            if start_letter < icd_letter < end_letter:
                return category  # Covers ranges like C00-D48
        
    return 'Unknown Category'

def disease_catygory(results):
    mapped_results = {icd: find_disease_category(icd) for icd in results['disease']}
    results['category'] = results['disease'].map(mapped_results)

    collected_dfs = []
    disease_num = dict()
    for category in results['category'].unique().tolist():
        subdf = results[results['category'] == category].copy()
        sum_df = create_summary(subdf, 'method', fused=False)
        sum_df['category'] = category
        collected_dfs.append(sum_df)
        disease_num[category] = len(subdf) / 2

    final_df = pd.concat(collected_dfs, ignore_index=True)

    category_order = final_df.groupby('category', observed=True)['auroc_mean'].mean().sort_values(ascending=False).index.tolist()
    final_df['category'] = pd.Categorical(final_df['category'], categories=category_order, ordered=True)

    show_df = final_df.sort_values(by=['category', 'auroc_mean'], ascending=[True, False]) \
        .loc[:, final_df.columns.str.contains('method|para|top_recall_25_mean|top_recall_300_mean|top_recall_10%_mean|top_recall_30%_mean|auroc_mean|rank_ratio_mean |bedroc_1_mean|bedroc_5_mean|bedroc_10_mean|bedroc_30_mean|weights_1_mean|weights_2_mean|weights_2_mean', case=False)] \
        .round(3) \
        .rename(columns=lambda x: x.replace('_mean', ''))

    # Highlight specific method
    def highlight_method(val):
        if isinstance(val, str) and val == 'random_pos_negative_bagging':
            return '<span style="color:red; font-weight:bold;">random_pos_negative_bagging</span>'
        return val

    styled_df = show_df.copy()
    if 'method' in styled_df.columns:
        styled_df['method'] = styled_df['method'].apply(highlight_method)

    # Assign background colors per category
    category_colors = {}
    base_colors = ['#ffdddd', "#dbf7db"]
    for i, cat in enumerate(category_order):
        category_colors[cat] = base_colors[i % len(base_colors)]

    def row_style(row):
        color = category_colors.get(row['category'], '#ffffff')
        return [f'background-color: {color}'] * len(row)


    # Display styled table
    from IPython.display import display, HTML
    display(HTML(styled_df.style.apply(row_style, axis=1).to_html(escape=False)))

def prcess_and_save_xlsx(fused_2019,all_avg_df,out_path):
    full_fused_2019 = fused_2019.round(3)
    mapped_results = {icd: find_disease_category(icd) for icd in full_fused_2019['disease']}
    full_fused_2019['category'] = full_fused_2019['disease'].map(mapped_results)
    # full_fused_2019 = pd.read_csv('/itf-fi-ml/shared/users/ziyuzh/svm/results/weighted_fused_2019.csv')
    full_fused_2019 = full_fused_2019[['method', 'para', 'top_recall_25','top_recall_300','top_recall_10%','top_recall_30%','auroc','rank_ratio','bedroc_1', 'bedroc_5', 'bedroc_10', 'bedroc_30', 'weights_1', 'weights_2', 'weights_3','disease', 'category']]

    full_fused_2019 = full_fused_2019.round(3)

    later_fused = [item for item in full_fused_2019['para'].unique().tolist() if 'nor_' in item or 'later' in item]
    feature_order = None
    if 'DL_early' in fused_2019['para'].unique():
        feature_order = ['DL_uniport_ppi_2019','DL_ppi_2019_dw_40','DL_diffusion_2019_2', 'DL_uniport_bio', 'DL_uniport_esm','DL_uniport_seq',
                         'DL_early', 'DL_early_ppi', 'DL_mid', 'DL_mid_ppi','DL_later_avg','DL_later_avg_ppi', 'DL_later_mlp', 'DL_later_mlp_ppi']
    elif 'RF_early' in fused_2019['para'].unique():
        feature_order = ['RF_uniport_ppi_2019','RF_ppi_2019_dw_40','RF_diffusion_2019_2', 'RF_uniport_bio', 'RF_uniport_esm','RF_uniport_seq',
                         'RF_early', 'RF_early_ppi','RF_later_avg','RF_later_avg_ppi', 'RF_later_rf', 'RF_later_rf_ppi']       
    elif 'gcn' in fused_2019['para'].unique():
        feature_order = ['gcn','sage']   
    else:
        print('11')
        feature_order = ['uniport_ppi_2019', 'ppi_2019_dw_40','diffusion_2019','uniport_bio', 'uniport_esm', 'uniport_seq', 
                    'linear_fused', 'linear_fused_ppi','geo_fused', 'geo_fused_ppi','early_fused','early_fused_ppi'] + later_fused

    # Convert 'feature' to a categorical type with that order
    full_fused_2019['para'] = pd.Categorical(full_fused_2019['para'], categories=feature_order, ordered=True)

    # Now sort by 'disease' first, then 'feature' by the custom order
    full_fused_2019 = full_fused_2019.sort_values(by=['disease', 'para'])

    target_cols = ['top_recall_25','top_recall_300','top_recall_10%','top_recall_30%','auroc', 'bedroc_1', 'bedroc_5', 'bedroc_10', 'bedroc_30', 'weights_1', 'weights_2', 'weights_3']

    # Define alternating background colors per disease
    base_colors = ['#ffdddd', '#dbf7db']
    diseases = full_fused_2019['disease'].unique()
    disease_color_map = {disease: base_colors[i % len(base_colors)] for i, disease in enumerate(diseases)}

    def combined_style(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)

        # Add background color row-wise
        for idx, row in df.iterrows():
            bg_color = disease_color_map[row['disease']]
            styles.loc[idx, :] = f'background-color: {bg_color};'

        # Highlight max in each group & each target column
        for category, group in df.groupby('disease'):
            for col in target_cols:
                max_val = group[col].max()
                max_indices = group[group[col] == max_val].index
                for idx in max_indices:
                    styles.loc[idx, col] += ' color: red; font-weight: bold;'
        
        return styles


    # Build mapping for background colors per disease
    # full_fused_2019['para'] = full_fused_2019['para'].str.replace('ppi_2016', 'ppi_2017', regex=False)
    # Identify numeric columns
    numeric_cols = full_fused_2019.select_dtypes(include=['number']).columns

    # Apply styling and format ONLY numeric columns
    full_fused_2019_disease = (
        full_fused_2019.style
        .apply(combined_style, axis=None)
        .format({col: "{:.3f}" for col in numeric_cols})
    )

    # fused_2019['para'] = fused_2019['para'].str.replace('ppi_2016', 'ppi_2017', regex=False)
    results = fused_2019
    mapped_results = {icd: find_disease_category(icd) for icd in results['disease']}
    results['category'] = results['disease'].map(mapped_results)

    collected_dfs = []
    disease_num = dict()
    for category in results['category'].unique().tolist():
        subdf = results[results['category'] == category].copy()
        # subdf = subdf.drop(columns='weights')
        sum_df = create_summary(subdf, ['method', 'para'], fused=True)
        sum_df['category'] = category
        collected_dfs.append(sum_df)
        disease_num[category] = len(subdf) / 2

    final_df = pd.concat(collected_dfs, ignore_index=True)

    category_order = final_df.groupby('category', observed=True)['auroc_mean'].mean().sort_values(ascending=False).index.tolist()
    final_df['category'] = pd.Categorical(final_df['category'], categories=category_order, ordered=True)

    show_df = final_df.sort_values(by=['category', 'auroc_mean'], ascending=[True, False]) \
        .loc[:, final_df.columns.str.contains('method|para|top_recall_25_mean|top_recall_300_mean|top_recall_10%_mean|top_recall_30%_mean|auroc_mean|bedroc_1_mean|bedroc_5_mean|bedroc_10_mean|bedroc_30_mean|weights_1_mean|weights_2_mean|weights_3_mean|category', case=False)] \
        .round(3) \
        .rename(columns=lambda x: x.replace('_mean', ''))

    # feature_order = ['ppi_2017', 'gene2vec', 'esm2', 'uniport', 
    #                  'linear_fused', 'geo_fused','weighted_linear_fused','weighted_geo_fused']

    # Convert 'feature' to a categorical type with that order
    show_df['para'] = pd.Categorical(show_df['para'], categories=feature_order, ordered=True)

    # Now sort by 'disease' first, then 'feature' by the custom order
    show_df = show_df.sort_values(by=['category', 'para'])

    # Define alternating background colors per disease
    base_colors = ['#ffdddd', '#dbf7db']


    # Build mapping for background colors per disease
    category = show_df['category'].unique()
    color_map = {disease: base_colors[i % len(base_colors)] for i, disease in enumerate(category)}

    target_cols = ['top_recall_25','top_recall_300','top_recall_10%','top_recall_30%','auroc', 'bedroc_1', 'bedroc_5', 'bedroc_10', 'bedroc_30','weights_1', 'weights_2', 'weights_3']

    def combined_style2(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)

        # Add background color row-wise
        for idx, row in df.iterrows():
            bg_color = color_map[row['category']]
            styles.loc[idx, :] = f'background-color: {bg_color};'

        # Highlight max in each group & each target column
        for category, group in df.groupby('category'):
            for col in target_cols:
                max_val = group[col].max()
                max_indices = group[group[col] == max_val].index
                for idx in max_indices:
                    styles.loc[idx, col] += ' color: red; font-weight: bold;'
        
        return styles


    numeric_cols = show_df.select_dtypes(include=['number']).columns

    # Apply styling and format ONLY numeric columns
    styled_df = (
        show_df.style
        .apply(combined_style2, axis=None)
        .format({col: "{:.3f}" for col in numeric_cols})
    )

    valid_feature_order = [f for f in feature_order if f in all_avg_df['para'].unique()]
    all_avg_df = all_avg_df.set_index('para').loc[valid_feature_order].reset_index()

    def highlight_max_font_red(s):
        is_max = s == s.max()
        return ['color: red; font-weight: bold' if v else '' for v in is_max]
    all_avg_df =all_avg_df[['para', 'top_recall_25','top_recall_300','top_recall_10%','top_recall_30%','auroc', 'rank_ratio', 'bedroc_1',
        'bedroc_5', 'bedroc_10', 'bedroc_30','weights_1', 'weights_2', 'weights_3']].round(3)
    # Apply styling
    styled_all_avg_df = all_avg_df.style.apply(highlight_max_font_red, subset=target_cols)

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        styled_all_avg_df.to_excel(writer, sheet_name='all (macro avg)', index=False)
        styled_df.to_excel(writer, sheet_name='category (macro avg)', index=False)
        full_fused_2019_disease.to_excel(writer, sheet_name='disease', index=False)

