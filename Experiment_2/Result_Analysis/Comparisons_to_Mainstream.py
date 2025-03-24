import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel

models = ['Random', 'MostPop', 'RP3beta', 'iALS']
cutoff = 50

dataset = 'lfm'
year = 2013
filtered = True



results_path_child = f'../Results/child_user_and_recommendation_genre_distributions.tsv'
results_path = f'../Results/user_and_recommendation_genre_distributions.tsv'

genre_path = f'../Results/genres.txt'


with open(genre_path, 'r') as f:
    genres = f.readlines()
genres = [genre.split('\n')[0] for genre in genres]


with open('../genres_lfm.txt', 'r') as f:
    genres_order_lfm = f.readlines()
genres_order_lfm = [genre.split('\n')[0] for genre in genres_order_lfm]

results = pd.read_csv(results_path, sep='\t')
def safe_literal_eval(val):
    if val is None:
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # Handle any case where literal_eval might fail (if needed)
        return val
    
for column in results.columns:
    if 'genre_distribution' in column:
        results[column] = results[column].apply(safe_literal_eval)

from scipy.spatial.distance import jensenshannon


def deviation_from_agp(df, models, cutoff):
    df['user_group'] = "mainstream"
    df.loc[df['age_at_listen'] < 17, 'user_group'] = "child"
    df.loc[df['age_at_listen'] > 29, 'user_group'] = "older"

    mainstream_distribution = df[df['user_group'] == "mainstream"]['train_genre_distribution'].apply(np.array).mean(axis=0)
    child_distribution = df[df['user_group'] == "child"]['train_genre_distribution'].apply(np.array).mean(axis=0)
    
    js_results_df = pd.DataFrame()

    for age_group, group in df.groupby('user_group'):

        js_df = pd.DataFrame()
        for model in models:
            js_df[f'js_mainstream_{model}_{cutoff}'] = None
            js_df[f'js_child_{model}_{cutoff}'] = None
            js_df[f'js_childsubset_mainstream_{model}_{cutoff}'] = None
            js_df[f'js_childsubset_child_{model}_{cutoff}'] = None
        js_df['user_id'] = None
        
        for i, row in group.iterrows():
            for model in models:
                recommendation_distribution = np.array(row[f'{model}_recommendation_genre_distribution_{cutoff}'], dtype=float)
                js_mainstream = jensenshannon(recommendation_distribution, mainstream_distribution)**2
                js_child = jensenshannon(recommendation_distribution, child_distribution)**2

                if np.isnan(js_mainstream):
                    js_mainstream = 0
                if np.isnan(js_child):
                    js_child = 0
                js_df.at[i, f'js_mainstream_{model}_{cutoff}'] = js_mainstream
                js_df.at[i, f'js_child_{model}_{cutoff}'] = js_child
 
            js_df.at[i, 'user_id'] = row['user_id']
            js_df['user_group'] = age_group
            
        
        js_results_df = pd.concat([js_results_df, js_df], axis=0)               

        if age_group == 'child':

            js_df = pd.DataFrame()
            for model in models:
                js_df[f'js_mainstream_{model}_{cutoff}'] = None
                js_df[f'js_child_{model}_{cutoff}'] = None
                js_df[f'js_childsubset_mainstream_{model}_{cutoff}'] = None
                js_df[f'js_childsubset_child_{model}_{cutoff}'] = None
            js_df['user_id'] = None
            
            for i, row in group.iterrows():
                for model in models:
                    recommendation_distribution = np.array(row[f'child_{model}_recommendation_genre_distribution_{cutoff}'], dtype=float)
                    js_mainstream = jensenshannon(recommendation_distribution, mainstream_distribution)**2
                    js_child = jensenshannon(recommendation_distribution, child_distribution)**2

                    if np.isnan(js_mainstream):
                        js_mainstream = 0
                    if np.isnan(js_child):
                        js_child = 0
                    js_df.at[i, f'js_mainstream_{model}_{cutoff}'] = js_mainstream
                    js_df.at[i, f'js_child_{model}_{cutoff}'] = js_child
    
                js_df.at[i, 'user_id'] = row['user_id']
                js_df['user_group'] = 'child_focused'
                
            
            js_results_df = pd.concat([js_results_df, js_df], axis=0)  
            
        
    for age_group, group in js_results_df.groupby('user_group'):
        for model in models:
            js_mainstream = group[f'js_mainstream_{model}_{cutoff}'].mean()
            js_child = group[f'js_child_{model}_{cutoff}'].mean()
            print(f'Average JSD between {model} recommendations and mainstream distribution for {age_group} users: {js_mainstream:.4f}')
            print(f'Average JSD between {model} recommendations and child distribution for {age_group} users: {js_child:.4f}')
            print()
            print()
            
            
    # Plotting
    plt.figure(figsize=(10, 8))


    # Calculate grouped means once
    js_grouped = js_results_df.groupby('user_group').mean()

    # Plot for both distributions
    for distribution in ['mainstream', 'child']:
        for model in models:
            plt.plot(js_grouped[f'js_{distribution}_{model}_{cutoff}'], label=f'{distribution} {model}', marker='o')

    # Labels and layout adjustments
    plt.xlabel('Age Group', fontsize=16)
    plt.ylabel('Average Jensen-Shannon Divergence', fontsize=16)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(title="Deviation from Age Profile")
    plt.tight_layout()
    plt.show()

    
    return js_results_df


js_results_df = deviation_from_agp(results, models, cutoff)



def test_jsd_across_groups(js_df, models, cutoff):
    for comparison in ['mainstream', 'child']:
        for model in models:
            print(f"Testing JSD to {comparison} for model: {model}")
            
            # Group JSD by age group for the given model
            js_df_clean = js_df[['user_group', f'js_{comparison}_{model}_{cutoff}']].dropna()
            js_df_clean = js_df_clean[js_df_clean['user_group'] != 'child_focused']
            groups = [group[f'js_{comparison}_{model}_{cutoff}'].values for name, group in js_df_clean.groupby('user_group')]
            
            # Print average scores
            print(f"Average JS divergence for {model}:")
            
            print(js_df_clean.groupby('user_group')[f'js_{comparison}_{model}_{cutoff}'].mean())
            
            # One-way ANOVA test
            anova_result = f_oneway(*groups)
            print(f"ANOVA result for {model}: F-statistic = {anova_result.statistic:.4f}, p-value = {anova_result.pvalue:.4f}")
            
            # If ANOVA is significant, perform Tukey HSD for post-hoc analysis
            if anova_result.pvalue < 0.05:
                print(f"ANOVA is significant for {model}, performing Tukey HSD test...")
                
                tukey_result = pairwise_tukeyhsd(pd.to_numeric(js_df_clean[f'js_{comparison}_{model}_{cutoff}']), js_df_clean['user_group'], alpha=0.01)
                print(tukey_result)
                
            # paired t-test between user_group = child_focused and user_group = child
            print('Paired t-test between child_focused and child')
            child_focused = js_df[js_df['user_group'] == 'child_focused'][f'js_{comparison}_{model}_{cutoff}']
            child = js_df[js_df['user_group'] == 'child'][f'js_{comparison}_{model}_{cutoff}']
            ttest_result = ttest_rel(child_focused, child)
            print(f"Paired t-test result: t-statistic = {ttest_result.statistic:.4f}, p-value = {ttest_result.pvalue:.4f}")
                


test_jsd_across_groups(js_results_df, models, cutoff)


    