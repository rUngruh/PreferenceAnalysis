import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

show_plots = ['avg_deviation_from_train']
models = ['MostPop', 'Random', 'RP3beta', 'iALS']
test_model = 'RP3beta'
cutoff = 50

dataset = 'lfm'
year = 2013





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

    
def avg_model_profile_deviation(df, models, cutoff, child=False):
    js_df = df[['user_id', 'age_at_listen', 'age_group', 'user_group']].copy()
    df =  df[['train_genre_distribution'] + [f'{"child_" if child else ""}{model}_recommendation_genre_distribution_{cutoff}' for model in models]]
    model_js = {}
    for model in models:
        js_df.loc[:,f'js_{model}_{cutoff}'] = None
    
    for i, row in df.iterrows():
        user_distribution = np.array(row['train_genre_distribution'], dtype=float)
        
        for model in models:
            model_distribution = np.array(row[f'{"child_" if child else ""}{model}_recommendation_genre_distribution_{cutoff}'], dtype=float)   

            js = jensenshannon(user_distribution, model_distribution)**2

            if np.isnan(js):
                js = 0
            model_js[model] = model_js.get(model, 0) + js
            js_df.at[i, f'js_{model}_{cutoff}'] = js
    model_js = {key: value/len(df) for key, value in model_js.items()}
    return model_js, js_df


    

if 'avg_deviation_from_train' in show_plots:

    results['child'] = results['age_at_listen']<17

    def plot_js_across_ages(results, models, cutoff):
        age_groups = sorted(results['user_group'].unique())
        
        # Dictionary to store JS scores for each model across age groups
        model_js_across_ages = {model: [] for model in models}
        
        js_results_df = pd.DataFrame()
        
        for age_group, group in results.groupby('user_group'):
            model_js_mean, js_df = avg_model_profile_deviation(group, models, cutoff)
            js_results_df = pd.concat([js_results_df, js_df], axis=0)
            if age_group == 'child':
                child_js_mean, js_df = avg_model_profile_deviation(group, models, cutoff, child=True)
                js_df['user_group'] = 'child_focused'
                js_results_df = pd.concat([js_results_df, js_df], axis=0)
                
            # Append the average JS divergence for each model
            for model in models:
                model_js_across_ages[model].append(model_js_mean[model])
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        for model in models:
            plt.plot(age_groups, model_js_across_ages[model], marker='o', label=model)
        
        plt.xlabel('Age Group')
        plt.ylabel('Average Jensen-Shannon Divergence')
        plt.title(f'Average Model Profile Deviation Across Age Groups (Cutoff: {cutoff})')
        plt.xticks(rotation=45)
        plt.legend(title="Models")
        plt.tight_layout()
        plt.show()
        
        return js_results_df

    # Call the plotting function
    js_results_df = plot_js_across_ages(results, models, cutoff)


    def test_jsd_across_groups(js_df, models, cutoff):
        for model in models:
            print(f"Testing JSD for model: {model}")
            
            # Group JSD by age group for the given model
            js_df_clean = js_df[['user_group', f'js_{model}_{cutoff}']].dropna()
            js_df_clean = js_df_clean[js_df_clean['user_group'] != 'child_focused']
            groups = [group[f'js_{model}_{cutoff}'].values for name, group in js_df_clean.groupby('user_group')]
            
            # Print average scores
            print(f"Average JS divergence for {model}:")
            
            print(js_df_clean.groupby('user_group')[f'js_{model}_{cutoff}'].mean())
            
            # One-way ANOVA test
            anova_result = f_oneway(*groups)
            print(f"ANOVA result for {model}: F-statistic = {anova_result.statistic:.4f}, p-value = {anova_result.pvalue:.4f}")
            
            # If ANOVA is significant, perform Tukey HSD for post-hoc analysis
            if anova_result.pvalue < 0.05:
                print(f"ANOVA is significant for {model}, performing Tukey HSD test...")
                
                tukey_result = pairwise_tukeyhsd(pd.to_numeric(js_df_clean[f'js_{model}_{cutoff}']), js_df_clean['user_group'], alpha=0.01)
                print(tukey_result)
                
            # paired t-test between user_group = child_focused and user_group = child
            print('Paired t-test between child_focused and child')
            child_focused = js_df[js_df['user_group'] == 'child_focused'][f'js_{model}_{cutoff}']
            child = js_df[js_df['user_group'] == 'child'][f'js_{model}_{cutoff}']
            ttest_result = ttest_rel(child_focused, child)
            print(f"Paired t-test result: t-statistic = {ttest_result.statistic:.4f}, p-value = {ttest_result.pvalue:.4f}")
                

            
    test_jsd_across_groups(js_results_df, models, cutoff)
 


if 'anova' in show_plots:
    print("Performing ANOVA analysis on genre distributions...")

    results['age_group'] = results['age_at_listen']<17
    
    genres = [genre.replace("'", "").replace("-", "").replace(" ", "") for genre in genres]
    genre_df = pd.DataFrame(results[f'{test_model}_recommendation_genre_distribution_{cutoff}'].apply(np.array).tolist())
    genre_df.columns = genres

    # Combine with the original DataFrame
    genre_df = pd.concat([results['age_group'], genre_df], axis=1)
    
    genre_df = genre_df.loc[:, (genre_df != 0).any(axis=0)]
    
    print(genre_df.head())
    from statsmodels.multivariate.manova import MANOVA
    
    # Fit the MANOVA model
    manova = MANOVA.from_formula(' + '.join(genre_df.columns[1:]) + ' ~ age_group', data=genre_df)
    manova_results = manova.mv_test()

    print(manova_results)
    
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Melt the DataFrame to long format for ANOVA
    melted_df = genre_df.melt(id_vars=['age_group'], value_vars=genre_df.columns, var_name='genre', value_name='value')

    
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    # Perform ANOVA for each genre
    anova_results = {}
    for genre in genre_df.drop('age_group', axis=1).columns:
        model = ols(f'{genre} ~ age_group', data=genre_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results[genre] = anova_table

    # Print ANOVA results for each genre
    for genre, result in anova_results.items():
        if result['PR(>F)'][0] < 0.05:
            print(f'ANOVA results for {genre}:')
            print(result)
            print('\n')

    # Tukey's HSD test for each genre
    tukey_results = {}
    for genre in genres:
        tukey = pairwise_tukeyhsd(endog=melted_df[melted_df['genre'] == genre]['value'],
                                groups=melted_df[melted_df['genre'] == genre]['age_group'],
                                alpha=0.01)
        tukey_results[genre] = tukey

    print('sigificant Genres')
    # Print Tukey's HSD results for each genre
    for genre, result in tukey_results.items():
        if result.reject.any():
            
            print(f'Tukey HSD results for {genre}:')
            print(result)
            print('\n')

    print('non-significant Genres')
    for genre, result in tukey_results.items():
        if not result.reject.any():
            print(f'Tukey HSD results for {genre}:')
            print(result)
            print('\n')
            