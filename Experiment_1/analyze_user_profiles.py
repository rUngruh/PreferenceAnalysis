import os
import pandas as pd
from scipy.special import kl_div
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

import argparse

parser = argparse.ArgumentParser(description='Analyze the user profiles of the filtered subsets.')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], required=True)
parser.add_argument('--analysis_types', type=str, nargs='+', help='Analysis types to perform', \
    choices=['anova', 'in-user-divergence-development', 'genre_distribution', 
             'intragroup_diversity', 'intergroup_divergence',  'intragroup_divergence', 
             'krippendorff_alpha', 'interquartile_range', 'kullback-leibler_divergence', 
             'cluster_analysis'], required=True)
parser.add_argument('--age_type', type=str, help='Type of age grouping', choices=['finegrained_age', 'binary_age', 'finegrained_child_ages'], required=True)
args = parser.parse_args()

age_type = args.age_type
dataset = args.dataset
distribution_type = 'normalized_genre_distribution'
show_plots = args.analysis_types
# 'anova', 'in-user-divergence-development', 'genre_distribution', 'intragroup_diversity', 'intergroup_divergence', 
# 'intragroup_divergence', 'krippendorff_alpha', 'interquartile_range', 'kullback-leibler_divergence', 'cluster_analysis'

dataset_dir = os.getenv("dataset_directory")
ml_data_dir = dataset_dir + '/processed/ml_with_age'
lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'



if dataset == 'ml':
    user_profile_path = ml_data_dir + '/user_profiles.tsv'
    user_profile_stats_path = ml_data_dir + '/user_profile_stats.tsv'
    
    
    
elif dataset == 'lfm':
    user_profile_path = lfm_data_dir + '/user_profiles_weighted.tsv'
    user_profile_stats_path = lfm_data_dir + '/user_profile_stats_weighted.tsv'

#user_profiles = pd.read_csv(user_profile_path, sep='\t')
user_stats = pd.read_csv(user_profile_stats_path, sep='\t')
#user_profiles['items'] = user_profiles['items'].apply(lambda x: ast.literal_eval(x))
#user_stats['top_genre_distribution'] = user_stats['top_genre_distribution'].apply(lambda x: ast.literal_eval(x))
user_stats['normalized_genre_distribution'] = user_stats['normalized_genre_distribution'].apply(lambda x: ast.literal_eval(x))

if dataset == 'lfm':
    if age_type == 'finegrained_age':
        ages_sort = ['12', '13', '14', '15', '16', '17', '18', '19-20', '21-22', '23-24', '25-29', '30-34', '35-44', '45-54', '55-65'] # Age group can be defined as a range (seperated by '_') or a single age
    elif age_type == 'binary_age':
        ages_sort = ['12-17', '18-65']
    elif age_type == 'finegrained_child_ages':
        ages_sort = ['12', '13', '14', '15', '16', '17', '18-65'] 
elif dataset == 'ml':
    if age_type == 'finegrained_age':
        ages_sort = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
    elif age_type == 'binary_age':
        ages_sort = ['Under 18', '18+']
    elif age_type == 'finegrained_child_ages':
        ages_sort = ['Under 18', '18-65']
# Define the age grouping function
def age_group(age):
    if dataset == 'lfm':

        min_age = int(ages_sort[0].split('-')[0]) if '-' in ages_sort[0] else int(ages_sort[0])
        if age < min_age:
            return None  # Exclude ages below the minimum age in ages_sort
        else:
            for age_range in ages_sort:
                if '-' in age_range:
                    start_age, end_age = map(int, age_range.split('-'))
                    if start_age <= age <= end_age:
                        return age_range
                else: 
                    if age == int(age_range):
                        return age_range
            return None
        
    elif dataset == 'ml':
        if age == 1:
            return "Under 18"
        
        for age_group in ages_sort:
            if '-' in age_group:
                start_age, end_age = map(int, age_group.split('-'))
            elif 'Under' in age_group:
                start_age = 0
                end_age = int(age_group.split(' ')[1])
            elif '+' in age_group:
                start_age = int(age_group.split('+')[0])
                end_age = 100
            else: 
                return None
            
            if start_age <= age <= end_age:
                return age_group
        return None
        

user_stats.sort_values('age', inplace=True)
user_stats['age_group'] = user_stats['age'].apply(age_group)
print(user_stats['age_group'].value_counts())    
user_stats = user_stats.groupby('age_group')



genres = []
genre_distributions = pd.DataFrame(columns=('age_group', 'genre_distribution'))
print("Calculating genre distributions for each age group...")

for age, group in user_stats:
    genre_sum = {}
    num_users = len(group)
    for i,user in group.iterrows():

        for user_genre, value in user[distribution_type].items():

            if user_genre not in genres:
                genres.append(user_genre)
            if user_genre in genre_sum:
                genre_sum[user_genre] += value
            else:
                genre_sum[user_genre] = value
       
    genre_avg = {genre: genre_sum.get(genre, 0) / num_users for genre in genres}
    if age in genre_distributions['age_group'].values:
        genre_distributions.loc[genre_distributions['age_group'] == age, 'genre_distribution'] = [genre_avg]
        print("this should not happen") # test statement
    else:
        genre_distributions = pd.concat([genre_distributions, pd.DataFrame({'age_group': [age], 'genre_distribution': [genre_avg]})], ignore_index=True)


for i, row in genre_distributions.iterrows():
    genre_distributions.at[i, 'genre_distribution'] = [row['genre_distribution'].get(genre, 0) for genre in genres]

user_stats = user_stats.obj
user_stats['genre_distribution'] = None
for i, user in user_stats.iterrows():
    user_stats.at[i, 'genre_distribution'] = [user[distribution_type].get(genre, 0) for genre in genres]
user_stats = user_stats.groupby('age_group')

genre_distributions['age_group'] = pd.Categorical(genre_distributions['age_group'], 
                                            categories=ages_sort, 
                                            ordered=True)

    
genre_distributions.sort_values('age_group', inplace=True)

if 'genre_distribution' in show_plots:
    genres = [genre.capitalize() for genre in genres]
    # Prepare data for stacked bar plot
    genre_data = pd.DataFrame(genre_distributions['genre_distribution'].tolist(), index=genre_distributions['age_group'], columns=genres)

    
    # Plot stacked bar chart
    plt.figure(figsize=(12, 8))
    sns.set(style='whitegrid')
    genre_data.plot(kind='bar', stacked=True, cmap='tab20', ax=plt.gca())
    plt.xlabel('Age Group', fontsize=16)
    plt.ylabel('Average Genre Distribution', fontsize=16)
    handles, labels = plt.gca().get_legend_handles_labels() 
    plt.legend(handles[::-1], labels[::-1], title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

from scipy.stats import entropy
if 'intragroup_diversity' in show_plots:
    from scipy.stats import entropy

    genre_distributions['entropy'] = genre_distributions['genre_distribution'].apply(lambda x: entropy(x))
    plt.figure(figsize=(12, 8))
    sns.barplot(x='age_group', y='entropy', data=genre_distributions, palette='tab10')
    plt.xlabel('Age Group', fontsize=16)
    plt.ylabel('Entropy', fontsize=16)
    plt.xticks(rotation=45, fontsize=16, ha='right')
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

if 'krippendorff_alpha' in show_plots:
    import krippendorff
    alpha_by_age_group = {}

    # Group by age_group
    for age, group_data in user_stats:
        # Extract the genre ratings for all users within this age group
        data_matrix = np.array([np.array(user['genre_distribution']) for i, user in group_data.iterrows()])
        print(data_matrix.shape)
        # Calculate Krippendorff's alpha for interval data
        alpha = krippendorff.alpha(reliability_data=data_matrix, level_of_measurement='ratio')
        
        # Store the result in the dictionary
        alpha_by_age_group[age] = alpha

    # Output the results
    for age, alpha in alpha_by_age_group.items():
        print(f"Krippendorff's alpha for age group {age}: {alpha}")
    
if 'intergroup_divergence' in show_plots:
    print("Calculating Jensen-Shannon Divergence between age groups...")
    inter_group_js_divergences = {}
    epsilon = 1e-10

    for i, row1 in genre_distributions.iterrows():
        age1 = row1['age_group']
        genre_distribution1 = np.array(row1['genre_distribution']) + epsilon
        
        for j, row2 in genre_distributions.iterrows():
            age2 = row2['age_group']
            genre_distribution2 = np.array(row2['genre_distribution']) + epsilon

            # Jensen-Shannon Divergence
            js_div = jensenshannon(genre_distribution1, genre_distribution2)**2
            print(js_div)
            inter_group_js_divergences[(age1, age2)] = js_div

    ages = genre_distributions['age_group'].unique()  

    heatmap_data = pd.DataFrame(index=ages, columns=ages)
    for (age1, age2), js_div in inter_group_js_divergences.items():
        #print(f"Jensen-Shannon Divergence between age {age1} and age {age2}: {js_divergence}")
        heatmap_data.at[age1, age2] = js_div
        heatmap_data.at[age2, age1] = js_div  # Ensure symmetry
        
    heatmap_data = heatmap_data.astype(float)
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_data, annot=True, cmap="coolwarm_r", linewidths=0.5, cbar=False, annot_kws={"size": 12})
    #plt.title("Jensen-Shannon Divergence Between Age Groups")

    plt.xlabel("Age Group", fontsize=16)
    plt.ylabel("Age Group", fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=16)  # Set rotation to 0 for horizontal

    plt.tight_layout()
    plt.show()

if 'interquartile_range' in show_plots:
    print("Calculating interquartile range of genre distributions across age groups...")
    genre_distributions['iqr'] = genre_distributions['genre_distribution'].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    plt.figure(figsize=(12, 8))
    sns.barplot(x='age_group', y='iqr', data=genre_distributions, palette='tab10')
    plt.xlabel('Age Group', fontsize=16)
    plt.ylabel('Interquartile Range', fontsize=16)
    plt.xticks(rotation=45, fontsize=16, ha='right')
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()
    
if 'kullback-leibler_divergence' in show_plots:
    print("Calculating Kullback-Leibler Divergence between age groups and the overall distribution...")
    overall_distribution = genre_distributions['genre_distribution'].apply(np.array).mean(axis=0)
    genre_distributions['kl_divergence'] = genre_distributions['genre_distribution'].apply(lambda x: entropy(x, overall_distribution))

    plt.figure(figsize=(10, 8))
    sns.barplot(x='age_group', y='kl_divergence', data=genre_distributions, palette='tab10')
    plt.xlabel('Age Group', fontsize=16)
    plt.ylabel('KL Divergence', fontsize=16)
    plt.xticks(rotation=45, fontsize=16, ha='right')
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()
    
if 'cluster_analysis' in show_plots:
    print("Performing cluster analysis on genre distributions...")
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage

    linked = linkage(genre_distributions['genre_distribution'].apply(np.array).tolist(), 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=genre_distributions['age_group'].values, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering of Age Groups Based on Genre Preferences')
    plt.xlabel('Age Group', fontsize=16)
    plt.ylabel('Distance', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()


if 'intragroup_divergence' in show_plots:
    epsilon = 1e-10
    print("Calculating average Jensen-Shannon Divergence within age groups...")
    # Initialize a DataFrame to store the average JSD values for each age group
    intra_group_js_divergences = pd.DataFrame(columns=['age_group', 'average_js_divergence'])

    from scipy.spatial.distance import pdist, jensenshannon

    # Iterate over each age group
    for age, group in user_stats:
        print(f"Processing age group: {age}")
        num_users = len(group)
        js_divergences = []
        # Transform genre distributions for vectorized operations
        genre_distributions = np.array([np.array(user['genre_distribution']) + epsilon for i, user in group.iterrows()])
        
        # Calculate pairwise JSD for all users within the same age group
        pairwise_jsd = pdist(genre_distributions, metric=lambda u, v: jensenshannon(u, v)**2)
        if np.isnan(pairwise_jsd):
                pairwise_jsd = 0
                
        # Average the pairwise JSDs
        average_js_divergence = np.mean(pairwise_jsd) if len(pairwise_jsd) > 0 else 0
        intra_group_js_divergences = pd.concat([intra_group_js_divergences, pd.DataFrame({'age_group': [age], 'average_js_divergence': [average_js_divergence]})], ignore_index=True)

    # Sort the DataFrame by age
    intra_group_js_divergences['age_group'] = pd.Categorical(intra_group_js_divergences['age_group'], categories=ages_sort, ordered=True)
    intra_group_js_divergences.sort_values('age_group', inplace=True)

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(data=intra_group_js_divergences, x='age_group', y='average_js_divergence', palette='viridis')
    plt.title('Average Jensen-Shannon Divergence Within Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Average JSD')
    plt.xticks(rotation=45)
    plt.show()

def compute_deviation(user):
        user = user.sort_values('age').reset_index(drop=True)
        init_genre_distribution_dict = user.iloc[0]['normalized_genre_distribution']
        init_genre_distribution = np.array([init_genre_distribution_dict.get(genre, 0) for genre in genres]) + epsilon

        deviations = []
        for i, row in user.iterrows():
            row_genre_distribution_dict = row['normalized_genre_distribution']
            row_genre_distribution = np.array([row_genre_distribution_dict.get(genre, 0) for genre in genres]) + epsilon
            js_div = jensenshannon(init_genre_distribution, row_genre_distribution)**2
            if np.isnan(js_div):
                js_div = 0
                
            deviations.append(js_div)
        
        user['deviation_from_initial'] = deviations
        return user
    

if dataset == 'lfm' and 'in-user-divergence-development' in show_plots:
    epsilon = 1e-10



    # Apply the function to each group and combine the results
    user_stats_grouped = user_stats.obj.groupby('user_id')
    user_stats_w_deviation = user_stats_grouped.apply(compute_deviation).reset_index(drop=True)
        
    # Pre-sort the DataFrame by 'user_id' and 'age'
    user_stats_w_deviation_sorted = user_stats_w_deviation.sort_values(by=['user_id', 'age']).reset_index(drop=True)

    # Group by 'user_id'
    user_stats_w_deviation_grouped = user_stats_w_deviation_sorted.groupby('user_id')

    # Function to collect deviations and starting age for each user
    def collect_age_developments(user):
        starting_age = user['age'].iloc[0]
        deviations = user['deviation_from_initial'].tolist()
        return {'user_id': user['user_id'].iloc[0], 'starting_age': starting_age, 'deviations': deviations}

    # Apply the function to each group and collect results
    age_developments = [collect_age_developments(user) for _, user in user_stats_w_deviation_grouped]

    # Convert to DataFrame
    age_developments_df = pd.DataFrame(age_developments)

    # Create a new figure for the plot
    plt.figure(figsize=(12, 8))

    # Group by starting age
    grouped = age_developments_df.groupby('starting_age')

    x = []
    # Plot mean deviations for each starting age
    for starting_age, group in grouped:
        sum_deviations = {}
        for _, row in group.iterrows():
            deviations = row['deviations']
            for i, deviation in enumerate(deviations):
                sum_deviations[i] = sum_deviations.get(i, []) + [deviation]
        mean_deviations = [np.mean(deviations) for deviations in sum_deviations.values()]
        ages = [x + starting_age for x in list(range(len(mean_deviations)))]

        
        # Plot the mean deviation
        plt.plot(ages, mean_deviations, label=f'Starting Age {starting_age}', linewidth=2)

    # Add labels and title
    plt.xlabel('Age Index')
    plt.ylabel('Mean Deviation from Initial')
    plt.title('Mean Deviations Over Time Grouped by Starting Age')
    #plt.legend(None)
    plt.grid(True)

    # Show the plot
    plt.show()
    
if dataset == 'lfm' and 'in-user_age_pairs' in show_plots:
    epsilon = 1e-10
    deviations_to_prev_year = {}
    # Group by user_id and age to prepare for deviation calculation
    user_stats_grouped = user_stats.obj.groupby('user_id')
    for user_id, user in user_stats_grouped:
        user.sort_values('age', inplace=True)
        age_0 = None
        for i, row in user.iterrows():
            age_1_age = row['age']
            age_1_dist = user['genre_distribution'] 

            if age_0 is not None and row['age'] - age_0[0] == 1:
                js_div = jensenshannon(age_0[1], age_1_dist)**2
                if np.isnan(js_div):
                    js_div = 0
                deviations_to_prev_year[age_1_age] = deviations_to_prev_year.get(age_1_age, []) + [js_div]
                
                    
            age_0 = (age_1_age, age_1_dist)

    # Calculate the average deviation for each age
    average_deviations = {age: np.mean(deviations) for age, deviations in deviations_to_prev_year.items()}
    average_deviations = {age: deviation for age, deviation in sorted(average_deviations.items(), key=lambda x: x[0])}
    print(average_deviations)
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(list(average_deviations.keys()), list(average_deviations.values()), marker='o')
    plt.xlabel('Age')
    plt.ylabel('Average Jensen-Shannon Divergence to Previous Year')
    plt.title('Average Jensen-Shannon Divergence to Previous Year by Age')
    plt.grid(True)
    plt.show()
    

if 'intergroup_correlation' in show_plots:
    print("Calculating correlation between age groups...")
    age_groups = genre_distributions['age_group'].unique()
    correlation_data = pd.DataFrame(index=age_groups, columns=age_groups)

    for i, age1 in enumerate(age_groups):
        for j, age2 in enumerate(age_groups):
            correlation_data.at[age1, age2] = np.corrcoef(
                genre_distributions.loc[genre_distributions['age_group'] == age1, 'genre_distribution'].values[0],
                genre_distributions.loc[genre_distributions['age_group'] == age2, 'genre_distribution'].values[0]
            )[0, 1]

    correlation_data = correlation_data.astype(float)
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(correlation_data, annot=True, cmap="coolwarm", linewidths=0.5, cbar=False)
    plt.title("Correlation of Genre Preferences Across Age Groups")
    plt.xlabel("Age Group", fontsize=16)
    plt.ylabel("Age Group", fontsize=16)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()
    
if 'anova' in show_plots:
    print("Performing ANOVA analysis on genre distributions...")

    genres = [genre.replace("'", "").replace("-", "").replace(" ", "") for genre in genres]
    genre_df = pd.DataFrame(user_stats.obj['genre_distribution'].apply(np.array).tolist())
    genre_df.columns = genres

    # Combine with the original DataFrame
    genre_df = pd.concat([user_stats.obj['age_group'], genre_df], axis=1)
    
    print(genre_df.head())
    from statsmodels.multivariate.manova import MANOVA

    # Fit the MANOVA model
    manova = MANOVA.from_formula(' + '.join(genres) + ' ~ age_group', data=genre_df)
    manova_results = manova.mv_test()

    print(manova_results)
    
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
            
            
