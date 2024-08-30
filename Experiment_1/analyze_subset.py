import os
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='Analyze the user profiles of the filtered subsets.')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], required=True)
parser.add_argument('--chunksize', type=int, default=1000000, help='Chunk size for processing listening events')
parser.add_argument('--analysis_types', type=str, nargs='+', help='Analysis types to perform', choices=['profile_size_distribution', 'user_stats_analysis'], required=True)
args = parser.parse_args()

dataset = args.dataset
chunksize = args.chunksize

distribution_type = 'normalized_genre_distribution'
show_plots = args.analysis_types

dataset_dir = os.getenv("dataset_directory")
ml_data_dir = dataset_dir + '/processed/ml_with_age'
lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'



if dataset == 'ml':
    user_profile_path = ml_data_dir + '/user_profiles.tsv'
    user_profile_stats_path = ml_data_dir + '/user_profile_stats.tsv'
    user_path = ml_data_dir + '/users.tsv'
    
    
elif dataset == 'lfm':
    user_profile_path = lfm_data_dir + '/user_profiles_weighted.tsv'
    user_profile_stats_path = lfm_data_dir + '/user_profile_stats_weighted.tsv'
    listening_events_path = lfm_data_dir + '/listening-events.tsv.bz2'
    artists_path = lfm_data_dir + '/artists.tsv'
    user_path = lfm_data_dir + '/users_corrected.tsv'

#user_profiles = pd.read_csv(user_profile_path, sep='\t')
user_stats = pd.read_csv(user_profile_stats_path, sep='\t')
#user_profiles['items'] = user_profiles['items'].apply(lambda x: ast.literal_eval(x))
user_stats['genre_distribution'] = user_stats[distribution_type].apply(lambda x: ast.literal_eval(x))

if dataset == 'lfm':
    artists = pd.read_csv(artists_path, sep='\t')
users = pd.read_csv(user_path, sep='\t')

        

if dataset == 'lfm':
    ages_sort = ['12', '13', '14', '15', '16', '17', '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65'] # Age group can be defined as a range (seperated by '_') or a single age
elif dataset == 'ml':
    ages_sort = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
# Define the age grouping function
def age_group(age):
    if dataset == 'lfm':

        min_age = int(ages_sort[0].split('_')[0]) if '_' in ages_sort[0] else int(ages_sort[0])
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
        elif age == 18:
            return "18-24"
        elif age == 25:
            return "25-34"
        elif age == 35:
            return "35-44"
        elif age == 45:
            return "45-49"
        elif age == 50:
            return "50-55"
        elif age == 56:
            return "56+"
        
user_stats['age_group'] = user_stats['age'].apply(age_group)
user_stats['age_group'] = pd.Categorical(user_stats['age_group'], categories=ages_sort, ordered=True)


print(user_stats['age_group'].value_counts())
listens_per_age = user_stats.groupby('age_group')['num_items'].sum().to_dict()
listens_per_age = {age: listens_per_age.get(age, 0) for age in ages_sort}
# Plot the distribution of profile sizes
if 'profile_size_distribution' in show_plots:
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(data=listens_per_age, marker='o', color='b', )
    ylabels = [f'{x:,}' for x in ax.get_yticks()]
    ax.set_yticklabels(ylabels)
    plt.title('Number of Ratings per Age Group')
    plt.xlabel('Age')
    plt.ylabel('Number of Ratings')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xticks(ha='center')  # Rotate labels by 45 degrees and align to the right
    plt.show()
    
    

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=user_stats, x='age_group', y='num_items')
    ax.set(ylim=(0,250))
    plt.title('Number of Ratings per Age Group')
    plt.xlabel('Age')
    plt.ylabel('Number of Ratings per User')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.xticks(ha='center')
    plt.show()
    
if dataset == 'ml':
    ages = user_stats['age_group'].value_counts().to_dict()
    ages = {age: ages.get(age, 0) for age in ages_sort}
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(ages.keys()), y=list(ages.values()), color='b')
    #plt.title('Number of Profiles per Age Group')
    plt.xlabel('Age', fontsize=16)
    plt.ylabel('Number of Profiles', fontsize=16)
    # Ensure that the x-ticks only show the desired ages
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

if dataset == "lfm" and "user_stats_analysis" in show_plots:
    stats_grouped = user_stats.groupby('user_id')
    num_users = len(stats_grouped)
    print(f"Number of users: {num_users}")
    print(f"Number of user profiles: {len(user_stats)}")
    
    num_profiles = {user_id: len(group) for user_id, group in stats_grouped}
    avg_num_profiles = sum(num_profiles.values()) / len(num_profiles)
    min_num_profiles = min(num_profiles.values())
    max_num_profiles = max(num_profiles.values())
    print(f"Average number of profiles: {avg_num_profiles}")
    print(f"Minimum number of profiles: {min_num_profiles}")
    print(f"Maximum number of profiles: {max_num_profiles}")
    print(f"Number of users with more than 1 profile: {sum(1 for num in num_profiles.values() if num > 1)}")
    print(f"Number of users with more than 5 profiles: {sum(1 for num in num_profiles.values() if num > 5)}")
    print(f"Number of users with more than 10 profiles: {sum(1 for num in num_profiles.values() if num > 10)}")
    
    profile_sizes = {user_id: group['num_items'].tolist() for user_id, group in stats_grouped}
    avg_profile_size = sum(sum(sizes) / len(sizes) for sizes in profile_sizes.values()) / len(profile_sizes)
    avg_listen_events_per_user = sum(sum(sizes) for sizes in profile_sizes.values()) / len(profile_sizes)
    print(f"Average profile size: {avg_profile_size}")
    print(f"Average listen events per user: {avg_listen_events_per_user}")
    print(f"Minimum profile size: {min(min(sizes) for sizes in profile_sizes.values())}")
    print(f"Maximum profile size: {max(max(sizes) for sizes in profile_sizes.values())}")
    print(f"Number of profiles with more than 1 listen event: {sum(sum(1 for size in sizes if size > 1) for sizes in profile_sizes.values())}")
    print(f"Number of profiles with more than 5 listen events: {sum(sum(1 for size in sizes if size > 5) for sizes in profile_sizes.values())}")
    print(f"Number of profiles with more than 10 listen events: {sum(sum(1 for size in sizes if size > 10) for sizes in profile_sizes.values())}")

    
    ages = user_stats['age'].value_counts().to_dict()
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=list(ages.keys()), y=list(ages.values()), color='b')
    #plt.title('Number of Profiles per Age Group')
    plt.xlabel('Age', fontsize=16)
    plt.ylabel('Number of Profiles', fontsize=16)
    # Create a list of x-tick positions. Start with the minimum age and add every 5 years.
    ages_to_display = [0, 3] + list(range(3, int(max(ages.keys()))-10, 5))

    # Ensure that the x-ticks only show the desired ages
    plt.xticks(ticks=ages_to_display, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()