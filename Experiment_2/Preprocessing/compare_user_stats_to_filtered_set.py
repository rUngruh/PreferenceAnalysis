import os
import ast
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Compare the age distribution of the overall user profile to the age distribution of the users in the filtered set that will be used for the recommender experiment.')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], default='lfm')
parser.add_argument('--years', type=int, nargs='+', help='Selecting the years for the recommender experiment. Add "filtered" if a version with k-core filtering was used (e.g., "filtered_2012")', default=[2012])

args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset
years = args.years

dataset_dir = os.getenv("dataset_directory")
ml_data_dir = dataset_dir + '/processed/ml_with_age'
lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'



if dataset == 'ml':

    user_profile_stats_path = ml_data_dir + f'/user_profile_stats.tsv'

    
    
elif dataset == 'lfm':
    user_profile_stats_path = lfm_data_dir + '/user_profile_stats_weighted.tsv'
    user_info_paths = lfm_data_dir + '/elliot_data/user_info_year.tsv'

#user_profiles = pd.read_csv(user_profile_path, sep='\t')
user_stats = pd.read_csv(user_profile_stats_path, sep='\t')
filtered_users_list  =[]
for y in years:
    filtered_users_list.append(pd.read_csv(user_info_paths.replace('year', str(y)), sep='\t'))


if dataset == 'lfm':
    ages_sort = ['12', '13', '14', '15', '16', '17', '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65'] # Age group can be defined as a range (seperated by '_') or a single age
elif dataset == 'ml':
    ages_sort = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
# Define the age grouping function
def age_group(age):
    age = int(age)
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

filtered_ratios = []
for i, year, filtered_users in zip(list(range(len(years))), years, filtered_users_list):
    filtered_users[f'age_group_{year}'] = filtered_users['age_at_listen'].apply(age_group)
    filtered_users[f'age_group_{year}'] = pd.Categorical(filtered_users[f'age_group_{year}'], categories=ages_sort, ordered=True)

    filtered_users.sort_values(by=f'age_group_{year}', inplace=True)
    filtered_stats_ratio = filtered_users['age_at_listen'].value_counts(normalize=True)
    filtered_users_list[i] = filtered_users
    filtered_ratios.append(filtered_stats_ratio)

user_stats.sort_values(by='age', inplace=True)
user_stats_ratio = user_stats['age'].value_counts(normalize=True)

merged_stats = pd.concat([user_stats_ratio] + filtered_ratios, axis=1)
merged_stats.columns = ['age'] + [f'age_at_listen_{year}' for year in years]


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(merged_stats.index, merged_stats['age'], label='Overall Age')
for year in years:
    year_label_start = f'{year}' if type(year) == int else f'{year.split('_')[1]} filtered'
    year_label_end = f'{year+1}' if type(year) == int else f'{int(year.split('_')[1]) + 1} filtered'
    plt.plot(merged_stats.index, merged_stats[f'age_at_listen_{year}'], label=f'Ages of users in the time of 31-10-{year_label_start} to 31-10-{year_label_end}')
plt.xlabel('Age')
plt.ylabel('Proportion of users')
plt.legend()
plt.show()


print(merged_stats)
for year in years:
    correlation = merged_stats['age'].corr(merged_stats[f'age_at_listen_{year}'])

    print(f"Correlation between overall age distribution and distribution of age at listen in {year}: {correlation}")

