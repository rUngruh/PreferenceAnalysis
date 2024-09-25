import os
import ast
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Gather the Results of the Experiment and compute Preference profiles.')
parser.add_argument('--age_type', type=str, help='Type of age grouping to use', choices=['finegrained_age', 'binary_age', 'finegrained_child_ages'], default='finegrained_age')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], default='lfm')
parser.add_argument('--year', type=int, help='Year of the experiment data for identification', default=2013)
parser.add_argument('--filtered', type=bool, help='Whether the data is filtered or not', default=True)
parser.add_argument('--models', type=str, help='Best models from the experiment', nargs='+', default=["MostPop", "Random_seed=42"])

args = parser.parse_args()

models = args.models

age_type = args.age_type  
dataset = args.dataset
year = args.year
filtered = args.filtered



insert = ''
    
if filtered == True:
    insert += '_filtered'


dataset_dir = os.getenv("dataset_directory")

lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'

recommendations_path = f'../Results/lfm_{year}'
child_recommendations_path = f'../Results/lfm_child_{year}'
results_path = f'../Results/user_and_recommendation_genre_distributions.tsv'


train_path = lfm_data_dir + f'/elliot_data/train{insert}_{str(year)}.tsv'
validation_path = lfm_data_dir + f'/elliot_data/validation{insert}_{str(year)}.tsv'
test_path = lfm_data_dir + f'/elliot_data/test{insert}_{str(year)}.tsv'
user_info_path = lfm_data_dir + f'/elliot_data/user_info{insert}_{str(year)}.tsv'
listening_events_path = lfm_data_dir + f'/elliot_data/listening-events_{str(year)}.tsv'


# Load the data
train = pd.read_csv(train_path, sep='\t', header=None, names=['user_id', 'track_id', 'count', 'timestamp'])
validation = pd.read_csv(validation_path, sep='\t', header=None, names=['user_id', 'track_id', 'count', 'timestamp'])
test = pd.read_csv(test_path, sep='\t', header=None, names=['user_id', 'track_id', 'count', 'timestamp'])
users = pd.read_csv(user_info_path, sep='\t')
    

train = pd.merge(train, users, on='user_id', how='left')
validation = pd.merge(validation, users, on='user_id', how='left')
test = pd.merge(test, users, on='user_id', how='left')

users = users[users['user_id'].isin(train['user_id'].unique())]

les = pd.read_csv(listening_events_path, sep='\t')
les = les[les['user_id'].isin(users['user_id'].unique())]

child_recommendations = {}
recommendations = {}
new_models = []
for model in models:
    model_name = model.split('_')[0] if '_' in model else model
    recommendations[model_name] = pd.read_csv(recommendations_path + f'/{model}.tsv', sep='\t', header=None, names=['user_id', 'track_id', 'score'])
    child_recommendations[model_name] = pd.read_csv(child_recommendations_path + f'/{model}.tsv', sep='\t', header=None, names=['user_id', 'track_id', 'score'])
    new_models.append(model_name)
models = new_models

print('Loaded User Data and Recommendation Data')

artists_path = lfm_data_dir + '/artists.tsv'
artists = pd.read_csv(artists_path, sep='\t')
artist_genre_dict = artists.set_index('artist_id')['genres'].to_dict()


genres = set()
items_to_remove = []

# Extract the genres from the artist_genre_dict
for key, value in artist_genre_dict.items():
    if isinstance(value, float):
        items_to_remove.append(key)
        continue
    extracted_genres = value.split(',')
    artist_genre_dict[key] = extracted_genres
    genres.update(extracted_genres)
    
for item in items_to_remove:
    _ = artist_genre_dict.pop(item)
genres = list(genres)

tracks_path = lfm_data_dir + '/tracks.tsv'

tracks = pd.read_csv(tracks_path, sep='\t')
tracks_artist_dict = tracks.set_index('track_id')['artist_id'].to_dict()

print("Loaded track and genre information.")

def compute_user_profiles(frame, cutoff=None, counts=False):
    # Compute the user profiles from a given frame
    
    frame['artist_id'] = frame['track_id'].apply(lambda x: tracks_artist_dict.get(x, None))

    user_genre_sums = {}
    profile_sizes = {}
    
    for user_id, group in frame.groupby('user_id'):
        if cutoff != None:
            group = group[:cutoff]
        user_genre_sums[user_id] = {}
        for i, track in group.iterrows():
            artist_id = track['artist_id']
            count = track['count'] if counts else 1
            artist_genres = artist_genre_dict.get(artist_id, [])
            for genre in artist_genres:
                if genre in user_genre_sums[user_id]:
                    user_genre_sums[user_id][genre] += ((1 / len(artist_genres)) * count)
                else:
                    user_genre_sums[user_id][genre] = ((1 / len(artist_genres)) * count)
        profile_sizes[user_id] = group['count'].sum() if counts else len(group)
        
    user_profiles = {}
    for key, genre_dict in user_genre_sums.items():
            total_value = sum(genre_dict.values())
            user_profiles[key] = {genre: value / total_value for genre, value in genre_dict.items()}
    
    return user_profiles, profile_sizes


def transform_dict_to_genre_list(d):
    return [d.get(genre, 0) for genre in genres]
    
    

if dataset == 'lfm':
    if age_type == 'finegrained_age':
        ages_sort = ['12', '13', '14', '15', '16', '17', '18', '19-20', '21-22', '23-24', '25-29', '30-34', '35-44', '45-54', '55-65'] # Age group can be defined as a range (seperated by '_') or a single age
    elif age_type == 'binary_age':
        ages_sort = ['12-16', '17-29']
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
        

        

users.sort_values('age_at_listen', inplace=True)
users['age_group'] = users['age_at_listen'].apply(age_group)
users = users.reset_index(drop=True)

users['user_group'] = "mainstream"
users.loc[users['age_at_listen'] < 17, 'user_group'] = "child"
users.loc[users['age_at_listen'] > 29, 'user_group'] = "older"



# Compute the user profiles
train_profiles, train_profile_sizes = compute_user_profiles(train, counts=False)
validation_profiles, validation_profile_sizes = compute_user_profiles(validation, counts=False)
test_profiles, test_profile_sizes = compute_user_profiles(test, counts=False)
print("Computed user profiles")

# Compute the recommendation profiles
model_profiles_5 = {}
model_profiles_10 = {}
model_profiles_20 = {}
model_profiles_50 = {}
model_profiles_100 = {}
model_profiles_200 = {}
child_model_profiles_5 = {}
child_model_profiles_10 = {}
child_model_profiles_20 = {}
child_model_profiles_50 = {}
child_model_profiles_100 = {}
child_model_profiles_200 = {}


for model in models:
    model_profiles_5[model], k = compute_user_profiles(recommendations[model], cutoff=5)
    model_profiles_10[model], k = compute_user_profiles(recommendations[model], cutoff=10)
    model_profiles_20[model], k = compute_user_profiles(recommendations[model], cutoff=20)
    model_profiles_50[model], k = compute_user_profiles(recommendations[model], cutoff=50)
    model_profiles_100[model], k = compute_user_profiles(recommendations[model], cutoff=100)
    model_profiles_200[model], k = compute_user_profiles(recommendations[model], cutoff=200)
    child_model_profiles_5[model], k = compute_user_profiles(child_recommendations[model], cutoff=5)
    child_model_profiles_10[model], k = compute_user_profiles(child_recommendations[model], cutoff=10)
    child_model_profiles_20[model], k = compute_user_profiles(child_recommendations[model], cutoff=20)
    child_model_profiles_50[model], k = compute_user_profiles(child_recommendations[model], cutoff=50)
    child_model_profiles_100[model], k = compute_user_profiles(child_recommendations[model], cutoff=100)
    child_model_profiles_200[model], k = compute_user_profiles(child_recommendations[model], cutoff=200)
    

print("Computed recommendation profiles.")

# Save the results to a tsv file
users['train_genre_distribution'] = None
users['train_profile_size'] = None
users['validation_genre_distribution'] = None
users['validation_profile_size'] = None
users['test_genre_distribution'] = None
users['test_profile_size'] = None
for type_insert in ['child_', '']:
    for model in models:
        users[f'{type_insert}{model}_recommendation_genre_distribution_5'] = None
        users[f'{type_insert}{model}_recommendation_genre_distribution_10'] = None
        users[f'{type_insert}{model}_recommendation_genre_distribution_20'] = None
        users[f'{type_insert}{model}_recommendation_genre_distribution_50'] = None
        users[f'{type_insert}{model}_recommendation_genre_distribution_100'] = None
        users[f'{type_insert}{model}_recommendation_genre_distribution_200'] = None
    
for i, user in users.iterrows():
    user_id = user['user_id']
    #users.at[i, 'le_genre_distribution'] = transform_dict_to_genre_list(le_profiles[user_id])
    users.at[i, 'train_genre_distribution'] = transform_dict_to_genre_list(train_profiles[user_id])
    users.at[i, 'train_profile_size'] = train_profile_sizes[user_id]
    users.at[i, 'validation_genre_distribution'] = transform_dict_to_genre_list(validation_profiles[user_id])
    users.at[i, 'validation_profile_size'] = train_profile_sizes[user_id]
    users.at[i, 'test_genre_distribution'] = transform_dict_to_genre_list(test_profiles[user_id])
    users.at[i, 'test_profile_size'] = train_profile_sizes[user_id]
    for model in models:
        users.at[i, f'{model}_recommendation_genre_distribution_5'] = transform_dict_to_genre_list(model_profiles_5[model][user_id])
        users.at[i, f'{model}_recommendation_genre_distribution_10'] = transform_dict_to_genre_list(model_profiles_10[model][user_id])
        users.at[i, f'{model}_recommendation_genre_distribution_20'] = transform_dict_to_genre_list(model_profiles_20[model][user_id])
        users.at[i, f'{model}_recommendation_genre_distribution_50'] = transform_dict_to_genre_list(model_profiles_50[model][user_id])
        users.at[i, f'{model}_recommendation_genre_distribution_100'] = transform_dict_to_genre_list(model_profiles_100[model][user_id])
        users.at[i, f'{model}_recommendation_genre_distribution_200'] = transform_dict_to_genre_list(model_profiles_200[model][user_id])
        if user['age_at_listen'] < 17:
            users.at[i, f'child_{model}_recommendation_genre_distribution_5'] = transform_dict_to_genre_list(child_model_profiles_5[model][user_id])
            users.at[i, f'child_{model}_recommendation_genre_distribution_10'] = transform_dict_to_genre_list(child_model_profiles_10[model][user_id])
            users.at[i, f'child_{model}_recommendation_genre_distribution_20'] = transform_dict_to_genre_list(child_model_profiles_20[model][user_id])
            users.at[i, f'child_{model}_recommendation_genre_distribution_50'] = transform_dict_to_genre_list(child_model_profiles_50[model][user_id])
            users.at[i, f'child_{model}_recommendation_genre_distribution_100'] = transform_dict_to_genre_list(child_model_profiles_100[model][user_id])
            users.at[i, f'child_{model}_recommendation_genre_distribution_200'] = transform_dict_to_genre_list(child_model_profiles_200[model][user_id])
users.to_csv(results_path, sep='\t')

# Save the list of genres as a txt
with open(f'../Results/genres.txt', 'w') as f:
    for genre in genres:
        f.write(f"{genre}\n")