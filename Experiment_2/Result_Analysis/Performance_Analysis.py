import os
import ast
import pandas as pd
import numpy as np


import argparse

parser = argparse.ArgumentParser(description='Compute performance for different age groups.')
parser.add_argument('--age_type', type=str, help='Type of age grouping to use', choices=['finegrained_age', 'binary_age', 'finegrained_child_ages'], default='finegrained_age')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], default='lfm')
parser.add_argument('--year', type=int, help='Year of the experiment data for identification', default=2013)
parser.add_argument('--filtered', type=bool, help='Whether the data is filtered or not', default=True)
parser.add_argument('--models', type=str, help='Best models from the experiment', nargs='+', default=["MostPop", "Random_seed=42"])
parser.add_argument('group_by', type=str, help='Group by user_group or age_group', choices=['user_group', 'age_group', 'age_at_listen'], default='user_group')
parser.add_argument('--cutoff', type=int, help='Cutoff for evaluation', default=50)

args = parser.parse_args()

models = args.models

age_type = args.age_type  
dataset = args.dataset
year = args.year
filtered = args.filtered
group_by = args.group_by
cutoff = args.cutoff


insert = ''
    
if filtered == True:
    insert += '_filtered'
    

dataset_dir = os.getenv("dataset_directory")

lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'

recommendations_path = f'../Results/lfm_{year}'
child_recommendations_path = f'../Results/lfm_child_{year}'

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

# Transform the data
train = train.groupby('user_id').agg({'track_id': list}).reset_index()
validation = validation.groupby('user_id').agg({'track_id': list}).reset_index()
test = test.groupby('user_id').agg({'track_id': list}).reset_index()


train = pd.merge(train, users, on='user_id', how='left')[['user_id', 'track_id']]
train.columns = ['user_id', 'train_ids']
validation = pd.merge(validation, users, on='user_id', how='left')[['user_id', 'track_id']]
validation.columns = ['user_id', 'validation_ids']
test = pd.merge(test, users, on='user_id', how='left')[['user_id', 'track_id']]
test.columns = ['user_id', 'test_ids']


users = users[users['user_id'].isin(train['user_id'].unique())]

# Load and transform the recommendations
child_recommendations = {}
recommendations = {}
new_models = []
for model in models:
    model_name = model.split('_')[0] if '_' in model else model
    recommendations[model_name] = pd.read_csv(recommendations_path + f'/{model}.tsv', sep='\t', header=None, names=['user_id', 'track_id', 'score'])
    child_recommendations[model_name] = pd.read_csv(child_recommendations_path + f'/{model}.tsv', sep='\t', header=None, names=['user_id', 'track_id', 'score'])
    new_models.append(model_name)
    recommendations[model_name] = recommendations[model_name].groupby('user_id').agg({'track_id': list}).reset_index()
    recommendations[model_name] = pd.merge(recommendations[model_name], users, on='user_id', how='left')[['user_id', 'track_id']]
    recommendations[model_name].columns = ['user_id', 'rec_ids']
    child_recommendations[model_name] = child_recommendations[model_name].groupby('user_id').agg({'track_id': list}).reset_index()
    child_recommendations[model_name] = pd.merge(child_recommendations[model_name], users, on='user_id', how='left')[['user_id', 'track_id']]
    child_recommendations[model_name].columns = ['user_id', 'rec_ids']
    
models = new_models




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



# Compute the nDCG for a user and a model at a given cutoff
def user_nDCG(user_id, test_items, recommendations, k):
    # Calculate the discounted cumulative gain at k
    DCG = 0
    for i in range(min(k, len(recommendations))):
        item = recommendations[i]
        if item in test_items:
            DCG += 1 / np.log2(i + 2)
    # Calculate the ideal discounted cumulative gain at k
    test_items = list(test_items)
    IDCG = 0
    for i in range(min(k, len(test_items))):
        item = test_items[i]
        IDCG += 1 / np.log2(i + 2)
    # Calculate the normalized discounted cumulative gain at k
    nDCG = DCG / IDCG
    return nDCG


def user_MRR(user_id, test_items, recommendations, k):
    for i in range(k):
        item = recommendations[i]
        if item in test_items:
            return 1 / (i + 1)
    return 0

def user_MAP(user_id, test_items, recommendations, k):
    AP = 0
    num_hits = 0
    for i in range(k):
        item = recommendations[i]
        if item in test_items:
            num_hits += 1
            AP += num_hits / (i + 1)
    if num_hits == 0:
        return 0
    return AP / num_hits

performance = {model: {'nDCG': [], 'MRR': [], 'MAP': []} for model in models}
ages = []

for age, group in users.groupby(group_by):
    print(age)
    print('Recommendations for all users')
    ages.append(age)
    
    group = pd.merge(group, test[['user_id', 'test_ids']], on='user_id', how='left')
    
    for model in models:
        print(model)
        
        group = pd.merge(group, recommendations[model][['user_id', 'rec_ids']], on='user_id', how='left')
        group[f'nDCG_{model}_{cutoff}'] = group.apply(lambda x: user_nDCG(x['user_id'], x['test_ids'], x['rec_ids'], cutoff), axis=1)
        group[f'MRR_{model}_{cutoff}'] = group.apply(lambda x: user_MRR(x['user_id'], x['test_ids'], x['rec_ids'], cutoff), axis=1)
        group[f'MAP_{model}_{cutoff}'] = group.apply(lambda x: user_MAP(x['user_id'], x['test_ids'], x['rec_ids'], cutoff), axis=1)
        
        # Store the means in the performance dictionary
        ndcg_mean = group[f'nDCG_{model}_{cutoff}'].mean()
        performance[model]['nDCG'].append(ndcg_mean)
        print(f'ndcg_mean: {ndcg_mean:.4f}')
        mrr_mean = group[f'MRR_{model}_{cutoff}'].mean()
        performance[model]['MRR'].append(mrr_mean)
        print(f'mrr_mean: {mrr_mean:.4f}')
        map_mean = group[f'MAP_{model}_{cutoff}'].mean()
        performance[model]['MAP'].append(map_mean)
        print(f'map_mean: {map_mean:.4f}')
        
        group = group.drop(columns=['rec_ids'])
    
    
    group = group[group['age_at_listen']<18]
    if group.shape[0] == 0:
        continue
    print('Recommendations for children')
    for model in models:
        print(model)

        group = pd.merge(group, child_recommendations[model][['user_id', 'rec_ids']], on='user_id', how='left')
        group[f'nDCG_{model}_{cutoff}'] = group.apply(lambda x: user_nDCG(x['user_id'], x['test_ids'], x['rec_ids'], cutoff), axis=1)
        group[f'MRR_{model}_{cutoff}'] = group.apply(lambda x: user_MRR(x['user_id'], x['test_ids'], x['rec_ids'], cutoff), axis=1)
        group[f'MAP_{model}_{cutoff}'] = group.apply(lambda x: user_MAP(x['user_id'], x['test_ids'], x['rec_ids'], cutoff), axis=1)

        # Store the means in the performance dictionary
        ndcg_mean = group[f'nDCG_{model}_{cutoff}'].mean()
        print(f'ndcg_mean: {ndcg_mean:.4f}')
        #performance[model]['nDCG'].append(ndcg_mean)
        mrr_mean = group[f'MRR_{model}_{cutoff}'].mean()
        print(f'mrr_mean: {mrr_mean:.4f}')
        #performance[model]['MRR'].append(mrr_mean)
        map_mean = group[f'MAP_{model}_{cutoff}'].mean()
        print(f'map_mean: {map_mean:.4f}')
        #performance[model]['MAP'].append(map_mean)
        
        group = group.drop(columns=['rec_ids'])


import matplotlib.pyplot as plt

for metric in ['nDCG', 'MRR', 'MAP']:
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.plot(ages, performance[model][metric], label=model)
    plt.title(f'{metric} over Age Groups')
    plt.xlabel('Age')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()

