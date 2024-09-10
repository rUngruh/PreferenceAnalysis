import os
import ast
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Analyze the user profiles in the train, validation, and test sets.')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], default='lfm')
parser.add_argument('--year', type=int, help='Selecting the year for the recommender experiment.', default=2012)
parser.add_argument('--filtered', type=bool, help='Whether to use the filtered version (k-core) of the dataset', default=True)
args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset
year = args.year
filtered = args.filtered

dataset_dir = os.getenv("dataset_directory")
ml_data_dir = dataset_dir + '/processed/ml_with_age'
lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'


if dataset == 'ml':
    ratings_path = ml_data_dir + '/ratings.csv'
    
    
elif dataset == 'lfm':
    train_path = lfm_data_dir + f'/train{"_filtered" if filtered == True else ""}_{str(year)}.tsv'
    validation_path = lfm_data_dir + f'/validation{"_filtered" if filtered == True else ""}_{str(year)}.tsv'
    test_path = lfm_data_dir + f'/test{"_filtered" if filtered == True else ""}_{str(year)}.tsv'
    user_info_path = lfm_data_dir + f'/user_info{"_filtered" if filtered == True else ""}_{str(year)}.tsv'
    listening_events_path = lfm_data_dir + f'/listening-events_{str(year)}.tsv'
    
    train = pd.read_csv(train_path, sep='\t', header=None, names=['user_id', 'track_id', 'count', 'timestamp'])
    validation = pd.read_csv(validation_path, sep='\t', header=None, names=['user_id', 'track_id', 'count', 'timestamp'])
    test = pd.read_csv(test_path, sep='\t', header=None, names=['user_id', 'track_id', 'count', 'timestamp'])
    users = pd.read_csv(user_info_path, sep='\t')
    
if dataset == 'lfm':
    train = pd.merge(train, users, on='user_id', how='inner')
    validation = pd.merge(validation, users, on='user_id', how='inner')
    test = pd.merge(test, users, on='user_id', how='inner')
    


    
    print(f'Train set: {len(train)} rows')
    print(f'Validation set: {len(validation)} rows')
    print(f'Test set: {len(test)} rows')
    
    print(f'Train set: {len(train["user_id"].unique())} unique users')
    print(f'Validation set: {len(validation["user_id"].unique())} unique users')
    print(f'Test set: {len(test["user_id"].unique())} unique users')
    
    train_les_per_user = train.groupby('user_id').size()
    print(f'Train set: {train_les_per_user.mean()} average listening events per user')
    
    validation_les_per_user = validation.groupby('user_id').size()
    print(f'Validation set: {validation_les_per_user.mean()} average listening events per user')
    
    test_les_per_user = test.groupby('user_id').size()
    print(f'Test set: {test_les_per_user.mean()} average listening events per user')
    
    train_les_per_age_group = train.groupby('age_at_listen')

    validation_les_per_age_group = validation.groupby('age_at_listen')
    
    test_les_per_age_group = test.groupby('age_at_listen')

    
    
    empty_train_profiles = users[users['user_id'].isin(train['user_id']) == False]
    empty_validation_profiles = users[users['user_id'].isin(validation['user_id']) == False]
    empty_test_profiles = users[users['user_id'].isin(test['user_id']) == False]
    
    print(f'Train set: {len(empty_train_profiles)} users with empty profiles')
    print(f'Validation set: {len(empty_validation_profiles)} users with empty profiles')
    print(f'Test set: {len(empty_test_profiles)} users with empty profiles')
    
    small_user_profiles = train_les_per_user[train_les_per_user < 10]
    print(f'Train set: {len(small_user_profiles)} users with less than 10 listening events')
    small_user_profiles = validation_les_per_user[validation_les_per_user < 10]
    print(f'Validation set: {len(small_user_profiles)} users with less than 10 listening events')
    small_user_profiles = test_les_per_user[test_les_per_user < 10]
    print(f'Test set: {len(small_user_profiles)} users with less than 10 listening events')
    
    print()
    print()
    
    print("Train set")
    for age, group in train_les_per_age_group:
        print(f'Age: {age}')
        print(f'Number of user profiles: {len(group["user_id"].unique())}')
        print(f'Average items in user profile: {group.groupby("user_id").size().mean()}')
        small_user_profiles = group.groupby("user_id").size()[group.groupby("user_id").size() < 10]
        print(f'Users with less than 10 listening events: {len(small_user_profiles)}')
    print()
    print()
    
    print("Validation set")
    for age, group in validation_les_per_age_group:
        print(f'Age: {age}')
        print(f'Number of user profiles: {len(group["user_id"].unique())}')
        print(f'Average items in user profile: {group.groupby("user_id").size().mean()}')
        small_user_profiles = group.groupby("user_id").size()[group.groupby("user_id").size() < 10]
        print(f'Users with less than 10 listening events: {len(small_user_profiles)}')
    print()
    print()   
    
    print("Test set")
    for age, group in test_les_per_age_group:
        print(f'Age: {age}')
        print(f'Number of user profiles: {len(group["user_id"].unique())}')
        print(f'Average items in user profile: {group.groupby("user_id").size().mean()}')
        small_user_profiles = group.groupby("user_id").size()[group.groupby("user_id").size() < 10]
        print(f'Users with less than 10 listening events: {len(small_user_profiles)}')
    print()
    print()
    