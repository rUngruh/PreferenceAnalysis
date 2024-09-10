import os
import ast
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Create train, validation and test split, including k-core filtering and removal of invalid profiles.')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], default='lfm')
parser.add_argument('--year', type=int, help='Selecting the year for the recommender experiment', default=2012)
parser.add_argument('--remove_missing_profiles', type=bool, help='Remove users with missing items in train, validation, or test sets', default=True)
parser.add_argument('--k_core_filtering', type=int, help='Minimum number of interactions per user and item, use None if no k_core_filtering', default=10)

args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset
year = args.year
remove_missing_profiles = args.remove_missing_profiles
k_core_filtering = args.k_core_filtering


dataset_dir = os.getenv("dataset_directory")
ml_data_dir = dataset_dir + '/processed/ml_with_age'
lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'

save_path = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags/elliot_data'

train_path = save_path + f'/train{"_filtered" if k_core_filtering else ""}_{str(year)}.tsv'
validation_path = save_path + f'/validation{"_filtered" if k_core_filtering else ""}_{str(year)}.tsv'
test_path = save_path + f'/test{"_filtered" if k_core_filtering else ""}_{str(year)}.tsv'
user_info_path = save_path + f'/user_info{"_filtered" if k_core_filtering else ""}_{str(year)}.tsv'

train_child_path = save_path + f'/train_child{"_filtered" if k_core_filtering else ""}_{str(year)}.tsv'
validation_child_path = save_path + f'/validation_child{"_filtered" if k_core_filtering else ""}_{str(year)}.tsv'
test_child_path = save_path + f'/test_child{"_filtered" if k_core_filtering else ""}_{str(year)}.tsv'


start_time = pd.to_datetime(f'{str(year)}-10-31 00:00:00')
end_time = pd.to_datetime(f'{str(year+1)}-10-31 00:00:00')
validation_start = pd.to_datetime(f'{str(year+1)}-09-01 00:00:00')
test_start = pd.to_datetime(f'{str(year+1)}-10-01 00:00:00')
if dataset == 'ml':
    ratings_path = ml_data_dir + '/ratings.csv'
    
    
elif dataset == 'lfm':
    listening_events_path = f'{lfm_data_dir}/listening-events_{str(year)}.tsv'


if dataset == 'lfm':
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(validation_path):
        os.remove(validation_path)
    if os.path.exists(test_path):
        os.remove(test_path)
    if os.path.exists(user_info_path):
        os.remove(user_info_path)
    if os.path.exists(train_child_path):
        os.remove(train_child_path)
    if os.path.exists(validation_child_path):
        os.remove(validation_child_path)
    if os.path.exists(test_child_path):
        os.remove(test_child_path)
        
    listening_events = pd.read_csv(listening_events_path, sep='\t')
    listening_events['timestamp'] = pd.to_datetime(listening_events['timestamp'])
    listening_events = listening_events[(listening_events['timestamp'] >= start_time) & (listening_events['timestamp'] < end_time)]
    
    listening_events = listening_events.sort_values(by=['user_id', 'timestamp'])
    
    user_info = listening_events[['user_id', 'age_at_listen']].drop_duplicates()
    
    if k_core_filtering:
        nothing_removed = False
        while nothing_removed == False:
            init_users = listening_events['user_id'].nunique()
            init_tracks = listening_events['track_id'].nunique()
            print(f"Initial number of users: {init_users}")
            print(f"Initial number of tracks: {init_tracks}")
            
            # Filter users and tracks that meet the k-core threshold
            user_profile_counts = listening_events.groupby('user_id').size()
            invalid_users = user_profile_counts[user_profile_counts < k_core_filtering].index
            print(f'Number of invalid users: {len(invalid_users)}')
            
            track_counts = listening_events.groupby('track_id').size()
            invalid_tracks = track_counts[track_counts < k_core_filtering].index
            print(f'Number of invalid_tracks: {len(invalid_tracks)}')
            
            # Keep only listening events with valid users and tracks
            listening_events = listening_events[~listening_events['user_id'].isin(invalid_users)]

            
            listening_events = listening_events[~listening_events['track_id'].isin(invalid_tracks)]
            
            # Update the user_info DataFrame to keep only valid users
            user_info = user_info[~user_info['user_id'].isin(invalid_users)]
    
            final_users = listening_events['user_id'].nunique()
            final_tracks = listening_events['track_id'].nunique()
            print(f"Final number of users: {final_users}")
            print(f"Final number of tracks: {final_tracks}")
            print()
            if final_users == init_users and final_tracks == init_tracks:
                nothing_removed = True
        print("Finished k-core filtering")
        
    child_chunk = listening_events[listening_events['age_at_listen'] < 18]
    child_chunk = child_chunk[['user_id', 'track_id', 'count', 'timestamp']]
    listening_events = listening_events[['user_id', 'track_id', 'count', 'timestamp']]
    print("Finished further processing")
    train_chunk = listening_events[listening_events['timestamp'] < validation_start]
    validation_chunk = listening_events[(listening_events['timestamp'] >= validation_start) & (listening_events['timestamp'] < test_start)]
    test_chunk = listening_events[listening_events['timestamp'] >= test_start]
    print("Finished splitting")
    
    del listening_events
    
    if remove_missing_profiles:
        print("Removing missing profiles")

        # Get unique user IDs from each dataset once
        train_user_ids = set(train_chunk['user_id'].unique())
        validation_user_ids = set(validation_chunk['user_id'].unique())
        test_user_ids = set(test_chunk['user_id'].unique())

        # Find valid user IDs that are present in all three datasets
        valid_user_ids = train_user_ids & validation_user_ids & test_user_ids

        print(f'Number of valid users: {len(valid_user_ids)}')

        # Filter user_info and chunks by valid user IDs
        user_info = user_info[user_info['user_id'].isin(valid_user_ids)]
        train_chunk = train_chunk[train_chunk['user_id'].isin(valid_user_ids)]
        validation_chunk = validation_chunk[validation_chunk['user_id'].isin(valid_user_ids)]
        test_chunk = test_chunk[test_chunk['user_id'].isin(valid_user_ids)]
        child_chunk = child_chunk[child_chunk['user_id'].isin(valid_user_ids)]
    
    
    train_child_chunk = child_chunk[child_chunk['timestamp'] < validation_start]
    validation_child_chunk = child_chunk[(child_chunk['timestamp'] >= validation_start) & (child_chunk['timestamp'] < test_start)]
    test_child_chunk = child_chunk[child_chunk['timestamp'] >= test_start]
    
    
        
    train_chunk.to_csv(train_path, sep='\t', index=False, header=False)
    validation_chunk.to_csv(validation_path, sep='\t', index=False, header=False)
    test_chunk.to_csv(test_path, sep='\t', index=False, header=False)
    header_written = os.path.exists(user_info_path)
    user_info.to_csv(user_info_path, sep='\t', index=False, header=not header_written)
    
    train_child_chunk.to_csv(train_child_path, sep='\t', index=False, header=False)
    validation_child_chunk.to_csv(validation_child_path, sep='\t', index=False, header=False)
    test_child_chunk.to_csv(test_child_path, sep='\t', index=False, header=False)