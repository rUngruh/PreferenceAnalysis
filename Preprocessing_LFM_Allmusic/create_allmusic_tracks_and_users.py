
import sys
import pandas as pd
import os
import logging
import ast

dataset = "lfm"
dataset_dir = os.getenv("dataset_directory")
save_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'
data_dir = dataset_dir + '/raw/LFM-2b'
original_users = data_dir + '/users_corrected.tsv.bz2'

processed_tracks = save_dir + '/tracks.tsv'

original_listening_events = data_dir + '/listening-events.tsv.bz2'
original_listening_counts = data_dir + '/listening-counts.tsv.bz2'

filtered_users = save_dir + '/users_corrected.tsv'
filtered_listening_counts = save_dir + '/listening-counts.tsv'

batch_size = 1000000

seed=42
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data path '{data_dir}' does not exist.")


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
        


delimiter = '\t' if dataset == "lfm" else '::'
compression='bz2' if dataset == "lfm" else None

#Process Tags
print('Processing tags...')
tracks = pd.read_csv(processed_tracks, delimiter=delimiter)
#track_artist_dict = dict(zip(tracks['track_id'], tracks['artist_id']))
valid_track_ids = set(tracks['track_id'].unique())
del tracks
print('Tags processed and saved')

print('Get user ids with valid ages...')
users = pd.read_csv(original_users, delimiter=delimiter, compression=compression)
valid_age_users = users[users['age_valid'] == True]['user_id'].unique()

# Process listening counts
print('Processing listening counts...')
num_removed_lines_invalid_age = 0
num_removed_lines_missing_genre = 0

valid_user_ids = []
if os.path.exists(filtered_listening_counts):
    os.remove(filtered_listening_counts)
header_written = False
for i, chunk in enumerate(pd.read_csv(original_listening_counts, delimiter=delimiter, chunksize=batch_size, compression=compression)):
    len_original_chunk = len(chunk)
    chunk = chunk[chunk['user_id'].isin(valid_age_users)]
    len_filtered_chunk = len(chunk)
    num_removed_lines_invalid_age += len_original_chunk - len_filtered_chunk
    len_original_chunk = len(chunk)
    chunk = chunk[chunk['track_id'].isin(valid_track_ids)]
    len_filtered_chunk = len(chunk)
    num_removed_lines_missing_genre += len_original_chunk - len_filtered_chunk
    valid_user_ids.extend(chunk['user_id'].unique())
    # Write chunk to CSV
    chunk.to_csv(filtered_listening_counts, sep='\t', index=False, header=not header_written, mode='a')

    # Set the flag to True after the first write
    if not header_written:
        header_written = True


valid_user_ids = set(valid_user_ids)
print(f'Number of removed user-track pairs because user with invalid age: {num_removed_lines_invalid_age}')
print(f'Number of removed user-track pairs because track with no tags: {num_removed_lines_missing_genre}')

print('Listening counts processed and saved')

# Process users
print('Processing users...')
if os.path.exists(filtered_users):
    os.remove(filtered_users)


all_ids = set(users['user_id'].unique())
users = users[users['user_id'].isin(valid_user_ids)]

users['creation_time'] = pd.to_datetime(users['creation_time'])

users.to_csv(filtered_users, sep='\t', index=False, header=True)

removed_users = all_ids - set(users['user_id'].unique())

removed_users_file = save_dir + '/removed_users.txt' 
with open(removed_users_file, 'w') as f: 
    for user_id in removed_users: 
        f.write(str(user_id) + '\n') 
        
print('Removed user IDs saved')
