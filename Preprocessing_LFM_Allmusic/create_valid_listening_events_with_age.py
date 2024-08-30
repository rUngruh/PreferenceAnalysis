import sys
import pandas as pd
import os
import logging
import ast

dataset = "lfm"
dataset_dir = os.getenv("dataset_directory")
save_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'
data_dir = dataset_dir + '/raw/LFM-2b'

original_listening_events = data_dir + '/listening-events.tsv.bz2'
original_users = data_dir + '/users_corrected.tsv.bz2'

filtered_users = save_dir + '/users_corrected.tsv'
filtered_tracks = save_dir + '/tracks.tsv'
filtered_listening_events = save_dir + '/listening-events.tsv.bz2'
filtered_listening_counts = save_dir + '/listening-counts.tsv'
filtered_removed_listening_events = save_dir + '/removed_listening-events.tsv.bz2'

batch_size = 10000000

seed=42
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data path '{data_dir}' does not exist.")


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
        


delimiter = '\t' if dataset == "lfm" else '::'
compression='bz2' if dataset == "lfm" else None

def custom_year_diff(date, base_date):
    # Calculate the year difference based on the base_date year
    year_diff = date.year - base_date.year
    
    # Determine if the date is before the custom year start in the given year
    current_year_start = pd.Timestamp(year=date.year, month=base_date.month, day=base_date.day)
    if date < current_year_start:
        year_diff -= 1

    return float(year_diff)
            

users = pd.read_csv(original_users, delimiter=delimiter, compression=compression)
valid_age_users = users[users['age_valid'] == True]['user_id'].unique()

tracks = pd.read_csv(filtered_tracks, delimiter=delimiter)
tracks_with_tags = set(tracks['track_id'].unique())
track_artist_dict = dict(zip(tracks['track_id'], tracks['artist_id']))
del tracks

if  os.path.exists(filtered_listening_events):
    os.remove(filtered_listening_events)

if os.path.exists(filtered_removed_listening_events):
    os.remove(filtered_removed_listening_events)
    
header_written = False

users_removed_for_invalid_age = []
listening_events_removed_for_invalid_age = 0
filtered_for_age_outliers = 0
listening_events_removed_for_missing_tag = 0
listening_events_removed_by_age = {}
overall_listening_events_by_age = {}
remaining_listening_events_by_age = {}
for i, chunk in enumerate(pd.read_csv(original_listening_events, delimiter='\t', chunksize=batch_size)):
    if i % 10 == 0:  # Print status every 10 chunks
        print(f'Processed {i * batch_size:,} rows; current chunk size: {len(chunk):,}')

    original_chunk_size = len(chunk)
    all_users = chunk['user_id'].unique()
    chunk = chunk[chunk['user_id'].isin(valid_age_users)]
    filtered_chunk_size = len(chunk)
    listening_events_removed_for_invalid_age += original_chunk_size - filtered_chunk_size
    users_removed_for_invalid_age.extend([user for user in all_users if user not in chunk['user_id'].unique()])
    
    
    chunk.loc[:, 'timestamp'] = pd.to_datetime(chunk['timestamp'])
    base_date = pd.to_datetime('2013-10-31 00:00:00')

    
    chunk.loc[:, 'age_at_listen'] = chunk['timestamp'].map(lambda date: custom_year_diff(date, base_date)) + \
        chunk['user_id'].map(users.set_index('user_id')['age_on_2013_10_31'])
    
    chunk = chunk[(chunk['age_at_listen'] >= 12) & (chunk['age_at_listen'] <= 65)]
    filtered_for_age_outliers += filtered_chunk_size - len(chunk)
    filtered_chunk_size = len(chunk)
    
    for age, count in chunk['age_at_listen'].value_counts().items():
        overall_listening_events_by_age[age] = overall_listening_events_by_age.get(age, 0) + count

    
    
    
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp']).astype('datetime64[s]')

    # Downcast numerical columns
    chunk['user_id'] = pd.to_numeric(chunk['user_id'], downcast='integer')
    chunk['track_id'] = pd.to_numeric(chunk['track_id'], downcast='integer')
    chunk['age_at_listen'] = pd.to_numeric(chunk['age_at_listen'], downcast='integer')
    
    
    removed_chunk = chunk[~chunk['track_id'].isin(tracks_with_tags)]
    chunk = chunk[chunk['track_id'].isin(tracks_with_tags)]
    chunk['artist_id'] = chunk['track_id'].map(track_artist_dict)
    chunk = chunk.drop(columns=['album_id'])
    chunk['artist_id'] = pd.to_numeric(chunk['artist_id'], downcast='integer')
    
    listening_events_removed_for_missing_tag += len(removed_chunk)
    for age, count in removed_chunk['age_at_listen'].value_counts().items():
        listening_events_removed_by_age[age] = listening_events_removed_by_age.get(age, 0) + count
        
    for age, count in chunk['age_at_listen'].value_counts().items():
        remaining_listening_events_by_age[age] = remaining_listening_events_by_age.get(age, 0) + count
            
    # Convert timestamp to datetime64 and then to seconds precision

    removed_chunk.to_csv(filtered_removed_listening_events, sep='\t', index=False, header=not header_written, mode='a', compression='bz2')
    
    # Write chunk to CSV
    chunk.to_csv(filtered_listening_events, sep='\t', index=False, header=not header_written, mode='a', compression='bz2')
    
    # Set the flag to True after the first write
    if not header_written:
        header_written = True
print('Processing complete.')

num_users_removed_for_invalid_age = len(set(users_removed_for_invalid_age))
print(f'Number of removed users because of invalid age: {num_users_removed_for_invalid_age}')
print(num_users_removed_for_invalid_age)
print()
print(f'Number of removed listening events because of invalid age: {listening_events_removed_for_invalid_age}')
print(listening_events_removed_for_invalid_age)
print()
print(f"Number of listening events removed for age outliers: {filtered_for_age_outliers}")
print()
print(f'Number of removed listening events because of missing tag: {listening_events_removed_for_missing_tag}')
print(listening_events_removed_for_missing_tag)
print()

print('Number of removed listening events by age of listen:')
for age, count in listening_events_removed_by_age.items():
    print(f'Age {age}: {count}')
print()
print()
print()
print('Number of listening events by age of listen:')
for age, count in overall_listening_events_by_age.items():
    print(f'Age {age}: {count}')
print()
print()
print()
print('Remaining listening events by age of listen:')
for age, count in remaining_listening_events_by_age.items():
    print(f'Age {age}: {count}')
print()
print()
print()
listening_events_removed_by_age = dict(sorted(listening_events_removed_by_age.items()))
listening_events_removed_by_age = pd.DataFrame(listening_events_removed_by_age.items(), columns=['user_id', 'age'])
listening_events_removed_by_age.to_csv('listening_events_removed_by_age', sep='\t', index=False)

remaining_listening_events_by_age = dict(sorted(remaining_listening_events_by_age.items()))
remaining_listening_events_by_age = pd.DataFrame(remaining_listening_events_by_age.items(), columns=['user_id', 'age'])
remaining_listening_events_by_age.to_csv('remaining_listening_events_by_age', sep='\t', index=False)

overall_listening_events_by_age = dict(sorted(overall_listening_events_by_age.items()))
overall_listening_events_by_age = pd.DataFrame(overall_listening_events_by_age.items(), columns=['user_id', 'age'])
overall_listening_events_by_age.to_csv('overall_listening_events_by_age', sep='\t', index=False)