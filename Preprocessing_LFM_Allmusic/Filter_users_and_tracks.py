import os
import ast
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Filter events by year and save.')
parser.add_argument('--chunksize', type=int, default=1000000, help='Chunk size for processing listening events')
#parser.add_argument('--start_date', type=str, help='Start date for the split', default='2012-10-31')
#parser.add_argument('--end_date', type=str, help='End date for the split', default='2013-10-31')

args = parser.parse_args()

chunksize = args.chunksize

# This script assumes data that was preprocessed according to the original paper's methodology. 
# Refer to the README for more details.

dataset_dir = os.getenv("dataset_directory")

dataset_dir = "C:/Users/rungruh/OneDrive - Delft University of Technology/Documents/datasets"

lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'


save_dir = lfm_data_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


listening_events_save_path = f'{save_dir}/listening-events_sampled.tsv'
tracks_save_path = f'{save_dir}/tracks_valid.tsv'
users_save_path = f'{save_dir}/users_valid.tsv'
artists_save_path = f'{save_dir}/artists_valid.tsv'
    


listening_events_path = lfm_data_dir + '/listening-events.tsv.bz2'
tracks_path = lfm_data_dir + '/tracks.tsv'
users_path = lfm_data_dir + '/users_corrected.tsv'
artists_path = lfm_data_dir + '/artists.tsv'

tracks = pd.read_csv(tracks_path, sep='\t')
users = pd.read_csv(users_path, sep='\t')
artists = pd.read_csv(artists_path, sep='\t')



user_ids = set()
track_ids = set()
artist_ids = set()



for i, chunk in enumerate(pd.read_csv(listening_events_path, sep='\t', chunksize=chunksize, compression='bz2')):
    if i % 10 == 0:  # Print status every 10 chunks
        print(f'Processed {i * chunksize:,} rows; current chunk size: {len(chunk):,} out of 1.13 billion ({i * chunksize / 1.13e9:.2%})')

    user_ids.update(chunk['user_id'].unique())
    track_ids.update(chunk['track_id'].unique())
    artist_ids.update(chunk['artist_id'].unique())
    
tracks = tracks[tracks['track_id'].isin(track_ids)]
users = users[users['user_id'].isin(user_ids)]
artists = artists[artists['artist_id'].isin(artist_ids)]
len(tracks)
tracks.to_csv(tracks_save_path, sep='\t', index=False)
users.to_csv(users_save_path, sep='\t', index=False)
artists.to_csv(artists_save_path, sep='\t', index=False)