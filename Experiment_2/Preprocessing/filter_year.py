import os
import ast
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Filter events by year and save.')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], default='lfm')
parser.add_argument('--chunksize', type=int, default=1000000, help='Chunk size for processing listening events')
parser.add_argument('--year', type=int, help='Selecting the year for the recommender experiment', default=2012)

args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset
chunksize = args.chunksize
year = args.year

dataset_dir = os.getenv("dataset_directory")
ml_data_dir = dataset_dir + '/processed/ml_with_age'
lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'


save_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags/elliot_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start_time = pd.to_datetime(f'{str(year)}-10-31 00:00:00')
end_time = pd.to_datetime(f'{str(year + 1)}-10-31 00:00:00')

if dataset == 'ml':
    ratings_path = ml_data_dir + '/ratings.csv'
    
    
elif dataset == 'lfm':

    listening_events_path = lfm_data_dir + '/listening-events.tsv.bz2'


if dataset == 'lfm':
    if os.path.exists(f'{save_dir}/listening-events_{year}.tsv'):
        os.remove(f'{save_dir}/listening-events_{year}.tsv')
    header_written = False
    num_unique_les = 0
    for i, chunk in enumerate(pd.read_csv(listening_events_path, sep='\t', chunksize=chunksize, compression='bz2')):
        if i % 10 == 0:  # Print status every 10 chunks
            print(f'Processed {i * chunksize:,} rows; current chunk size: {len(chunk):,} out of 1.13 billion ({i * chunksize / 1.13e9:.2%})')

        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        chunk = chunk[(chunk['timestamp'] >= start_time) & (chunk['timestamp'] < end_time)]
        
        chunk = chunk.sort_values(by=['user_id', 'track_id', 'timestamp'])
        
        # Add a 'count' column to track the number of listening events per user-track combination
        chunk['count'] = chunk.groupby(['user_id', 'track_id'])['timestamp'].transform('count')
        
        # Remove duplicate listening events, keeping only the first one per user-track combination
        chunk = chunk.drop_duplicates(subset=['user_id', 'track_id'], keep='first')
        
        num_unique_les += len(chunk)

        # Save the processed chunk to a new TSV file
        chunk.to_csv(f'{save_dir}/listening-events_{year}.tsv', sep='\t', index=False, header=not header_written, mode='a')
        if not header_written:
            header_written = True

        
