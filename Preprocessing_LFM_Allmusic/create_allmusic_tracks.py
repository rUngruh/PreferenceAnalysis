import os
import pandas as pd

dataset = "lfm"
dataset_dir = os.getenv("dataset_directory")
save_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'
data_dir = dataset_dir + '/raw/LFM-2b'

original_tracks = data_dir + '/tracks.tsv.bz2'

allmusic_artists = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags/artists.tsv'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


filtered_tracks = save_dir + '/tracks.tsv'

batch_size = 1000000

if os.path.exists(filtered_tracks):
    os.remove(filtered_tracks)
artist_df = pd.read_csv(allmusic_artists, sep='\t', encoding='utf-8')
print(f"Number of artists initially: {len(artist_df)}")
artist_df = artist_df[(artist_df['artist'].notnull()) & (artist_df['genres'].notnull())]

print(f"Number of artists after filtering for those with attached Genre: {len(artist_df)}")

num_overall_tracks = 0
num_filtered_tracks = 0

header_written = False
for i, track_chunk in enumerate(pd.read_csv(original_tracks, sep='\t', compression='bz2', encoding='utf-8', on_bad_lines='skip', chunksize=batch_size)):
    if i % 10 == 0:
        print(f"Processing chunk {i}...")
    num_overall_tracks += len(track_chunk)
    track_chunk = track_chunk.merge(artist_df, how='inner', left_on='artist_name', right_on='artist')
    
    track_chunk = track_chunk.drop(['artist', 'genres', 'artist_name'], axis=1)
    num_filtered_tracks += len(track_chunk)
    
    track_chunk.to_csv(filtered_tracks, sep='\t', index=False, header=not header_written, mode='a')
    if not header_written:
        header_written = True
    

print(f"Number of tracks initially: {num_overall_tracks}")
print(f"Number of tracks after filtering for those with attached Genre: {num_filtered_tracks}")
