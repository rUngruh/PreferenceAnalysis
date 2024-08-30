import os
import pandas as pd

dataset = "lfm"
dataset_dir = os.getenv("dataset_directory")
save_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'
allmusic_artists = dataset_dir + '/raw/LFM-1b_UGP/LFM-1b_artist_genres_allmusic.txt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create a dictionary to store the artist profiles
artist_profiles = {}
genre_dict = {}
with open(dataset_dir + '/raw/LFM-1b_UGP/genres_allmusic.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line_num, line in enumerate(lines):
        genre_dict[line_num] = line.strip()

with open(allmusic_artists, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    # Process each line in the file
    for line in lines[1:]:
        # Split the line by tab
        values = line.strip().split('\t')
        
        # Extract the artist name and genres
        artist_name = values[0]
        genres = values[1:]
        
        # Store the artist profile in the dictionary
        
        artist_profiles[artist_name] = ','.join([genre_dict[int(genre)] for genre in genres])
        

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(artist_profiles, orient='index')

df = df.reset_index()
df.columns = ['artist', 'genres']
df['artist_id'] = df.index
df = df[['artist_id', 'artist', 'genres']]

df.to_csv(save_dir + '/artists.tsv', sep='\t', index=False)