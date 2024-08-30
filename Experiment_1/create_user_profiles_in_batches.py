import os
import pandas as pd
from scipy.special import kl_div
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import ast
import argparse

dataset_dir = os.getenv("dataset_directory")
ml_data_dir = dataset_dir + '/processed/ml_with_age'
lfm_data_dir = dataset_dir + '/processed/lfm_with_lfm1b_allmusic_tags'

parser = argparse.ArgumentParser(description='Create user profiles in batches.')
parser.add_argument('--dataset', type=str, help='Dataset to use (ml or lfm)', choices=['ml', 'lfm'], required=True)
parser.add_argument('--chunksize', type=int, default=1000000, help='Chunk size for processing listening events')
parser.add_argument('--weighted', type=bool, help='Use weighted listening events', default=True)

args = parser.parse_args()

# Use the dataset argument
dataset = args.dataset
chunksize = args.chunksize
weighted = args.weighted


if dataset == 'ml':
    user_profile_path = ml_data_dir + '/user_profiles.tsv' if not weighted else ml_data_dir + '/user_profiles_weighted.tsv'
    user_profile_stats_path = ml_data_dir + '/user_profile_stats.tsv' if not weighted else ml_data_dir + '/user_profile_stats_weighted.tsv'
    ratings_path = ml_data_dir + '/ratings.tsv'
    tags_path = ml_data_dir + '/genres.tsv'
    user_path = ml_data_dir + '/users.tsv'
    
    ratings = pd.read_csv(ratings_path, sep='\t')
    tags = pd.read_csv(tags_path, sep='\t')
    tags['tags'] = tags['tags'].apply(lambda x: ast.literal_eval(x))
    users = pd.read_csv(user_path, sep='\t')
    
if dataset == 'lfm':
    user_profile_path = lfm_data_dir + '/user_profiles.tsv' if not weighted else lfm_data_dir + '/user_profiles_weighted.tsv'
    user_profile_stats_path = lfm_data_dir + '/user_profile_stats.tsv' if not weighted else lfm_data_dir + '/user_profile_stats_weighted.tsv'


if dataset == 'lfm':
    listening_events_path = lfm_data_dir + '/listening-events.tsv.bz2'
    artists_path = lfm_data_dir + '/artists.tsv'
    user_path = lfm_data_dir + '/users_corrected.tsv'
    


    artists = pd.read_csv(artists_path, sep='\t')
    artist_genre_dict = artists.set_index('artist_id')['genres'].to_dict()
    items_to_remove = []
    for key, value in artist_genre_dict.items():
        if isinstance(value, float):
            items_to_remove.append(key)
            continue
        artist_genre_dict[key] = value.split(',')
    for item in items_to_remove:
        artist_genre_dict.pop(item)
    
   
if dataset == 'lfm':                
    user_profile_data = []
    stats_profile_data = []


    user_genre_sums = {}
    les = 0

    user_tracks = {}
    tracks_per_user = {}
    
    for i, chunk in enumerate(pd.read_csv(listening_events_path, sep='\t', compression='bz2', chunksize=chunksize)):
        les += len(chunk)
        if i % 10 == 0:
            print(f"Processing batch {i}, current batch size: {chunksize}; out of 1.3 billion, percentage: {(i * chunksize / 1131465529) * 100:.3f}%")
        
        if weighted:
            chunk_user_listening_events_dict = (chunk
            .groupby(['user_id', 'age_at_listen'])
            .apply(lambda df: list(zip(df['track_id'], df['artist_id'])))
            .to_dict()
        )
        else:
            chunk_user_listening_events_dict = (chunk
                .groupby(['user_id', 'age_at_listen'])
                .apply(lambda df: list(set(zip(df['track_id'], df['artist_id']))))
                .to_dict()
            )
        del chunk

        
        for key, track_artist_tuples in chunk_user_listening_events_dict.items():
            
            if key in user_tracks:
                unique_track_artist_tuples = [t for t in list(set(track_artist_tuples)) if t[0] not in user_tracks[key]]
                if not weighted:
                    track_artist_tuples = unique_track_artist_tuples
                    
            else:
                unique_track_artist_tuples = [t for t in list(set(track_artist_tuples))]
                # If the key does not exist in user_tracks, initialize it with the current track_ids
                user_tracks[key] = []
                user_genre_sums[key] = {}
            
            user_tracks[key].extend([t[0] for t in unique_track_artist_tuples])
            
            tracks_per_user[key] = len(track_artist_tuples)
            
            for track_id, artist_id in track_artist_tuples:
                genres = artist_genre_dict.get(artist_id, [])
                for genre in genres:
                    if genre in user_genre_sums[key]:
                        user_genre_sums[key][genre] += 1 / len(genres)
                    else:
                        user_genre_sums[key][genre] = 1 / len(genres)
        
    print(f"Processed {les} listening events.")
    print('Processed listening events. Now, computing genre distributions across users...')
    for key, genre_dict in user_genre_sums.items():
        total_value = sum(genre_dict.values())
        user_genre_sums[key] = {genre: value / total_value for genre, value in genre_dict.items()}

    print('Processed user profiles.')


    print('Creating dataframes...')
    users = pd.read_csv(user_path, sep='\t')
    user_profile_data = []
    for key, items in user_tracks.items():

        user_profile_data.append({
            'user_id': key[0],
            'age': key[1],
            'items': ','.join(map(str, items)),
            #'top_genres': ','.join(user_genres),
            'gender': users[users['user_id'] == key[0]]['gender'].values[0],
            
        })
    del user_tracks
    
    stats_profile_data = []
    for key, user_genre_distribution in user_genre_sums.items():
            stats_profile_data.append({
                'user_id': key[0],
                'age': key[1],
                'gender': users[users['user_id'] == key[0]]['gender'].values[0],
                'num_items': tracks_per_user[key],
                'normalized_genre_distribution': user_genre_distribution,
                #'normalized_top_genre_distribution': user_top_genre_distribution,
            })


    print('Processing complete.')
    
    
    
if dataset == 'ml':
    user_profile_data = []
    stats_profile_data = []

    # Create a dictionary to map user_ids to their listening events
    if weighted:
        user_listening_events_dict = ratings.groupby('user_id').apply(lambda x: x['movie_id'].tolist()).to_dict()
    else:
        user_listening_events_dict = ratings.groupby('user_id').apply(lambda x: x['movie_id'].unique().tolist()).to_dict()
        
    # Create a dictionary to map movie_ids to their genres
    movie_genres_dict = {row['movie_id'] : row['tags'] for i, row in tags[['movie_id', 'tags']].iterrows()}

    for i, user in users.iterrows():
        if i % 100 == 0:
            print(f"Processing user {i} of {len(users)}; {i/len(users)*100:.2f}%")
        user_id = user['user_id']

        user_items = user_listening_events_dict.get(user_id, [])
        gender = user['gender']
        age = user['age']

        # Collect genres for the user's items
        
        user_genres = [movie_genres_dict[movie_id] for movie_id in user_items]
        
        user_top_genre_distribution = pd.Series([genres[0] for genres in user_genres], dtype='str').value_counts(normalize=True).to_dict()
        
        # Normalize genres
        user_normalized_genres = [{genre: 1 / len(movie_genres_dict[movie_id]) for genre in movie_genres_dict[movie_id]} for movie_id in user_items if movie_id in movie_genres_dict]
        user_normalized_genre_distribution = {}
        
        for item_genres in user_normalized_genres:
            for genre, value in item_genres.items():
                if genre in user_normalized_genre_distribution:
                    user_normalized_genre_distribution[genre] += value
                else:
                    user_normalized_genre_distribution[genre] = value
        
        # Normalize the aggregated dictionary
        total_value = sum(user_normalized_genre_distribution.values())
        if total_value > 0:
            user_normalized_genre_distribution = {
                genre: value / total_value for genre, value in user_normalized_genre_distribution.items()
            }
       
        #print(user_genres)

        user_profile_data.append({
            'user_id': user_id,
            'items': ','.join(map(str, user_items)),
            #'top_genres': ','.join(user_genres),
            'gender': gender,
            'age': age,
        })

        if user_top_genre_distribution:
            stats_profile_data.append({
                'user_id': user_id,
                'gender': gender,
                'age': age,
                'num_items': len(user_items),
                'top_genre_distribution': user_top_genre_distribution,
                'normalized_genre_distribution': user_normalized_genre_distribution
            })

    print('Processing complete.')
    
    
print('Saving user profiles...')
if os.path.exists(user_profile_path):
    os.remove(user_profile_path)
if os.path.exists(user_profile_stats_path):
    os.remove(user_profile_stats_path)
    
user_profile = pd.DataFrame(user_profile_data)


# Save the DataFrame to a CSV file
user_profile.to_csv(user_profile_path, sep='\t', index=False)

stats_profile = pd.DataFrame(stats_profile_data)

stats_profile.to_csv(user_profile_stats_path, sep='\t', index=False)




print("Unique users: ", len(user_profile['user_id'].unique()))
print("Differen profiles: ", len(user_profile))
print("Average number of items per user per age: ", user_profile['items'].apply(lambda x: len(x.split(','))).mean())
