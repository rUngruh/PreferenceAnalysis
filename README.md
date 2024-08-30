# Homogenization by Recommender Systems and Its Long-term Effects on Children

## Datasets
Create a directory in which you save [LFM-2b](https://www.cp.jku.at/datasets/LFM-2b/), [LFM-1b](https://www.cp.jku.at/datasets/LFM-1b/), and [MovieLens-1m](https://grouplens.org/datasets/movielens/).

```
├── datasets
│   ├── raw
│   │   ├── LFM-2b
│   │   ├── LFM-1b_UGP
│   │   ├── movielens-1m
│   ├── processed
```
In order to streamline the process, add the path of the `datasets` directory to `paths.env`.


## Preprocessing
### LFM-2b with Allmusic genres
1. Run `Preprocessing_LFM_Allmusic/create_allmusic_artist_profiles.py`
2. Run `Preprocessing_LFM_Allmusic/create_allmusic_tracks.py`
3. Run `Preprocessing_LFM_Allmusic/create_allmusic_tracks_and_users.py`
4. Run `Preprocessing_LFM_Allmusic/create_valid_listening_events_with_age.py`

### Movielens-1m
1. Run `Preprocessing_ML/process_ratings.py`

## Experiment 1: Data Analysis
Run the following scripts with the dataset argument (`ml` or `lfm`) in order to do the analyses for  Experiment 1.

1. Run `Experiment_1/create_user_profiles_in_batches.py` 
2. Run `Experiment_1/analyze_subset.py` and `Experiment_1/analyze_user_profiles.py` in order to further analyze the user profiles.