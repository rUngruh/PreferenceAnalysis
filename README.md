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

## Experiment 2:
### Step1: Dataset Processing and Splitting
Here, we select a subset of LFM-2b for our recommendation analysis, make an informed decision about the selection of the subset, and create a temporal split for validation and testing. 
1. Run `Experiment_2/Preprocessing/filter_year.py` and select years that could be valid for the Experiment.
2. Run `Experiment_2/Preprocessing/split_set.py`
3. Analyze and compare subsets with `Experiment_2/Preprocessing/analyze_subset.py` and `Experiment_2/Preprocessing/compare_user_stats_to_filtered_set.py` in order to validate that the filtered sets for the training are in accordance with the original data from Exp1.

Now, we utilize the selected subset for training and evaluating using Elliot.


### Step2: Preparing Elliot
- Create conda environment
```
conda create --yes --name elliot-env python=3.8
conda activate elliot-env
cd Experiment_2 
git clone https://github.com//sisinflab/elliot.git && cd elliot
pip install --upgrade pip
pip install -e . --verbose
```

In our case, we had to downgrade protobuf
```
pip install protobuf==3.20.3
```

Add the train, validation, and test splits from the previous steps to the data directory in elliot.
1. From `processed/lfm_with_lfm1b_allmusic_tags/elliot_data` add `train_child_filtered_2012`, `validation_child_filtered_2012`, and `test_child_filtered_2012` to `Experiment_2/elliot/data/lfm_child_2012` as `train.tsv`, `validation.tsv`, and `test.tsv`, respectively.
2. From `processed/lfm_with_lfm1b_allmusic_tags/elliot_data` add `train_filtered_2012`, `validation_filtered_2012`, and `test_filtered_2012` to `Experiment_2/elliot/data/lfm_2011` as `train.tsv`, `validation.tsv`, and `test.tsv`, respectively.

Finally, add the config_files (`Experiment_2/config_files`) to `Experiment_2/elliot/config_files`


### Step3: Running Experiments using Elliot
Run experiments with Elliot for the full set and for children only.
```
python start-experiments.py --config child_config
```
```
python start-experiments.py --config all_user_config
```


### Step 4: Post-Process the Results
- Move the best performing models on the validation set to `Experiment_2/Results`
- Run `Experiment_2/Result_Analysis/Process_Results.py` in order to compute the User Genre Profiles and Recommendation Genre Profiles.


### Step 5: Evaluate the Results of the Experiments
- Run `Experiment_2/Results_Analysis/Performance_Analysis.py` in order to compute the performance for users of different age groups.