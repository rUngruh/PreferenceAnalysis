import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd


models = ['Random', 'MostPop', 'RP3beta', 'iALS']
test_model = 'RP3beta'
cutoff = 50

dataset = 'lfm'
year = 2013

results_path = f'../Results/user_and_recommendation_genre_distributions.tsv'

genre_path = f'../Results/genres.txt'


with open(genre_path, 'r') as f:
    genres = f.readlines()
genres = [genre.split('\n')[0].capitalize() for genre in genres]


with open('../genres_lfm.txt', 'r') as f:
    genres_order_lfm = f.readlines()
genres_order_lfm = [genre.split('\n')[0].capitalize() for genre in genres_order_lfm]

results = pd.read_csv(results_path, sep='\t')

def safe_literal_eval(val):
    if val is None:
        return None
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # Handle any case where literal_eval might fail (if needed)
        return val
    
for column in results.columns:
    if 'genre_distribution' in column:
        results[column] = results[column].apply(safe_literal_eval)

from scipy.spatial.distance import jensenshannon

results['user_group'] = "Mainstream"
results.loc[results['age_at_listen'] < 17, 'user_group'] = "Children"
results.loc[results['age_at_listen'] > 29, 'user_group'] = "NMA"

age_groups = results['age_at_listen'].unique()
user_groups = results['user_group'].unique()
mainstream_distribution = results[results['user_group'] == "mainstream"]['train_genre_distribution'].apply(np.array).mean(axis=0)
child_distribution = results[results['user_group'] == "child"]['train_genre_distribution'].apply(np.array).mean(axis=0)


results_grouped = results.groupby('user_group')

data = []

for age, age_group in results_grouped:
    mean_genre_dict = {'age': f'{age}', 'model': f'AGP$_{{{age}}}$', 'dataset' : ''}
    for index, genre in enumerate(genres):
        mean_genre_dict[genre] = age_group['train_genre_distribution'].apply(lambda x: x[index]).mean()
    data.append(mean_genre_dict)
    
    if age == 'Children':
        for model in models:
            mean_genre_dict = {'age': f'{age}', 'model': model, 'dataset' : 'child-focused'}
            for index, genre in enumerate(genres):
                mean_genre_dict[genre] = age_group[f'child_{model}_recommendation_genre_distribution_{cutoff}'].apply(lambda x: x[index]).mean()
            data.append(mean_genre_dict)
       
    for model in models:
        mean_genre_dict = {'age': f'{age}', 'model': model, 'dataset' : 'unbalanced'}
        for index, genre in enumerate(genres):
            mean_genre_dict[genre] = age_group[f'{model}_recommendation_genre_distribution_{cutoff}'].apply(lambda x: x[index]).mean()
        data.append(mean_genre_dict)
    

mean_genre_df = pd.DataFrame(data)

genres_order_lfm.reverse()
mean_genre_df = mean_genre_df[['age', 'model', 'dataset'] + genres_order_lfm]

print(mean_genre_df)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the plot style
plt.figure(figsize=(20, 8))  
sns.set(style='whitegrid')

# Create a unique index combining age, dataset, and model
mean_genre_df['age_dataset_model'] = mean_genre_df['age'] + ' ' + mean_genre_df['dataset'] + ' ' + mean_genre_df['model']

mean_genre_df = mean_genre_df.drop(columns=['age', 'dataset', 'model'])

# Set this as the index
mean_genre_df = mean_genre_df.set_index('age_dataset_model')

# Prepare variables
bar_positions = np.arange(len(mean_genre_df), dtype=float)  # Ensure positions are float
bar_widths = np.ones_like(bar_positions) * 0.8  # Default width for all bars

wider_bars = [0, 9, 14]  # Indexes of bars to make wider
extra_width = 0.2  # Additional width for the wider bars
# Modify specific bars
bar_widths[wider_bars] = 1 + extra_width  # Make the 1st, 10th, and 15th bars wider

# Insert gaps after specific bars by adding space to positions
gaps = [8, 13]  # After which bars to insert space
space = 1  # Amount of space to insert

small_gaps = [0, 4, 9, 14]
small_space = 0.25

for gap in gaps:
    bar_positions[gap + 1:] += space  # Shift bars after the specified ones

for bar in wider_bars:
    bar_positions[bar + 1:] += extra_width  # Shift bars after the specified ones

for gap in small_gaps:
    bar_positions[gap + 1:] += small_space  # Shift bars after the specified ones

# Plot manually for each column in the DataFrame (each genre)
bottoms = np.zeros(len(mean_genre_df))

colors = sns.color_palette('tab20', n_colors=len(mean_genre_df.columns))
for genre in mean_genre_df.columns:
    plt.bar(bar_positions, mean_genre_df[genre], width=bar_widths, bottom=bottoms, label=genre, color=colors.pop(0))
    print(mean_genre_df[genre])
    bottoms += mean_genre_df[genre]

# Add text centered between the first and 9th bar
plt.text(np.mean(bar_positions[:9]), 1.05, 'Children', ha='center', fontsize=24, fontweight='bold')
plt.text(np.mean(bar_positions[9:14]), 1.05, 'Mainstream', ha='center', fontsize=24, fontweight='bold')
plt.text(np.mean(bar_positions[14:]), 1.05, 'Older', ha='center', fontsize=24, fontweight='bold')

plt.text(np.mean(bar_positions[1:4]), -0.25, 'Child Subset', ha='center', fontsize=24, style='italic')
plt.text(np.mean(bar_positions[5:9]), -0.25, 'Unbalanced', ha='center', fontsize=24, style='italic')

plt.vlines(bar_positions[9]-1.15, 0, 1.06, color='grey', linewidth=1, linestyles='dashed')
plt.vlines(bar_positions[14]-1.15, 0, 1.06, color='grey', linewidth=1, linestyles='dashed')

plt.ylabel('Average Genre Distribution', fontsize=24)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title='Genre', 
           bbox_to_anchor=(1, 1), loc='upper left', fontsize=17)


x_labels = [model for age_group, dataset, model in mean_genre_df.index.str.split(" ")]

# Setting xticks and yticks formatting
plt.xticks(bar_positions, x_labels, rotation=45, fontsize=20, ha='right')
plt.yticks(fontsize=20)

plt.xlim(bar_positions.min() - 0.7, bar_positions.max() + 0.5)

# Adjust layout to fit all elements
plt.tight_layout()

# Show the plot
plt.show()