import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import requests, bs4
import os
import lxml
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from fitter import Fitter, get_common_distributions, get_distributions
import math
import scipy.special
import pickle

def findClientIPool(client_i, splitted):
    for i, val in enumerate(splitted):
        if client_i in val:
            return (i)


def theoretical_earnings(p, mu, s):
    earnings = 0
    for i in range(1, s):
        earnings += i * scipy.special.comb((s - 1), i) * (p ** i) * ((1 - p) ** (s - 1 - i))
    earnings = earnings * mu / s
    return earnings

# %%

df = pd.read_csv(Path.cwd() / 'data' / 'df_players.csv', index_col=0)
df['major_urls'] = df['major_urls'].astype(str)

df_minor = df.loc[
    df.salary >= 0].copy()  # First remove any players that had negative salary meaning they earned before 2000
df_minor['contribution'] = np.where((df_minor['salary'] > 1600000), (df_minor['salary'] - 1600000) * 0.1,
                                    0)  # determine that player's contribution
df_minor['contribution'] = np.where((df_minor['contribution'] > 20000000), 20000000,
                                    df_minor['contribution'])  # cap the contribution at 20 mill
df_minor['made_it_salary'] = np.where((df_minor['salary'] > 0), 1,
                                      0)  # Determine those who made it based on those who earned a salary
df_minor['made_it_appearance'] = np.where(df_minor['major_urls'] == 'nan', 0,
                                          1)  # Determine those who made it based on a major_url
df_minor['made_it_contribution'] = np.where((df_minor['contribution'] > 0), 1,
                                            0)  # Determine those who made it based on those who earned a salary

df_major = df_minor.loc[df_minor.made_it_salary == 1].copy()

# %%

## Create simulation

random_seeds = np.arange(1,5)

sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18,
         20, 25, 30, 35, 40, 45, 50, 60, 80, 100]

sizes = np.arange(2, 20 + 1)

num_scen = 100
num_clients = 100
client_size_K_neg = 20
client_size_K_pos = 5

size_average_rand_dict = {}

for seed in random_seeds:
    print("seed = {}".format(seed))
    np.random.seed(seed)

    # Sample from the df, just K players
    df_minor_sample = df_minor.sample(num_clients).reset_index()
    client_list = np.arange(num_clients)

    # Determine the clients we're analyzing, split into K- and K+
    df_minor_sample_made_it = df_minor_sample[df_minor_sample.made_it_contribution == 1]
    df_minor_sample_didnt_make_it = df_minor_sample[df_minor_sample.made_it_contribution == 0]
    # client_list_made_it = np.random.choice(df_minor_sample_made_it.index, size=client_size_K_pos, replace=False)
    # client_list_didnt_make_it = np.random.choice(df_minor_sample_didnt_make_it.index, size=client_size_K_neg, replace=False)
    client_list_made_it = df_minor_sample_made_it.index
    client_list_didnt_make_it = df_minor_sample_didnt_make_it.index

    # Create the array to save the average results per size
    # size_average = pd.Series(index=sizes, dtype=float)
    # Create the array to save the average results per size
    size_average = pd.DataFrame(index=sizes, columns=['didnt_make_it', 'made_it', 'both'], dtype=float)

    for type in ['didnt_make_it', 'made_it']:

        # Pull the right client list (either from K- or K+)
        if type == 'made_it':
            client_list = client_list_made_it
        else:
            client_list = client_list_didnt_make_it

        for size in sizes:
            print("size = {}".format(size))

            # Create array for the average earnings per client
            client_average = pd.Series(index=client_list, dtype=float)

            for client_i in client_list:  # for all clients, range(num_clients)

                # Create the array for the average earnings over the scenarios
                scenario_average = np.zeros(num_scen)
                for scen in range(num_scen):
                    # create player indices of everyone in K and shuffle them
                    indices = np.arange(num_clients)
                    np.random.shuffle(indices)
                    # create the split points for the group size and create the groups
                    cuts = np.arange(size, num_clients, size)
                    splitted = [list(x) for x in np.split(indices, cuts)]
                    # create a mini df that contains client i and get their average earnings
                    i = findClientIPool(client_i, splitted)

                    # if the pools aren't even sized and the client is in the smaller group, ignore this
                    if (num_clients % size != 0) and (i == (len(splitted) - 1)):
                        scenario_average[scen] = np.nan
                    else:
                        # average minor salary (10,000*6.55) + clients major salary - clients contribution + the shared contribution from the entire pool
                        # earnings = 65500 + df_minor_sample.loc[client_i].salary - df_minor_sample.loc[client_i].contribution + ((df_minor_sample.loc[splitted[i]]['contribution'].sum()) / size)
                        # Below line is for threshold = 0, so don't use contribution column, just 10% of salary
                        earnings = 65500 + df_minor_sample.loc[client_i].salary - 0.1*df_minor_sample.loc[client_i].salary + (sum(0.1*df_minor_sample.loc[splitted[i]]['salary']) / size)
                        earnings = np.where(earnings > 0.0000000001, earnings, 0.0001)
                        scenario_average[scen] = np.sqrt(earnings)

                # Average per client
                client_average[client_i] = np.nanmean(scenario_average)
            # Average per size
            size_average.loc[size, type] = np.nanmean(client_average)

    size_average['both'] = size_average['didnt_make_it']*len(client_list_didnt_make_it)/num_clients + size_average['made_it']*len(client_list_made_it)/num_clients

    size_average_rand_dict[seed] = size_average

#%%

# Plot the results
plt.figure()
plt.plot(sizes, size_average['made_it'], marker='x', color='red', label='Made it')
plt.plot(sizes, size_average['didnt_make_it'], marker='x', color='blue', label='Didnt Make it')
plt.plot(sizes, size_average['both'], marker='x', color='green', label='Both')
plt.title('Relationship between Sqrt Earnings and Pool Size')
plt.legend()
plt.xlabel('Pool Size')
plt.ylabel('Average Log Earnings')

#%%

# plt.figure()
earnings_df = pd.concat([v for k,v in size_average_rand_dict.items()])['didnt_make_it']
sns.lineplot(x=np.array(earnings_df.index), y=earnings_df.values, ci=95, label='Simulated (95% CI)')