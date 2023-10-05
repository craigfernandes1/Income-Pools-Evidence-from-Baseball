import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import binom
from matplotlib.offsetbox import AnchoredText
from poibin import PoiBin
import networkx as nx
from scipy import stats
import math
from scipy.signal import savgol_filter
from itertools import compress
import random


# %%

def importData():
    df = pd.read_csv(Path.cwd() / 'data' / 'df_players.csv', index_col=0)
    # df = pd.read_csv(Path.cwd() / 'df_players.csv', index_col=0)
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
    df_minor['draft_round'] = df_minor['draft_round'].replace('np.nan', np.nan)
    df_minor['draft_round'] = pd.to_numeric(df_minor['draft_round'])
    df_minor = df_minor.reset_index()
    df_minor = df_minor.drop(['index'], axis=1)
    return df_minor


def compute_pool_utility(players, n_a, n_b, c, pool_type, assigned_type,compute_ub=False):
    if n_a > 0 and n_b > 0:
        pool_a = players[(players.level == 'alpha') & (players[assigned_type] == 0)].sample(n_a)
        pool_b = players[(players.level == 'beta') & (players[assigned_type] == 0)].sample(n_b)
        pool = pd.concat((pool_a, pool_b))
    elif n_a > 0 and n_b == 0:
        pool = players[(players.level == 'alpha') & (players[assigned_type] == 0)].sample(n_a)
    elif n_a == 0 and n_b > 0:
        pool = players[(players.level == 'beta') & (players[assigned_type] == 0)].sample(n_b)

    size = pool.shape[0]

    pot = c * pool.salary.sum()

    for index, row in pool.iterrows():
        new_earnings = 0.9 * row['salary'] + pot / size
        players.loc[index, pool_type] = 1 - math.exp(-gamma * new_earnings)
        if compute_ub==False:
            players.loc[index, assigned_type] = 1

def make_mixed_pools_gamma_0_1(players, n_a_unassigned, n_b_unassigned, k):

    if n_a_unassigned >= 1 and n_b_unassigned >= k - 1:
        compute_pool_utility(players, 1, k - 1, c, 'mixed_pool_utility', 'assigned_mixed_pool')

    elif n_a_unassigned >= k:
        compute_pool_utility(players, k, 0, c, 'mixed_pool_utility', 'assigned_mixed_pool')

    elif n_b_unassigned >= k:
        compute_pool_utility(players, 0, k, c, 'mixed_pool_utility', 'assigned_mixed_pool')

    elif n_a_unassigned < k and n_a_unassigned > 0:
        compute_pool_utility(players, n_a_unassigned, 0, c, 'mixed_pool_utility', 'assigned_mixed_pool')

    elif n_b_unassigned < k and n_b_unassigned > 0:
        compute_pool_utility(players, 0, n_b_unassigned, c, 'mixed_pool_utility', 'assigned_mixed_pool')


def make_mixed_pools_gamma_1(players, n_a_unassigned, n_b_unassigned, k):

    stable_pools = [[1,k-1], [2,k-2]]
    remaining_stable_pools = [(n_a_unassigned >= 1 and n_b_unassigned >= k - 1), (n_a_unassigned >= 2 and n_b_unassigned >= k - 2)]

    if any(remaining_stable_pools):

        remaining_stable_pools = list(compress(stable_pools, remaining_stable_pools))

        pool = random.choice(remaining_stable_pools)

        compute_pool_utility(players, pool[0], pool[1], c, 'mixed_pool_utility', 'assigned_mixed_pool')

    elif n_a_unassigned >= k:
        compute_pool_utility(players, k, 0, c, 'mixed_pool_utility', 'assigned_mixed_pool')

    elif n_b_unassigned >= k:
        compute_pool_utility(players, 0, k, c, 'mixed_pool_utility', 'assigned_mixed_pool')

    elif n_a_unassigned < k and n_a_unassigned > 0:
        compute_pool_utility(players, n_a_unassigned, 0, c, 'mixed_pool_utility', 'assigned_mixed_pool')

    elif n_b_unassigned < k and n_b_unassigned > 0:
        compute_pool_utility(players, 0, n_b_unassigned, c, 'mixed_pool_utility', 'assigned_mixed_pool')


def make_mixed_pools(players, n_a_unassigned, n_b_unassigned, k, gamma):

    if gamma == 0.1:
        make_mixed_pools_gamma_0_1(players, n_a_unassigned, n_b_unassigned, k)

    elif gamma == 1:
        make_mixed_pools_gamma_1(players, n_a_unassigned, n_b_unassigned, k)

def make_homo_pools(players, n_a_unassigned, n_b_unassigned, k):
    if n_a_unassigned >= k:
        compute_pool_utility(players, k, 0, c, 'homo_pool_utility', 'assigned_homo_pool')

    elif n_b_unassigned >= k:
        compute_pool_utility(players, 0, k, c, 'homo_pool_utility', 'assigned_homo_pool')

    elif n_a_unassigned < k and n_a_unassigned > 0:
        compute_pool_utility(players, n_a_unassigned, 0, c, 'homo_pool_utility', 'assigned_homo_pool')

    elif n_b_unassigned < k and n_b_unassigned > 0:
        compute_pool_utility(players, 0, n_b_unassigned, c, 'homo_pool_utility', 'assigned_homo_pool')


def t_mean_confidence_interval(data, alpha):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a, ddof=1)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se / np.sqrt(n)
    return m, h


# %%

df = importData()

# alphas = df[df['draft_round'].isin(np.arange(1, 6))]
# betas = df[~df['draft_round'].isin(np.arange(1, 6))]
# p_a = np.round(alphas.made_it_salary.mean(), 3)
# p_b = np.round(betas.made_it_salary.mean(), 3)

alphas = df[df['draft_round'].isin(np.arange(1, 6))]
betas = df[df['draft_round'].isin(np.arange(6, 12))]
charlies = df[~df['draft_round'].isin(np.arange(1, 12))]

p_a = np.round(alphas.made_it_salary.mean(), 3)
p_b = np.round(betas.made_it_salary.mean(), 3)
p_c = np.round(charlies.made_it_salary.mean(), 3)

gamma = 1
c = 0.1
scale = 1000000
s_plus = df[df.salary > 0].salary.mean()

df['salary'] = df['salary'] / scale
df['solo_utility'] = 1 - np.exp(-gamma * (df['salary']))


# %%

numiter = 30
n = 100

# sizes = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
sizes = [2,3,4,5,6,7]
# sizes = [6]


solo_sw_sizes = []
solo_sw_A_sizes = []
solo_sw_B_sizes = []

mixed_pool_sw_sizes = []
mixed_pool_sw_A_sizes = []
mixed_pool_sw_B_sizes = []

homo_pool_sw_sizes = []
homo_pool_sw_A_sizes = []
homo_pool_sw_B_sizes = []

solo_sw_hw_sizes = []
solo_sw_A_hw_sizes = []
solo_sw_B_hw_sizes = []

mixed_pool_sw_hw_sizes = []
mixed_pool_sw_A_hw_sizes = []
mixed_pool_sw_B_hw_sizes = []

homo_pool_sw_hw_sizes = []
homo_pool_sw_A_hw_sizes = []
homo_pool_sw_B_hw_sizes = []

for k in sizes:

    solo_sw = []
    solo_sw_A = []
    solo_sw_B = []

    mixed_pool_sw = []
    mixed_pool_sw_A = []
    mixed_pool_sw_B = []

    homo_pool_sw = []
    homo_pool_sw_A = []
    homo_pool_sw_B = []

    for iter in range(numiter):


        players = df.sample(n, random_state=iter, ignore_index=False)
        conditions = [(players['draft_round'] <= 5),
                      (players['draft_round'] > 5) | (players['draft_round'].isna())]
        values = ['alpha', 'beta']

        players['level'] = np.select(conditions, values)
        players = players[['salary', 'draft_round', 'made_it_salary', 'solo_utility', 'level']]

        n_a = players[players.level == 'alpha'].shape[0]
        n_b = players[players.level == 'beta'].shape[0]

        players['assigned_mixed_pool'] = 0
        players['assigned_homo_pool'] = 0
        players['mixed_pool_utility'] = 0
        players['homo_pool_utility'] = 0

        players['global_UB_utility'] = 0
        players['level_UB_utility'] = 0
        players['assigned_UB'] = 0

        while players.assigned_mixed_pool.sum() != n:
            n_a_unassigned = n_a - players[players.level == 'alpha'].assigned_mixed_pool.sum()
            n_b_unassigned = n_b - players[players.level == 'beta'].assigned_mixed_pool.sum()
            make_mixed_pools(players, n_a_unassigned, n_b_unassigned,k, gamma)

        while players.assigned_homo_pool.sum() != n:
            n_a_unassigned = n_a - players[players.level == 'alpha'].assigned_homo_pool.sum()
            n_b_unassigned = n_b - players[players.level == 'beta'].assigned_homo_pool.sum()
            make_homo_pools(players, n_a_unassigned, n_b_unassigned,k)

        # Compute upper bounds (as grand pool)
        compute_pool_utility(players, n_a, n_b, c, 'global_UB_utility', 'assigned_UB', compute_ub=True)
        compute_pool_utility(players, n_a, 0, c, 'level_UB_utility', 'assigned_UB', compute_ub=True)
        compute_pool_utility(players, 0, n_b, c, 'level_UB_utility', 'assigned_UB', compute_ub=True)

        # Compute lower bound (as solo pool)
        solo_sw_lb = players.solo_utility.mean()
        solo_sw_A_lb = players[players.level == 'alpha'].solo_utility.mean()
        solo_sw_B_lb = players[players.level == 'beta'].solo_utility.mean()

        # solo_sw_lb = 1
        # solo_sw_A_lb = 1
        # solo_sw_B_lb = 1

        # Save the means for these particular players
        solo_sw.append(players.solo_utility.mean()/solo_sw_lb)
        solo_sw_A.append(players[players.level == 'alpha'].solo_utility.mean()/solo_sw_A_lb)
        solo_sw_B.append(players[players.level == 'beta'].solo_utility.mean()/solo_sw_B_lb)

        mixed_pool_sw.append(players.mixed_pool_utility.mean()/solo_sw_lb)
        mixed_pool_sw_A.append(players[players.level == 'alpha'].mixed_pool_utility.mean()/solo_sw_A_lb)
        mixed_pool_sw_B.append(players[players.level == 'beta'].mixed_pool_utility.mean()/solo_sw_B_lb)

        homo_pool_sw.append(players.homo_pool_utility.mean()/solo_sw_lb)
        homo_pool_sw_A.append(players[players.level == 'alpha'].homo_pool_utility.mean()/solo_sw_A_lb)
        homo_pool_sw_B.append(players[players.level == 'beta'].homo_pool_utility.mean()/solo_sw_B_lb)

    # Compute the means and half-widths for this pool size
    solo_sw_m, solo_sw_h = t_mean_confidence_interval(solo_sw, 0.05)
    solo_sw_A_m, solo_sw_A_h = t_mean_confidence_interval(solo_sw_A, 0.05)
    solo_sw_B_m, solo_sw_B_h = t_mean_confidence_interval(solo_sw_B, 0.05)

    mixed_pool_sw_m, mixed_pool_sw_h = t_mean_confidence_interval(mixed_pool_sw, 0.05)
    mixed_pool_sw_A_m, mixed_pool_sw_A_h = t_mean_confidence_interval(mixed_pool_sw_A, 0.05)
    mixed_pool_sw_B_m, mixed_pool_sw_B_h = t_mean_confidence_interval(mixed_pool_sw_B, 0.05)

    homo_pool_sw_m, homo_pool_sw_h = t_mean_confidence_interval(homo_pool_sw, 0.05)
    homo_pool_sw_A_m, homo_pool_sw_A_h = t_mean_confidence_interval(homo_pool_sw_A, 0.05)
    homo_pool_sw_B_m, homo_pool_sw_B_h = t_mean_confidence_interval(homo_pool_sw_B, 0.05)

    # Save the means and half-widths for this pool size
    solo_sw_sizes.append(solo_sw_m)
    solo_sw_A_sizes.append(solo_sw_A_m)
    solo_sw_B_sizes.append(solo_sw_B_m)
    solo_sw_hw_sizes.append(solo_sw_h)
    solo_sw_A_hw_sizes.append(solo_sw_A_h)
    solo_sw_B_hw_sizes.append(solo_sw_B_h)

    mixed_pool_sw_sizes.append(mixed_pool_sw_m)
    mixed_pool_sw_A_sizes.append(mixed_pool_sw_A_m)
    mixed_pool_sw_B_sizes.append(mixed_pool_sw_B_m)
    mixed_pool_sw_hw_sizes.append(mixed_pool_sw_h)
    mixed_pool_sw_A_hw_sizes.append(mixed_pool_sw_A_h)
    mixed_pool_sw_B_hw_sizes.append(mixed_pool_sw_B_h)

    homo_pool_sw_sizes.append(homo_pool_sw_m)
    homo_pool_sw_A_sizes.append(homo_pool_sw_A_m)
    homo_pool_sw_B_sizes.append(homo_pool_sw_B_m)
    homo_pool_sw_hw_sizes.append(homo_pool_sw_h)
    homo_pool_sw_A_hw_sizes.append(homo_pool_sw_A_h)
    homo_pool_sw_B_hw_sizes.append(homo_pool_sw_B_h)

print('Numerical analysis complete')

#%%

size = 6

print('Solo utility: {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}'.format(
    solo_sw_sizes[size-2], solo_sw_hw_sizes[size-2],solo_sw_A_sizes[size-2], solo_sw_A_hw_sizes[size-2],solo_sw_B_sizes[size-2], solo_sw_B_hw_sizes[size-2]))
print('Pool utility: {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}'.format(
    mixed_pool_sw_sizes[size-2], mixed_pool_sw_hw_sizes[size-2],mixed_pool_sw_A_sizes[size-2], mixed_pool_sw_A_hw_sizes[size-2],mixed_pool_sw_B_sizes[size-2], mixed_pool_sw_B_hw_sizes[size-2]))
print('Homo utility: {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}'.format(
    homo_pool_sw_sizes[size-2], homo_pool_sw_hw_sizes[size-2],homo_pool_sw_A_sizes[size-2], homo_pool_sw_A_hw_sizes[size-2],homo_pool_sw_B_sizes[size-2], homo_pool_sw_B_hw_sizes[size-2]))

results_df = pd.DataFrame(columns=['Pooling Type', 'Level', 'hw', 'sw'])
results_df.loc[len(results_df)] = ['Solo Pool', 'Overall', solo_sw_hw_sizes[size-2], solo_sw_sizes[size-2],]
results_df.loc[len(results_df)] = ['Solo Pool', 'Alpha', solo_sw_A_hw_sizes[size-2], solo_sw_A_sizes[size-2],]
results_df.loc[len(results_df)] = ['Solo Pool', 'Beta', solo_sw_B_hw_sizes[size-2], solo_sw_B_sizes[size-2],]

results_df.loc[len(results_df)] = ['Mixed Pool', 'Overall', mixed_pool_sw_hw_sizes[size-2], mixed_pool_sw_sizes[size-2],]
results_df.loc[len(results_df)] = ['Mixed Pool', 'Alpha', mixed_pool_sw_A_hw_sizes[size-2], mixed_pool_sw_A_sizes[size-2],]
results_df.loc[len(results_df)] = ['Mixed Pool', 'Beta', mixed_pool_sw_B_hw_sizes[size-2], mixed_pool_sw_B_sizes[size-2],]

results_df.loc[len(results_df)] = ['Homogenous Pool', 'Overall', homo_pool_sw_hw_sizes[size-2], homo_pool_sw_sizes[size-2],]
results_df.loc[len(results_df)] = ['Homogenous Pool', 'Alpha', homo_pool_sw_A_hw_sizes[size-2], homo_pool_sw_A_sizes[size-2],]
results_df.loc[len(results_df)] = ['Homogenous Pool', 'Beta', homo_pool_sw_B_hw_sizes[size-2], homo_pool_sw_B_sizes[size-2],]

dfp = results_df.pivot(index='Level', columns='Pooling Type', values='sw')
yerr = results_df.pivot(index='Level', columns='Pooling Type', values='hw')

dfp.plot(kind='bar', yerr=yerr, rot=0)
plt.ylabel('Mean Social Welfare')

#%%

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11,8))
ax1.errorbar(sizes, solo_sw_A_sizes, yerr=solo_sw_A_hw_sizes, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax1.errorbar(sizes, mixed_pool_sw_A_sizes, yerr=mixed_pool_sw_A_hw_sizes, fmt="o", label='Mixed Pool', alpha=0.6, color='orange')
ax1.errorbar(sizes, homo_pool_sw_A_sizes, yerr=homo_pool_sw_A_hw_sizes, fmt="o", label='Homogenous Pool', alpha=0.6, color='#1f77b4')
ax1.set_title('Alpha')
ax1.set_xticks(sizes[::2])
ax1.set_xlabel('Max Pool Size')
ax1.set_ylabel('Mean Social Welfare')


ax2.errorbar(sizes, solo_sw_B_sizes, yerr=solo_sw_B_hw_sizes, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax2.errorbar(sizes, mixed_pool_sw_B_sizes, yerr=mixed_pool_sw_B_hw_sizes, fmt="o", label='Mixed Pool', alpha=0.6, color='orange')
ax2.errorbar(sizes, homo_pool_sw_B_sizes, yerr=homo_pool_sw_B_hw_sizes, fmt="o", label='Homogenous Pool', alpha=0.6, color='#1f77b4')
ax2.set_title('Beta')
ax2.set_xticks(sizes[::2])
ax2.set_xlabel('Max Pool Size')
# ax2.set_ylabel('Mean Social Welfare')


ax3.errorbar(sizes, solo_sw_sizes, yerr=solo_sw_hw_sizes, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax3.errorbar(sizes, mixed_pool_sw_sizes, yerr=mixed_pool_sw_hw_sizes, fmt="o", label='Mixed Pool', alpha=0.6, color='orange')
ax3.errorbar(sizes, homo_pool_sw_sizes, yerr=homo_pool_sw_hw_sizes, fmt="o", label='Homogenous Pool', alpha=0.6, color='#1f77b4')
ax3.set_title('Overall')
ax3.set_xticks(sizes[::2])
ax3.set_xlabel('Max Pool Size')
# ax3.set_ylabel('Mean Social Welfare')

# ax1.set_ylim(0.9, 3)
# ax2.set_ylim(0.9, 3)
# ax3.set_ylim(0.9, 3)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')



# %%


# %%

# Compare n, p and s^+ (no contribution) for each draft round
s = []
s_std = []
p = []
n = []
rounds = np.sort(df['draft_round'].unique())
rounds = np.arange(1, 31)

for i in rounds:
    s.append(df[((df['draft_round'] == i)) & (df['made_it_salary'] == 1)]['salary'].mean())
    s_std.append(df[((df['draft_round'] == i)) & (df['made_it_salary'] == 1)]['salary'].std())
    p.append(df[(df['draft_round'] == i)]['made_it_salary'].mean())
    n.append(df[(df['draft_round'] == i)].shape[0])

s.append(df[((df['draft_round'] > 30)) & (df['made_it_salary'] == 1)]['salary'].mean())
s_std.append(df[((df['draft_round'] > 30)) & (df['made_it_salary'] == 1)]['salary'].std())
p.append(df[(df['draft_round'] > 30)]['made_it_salary'].mean())
n.append(df[(df['draft_round'] > 30)].shape[0])

s.append(df[((df['draft_round'].isna())) & (df['made_it_salary'] == 1)]['salary'].mean())
s_std.append(df[((df['draft_round'].isna())) & (df['made_it_salary'] == 1)]['salary'].std())
p.append(df[(df['draft_round'].isna())]['made_it_salary'].mean())
n.append(df[(df['draft_round'].isna())].shape[0])

# rounds[-1] = 100
# s[-1] = df[((df['draft_round'].isna())) & (df['made_it_salary'] == 1)]['salary'].mean()
# p[-1] = df[(df['draft_round'].isna())]['made_it_salary'].mean()
# n[-1] = df[(df['draft_round'].isna())].shape[0]

rounds = np.arange(1, 33)
xt = np.arange(1, 29, 5)
xt = np.append(xt, [30, 31, 32])

plt.figure()
plt.subplot(1, 2, 2)
ax = plt.gca()
plt.plot(rounds, s, marker="o", label='mean')
plt.plot(rounds, s_std, marker="o", label='std')
plt.legend()
plt.title("MLB Salary vs Draft Round")
plt.ylabel('MLB Salary', fontsize=14)
plt.xlabel('Draft Round', fontsize=14)
plt.xticks(xt)
a = ax.get_xticks().tolist()
a[-2] = '31+'
a[-1] = 'Undrafted'
ax.set_xticklabels(a, rotation=45)
plt.gca().xaxis.grid(linestyle='--')
plt.gca().yaxis.grid(linestyle='--')

plt.subplot(1, 2, 1)
# plt.figure()
ax = plt.gca()
plt.gca().xaxis.grid(linestyle='--')
plt.gca().yaxis.grid(linestyle='--')

plt.plot(rounds, p, color="red", marker="o", label='probability')
p_smoothed = savgol_filter(p, 11, 3)
plt.plot(rounds, p_smoothed, color="darkred", marker="o", alpha=0.3, label='smoothed probability')
plt.legend()
plt.ylim(0, 0.5)
plt.title("Probability of Success vs Draft Round")
ax = plt.gca()
ax.set_ylabel("Probability", color="black", fontsize=14)
plt.xlabel('Draft Round', fontsize=14)
# ax2=ax.twinx()
# ax2.plot(rounds, n,color="blue",marker="o", label='draft size')
# plt.ylim(0,1600)
# plt.yticks(np.arange(0, 1600, 100), )
# plt.gca().yaxis.grid(linestyle='--')
# ax2.set_ylabel("Size",color="blue",fontsize=14)
plt.xticks(xt)
a = ax.get_xticks().tolist()
a[-2] = '31+'
a[-1] = 'Undrafted'
ax.set_xticklabels(a, rotation=45)
plt.legend()

#%%

alphas = df[df['draft_round'].isin(np.arange(1, 6))]
betas = df[df['draft_round'].isin(np.arange(6, 12))]
charlies = df[~df['draft_round'].isin(np.arange(1, 12))]

p_a = np.round(alphas.made_it_salary.mean(), 3)
p_b = np.round(betas.made_it_salary.mean(), 3)
p_c = np.round(charlies.made_it_salary.mean(), 3)

print(p_a, p_b, p_c)