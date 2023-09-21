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
from datetime import datetime
from gurobipy import *
from statistics import mean


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


def compute_pool_utility(players, n_a, n_b, n_c, perc, pool_type, assigned_type, compute_ub=False, in_homo_pool_while_mixed=False):

    # Sample n_a players for the pool
    if n_a > 0:
        pool_a = players[(players.level == 'Class 1') & (players[assigned_type] == 0)].sample(n_a)
    else:
        pool_a = players.sample(0)

    # Sample n_b players for the pool
    if n_b > 0:
        pool_b = players[(players.level == 'Class 2') & (players[assigned_type] == 0)].sample(n_b)
    else:
        pool_b = players.sample(0)

    # Sample n_c players for the pool
    if n_c > 0:
        pool_c = players[(players.level == 'Class 3') & (players[assigned_type] == 0)].sample(n_c)
    else:
        pool_c = players.sample(0)

    # Combine to make the pool, compute the size and the shared pot
    pool = pd.concat((pool_a, pool_b, pool_c))
    size = pool.shape[0]
    pot = perc * pool.salary.sum()

    # Determine the new earnings and utilities for each pool member
    for index, row in pool.iterrows():
        new_earnings = (1-perc) * row['salary'] + pot / size + row['minor_salary']
        players.loc[index, pool_type] = 1 - math.exp(-gamma * new_earnings)
        if in_homo_pool_while_mixed == True:
            players.loc[index, 'in_homo_pool_while_mixed'] = 1
        if compute_ub == False:
            players.loc[index, assigned_type] = 1


def make_mixed_pools(players, n_a_unassigned, n_b_unassigned,n_c_unassigned, max_size, stable_pools):

    # create list of whether or not we can make each stable pool
    remaining_stable_pools = []

    # check if we have enough people of each level
    for stable_pool in stable_pools:
        if n_a_unassigned >= stable_pool[0] and n_b_unassigned >= stable_pool[1] and n_c_unassigned >= stable_pool[2]:
            remaining_stable_pools.append(True)
        else:
            remaining_stable_pools.append(False)

    # First try to make a mixed pool, if one is possible
    if any(remaining_stable_pools):
        # combine boolean list and stable_pools list and choose one
        remaining_stable_pools = list(compress(stable_pools, remaining_stable_pools))
        pool = random.choice(remaining_stable_pools)

        compute_pool_utility(players, pool[0], pool[1], pool[2], perc, 'mixed_pool_utility', 'assigned_mixed_pool')

    # For no more stable pools possible
    # else:
    #     players.loc[players['assigned_mixed_pool']==0,'assigned_mixed_pool'] = -1

        # if no mixed pool possible, make a homogenous pool of max_size
    elif n_a_unassigned >= max_size:
        compute_pool_utility(players, max_size, 0, 0, perc, 'mixed_pool_utility', 'assigned_mixed_pool', in_homo_pool_while_mixed=True)
    elif n_b_unassigned >= max_size:
        compute_pool_utility(players, 0, max_size, 0, perc, 'mixed_pool_utility', 'assigned_mixed_pool',in_homo_pool_while_mixed=True)
    elif n_c_unassigned >= max_size:
        compute_pool_utility(players, 0, 0, max_size, perc, 'mixed_pool_utility', 'assigned_mixed_pool',in_homo_pool_while_mixed=True)

    # if no mixed pool possible or full homogenous pool, make a smaller homogenous pool
    elif n_a_unassigned < max_size and n_a_unassigned > 0:
        compute_pool_utility(players, n_a_unassigned, 0, 0, perc, 'mixed_pool_utility', 'assigned_mixed_pool',in_homo_pool_while_mixed=True)
    elif n_b_unassigned < max_size and n_b_unassigned > 0:
        compute_pool_utility(players, 0, n_b_unassigned, 0, perc, 'mixed_pool_utility', 'assigned_mixed_pool',in_homo_pool_while_mixed=True)
    elif n_c_unassigned < max_size and n_c_unassigned > 0:
        compute_pool_utility(players, 0, 0, n_c_unassigned, perc, 'mixed_pool_utility', 'assigned_mixed_pool',in_homo_pool_while_mixed=True)


def make_homo_pools(players, n_a_unassigned, n_b_unassigned,n_c_unassigned, max_size):

    # if no mixed pool possible, make a homogenous pool of max_size
    if n_a_unassigned >= max_size:
        compute_pool_utility(players, max_size, 0, 0, perc, 'homo_pool_utility', 'assigned_homo_pool')
    elif n_b_unassigned >= max_size:
        compute_pool_utility(players, 0, max_size, 0, perc, 'homo_pool_utility', 'assigned_homo_pool')
    elif n_c_unassigned >= max_size:
        compute_pool_utility(players, 0, 0, max_size, perc, 'homo_pool_utility', 'assigned_homo_pool')

    # For no more homogeneous pools of size N
    # else:
    #     players.loc[players['assigned_homo_pool']==0,'assigned_homo_pool'] = -1


    # if no mixed pool possible or full homogenous pool, make a smaller homogenous pool
    elif n_a_unassigned < max_size and n_a_unassigned > 0:
        compute_pool_utility(players, n_a_unassigned, 0, 0, perc, 'homo_pool_utility', 'assigned_homo_pool')
    elif n_b_unassigned < max_size and n_b_unassigned > 0:
        compute_pool_utility(players, 0, n_b_unassigned, 0, perc, 'homo_pool_utility', 'assigned_homo_pool')
    elif n_c_unassigned < max_size and n_c_unassigned > 0:
        compute_pool_utility(players, 0, 0, n_c_unassigned, perc, 'homo_pool_utility', 'assigned_homo_pool')


def t_mean_confidence_interval(data, alpha):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a, ddof=1)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * se / np.sqrt(n)
    return m, h


def compute_distance(P,N=3,Q=[7,0,0]):
    """Computes "distance" from P ("source" distribution) to Q ("demand" distribution)
    """
    # variables
    m = Model()
    x = m.addVars(N,N,lb=0)
    # parameter
    c = np.array([[0,1,2],
    [1,0,1],
    [2,1,0]]) # represents unit "transportation" cost
    # objective (computes total "transportation" cost)
    m.setObjective(quicksum( quicksum(c[i,j]*x[i,j] for i in range(N))
    for j in range(N)))
    # constraints
    m.addConstrs(quicksum(x[i,j] for i in range(N))==Q[j] for j in range(N))
    m.addConstrs(quicksum(x[i,j] for j in range(N))==P[i] for i in range(N))
    m.params.logtoconsole = 0
    m.optimize()
    # print(m.objVal) # get obj value

    return m.objVal


# %%

df = importData()

alphas = df[df['draft_round'].isin(np.arange(11,21))]
betas = df[df['draft_round'].isin(np.arange(21,31))]
charlies = df[df['draft_round'].isin(np.arange(31,99))]

p = np.round(df.made_it_salary.mean(), 3)
p_a = np.round(alphas.made_it_salary.mean(), 3)
p_b = np.round(betas.made_it_salary.mean(), 3)
p_c = np.round(charlies.made_it_salary.mean(), 3)

print(p, p_a, p_b, p_c)

n_a = alphas.shape[0]
n_b = betas.shape[0]
n_c = charlies.shape[0]
n = n_a + n_b + n_c
print(np.round(n_a/n,3), np.round(n_b/n,3), np.round(n_c/n,3))

s_plus = df[(df.draft_round.isin(np.arange(11,99))) & (df.salary > 0)].salary.mean()
s_plus_a = alphas[alphas.salary > 0].salary.mean()
s_plus_b = betas[betas.salary > 0].salary.mean()
s_plus_c = charlies[charlies.salary > 0].salary.mean()

print(np.round(s_plus), np.round(s_plus_a), np.round(s_plus_b), np.round(s_plus_c))

perc = 0.1

#%%
# Check for statistically signficiant same means of salary in each class
A = alphas[alphas.salary > 0].salary
B = betas[betas.salary > 0].salary
C = charlies[charlies.salary > 0].salary

t,p = stats.ttest_ind(B,C)
print(p/2 > 0.05)
print(p)
print(t)

# %%

# datetime object containing current date and time
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

numiter = 1000
n = 200
gamma = 0.10
perc = 0.10
max_size = 5

divider = 1000000
s_minus = 60000 / divider  # based on avg 6.55 years at $10K
s_plus = 14200000 / divider

p_a = 0.12
p_b = 0.09
p_c = 0.07

df = importData()
df['salary'] = df['salary'] / divider
df['minor_salary'] = 10000/divider * df['years_in_minors']
df = df[df.draft_round>=11]
conditions = [(df['draft_round'].isin(np.arange(11, 21))),
              (df['draft_round'].isin(np.arange(21, 31))),
              (df['draft_round'].isin(np.arange(31, 99)))]
# conditions = [(players['draft_round'] <= 2),
#               (players['draft_round'] >= 3) & (players['draft_round'] <= 11),
#               (players['draft_round'] >= 12) | (players['draft_round'].isna())]

values = ['Class 1', 'Class 2', 'Class 3']
df['level'] = np.select(conditions, values)

# Make theoretical players
# # Colapse
# prob_dict = {'Class 1': p_a, 'Class 2': p_b, 'Class 3': p_c, }
# df['probability'] = df['level'].copy().replace(prob_dict)
# df['made_it'] = np.random.rand(df.shape[0]) < df['probability']
# df['salary'] = df['made_it'] * s_plus
# df['minor_salary'] = (1-df['made_it']) * s_minus

sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 200]
gammas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
percents = [0.06, 0.08, 0.1, 0.12, 0.14]

sizes = [5]
gammas = [0.10]
percents = [0.1]

gamma = 0.10
perc = 0.10
max_size = 5

sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 200]
# sizes = [5]
# sizes = [ 200]


make_arrays = True
if make_arrays == True:
    # start
    solo_sw_array = []
    solo_sw_A_array = []
    solo_sw_B_array = []
    solo_sw_C_array = []

    mixed_pool_sw_array = []
    mixed_pool_sw_A_array = []
    mixed_pool_sw_B_array = []
    mixed_pool_sw_C_array = []

    homo_pool_sw_array = []
    homo_pool_sw_A_array = []
    homo_pool_sw_B_array = []
    homo_pool_sw_C_array = []

    solo_sw_hw_array = []
    solo_sw_A_hw_array = []
    solo_sw_B_hw_array = []
    solo_sw_C_hw_array = []

    mixed_pool_sw_hw_array = []
    mixed_pool_sw_A_hw_array = []
    mixed_pool_sw_B_hw_array = []
    mixed_pool_sw_C_hw_array = []

    homo_pool_sw_hw_array = []
    homo_pool_sw_A_hw_array = []
    homo_pool_sw_B_hw_array = []
    homo_pool_sw_C_hw_array = []

    homo_in_mixed_array = []
    homo_in_mixed_A_array = []
    homo_in_mixed_B_array = []
    homo_in_mixed_C_array = []

# # end

# for gamma in gammas:
for max_size in sizes:
# for perc in percents:

    # Define solo utility for particular gamma
    df['solo_utility'] = 1 - np.exp(-gamma * (df['salary'] + df['minor_salary']))

    # Load graph and then the stable pools
    filename = "G_gamma_{}_perc_{}_pa_{}.gpickle".format(gamma, perc, p_a)
    G = nx.read_gpickle(filename)

    # remove nodes greater than a size
    for a in range(0, 12+2):
        for b in range(0, 12+2):
            for c in range(0, 12+2):
                if a + b + c > max_size:
                    if (a, b, c) in G:
                        G.remove_node((a, b, c))

    # find stable pools\=
    # These are pools with size == max_size, and no out_arrows and no homoegenous pools
    stable_pools = []
    for node in G.nodes():
        if G.out_degree(node) == 0:
            if (node[0] + node[1] + node[2] == max_size) & (node[0] != max_size) & (node[1] != max_size) & (node[2] != max_size):
                stable_pools.append(node)

    print(max_size,gamma,perc, ': ', stable_pools)

    # for epsilon stable
    # stable_pools = [[n_a,n_b,n_c]]

    solo_sw = []
    solo_sw_A = []
    solo_sw_B = []
    solo_sw_C = []

    mixed_pool_sw = []
    mixed_pool_sw_A = []
    mixed_pool_sw_B = []
    mixed_pool_sw_C = []

    homo_pool_sw = []
    homo_pool_sw_A = []
    homo_pool_sw_B = []
    homo_pool_sw_C = []

    UB_pool_sw = []
    UB_pool_sw_A = []
    UB_pool_sw_B = []
    UB_pool_sw_C = []

    homo_in_mixed = []
    homo_in_mixed_A = []
    homo_in_mixed_B = []
    homo_in_mixed_C = []

    for iter in range(numiter):

        players = df.sample(n, random_state=iter, ignore_index=False)

        players = players[['salary', 'draft_round', 'made_it_salary', 'solo_utility', 'level', 'minor_salary']]

        n_a = players[players.level == 'Class 1'].shape[0]
        n_b = players[players.level == 'Class 2'].shape[0]
        n_c = players[players.level == 'Class 3'].shape[0]

        players['assigned_mixed_pool'] = 0
        players['assigned_homo_pool'] = 0
        players['mixed_pool_utility'] = 0
        players['homo_pool_utility'] = 0

        players['in_homo_pool_while_mixed'] = 0

        players['global_UB_utility'] = 0
        players['level_UB_utility'] = 0
        players['assigned_UB'] = 0

        # stable_pools = [[n_a, n_b, n_c]]

        while sum(players.assigned_mixed_pool.abs()) != n:
            n_a_unassigned = n_a - players[players.level == 'Class 1'].assigned_mixed_pool.sum()
            n_b_unassigned = n_b - players[players.level == 'Class 2'].assigned_mixed_pool.sum()
            n_c_unassigned = n_c - players[players.level == 'Class 3'].assigned_mixed_pool.sum()
            make_mixed_pools(players, n_a_unassigned, n_b_unassigned, n_c_unassigned, max_size, stable_pools)

        while sum(players.assigned_homo_pool.abs()) != n:
            n_a_unassigned = n_a - players[players.level == 'Class 1'].assigned_homo_pool.sum()
            n_b_unassigned = n_b - players[players.level == 'Class 2'].assigned_homo_pool.sum()
            n_c_unassigned = n_c - players[players.level == 'Class 3'].assigned_homo_pool.sum()
            make_homo_pools(players, n_a_unassigned, n_b_unassigned, n_c_unassigned, max_size)


        # Compute fraction of players in homo pools while mixing
        # homo_in_mixed.append(players[(players['in_homo_pool_while_mixed'] == 1)].shape[0] / n)
        # homo_in_mixed_A.append(players[(players.level == 'Class 1') & (players['in_homo_pool_while_mixed'] == 1)].shape[0]/n_a)
        # homo_in_mixed_B.append(players[(players.level == 'Class 2') & (players['in_homo_pool_while_mixed'] == 1)].shape[0]/n_b)
        # homo_in_mixed_C.append(players[(players.level == 'Class 3') & (players['in_homo_pool_while_mixed'] == 1)].shape[0]/n_c)

        # Compute upper bounds (as grand pool)
        # compute_pool_utility(players, n_a, n_b, n_c, perc, 'global_UB_utility', 'assigned_UB', compute_ub=True)
        # compute_pool_utility(players, n_a, 0, 0, perc, 'level_UB_utility', 'assigned_UB', compute_ub=True)
        # compute_pool_utility(players, 0, n_b, 0, perc, 'level_UB_utility', 'assigned_UB', compute_ub=True)
        # compute_pool_utility(players, 0, 0, n_c, perc, 'level_UB_utility', 'assigned_UB', compute_ub=True)

        # Compute lower bound (as solo pool) for homo
        solo_sw_lb_homo = players[(players.assigned_homo_pool == 1)].solo_utility.mean()
        solo_sw_A_lb_homo = players[(players.level == 'Class 1') & (players.assigned_homo_pool == 1)].solo_utility.mean()
        solo_sw_B_lb_homo = players[(players.level == 'Class 2') & (players.assigned_homo_pool == 1)].solo_utility.mean()
        solo_sw_C_lb_homo = players[(players.level == 'Class 3') & (players.assigned_homo_pool == 1)].solo_utility.mean()

        # Compute lower bound (as solo pool) for mixed
        solo_sw_lb_mixed = players[(players.assigned_mixed_pool == 1)].solo_utility.mean()
        solo_sw_A_lb_mixed = players[(players.level == 'Class 1') & (players.assigned_mixed_pool == 1)].solo_utility.mean()
        solo_sw_B_lb_mixed = players[(players.level == 'Class 2') & (players.assigned_mixed_pool == 1)].solo_utility.mean()
        solo_sw_C_lb_mixed = players[(players.level == 'Class 3') & (players.assigned_mixed_pool == 1)].solo_utility.mean()

        # Save the means for these particular players
        # solo_sw.append(100*(players.solo_utility.mean() - solo_sw_lb) / solo_sw_lb)
        # solo_sw_A.append(100*(players[players.level == 'Class 1'].solo_utility.mean() - solo_sw_A_lb) / solo_sw_A_lb)
        # solo_sw_B.append(100*(players[players.level == 'Class 2'].solo_utility.mean() - solo_sw_B_lb)/ solo_sw_B_lb)
        # solo_sw_C.append(100*(players[players.level == 'Class 3'].solo_utility.mean() - solo_sw_C_lb) / solo_sw_C_lb)

        mixed_pool_sw.append(100*(players[(players.assigned_mixed_pool==1)].mixed_pool_utility.mean() - solo_sw_lb_mixed) / solo_sw_lb_mixed)
        mixed_pool_sw_A.append(100*(players[(players.level == 'Class 1') & (players.assigned_mixed_pool==1)].mixed_pool_utility.mean() - solo_sw_A_lb_mixed) / solo_sw_A_lb_mixed)
        mixed_pool_sw_B.append(100*(players[(players.level == 'Class 2') & (players.assigned_mixed_pool==1)].mixed_pool_utility.mean() - solo_sw_B_lb_mixed) / solo_sw_B_lb_mixed)
        mixed_pool_sw_C.append(100*(players[(players.level == 'Class 3') & (players.assigned_mixed_pool==1)].mixed_pool_utility.mean() - solo_sw_C_lb_mixed) / solo_sw_C_lb_mixed)

        homo_pool_sw.append(100*(players[(players.assigned_homo_pool==1)].homo_pool_utility.mean() - solo_sw_lb_homo) / solo_sw_lb_homo)
        homo_pool_sw_A.append(100*(players[(players.level == 'Class 1')  & (players.assigned_homo_pool==1)].homo_pool_utility.mean() - solo_sw_A_lb_homo) / solo_sw_A_lb_homo)
        homo_pool_sw_B.append(100*(players[(players.level == 'Class 2')  & (players.assigned_homo_pool==1)].homo_pool_utility.mean() - solo_sw_B_lb_homo) / solo_sw_B_lb_homo)
        homo_pool_sw_C.append(100*(players[(players.level == 'Class 3')  & (players.assigned_homo_pool==1)].homo_pool_utility.mean() - solo_sw_C_lb_homo) / solo_sw_C_lb_homo)

        # # Save upperbounds just once
        # UB_pool_sw.append(100*(players.global_UB_utility.mean() - solo_sw_lb) / solo_sw_lb)
        # UB_pool_sw_A.append(100*(players[players.level == 'Class 1'].level_UB_utility.mean()  - solo_sw_A_lb) / solo_sw_A_lb)
        # UB_pool_sw_B.append(100*(players[players.level == 'Class 2'].level_UB_utility.mean()  - solo_sw_B_lb) / solo_sw_B_lb)
        # UB_pool_sw_C.append(100*(players[players.level == 'Class 3'].level_UB_utility.mean()  - solo_sw_C_lb) / solo_sw_C_lb)


    if max_size <= n:

        # Compute the means and half-widths for this pool size
        # solo_sw_m, solo_sw_h = t_mean_confidence_interval(solo_sw, 0.05)
        # solo_sw_A_m, solo_sw_A_h = t_mean_confidence_interval(solo_sw_A, 0.05)
        # solo_sw_B_m, solo_sw_B_h = t_mean_confidence_interval(solo_sw_B, 0.05)
        # solo_sw_C_m, solo_sw_C_h = t_mean_confidence_interval(solo_sw_C, 0.05)


        mixed_pool_sw_m, mixed_pool_sw_h = t_mean_confidence_interval(mixed_pool_sw, 0.05)
        mixed_pool_sw_A_m, mixed_pool_sw_A_h = t_mean_confidence_interval(mixed_pool_sw_A, 0.05)
        mixed_pool_sw_B_m, mixed_pool_sw_B_h = t_mean_confidence_interval(mixed_pool_sw_B, 0.05)
        mixed_pool_sw_C_m, mixed_pool_sw_C_h = t_mean_confidence_interval(mixed_pool_sw_C, 0.05)

        homo_pool_sw_m, homo_pool_sw_h = t_mean_confidence_interval(homo_pool_sw, 0.05)
        homo_pool_sw_A_m, homo_pool_sw_A_h = t_mean_confidence_interval(homo_pool_sw_A, 0.05)
        homo_pool_sw_B_m, homo_pool_sw_B_h = t_mean_confidence_interval(homo_pool_sw_B, 0.05)
        homo_pool_sw_C_m, homo_pool_sw_C_h = t_mean_confidence_interval(homo_pool_sw_C, 0.05)

        # Save the means and half-widths for this pool size
        # solo_sw_array.append(solo_sw_m)
        # solo_sw_A_array.append(solo_sw_A_m)
        # solo_sw_B_array.append(solo_sw_B_m)
        # solo_sw_C_array.append(solo_sw_C_m)
        # solo_sw_hw_array.append(solo_sw_h)
        # solo_sw_A_hw_array.append(solo_sw_A_h)
        # solo_sw_B_hw_array.append(solo_sw_B_h)
        # solo_sw_C_hw_array.append(solo_sw_C_h)

        mixed_pool_sw_array.append(mixed_pool_sw_m)
        mixed_pool_sw_A_array.append(mixed_pool_sw_A_m)
        mixed_pool_sw_B_array.append(mixed_pool_sw_B_m)
        mixed_pool_sw_C_array.append(mixed_pool_sw_C_m)
        mixed_pool_sw_hw_array.append(mixed_pool_sw_h)
        mixed_pool_sw_A_hw_array.append(mixed_pool_sw_A_h)
        mixed_pool_sw_B_hw_array.append(mixed_pool_sw_B_h)
        mixed_pool_sw_C_hw_array.append(mixed_pool_sw_C_h)

        homo_pool_sw_array.append(homo_pool_sw_m)
        homo_pool_sw_A_array.append(homo_pool_sw_A_m)
        homo_pool_sw_B_array.append(homo_pool_sw_B_m)
        homo_pool_sw_C_array.append(homo_pool_sw_C_m)
        homo_pool_sw_hw_array.append(homo_pool_sw_h)
        homo_pool_sw_A_hw_array.append(homo_pool_sw_A_h)
        homo_pool_sw_B_hw_array.append(homo_pool_sw_B_h)
        homo_pool_sw_C_hw_array.append(homo_pool_sw_C_h)

        # homo_in_mixed_array.append(mean(homo_in_mixed))
        # homo_in_mixed_A_array.append(mean(homo_in_mixed_A))
        # homo_in_mixed_B_array.append(mean(homo_in_mixed_B))
        # homo_in_mixed_C_array.append(mean(homo_in_mixed_C))

# mean UB
# UB_sw_m, UB_sw_h = t_mean_confidence_interval(UB_pool_sw, 0.05)
# UB_sw_A_m, UB_sw_A_h = t_mean_confidence_interval(UB_pool_sw_A, 0.05)
# UB_sw_B_m, UB_sw_B_h = t_mean_confidence_interval(UB_pool_sw_B, 0.05)
# UB_sw_C_m, UB_sw_C_h = t_mean_confidence_interval(UB_pool_sw_C, 0.05)

sizes = sizes[:-1]
print('Numerical analysis complete')

# %%

########################################################################
# Compare social welfare for a single value of MAX SIZE
########################################################################
size = 5
if len(sizes)==0:
    size = 2

print('Pool utility: {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}'.format(
    mixed_pool_sw_array[size - 2], mixed_pool_sw_hw_array[size - 2], mixed_pool_sw_A_array[size - 2],
    mixed_pool_sw_A_hw_array[size - 2], mixed_pool_sw_B_array[size - 2], mixed_pool_sw_B_hw_array[size - 2],
    mixed_pool_sw_C_array[size - 2], mixed_pool_sw_C_hw_array[size - 2]))
print('Homo utility: {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}, {:.4f} +/- {:.4f}'.format(
    homo_pool_sw_array[size - 2], homo_pool_sw_hw_array[size - 2], homo_pool_sw_A_array[size - 2],
    homo_pool_sw_A_hw_array[size - 2], homo_pool_sw_B_array[size - 2], homo_pool_sw_B_hw_array[size - 2],
    homo_pool_sw_C_array[size - 2], homo_pool_sw_C_hw_array[size - 2]))

results_df = pd.DataFrame(columns=['Pooling Type', 'Level', 'hw', 'sw'])
# results_df.loc[len(results_df)] = ['Solo Pool', 'Overall', solo_sw_hw_array[size - 2], solo_sw_array[size - 2], ]
# results_df.loc[len(results_df)] = ['Solo Pool', 'Class 1', solo_sw_A_hw_array[size - 2], solo_sw_A_array[size - 2], ]
# results_df.loc[len(results_df)] = ['Solo Pool', 'Class 2', solo_sw_B_hw_array[size - 2], solo_sw_B_array[size - 2], ]
# results_df.loc[len(results_df)] = ['Solo Pool', 'Class 3', solo_sw_C_hw_array[size - 2], solo_sw_C_array[size - 2], ]


results_df.loc[len(results_df)] = ['Mixed Pool', 'Overall', mixed_pool_sw_hw_array[size - 2],
                                   mixed_pool_sw_array[size - 2], ]
results_df.loc[len(results_df)] = ['Mixed Pool', 'Class 1', mixed_pool_sw_A_hw_array[size - 2],
                                   mixed_pool_sw_A_array[size - 2], ]
results_df.loc[len(results_df)] = ['Mixed Pool', 'Class 2', mixed_pool_sw_B_hw_array[size - 2],
                                   mixed_pool_sw_B_array[size - 2], ]
results_df.loc[len(results_df)] = ['Mixed Pool', 'Class 3', mixed_pool_sw_C_hw_array[size - 2],
                                   mixed_pool_sw_C_array[size - 2], ]

results_df.loc[len(results_df)] = ['Homogenous Pool', 'Overall', homo_pool_sw_hw_array[size - 2],
                                   homo_pool_sw_array[size - 2], ]
results_df.loc[len(results_df)] = ['Homogenous Pool', 'Class 1', homo_pool_sw_A_hw_array[size - 2],
                                   homo_pool_sw_A_array[size - 2], ]
results_df.loc[len(results_df)] = ['Homogenous Pool', 'Class 2', homo_pool_sw_B_hw_array[size - 2],
                                   homo_pool_sw_B_array[size - 2], ]
results_df.loc[len(results_df)] = ['Homogenous Pool', 'Class 3', homo_pool_sw_C_hw_array[size - 2],
                                   homo_pool_sw_C_array[size - 2], ]

fig, ax = plt.subplots(1, 1, figsize=(8.5,7))
dfp = results_df.pivot(index='Level', columns='Pooling Type', values='sw')
yerr = results_df.pivot(index='Level', columns='Pooling Type', values='hw')

dfp.plot(kind='bar', yerr=yerr, rot=0, ax=ax,width=0.75, alpha=0.8)
plt.ylabel('Relative Increase in\nMean Social Welfare (%)', labelpad=10, fontsize = 16)
plt.ylim(0,35)
plt.legend(loc = 2, prop={'size': 12})

# # Shrink current axis by 20%
# box = plt.gca().get_position()
# plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # Put a legend to the right of the current axis
# plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))# plt.ylim(1,1.8)

plt.xlabel('')
plt.gca().tick_params(axis='both', which='major', labelsize=16)


# %%

########################################################################
# Compare different values of MAX SIZE for gamma 0.10 and perc = 0.10
########################################################################

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7.25))

# ax1.errorbar(sizes, solo_sw_A_array, yerr=solo_sw_A_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax1.errorbar(sizes, mixed_pool_sw_A_array, yerr=mixed_pool_sw_A_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax1.errorbar(sizes, homo_pool_sw_A_array, yerr=homo_pool_sw_A_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax1.axhline(y = UB_sw_A_m, color = 'r', linestyle = '--', alpha=0.25)
ax1.set_title('Class 1')
ax1.set_xticks(sizes[::2])
ax1.set_xlabel('Upper Bound on Pool Size')
ax1.set_ylabel('Relative Increase in\nMean Social Welfare (%)')

# ax2.errorbar(sizes, solo_sw_B_array, yerr=solo_sw_B_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax2.errorbar(sizes, mixed_pool_sw_B_array, yerr=mixed_pool_sw_B_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax2.errorbar(sizes, homo_pool_sw_B_array, yerr=homo_pool_sw_B_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax2.axhline(y = UB_sw_B_m, color = 'r', linestyle = '--', alpha=0.25)
ax2.set_title('Class 2')
ax2.set_xticks(sizes[::2])
ax2.set_xlabel('Upper Bound on Pool Size')
ax2.set_ylabel('Relative Increase in\nMean Social Welfare (%)')


# ax3.errorbar(sizes, solo_sw_C_array, yerr=solo_sw_C_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax3.errorbar(sizes, mixed_pool_sw_C_array, yerr=mixed_pool_sw_C_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax3.errorbar(sizes, homo_pool_sw_C_array, yerr=homo_pool_sw_C_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax3.axhline(y = UB_sw_C_m, color = 'r', linestyle = '--', alpha=0.25)
ax3.set_title('Class 3')
ax3.set_xticks(sizes[::2])
ax3.set_xlabel('Upper Bound on Pool Size')
ax3.set_ylabel('Relative Increase in\nMean Social Welfare (%)')


# ax4.errorbar(sizes, solo_sw_array, yerr=solo_sw_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax4.errorbar(sizes, mixed_pool_sw_array, yerr=mixed_pool_sw_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax4.errorbar(sizes, homo_pool_sw_array, yerr=homo_pool_sw_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax4.axhline(y = UB_sw_m, color = 'r', linestyle = '--', alpha=0.25)
ax4.set_title('Overall')
ax4.set_xticks(sizes[::2])
ax4.set_xlabel('Upper Bound on Pool Size')
ax4.set_ylabel('Relative Increase in\nMean Social Welfare (%)')


low_lim = 0
high_lim = 60

ax1.set_ylim(low_lim, high_lim)
ax2.set_ylim(low_lim, high_lim)
ax3.set_ylim(low_lim, high_lim)
ax4.set_ylim(low_lim, high_lim)
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.89, wspace=0.35, hspace=0.45)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

#%%

####
# PLOT JUST SIZE AND OVERALL SW WITH gamma = 0.10 and perc = 0.10
####

fig, ax = plt.subplots(1, 1, figsize=(8.5,7))


ax.errorbar(sizes, homo_pool_sw_array[:-1], yerr=homo_pool_sw_hw_array[:-1], fmt="o", label='Homogenous Pool', alpha=0.8,
             color='#1f77b4',ms=15,)
ax.errorbar(sizes, mixed_pool_sw_array[:-1], yerr=mixed_pool_sw_hw_array[:-1], fmt="o", label='Mixed Pool', alpha=0.8,
             color='orange',ms=15,)

plt.plot(sizes, mixed_pool_sw_array[:-1], alpha=0.6,
             color='orange',)
plt.plot(sizes, homo_pool_sw_array[:-1], alpha=0.6,
             color='#1f77b4',)


# ax4.axhline(y = UB_sw_m, color = 'r', linestyle = '--', alpha=0.25)
# ax4.set_title('Overall')
ax.set_xticks(sizes[::2])
plt.xlabel('Upper Bound on Pool Size', fontsize=16, labelpad=10)
plt.ylabel('Relative Increase in\nMean Social Welfare (%)', fontsize=16, labelpad=10)
plt.legend(loc = 2, prop={'size': 12})
plt.ylim(0,35)
# plt.ylim(10,30)
plt.gca().tick_params(axis='both', which='major', labelsize=16)

#%%

# PLOT MAX SIZE AND DIVERSITY OF STABLE POOLS

##############
# Compute distance of stable pools
###############

sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

gamma = 0.1
max_size = 5
perc = 0.10
p_a = 0.12

array = [0.1]

df_distance_array = pd.DataFrame(np.zeros((len(array),8)), index=array, columns=['Class 1', 'Class 2', 'Class 3', 'Overall',
                                                                                 'Class 1 Num', 'Class 2 Num', 'Class 3 Num', 'Overall Num'])
for value in array:
    # Load graph and then the stable pools
    gamma = value
    filename = "G_gamma_{}_perc_{}_pa_{}.gpickle".format(gamma, perc, p_a)
    G = nx.read_gpickle(filename)

    # remove nodes greater than a size
    for a in range(0, 12+2):
        for b in range(0, 12+2):
            for c in range(0, 12+2):
                if a + b + c > max_size:
                    if (a, b, c) in G:
                        G.remove_node((a, b, c))

    # find stable pools\=
    # These are pools with size == max_size, and no out_arrows and no homoegenous pools
    stable_pools = []
    for node in G.nodes():
        if G.out_degree(node) == 0:
            if (node[0] + node[1] + node[2] == max_size) & (node[0] != max_size) & (node[1] != max_size) & (node[2] != max_size):
                stable_pools.append(node)

    df_distance = pd.DataFrame(np.zeros((4,3)), index=['Class 1', 'Class 2', 'Class 3', 'Overall'], columns=['homogeneous', 'mixed', 'num_pools'])

    for P in stable_pools:
        distance = compute_distance(P,N=3,Q=[max_size,0,0])
        if P[0] > 0:
            df_distance.loc['Class 1', 'num_pools'] += 1
            df_distance.loc['Class 1', 'mixed'] += distance
        if P[1] > 0:
            df_distance.loc['Class 2', 'num_pools'] += 1
            df_distance.loc['Class 2', 'mixed'] += distance
        if P[2] > 0:
            df_distance.loc['Class 3', 'num_pools'] += 1
            df_distance.loc['Class 3', 'mixed'] += distance
        df_distance.loc['Overall', 'num_pools'] += 1
        df_distance.loc['Overall', 'mixed'] += distance

    df_distance.loc['Class 1', 'mixed'] /= df_distance.loc['Class 1', 'num_pools']
    df_distance.loc['Class 2', 'mixed'] /= df_distance.loc['Class 2', 'num_pools']
    df_distance.loc['Class 3', 'mixed'] /= df_distance.loc['Class 3', 'num_pools']
    df_distance.loc['Overall', 'mixed'] /= df_distance.loc['Overall', 'num_pools']

    df_distance.loc['Class 1', 'homogeneous'] = compute_distance([max_size,0,0],N=3,Q=[max_size,0,0])
    df_distance.loc['Class 2', 'homogeneous'] = compute_distance([0,max_size,0],N=3,Q=[max_size,0,0])
    df_distance.loc['Class 3', 'homogeneous'] = compute_distance([0,0,max_size],N=3,Q=[max_size,0,0])

    df_distance[['homogeneous', 'mixed']] /= compute_distance([0,0,max_size],N=3,Q=[max_size,0,0])

    df_distance_array.loc[value,'Class 1'] = df_distance.loc['Class 1', 'mixed']
    df_distance_array.loc[value,'Class 2'] = df_distance.loc['Class 2', 'mixed']
    df_distance_array.loc[value,'Class 3'] = df_distance.loc['Class 3', 'mixed']
    df_distance_array.loc[value,'Overall'] = df_distance.loc['Overall', 'mixed']
    df_distance_array.loc[value,'Class 1 Num'] = df_distance.loc['Class 1', 'num_pools']
    df_distance_array.loc[value,'Class 2 Num'] = df_distance.loc['Class 2', 'num_pools']
    df_distance_array.loc[value,'Class 3 Num'] = df_distance.loc['Class 3', 'num_pools']
    df_distance_array.loc[value,'Overall Num'] = df_distance.loc['Overall', 'num_pools']

    # Just for gammas
    # df_distance_array.loc[0.01,'Class 1'] = compute_distance([max_size,0,0],N=3,Q=[max_size,0,0])/compute_distance([0,0,max_size],N=3,Q=[max_size,0,0])
    # df_distance_array.loc[0.01,'Class 2'] = compute_distance([0,max_size,0],N=3,Q=[max_size,0,0])/compute_distance([0,0,max_size],N=3,Q=[max_size,0,0])
    # df_distance_array.loc[0.01,'Class 3'] = compute_distance([0,0,max_size],N=3,Q=[max_size,0,0])/compute_distance([0,0,max_size],N=3,Q=[max_size,0,0])
    # df_distance_array.loc[0.01,'Overall'] = compute_distance([0,max_size,0],N=3,Q=[max_size,0,0])/compute_distance([0,0,max_size],N=3,Q=[max_size,0,0])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7.25))

ax1.scatter(array, df_distance_array.loc[:,'Class 1'],  label='Mixed Pool', alpha=0.6,
             color='orange')
ax1.set_title('Class 1')
ax1.set_xticks(array[::2])
# ax1.set_xlabel('Upper Bound on Pool Size')
ax1.set_xlabel('Risk Aversion Parameter')
ax1.set_ylabel('Average Normalized \'Distance\'\nto Stacked Pool')

ax2.scatter(array, df_distance_array.loc[:,'Class 2'],  label='Mixed Pool', alpha=0.6,
             color='orange')
ax2.set_title('Class 2')
ax2.set_xticks(array[::2])
# ax2.set_xlabel('Upper Bound on Pool Size')
ax2.set_xlabel('Risk Aversion Parameter')
ax2.set_ylabel('Average Normalized \'Distance\'\nto Stacked Pool')

ax3.scatter(array, df_distance_array.loc[:,'Class 3'],  label='Mixed Pool', alpha=0.6,
             color='orange')
ax3.set_title('Class 3')
ax3.set_xticks(array[::2])
# ax3.set_xlabel('Upper Bound on Pool Size')
ax3.set_xlabel('Risk Aversion Parameter')
ax3.set_ylabel('Average Normalized \'Distance\'\nto Stacked Pool')

ax4.scatter(array, df_distance_array.loc[:,'Overall'],  label='Mixed Pool', alpha=0.6,
             color='orange')
ax4.set_title('Overall')
ax4.set_xticks(array[::2])
# ax4.set_xlabel('Upper Bound on Pool Size')
ax4.set_xlabel('Risk Aversion Parameter')
ax4.set_ylabel('Mean Normalized \'Distance\'\nto Stacked Pool')

low_lim = -0.1
high_lim = 1.1

ax1.set_ylim(low_lim, high_lim)
ax2.set_ylim(low_lim, high_lim)
ax3.set_ylim(low_lim, high_lim)
ax4.set_ylim(low_lim, high_lim)

plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.89, wspace=0.35, hspace=0.45)


#%%

# PLOT MAX SIZE AND NUMBER OF HOMOGENEOUS POOLS IN MIXED POOL OPTION

array = percents

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7.25))

ax1.scatter(array, np.ones(len(array))-homo_in_mixed_A_array,  label='Mixed Pool', alpha=0.6,
             color='orange')
ax1.set_title('Class 1')
ax1.set_xticks(array[::1])
# ax1.set_xlabel('Risk Aversion Parameter')
ax1.set_xlabel('Contribution Percentage')
ax1.set_ylabel('Average Fraction of Agents in Mixed Pools')

ax2.scatter(array, np.ones(len(array))- homo_in_mixed_B_array,  label='Mixed Pool', alpha=0.6,
             color='orange')
ax2.set_title('Class 2')
ax2.set_xticks(array[::1])
# ax2.set_xlabel('Risk Aversion Parameter')
ax2.set_xlabel('Contribution Percentage')
ax2.set_ylabel('Average Fraction of Agents in Mixed Pools')

ax3.scatter(array, np.ones(len(array))-homo_in_mixed_C_array,  label='Mixed Pool', alpha=0.6,
             color='orange')
ax3.set_title('Class 3')
ax3.set_xticks(array[::1])
# ax3.set_xlabel('Risk Aversion Parameter')
ax3.set_xlabel('Contribution Percentage')
ax3.set_ylabel('Average Fraction of Agents in Mixed Pools')

ax4.scatter(array, np.ones(len(array))-homo_in_mixed_array,  label='Mixed Pool', alpha=0.6,
             color='orange')
ax4.set_title('Overall')
ax4.set_xticks(array[::1])
# ax4.set_xlabel('Risk Aversion Parameter')
ax4.set_xlabel('Contribution Percentage')
ax4.set_ylabel('Average Fraction of Agents in Mixed Pools')

plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.89, wspace=0.35, hspace=0.45)

low_lim = 0.1
high_lim = 1.1

ax1.set_ylim(low_lim, high_lim)
ax2.set_ylim(low_lim, high_lim)
ax3.set_ylim(low_lim, high_lim)
ax4.set_ylim(low_lim, high_lim)

# %%

########################################################################
# Compare different values of GAMMA for max size 7 and perc = 0.10
########################################################################

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7.25))
# ax1.errorbar(sizes, solo_sw_A_array, yerr=solo_sw_A_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax1.errorbar(gammas, mixed_pool_sw_A_array, yerr=mixed_pool_sw_A_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax1.errorbar(gammas, homo_pool_sw_A_array, yerr=homo_pool_sw_A_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax1.axhline(y = UB_sw_A_m, color = 'r', linestyle = '--', alpha=0.25)
ax1.set_title('Class 1')
ax1.set_xticks(gammas[::2])
ax1.set_xlabel('Risk Aversion Parameter')
ax1.set_ylabel('Relative Increase in\nMean Social Welfare (%)')

# ax2.errorbar(gammas, solo_sw_B_array, yerr=solo_sw_B_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax2.errorbar(gammas, mixed_pool_sw_B_array, yerr=mixed_pool_sw_B_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax2.errorbar(gammas, homo_pool_sw_B_array, yerr=homo_pool_sw_B_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax2.axhline(y = UB_sw_B_m, color = 'r', linestyle = '--', alpha=0.25)
ax2.set_title('Class 2')
ax2.set_xticks(gammas[::2])
ax2.set_xlabel('Risk Aversion Parameter')
ax2.set_ylabel('Relative Increase in\nMean Social Welfare (%)')

# ax2.set_ylabel('Mean Social Welfare')


# ax3.errorbar(gammas, solo_sw_C_array, yerr=solo_sw_C_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax3.errorbar(gammas, mixed_pool_sw_C_array, yerr=mixed_pool_sw_C_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax3.errorbar(gammas, homo_pool_sw_C_array, yerr=homo_pool_sw_C_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax3.axhline(y = UB_sw_C_m, color = 'r', linestyle = '--', alpha=0.25)
ax3.set_title('Class 3')
ax3.set_xticks(gammas[::2])
ax3.set_xlabel('Risk Aversion Parameter')
ax3.set_ylabel('Relative Increase in\nMean Social Welfare (%)')


# ax4.errorbar(gammas, solo_sw_array, yerr=solo_sw_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax4.errorbar(gammas, mixed_pool_sw_array, yerr=mixed_pool_sw_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax4.errorbar(gammas, homo_pool_sw_array, yerr=homo_pool_sw_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax4.axhline(y = UB_sw_m, color = 'r', linestyle = '--', alpha=0.25)
ax4.set_title('Overall')
ax4.set_xticks(gammas[::2])
ax4.set_xlabel('Risk Aversion Parameter $\gamma$')
ax4.set_ylabel('Relative Increase in\nMean Social Welfare (%)')


low_lim = 0
high_lim = 60

ax1.set_ylim(low_lim, high_lim)
ax2.set_ylim(low_lim, high_lim)
ax3.set_ylim(low_lim, high_lim)
ax4.set_ylim(low_lim, high_lim)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.89, wspace=0.35, hspace=0.45)


# %%

########################################################################
# Compare different values of PERCENT for max size 7 and gamma = 0.35
########################################################################

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7.25))
# ax1.errorbar(sizes, solo_sw_A_array, yerr=solo_sw_A_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax1.errorbar(percents, mixed_pool_sw_A_array, yerr=mixed_pool_sw_A_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax1.errorbar(percents, homo_pool_sw_A_array, yerr=homo_pool_sw_A_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax1.axhline(y = UB_sw_A_m, color = 'r', linestyle = '--', alpha=0.25)
ax1.set_title('Class 1')
ax1.set_xticks(percents[::1])
ax1.set_xlabel('Contribution Percent')
ax1.set_ylabel('Relative Increase in\nMean Social Welfare (%)')

# ax2.errorbar(gammas, solo_sw_B_array, yerr=solo_sw_B_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax2.errorbar(percents, mixed_pool_sw_B_array, yerr=mixed_pool_sw_B_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax2.errorbar(percents, homo_pool_sw_B_array, yerr=homo_pool_sw_B_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax2.axhline(y = UB_sw_B_m, color = 'r', linestyle = '--', alpha=0.25)
ax2.set_title('Class 2')
ax2.set_xticks(percents[::1])
ax2.set_xlabel('Contribution Percent')
ax2.set_ylabel('Relative Increase in\nMean Social Welfare (%)')


# ax3.errorbar(gammas, solo_sw_C_array, yerr=solo_sw_C_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax3.errorbar(percents, mixed_pool_sw_C_array, yerr=mixed_pool_sw_C_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax3.errorbar(percents, homo_pool_sw_C_array, yerr=homo_pool_sw_C_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax3.axhline(y = UB_sw_C_m, color = 'r', linestyle = '--', alpha=0.25)
ax3.set_title('Class 3')
ax3.set_xticks(percents[::1])
ax3.set_xlabel('Contribution Percent')
ax3.set_ylabel('Relative Increase in\nMean Social Welfare (%)')


# ax4.errorbar(gammas, solo_sw_array, yerr=solo_sw_hw_array, fmt="o", label='Solo Pool', alpha=0.6, color='green')
ax4.errorbar(percents, mixed_pool_sw_array, yerr=mixed_pool_sw_hw_array, fmt="o", label='Mixed Pool', alpha=0.6,
             color='orange')
ax4.errorbar(percents, homo_pool_sw_array, yerr=homo_pool_sw_hw_array, fmt="o", label='Homogenous Pool', alpha=0.6,
             color='#1f77b4')
# ax4.axhline(y = UB_sw_m, color = 'r', linestyle = '--', alpha=0.25)
ax4.set_title('Overall')
ax4.set_xticks(percents[::1])
ax4.set_xlabel('Contribution Percent')
ax4.set_ylabel('Relative Increase in\nMean Social Welfare (%)')

low_lim = 0
high_lim = 80

ax1.set_ylim(low_lim, high_lim)
ax2.set_ylim(low_lim, high_lim)
ax3.set_ylim(low_lim, high_lim)
ax4.set_ylim(low_lim, high_lim)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.89, wspace=0.35, hspace=0.45)

#%%

######################
# Compute number of stable pools for gamma and perc
#####################

stable_df = pd.DataFrame(np.zeros((len(percents), len(gammas))), index=percents, columns=gammas)

for perc in percents:
    for gamma in gammas:

        # Define solo utility for particular gamma
        df['solo_utility'] = 1 - np.exp(-gamma * (df['salary'] + df['minor_salary']))

        # Load graph and then the stable pools
        filename = "G_gamma_{}_perc_{}_pa_{}.gpickle".format(gamma, perc, p_a)
        G = nx.read_gpickle(filename)

        # remove nodes greater than a size
        for a in range(0, 12+2):
            for b in range(0, 12+2):
                for c in range(0, 12+2):
                    if a + b + c > max_size:
                        if (a, b, c) in G:
                            G.remove_node((a, b, c))

        # find stable pools\=
        # These are pools with size == max_size, and no out_arrows and no homoegenous pools
        stable_pools = []
        for node in G.nodes():
            if G.out_degree(node) == 0:
                if (node[0] + node[1] + node[2] == max_size) & (node[0] != max_size) & (node[1] != max_size) & (node[2] != max_size):
                    stable_pools.append(node)

        stable_df.loc[perc, gamma] = len(stable_pools)

plt.figure()
sns.heatmap(stable_df, cmap='RdYlGn', annot=True, linecolor='black',linewidths=0.5)
plt.xlabel('Risk Aversion')
plt.ylabel('Contribution Percent')

#%%

######################
# Compute number of stable pools for size
#####################

stable_df = pd.DataFrame(np.zeros((len(sizes), 1)), index=sizes)

for max_size in sizes:


    # Define solo utility for particular gamma
    df['solo_utility'] = 1 - np.exp(-gamma * (df['salary'] + df['minor_salary']))

    # Load graph and then the stable pools
    filename = "G_gamma_{}_perc_{}_pa_{}.gpickle".format(gamma, perc, p_a)
    G = nx.read_gpickle(filename)

    # remove nodes greater than a size
    for a in range(0, 12+2):
        for b in range(0, 12+2):
            for c in range(0, 12+2):
                if a + b + c > max_size:
                    if (a, b, c) in G:
                        G.remove_node((a, b, c))

    # find stable pools\=
    # These are pools with size == max_size, and no out_arrows and no homoegenous pools
    stable_pools = []
    for node in G.nodes():
        if G.out_degree(node) == 0:
            if (node[0] + node[1] + node[2] == max_size) & (node[0] != max_size) & (node[1] != max_size) & (node[2] != max_size):
                stable_pools.append(node)

    stable_df.loc[max_size,0] = len(stable_pools)

plt.figure()
plt.plot(sizes, stable_df[0])
plt.xlabel('Max Size')
plt.ylabel('Number of Stable Pools')


# %%

# EDA Compare n, p and s^+ (no contribution) for each draft round

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



