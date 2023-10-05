import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import requests, bs4
import os
# import lxml
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
# from fitter import Fitter, get_common_distributions, get_distributions
import math
import scipy.special
import pickle
from scipy.stats import binom
from matplotlib.offsetbox import AnchoredText
from scipy.signal import savgol_filter
from gurobipy import *

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

df_minor = df.loc[df.salary >= 0].copy()  # First remove any players that had negative salary meaning they earned before 2000
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

#%%

# Compare n, p and s^+ (no contribution) for each draft round

s = []
s_std = []
p = []
n = []
rounds = np.sort(df_minor['draft_round'].unique())
rounds = np.arange(1,31)

for i in rounds:
    s.append(df_minor[((df_minor['draft_round'] == i)) & (df_minor['made_it_salary'] == 1)]['salary'].mean())
    s_std.append(df_minor[((df_minor['draft_round'] == i)) & (df_minor['made_it_salary'] == 1)]['salary'].std())
    p.append(df_minor[(df_minor['draft_round'] == i)]['made_it_salary'].mean())
    n.append(df_minor[(df_minor['draft_round'] == i)].shape[0])


s.append(df_minor[((df_minor['draft_round'] > 30)) & (df_minor['made_it_salary'] == 1)]['salary'].mean())
s_std.append(df_minor[((df_minor['draft_round'] > 30)) & (df_minor['made_it_salary'] == 1)]['salary'].std())
p.append(df_minor[(df_minor['draft_round'] > 30)]['made_it_salary'].mean())
n.append(df_minor[(df_minor['draft_round'] > 30)].shape[0])

s.append(df_minor[((df_minor['draft_round'].isna())) & (df_minor['made_it_salary'] == 1)]['salary'].mean())
s_std.append(df_minor[((df_minor['draft_round'].isna())) & (df_minor['made_it_salary'] == 1)]['salary'].std())
p.append(df_minor[(df_minor['draft_round'].isna())]['made_it_salary'].mean())
n.append(df_minor[(df_minor['draft_round'].isna())].shape[0])


# rounds[-1] = 100
# s[-1] = df_minor[((df_minor['draft_round'].isna())) & (df_minor['made_it_salary'] == 1)]['salary'].mean()
# p[-1] = df_minor[(df_minor['draft_round'].isna())]['made_it_salary'].mean()
# n[-1] = df_minor[(df_minor['draft_round'].isna())].shape[0]

rounds = np.arange(1,33)
xt = np.arange(1,29, 5)
xt = np.append(xt, [30, 31,32])

plt.figure()
plt.subplot(1,2,2)
ax = plt.gca()
plt.plot(rounds, s,  marker="o", label='mean')
plt.plot(rounds, s_std,  marker="o", label='std')
plt.legend()
plt.title("MLB Salary vs Draft Round")
plt.ylabel('MLB Salary',fontsize=14)
plt.xlabel('Draft Round',fontsize=14)
plt.xticks(xt)
a=ax.get_xticks().tolist()
a[-2]='31+'
a[-1]='Undrafted'
ax.set_xticklabels(a, rotation=45)
plt.gca().xaxis.grid(linestyle='--')
plt.gca().yaxis.grid(linestyle='--')

plt.subplot(1,2,1)
# plt.figure()
ax = plt.gca()
plt.gca().xaxis.grid(linestyle='--')
plt.gca().yaxis.grid(linestyle='--')

plt.plot(rounds, p, color="red", marker="o", label='probability')
p_smoothed = savgol_filter(p, 11, 3)
plt.plot(rounds, p_smoothed, color="darkred", marker="o",alpha=0.3, label='smoothed probability' )
plt.legend()
plt.ylim(0,0.5)
plt.title("Probability of Success vs Draft Round")
ax = plt.gca()
ax.set_ylabel("Probability",color="black",fontsize=14)
plt.xlabel('Draft Round',fontsize=14)
# ax2=ax.twinx()
# ax2.plot(rounds, n,color="blue",marker="o", label='draft size')
# plt.ylim(0,1600)
# plt.yticks(np.arange(0, 1600, 100), )
# plt.gca().yaxis.grid(linestyle='--')
# ax2.set_ylabel("Size",color="blue",fontsize=14)
plt.xticks(xt)
a=ax.get_xticks().tolist()
a[-2]='31+'
a[-1]='Undrafted'
ax.set_xticklabels(a, rotation=45)
plt.legend()


#%%
## Probability of contributing given you make it to the majors
pp = []
for i in range(1,30):
    # print(df_minor[(df_minor['draft_round'] == i) & (df_minor['made_it_salary'] == 1) & (df_minor['made_it_contribution'] == 1)]['salary'].mean())
    pp.append(df_minor[(df_minor['draft_round'] == i) & (df_minor['made_it_salary'] == 1)]['made_it_contribution'].mean())

pp = np.array(pp)
print(pp.mean())
print(pp.std())

#%%

# Salary distrbution

plt.figure()
# plt.hist(df_minor[df_minor['made_it_salary']==1]['salary'], bins=100)
# plt.hist(df_minor[(df_minor['made_it_salary']==1) & (df_minor['made_it_contribution']==1)]['salary'], bins=100)
plt.hist(df_minor[(df_minor['made_it_salary']==1) & (df_minor['made_it_contribution']==0)]['salary'], bins=100)
# plt.hist(df_minor[(df_minor['made_it_salary']==1) & (df_minor['made_it_contribution']==1)]['contribution'], bins=100)

print('minor = {}\nmajor = {}\nmajor thresh = {}\noverall salary = {}'.format(65500,
                                                         df_minor[((df_minor['draft_round'] == 1) )&(df_minor['made_it_salary']==1) & (df_minor['made_it_contribution']==0)]['salary'].mean(),
                                                         df_minor[((df_minor['draft_round'] == 1) )&(df_minor['made_it_salary']==1) & (df_minor['made_it_contribution']==1)]['salary'].mean(),
                                                         df_minor[((df_minor['draft_round'] == 1) )&(df_minor['made_it_salary']==1)]['salary'].mean()))


#%%

# Draft round analysis

df_minor['draft_round'] = df_minor['draft_round'].replace('np.nan', np.nan)
df_minor['draft_round'] = pd.to_numeric(df_minor['draft_round'])
n = 5095
p = df_minor.made_it_salary.mean()
mu = df_minor[df_minor['made_it_salary'] == 1]['salary'].mean()
mu_std = df_minor[df_minor['made_it_salary'] == 1]['salary'].std()
p_sal_no_contr = df_minor[(df_minor['made_it_contribution'] == 0) & (df_minor['made_it_salary'] == 1)].shape[0]/n
print('[Overall]', n, p, mu, mu_std)

for round in [[1,2], [3,4,5,6,7,8,9,10,11]]:
    n = df_minor[df_minor['draft_round'].isin(round)].shape[0]
    p = df_minor[df_minor['draft_round'].isin(round)].made_it_salary.mean()
    mu = df_minor[(df_minor['draft_round'].isin(round)) & (df_minor['made_it_salary'] == 1)]['salary'].mean()
    mu_std = df_minor[(df_minor['draft_round'].isin(round)) & (df_minor['made_it_salary'] == 1)]['salary'].std()
    p_sal_no_contr = df_minor[(df_minor['draft_round'].isin(round)) & (df_minor['made_it_contribution'] == 0) & (df_minor['made_it_salary'] == 1)].shape[0] / n
    print(round, n, p, mu, mu_std )


n = df_minor[~df_minor.draft_round.isin(np.arange(1,12))].shape[0]
p = df_minor[~df_minor.draft_round.isin(np.arange(1,12))].made_it_salary.mean()
mu = df_minor[(~df_minor.draft_round.isin(np.arange(1,12))) & (df_minor['made_it_salary'] == 1)]['salary'].mean()
mu_std = df_minor[(~df_minor.draft_round.isin(np.arange(1,12))) & (df_minor['made_it_salary'] == 1)]['salary'].std()
p_sal_no_contr = df_minor[(~df_minor.draft_round.isin(np.arange(1,12))) & (df_minor['made_it_contribution'] == 0) & (df_minor['made_it_salary'] == 1)].shape[0] / n

print('[11+]', n, p, mu, mu_std)

#%%





















