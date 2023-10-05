import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import requests, bs4
import time
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from fitter import Fitter, get_common_distributions, get_distributions
import math
import scipy.special
import pickle
from scipy.stats import binom
from matplotlib.offsetbox import AnchoredText
from poibin import PoiBin
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from datetime import datetime

# %%

# Define functions
def expon_util(x, gamma):
    return (1 - math.exp(-gamma * x))

def sqrt_util(x, gamma):
    return math.sqrt(x)

def compute_S1(a, b, c):
    k = a + b + c - 1

    S1_A_util = 0
    S1_B_util = 0
    S1_C_util = 0

    if a > 0:
        # E[U(alpha stay)]
        S1_A_util = 0
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                for m in range(0, c + 1):
                    S1_A_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(alpha leave)]
        S1_A_util -= p_A * (1 - math.exp(-gamma * s_plus)) + (1 - p_A) * (1 - math.exp(-gamma * s_minus))

    if b > 0:
        # E[U(beta stay)]
        S1_B_util = 0
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    S1_B_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c, p_C)* (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(beta leave)]
        S1_B_util -= p_B * (1 - math.exp(-gamma * s_plus)) + (1 - p_B) * (1 - math.exp(-gamma * s_minus))

    if c > 0:
        # E[U(charlie stay)]
        S1_C_util = 0
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                for m in range(0, c - 1 + 1):
                    S1_C_util += binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c-1, p_C)* (p_C * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                (1 - p_C) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(charlie leave)]
        S1_C_util -= p_C * (1 - math.exp(-gamma * s_plus)) + (1 - p_C) * (1 - math.exp(-gamma * s_minus))

    return S1_A_util >= 0, S1_B_util >= 0, S1_C_util >= 0


def compute_S2(a, b, c):
    S2_A_keep_A_util = -1
    S2_B_keep_A_util = -1
    S2_C_keep_A_util = -1
    S2_A_keep_B_util = -1
    S2_B_keep_B_util = -1
    S2_C_keep_B_util = -1
    S2_A_keep_C_util = -1
    S2_B_keep_C_util = -1
    S2_C_keep_C_util = -1

    k = a + b + c - 1

    if a > 0:
        if a != 1:
            S2_A_keep_A_util = 0
            # E[U(alpha keep an alpha)]
            for i in range(0, a - 1 + 1):
                for j in range(0, b + 1):
                    for m in range(0, c + 1):
                        S2_A_keep_A_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B)  * binom.pmf(m, c, p_C) * (p_A * (
                                1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                               (1 - p_A) * (1 - math.exp(
                                    -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

            # E[U(alpha remove an alpha)]
            for i in range(0, a - 2 + 1):
                for j in range(0, b + 1):
                    for m in range(0, c + 1):
                        S2_A_keep_A_util -= binom.pmf(i, a - 2, p_A) * binom.pmf(j, b, p_B)  * binom.pmf(m, c, p_C) * (p_A * (
                                1 - math.exp(-gamma * s_plus) * math.exp(
                            -gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_A) * (1 - math.exp(
                                    -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))

    if a > 0 and b > 0:
        # E[U(beta keep an alpha)]
        S2_B_keep_A_util = 0
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    S2_B_keep_A_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c, p_C) * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                           (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(beta remove an alpha)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    S2_B_keep_A_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c, p_C)  * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))


    if a > 0 and c > 0:
        # E[U(charlie keep an alpha)]
        S2_C_keep_A_util = 0
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                for m in range(0, c - 1 + 1):
                    S2_C_keep_A_util += binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c - 1, p_C) * (p_C * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                           (1 - p_C) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(charlie remove an alpha)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                for m in range(0, c - 1 + 1):
                    S2_C_keep_A_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c - 1, p_C)  * (p_C * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_C) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))


    if a > 0:
        # Check if both players prefer to keep the alpha (i.e., stable)
        if (S2_A_keep_A_util >= 0) or (S2_B_keep_A_util >= 0) or (S2_C_keep_A_util >= 0):
            S2_A = True
        else:
            S2_A = False

    ###########################################

    if a > 0 and b > 0:
        # E[U(alpha keep a beta)]
        S2_A_keep_B_util = 0
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                for m in range(0, c + 1):
                    S2_A_keep_B_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                           (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(alpha remove a beta)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    S2_A_keep_B_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c, p_C) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))


    if c > 0 and b > 0:
        # E[U(charlie keep a beta)]
        S2_C_keep_B_util = 0
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                for m in range(0, c - 1 + 1):
                    S2_C_keep_B_util += binom.pmf(i, a , p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c - 1, p_C) * (p_C * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                           (1 - p_C) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(charlie remove a beta)]
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c - 1 + 1):
                    S2_C_keep_B_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c - 1, p_C) * (p_C * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_C) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))


    if b > 0:
        if b != 1:

            S2_B_keep_B_util = 0
            # E[U(beta keep a beta)]
            for i in range(0, a + 1):
                for j in range(0, b - 1 + 1):
                    for m in range(0, c + 1):
                        S2_B_keep_B_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c, p_C) * (p_B * (
                                1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                               (1 - p_B) * (1 - math.exp(
                                    -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

            # E[U(beta remove a beta)]
            for i in range(0, a + 1):
                for j in range(0, b - 2 + 1):
                    for m in range(0, c + 1):
                        S2_B_keep_B_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b - 2, p_B) * binom.pmf(m, c, p_C) * (p_B * (
                                1 - math.exp(-gamma * s_plus) * math.exp(
                            -gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_B) * (1 - math.exp(
                                    -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))



    # Check if both players prefer to keep the beta (i.e., stable)
    if b > 0:
        if (S2_A_keep_B_util >= 0) or (S2_B_keep_B_util >= 0) or (S2_C_keep_B_util >= 0):
            S2_B = True
        else:
            S2_B = False




    ###########################################

    if a > 0 and c > 0:
        # E[U(alpha keep a charlie)]
        S2_A_keep_C_util = 0
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                for m in range(0, c + 1):
                    S2_A_keep_C_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                           (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(alpha remove a charlie)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                for m in range(0, c - 1 + 1):
                    S2_A_keep_C_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c - 1, p_C) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))


    if c > 0 and b > 0:
        # E[U(beta keep a charlie)]
        S2_B_keep_C_util = 0
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    S2_B_keep_C_util += binom.pmf(i, a , p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c, p_C) * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                           (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(beta remove a charlie)]
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c - 1 + 1):
                    S2_B_keep_C_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c - 1, p_C) * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))

    if c > 0:
        if c != 1:

            S2_C_keep_C_util = 0
            # E[U(charlie keep a charlie)]
            for i in range(0, a + 1):
                for j in range(0, b + 1):
                    for m in range(0, c - 1 + 1):
                        S2_C_keep_C_util += binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c - 1, p_C) * (p_C * (
                                1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                               (1 - p_C) * (1 - math.exp(
                                    -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

            # E[U(charlie remove a charlie)]
            for i in range(0, a + 1):
                for j in range(0, b + 1):
                    for m in range(0, c - 2 + 1):
                        S2_C_keep_C_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b , p_B) * binom.pmf(m, c - 2, p_C) * (p_C * (
                                1 - math.exp(-gamma * s_plus) * math.exp(
                            -gamma * perc * s_plus * (i + j + m - k + 1) / (a + b + c - 1))) +
                                                                                               (1 - p_C) * (1 - math.exp(
                                    -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c - 1))))

    # Check if both players prefer to keep the charlie (i.e., stable)
    if c > 0:
        if (S2_A_keep_C_util >= 0) or (S2_B_keep_C_util >= 0) or (S2_C_keep_C_util >= 0):
            S2_C = True
        else:
            S2_C = False


    ###################################################################################

    if a == 0:
        S2_A = True
    if b == 0:
        S2_B = True
    if c == 0:
        S2_C = True

    return S2_A, S2_B, S2_C


def compute_G1(a, b, c):
    G1_A_util = 0
    G1_B_util = 0
    G1_C_util = 0
    k = a + b + c - 1

    # E[U(alpha do not join)]
    G1_A_util += p_A * (1 - math.exp(-gamma * s_plus)) + (1 - p_A) * (1 - math.exp(-gamma * s_minus))

    # E[U(alpha join)]
    for i in range(0, a + 1):
        for j in range(0, b + 1):
            for m in range(0, c + 1):
                G1_A_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                            (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c + 1))))


    # E[U(beta do not join)]
    G1_B_util += p_B * (1 - math.exp(-gamma * s_plus)) + (1 - p_B) * (1 - math.exp(-gamma * s_minus))

    # E[U(beta join)]
    for i in range(0, a + 1):
        for j in range(0, b + 1):
            for m in range(0, c + 1):
                G1_B_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C)  * (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                            (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c + 1))))


    # E[U(charlie do not join)]
    G1_C_util += p_C * (1 - math.exp(-gamma * s_plus)) + (1 - p_C) * (1 - math.exp(-gamma * s_minus))

    # E[U(charlie join)]
    for i in range(0, a + 1):
        for j in range(0, b + 1):
            for m in range(0, c + 1):
                G1_C_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C) * (p_C * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                            (1 - p_C) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c + 1))))

    return G1_A_util >= 0, G1_B_util >= 0, G1_C_util >= 0


def compute_G2(a, b, c):
    G2_A_dont_allow_A_util = -1
    G2_B_dont_allow_A_util = -1
    G2_C_dont_allow_A_util = -1
    G2_A_dont_allow_B_util = -1
    G2_B_dont_allow_B_util = -1
    G2_C_dont_allow_B_util = -1
    G2_A_dont_allow_C_util = -1
    G2_B_dont_allow_C_util = -1
    G2_C_dont_allow_C_util = -1

    k = a + b + c - 1

    if a > 0:
        G2_A_dont_allow_A_util = 0
        # E[U(alpha do not allow an alpha)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                for m in range(0, c + 1):
                    G2_A_dont_allow_A_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B)* binom.pmf(m, c, p_C) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                                 (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(alpha allow an alpha)]
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                for m in range(0, c + 1):
                    G2_A_dont_allow_A_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                                             (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c + 1))))

    if b > 0:

        G2_B_dont_allow_A_util = 0
        # E[U(beta do not allow an alpha)]
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    G2_B_dont_allow_A_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B)* binom.pmf(m, c, p_C) * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                                 (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(beta allow an alpha)]
        for i in range(0, a + 1 + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    G2_B_dont_allow_A_util -= binom.pmf(i, a + 1, p_A) * binom.pmf(j, b - 1, p_B)* binom.pmf(m, c, p_C) * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                                                     (1 - p_B) * (
                                                                                                                 1 - math.exp(
                                                                                                             -gamma * s_minus) * math.exp(
                                                                                                             -gamma * perc * s_plus * (
                                                                                                                         i + j + m) / (
                                                                                                                         a + b + c + 1))))

    if c > 0:

        G2_C_dont_allow_A_util = 0
        # E[U(charlie do not allow an alpha)]
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                for m in range(0, c - 1 + 1):
                    G2_C_dont_allow_A_util += binom.pmf(i, a, p_A) * binom.pmf(j, b , p_B)* binom.pmf(m, c -1, p_C) * (p_C * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                                 (1 - p_C) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(charlie allow an alpha)]
        for i in range(0, a + 1 + 1):
            for j in range(0, b  + 1):
                for m in range(0, c - 1 + 1):
                    G2_C_dont_allow_A_util -= binom.pmf(i, a + 1, p_A) * binom.pmf(j, b, p_B)* binom.pmf(m, c - 1, p_C) * (p_C * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                                                     (1 - p_C) * (
                                                                                                                 1 - math.exp(
                                                                                                             -gamma * s_minus) * math.exp(
                                                                                                             -gamma * perc * s_plus * (
                                                                                                                         i + j + m) / (
                                                                                                                         a + b + c + 1))))

    # Check if both players prefer to not allow the alpha (i.e., stable)
    if (G2_A_dont_allow_A_util >= 0) or (G2_B_dont_allow_A_util >= 0) or (G2_C_dont_allow_A_util >= 0):
        G2_A = True
    else:
        G2_A = False

    #############################################

    if a > 0:

        G2_A_dont_allow_B_util = 0
        # E[U(alpha do not allow an beta)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                for m in range(0, c + 1):
                    G2_A_dont_allow_B_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                                 (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(alpha allow an beta)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1 + 1):
                for m in range(0, c + 1):
                    G2_A_dont_allow_B_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b + 1, p_B) * binom.pmf(m, c, p_C) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                                                     (1 - p_A) * (
                                                                                                                 1 - math.exp(
                                                                                                             -gamma * s_minus) * math.exp(
                                                                                                             -gamma * perc * s_plus * (
                                                                                                                         i + j + m) / (
                                                                                                                         a + b + c + 1))))

    # E[U(beta do not allow an beta)] - E[U(beta allow an beta)]
    if b > 0:
        G2_B_dont_allow_B_util = 0
        # E[U(beta do not allow an beta)]
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    G2_B_dont_allow_B_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B)* binom.pmf(m, c, p_C) * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                                 (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(beta allow an beta)]
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                for m in range(0, c + 1):
                    G2_B_dont_allow_B_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C)  * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                                             (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c + 1))))

    if c > 0:

        G2_C_dont_allow_B_util = 0
        # E[U(charlie do not allow an beta)]
        for i in range(0, a  + 1):
            for j in range(0, b + 1):
                for m in range(0, c - 1 + 1):
                    G2_C_dont_allow_B_util += binom.pmf(i, a , p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c - 1, p_C) * (p_C * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                                                                                 (1 - p_C) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(charlie allow an beta)]
        for i in range(0, a + 1):
            for j in range(0, b + 1 + 1):
                for m in range(0, c - 1 + 1):
                    G2_C_dont_allow_B_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b + 1, p_B) * binom.pmf(m, c - 1, p_C) * (p_C * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                                                                     (1 - p_C) * (
                                                                                                                 1 - math.exp(
                                                                                                             -gamma * s_minus) * math.exp(
                                                                                                             -gamma * perc * s_plus * (
                                                                                                                         i + j + m) / (
                                                                                                                         a + b + c + 1))))

    # Check if both players prefer to not allow the beta (i.e., stable)
    if (G2_A_dont_allow_B_util >= 0) or (G2_B_dont_allow_B_util >= 0) or (G2_C_dont_allow_B_util >= 0):
        G2_B = True
    else:
        G2_B = False

    #############################################

    if a > 0:

        G2_A_dont_allow_C_util = 0
        # E[U(alpha do not allow an charlie)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                for m in range(0, c + 1):
                    G2_A_dont_allow_C_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c, p_C) * (
                                p_A * (
                                1 - math.exp(-gamma * s_plus) * math.exp(
                            -gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(alpha allow an charlie)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b  + 1):
                for m in range(0, c + 1 + 1):
                    G2_A_dont_allow_C_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b , p_B) * binom.pmf(m, c + 1,
                                                                                                              p_C) * (
                                                          p_A * (
                                                          1 - math.exp(-gamma * s_plus) * math.exp(
                                                      -gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                          (1 - p_A) * (
                                                                  1 - math.exp(
                                                              -gamma * s_minus) * math.exp(
                                                              -gamma * perc * s_plus * (
                                                                      i + j + m) / (
                                                                      a + b + c + 1))))

    if b > 0:
        G2_B_dont_allow_C_util = 0
        # E[U(beta do not allow a charlie)]
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1):
                    G2_B_dont_allow_C_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c, p_C) * (
                                p_B * (
                                1 - math.exp(-gamma * s_plus) * math.exp(
                            -gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(beta allow a charlie)]
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                for m in range(0, c + 1 + 1):
                    G2_B_dont_allow_C_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * binom.pmf(m, c + 1, p_C) * (
                                p_B * (
                                1 - math.exp(-gamma * s_plus) * math.exp(
                            -gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c + 1))))

    if c > 0:

        G2_C_dont_allow_C_util = 0
        # E[U(charlie do not allow an charlie)]
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                for m in range(0, c - 1 + 1):
                    G2_C_dont_allow_C_util += binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c - 1, p_C) * (
                                p_C * (
                                1 - math.exp(-gamma * s_plus) * math.exp(
                            -gamma * perc * s_plus * (i + j + m - k) / (a + b + c))) +
                                (1 - p_C) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j + m) / (a + b + c))))

        # E[U(charlie allow an charlie)]
        for i in range(0, a + 1):
            for j in range(0, b  + 1):
                for m in range(0, c  + 1):
                    G2_C_dont_allow_C_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * binom.pmf(m, c,
                                                                                                          p_C) * (
                                                          p_C * (
                                                          1 - math.exp(-gamma * s_plus) * math.exp(
                                                      -gamma * perc * s_plus * (i + j + m - k - 1) / (a + b + c + 1))) +
                                                          (1 - p_C) * (
                                                                  1 - math.exp(
                                                              -gamma * s_minus) * math.exp(
                                                              -gamma * perc * s_plus * (
                                                                      i + j + m) / (
                                                                      a + b + c + 1))))

    # Check if both players prefer to not allow the beta (i.e., stable)
    if (G2_A_dont_allow_C_util >= 0) or (G2_B_dont_allow_C_util >= 0) or (G2_C_dont_allow_C_util >= 0):
        G2_C = True
    else:
        G2_C = False


    return G2_A, G2_B, G2_C


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

# %%

# Parameter definitions
divider = 1000000
s_minus = 60000 / divider  # based on avg 6.55 years at $10K
s_plus = 14200000 / divider

gamma = 0.05
perc = 0.1

# for perc in [0.06, 0.08, 0.1, 0.12, 0.14]:
for perc in [0.1]:

    for gamma in [0.05]:

        # datetime object containing current date and time
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)

        tic = time.perf_counter()

        print('gamma: ', gamma)
        print('perc: ', perc)

        p_A = 0.405
        p_B = 0.186
        p_C = 0.101

        p_A = 0.20
        p_B = 0.15
        p_C = 0.065

        S1_A_false_count = 0
        S1_B_false_count = 0
        S1_C_false_count = 0

        # Create graph
        G = nx.DiGraph()
        A = 5
        B = 5
        C = 5

        max_size = 5

        for a in range(0, A + 1):
            print('Outter loop ', a)
            for b in range(0, B + 1):
                print('   Middle loop ', b)
                for c in range(0, C + 1):
                    S1_A, S1_B, S1_C = True, True, True
                    S2_A, S2_B, S2_C = True, True, True
                    G1_A, G1_B, G1_C = True, True, True
                    G2_A, G2_B, G2_C = True, True, True

                    if (a == 0) and (b == 0) and (c == 0):
                        continue

                    S1_A, S1_B, S1_C = compute_S1(a, b, c)

                    if S1_A == False:
                        G.add_edges_from([((a, b, c), (a - 1, b, c))], color='darkred')
                        S1_A_false_count += 1
                    if S1_B == False:
                        G.add_edges_from([((a, b, c), (a, b - 1, c))], color='darkred')
                        S1_B_false_count += 1
                    if S1_C == False:
                        G.add_edges_from([((a, b, c), (a, b, c - 1))], color='darkred')
                        S1_B_false_count += 1

                    ## If we're S1 stable, then check S2, otherwise skip
                    # if (S1_A == True and S1_B == True):
                    S2_A, S2_B, S2_C = compute_S2(a, b, c)
                    if S2_A == False:
                        G.add_edges_from([((a, b, c), (a - 1, b, c))], color='r')
                    if S2_B == False:
                        G.add_edges_from([((a, b, c), (a, b - 1, c))], color='r')
                    if S2_C == False:
                        G.add_edges_from([((a, b, c), (a, b, c - 1))], color='r')

                    # Special colour if both S1 and S2 hold
                    if S1_A == False and S2_A == False:
                        G.add_edges_from([((a, b, c), (a - 1, b, c))], color='purple')
                    if S1_B == False and S2_B == False:
                        G.add_edges_from([((a, b, c), (a, b - 1, c))], color='purple')
                    if S1_C == False and S2_C == False:
                        G.add_edges_from([((a, b, c), (a, b, c - 1))], color='purple')

                    ## If we're S1 and S2 stable, then check G1/G2
                    # if (S1_A == True and S1_B == True and S2_A == True and S2_B == True ):
                    G1_A, G1_B, G1_C = compute_G1(a, b, c)
                    G2_A, G2_B, G2_C = compute_G2(a, b, c)

                    if (G1_A == False and G2_A == False):
                        G.add_edges_from([((a, b, c), (a + 1, b, c))], color='b')
                    if (G1_B == False and G2_B == False):
                        G.add_edges_from([((a, b, c), (a, b + 1, c))], color='b')
                    if (G1_C == False and G2_C == False):
                        G.add_edges_from([((a, b, c), (a, b, c + 1))], color='b')

                    # if (G2_A == False):
                    #     G.add_edges_from([((a, b), (a + 1, b))], color='b')
                    # if (G2_B == False):
                    #     G.add_edges_from([((a, b), (a, b + 1))], color='b')

        # Remove insensible edges to (0,0)
        G.remove_edges_from([((0, 1,0), (0, 0,0)), ((1, 0,0), (0, 0,0))])  # edges from origin to solo pool
        G.remove_node((0,0,0))

        # filename = "G_gamma_{}_perc_{}_pa_{}.gpickle".format(gamma, perc, p_A)
        # nx.write_gpickle(G, filename)

        print("Calculations Complete")

        toc = time.perf_counter()
        print(f"Completed the calculations in {(toc - tic)/3600:0.4f} hours")

#%%


gamma = 0.01
perc = 0.06
p_A = 0.12
max_size = 6


filename = "pool_networks/G_gamma_{}_perc_{}_pa_{}.gpickle".format(gamma, perc, p_A)
G = nx.read_gpickle(filename)


#%%
# remove nodes greater than a size
for a in range(0, 12+2):
    for b in range(0,12+2):
        for c in range(0,12+2):
            if a+b+c > max_size:
                if (a,b,c) in G:
                    G.remove_node((a,b,c))


# find stable pools\=
# These are pools with size == max_size, and no out_arrows and no homoegenous pools
stable_pools = []
for node in G.nodes():
    if G.out_degree(node) == 0:
        # if (node[0] + node[1] + node[2] == max_size) & (node[0] != max_size) & (node[1] != max_size) & (node[2] != max_size):
        if (node[0] + node[1] + node[2] == max_size):
            stable_pools.append(node)


# add dummy edges from origin
G.add_edges_from([((0,0,0), (1,0,0))], color='black')
G.add_edges_from([((0,0,0), (0,1,0))], color='black')
G.add_edges_from([((0,0,0), (0,0,1))], color='black')

color_map = []
for node in G:
    if node[0] == 5 or node[1] == 5:
        color_map.append('white')
    else:
        color_map.append('silver')


# Plot 3D Graph
# Extract node and edge positions from the layout
pos = {(x, y, z): (x, y, z) for x, y, z in G.nodes()}
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

# Create the 3D figure
fig = plt.figure()
fig.set_size_inches(11,9.25)
ax = fig.add_subplot(111, projection="3d")
# ax.tick_params(labelsize=12)

ax.tick_params(axis='both', which='major', labelsize=22)


# Plot the nodes - alpha is scaled by "depth" automatically
# ax.scatter(*node_xyz.T, s=500, ec="g", alpha=0.5, c="g")

for viznode in node_xyz:
    if sum(viznode) != max_size:
        ax.scatter(viznode[0],viznode[1],viznode[2], s=1100, ec="silver", alpha=0.4, c="silver",zorder=2)
    else:
        ax.scatter(viznode[0], viznode[1], viznode[2], s=1100, ec="silver", alpha=0.55, c="silver",zorder=2)

for node in stable_pools:
    ax.scatter(node[0], node[1], node[2], s=1100, ec="grey", alpha=0.5, c="grey",zorder=2) # linewidth=3
    # ax.text(node[0], node[1], node[2], '(%s, %s, %s)' % (str(node[0]), str(node[1]), str(node[2])), size=20, zorder=0,color='k')


# Plot the edges
colors = [G[u][v]['color'] for u, v in G.edges()]
ii = 0
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color=colors[ii])
    ii += 1

# for vizedge in edge_xyz:
#     ax.arrow3D(vizedge[0][0],vizedge[0][1],vizedge[0][2],
#                vizedge[1][0] - vizedge[0][0],vizedge[1][1] - vizedge[0][1],vizedge[1][2] - vizedge[0][2],
#                color=colors[ii],
#                mutation_scale=10,
#                arrowstyle="-|>",
#                linestyle='dashed')
#     ii += 1



ax.set_xlabel("Number Class 1 Agents", labelpad=20, fontsize=26)
ax.set_ylabel("Number Class 2 Agents", labelpad=20, fontsize=26)
ax.set_zlabel("Number Class 3 Agents", labelpad=20, fontsize=26)

# plt.rc('axes', labelsize=16)
# matplotlib.rc('font', size=36)

ax.set_xticks(np.arange(0,max_size+1))
ax.set_yticks(np.arange(0,max_size+1))
ax.set_zticks(np.arange(0,max_size+1))

ax.view_init(elev=22., azim=46)

# style = dict(size=10, color='red')
# ax.text(5,0,0, "(5,0,0)", ha='center', **style,zorder=1)



# ax.set_xlim(0,A)
# ax.set_ylim(0,B)
# ax.set_zlim(0,C)

ax.set_title('$K={}$, $c={}$, $\gamma$ = {}'.format(max_size, perc, gamma))

fig.tight_layout()
plt.show()

style = dict(size=10, color='red')
for stable_pool in stable_pools:
    if (stable_pool[0] != 0) & (stable_pool[1] != 0) & (stable_pool[2] != 0):
        ax.text(stable_pool[0], stable_pool[1], stable_pool[2], stable_pool, ha='center', **style, zorder=5)

for node in G.nodes:
    ax.text(node[0], node[1], node[2], node, ha='center', **style, zorder=5)
