import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import requests, bs4
import os
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
from poibin import PoiBin

import networkx as nx


# %%

# Define functions

def compute_S1_old(a, b):
    S1_A_util = 0
    S1_B_util = 0

    # E[U(alpha stay)] - E[U(alpha leave)]
    if a > 0:
        S1_A_util = p_A * (
                math.exp(-gamma * s_plus) - math.exp(-gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
            ((a - 1) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                    (1 - p_A) * (math.exp(-gamma * s_minus) - math.exp(-gamma * s_minus) * math.exp(
            ((a - 1) * p_A + b * p_B) *
            (math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    # E[U(beta stay)] - E[U(beta leave)]
    if b > 0:
        S1_B_util = p_B * (
                math.exp(-gamma * s_plus) - math.exp(-gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
            ((a) * p_A + (b - 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                    (1 - p_B) * (math.exp(-gamma * s_minus) - math.exp(-gamma * s_minus) * math.exp(
            ((a) * p_A + (b - 1) * p_B) *
            (math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    return S1_A_util >= 0, S1_B_util >= 0, S1_A_util


def compute_S1(a, b):
    k = a + b - 1

    S1_A_util = 0
    S1_B_util = 0

    if a > 0:
        # E[U(alpha stay)]
        S1_A_util = 0
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                S1_A_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

        # E[U(alpha leave)]
        S1_A_util -= p_A * (1 - math.exp(-gamma * s_plus))/gamma + (1 - p_A) * (1 - math.exp(-gamma * s_minus))/gamma

    if b > 0:
        # E[U(beta stay)]
        S1_B_util = 0
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                S1_B_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

        # E[U(beta leave)]
        S1_B_util -= p_B * (1 - math.exp(-gamma * s_plus))/gamma + (1 - p_B) * (1 - math.exp(-gamma * s_minus))/gamma

    return S1_A_util >= 0, S1_B_util >= 0, S1_A_util


def compute_S2_old(a, b):
    S2_A_keep_A_util = -1
    S2_B_keep_A_util = -1
    S2_A_keep_B_util = -1
    S2_B_keep_B_util = -1

    if a > 0:
        if a != 1:
            # E[U(alpha keep an alpha)] - E[U(alpha remove an alpha)]
            S2_A_keep_A_util = p_A * (math.exp(-gamma * s_plus * (1 - (c * (a + b - 2) / (a + b - 1)))) * math.exp(
                ((a - 2) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b - 1)) - 1)) - math.exp(
                -gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
                ((a - 1) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                               (1 - p_A) * (math.exp(-gamma * s_minus) * math.exp(((a - 2) * p_A + b * p_B) * (
                    math.exp(-gamma * perc * s_plus / (a + b - 1)) - 1)) - math.exp(-gamma * s_minus) * math.exp(
                ((a - 1) * p_A + b * p_B) * (
                        math.exp(-gamma * perc * s_plus / (a + b)) - 1)))
    if a > 0 and b > 0:
        # E[U(beta keep an alpha)] - E[U(beta remove an alpha)]
        S2_B_keep_A_util = p_B * (math.exp(-gamma * s_plus * (1 - (c * (a + b - 2) / (a + b - 1)))) * math.exp(
            ((a - 1) * p_A + (b - 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b - 1)) - 1)) - math.exp(
            -gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
            ((a) * p_A + (b - 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                           (1 - p_B) * (math.exp(-gamma * s_minus) * math.exp(((a - 1) * p_A + (b - 1) * p_B) * (
                math.exp(-gamma * perc * s_plus / (a + b - 1)) - 1)) - math.exp(-gamma * s_minus) * math.exp(
            ((a) * p_A + (b - 1) * p_B) * (
                    math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    if a > 0:
        # Check if both players prefer to keep the alpha (i.e., stable)
        if (S2_A_keep_A_util >= 0) or (S2_B_keep_A_util >= 0):
            S2_A = True
        else:
            S2_A = False

    if a > 0 and b > 0:
        # E[U(alpha keep a beta)] - E[U(alpha remove a beta)]
        S2_A_keep_B_util = p_A * (math.exp(-gamma * s_plus * (1 - (c * (a + b - 2) / (a + b - 1)))) * math.exp(
            ((a - 1) * p_A + (b - 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b - 1)) - 1)) - math.exp(
            -gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
            ((a - 1) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                           (1 - p_A) * (math.exp(-gamma * s_minus) * math.exp(((a - 1) * p_A + (b - 1) * p_B) * (
                math.exp(-gamma * perc * s_plus / (a + b - 1)) - 1)) - math.exp(-gamma * s_minus) * math.exp(
            ((a - 1) * p_A + b * p_B) * (
                    math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    if b > 0:
        if b != 1:
            # E[U(beta keep a beta)] - E[U(beta remove a beta)]
            S2_B_keep_B_util = p_B * (math.exp(-gamma * s_plus * (1 - (c * (a + b - 2) / (a + b - 1)))) * math.exp(
                ((a) * p_A + (b - 2) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b - 1)) - 1)) - math.exp(
                -gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
                ((a) * p_A + (b - 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                               (1 - p_B) * (math.exp(-gamma * s_minus) * math.exp(((a) * p_A + (b - 2) * p_B) * (
                    math.exp(-gamma * perc * s_plus / (a + b - 1)) - 1)) - math.exp(-gamma * s_minus) * math.exp(
                ((a) * p_A + (b - 1) * p_B) * (
                        math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    # Check if both players prefer to keep the beta (i.e., stable)
    if b > 0:
        if (S2_A_keep_B_util >= 0) or (S2_B_keep_B_util >= 0):
            S2_B = True
        else:
            S2_B = False

    if a == 0 or b == 0:
        S2_A = True
        S2_B = True

    return S2_A, S2_B


def compute_S2(a, b):
    S2_A_keep_A_util = -1
    S2_B_keep_A_util = -1
    S2_A_keep_B_util = -1
    S2_B_keep_B_util = -1

    k = a + b - 1

    if a > 0:
        if a != 1:
            S2_A_keep_A_util = 0
            # E[U(alpha keep an alpha)]
            for i in range(0, a - 1 + 1):
                for j in range(0, b + 1):
                    S2_A_keep_A_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                           (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

            # E[U(alpha remove an alpha)]
            for i in range(0, a - 2 + 1):
                for j in range(0, b + 1):
                    S2_A_keep_A_util -= binom.pmf(i, a - 2, p_A) * binom.pmf(j, b, p_B) * (p_A * (
                            1 - math.exp(-gamma * s_plus) * math.exp(
                        -gamma * perc * s_plus * (i + j - k + 1) / (a + b - 1)))/gamma +
                                                                                           (1 - p_A) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b - 1)))/gamma)

    if a > 0 and b > 0:
        # E[U(beta keep an alpha)]

        S2_B_keep_A_util = 0
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                S2_B_keep_A_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                       (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

        # E[U(beta remove an alpha)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b - 1 + 1):
                S2_B_keep_A_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b - 1, p_B) * (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k + 1) / (a + b - 1)))/gamma +
                                                                                           (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b - 1)))/gamma)

    if a > 0:
        # Check if both players prefer to keep the alpha (i.e., stable)
        if (S2_A_keep_A_util >= 0) or (S2_B_keep_A_util >= 0):
            S2_A = True
        else:
            S2_A = False

    if a > 0 and b > 0:
        # E[U(alpha keep a beta)]
        S2_A_keep_B_util = 0
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                S2_A_keep_B_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                       (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

        # E[U(alpha remove a beta)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b - 1 + 1):
                S2_A_keep_B_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b - 1, p_B) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k + 1) / (a + b - 1)))/gamma +
                                                                                           (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b - 1)))/gamma)

    if b > 0:
        if b != 1:

            S2_B_keep_B_util = 0
            # E[U(beta keep a beta)]
            for i in range(0, a + 1):
                for j in range(0, b - 1 + 1):
                    S2_B_keep_B_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                           (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

            # E[U(beta remove a beta)]
            for i in range(0, a + 1):
                for j in range(0, b - 2 + 1):
                    S2_B_keep_B_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b - 2, p_B) * (p_B * (
                            1 - math.exp(-gamma * s_plus) * math.exp(
                        -gamma * perc * s_plus * (i + j - k + 1) / (a + b - 1)))/gamma +
                                                                                           (1 - p_B) * (1 - math.exp(
                                -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b - 1)))/gamma)

    # Check if both players prefer to keep the beta (i.e., stable)
    if b > 0:
        if (S2_A_keep_B_util >= 0) or (S2_B_keep_B_util >= 0):
            S2_B = True
        else:
            S2_B = False

    if a == 0 or b == 0:
        S2_A = True
        S2_B = True

    return S2_A, S2_B


def compute_G1_old(a, b):
    # E[U(alpha do not join)] - E[U(alpha join)]
    G1_A_util = p_A * (math.exp(-gamma * s_plus * (1 - (c * (a + b) / (a + b + 1)))) * math.exp(
        ((a) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(-gamma * s_plus)) + \
                (1 - p_A) * (math.exp(-gamma * s_minus) * math.exp(((a) * p_A + b * p_B) * (
            math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(-gamma * s_minus))

    # E[U(beta do not join)] - E[U(beta join)]
    G1_B_util = p_B * (math.exp(-gamma * s_plus * (1 - (c * (a + b) / (a + b + 1)))) * math.exp(
        ((a) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(-gamma * s_plus)) + \
                (1 - p_B) * (math.exp(-gamma * s_minus) * math.exp(((a) * p_A + b * p_B) * (
            math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(-gamma * s_minus))

    return G1_A_util >= 0, G1_B_util >= 0


def compute_G1(a, b):
    G1_A_util = 0
    G1_B_util = 0
    k = a + b - 1

    # E[U(alpha do not join)]
    G1_A_util += p_A * (1 - math.exp(-gamma * s_plus))/gamma + (1 - p_A) * (1 - math.exp(-gamma * s_minus))/gamma

    # E[U(alpha join)]
    for i in range(0, a + 1):
        for j in range(0, b + 1):

            G1_A_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * (p_A * (
                    1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k - 1) / (a + b + 1)))/gamma +
                                                                        (1 - p_A) * (1 - math.exp(
                        -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b + 1)))/gamma)

    # E[U(beta do not join)]
    G1_B_util += p_B * (1 - math.exp(-gamma * s_plus))/gamma + (1 - p_B) * (1 - math.exp(-gamma * s_minus))/gamma

    # E[U(beta join)]
    for i in range(0, a + 1):
        for j in range(0, b + 1):
            G1_B_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * (p_B * (
                    1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k - 1) / (a + b + 1)))/gamma +
                                                                        (1 - p_B) * (1 - math.exp(
                        -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b + 1)))/gamma)

    return G1_A_util >= 0, G1_B_util >= 0


def compute_G2_old(a, b):
    G2_A_dont_allow_A_util = -1
    G2_B_dont_allow_A_util = -1
    G2_A_dont_allow_B_util = -1
    G2_B_dont_allow_B_util = -1

    # E[U(alpha do not allow an alpha)] - E[U(alpha allow an alpha)]
    if a > 0:
        G2_A_dont_allow_A_util = p_A * (math.exp(-gamma * s_plus * (1 - (c * (a + b) / (a + b + 1)))) * math.exp(
            ((a) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(
            -gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
            ((a - 1) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                                 (1 - p_A) * (math.exp(-gamma * s_minus) * math.exp(((a) * p_A + b * p_B) * (
                math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(-gamma * s_minus) * math.exp(
            ((a - 1) * p_A + b * p_B) * (
                    math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    # E[U(beta do not allow an alpha)] - E[U(beta allow an alpha)]
    if b > 0:
        G2_B_dont_allow_A_util = p_B * (math.exp(-gamma * s_plus * (1 - (c * (a + b) / (a + b + 1)))) * math.exp(
            ((a + 1) * p_A + (b - 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(
            -gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
            ((a) * p_A + (b - 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                                 (1 - p_B) * (math.exp(-gamma * s_minus) * math.exp(((a + 1) * p_A + (b - 1) * p_B) * (
                math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(-gamma * s_minus) * math.exp(
            ((a) * p_A + (b - 1) * p_B) * (
                    math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    # Check if both players prefer to not allow the alpha (i.e., stable)
    if (G2_A_dont_allow_A_util >= 0) or (G2_B_dont_allow_A_util >= 0):
        G2_A = True
    else:
        G2_A = False

    # E[U(alpha do not allow an beta)] - E[U(alpha allow an beta)]
    if a > 0:
        G2_A_dont_allow_B_util = p_A * (math.exp(-gamma * s_plus * (1 - (c * (a + b) / (a + b + 1)))) * math.exp(
            ((a - 1) * p_A + (b + 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(
            -gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
            ((a - 1) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                                 (1 - p_A) * (math.exp(-gamma * s_minus) * math.exp(((a - 1) * p_A + (b + 1) * p_B) * (
                math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(-gamma * s_minus) * math.exp(
            ((a - 1) * p_A + b * p_B) * (
                    math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    # E[U(beta do not allow an beta)] - E[U(beta allow an beta)]
    if b > 0:
        G2_B_dont_allow_B_util = p_B * (math.exp(-gamma * s_plus * (1 - (c * (a + b) / (a + b + 1)))) * math.exp(
            ((a) * p_A + b * p_B) * (math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(
            -gamma * s_plus * (1 - (c * (a + b - 1) / (a + b)))) * math.exp(
            ((a) * p_A + (b - 1) * p_B) * (math.exp(-gamma * perc * s_plus / (a + b)) - 1))) + \
                                 (1 - p_B) * (math.exp(-gamma * s_minus) * math.exp(((a) * p_A + b * p_B) * (
                math.exp(-gamma * perc * s_plus / (a + b + 1)) - 1)) - math.exp(-gamma * s_minus) * math.exp(
            ((a) * p_A + (b - 1) * p_B) * (
                    math.exp(-gamma * perc * s_plus / (a + b)) - 1)))

    # Check if both players prefer to not allow the beta (i.e., stable)
    if (G2_A_dont_allow_B_util >= 0) or (G2_B_dont_allow_B_util >= 0):
        G2_B = True
    else:
        G2_B = False

    return G2_A, G2_B


def compute_G2(a, b):
    G2_A_dont_allow_A_util = -1
    G2_B_dont_allow_A_util = -1
    G2_A_dont_allow_B_util = -1
    G2_B_dont_allow_B_util = -1
    k = a + b - 1

    if a > 0:
        G2_A_dont_allow_A_util = 0
        # E[U(alpha do not allow an alpha)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                G2_A_dont_allow_A_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                             (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

        # E[U(alpha allow an alpha)]
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                G2_A_dont_allow_A_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k - 1) / (a + b + 1)))/gamma +
                                                                                         (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b + 1)))/gamma)

    if b > 0:

        G2_B_dont_allow_A_util = 0
        # E[U(beta do not allow an alpha)]
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                G2_B_dont_allow_A_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                             (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

        # E[U(beta allow an alpha)]
        for i in range(0, a + 1 + 1):
            for j in range(0, b - 1 + 1):
                G2_B_dont_allow_A_util -= binom.pmf(i, a + 1, p_A) * binom.pmf(j, b - 1, p_B) * (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k - 1) / (a + b + 1)))/gamma +
                                                                                                 (1 - p_B) * (
                                                                                                             1 - math.exp(
                                                                                                         -gamma * s_minus) * math.exp(
                                                                                                         -gamma * perc * s_plus * (
                                                                                                                     i + j) / (
                                                                                                                     a + b + 1)))/gamma)

    # Check if both players prefer to not allow the alpha (i.e., stable)
    if (G2_A_dont_allow_A_util >= 0) or (G2_B_dont_allow_A_util >= 0):
        G2_A = True
    else:
        G2_A = False

    if a > 0:

        G2_A_dont_allow_B_util = 0
        # E[U(alpha do not allow an beta)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1):
                G2_A_dont_allow_B_util += binom.pmf(i, a - 1, p_A) * binom.pmf(j, b, p_B) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                             (1 - p_A) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

        # E[U(alpha allow an beta)]
        for i in range(0, a - 1 + 1):
            for j in range(0, b + 1 + 1):
                G2_A_dont_allow_B_util -= binom.pmf(i, a - 1, p_A) * binom.pmf(j, b + 1, p_B) * (p_A * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k - 1) / (a + b + 1)))/gamma +
                                                                                                 (1 - p_A) * (
                                                                                                             1 - math.exp(
                                                                                                         -gamma * s_minus) * math.exp(
                                                                                                         -gamma * perc * s_plus * (
                                                                                                                     i + j) / (
                                                                                                                     a + b + 1)))/gamma)

    # E[U(beta do not allow an beta)] - E[U(beta allow an beta)]
    if b > 0:
        G2_B_dont_allow_B_util = 0
        # E[U(beta do not allow an beta)]
        for i in range(0, a + 1):
            for j in range(0, b - 1 + 1):
                G2_B_dont_allow_B_util += binom.pmf(i, a, p_A) * binom.pmf(j, b - 1, p_B) * (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k) / (a + b)))/gamma +
                                                                                             (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b)))/gamma)

        # E[U(beta allow an beta)]
        for i in range(0, a + 1):
            for j in range(0, b + 1):
                G2_B_dont_allow_B_util -= binom.pmf(i, a, p_A) * binom.pmf(j, b, p_B) * (p_B * (
                        1 - math.exp(-gamma * s_plus) * math.exp(-gamma * perc * s_plus * (i + j - k - 1) / (a + b + 1)))/gamma +
                                                                                         (1 - p_B) * (1 - math.exp(
                            -gamma * s_minus) * math.exp(-gamma * perc * s_plus * (i + j) / (a + b + 1)))/gamma)

    # Check if both players prefer to not allow the beta (i.e., stable)
    if (G2_A_dont_allow_B_util >= 0) or (G2_B_dont_allow_B_util >= 0):
        G2_B = True
    else:
        G2_B = False

    return G2_A, G2_B


def strictly_increasing(L):
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))


# %%

# Parameter definitions
divider = 1000000
s_minus = 65000 / divider  # based on avg 6.55 years at $10K
# s_minus = 0 / divider  # based on avg 6.55 years at $10K
s_plus = 18000000 / divider

perc = 0.1

p_A = 0.15
p_B = 0.065

gamma = 0.1

# Create graph
G = nx.DiGraph()
A = 8
B = 8

max_size = 20

# G.add_edges_from([((0, 0), (1, 0)), ((0, 0), (0, 1))], color='b')  # edges from origin to solo pool

S1_A_false_count = 0
S1_B_false_count = 0

# G.add_node('$n_1$ , $n_2$')
# plt.figure(figsize=(6, 6))
# nx.draw(G,node_color='lightgreen', with_labels=True,arrows=True, arrowsize=10, node_size=25000, font_size=36, )

mat_A_s1_util = np.zeros((B + 1, A + 1))

for a in range(0, A + 1):
    for b in range(0, B + 1):

        S1_A, S1_B = True, True
        S2_A, S2_B = True, True
        G1_A, G1_B = True, True
        G2_A, G2_B = True, True

        if (a == 0) and (b == 0):
            continue

        if (a == 1) and (b == 1):
            tmp = 0
        if (a == 4) and (b == 9):
            tmp = 0

        S1_A, S1_B, mat_A_s1_util[b, a] = compute_S1(a, b)

        if S1_A == False:
            G.add_edges_from([((a, b), (a - 1, b))], color='moccasin')
            S1_A_false_count += 1
        if S1_B == False:
            G.add_edges_from([((a, b), (a, b - 1))], color='moccasin')
            S1_B_false_count += 1

        ## If we're S1 stable, then check S2, otherwise skip
        # if (S1_A == True and S1_B == True):
        S2_A, S2_B = compute_S2(a, b)
        if S2_A == False:
            G.add_edges_from([((a, b), (a - 1, b))], color='lightcoral')
        if S2_B == False:
            G.add_edges_from([((a, b), (a, b - 1))], color='lightcoral')

        # Special colour if both S1 and S2 hold
        if S1_A == False and S2_A == False:
            G.add_edges_from([((a, b), (a - 1, b))], color='purple')
        if S1_B == False and S2_B == False:
            G.add_edges_from([((a, b), (a, b - 1))], color='purple')

        ## If we're S1 and S2 stable, then check G1/G2
        # if (S1_A == True and S1_B == True and S2_A == True and S2_B == True ):
        G1_A, G1_B = compute_G1(a, b)
        G2_A, G2_B = compute_G2(a, b)

        if (G1_A == False and G2_A == False):
            G.add_edges_from([((a, b), (a + 1, b))], color='lightskyblue')
        if (G1_B == False and G2_B == False):
            G.add_edges_from([((a, b), (a, b + 1))], color='lightskyblue')

        # if (G2_A == False):
        #     G.add_edges_from([((a, b), (a + 1, b))], color='b')
        # if (G2_B == False):
        #     G.add_edges_from([((a, b), (a, b + 1))], color='b')

# Remove insensible edges to (0,0)
G.remove_edges_from([((0, 1), (0, 0)), ((1, 0), (0, 0))])  # edges from origin to solo pool

# remove nodes greater than a size
for a in range(0, A+1 +1):
    for b in range(0,B +1 + 1):
            if a+b > max_size:
                if (a,b) in G:
                    G.remove_node((a,b))
# G.add_edge((0,4),(0,5),color='white')
# G.add_edge((4,0),(5,0),color='white')

stable_pools = []
for node in G.nodes():
    if G.out_degree(node) == 0:
        # if (node[0] + node[1] + node[2] == max_size) & (node[0] != max_size) & (node[1] != max_size) & (node[2] != max_size):
        if (node[0] + node[1] == max_size):
            stable_pools.append(node)
#

if gamma == 0.5:
    stable_pools = [(0,4), (4,0),(1,3),(2,2)]
elif gamma == 0.01:
    stable_pools = [(0,4), (4,0)]

# Draw graph
color_map = []
for node in G:
    # if node[0] == 5 or node[1] == 5:
    #     color_map.append('white')
    if node in stable_pools:
        color_map.append('grey')
    else:
        color_map.append('silver')

line_width_map = []
for node in G:
    if node in stable_pools:
        line_width_map.append(5)
    else:
        line_width_map.append(3)

plt.figure(figsize=(6, 6))
pos = {(x, y): (x, y) for x, y in G.nodes()}
colors = [G[u][v]['color'] for u, v in G.edges()]
nx.draw(G,
        pos=pos, node_color=color_map, edge_color=colors, with_labels=True,
        arrows=True, arrowsize=15, width=3, edge_vmax=line_width_map, node_size=1000, font_size=11, )

# plt.gca().plot((-0.25,4.75), (4.75,-0.25), color='black', linestyle='--')
# plt.text(2., -0.35, 'Class 1', fontsize=14)
# plt.text(-0.35,2, 'Class 2', rotation=90, fontsize=14)
plt.text((A + 1) / 2 - 4.5, B + 1.5, '$p_1$ = {}, $p_1$ = {}, $\gamma$ = {}, $s^+$={}M, $s^-$={}M, $c$={}'.format(p_A, p_B, gamma, s_plus,s_minus, perc), fontsize=12)
# plt.text((A + 1) / 2 , B + 1.5, '$p_1$ = {}, $p_2$ = {}, $\gamma$ = {}, $c$={}, $s^+$={},$s^-={}}'.format(p_A, p_B, gamma, perc, s_plus, s_minus), fontsize=12)


# %%

# find stable pools
# These are pools with size == max_size, and no out_arrows
stable_pools = []
for node in G.nodes():
    if G.out_degree(node) == 0:
        if node[0] + node[1] == max_size:
            stable_pools.append(node)

stable_pools

#%%


# Find the shortest paths
# print([p for p in nx.all_shortest_paths(G, source=(0, 0), target=(4, 3))])

# %%

###### Stochastic Dominance of S2 with all Alphas

from scipy.stats import poisson

p_A = 0.3
A = 4

probs = [p_A] * (A - 1)
pb = PoiBin(probs)  # create the dist

dont_allow_pmf = []
allow_pmf = []
for i in range(A + 10):
    dont_allow_pmf.append(poisson.cdf(i, (A - 1) * p_A))
    allow_pmf.append(poisson.cdf(i, (A) * p_A))

plt.figure()
plt.plot(allow_pmf, label='allow')
plt.plot(dont_allow_pmf, label='dont_allow')
plt.legend()

####################################
# %%

# PLOT CDF for SOSD for S1

data = np.array([-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17, 18])

y = np.array([0.02251799813685250, 0.11258999068426200, 0.28147497671065600, 0.47850746040811500, 0.63859635341230100,
              0.73464968921481200, 0.77867413479096300, 0.79439715106816000, 0.79881924939612200, 0.79980193791344600,
              0.79997390840397800, 0.79999735892541400, 0.79999980168806400, 0.79999998959288300, 0.79999999965921300,
              0.79999999999475700, 0.80000000000000000, 0.8056295, 0.828147498, 0.870368744, 0.919626865, 0.959649088,
              0.983662422,
              0.994668534, 0.998599288, 0.999704812, 0.999950484, 0.999993477, 0.99999934, 0.99999995, 0.999999997, 1,
              1, 1, ])
yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='Stay')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

data = np.array([-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17, 18])

y = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
              0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, ])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='Leave')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')
# ax.set_xticks(data)

# %%

y = np.array([0.02251799813685250, 0.11258999068426200, 0.28147497671065600, 0.47850746040811500, 0.63859635341230100,
              0.73464968921481200, 0.77867413479096300, 0.79439715106816000, 0.79881924939612200, 0.79980193791344600,
              0.79997390840397800, 0.79999735892541400, 0.79999980168806400, 0.79999998959288300, 0.79999999965921300,
              0.79999999999475700, 0.80000000000000000, 0.8056295, 0.828147498, 0.870368744, 0.919626865, 0.959649088,
              0.983662422,
              0.994668534, 0.998599288, 0.999704812, 0.999950484, 0.999993477, 0.99999934, 0.99999995, 0.999999997, 1,
              1, 1, ])
yn_stay = np.insert(y, 0, 0)

y = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
              0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, ])

yn_leave = np.insert(y, 0, 0)

diff = yn_stay - yn_leave
data = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17, 18])
plt.figure()
plt.scatter(data, diff)
# %%

# PLOT CDF for SOSD for S1 SMALL

data = np.array([-1, 0, '$cs^+/n$','$s^+ - cs^+(n-1)/n$', '$s^+$' ])

y = np.array([0,0.71680000000000000,0.79360000000000000,0.80000000000000000,0.90240000000000000,0.97920000000000000,0.99840000000000000,1.00000000000000000,])
yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label=r'$F_i^{(n,p)}$')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

data = np.array([-1, 1,1.425,1.85,2.275,15.725,16.15,16.575,17, 18])


y = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, ])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label=r'$F_i^{\emptyset}$')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')
# ax.set_xticks(data)

data = np.array([1,1.425,1.85,2.275,15.725,16.15,16.575,17])
ax.set_xticks(data)
ax.set_xticklabels(ax.get_xticks(), rotation=90)

# %%

y = np.array([0.02251799813685250, 0.11258999068426200, 0.28147497671065600, 0.47850746040811500, 0.63859635341230100,
              0.73464968921481200, 0.77867413479096300, 0.79439715106816000, 0.79881924939612200, 0.79980193791344600,
              0.79997390840397800, 0.79999735892541400, 0.79999980168806400, 0.79999998959288300, 0.79999999965921300,
              0.79999999999475700, 0.80000000000000000, 0.8056295, 0.828147498, 0.870368744, 0.919626865, 0.959649088,
              0.983662422,
              0.994668534, 0.998599288, 0.999704812, 0.999950484, 0.999993477, 0.99999934, 0.99999995, 0.999999997, 1,
              1, 1, ])
yn_stay = np.insert(y, 0, 0)

y = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
              0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, ])

yn_leave = np.insert(y, 0, 0)

diff = yn_stay - yn_leave
data = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17, 18])
plt.figure()
plt.scatter(data, diff)



# %%

#############
# computing area for S1 dominance

###########

# tot = 0
#
# for i in range(0,17):
#     for j in range(0,i):
#         tot += binom.pmf(j, 16, 0.2)
#
# pos_area = 0.1*17*0.8*16/17 - tot*0.1*17*0.8/17
#
#
# tot = 0
#
# for i in range(0,17):
#     for j in range(0,i):
#         tot += binom.pmf(j, 16, 0.2)
#     print(tot)
#
# neg_area = tot*0.1*17*0.2/17
#
# from scipy.stats import binom
#
#
# p = 0.475
# q = 1-p
# n = 27
#
# tot = 0
# for i in range(0,n+1):
#     for j in range(0,i):
#         tot += binom.pmf(j, n, p)
#
# print(tot, q*n)


# %%
# PLOT CDF for SOSD for S2

# Allow


data = np.array([-1, 0.000000, 0.094444, 0.188889, 0.283333, 0.377778, 0.472222, 0.566667,
                 0.661111, 0.755556, 0.850000, 0.944444, 1.038889, 1.133333, 1.227778, 1.322222,
                 1.416667, 1.511111, 1.605556, 15.394444, 15.488889, 15.583333, 15.677778,
                 15.772222, 15.866667, 15.961111, 16.055556, 16.150000, 16.244444, 16.338889,
                 16.433333, 16.527778, 16.622222, 16.716667, 16.811111, 16.905556, 17.000000, 18])

y = np.array([0.0180143985, 0.0945755922, 0.2476979795, 0.4391009637, 0.6065785748, 0.7154390221, 0.7698692457,
              0.7912525478, 0.7979348297, 0.7996054002, 0.7999395143, 0.7999926688, 0.7999993131, 0.7999999520,
              0.7999999976, 0.7999999999, 0.8000000000, 0.8000000000, 0.8045035996, 0.8236438980, 0.8619244949,
              0.9097752409, 0.9516446437, 0.9788597555, 0.9924673114, 0.9978131370, 0.9994837074, 0.9999013501,
              0.9999848786, 0.9999981672, 0.9999998283, 0.9999999880, 0.9999999994, 1.0000000000, 1.0000000000,
              1.0000000000, ])

yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

# Dont Allow

data = np.array([-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17, 18])

y = np.array([0.02251799813685250, 0.11258999068426200, 0.28147497671065600, 0.47850746040811500, 0.63859635341230100,
              0.73464968921481200, 0.77867413479096300, 0.79439715106816000, 0.79881924939612200, 0.79980193791344600,
              0.79997390840397800, 0.79999735892541400, 0.79999980168806400, 0.79999998959288300, 0.79999999965921300,
              0.79999999999475700, 0.80000000000000000, 0.8056295, 0.828147498, 0.870368744, 0.919626865, 0.959649088,
              0.983662422,
              0.994668534, 0.998599288, 0.999704812, 0.999950484, 0.999993477, 0.99999934, 0.99999995, 0.999999997, 1,
              1, 1, ])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='Dont Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')

full_data = [0.00, 0.09, 0.10, 0.19, 0.20, 0.28, 0.30, 0.38, 0.40, 0.47, 0.50, 0.57, 0.60, 0.66, 0.70, 0.76, 0.80, 0.85,
             0.90, 0.94, 1.00, 1.04, 1.10, 1.13, 1.20, 1.23,
             1.30, 1.32, 1.40, 1.42, 1.50, 1.51, 1.60, 1.61, 15.39, 15.40, 15.49, 15.50, 15.58, 15.60, 15.68, 15.70,
             15.77, 15.80, 15.87, 15.90, 15.96, 16.00, 16.06, 16.10,
             16.15, 16.20, 16.24, 16.30, 16.34, 16.40, 16.43, 16.50, 16.53, 16.60, 16.62, 16.70, 16.72, 16.80, 16.81,
             16.90, 16.91, 17.00, ]

ax.set_xticks(full_data)
ax.set_xticklabels(ax.get_xticks(), rotation=45)

# %%
# PLOT CDF for SOSD for S2 small

# Allow


data = np.array(
    [-1, 0.000000, 0.340000, 0.680000, 1.020000, 1.360000, 15.640000, 15.980000, 16.320000, 16.660000, 17.000000, 18])

y = np.array(
    [0.3276800000, 0.6553600000, 0.7782400000, 0.7987200000, 0.8000000000, 0.8819200000, 0.9638400000, 0.9945600000,
     0.9996800000, 1.0000000000, ])

yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

# Dont Allow

data = np.array([-1, 0.00, 0.43, 0.85, 1.28, 15.73, 16.15, 16.58, 17.00, 18])

y = np.array(
    [0.409600000000, 0.716800000000, 0.793600000000, 0.800000000000, 0.902400000000, 0.979200000000, 0.998400000000,
     1.000000000000, ])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='Dont Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')

full_data = [0.00000000, 0.34000000, 0.42500000, 0.68000000, 0.85000000, 1.02000000, 1.27500000, 1.36000000,
             15.64000000, 15.72500000, 15.98000000, 16.15000000, 16.32000000, 16.57500000, 16.66000000, 17.00000000, ]

ax.set_xticks(full_data)
ax.set_xticklabels(ax.get_xticks(), rotation=90)

# %%

# s-cdf for S2

income = [0.00, 0.09, 0.10, 0.19, 0.20, 0.28, 0.30, 0.38, 0.40, 0.47, 0.50, 0.57, 0.60, 0.66, 0.70, 0.76, 0.80, 0.85,
          0.90, 0.94, 1.00, 1.04, 1.10, 1.13, 1.20, 1.23,
          1.30, 1.32, 1.40, 1.42, 1.50, 1.51, 1.60, 1.61, 15.39, 15.40, 15.49, 15.50, 15.58, 15.60, 15.68, 15.70, 15.77,
          15.80, 15.87, 15.90, 15.96, 16.00, 16.06, 16.10,
          16.15, 16.20, 16.24, 16.30, 16.34, 16.40, 16.43, 16.50, 16.53, 16.60, 16.62, 16.70, 16.72, 16.80, 16.81,
          16.90, 16.91, 17.00, ]

dont_allow = [0.000000, 0.002127, 0.002252, 0.012260, 0.013511, 0.036967, 0.041658, 0.078876, 0.089509, 0.135630,
              0.153369, 0.202345,
              0.226834, 0.274419, 0.304701, 0.348834, 0.384141, 0.424082, 0.464023, 0.499569, 0.544003, 0.575113,
              0.624000, 0.650667,
              0.704000, 0.726222, 0.784000, 0.801778, 0.864000, 0.877333, 0.944000, 0.952889, 1.024000, 1.028444,
              12.059556, 12.064000,
              12.135612, 12.144563, 12.213575, 12.227378, 12.295073, 12.314415, 12.380832, 12.406377, 12.470354,
              12.502342, 12.562455,
              12.600708, 12.655968, 12.700175, 12.750105, 12.800035, 12.844467, 12.900006, 12.938893, 13.000001,
              13.033334, 13.100000,
              13.127778, 13.200000, 13.222222, 13.300000, 13.316667, 13.400000, 13.411111, 13.500000, 13.505556,
              13.600000, ]

allow = [0.000000, 0.001701, 0.002227, 0.010633, 0.013386, 0.034027, 0.041346, 0.075498, 0.088977, 0.132786, 0.152659,
         0.200355, 0.226017,
         0.273065, 0.303836, 0.347794, 0.383258, 0.423155, 0.463135, 0.498673, 0.543114, 0.574223, 0.623111, 0.649778,
         0.703111, 0.725333,
         0.783111, 0.800889, 0.863111, 0.876444, 0.943111, 0.952000, 1.023111, 1.027556, 12.058667, 12.063136,
         12.134648, 12.143799, 12.212436,
         12.226802, 12.293840, 12.314057, 12.379763, 12.406198, 12.469641, 12.502270, 12.562089, 12.600685, 12.655822,
         12.700169, 12.750060,
         12.800034, 12.844455, 12.900005, 12.938891, 13.000001, 13.033334, 13.100000, 13.127778, 13.200000, 13.222222,
         13.300000, 13.316667,
         13.400000, 13.411111, 13.500000, 13.505556, 13.600000, ]

plt.figure()
plt.plot(income, dont_allow, label='dont allow')
plt.plot(income, allow, label='allow')

diff = np.array(allow) - np.array(dont_allow)

plt.figure()
plt.plot(income, diff)

# %%

# PLOT CDF for SOSD for S1 SMALL

data = np.array([-1, 0, 0.425, 0.85, 1.275, 15.725, 16.15, 16.575, 17, 18])

y = np.array([0.40960000000000000, 0.71680000000000000, 0.79360000000000000, 0.80000000000000000, 0.90240000000000000,
              0.97920000000000000, 0.99840000000000000, 1.00000000000000000, ])

yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='Stay')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

data = np.array([-1, 0, 0.425, 0.85, 1.275, 15.725, 16.15, 16.575, 17, 18])

y = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, ])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='Leave')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')
# ax.set_xticks(data)


# %%

# PLOT CDF for SOSD for S1 SMALL, c=0.7

data = np.array([-1, 0, 2.975, 5.95, 8.075, 8.925, 11.05, 14.025, 17, 18])

y = np.array([0.40960000000000000, 0.71680000000000000, 0.79360000000000000, 0.89600000000000000, 0.90240000000000000,
              0.97920000000000000, 0.99840000000000000, 1.00000000000000000, ])

yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='Stay')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

data = np.array([-1, 0, 2.975, 5.95, 8.075, 8.925, 11.05, 14.025, 17, 18])

y = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1, ])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='Leave')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')
# ax.set_xticks(data)


# %%
# PLOT CDF for SOSD for G2 allowing an Alpha from Alphas perspective

# Allow


data = np.array(
    [-1, 0.00000, 0.24286, 0.24286, 0.48571, 0.48571, 0.48571, 0.72857, 0.72857, 0.72857, 0.72857, 0.97143, 0.97143,
     0.97143,
     1.21429, 1.21429, 1.45714, 15.54286, 15.78571, 15.78571, 16.02857, 16.02857, 16.02857, 16.27143, 16.27143,
     16.27143,
     16.27143, 16.51429, 16.51429, 16.51429, 16.75714, 16.75714, 17.00000, 18])

y = np.array(
    [0.288755302400, 0.395821875200, 0.612388352000, 0.625621299200, 0.705921228800, 0.760062848000, 0.760608025600,
     0.770532736000, 0.790607718400, 0.795119520000, 0.795528403200, 0.798009580800, 0.799682496000, 0.799784716800,
     0.799991481600, 0.800000000000, 0.872188825600, 0.898955468800, 0.953097088000, 0.956405324800, 0.976480307200,
     0.990015712000, 0.990152006400, 0.992633184000, 0.997651929600, 0.998779880000, 0.998882100800, 0.999502395200,
     0.999920624000, 0.999946179200, 0.999997870400, 1.000000000000, ])

yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.set_xticks(data)

# Allow SHIFTED

# data = np.array(
#     [-1, 0.00000, 0.24286, 0.24286, 0.48571, 0.48571, 0.48571, 0.72857, 0.72857, 0.72857, 0.72857, 0.97143, 0.97143,
#      0.97143,
#      1.21429, 1.21429, 1.45714, 15.5833333333333,15.8261904761905,15.8261904761905,16.0690476190476,16.0690476190476,16.0690476190476,16.3119047619048,16.3119047619048,16.3119047619048,16.3119047619048,16.5547619047619,16.5547619047619,16.5547619047619,16.7976190476190,16.7976190476190,17.0404761904762, 18])
# # add shift
# # data += 0.1*18/(6*7)
# # shift_data = data
#
# y = np.array(
#     [0.288755302400, 0.395821875200, 0.612388352000, 0.625621299200, 0.705921228800, 0.760062848000, 0.760608025600,
#      0.770532736000, 0.790607718400, 0.795119520000, 0.795528403200, 0.798009580800, 0.799682496000, 0.799784716800,
#      0.799991481600, 0.800000000000, 0.872188825600, 0.898955468800, 0.953097088000, 0.956405324800, 0.976480307200,
#      0.990015712000, 0.990152006400, 0.992633184000, 0.997651929600, 0.998779880000, 0.998882100800, 0.999502395200,
#      0.999920624000, 0.999946179200, 0.999997870400, 1.000000000000, ])
#
# yn = np.insert(y, 0, 0)
#
# # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
# ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
#           color='black', zorder=1, label='Allow Shift')
#
# # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
# ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='black',
#           linestyle='dashed', zorder=1)
#
# ax.scatter(data[1:-1], y, color='black', s=18, zorder=2)
# ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
#            edgecolor='black')
# ax.grid(False)
# ax.set_xlim(data[0], data[-1])
# ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

# Dont Allow

data = np.array(
    [-1, 0.00000, 0.28333, 0.28333, 0.56667, 0.56667, 0.56667, 0.85000, 0.85000, 0.85000, 1.13333, 1.13333, 1.41667,
     15.58333,
     15.86667, 15.86667, 16.15000, 16.15000, 16.15000, 16.43333, 16.43333, 16.43333, 16.71667, 16.71667, 17.00000, 18])

y = np.array(
    [0.360944128000, 0.494777344000, 0.675249408000, 0.691790592000, 0.758707200000, 0.781266208000, 0.781947680000,
     0.790218272000, 0.798582848000,
     0.798923584000, 0.799957408000, 0.800000000000, 0.890236032000, 0.923694336000, 0.968812352000, 0.972947648000,
     0.989676800000, 0.995316552000, 0.995486920000,
     0.997554568000, 0.999645712000, 0.999730896000, 0.999989352000, 1.000000000000, ])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='Dont Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')

full_data = [0.00, 0.24, 0.28, 0.49, 0.57, 0.73, 0.85, 0.97, 1.13, 1.21, 1.42, 1.46, 15.54, 15.58, 15.79, 15.87, 16.03,
             16.15, 16.27, 16.43, 16.51, 16.72, 16.76, 17.00, ]
ax.set_xticks(full_data)
ax.set_xticklabels(ax.get_xticks(), rotation=45)

#%%%

# %%
# PLOT CDF for SOSD for G2 allowing an Alpha from Alphas perspective SMALL

a = 1
b = 1
p_a = 0.2
p_b = 0.1
q_a = 1 - p_a
q_b = 1-p_b

# Allow

data = np.array(
    [-1, 0.00000,0.56667,1.13333,15.86667,16.43333,17.00000, 18])

y = np.array(
    [0.57600,0.78400,0.80000,0.94400,0.99600,1.00000, ])

yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.set_xticks(data)

# Allow SHIFTED
#
# data = np.array(
#     [-1, 0.00000, 0.24286, 0.24286, 0.48571, 0.48571, 0.48571, 0.72857, 0.72857, 0.72857, 0.72857, 0.97143, 0.97143,
#      0.97143,
#      1.21429, 1.21429, 1.45714, 15.5833333333333,15.8261904761905,15.8261904761905,16.0690476190476,16.0690476190476,16.0690476190476,16.3119047619048,16.3119047619048,16.3119047619048,16.3119047619048,16.5547619047619,16.5547619047619,16.5547619047619,16.7976190476190,16.7976190476190,17.0404761904762, 18])
# # add shift
# # data += 0.1*17/(6*7) (cS/((a+b)*(a+b+1))
# # shift_data = data
#
# y = np.array(
#     [0.288755302400, 0.395821875200, 0.612388352000, 0.625621299200, 0.705921228800, 0.760062848000, 0.760608025600,
#      0.770532736000, 0.790607718400, 0.795119520000, 0.795528403200, 0.798009580800, 0.799682496000, 0.799784716800,
#      0.799991481600, 0.800000000000, 0.872188825600, 0.898955468800, 0.953097088000, 0.956405324800, 0.976480307200,
#      0.990015712000, 0.990152006400, 0.992633184000, 0.997651929600, 0.998779880000, 0.998882100800, 0.999502395200,
#      0.999920624000, 0.999946179200, 0.999997870400, 1.000000000000, ])
#
# yn = np.insert(y, 0, 0)
#
# # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
# ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
#           color='black', zorder=1, label='Allow Shift')
#
# # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
# ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='black',
#           linestyle='dashed', zorder=1)
#
# ax.scatter(data[1:-1], y, color='black', s=18, zorder=2)
# ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
#            edgecolor='black')
# ax.grid(False)
# ax.set_xlim(data[0], data[-1])
# ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

# Dont Allow

data = np.array(
    [-1, 0.00000,0.85000,16.15000,17.00000, 18])

y = np.array(
    [0.720000000,0.800000000,0.980000000,1.000000000,])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='Dont Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')

full_data = [0.00000,0.56667,0.85000,1.13333,15.86667,16.15000,16.43333,17.00000,]
ax.set_xticks(full_data)
ax.set_xticklabels(ax.get_xticks(), rotation=45)

#%%

# PLOT s-CDF for SOSD for G2 allowing an Alpha from Alphas perspective SMALL
fig, ax = plt.subplots()


plt.plot([0,0.85], [0,0.612],'o', color='orange',  linestyle="-", label='Dont Allow')
plt.plot([0.85,16.15], [0.612, 12.85],'o',   color='orange',linestyle="-")
plt.plot([16.15, 17], [12.85, 13.685],'o',  color='orange', linestyle="-")

plt.plot([0,0.5667], [0,0.32640],'o', color='blue',  linestyle="-", label='Allow')
plt.plot([0.5667, 1.13333], [0.32640, 0.77067],'o', color='blue', linestyle="-")
plt.plot([1.13333, 15.86667], [0.77067, 12.55733], 'o',color='blue',linestyle="-")
plt.plot([15.86667, 16.43333], [12.55733, 13.09227],'o', color='blue', linestyle="-")
plt.plot([16.43333, 17], [13.09227, 13.65667],'o', color='blue', linestyle="-")


full_data = [0.00000,0.56667,0.85000,1.13333,15.86667,16.15000,16.43333,17.00000,]
ax.set_xticks(full_data)
ax.set_xticklabels(ax.get_xticks(), rotation=45)
plt.legend()
# %%
# PLOT CDF for SOSD for G2 THREE 3 MULTI Class

a = 1
b = 1
c = 1
p_a = 0.2
p_b = 0.15
p_c = 0.1
q_a = 1 - p_a
q_b = 1-p_b
q_c = 1-p_c

# Allow

data = np.array(
    [-1, 0.00000,0.42500,0.85000,1.27500,15.72500,16.15000,16.57500,17.00000,18])

y = np.array(
    [0.48960,0.75280,0.79760,0.80000,0.92240,0.98820,0.99940,1.00000,])

yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.set_xticks(data)

#  Dont Allow

data = np.array(
    [-1, 0.00000,0.56667,1.13333,15.86667,16.43333,17.00000, 18])

y = np.array(
    [0.612000000,0.788000000,0.800000000,0.953000000,0.997000000,1.000000000,])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='Dont Allow')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend()
ax.set_xlabel('Income')
ax.set_ylabel('Cumulative Probability')

full_data = [0.00000,0.42500,0.56667,0.85000,1.13333,1.27500,15.72500,15.86667,16.15000,16.43333,16.57500,17.00000,]
ax.set_xticks(full_data)
ax.set_xticklabels(ax.get_xticks(), rotation=45)


# %%

tot_lhs = 0
tot_rhs = 0
tot_simple = 0

a = 3
b = 2
k = 2

p_A = 0.2
p_B = 0.11

for i in range(k - 1 + 1):
    for j in range(i + 1):
        tot_lhs += binom.pmf(j, a - 1, p_A) * binom.pmf(i - j, b, p_B)

for i in range(k - 1 + 1):
    for j in range(i + 1):
        tot_rhs += binom.pmf(j - 1, a - 1, p_A) * binom.pmf(i - j, b, p_B)

for j in range(k - 1 + 1):
    tot_simple += binom.pmf(j, a - 1, p_A) * binom.pmf(k - 1 - j, b, p_B)

print(tot_lhs, tot_rhs, tot_simple)
print(tot_lhs - tot_rhs, tot_simple)

# %%

# %%

# %%
# TRYING TO PLOT CDF OF CONVOLUTION OF BINOMIALS, WITH (a-1) PEOPLE VS (b-1) PEOPLE

# Allow


data = np.array([-1, 0.000000, 1, 2, 3, 4, 5, 6])

y = np.array([0.45118016, 0.844062, 0.976583, 0.998229, 0.999947, 1.000000, ])

yn = np.insert(y, 0, 0)

fig, ax = plt.subplots()
ax.set_facecolor('white')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='blue', zorder=1, label='(a-1,b)')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='blue',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='blue', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='blue')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
# ax.set_xticks(data)

# Dont Allow

data = np.array([-1, 0.000000, 1, 2, 3, 4, 5, 6])

y = np.array([0.405555, 0.809971, 0.967395, 0.997175, 0.999903, 1.000000, ])

yn = np.insert(y, 0, 0)

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hlines.html
ax.hlines(y=yn, xmin=data[:-1], xmax=data[1:],
          color='orange', zorder=1, label='(a,b-1)')

# https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.vlines.html
ax.vlines(x=data[1:-1], ymin=yn[:-1], ymax=yn[1:], color='orange',
          linestyle='dashed', zorder=1)

ax.scatter(data[1:-1], y, color='orange', s=18, zorder=2)
ax.scatter(data[1:-1], yn[:-1], color='white', s=18, zorder=2,
           edgecolor='orange')
ax.grid(False)
ax.set_xlim(data[0], data[-1])
ax.set_ylim([-0.01, 1.01])
ax.legend(loc=4)
ax.set_xlabel('Num Success')
ax.set_ylabel('Cumulative Probability')

full_data = [0.000000, 1, 2, 3, 4, 5, ]

ax.set_xticks(full_data)
ax.set_xticklabels(ax.get_xticks(), rotation=90)
plt.text(3.5, 0.5, '$a = 3, $b= 3', fontsize=12)
