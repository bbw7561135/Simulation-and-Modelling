#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:04:33 2017

@author: saultyevil
"""

import numpy as np


def f(x):
    return x * x


def mc_integrate_multid(f, R, num_dimen, N=100000):
    """
    Monte Carlo integration in arbitrary dimensions (read from the size of
    the domain): to be completed
    """

    # the domain is going to be a list of lists
    # volume is the volume of the box, i.e. the volume of the domain

    volume = (2 * R) ** num_dimen

    xs = np.zeros((num_dimen, N))
    for i in range(num_dimen):
        xs[i] = R * np.random.rand(N)

    k = 0
    for j in range(N):
        sum = 0
        for i in range(num_dimen):
            comp = xs[i][j]
            sum += f(comp)
        if sum <= R ** 2:
            k += 1

    return volume * k / N

print(mc_integrate_multid(f, 1, 3))
