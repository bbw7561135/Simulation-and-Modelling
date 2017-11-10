#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 09:58:32 2017

@author: saultyevil
"""

import numpy as np


def eq_circle(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def mc_int_circle(f, R, N=10000):
    """
    Monte Carlo integration to find the area of a circle. Returns the
    approximate area of a circle. Assumes the circle is at the origin.

    Parameters
    -----------
    radius: number; the radius of the circle
    """

    x = R * np.random.rand(N)
    y = R * np.random.rand(N)

    area_of_sq = (2 * R) ** 2

    k = np.sum(f(x, y) <= R)

    return area_of_sq * k / N


def eq_sphere(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def mc_int_sphere(f, R, N=10000):
    """
    Monte Carlo integration to find the area of a circle. Returns the
    approximate area of a circle. Assumes the circle is at the origin.

    Parameters
    -----------
    radius: number; the radius of the circle
    """

    x = R * np.random.rand(N)
    y = R * np.random.rand(N)
    z = R * np.random.rand(N)

    volume_of_cube = (2 * R) ** 3

    k = np.sum(f(x, y, z) <= R)

    return volume_of_cube * k / N

print(mc_int_circle(eq_circle, 1))
print(mc_int_sphere(eq_sphere, 1))
