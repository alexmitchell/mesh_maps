#!/usr/bin/env python3

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

# Model runs
n_steps = 100

# Define map boundaries
x_min, x_max = 0, 10
y_min, y_max = 0, 10

width = x_max - x_min
height = y_max - y_min

def get_border_points():
    border_x = np.arange(x_min, x_max+1) # Full width
    #border_y = np.arange(y_min+1, y_max) # Height between top and bottom
    border_y = np.arange(y_min, y_max+1) # Height between top and bottom
    px, py = np.meshgrid(border_x, border_y)

    edges = []
    for y_lim in y_min, y_max:
        zero_y = np.zeros_like(border_x, dtype=np.float)
        edges.append(np.vstack((border_x, zero_y + y_lim, zero_y)).T)

    for x_lim in x_min, x_max:
        zero_x = np.zeros_like(border_y, dtype=np.float)
        edges.append(np.vstack((zero_x + x_lim, border_y, zero_x)).T)

    return np.concatenate(edges)

def get_random_points(n_rand_pts=100):
    n_rand_pts = 100
    rand_p = np.random.rand(n_rand_pts, 3)
    rand_p[:, 0] *= width
    rand_p[:, 1] *= height
    rand_p[:,2] = 0

    return rand_p



border_p = get_border_points()
rand_p = get_random_points()
points_3D = np.concatenate((border_p, rand_p))

p2D = points_3D[:, 0:2]
x, y = p2D.T

triangles = Delaunay(p2D)

plt.triplot(x, y, triangles.simplices.copy())
#plt.scatter(x, y)
plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))
plt.show()



