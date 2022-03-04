import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')


def make_ball_gaussian(n=100, center=(0.0, 0.0, 0.0), center_std=0.1, radius=1.0, radius_std=0.1):
    points = np.random.randn(3, n)
    points /= np.linalg.norm(points, axis=0)
    points *= radius
    thing = points.copy()
    thing[0] += center[0]
    thing[1] += center[1]
    thing[2] += center[2]
    points[0] = np.random.normal(center[0] + points[0], center_std)
    points[1] = np.random.normal(center[1] + points[1], center_std)
    points[2] = np.random.normal(center[2] + points[2], center_std)
    ax.plot(points[0], points[1], points[2], 'bo')
    ax.plot(thing[0], thing[1], thing[2], 'r+')
    plt.show()
    return points


# print(make_ball_gaussian(n=1000, center=(1, 0, 1), center_std=0.01, radius=.150, radius_std=0.002))
print(make_ball_gaussian())
