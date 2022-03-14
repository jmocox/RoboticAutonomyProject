import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')


def distance(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def make_ball_gaussian(n=100, center=(0.0, 0.0, 0.0), center_std=0.1, radius=1.0):
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
    # ax.plot(points[0], points[1], points[2], 'bo')
    # ax.plot(thing[0], thing[1], thing[2], 'r+')
    # plt.show()
    return points


def get_facing_points(px, py, pz, center=(0.0, 0.0, 0.0), opposite=False):
    center_distance = distance(center[0], center[1], center[2])
    center = np.asarray(center)
    center_normalized = center / np.linalg.norm(center)

    facing_x, facing_y, facing_z = [], [], []
    for i in range(len(px)):
        point_distance = distance(px[i], py[i], pz[i])
        point = np.array([px[i], py[i], pz[i]])
        point_normalized = point / np.linalg.norm(point)
        dot = np.dot(center_normalized, point_normalized)
        if opposite:
            if center_distance < dot * point_distance:
                facing_x.append(px[i])
                facing_y.append(py[i])
                facing_z.append(pz[i])
        else:
            if center_distance > dot * point_distance:
                facing_x.append(px[i])
                facing_y.append(py[i])
                facing_z.append(pz[i])

    return facing_x, facing_y, facing_z


def select(points, n):
    selection = np.random.randint(0, len(points[0]), n)
    ch_x = np.asarray([points[0][s] for s in selection])
    ch_y = np.asarray([points[1][s] for s in selection])
    ch_z = np.asarray([points[2][s] for s in selection])

    return ch_x, ch_y, ch_z


x, y, z, r = 2, 1, 1, 0.15

s_x, s_y, s_z = make_ball_gaussian(n=100, center=(x, y, z), center_std=0.02, radius=r)
facing_x, facing_y, facing_z = get_facing_points(s_x, s_y, s_z, center=(x, y, z))
# o_x, o_y, o_z = get_facing_points(s_x, s_y, s_z, center=(x, y, z), opposite=True)
# o_center = [np.average(o_x), np.average(o_y), np.average(o_z)]

occ_x, occ_y, occ_z = select((facing_x, facing_y, facing_z), 1)
for i in range(len(facing_x)):
    if facing_x[i] == occ_x[0] and facing_y[i] == occ_y[0] and facing_z[i] == occ_z[0]:
        continue
    d = distance(occ_x[0] - facing_x[i], occ_y[0] - facing_y[i], occ_z[0] - facing_z[i])
    print(d)
    if 0.15 > d:
        print('asdf')
        occ_x = np.append(occ_x, facing_x[i])
        occ_y = np.append(occ_y, facing_y[i])
        occ_z = np.append(occ_z, facing_z[i])
print(occ_x)

guess_center = np.asarray([np.average(facing_x), np.average(facing_y), np.average(facing_z)])
guess_occ_center = np.asarray([np.average(occ_x), np.average(occ_y), np.average(occ_z)])
delta_center = (guess_center * (r / 2)) / np.linalg.norm(guess_center)
delta_occ_center = (guess_occ_center * (r / 2)) / np.linalg.norm(guess_occ_center)
guess_center += delta_center
guess_occ_center += delta_occ_center
# ch_x, ch_y, ch_z = select((facing_x, facing_y, facing_z), 3)
de_x, de_y, de_z = [], [], []
for i in range(len(facing_x)):
    dx = facing_x[i] - guess_center[0]
    dy = facing_y[i] - guess_center[1]
    dz = facing_z[i] - guess_center[2]
    est_r = distance(dx, dy, dz)
    delta_r = est_r - r
    dv = np.asarray([dx, dy, dz])
    dv = (dv * delta_r) / np.linalg.norm(dv)
    dv += guess_center
    # print(dv)
    de_x.append(dv[0])
    de_y.append(dv[1])
    de_z.append(dv[2])

second_guess_occ_center = guess_occ_center
for i in range(5):
    deo_x, deo_y, deo_z = [], [], []
    for i in range(len(occ_x)):
        dx = occ_x[i] - guess_center[0]
        dy = occ_y[i] - guess_center[1]
        dz = occ_z[i] - guess_center[2]
        est_r = distance(dx, dy, dz)
        delta_r = est_r - r
        dv = np.asarray([dx, dy, dz])
        dv = (dv * delta_r) / np.linalg.norm(dv)
        dv += second_guess_occ_center
        # print(dv)
        deo_x.append(dv[0])
        deo_y.append(dv[1])
        deo_z.append(dv[2])

    second_guess_occ_center = np.asarray([np.average(deo_x), np.average(deo_y), np.average(deo_z)])
    ax.plot(deo_x, deo_y, deo_z, 'm+')
    ax.plot(second_guess_occ_center[0], second_guess_occ_center[1], second_guess_occ_center[2], 'm*')

second_guess_center = np.asarray([np.average(de_x), np.average(de_y), np.average(de_z)])


ax.plot(s_x, s_y, s_z, 'r+')
ax.plot(facing_x, facing_y, facing_z, 'b+')
ax.plot(de_x, de_y, de_z, 'y+')
# ax.plot(deo_x, deo_y, deo_z, 'm+')
ax.plot(occ_x, occ_y, occ_z, 'mo')
# ax.plot(guess_center[0], guess_center[1], guess_center[2], 'ks')
ax.plot(second_guess_center[0], second_guess_center[1], second_guess_center[2], 'ks')
ax.plot(second_guess_occ_center[0], second_guess_occ_center[1], second_guess_occ_center[2], 'ms')
# ax.plot(second_guess_occ_center[0], second_guess_occ_center[1], second_guess_occ_center[2], 'ms')
plt.show()

# print(np.average(ds))
