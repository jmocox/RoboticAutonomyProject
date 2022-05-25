from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import scipy.stats

from scipy.spatial.transform import Rotation

# t265/odom/sample nav_msgs/msg/Odometry
# ball_belief geometry_msgs/msg/PoseWithCovarianceStamped
# covariance_trace std_msgs/msg/Float32
# entropy_trace std_msgs/msg/Float32
# kalman_gain_trace std_msgs/msg/Float32

bags = [
    '2022-05-04-10-44-27.bag',
    '2022-05-04-10-56-28.bag',
    '2022-05-04-11-00-59.bag',
    '2022-05-04-11-09-17.bag',
]

# create reader instance
with Reader(f'/Users/tylerhaden/Downloads/bagel_bag/{bags[0]}') as reader:
    # for connection in reader.connections.values():
    #     print(connection.topic, connection.msgtype)



    x, y, z, t = [], [], [], []
    # connections = [x for x in reader.connections.values() if x.topic == '/move_base/TrajectoryPlannerROS/global_plan']
    connections = [x for x in reader.connections.values() if x.topic == 'kalman_gain_trace']
    i = 0
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        i += 1
        # if i % 5 != 0:
        #     continue
        msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
        x.append(msg.data)
        # x.append(msg.pose.pose.position.x)
        # y.append(msg.pose.pose.position.y)
        # rot = Rotation.from_quat([
        #     msg.pose.pose.orientation.x,
        #     msg.pose.pose.orientation.y,
        #     msg.pose.pose.orientation.z,
        #     msg.pose.pose.orientation.w,
        # ])
        # rot_euler = rot.as_euler('xyz')
        # #print(msg.header.stamp)
        # z.append(rot_euler[2])
        #t.append(msg.header.stamp.sec + (msg.header.stamp.nanosec * 10e-10))

        #print(msg.transforms[0].transform)

    print(x, t)

    c = [[i / len(x), i / len(x), i / len(x)] for i in range(len(x))]

    # fig, (ax0) = plt.subplots(1, 1)
    # ax0.scatter(np.asarray(y), x, c=c, edgecolors='black')
    # ax0.set_ylim((1.015, 1.03))
    # ax0.set_xlim((-0.095, -0.08))
    # ax0.set_ylabel('X coord (meters)')
    # ax0.set_xlabel('Y coord (meters)')
    # ax0.set_title('Pose over Time')

    # t0 = t[0]
    # t = [tt - t0 for tt in t]

    # fig, (ax0) = plt.subplots(1, 1)
    # # ax0.scatter(t, z, c=c, edgecolors='black')
    # ax0.plot(t, z, c='black')
    # # ax0.set_ylim((1.015, 1.03))
    # # ax0.set_xlim((-0.095, -0.08))
    # ax0.set_xlabel('Time Elapsed (seconds)')
    # ax0.set_ylabel('Yaw (radians)')
    # ax0.set_title('Yaw over Time')

    plt.plot(x)
    plt.show()
    quit()

    fig, (ax0) = plt.subplots(1, 1)
    d = []
    p = []
    x9, y9 = x[-1], y[-1]
    gaus = scipy.stats.norm(1, 0.7)
    for i in range(len(x)):
        d.append(np.sqrt(((y[i] -y9) ** 2) + ((x[i] - x9) ** 2)) + 1)
        p.append(gaus.pdf(d[-1]))

    color = 'blue'
    ax0.set_ylabel('Distance From Ball (meters)', color=color)
    ax0.plot(t, d, c=color)
    ax0.tick_params(axis='y', labelcolor=color)

    ax2 = ax0.twinx()

    color = 'black'
    ax2.set_ylabel('Observation Likelihood', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, p, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # ax0.plot(t, d, c='blue')
    #ax0.plot(t, p, c='black')
    # ax0.set_ylim((1.015, 1.03))
    # ax0.set_xlim((-0.095, -0.08))
    # ax0.set_ylabel('X coord (meters)')
    ax0.set_xlabel('Time Elapsed (seconds)')
    ax0.set_title('Modeled Observation Likelihood')

    def gauss2d(x, y, m, s):
        det = np.linalg.det(s)
        inv = np.linalg.inv(s)
        nx, ny = x.shape
        k = 2 * np.pi * np.sqrt(det)
        e1 = x - m[0]
        e2 = y - m[1]
        M = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                e = np.asarray([e1[i, j], e2[i, j]])
                M[i, j] = np.matmul(np.matmul(-0.5 * e.T, inv), e)

        return np.exp(M) / k


    # fig = plt.figure(figsize=(10, 8), dpi=80)
    # ax = plt.axes(projection='3d')
    #
    # side = 6
    # x = np.linspace(-side, side, 30)
    # y = np.linspace(-side, side, 30)
    #
    # X, Y = np.meshgrid(x, y)
    # m = [3.5, 0]
    # sss = 0.7
    # s = [[1 * sss, 0], [0, 2 * sss]]
    # Z = gauss2d(X, Y, m, s)
    #
    # cont = ax.contour3D(X, Y, Z, 70, cmap='plasma')
    # ax.set_xlabel('X coord (meters)')
    # ax.set_ylabel('Y coord (meters)')
    # ax.set_zlabel('Detection Likelihood')
    # ax.set_title('Likelihood of Detection from Robot\'s Pose')
    #
    # rad = 1
    # ss = np.asarray([[0, -rad], [-2*rad, -rad], [-2*rad, rad], [0, rad]]).T
    #
    # plt.plot(
    #     np.concatenate((ss[0, :], [ss[0, 0]])),
    #     np.concatenate((ss[1, :], [ss[1, 0]]))
    # )





    plt.show()

