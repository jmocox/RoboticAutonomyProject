from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

import json
from matplotlib.ticker import FormatStrFormatter
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

# fig, axs = plt.subplots(1, 4)
for bag_i in [1]:
    with Reader(f'/Users/tylerhaden/Downloads/bagel_bag/{bags[bag_i]}') as reader:
        # for connection in reader.connections.values():
        #     print(connection.topic, connection.msgtype)

        robot_x, robot_y, ball_x, ball_y = [], [], [], []
        time_start, time_end = [], []
        cov_trace, kal_trace, entropy = [], [], []

        odom_con = [x for x in reader.connections.values() if x.topic == 't265/odom/sample']
        odom_len = len(list(reader.messages(connections=odom_con)))
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=odom_con)):
            msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)

            if i == 0:
                time_start.append(msg.header.stamp.sec + (msg.header.stamp.nanosec * 10e-10))
            elif i == odom_len - 1:
                time_end.append(msg.header.stamp.sec + (msg.header.stamp.nanosec * 10e-10))

            robot_x.append(msg.pose.pose.position.x)
            robot_y.append(msg.pose.pose.position.y)

        print(f'Odom: sample#={len(robot_x)}, time_elapsed={(time_end[-1] - time_start[-1]) / 60}s')

        ball_con = [x for x in reader.connections.values() if x.topic == 'ball_belief']
        ball_len = len(list(reader.messages(connections=odom_con)))
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=ball_con)):
            msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)

            if i == 0:
                time_start.append(msg.header.stamp.sec + (msg.header.stamp.nanosec * 10e-10))
            elif i == odom_len - 1:
                time_end.append(msg.header.stamp.sec + (msg.header.stamp.nanosec * 10e-10))

            ball_x.append(msg.pose.pose.position.x)
            ball_y.append(msg.pose.pose.position.y)

        print(f'Ball: sample#={len(ball_x)}, time_elapsed={(time_end[-1] - time_start[-1]) / 60}s')

        cov_con = [x for x in reader.connections.values() if x.topic == 'covariance_trace']
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=cov_con)):
            msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            cov_trace.append(msg.data)
        print(f'Cov: sample#={len(cov_trace)}')

        kal_con = [x for x in reader.connections.values() if x.topic == 'kalman_gain_trace']
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=kal_con)):
            msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            kal_trace.append(msg.data)
        print(f'Kal: sample#={len(kal_trace)}')

        entropy_con = [x for x in reader.connections.values() if x.topic == 'entropy_trace']
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=entropy_con)):
            msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
            entropy.append(msg.data)
        print(f'Ent: sample#={len(entropy)}')

        elapsed_time = np.average(time_end) - np.average(time_start)

        # # cov, kal, ent over time
        # fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        #
        # t = np.arange(0, elapsed_time, elapsed_time / len(entropy))
        # if bag_i in [0, 1, 2, 3]:
        #     start = [0, 95, 75, 25][bag_i]
        #     stop = [175, 190, len(cov_trace), 80][bag_i]
        #     cov_trace = cov_trace[start:stop]
        #     kal_trace = kal_trace[start:stop]
        #     entropy = entropy[start:stop]
        #     t = t[start:stop]
        #     t = [_t - t[0] for _t in t]
        #
        # ax0.plot(t, cov_trace)
        # ax1.plot(t, kal_trace)
        # ax2.plot(t, entropy)
        #
        # ax0.set_ylabel('Covariance Tr')
        # ax1.set_ylabel('Kalman Gain Tr')
        # ax2.set_ylabel('Entropy (bits)')
        #
        # ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #
        # ax2.set_xlabel('Elapsed Time (seconds)')
        # ax0.set_title('Effect of Ball Uncertainty on Kalman Gain and Entropy')
        #
        # plt.show()
        # quit()

        # # pose over time
        # if bag_i in [0, 1, 2, 3]:
        #     start = [0, 25000, 30000, 6000][bag_i]
        #     stop = [27000, 30000, 46320, 17554][bag_i]
        #     robot_x = robot_x[start:stop]
        #     robot_y = robot_y[start:stop]
        #
        # axs[bag_i].plot(-np.asarray(robot_y), robot_x)
        # axs[bag_i].set_ylim(-0.5, 6)
        # axs[bag_i].set_xlim(-2, 2)

        # # # ball location over time
        if bag_i in [0, 1, 2, 3]:
            og_len = len(ball_x)
            start = [125, 95, 75, 45][bag_i]
            stop = [len(ball_x) - 100, len(ball_x) - 62, len(ball_x), len(ball_x)][bag_i]
            ball_x = ball_x[start:stop]
            ball_y = ball_y[start:stop]

            robot_x = robot_x[int(start * len(robot_x) / og_len):int(stop * len(robot_x) / og_len)]
            robot_y = robot_y[int(start * len(robot_y) / og_len):int(stop * len(robot_y) / og_len)]

        goal_x, goal_y = [], []
        for i in range(len(ball_x)):
            print(int(i * len(robot_x) / len(ball_x)), len(robot_x))
            rx = robot_x[int(i * len(robot_x) / len(ball_x))]
            ry = robot_y[int(i * len(robot_y) / len(ball_y))]
            distance = np.sqrt(((rx - ball_x[i]) ** 2) + ((ry - ball_y[i]) ** 2))
            prop = 0.6 / distance
            goal_x.append(ball_x[i] - (prop * rx))
            goal_y.append(ball_y[i] - (prop * ry))
            print(goal_x[-1])
            print(goal_y[-1])

        # c = [[i / len(goal_x), i / len(goal_x), i / len(goal_x)] for i in range(len(goal_x))]
        fig, axs = plt.subplots()
        axs.scatter(-np.asarray(goal_y), goal_x, c='green', edgecolors='black')
        axs.set_ylim(-0.5, 5)
        axs.set_xlim(-3, 3)
        axs.set_xlabel('Goal Position Y (m)')
        axs.set_ylabel('Goal Position X (m)')

        fig.suptitle(f'Control Goal Position - Round {["1", "2a", "2b", "3"][bag_i]}')

        plt.show()

        # # # ball location over time
        # if bag_i in [0, 1, 2, 3]:
        #     start = [125, 95, 75, 45][bag_i]
        #     stop = [len(ball_x) - 100, len(ball_x), len(ball_x), len(ball_x)][bag_i]
        #     ball_x = ball_x[start:stop]
        #     ball_y = ball_y[start:stop]
        #
        # if bag_i == 0:
        #     ball_x = ball_x[:15] + ball_x[22:]
        #     ball_y = ball_y[:15] + ball_y[22:]
        #
        #     new_time_elapsed = len(ball_x)
        #     new_vel = 3 / new_time_elapsed
        #     print(new_vel)
        #
        #     for i in range(1, len(ball_x)):
        #         ball_x[i] += new_vel * i
        #
        #
        # c = [[i / len(ball_x), i / len(ball_x), i / len(ball_x)] for i in range(len(ball_x))]
        # print(len(c))
        # fig, axs = plt.subplots()
        # axs.scatter(-np.asarray(ball_y), ball_x, c=c, edgecolors='black')
        # axs.set_ylim(-0.5, 6)
        # axs.set_xlim(-3, 3)
        # axs.set_xlabel('Ball Position Y (m)')
        # axs.set_ylabel('Ball Position X (m)')
        # axs.set_title(f'Estimated Ball Position - Round {["1", "2a", "2b", "3"][bag_i]}')
        # plt.show()

# axs[0].set_xlabel('Position Y (m)\nRound 1')
# axs[1].set_xlabel('Position Y (m)\nRound 2a')
# axs[2].set_xlabel('Position Y (m)\nRound 2b')
# axs[3].set_xlabel('Position Y (m)\nRound 3')
# axs[0].set_ylabel('Position X (m)')
# fig.suptitle('Trajectories of Robot')
# plt.show()

# axs[0].set_xlabel('Position Y (m)\nRound 1')
# axs[1].set_xlabel('Position Y (m)\nRound 2a')
# axs[2].set_xlabel('Position Y (m)\nRound 2b')
# axs[3].set_xlabel('Position Y (m)\nRound 3')
# axs[0].set_ylabel('Position X (m)')
# fig.suptitle('Estimated Ball Positions')
# plt.show()
