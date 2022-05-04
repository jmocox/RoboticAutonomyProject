#!/usr/bin/env python

import numpy as np
import time
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
#from move_base_msgs.msg import MoveBaseGoal 
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry
from scipy.stats import multivariate_normal, entropy
from std_msgs.msg import Float32
from tf.transformations import quaternion_from_euler


class EKF():
    def __init__(self, nk, dt, X, U, color):
        self.start = time.time()
        self.nk = nk
        self.dt = dt
        self.X = X
        #self.X_predicted_steps = zeros(nk)
        self.U = U
        self.A = np.identity(3)
        self.C = np.identity(3)
        self.x_km2_km1 = np.zeros(3)

        self.Sigma_init = np.array(
            [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.1]])  # <--------<< Initialize correction covariance
        self.sigma_measure = np.array([[0.05, 0, 0], [0, 0.05, 0],
                                       [0, 0, 0.1]])  # <--------<< Should be updated with variance from the measurement
        self.sigma_motion = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self.KalGain = np.random.rand(3, 3)  # <--------<< Initialize Kalman Gain

        self.z_k = None
        self.z_height_k = None
        self.z_height_variance = None
        rospy.Subscriber(
            '/ballxyz/' + color,
            PoseWithCovarianceStamped,
            self.measurement_cb
        )

        self.Sx_k_k = self.Sigma_init

        self.last_recorded_positions = [[0], [0]]
        self.steps_ball_unfound = 0

        self.belief_pub = rospy.Publisher('/ball_belief', PoseWithCovarianceStamped, queue_size=5)
        self.future_pub = rospy.Publisher('/future', MarkerArray, queue_size=5)
        self.kalman_gain_pub = rospy.Publisher('/kalman_gain_trace', Float32, queue_size=5)
        self.covariance_pub = rospy.Publisher('/covariance_trace', Float32, queue_size=5)
        self.entropy_pub = rospy.Publisher('/entropy_trace', Float32, queue_size=5)
        #self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)\

        self.kalman_gain_det = 0
        self.covariance_det = 0
        self.covariance = np.zeros((6, 6))

    def prediction(self, x_km1_km1, Sigma_km1_km1):

        # ADDED THIS BIT _______________
        # Calculate nk time steps ahead
        x_predictions = []

        ## Only predict steps while ball is observed
        for time_steps in range(self.nk):
        
            only_dot = True
            if only_dot:
        
                x_mean_kn_km1 = self.dotX(x_km1_km1, time_steps + 1)
            else:
            
                x_mean_kn_km1 = self.dotX(
                    np.matmul(np.linalg.matrix_power(self.A, time_steps + 1), x_km1_km1), 
                    time_steps + 1
                )
            
            x_predictions.append(x_mean_kn_km1)
        #_______________________________
        # Defining A utilizing Jacobian... A = (I+Jacobian)
        """
        jacobian = np.asarray([
            [x_km1_km1[0]-self.x_km2_km1[0], 0, 0],
            [0, x_km1_km1[1]-self.x_km2_km1[1], 0],
            [-1*np.sin(x_km1_km1[2]-self.x_km2_km1[2]),np.cos(x_km1_km1[2]-self.x_km2_km1[2]),x_km1_km1[2]-self.x_km2_km1[2]]
        ])
        """
        x_mean_k_km1 = np.matmul(self.A, x_km1_km1)
        Jacobian = np.asarray([
            [x_mean_k_km1[0] - x_km1_km1[0], 0, 0],
            [0, x_mean_k_km1[1] - x_km1_km1[1], 0],
            [-1*np.sin(x_mean_k_km1[2]-x_km1_km1[2]),np.cos(x_mean_k_km1[2]-x_km1_km1[2]),x_mean_k_km1[2] - x_km1_km1[2]]
        ])
        
        
        """
        Jacobian = np.asarray([
            [x_mean_k_km1[0] - x_km1_km1[0], 0, 0],
            [0, x_mean_k_km1[1] - x_km1_km1[1], 0],
            [0, 0,x_mean_k_km1[2] - x_km1_km1[2]]
        ])
        """
        """
        Jacobian = np.asarray([
            [x_km1_km1[0] - self.x_km2_km1[0], 0, 0],
            [0, x_km1_km1[1] - self.x_km2_km1[1], 0],
            [0, 0,x_km1_km1[2] - self.x_km2_km1[2]]
        ])
        """
        
        Jacobian = np.divide(Jacobian, self.dt)
        self.A = np.add(np.identity(3), Jacobian)
        
        # print(self.A, self.dt)
        
        #x_mean_k_km1 = np.matmul(self.A, x_km1_km1)

        Sigma_k_km1 = np.matmul(np.matmul(self.A, Sigma_km1_km1), self.A.T) + self.sigma_motion

        self.x_km2_km1 = x_km1_km1

        return x_predictions, x_mean_k_km1, Sigma_k_km1

    def correction(self, x_mean_k_km1, Sx_k_km1, kalman_gain):
        x_mean_k_k = x_mean_k_km1 + np.matmul(kalman_gain, self.z_k - x_mean_k_km1)

        inner = np.identity(3) - np.matmul(kalman_gain, self.C)
        Sigma_k_k = np.matmul(inner, Sx_k_km1)

        return x_mean_k_k, Sigma_k_k

    def compute_gain(self, Sx_k_km1):
        inner = np.matmul(np.matmul(self.C, Sx_k_km1), self.C.T) + self.sigma_measure
        return np.matmul(np.matmul(Sx_k_km1, self.C.T), np.linalg.inv(inner))

    def update(self):
        if self.z_k is None:
            return
        ball_detected = self.z_k[0] != 0 and self.z_k[1] != 0
        ## Publish to nav stack position of ball when ball is observed    
        if ball_detected:
            self.steps_ball_unfound = 0
            # Output matrix of means of all predicted steps
            x_prediction_means, x_mean_k_km1, Sx_k_km1 = self.prediction(self.X, self.Sx_k_k)
            print('predict', x_mean_k_km1)
            #x_mean_k_km1 = x_prediction_means[0] #Changed this line too
            kalman_gain = self.compute_gain(Sx_k_km1)
            self.kalman_gain_det = np.trace(kalman_gain)
            self.X, self.Sx_k_k = self.correction(x_mean_k_km1, Sx_k_km1, kalman_gain)
            self.X[2] = np.arctan2(np.sin(self.X[2]), np.cos(self.X[2]))
            print('correct', self.z_k, self.X)
            """for i in range(nk):
                self.X_predicted_steps[i],Sx_k_k_step = self.correction(s_prediction_means[i], Sx_k_km1, kalman_gain)"""
            #print(self.X, self.Sx_k_k)
            xs, ys = [], []
            for x, y, z in x_prediction_means:
                xs.append(x)
                ys.append(y)
            self.last_recorded_positions = [xs, ys]
            print('Belief Pose => ', self.X)
            print('Belief Covariance => ', self.Sx_k_k)
            self.publish_future(xs, ys, self.z_height_k)
        ## Publish to navstack predicted position of ball, K amount of steps ahead    
        if not ball_detected:
            #Count how many time steps ball isn't observed
            self.steps_ball_unfound += 1

            print(self.steps_ball_unfound,  len(self.last_recorded_positions[0]))
            if self.steps_ball_unfound < len(self.last_recorded_positions[0]):
                future_position_index = self.steps_ball_unfound
            else:
                future_position_index = -1
            self.X = [
                self.last_recorded_positions[0][future_position_index],
                self.last_recorded_positions[1][future_position_index], 
                self.X[2]
            ]
            self.Sx_k_k *= 1.05
        self.publish_ball_belief()
    
    def dotX(self, x, time_step):
        x_dot = np.asarray([
            self.U[0] * np.cos(x[2]) * self.dt,
            self.U[0] * np.sin(x[2]) * self.dt,
            self.U[1] * self.dt,
        ])

        #return (x + x_dot), (x + self.nk * x_dot)
        return (x + time_step * x_dot)

    #def odom_cb(self, pose_of_bot):


    def measurement_cb(self, pwcs):
        if self.z_k is None:
            theta = 0
        else:
            theta = np.arctan2(pwcs.pose.pose.position.y - self.z_k[1], pwcs.pose.pose.position.x - self.z_k[0])
        ## Current pwcs pose is in reference of the sensor frame, pose needs to be adjsuted for global (add robot motion)
        self.z_k = np.asarray([pwcs.pose.pose.position.x, pwcs.pose.pose.position.y, theta])

        self.sigma_measure = np.asarray([
            [pwcs.pose.covariance[0], 0, 0],
            [0, pwcs.pose.covariance[7], 0],
            [0, 0, 0.1],
        ])

        self.z_height_k = pwcs.pose.pose.position.z
        self.z_height_variance = min(pwcs.pose.covariance[14], .1)

    def publish_ball_belief(self):
        pwcs = PoseWithCovarianceStamped()
        pwcs.header.stamp = rospy.get_rostime()
        pwcs.header.frame_id = 'd400_link'

        x, y, z, w = quaternion_from_euler(0, 0, self.X[2])
        pwcs.pose.pose.orientation.x = x
        pwcs.pose.pose.orientation.y = y
        pwcs.pose.pose.orientation.z = z
        pwcs.pose.pose.orientation.w = w

        pwcs.pose.pose.position.x = self.X[0]
        pwcs.pose.pose.position.y = self.X[1]
        pwcs.pose.pose.position.z = self.z_height_k

        pwcs.pose.covariance = [
            self.Sx_k_k[0, 0], 0, 0, 0, 0, 0,
            0, self.Sx_k_k[1, 1], 0, 0, 0, 0,
            0, 0, self.z_height_variance, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]

        self.covariance = np.reshape(np.asarray(pwcs.pose.covariance), (6, 6))
        self.covariance_det = np.trace(self.covariance)

        self.belief_pub.publish(pwcs)
    
    def publish_future(self, xs, ys, z, diameter=0.15):

        markers = MarkerArray()
        markers.markers = []
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            
            marker = Marker()
            marker.header.frame_id = '/d400_link'
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.ns = 'FutureBalls'
            marker.id = i
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            marker.color.a = 0.5 + (len(xs) - i) * 0.5
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
            marker.pose.orientation.w = 1
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            
            markers.markers.append(marker)

        self.future_pub.publish(markers)

    def publish_entropy(self):
        x_var = self.covariance[0, 0]
        y_var = self.covariance[1, 1]

        cov = np.diag(np.array([x_var, y_var]))

        try:
            np.linalg.inv(cov)
        except:
            return

        x, y = np.mgrid[-10.0:10.0:100j, -10.0:10.0:100j]
        xy = np.column_stack([x.flat, y.flat])
        mu = np.array([0.0, 0.0])

        w = multivariate_normal.pdf(xy, mean=mu, cov=cov)
        e = entropy(w, base=2)

        print('entroy =', e)
        f32 = Float32()
        f32.data = e
        self.entropy_pub.publish(f32)





if __name__ == '__main__':
    rospy.init_node('EKF')

    # ---------------Define initial conditions --------------- #
    dt = .8  # Sampling duration (seconds)
    nk = 5  # Look ahead duration (seconds)

    # Initial State of the ball
    X = [0, 0, 0]

    # Control input, always assumed to be going straight at constant velocity
    U = [
        0.17,  # Forward velocity (meters / second)
        0  # Turning velocity (radians / second)
    ]

    color = rospy.get_param('/ball_color')
    extended_kalman_filter = EKF(nk, dt, X, U, color=color)

    hz = 1.0 / dt
    rate = rospy.Rate(hz)
    while not rospy.is_shutdown():

        extended_kalman_filter.update()
        f32 = Float32()
        f32.data = extended_kalman_filter.kalman_gain_det
        extended_kalman_filter.kalman_gain_pub.publish(f32)
        f320 = Float32()
        f320.data = extended_kalman_filter.covariance_det
        extended_kalman_filter.covariance_pub.publish(f320)

        if extended_kalman_filter.covariance is not None:
            extended_kalman_filter.publish_entropy()
        rate.sleep()
