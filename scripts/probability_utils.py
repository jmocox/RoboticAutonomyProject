import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np


def gauss1d(x, m, S):
    K = np.sqrt(2 * np.pi * S)
    M = (-(x - m) ** 2) / (2 * S)
    return np.exp(M) / K


def gauss2d(x, y, m, S):
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    nx, ny = x.shape

    K = 2 * np.pi * np.sqrt(det)
    e1 = x - m[0]
    e2 = y - m[1]

    M = np.ndarray((nx, ny))
    for i in range(nx):
        for j in range(ny):
            e = np.asarray([[e1[i, j]], [e2[i, j]]])
            M[i, j] = np.matmul(np.matmul(-0.5 * e.T, inv), e)

    return np.exp(M) / K


class Space2d:
    def __init__(self, xmin, xmax, nx, ymin, ymax, ny):
        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny

        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx

        self.ymin = ymin
        self.ymax = ymax
        self.ny = ny

        self.xi, self.yi = np.mgrid[xmin:xmax:nx * 1j, ymin:ymax:ny * 1j]


def gauss2d_test():
    xmin = -10
    xmax = 10
    nx = 100
    ymin = -8
    ymax = 8
    ny = 100
    X, Y = np.mgrid[xmin:xmax:nx * 1j, ymin:ymax:ny * 1j]

    Z = gauss2d(X, Y, np.asarray([2, 3]), np.asarray([[2.5, 1], [02.1, 2]]))

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def intFxWx(x, fx):
    return sum(fx) * (x[-1] - x[0]) / (len(x) - 1)


def intFxyWxy(x, y, fxy):
    Fy = np.ndarray(y.shape)
    for i in range(len(y)):
        Fy[i] = intFxWx(x, fxy[:, i])
    return Fy


def intF1xyzF2xyWxy(x, y, f1xyz, f2xy):
    Fz = np.ndarray((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            f1xy = f1xyz[i, j, :, :]
            fxyz = np.matmul(f1xy, f2xy)
            Fz[i, j] = intFxyWxy(x, y, fxyz)
    return Fz


def transform_p2d(x, a):
    x_t, y_t, th_t = a
    z_in = np.asarray([x[0], x[1], 1])

    T = np.asarray([
        [np.cos(th_t), -np.sin(th_t), x_t],
        [np.sin(th_t), np.cos(th_t), y_t],
        [0, 0, 1]
    ])

    z_out = np.matmul(T, z_in)

    return np.asarray([z_out[0], z_out[1]])


def hfov(di, pi):
    return np.matmul(di, np.sin(pi)), np.matmul(di, np.cos(pi))
def test():
    d_min = 1
    d_max = 5
    nd = 10
    angle_min = -1.2  # radians
    angle_max = 1.2  # radians
test()

class RBE1D:
    def __init__(self, xmin, xmax, nx):
        self.nx = nx
        self.dx = (xmax - xmin) / self.nx
        self.xmin = xmin + self.dx / 2
        self.xmax = xmax - self.dx / 2
        self.X = np.linspace(self.xmin, self.xmax, self.nx)

        self.belief = None
        self.prediction_posterior = None

        self.motion_model_mean = lambda x: x
        self.motion_model_S = 0

        self.sensors = []

    def predict(self, prior_probability, motion_model_probability):
        return self.intF1xzF2xWx(prior_probability, motion_model_probability)

    def correct(self, prior_probability, observation_likelihood):
        joint_probability = prior_probability * observation_likelihood
        return joint_probability / self.intFxWx(joint_probability)

    def intFxWx(self, fx):
        return sum(fx) * (self.X[-1] - self.X[0]) / (self.nx - 1)

    def intF1xzF2xWx(self, f1xz, f2x):
        d = []
        for i in range(self.nx):
            print(np.reshape(f1xz, (1, 100)).shape, f2x.shape)
            j = np.matmul(np.reshape(f1xz, (1, 100)), f2x)
            # j = f1xz[i] * f2x
            print('jj', j.shape)
            k = self.intFxWx(j)
            print('kk', k.shape)
            d.append(k)
            print('d', np.asarray(d).shape)
        return np.asarray(d)

    def set_motion_model(self, m_function, S):
        self.motion_model_mean = m_function
        self.motion_model_S = S

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def step(self, plt):
        if self.belief is None:
            initial_belief = np.ones(self.nx)
            for sensor in self.sensors:
                k = gauss1d(self.X, sensor.init_m, sensor.init_s)
                # plt.plot(self.X, k, 'g')
                initial_belief *= k

            self.belief = initial_belief / self.intFxWx(initial_belief)
            return

        # prediction
        prob_current_target_location = [gauss1d(self.X, self.motion_model_mean(x), self.motion_model_S) for x in self.X]
        prob_current_target_location = np.asarray(prob_current_target_location).T
        print(prob_current_target_location.shape, self.belief.shape)
        self.prediction_posterior = self.predict(self.belief, prob_current_target_location)
        print(self.prediction_posterior.shape)

        # correction
        likelilhood = np.ones(self.nx)
        for sensor in self.sensors:
            likelilhood *= gauss1d(self.X, sensor.get_next_reading() - sensor.error, sensor.noise)
        self.belief = self.correct(self.prediction_posterior, likelilhood)


class SensorNode:
    def __init__(self, values, x=0, error=0, noise=0, init=(0, 1)):
        self.values = values
        self.i = 0

        self.x = x
        self.error = error
        self.noise = noise

        self.init_m = init[0]
        self.init_s = init[1]

    def get_next_reading(self):
        reading = self.values[self.i]
        self.i += 1
        return reading
