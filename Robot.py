
from math import sqrt, pi, cos, sin, exp
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 1000  # Number of particles
world_size = 100  # Dimension of the map
landmarks = [
    [10, 10], [10, 20], [10, 30], [10, 40], [10, 50],
    [10, 60], [10, 70], [10, 80], [10, 90], [90, 90],
    [90, 80], [90, 70], [90, 60], [90, 50], [90, 40],
    [90, 30], [90, 20], [90, 10]
]  # location of the landmarks


class Robot:
    def __init__(self):  # initialize the robot's parameters
        self.x = world_size * random.random()
        self.y = world_size * random.random()
        self.yaw = 2 * pi * random.random()
        self.forward_noise = 0.1
        self.turn_noise = 0.1
        self.sense_noise = 5

    def set_pose(
        self, new_x, new_y, new_yaw
    ):  # manual values of position and orientation
        self.x = new_x % world_size
        self.y = new_y % world_size
        self.yaw = new_yaw % (2 * pi)

    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):  # set the noise value
        self.forward_noise = new_f_noise
        self.turn_noise = new_t_noise
        self.sense_noise = new_s_noise

    def sense(self):  # distance from the landmarks
        Z = []
        for lm in landmarks:
            dist = sqrt((self.x - lm[0]) ** 2 + (self.y - lm[1]) ** 2)
            dist += random.gauss(0, self.sense_noise)
            Z.append(dist)
        return Z

    def Gaussian(self, mean, var, x):
        return exp(-((mean - x) ** 2) / (2 * var**2)) / sqrt(2 * pi * var**2)

    def measurement_prob(self, measurements):  # measurement for the each particle
        prob = 1.0
        for lm, meas in zip(landmarks, measurements):
            dist = sqrt((self.x - lm[0]) ** 2 + (self.y - lm[1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, meas)
        return prob

    def move(self, turn, forward):  # movemet of the robot
        yaw = self.yaw + turn + random.gauss(0, self.turn_noise)
        yaw = yaw % (2 * pi)

        dist = forward + random.gauss(0, self.forward_noise)
        x = self.x + dist * cos(yaw)
        y = self.y + dist * sin(yaw)
        x = x % world_size
        y = y % world_size

        p = Robot()
        p.set_pose(x, y, yaw)
        p.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return p

    def __repr__(self):
        return "[x = %.6f y = %.6f yaw = %.6f]" % (self.x, self.y, self.yaw)


def resample(p, w):
    new_p = []
    index = int(N * random.random())
    beta = 0.0
    mw = max(w)
    # print(w)
    for i in range(N):
        beta += 2 * mw * random.random()
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        new_p.append(p[index])
    return new_p
