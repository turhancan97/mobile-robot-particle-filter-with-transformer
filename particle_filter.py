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


fig, ax = plt.subplots()  # for animation
robot = Robot()  # call for robot
robot.set_pose(50, 50, 0)  # set the initial position
particles = [Robot() for _ in range(N)]  # set N particles randomly
weights = [0 for _ in range(N)]  # set weights with zero values
# For animation
(robo_dot,) = ax.plot(0, 0, "ks", label="Robot")
sense_lines = [ax.plot([0, 0], [0, 0], "k--") for _ in landmarks]
particle_dots = [ax.plot(0, 0, ".") for _ in range(N)]


def animate(_):
    global particles, weights, robot
    w = []
    rand_turn = random.uniform(0, 0.1)  # Random orientation
    rand_forward = random.uniform(0, 1)  # Random forward
    robot = robot.move(
        rand_turn, rand_forward
    )  # robot moves with random from rand_turn and rand_forward
    Z = robot.sense()  # Calculate eucledian distance from landmarks
    robo_dot.set_data(robot.x, robot.y)  # set robots position
    for i in range(N):  # measurement from particle filters
        p = particles[i].move(rand_turn, rand_forward)
        particles[i] = p
        weights[i] = p.measurement_prob(Z)
        particle_dots[i][0].set_data(p.x, p.y)
    norm = sum(weights)
    weights = [w / norm for w in weights]
    particles = resample(particles, weights)
    for i in range(len(landmarks)):  # visualization
        sense_lines[i][0].set_data(
            [robot.x, landmarks[i][0]], [robot.y, landmarks[i][1]]
        )
        sense_lines[i][0].set_color("blue")
        sense_lines[i][0].set_linewidth(0.8)
        sense_lines[i][0].set_alpha(0.5)
    return robo_dot, particle_dots, sense_lines


def init():
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Particle Filter Algorithm in 2D Environment")
    for lm in landmarks:
        ax.plot(lm[0], lm[1], "g*")
    ax.legend()
    return (
        robo_dot,
        particle_dots,
    )


anim = animation.FuncAnimation(fig, animate, 250, interval=100, init_func=init)
plt.show()
# anim.save("../../docs/images/particle-localization_2D.gif", writer="imagemagick")
