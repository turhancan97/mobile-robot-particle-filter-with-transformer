import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt, pi, cos, sin, exp
from PIL import Image, ImageDraw

# Constants
N = 1000  # Number of particles
world_size = 100  # Dimension of the map
img_size = 256*2  # Image size for training data
alpha = 1.5  # Distance threshold in meters
beta = pi / 2  # Orientation threshold in radians

# Landmarks
landmarks = [
    [10, 10], [10, 20], [10, 30], [10, 40], [10, 50],
    [10, 60], [10, 70], [10, 80], [10, 90], [90, 90],
    [90, 80], [90, 70], [90, 60], [90, 50], [90, 40],
    [90, 30], [90, 20], [90, 10]
]

class Robot:
    def __init__(self):
        self.x = world_size * random.random()
        self.y = world_size * random.random()
        self.yaw = 2 * pi * random.random()
        self.forward_noise = 0.1
        self.turn_noise = 0.1
        self.sense_noise = 5

    def set_pose(self, new_x, new_y, new_yaw):
        self.x = new_x % world_size
        self.y = new_y % world_size
        self.yaw = new_yaw % (2 * pi)

    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        self.forward_noise = new_f_noise
        self.turn_noise = new_t_noise
        self.sense_noise = new_s_noise

    def sense(self):
        Z = []
        for lm in landmarks:
            dist = sqrt((self.x - lm[0]) ** 2 + (self.y - lm[1]) ** 2)
            dist += random.gauss(0, self.sense_noise)
            Z.append(dist)
        return Z

    def Gaussian(self, mean, var, x):
        return exp(-((mean - x) ** 2) / (2 * var**2)) / sqrt(2 * pi * var**2)

    def measurement_prob(self, measurements):
        prob = 1.0
        for lm, meas in zip(landmarks, measurements):
            dist = sqrt((self.x - lm[0]) ** 2 + (self.y - lm[1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, meas)
        return prob

    def move(self, turn, forward):
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
    for i in range(N):
        beta += 2 * mw * random.random()
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        new_p.append(p[index])
    return new_p

def generate_particle_image(particles):
    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    for p in particles:
        x = int(p.x / world_size * img_size)
        y = int(p.y / world_size * img_size)
        draw.point((x, y), fill=255)
    return np.array(img)

def estimate_pose(particles):
    x_est = np.mean([p.x for p in particles])
    y_est = np.mean([p.y for p in particles])
    yaw_est = np.mean([p.yaw for p in particles])
    return x_est, y_est, yaw_est

def label_data(x_GT, y_GT, theta_GT, x_PF, y_PF, theta_PF, alpha, beta):
    dist = sqrt((x_GT - x_PF) ** 2 + (y_GT - y_PF) ** 2)
    orient_diff = abs(theta_GT - theta_PF)
    if dist < alpha and orient_diff < beta:
        return 1  # Localized
    else:
        return 0  # Delocalized

def generate_training_data(robot, particles, num_samples, alpha, beta):
    data = []
    for _ in range(num_samples):
        # Move robot and particles
        rand_turn = random.uniform(0, 0.1)
        rand_forward = random.uniform(0, 1)
        robot = robot.move(rand_turn, rand_forward)
        Z = robot.sense()
        
        for i in range(N):
            particles[i] = particles[i].move(rand_turn, rand_forward)
        
        weights = [p.measurement_prob(Z) for p in particles]
        weights = [w / sum(weights) for w in weights]
        particles = resample(particles, weights)

        # Generate image and label
        particle_image = generate_particle_image(particles)
        x_PF, y_PF, theta_PF = estimate_pose(particles)
        label = label_data(robot.x, robot.y, robot.yaw, x_PF, y_PF, theta_PF, alpha, beta)
        
        data.append((particle_image, label))
    return data

# Initialize robot and particles
robot = Robot()
robot.set_pose(50, 50, 0)
particles = [Robot() for _ in range(N)]

# Generate training data
num_samples = 100
training_data = generate_training_data(robot, particles, num_samples, alpha, beta)

# Save training data
for i, (img, label) in enumerate(training_data):
    if label == 0:
        plt.imsave(f"training_data/Delocalized/img_{i}.png", img, cmap='gray')
    else:
        plt.imsave(f"training_data/Localized/img_{i}.png", img, cmap='gray')
