import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt, pi, cos, sin, exp
from Robot import Robot, resample, world_size, landmarks, N
from PIL import Image, ImageDraw

# Constants
img_size = 224 # 36  # Image size for training data
alpha = 1.5  # Distance threshold in meters
beta = pi / 2  # Orientation threshold in radians

# def generate_particle_image(particles):
#     img = Image.new('L', (img_size, img_size), 0)
#     draw = ImageDraw.Draw(img)
#     for p in particles:
#         x = int(p.x / world_size * img_size)
#         y = int(p.y / world_size * img_size)
#         draw.point((x, y), fill=255)
#     return np.array(img)

def generate_particle_image(particles, img_size=36, world_size=100, s=1.5):
    # Compute mean position and orientation
    mean_x = np.mean([p.x for p in particles])
    mean_y = np.mean([p.y for p in particles])
    mean_yaw = np.mean([p.yaw for p in particles])

    # Create a blank image
    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    
    # Calculate scale factor to convert world coordinates to image coordinates
    scale_factor = img_size / s
    
    for p in particles:
        # Transform particle coordinates to the image frame
        dx = p.x - mean_x
        dy = p.y - mean_y
        
        # Rotate the coordinates based on the mean orientation
        x_rot = dx * np.cos(mean_yaw) + dy * np.sin(mean_yaw)
        y_rot = -dx * np.sin(mean_yaw) + dy * np.cos(mean_yaw)
        
        # Scale and translate to the center of the image
        x_img = int((x_rot * scale_factor) + (img_size / 2))
        y_img = int((y_rot * scale_factor) + (img_size / 2))
        
        # Draw the particle on the image
        if 0 <= x_img < img_size and 0 <= y_img < img_size:
            draw.point((x_img, y_img), fill=255)
    
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
        particle_image = generate_particle_image(particles, img_size, world_size, s=1.5)
        x_PF, y_PF, theta_PF = estimate_pose(particles)
        label = label_data(robot.x, robot.y, robot.yaw, x_PF, y_PF, theta_PF, alpha, beta)
        
        data.append((particle_image, label))
    return data

# Initialize robot and particles
robot = Robot()
robot.set_pose(50, 50, 0)
particles = [Robot() for _ in range(N)]

# Generate training data
num_samples = 1000
training_data = generate_training_data(robot, particles, num_samples, alpha, beta)

# Save training data
for i, (img, label) in enumerate(training_data):
    if label == 0:
        plt.imsave(f"training_data/Delocalized/img_{i}.png", img, cmap='gray')
    else:
        plt.imsave(f"training_data/Localized/img_{i}.png", img, cmap='gray')
