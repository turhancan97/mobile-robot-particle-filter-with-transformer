import torch
import timm
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn as nn
import random
from math import sqrt, pi, cos, sin, exp

# Define the Vision Transformer model class (same as used during training)
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Load the trained model
model = ViTClassifier()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Define transformations (same as used during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to generate particle image (same as used during training)
def generate_particle_image(particles, img_size=256*2):
    img = Image.new('L', (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    for p in particles:
        x = int(p.x / world_size * img_size)
        y = int(p.y / world_size * img_size)
        draw.point((x, y), fill=255)
    return np.array(img)

# Function to perform inference
def infer(model, particles):
    particle_image = generate_particle_image(particles)
    particle_image = transform(particle_image).unsqueeze(0).to(device).float()
    with torch.no_grad():
        outputs = model(particle_image)
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted: {predicted.item()}")
    return predicted.item()

# Visualization and Monitoring
fig, ax = plt.subplots()

def init():
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Robot Localization")
    return []

def animate(i):
    global robot, particles

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

    # Perform inference
    localized = infer(model, particles)

    # Clear the previous plot
    ax.clear()
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Robot {'Localized' if localized else 'Delocalized'}")

    for lm in landmarks:
        ax.plot(lm[0], lm[1], "g*")

    ax.plot(robot.x, robot.y, "ks", label="Robot")
    for p in particles:
        ax.plot(p.x, p.y, ".", color="blue" if localized else "red", alpha=0.5)

    ax.legend()
    return []

# Simulation parameters
world_size = 100
N = 1000
landmarks = [
    [10, 10], [10, 20], [10, 30], [10, 40], [10, 50],
    [10, 60], [10, 70], [10, 80], [10, 90], [90, 90],
    [90, 80], [90, 70], [90, 60], [90, 50], [90, 40],
    [90, 30], [90, 20], [90, 10]
]

# Assume these are defined (from your earlier code)
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

# Initialize robot and particles
robot = Robot()
robot.set_pose(50, 50, 0)
particles = [Robot() for _ in range(N)]

# Set up the animation
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=200, blit=False)

plt.show()
