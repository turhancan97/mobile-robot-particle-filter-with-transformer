import torch
import timm
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn as nn
import random
from Robot import Robot, resample, world_size, landmarks, N

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
# def generate_particle_image(particles, img_size=256*2):
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

# Function to perform inference
def infer(model, particles):
    particle_image = generate_particle_image(particles, img_size = 224, world_size=100, s=1.5)
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

# Initialize robot and particles
robot = Robot()
robot.set_pose(50, 50, 0)
particles = [Robot() for _ in range(N)]

# Set up the animation
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=200, blit=False)

plt.show()
