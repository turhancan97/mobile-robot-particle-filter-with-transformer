
# Mobile Robot Particle Filter with Vision Transformer

This project simulates a mobile robot's localization using a particle filter algorithm and evaluates its state (localized or delocalized) using a Vision Transformer (ViT) model.

## Table of Contents
- [Mobile Robot Particle Filter with Vision Transformer](#mobile-robot-particle-filter-with-vision-transformer)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Project Structure](#project-structure)
    - [Simulation](#simulation)
    - [Training the Model](#training-the-model)
    - [Inference](#inference)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)

## Overview

This project involves the following key components:
- A simulation of a mobile robot using a particle filter for localization.
- Generation of training data by simulating different states of the robot.
- Training a Vision Transformer (ViT) model to classify the robot's state as localized or delocalized.
- Running the simulation and using the trained model for real-time inference.

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/turhancan97/mobile-robot-particle-filter-with-transformer.git
   cd mobile-robot-particle-filter-with-transformer
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   conda create -n particle-filter python=3.9
   conda activate particle-filter
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Project Structure

- `train.py`: Script for training the Vision Transformer model.
- `inference.py`: Script for running the simulation and performing real-time inference.
- `data_generation.py`: Function for generating particle distribution images.
- `particle_filter_animation.py`: Contains the `Robot` class and particle filter logic.
- `README.md`: This file.

### Simulation

The simulation involves a mobile robot moving within a predefined environment, using particle filters to estimate its position and orientation.

### Training the Model

1. **Generate Training Data**:
   - Run the simulation to generate images representing different states (localized and delocalized).
   - Save the images in separate directories for training the Vision Transformer model.

2. **Train the Vision Transformer Model**:
   - Use `train.py` to train the model on the generated data.
   ```bash
   python train.py
   ```

### Inference

To run the simulation and monitor the robot's state in real-time, use `inference.py`.

```bash
python inference.py
```

This script will:
- Initialize the robot and particles.
- Move the robot and particles at each step.
- Use the trained Vision Transformer model to classify the robot's state.
- Visualize the simulation and classification results.

## Acknowledgements

This project is based on concepts from the paper:

- **Title**: Creating a robot localization monitor using particle filter and machine learning approaches
- **Authors**: Matthias Eder, Michael Reip, Gerald Steinbauer
- **Published in**: Springer / Applied Intelligence
- **Link**: [URL to the paper](https://link.springer.com/article/10.1007/s10489-020-02157-6)

The Vision Transformer model is implemented using the `timm` library.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.