import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Robot import Robot, resample, world_size, landmarks, N # Import Robot class and resample function from Robot.py file


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
    '''Initialize the plot'''
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
